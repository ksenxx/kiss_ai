// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as crypto from 'crypto';
import * as vscode from 'vscode';
import {editorTabBodyAttrs, EditorTabInit} from './SorcarTab';
import {
  PanelEvent,
  RegistryTabEntry,
  SorcarSidebarView,
} from './SorcarSidebarView';
import {
  clearWebviewNotificationPoster,
  setWebviewNotificationPoster,
} from './WebviewNotifications';
import {ToWebviewMessage} from './types';

/** The webview panel viewType of an editor-tab chat. */
export const CHAT_PANEL_VIEW_TYPE = 'kissSorcar.chatTab';

const DEFAULT_PANEL_TITLE = 'KISS Sorcar';

type NotificationMessage = Extract<ToWebviewMessage, {type: 'notification'}>;

interface ChatPanel {
  /** The chat's tab id in the daemon's shared registry. */
  tabId: string;
  /** The backend chat id once the tab bound to one ('' before). */
  chatId: string;
  panel: vscode.WebviewPanel;
  controller: SorcarSidebarView;
  /**
   * Raised before a NON-user disposal (mode switch off, closeSelf after
   * another client already closed the tab, extension teardown) so the
   * dispose handler does not retire the chat from the daemon registry.
   */
  suppressCloseTab: boolean;
}

function randomTabId(): string {
  return crypto.randomUUID();
}

/** Normalize a directory for comparison (Windows: \ and case). */
function normDir(dir: string): string {
  let s = dir.replace(/\\/g, '/').replace(/\/+$/, '');
  if (process.platform === 'win32') s = s.toLowerCase();
  return s;
}

/** True when *dir* is *root* or a subdirectory of it. */
function isDirInside(dir: string, root: string): boolean {
  const d = normDir(dir);
  const r = normDir(root);
  if (d === r) return true;
  return d.startsWith(r + '/');
}

/**
 * Editor-tabs mode: one VS Code EDITOR TAB (WebviewPanel) per chat tab,
 * replacing the secondary sidebar's internal tab bar. Each panel hosts
 * the same chat webview pinned to a single root chat tab (see
 * editorTabBodyAttrs / main.js EDITOR_TAB_MODE) and is driven by its
 * own SorcarSidebarView controller with its own daemon connection —
 * the same client-per-surface model the remote web app uses.
 */
export class SorcarPanelManager {
  private _panels: Map<string, ChatPanel> = new Map();
  private _active: ChatPanel | undefined;
  private _poster: ((message: NotificationMessage) => void) | undefined;
  // Terminal teardown (deactivate / window reload): panel disposals
  // after this are not user closes and must not retire chats from the
  // daemon's registry.
  private _shuttingDown: boolean = false;

  /**
   * @param _extensionUri The extension's root uri (chat HTML assets).
   * @param _retireTab Retires a chat tab from the daemon registry
   *     through a LONG-LIVED client (the sidebar controller's). A
   *     panel's own client dies with the panel, so a closeTab queued
   *     on it while the daemon is briefly unreachable would be lost.
   *     Optional: without it the panel's own controller is used.
   */
  constructor(
    private readonly _extensionUri: vscode.Uri,
    private readonly _retireTab?: (tabId: string) => void,
  ) {}

  /** Whether editor-tabs mode is currently switched on. */
  public static modeEnabled(): boolean {
    // Guarded like the other optional host APIs (test stubs may not
    // model configuration); the mode then simply reads as off.
    if (typeof vscode.workspace?.getConfiguration !== 'function') {
      return false;
    }
    return (
      vscode.workspace
        .getConfiguration('kissSorcar')
        .get<boolean>('editorTabsMode') === true
    );
  }

  /**
   * Register the serializer that revives chat panels across window
   * reloads. The webview persists its root tab id (main.js
   * persistTabState, editorRootTabId), so a revived panel re-adopts
   * the same chat and the daemon's replay refills its transcript.
   */
  public registerSerializer(): vscode.Disposable {
    // Older engines/tests may lack the API; panels then simply do not
    // survive a reload.
    if (typeof vscode.window.registerWebviewPanelSerializer !== 'function') {
      return {dispose: () => {}};
    }
    return vscode.window.registerWebviewPanelSerializer(CHAT_PANEL_VIEW_TYPE, {
      deserializeWebviewPanel: (
        panel: vscode.WebviewPanel,
        state: unknown,
      ): Thenable<void> => {
        if (!SorcarPanelManager.modeEnabled()) {
          // The mode was switched off since the panel was serialized;
          // its chat lives on in the sidebar view.
          panel.dispose();
          return Promise.resolve();
        }
        const s = (state || {}) as {editorRootTabId?: unknown};
        const tabId =
          typeof s.editorRootTabId === 'string' && s.editorRootTabId
            ? s.editorRootTabId
            : randomTabId();
        this._adoptPanel(panel, {tabId, title: panel.title});
        return Promise.resolve();
      },
    });
  }

  /** How many chat panels are open. */
  public get panelCount(): number {
    return this._panels.size;
  }

  /** The controller of the most recently active chat panel, if any. */
  public activeController(): SorcarSidebarView | undefined {
    return this._activePanel()?.controller;
  }

  /** Open a fresh conversation in a new editor tab and focus it. */
  public openNewChat(): SorcarSidebarView {
    return this._createPanel({tabId: randomTabId()}).controller;
  }

  /**
   * Reveal the most recently active chat panel, opening a fresh one
   * when none exists, and return its controller.
   */
  public revealActiveOrCreate(): SorcarSidebarView {
    const active = this._activePanel();
    if (active) {
      active.panel.reveal();
      return active.controller;
    }
    return this.openNewChat();
  }

  /**
   * Open the settings UI: reveal (or open) a chat panel and show its
   * settings panel — the editor-title gear button's action.
   */
  public async openSettings(): Promise<void> {
    const controller = this.revealActiveOrCreate();
    await controller.openSettingsUI();
  }

  /**
   * Materialize editor-tab panels for the daemon registry's chat tabs
   * scoped to *workspaceDir* — called when the user switches the mode
   * on, so the sidebar's chats migrate to editor tabs. Opens a fresh
   * chat when the registry has none for this workspace.
   */
  public enterMode(entries: RegistryTabEntry[], workspaceDir: string): void {
    for (const entry of entries) {
      if (this._panels.has(entry.tabId)) continue;
      const scope = entry.scopeWorkDir || entry.workDir;
      if (scope && workspaceDir && !isDirInside(scope, workspaceDir)) continue;
      this._createPanel({
        tabId: entry.tabId,
        title: entry.title,
        // Registry-born: the webview may treat the tab's absence from
        // its first snapshot as a close by another client.
        inRegistry: true,
      });
    }
    if (this._panels.size === 0) this.openNewChat();
  }

  /**
   * Close every chat panel WITHOUT retiring the chats from the daemon
   * registry — used when the user switches the mode off (the sidebar
   * view re-adopts the same tabs from the next `tabs_state`).
   */
  public closeAll(): void {
    for (const cp of [...this._panels.values()]) {
      cp.suppressCloseTab = true;
      cp.panel.dispose();
    }
  }

  /**
   * Mark terminal teardown: panel disposals from here on (deactivate,
   * window reload) are not user closes and keep the chats registered.
   */
  public markShutdown(): void {
    this._shuttingDown = true;
  }

  public dispose(): void {
    // Terminal teardown (deactivate / window reload): dispose the
    // controllers but LEAVE the panels standing. Disposing them here
    // would close the editor tabs the workbench is about to persist,
    // defeating the serializer's revival on the next load.
    this.markShutdown();
    for (const cp of [...this._panels.values()]) {
      cp.suppressCloseTab = true;
      cp.controller.dispose();
    }
    this._panels.clear();
    this._active = undefined;
    this._refreshPoster();
  }

  private _activePanel(): ChatPanel | undefined {
    if (this._active && this._panels.has(this._active.tabId)) {
      return this._active;
    }
    return this._panels.values().next().value;
  }

  private _createPanel(init: EditorTabInit): ChatPanel {
    const panel = vscode.window.createWebviewPanel(
      CHAT_PANEL_VIEW_TYPE,
      init.title || DEFAULT_PANEL_TITLE,
      vscode.ViewColumn.Active,
      {
        enableScripts: true,
        // A backgrounded chat keeps its transcript, composer draft and
        // daemon connection exactly like a hidden sidebar view does.
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(this._extensionUri, 'media'),
          vscode.Uri.joinPath(this._extensionUri, 'out'),
        ],
      },
    );
    return this._adoptPanel(panel, init);
  }

  private _adoptPanel(
    panel: vscode.WebviewPanel,
    init: EditorTabInit,
  ): ChatPanel {
    panel.iconPath = vscode.Uri.joinPath(
      this._extensionUri,
      'media',
      'kiss-icon.svg',
    );
    const cp: ChatPanel = {
      tabId: init.tabId,
      chatId: init.resumeChatId || '',
      panel,
      controller: undefined as unknown as SorcarSidebarView,
      suppressCloseTab: false,
    };
    cp.controller = new SorcarSidebarView(this._extensionUri, {
      rootTabId: init.tabId,
      onEvent: (event: PanelEvent) => this._onPanelEvent(cp, event),
    });
    cp.controller.attachWebviewHost(
      {
        webview: panel.webview,
        get visible() {
          return panel.visible;
        },
        show: () => panel.reveal(undefined, false),
        // onDidChangeViewState covers visibility changes (and more);
        // the controller only re-reads host.visible on each firing.
        onDidChangeVisibility:
          panel.onDidChangeViewState as vscode.Event<unknown>,
        onDidDispose: panel.onDidDispose,
      },
      editorTabBodyAttrs(init),
    );
    cp.controller.syncWorkDir();
    this._panels.set(cp.tabId, cp);
    // A revived background panel must not steal the "active" slot from
    // the panel the user is actually on.
    if (panel.active !== false) this._active = cp;
    panel.onDidChangeViewState(e => {
      if (e.webviewPanel.active) this._active = cp;
    });
    panel.onDidDispose(() => {
      this._panels.delete(cp.tabId);
      if (this._active === cp) this._active = undefined;
      // A user close retires the chat (the sidebar webview does the
      // same for its internal tabs); a mode switch, a closeSelf echo
      // or extension teardown must not.
      if (!this._shuttingDown && !cp.suppressCloseTab) {
        this._retire(cp);
      }
      cp.controller.dispose();
      this._refreshPoster();
    });
    this._refreshPoster();
    return cp;
  }

  /** Retire *cp*'s chat tab from the daemon's shared registry. */
  private _retire(cp: ChatPanel): void {
    if (this._retireTab) this._retireTab(cp.tabId);
    else cp.controller.closeChatTab(cp.tabId);
  }

  private _onPanelEvent(cp: ChatPanel, event: PanelEvent): void {
    switch (event.kind) {
      case 'title': {
        const title = (event.title || '').trim();
        cp.panel.title = title || DEFAULT_PANEL_TITLE;
        break;
      }
      case 'chatBound':
        cp.chatId = event.chatId;
        break;
      case 'closeSelf':
        // The dispose handler must not send a second close; when the
        // USER closed the root chat inside the panel (retire), the
        // registry entry is retired through the long-lived client here.
        cp.suppressCloseTab = true;
        if (event.retire) this._retire(cp);
        cp.panel.dispose();
        break;
      case 'openChat':
        this.openChat(event);
        break;
    }
  }

  /**
   * Open a chat as an editor tab: reveal the panel already bound to
   * `event.chatId`, or create a new panel resuming it (a fresh
   * conversation when the id is empty). Serves both an editor-tab
   * webview's own history opens and the primary-sidebar history
   * panel's clicks.
   *
   * @param event The chat to open — backend chat id ('' or absent for
   *     a fresh one), the task to scroll to, and the panel title.
   */
  public openChat(event: {
    chatId?: string;
    taskId?: string | number | null;
    title?: string;
  }): void {
    const chatId = event.chatId ? String(event.chatId) : '';
    if (chatId) {
      // One chat, one panel: a history open of a chat that already
      // has an editor tab reveals that tab (the daemon registry
      // enforces the same one-tab-per-chat invariant) and brings the
      // clicked task on screen instead of leaving the panel parked on
      // whatever task it was showing.
      for (const other of this._panels.values()) {
        if (other.chatId === chatId) {
          other.panel.reveal();
          if (
            event.taskId !== undefined &&
            event.taskId !== null &&
            event.taskId !== ''
          ) {
            other.controller.showTask(String(event.taskId));
          }
          return;
        }
      }
    }
    this._createPanel({
      tabId: randomTabId(),
      title: event.title,
      resumeChatId: chatId || undefined,
      resumeTaskId:
        event.taskId === undefined || event.taskId === null
          ? undefined
          : String(event.taskId),
    });
  }

  /**
   * Keep the shared webview toast poster pointing at the active chat
   * panel while any panel exists; release it (identity-checked) when
   * the last panel closes so the sidebar view can claim it back.
   */
  private _refreshPoster(): void {
    if (this._panels.size === 0) {
      if (this._poster) {
        clearWebviewNotificationPoster(this._poster);
        this._poster = undefined;
      }
      return;
    }
    if (this._poster) return;
    const poster = (message: NotificationMessage) => {
      this._activePanel()?.panel.webview.postMessage(message);
    };
    this._poster = poster;
    setWebviewNotificationPoster(poster);
  }
}
