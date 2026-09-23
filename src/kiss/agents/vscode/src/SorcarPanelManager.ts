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
import {MetaPanelValues, TaskUpdateState, ToWebviewMessage} from './types';
import {PRODUCT_NAME} from './brand';

/** The webview panel viewType of an editor-tab chat. */
export const CHAT_PANEL_VIEW_TYPE = 'kissSorcar.chatTab';

const DEFAULT_PANEL_TITLE = PRODUCT_NAME;

// Editor-tab status — the editor-tab analogue of the sidebar tab
// strip's status icon (.chat-tab-spinner / .chat-tab-ok / .chat-tab-fail
// in main.css).  While the task runs the tab ICON is the green ring
// spinner (media/spinner-running.svg, the same ring the composer's
// wait spinner and the history rows draw, animated inside the SVG so
// the workbench turns it without any clock of ours); a title is plain
// text, so the finished states are a green tick / red cross prefix.
const STATUS_OK_PREFIX = '\u2705 '; // ✅
const STATUS_FAIL_PREFIX = '\u274C '; // ❌
// Also matches the braille spinner frames and the 🟢 / 🔴 / ⚪ circles
// older versions persisted in the title, so a panel revived from a
// previous session still comes back clean.
const STATUS_PREFIX_RE =
  /^(?:\u2705|\u274C|[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]|\u{1F7E2}|\u{1F534}|\u{26AA})\s+/u;

/** Drop a status prefix a previous session left in a panel title. */
function stripStatusPrefix(title: string): string {
  return (title || '').replace(STATUS_PREFIX_RE, '');
}

type NotificationMessage = Extract<ToWebviewMessage, {type: 'notification'}>;

interface ChatPanel {
  /** The chat's tab id in the daemon's shared registry. */
  tabId: string;
  /** The backend chat id once the tab bound to one ('' before). */
  chatId: string;
  /**
   * True once a `tabs_state` snapshot confirmed the panel's root tab
   * bound to its chat (chatBound). A resume-born chatId without this
   * flag means the registration is still in flight, so the tab's
   * absence from a snapshot is not yet meaningful.
   */
  registryBound: boolean;
  /** The webview-reported chat title WITHOUT the status prefix. */
  baseTitle: string;
  /** Root task status: '', 'running', 'ok' or 'fail'. */
  status: string;
  panel: vscode.WebviewPanel;
  controller: SorcarSidebarView;
  /**
   * Raised before a NON-user disposal (mode switch off, closeSelf after
   * another client already closed the tab, extension teardown) so the
   * dispose handler does not retire the chat from the daemon registry.
   */
  suppressCloseTab: boolean;
  /**
   * Raised when the USER closed the root chat from inside the webview
   * (closeSelf with retire): the replacement chat that keeps the
   * one-tab invariant then takes the keyboard focus, exactly like the
   * sidebar strip's createNewTab focuses the fresh composer.
   */
  userClosed: boolean;
  /**
   * The panel's last reported task-info values (metaUpdate), cached so
   * a panel switch can repaint the secondary sidebar's Task Info view
   * without waiting for the panel's next report.
   */
  metaValues?: MetaPanelValues;
  /** The task-update report state that came with metaValues. */
  metaTaskUpdate?: TaskUpdateState | null;
  /**
   * The chat id and task id the panel last reported showing
   * (activeTask), cached so a panel switch can point the history
   * panel at the right row without waiting for the next report.
   */
  activeChatId?: string;
  activeTaskId?: string;
}

/** True when *tab* is an editor tab hosting a chat webview (live or a
 * serialized placeholder the workbench has not revived yet). */
function isChatEditorTab(tab: vscode.Tab): boolean {
  const viewType = (tab.input as {viewType?: unknown} | null)?.viewType;
  return (
    typeof viewType === 'string' && viewType.includes(CHAT_PANEL_VIEW_TYPE)
  );
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
  // Registry tabs whose adoption a same-chat panel with an
  // UNCONFIRMED registration blocked, keyed by chat id. If that
  // panel's registration goes through, the daemon displaces these
  // tabs and the next snapshot prunes them; if it never does (the
  // queued command was dropped) or the panel closes, the remembered
  // tab is adopted — otherwise the chat's real tab would be skipped
  // forever on the strength of a claim that can no longer bind.
  private _pendingAdoptions: Map<
    string,
    {entry: RegistryTabEntry; workspaceDir: string}
  > = new Map();
  private _poster: ((message: NotificationMessage) => void) | undefined;
  // Where the ACTIVE panel's task-info values go: the secondary
  // sidebar's Task Info view (see extension.ts setMetaSink wiring).
  private _metaSink?: (
    values: MetaPanelValues | null,
    taskUpdate: TaskUpdateState | null,
  ) => void;
  // Where the ACTIVE panel's chat / task ids go: the primary sidebar's
  // history panel (see extension.ts setActiveTaskSink wiring).
  private _activeTaskSink?: (chatId: string, taskId: string) => void;
  // The two editor-tab icons: the KISS logo, and the green ring spinner
  // shown while the panel's task runs.  Built once so a repaint that
  // keeps the icon does not push a fresh Uri to the workbench.
  private readonly _kissIcon: vscode.Uri;
  private readonly _spinnerIcon: vscode.Uri;
  // Terminal teardown (deactivate / window reload): panel disposals
  // after this are not user closes and must not retire chats from the
  // daemon's registry.
  private _shuttingDown: boolean = false;
  // Raised while closeAll (mode switch off) disposes the panels: those
  // disposals must not spawn a replacement chat tab — the chats move to
  // the sidebar view.
  private _closingAll: boolean = false;

  /**
   * @param _extensionUri The extension's root uri (chat HTML assets).
   * @param _retireTab Retires a chat tab from the daemon registry
   *     through a LONG-LIVED client (the sidebar controller's). A
   *     panel's own client dies with the panel, so a closeTab queued
   *     on it while the daemon is briefly unreachable would be lost.
   *     Optional: without it the panel's own controller is used.
   * @param _recordPanelTab Called with (tabId, true) when a panel
   *     opens and (tabId, false) when one closes — except during
   *     terminal teardown, which leaves the editor tabs standing for
   *     the serializer. The extension keeps a persistent record of
   *     the open panels' root tab ids in workspaceState so the NEXT
   *     session knows which registry tabs its serialized placeholders
   *     hold (see extension.ts registry adoption). Per-tab deltas
   *     rather than whole-set snapshots: serializer revival rebuilds
   *     the set one panel at a time, and a whole-set write mid-revival
   *     would replace the record with a partial one — a crash right
   *     then would let the next session adopt tabs whose placeholders
   *     still exist.
   */
  constructor(
    private readonly _extensionUri: vscode.Uri,
    private readonly _retireTab?: (tabId: string) => void,
    private readonly _recordPanelTab?: (tabId: string, open: boolean) => void,
  ) {
    const media = vscode.Uri.joinPath(_extensionUri, 'media');
    this._kissIcon = vscode.Uri.joinPath(media, 'kiss-icon.svg');
    this._spinnerIcon = vscode.Uri.joinPath(media, 'spinner-running.svg');
  }

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
        if (this._panels.has(tabId)) {
          // The registry adoption path opened this tab before the
          // workbench revived its serialized placeholder (possible
          // only when the persisted panel-id record missed it, e.g.
          // after a crash): the husk duplicates a live panel — drop
          // it instead of corrupting the id-keyed panel map.
          panel.dispose();
          return Promise.resolve();
        }
        // The workbench persisted the decorated title; the status it
        // carried belongs to the previous session.
        this._adoptPanel(panel, {tabId, title: stripStatusPrefix(panel.title)});
        return Promise.resolve();
      },
    });
  }

  /** How many chat panels are open. */
  public get panelCount(): number {
    return this._panels.size;
  }

  /**
   * How many editor tabs host a chat — live panels AND the serialized
   * placeholders a window reload restores (indistinguishable in the
   * tabGroups API; a placeholder revives into a panel only when its
   * tab is first shown). -1 when the API is absent (test stubs).
   */
  public chatEditorTabCount(): number {
    const groups = vscode.window.tabGroups?.all;
    if (!groups) return -1;
    let count = 0;
    for (const group of groups) {
      for (const tab of group.tabs) {
        if (isChatEditorTab(tab)) count += 1;
      }
    }
    return count;
  }

  /**
   * True when ANY editor tab hosts a chat: a live panel or a restored
   * placeholder awaiting revival. Opening a "first" chat next to a
   * placeholder would be a duplicate, so every "is a chat open?"
   * decision goes through this rather than panelCount alone.
   */
  public hasChatEditorTab(): boolean {
    if (this._panels.size > 0) return true;
    const count = this.chatEditorTabCount();
    // Guarded like the other optional host APIs (absent in test stubs).
    if (count < 0) return false;
    return count > 0;
  }

  /**
   * The editor-tabs-mode invariant: at least one chat editor tab is
   * always open, exactly as the sidebar strip always keeps one chat
   * tab (main.js closeTab -> createNewTab, reconcileTabs' placeholder).
   * Opens a fresh conversation when no chat editor tab — live panel or
   * restored placeholder — exists. A no-op while the mode is off, while
   * closeAll is moving the chats to the sidebar, and during terminal
   * teardown.
   *
   * @param opts preserveFocus: open the replacement in the background
   *     (an automatic open the user did not ask for — activation, a
   *     chat closed by another client); otherwise the fresh chat takes
   *     the keyboard focus like the sidebar's createNewTab does.
   * @returns The new chat's controller, or undefined when nothing
   *     needed opening.
   */
  public ensureChatOpen(opts?: {
    preserveFocus?: boolean;
  }): SorcarSidebarView | undefined {
    if (this._shuttingDown || this._closingAll) return undefined;
    if (!SorcarPanelManager.modeEnabled()) return undefined;
    if (this.hasChatEditorTab()) return undefined;
    if (opts?.preserveFocus) {
      return this._createPanel({tabId: randomTabId()}, {preserveFocus: true})
        .controller;
    }
    const controller = this.openNewChat();
    void controller.focusChatInput();
    return controller;
  }

  /**
   * Backstop for the one-tab invariant that the panel dispose handler
   * cannot see: a chat editor tab closed WITHOUT ever becoming a live
   * panel (a serialized placeholder the user closes before the
   * workbench revives it fires no WebviewPanel.onDidDispose). Every
   * chat tab close re-checks the invariant from the tabGroups model.
   */
  public watchEditorTabs(): vscode.Disposable {
    const tabGroups = vscode.window.tabGroups;
    // Guarded like the other optional host APIs (absent in test stubs).
    if (!tabGroups || typeof tabGroups.onDidChangeTabs !== 'function') {
      return {dispose: () => {}};
    }
    return tabGroups.onDidChangeTabs(e => {
      if (!e.closed.some(isChatEditorTab)) return;
      this.ensureChatOpen({preserveFocus: true});
    });
  }

  /** The controller of the most recently active chat panel, if any. */
  public activeController(): SorcarSidebarView | undefined {
    return this._activePanel()?.controller;
  }

  /**
   * Adopt *sink* as the destination of the active panel's task-info
   * values and push the current state right away, so a Task Info view
   * registered after the panels never starts stale.
   */
  public setMetaSink(
    sink: (
      values: MetaPanelValues | null,
      taskUpdate: TaskUpdateState | null,
    ) => void,
  ): void {
    this._metaSink = sink;
    this._pushActiveMeta();
  }

  /**
   * The Task Info view's refresh button: ask the ACTIVE chat editor
   * panel to poll the daemon for its task update with `refresh: true`
   * (the panel's metainfo block owns the poll; the Task Info view
   * never polls).  No-op without an active panel.
   */
  public refreshActiveTaskUpdate(): void {
    this._activePanel()?.panel.webview.postMessage({
      type: 'refreshTaskUpdate',
    } as ToWebviewMessage);
  }

  /**
   * Adopt *sink* as the destination of the active panel's chat / task
   * ids and push the current state right away, so a history panel
   * registered after the panels highlights the right row from the
   * start.
   */
  public setActiveTaskSink(
    sink: (chatId: string, taskId: string) => void,
  ): void {
    this._activeTaskSink = sink;
    this._pushActiveTask();
  }

  /**
   * Relay the ACTIVE panel's cached task-info values to the meta sink
   * (placeholder state when no panel exists or none reported yet) and
   * its chat / task ids to the active-task sink ('' when none).
   * Called on every metaUpdate of the active panel, on panel
   * activations and on panel closes, so the Task Info view and the
   * history panel always describe the chat editor tab the user is on.
   */
  private _pushActiveMeta(): void {
    this._pushActiveTask();
    if (!this._metaSink) return;
    const cp = this._activePanel();
    this._metaSink(cp?.metaValues ?? null, cp?.metaTaskUpdate ?? null);
  }

  /** Relay the ACTIVE panel's cached chat / task ids to its sink. */
  private _pushActiveTask(): void {
    if (!this._activeTaskSink) return;
    const cp = this._activePanel();
    this._activeTaskSink(cp?.activeChatId ?? '', cp?.activeTaskId ?? '');
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
    this.ensureChatOpen();
  }

  /**
   * Materialize editor-tab panels for registry tabs another client
   * created — or first bound to a chat — since the last snapshot: a
   * task run in the remote web app (or another window) opens as an
   * editor tab here, exactly like sidebar mode adopts the tab into
   * its internal strip. Tabs already open as a panel (by id or by
   * chat) are left alone, tabs scoped to another workspace are
   * skipped (same scoping as enterMode), and the new panel never
   * steals the user's keyboard focus.
   *
   * @param entries The snapshot's newly added tabs (see
   *     RegistryTabsDelta.added), possibly filtered by the caller.
   * @param workspaceDir This window's workspace root ('' = adopt all).
   * @param listed EVERY tab the snapshot lists (not just the added
   *     ones) — the source of truth for the displacement decision.
   */
  public adoptRegistryTabs(
    entries: RegistryTabEntry[],
    workspaceDir: string,
    listed?: RegistryTabEntry[],
  ): void {
    let listedIds: Set<string> | undefined;
    if (listed) {
      // Sync open panels' chat bindings from THIS snapshot before
      // deciding anything. Each panel's own daemon socket reports the
      // same binding as a `chatBound` event, but that is a separate
      // connection with no ordering guarantee against this snapshot
      // stream: if it lags, a panel whose tab an earlier snapshot
      // already confirmed would still read registryBound=false here,
      // and a displaced chat's replacement tab would be skipped for
      // good while the stale panel closes itself.
      listedIds = new Set();
      for (const entry of listed) {
        listedIds.add(entry.tabId);
        const cp = this._panels.get(entry.tabId);
        if (cp && entry.chatId) {
          cp.chatId = entry.chatId;
          cp.registryBound = true;
        }
      }
      // A remembered tab the registry no longer lists was displaced
      // or closed remotely; forget it.
      for (const [chatId, pending] of this._pendingAdoptions) {
        if (!listedIds.has(pending.entry.tabId)) {
          this._pendingAdoptions.delete(chatId);
        }
      }
    }
    for (const entry of entries) {
      if (this._panels.has(entry.tabId)) continue;
      if (entry.chatId) {
        const dup = [...this._panels.values()].find(
          cp => cp.chatId === entry.chatId,
        );
        // Skip a chat some open panel already shows — UNLESS the
        // registry DISPLACED that panel's tab (one tab per chat, the
        // newest bind wins: the panel's registry-confirmed tab is no
        // longer listed and its webview is about to close the panel),
        // in which case the replacement tab must be adopted or the
        // chat would lose its editor tab. A panel whose registration
        // is still in flight (resume-born chatId, never yet listed in
        // a snapshot) is not displaced — its absence from the list
        // means nothing yet, and its own pending bind will displace
        // this entry's tab in a moment.
        if (
          dup &&
          (!dup.registryBound || !listedIds || listedIds.has(dup.tabId))
        ) {
          if (!dup.registryBound) {
            // Remember the skip: the panel's claim may never bind
            // (see _pendingAdoptions).
            this._pendingAdoptions.set(entry.chatId, {entry, workspaceDir});
          }
          continue;
        }
      }
      const scope = entry.scopeWorkDir || entry.workDir;
      if (scope && workspaceDir && !isDirInside(scope, workspaceDir)) continue;
      this._createPanel(
        {tabId: entry.tabId, title: entry.title, inRegistry: true},
        {preserveFocus: true},
      );
    }
  }

  /**
   * Close every chat panel WITHOUT retiring the chats from the daemon
   * registry — used when the user switches the mode off (the sidebar
   * view re-adopts the same tabs from the next `tabs_state`).
   */
  public closeAll(): void {
    // Tabs remembered for adoption belong to the editor-tab surface;
    // once the mode is off the sidebar webview re-adopts everything
    // from `tabs_state`, and a stale memory could otherwise open a
    // long-vanished tab when the mode comes back.
    this._pendingAdoptions.clear();
    this._closingAll = true;
    try {
      for (const cp of [...this._panels.values()]) {
        cp.suppressCloseTab = true;
        cp.panel.dispose();
      }
    } finally {
      this._closingAll = false;
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
    this._pendingAdoptions.clear();
    this._active = undefined;
    this._refreshPoster();
  }

  private _activePanel(): ChatPanel | undefined {
    if (this._active && this._panels.has(this._active.tabId)) {
      return this._active;
    }
    return this._panels.values().next().value;
  }

  private _createPanel(
    init: EditorTabInit,
    opts?: {preserveFocus?: boolean},
  ): ChatPanel {
    const panel = vscode.window.createWebviewPanel(
      CHAT_PANEL_VIEW_TYPE,
      init.title || DEFAULT_PANEL_TITLE,
      // Adopted remote tabs open without stealing the user's keyboard
      // focus; user-initiated opens take it as any new editor would.
      {
        viewColumn: vscode.ViewColumn.Active,
        preserveFocus: !!opts?.preserveFocus,
      },
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
    const cp: ChatPanel = {
      tabId: init.tabId,
      chatId: init.resumeChatId || '',
      registryBound: false,
      baseTitle: init.title || '',
      status: '',
      panel,
      controller: undefined as unknown as SorcarSidebarView,
      suppressCloseTab: false,
      userClosed: false,
    };
    // A revived panel may still carry the previous session's status
    // prefix in its persisted title (the serializer strips it from
    // init.title); repaint from the clean slate.  This also gives the
    // panel its icon.
    this._applyPanelTitle(cp);
    cp.controller = new SorcarSidebarView(this._extensionUri, {
      rootTabId: init.tabId,
      onEvent: (event: PanelEvent) => this._onPanelEvent(cp, event),
    });
    cp.controller.onActiveTask = (chatId: string, taskId: string) => {
      cp.activeChatId = chatId;
      cp.activeTaskId = taskId;
      if (this._activePanel() === cp) this._pushActiveTask();
    };
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
    if (this._recordPanelTab) this._recordPanelTab(cp.tabId, true);
    // A revived background panel must not steal the "active" slot from
    // the panel the user is actually on.
    if (panel.active !== false) {
      this._active = cp;
      this._pushActiveMeta();
    }
    panel.onDidChangeViewState(e => {
      if (e.webviewPanel.active) {
        this._active = cp;
        // Repaint the Task Info view from the newly active panel's
        // cached values instead of waiting for its next report.
        this._pushActiveMeta();
      }
    });
    panel.onDidDispose(() => {
      this._panels.delete(cp.tabId);
      // During terminal teardown the editor tabs stay open for the
      // serializer, so the persisted record must keep their ids.
      if (!this._shuttingDown && this._recordPanelTab) {
        this._recordPanelTab(cp.tabId, false);
      }
      if (this._active === cp) this._active = undefined;
      // The Task Info view must not keep describing a closed panel.
      this._pushActiveMeta();
      // A user close retires the chat (the sidebar webview does the
      // same for its internal tabs); a mode switch, a closeSelf echo
      // or extension teardown must not.
      if (!this._shuttingDown && !cp.suppressCloseTab) {
        this._retire(cp);
        // The closed panel may have been the unconfirmed claim that
        // blocked a same-chat registry tab; that tab may open now.
        this._adoptPendingFor(cp.chatId);
      }
      cp.controller.dispose();
      this._refreshPoster();
      // The last chat editor tab closing — by the user (tab X, or the
      // root chat closed inside the webview), or because another
      // client closed / displaced the chat — leaves the window without
      // a chat, which the sidebar strip never allows: open a fresh one.
      // The tabGroups model already reflects this close (the workbench
      // updates it before disposing the editor input), so a restored
      // placeholder still standing keeps this from duplicating it.
      // Mode-off closes (closeAll) and teardown are skipped inside.
      const userClose = !cp.suppressCloseTab || cp.userClosed;
      this.ensureChatOpen({preserveFocus: !userClose});
    });
    this._refreshPoster();
    return cp;
  }

  /**
   * Adopt the registry tab remembered for *chatId*, if any — called
   * when the same-chat panel that blocked its adoption released the
   * claim (registration dropped) or closed.
   */
  private _adoptPendingFor(chatId: string): void {
    if (!chatId) return;
    const pending = this._pendingAdoptions.get(chatId);
    if (!pending) return;
    this._pendingAdoptions.delete(chatId);
    this.adoptRegistryTabs([pending.entry], pending.workspaceDir);
  }

  /** Retire *cp*'s chat tab from the daemon's shared registry. */
  private _retire(cp: ChatPanel): void {
    if (this._retireTab) this._retireTab(cp.tabId);
    else cp.controller.closeChatTab(cp.tabId);
  }

  /**
   * Paint *cp*'s editor tab: the green ring spinner as the tab icon
   * while the task runs (the KISS logo otherwise), and a green tick /
   * red cross prefix on the title once it finished — the editor-tab
   * analogue of the sidebar strip's status icon.
   */
  private _applyPanelTitle(cp: ChatPanel): void {
    const icon = cp.status === 'running' ? this._spinnerIcon : this._kissIcon;
    if (cp.panel.iconPath !== icon) cp.panel.iconPath = icon;
    let prefix = '';
    if (cp.status === 'ok') {
      prefix = STATUS_OK_PREFIX;
    } else if (cp.status === 'fail') {
      prefix = STATUS_FAIL_PREFIX;
    }
    cp.panel.title = prefix + (cp.baseTitle || DEFAULT_PANEL_TITLE);
  }

  private _onPanelEvent(cp: ChatPanel, event: PanelEvent): void {
    switch (event.kind) {
      case 'title': {
        // The webview's title is authoritative and never decorated —
        // a chat legitimately titled "✅ deploy status" keeps its
        // tick (only the serializer strips, and only the one prefix
        // a previous session's decoration added).
        cp.baseTitle = (event.title || '').trim();
        cp.status = event.state || '';
        this._applyPanelTitle(cp);
        break;
      }
      case 'reveal':
        // A finished task brings its editor tab forward the way
        // sidebar mode switches its internal tab — without stealing
        // the user's keyboard focus.
        cp.panel.reveal(undefined, true);
        break;
      case 'metaUpdate':
        cp.metaValues = event.values;
        cp.metaTaskUpdate = event.taskUpdate;
        if (this._activePanel() === cp) this._pushActiveMeta();
        break;
      case 'chatBound':
        cp.chatId = event.chatId;
        cp.registryBound = true;
        // The daemon accepted THIS panel's binding, so any same-chat
        // tab remembered for adoption was displaced by it. The next
        // controller snapshot prunes it too, but a user close racing
        // that snapshot must not resurrect the displaced tab.
        this._pendingAdoptions.delete(event.chatId);
        break;
      case 'registrationDropped': {
        // The panel's registry registration never reached the daemon
        // (its queued command was dropped after an outage) and the
        // webview does not retry it: the chat claim is void. Release
        // it and open any same-chat registry tab the claim blocked —
        // otherwise this window would keep an editor tab that can
        // never bind while the chat's real tab never opens.
        if (cp.registryBound) break;
        const claimed = cp.chatId;
        cp.chatId = '';
        this._adoptPendingFor(claimed);
        break;
      }
      case 'closeSelf':
        // The dispose handler must not send a second close; when the
        // USER closed the root chat inside the panel (retire), the
        // registry entry is retired through the long-lived client here.
        cp.suppressCloseTab = true;
        // retire = the USER closed the root chat inside the webview;
        // otherwise the registry dropped the tab (closed elsewhere).
        cp.userClosed = !!event.retire;
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
   *     a fresh one), the task to scroll to, the panel title, and (for
   *     fresh chats) the opener's composer draft to seed the new
   *     panel's textarea with.
   */
  public openChat(event: {
    chatId?: string;
    taskId?: string | number | null;
    title?: string;
    pendingText?: string;
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
      // Only a FRESH chat starts from the opener's draft; a resume
      // shows the resumed chat's own composer state.
      pendingText: chatId ? undefined : event.pendingText || undefined,
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
