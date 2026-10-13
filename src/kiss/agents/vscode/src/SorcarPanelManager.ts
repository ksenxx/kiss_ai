// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import {EDITOR_TAB_BODY_ATTRS} from './SorcarTab';
import {PanelEvent, SorcarSidebarView} from './SorcarSidebarView';
import {
  clearWebviewNotificationPoster,
  setWebviewNotificationPoster,
} from './WebviewNotifications';
import {MetaPanelValues, TaskUpdateState, ToWebviewMessage} from './types';
import {PRODUCT_NAME, mediaIconPath} from './brand';

/** The webview panel viewType of the chat editor tab. */
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

/** True when *tab* is an editor tab hosting the chat webview (live or a
 * serialized placeholder the workbench has not revived yet). */
function isChatEditorTab(tab: vscode.Tab): boolean {
  const viewType = (tab.input as {viewType?: unknown} | null)?.viewType;
  return (
    typeof viewType === 'string' && viewType.includes(CHAT_PANEL_VIEW_TYPE)
  );
}

/**
 * Editor-tabs mode: the chat lives in ONE VS Code editor tab (a
 * WebviewPanel) instead of the secondary sidebar's chat view. The panel
 * hosts the same chat webview as the sidebar view — the whole daemon
 * tab registry mirrored into it, chats switched in place from the
 * history panel, "+" reusing an idle new chat — exactly as the remote
 * web app shows every chat in one browser tab. The panel is driven by
 * one SorcarSidebarView controller with its own daemon connection.
 */
export class SorcarPanelManager {
  private _panel: vscode.WebviewPanel | undefined;
  private _controller: SorcarSidebarView | undefined;
  // The webview-reported title of the chat on screen (no status
  // prefix) and that chat's task status: '', 'running', 'ok' or 'fail'.
  private _baseTitle: string = '';
  private _status: string = '';
  // The last task-info values the panel reported (metaUpdate), cached
  // so a Task Info view registered later starts from them.
  private _metaValues: MetaPanelValues | null = null;
  private _metaTaskUpdate: TaskUpdateState | null = null;
  // The chat / task the panel last reported showing (activeTask).
  private _activeChatId: string = '';
  private _activeTaskId: string = '';
  private _poster: ((message: NotificationMessage) => void) | undefined;
  // Where the panel's task-info values go: the secondary sidebar's
  // Task Info view (see extension.ts setMetaSink wiring).
  private _metaSink?: (
    values: MetaPanelValues | null,
    taskUpdate: TaskUpdateState | null,
  ) => void;
  // Where the panel's chat / task ids go: the primary sidebar's
  // history panel (see extension.ts setActiveTaskSink wiring).
  private _activeTaskSink?: (chatId: string, taskId: string) => void;
  /**
   * Called when a task starts in the chat on screen, as the Task Info
   * view is revealed for it. extension.ts has the view arrange its
   * sections for the run (postShowForRun).
   */
  public onTaskStarted?: () => void;
  // The two editor-tab icons: the KISS logo, and the green ring spinner
  // shown while the shown chat's task runs.  Built once so a repaint
  // that keeps the icon does not push a fresh Uri to the workbench.
  private readonly _kissIcon: vscode.Uri;
  private readonly _spinnerIcon: vscode.Uri;
  // Terminal teardown (deactivate / window reload): the panel is left
  // standing for the serializer.
  private _shuttingDown: boolean = false;

  /** @param _extensionUri The extension's root uri (chat HTML assets). */
  constructor(private readonly _extensionUri: vscode.Uri) {
    const root = _extensionUri.fsPath;
    this._kissIcon = vscode.Uri.joinPath(
      _extensionUri,
      mediaIconPath(root, 'kiss-icon.svg'),
    );
    this._spinnerIcon = vscode.Uri.joinPath(
      _extensionUri,
      mediaIconPath(root, 'spinner-running.svg'),
    );
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
   * Register the serializer that revives the chat panel across window
   * reloads. The daemon's registry holds the tab set and replays every
   * chat's transcript into the revived webview, so nothing beyond the
   * panel itself needs restoring.
   */
  public registerSerializer(): vscode.Disposable {
    // Older engines/tests may lack the API; the panel then simply does
    // not survive a reload.
    if (typeof vscode.window.registerWebviewPanelSerializer !== 'function') {
      return {dispose: () => {}};
    }
    return vscode.window.registerWebviewPanelSerializer(CHAT_PANEL_VIEW_TYPE, {
      deserializeWebviewPanel: (panel: vscode.WebviewPanel): Thenable<void> => {
        // The mode was switched off since the panel was serialized (the
        // chat lives in the sidebar view), or a chat was opened before
        // the workbench revived this placeholder (one panel only): the
        // husk is dropped.
        if (!SorcarPanelManager.modeEnabled() || this._panel) {
          panel.dispose();
        } else {
          // The workbench persisted the decorated title; the status it
          // carried belongs to the previous session.
          this._adopt(panel, stripStatusPrefix(panel.title));
        }
        return Promise.resolve();
      },
    });
  }

  /** The chat panel's controller, if the panel is open. */
  public controller(): SorcarSidebarView | undefined {
    return this._controller;
  }

  /**
   * Reveal the chat panel, opening it when none exists, and return its
   * controller.
   *
   * @param opts preserveFocus: keep the keyboard focus where it is
   *     (an automatic open the user did not ask for); otherwise the
   *     panel takes the focus as any new editor would.
   */
  public open(opts?: {preserveFocus?: boolean}): SorcarSidebarView {
    if (this._panel && this._controller) {
      this._panel.reveal(undefined, !!opts?.preserveFocus);
      return this._controller;
    }
    const panel = vscode.window.createWebviewPanel(
      CHAT_PANEL_VIEW_TYPE,
      DEFAULT_PANEL_TITLE,
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
    return this._adopt(panel, '');
  }

  /**
   * Open the chat panel in the background when the window has no chat
   * editor tab at all — neither a live panel nor a serialized
   * placeholder the workbench has not revived yet (opening next to a
   * placeholder would duplicate it). A no-op while the mode is off and
   * during extension teardown.
   */
  public ensureOpen(): void {
    if (this._shuttingDown || this._panel) return;
    if (!SorcarPanelManager.modeEnabled()) return;
    // Guarded like the other optional host APIs (absent in test stubs).
    const groups = vscode.window.tabGroups?.all;
    if (groups && groups.some(g => g.tabs.some(isChatEditorTab))) return;
    this.open({preserveFocus: true});
  }

  /**
   * Open the settings UI: reveal (or open) the chat panel and show its
   * settings panel — the editor-title gear button's action.
   */
  public async openSettings(): Promise<void> {
    await this.open().openSettingsUI();
  }

  /**
   * Show a chat in the panel — a primary-sidebar history panel click,
   * the Task Info view's Apps subpanel launching a "connect this app"
   * task (a new chat whose first prompt is submitted at once), or its
   * Recommendations subpanel putting a suggested task in a new chat's
   * composer (not submitted: the user reads it first).
   *
   * @param event The chat to open: backend chat id ('' or absent for a
   *     new chat), the task to scroll to, the task text, the remote
   *     machine the chat lives on ('' or absent for this one); or
   *     pendingText, the prompt a new chat's composer receives,
   *     submitted at once with autoSubmit.
   */
  public openChat(event: {
    chatId?: string;
    taskId?: string | number | null;
    title?: string;
    machine?: string;
    pendingText?: string;
    autoSubmit?: boolean;
  }): void {
    const controller = this.open();
    if (event.pendingText) {
      controller.newConversation();
      if (event.autoSubmit) controller.submitWhenReady(event.pendingText);
      else void controller.appendToInput(event.pendingText, true);
      return;
    }
    void controller.openChatFromHistory(event);
  }

  /**
   * Adopt *sink* as the destination of the panel's task-info values
   * and push the current state right away, so a Task Info view
   * registered after the panel never starts stale.
   */
  public setMetaSink(
    sink: (
      values: MetaPanelValues | null,
      taskUpdate: TaskUpdateState | null,
    ) => void,
  ): void {
    this._metaSink = sink;
    this._pushMeta();
  }

  /**
   * The Task Info view's refresh button: ask the chat panel to poll
   * the daemon for its task update with `refresh: true` (the panel's
   * metainfo block owns the poll; the Task Info view never polls).
   * No-op without a panel.
   */
  public refreshActiveTaskUpdate(): void {
    this._panel?.webview.postMessage({
      type: 'refreshTaskUpdate',
    } as ToWebviewMessage);
  }

  /**
   * Adopt *sink* as the destination of the panel's chat / task ids and
   * push the current state right away, so a history panel registered
   * after the panel highlights the right row from the start.
   */
  public setActiveTaskSink(
    sink: (chatId: string, taskId: string) => void,
  ): void {
    this._activeTaskSink = sink;
    this._pushActiveTask();
  }

  /**
   * Relay the panel's cached task-info values to the meta sink
   * (placeholder state when no panel exists or none reported yet) and
   * its chat / task ids to the active-task sink ('' when none).
   */
  private _pushMeta(): void {
    this._pushActiveTask();
    this._metaSink?.(this._metaValues, this._metaTaskUpdate);
  }

  private _pushActiveTask(): void {
    this._activeTaskSink?.(this._activeChatId, this._activeTaskId);
  }

  /**
   * Close the chat panel WITHOUT touching the daemon registry — used
   * when the user switches the mode off (the sidebar view re-adopts
   * the same tabs from the next `tabs_state`).
   */
  public close(): void {
    this._panel?.dispose();
  }

  public dispose(): void {
    // Terminal teardown (deactivate / window reload): dispose the
    // controller but LEAVE the panel standing. Disposing it here would
    // close the editor tab the workbench is about to persist,
    // defeating the serializer's revival on the next load.
    this._shuttingDown = true;
    this._controller?.dispose();
    this._controller = undefined;
    this._panel = undefined;
    this._refreshPoster();
  }

  private _adopt(panel: vscode.WebviewPanel, title: string): SorcarSidebarView {
    this._panel = panel;
    this._baseTitle = title;
    this._status = '';
    // A revived panel may still carry the previous session's status
    // prefix in its persisted title (stripped by the serializer);
    // repaint from the clean slate.  This also gives the panel its icon.
    this._applyTitle();
    const controller = new SorcarSidebarView(this._extensionUri, {
      onEvent: (event: PanelEvent) => this._onPanelEvent(event),
    });
    this._controller = controller;
    controller.onActiveTask = (chatId: string, taskId: string) => {
      this._activeChatId = chatId;
      this._activeTaskId = taskId;
      this._pushActiveTask();
    };
    controller.attachWebviewHost(
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
      EDITOR_TAB_BODY_ATTRS,
    );
    controller.syncWorkDir();
    panel.onDidDispose(() => {
      // Closing the panel closes the chat SURFACE, like closing the
      // remote web app's browser tab: the chats stay in the daemon's
      // registry for the next open.
      this._panel = undefined;
      this._controller = undefined;
      this._metaValues = null;
      this._metaTaskUpdate = null;
      this._activeChatId = '';
      this._activeTaskId = '';
      // The Task Info view must not keep describing a closed panel.
      this._pushMeta();
      controller.dispose();
      this._refreshPoster();
    });
    this._pushMeta();
    this._refreshPoster();
    return controller;
  }

  /**
   * Paint the editor tab: the green ring spinner as the tab icon while
   * the shown chat's task runs (the KISS logo otherwise), and a green
   * tick / red cross prefix on the title once it finished — the
   * editor-tab analogue of the sidebar strip's status icon.
   */
  private _applyTitle(): void {
    if (!this._panel) return;
    const icon =
      this._status === 'running' ? this._spinnerIcon : this._kissIcon;
    if (this._panel.iconPath !== icon) this._panel.iconPath = icon;
    let prefix = '';
    if (this._status === 'ok') {
      prefix = STATUS_OK_PREFIX;
    } else if (this._status === 'fail') {
      prefix = STATUS_FAIL_PREFIX;
    }
    this._panel.title = prefix + (this._baseTitle || DEFAULT_PANEL_TITLE);
  }

  private _onPanelEvent(event: PanelEvent): void {
    switch (event.kind) {
      case 'title':
        // The webview's title is authoritative and never decorated —
        // a chat legitimately titled "✅ deploy status" keeps its
        // tick (only the serializer strips, and only the one prefix
        // a previous session's decoration added).
        this._baseTitle = (event.title || '').trim();
        this._status = event.state || '';
        this._applyTitle();
        break;
      case 'reveal':
        // A finished task brings the editor tab forward the way
        // sidebar mode switches its internal tab — without stealing
        // the user's keyboard focus, and never while the user is in a
        // file: revealing would replace the document they are reading
        // in its group. The tab's ✅ / ❌ title decoration already tells
        // them the task ended. A waiting question (`force`) is the
        // exception: the agent is blocked until they answer it.
        if (vscode.window.activeTextEditor && !event.force) break;
        this._panel?.reveal(undefined, true);
        break;
      case 'taskStarted':
        // The run gets the window: the Task Info view comes up in the
        // secondary side bar beside the chat, the keyboard focus stays
        // in the chat the user just submitted from. Only for a chat on
        // screen: a start behind a text editor (a task launched from
        // another surface) brings nothing up over what the user is
        // looking at. The view arranges its sections for the run
        // (onTaskStarted) whether or not the reveal succeeds.
        if (!this._panel?.visible) break;
        this.onTaskStarted?.();
        void vscode.commands
          .executeCommand('kissSorcar.metaViewSecondary.focus', {
            preserveFocus: true,
          })
          .then(undefined, err =>
            console.error('[KISS Sorcar] Task Info reveal failed:', err),
          );
        break;
      case 'metaUpdate':
        this._metaValues = event.values;
        this._metaTaskUpdate = event.taskUpdate;
        this._pushMeta();
        break;
      case 'openChat':
        this.openChat(event);
        break;
    }
  }

  /**
   * Keep the shared webview toast poster installed while the panel
   * exists; release it (identity-checked) when the panel closes so the
   * sidebar view can claim it back.
   */
  private _refreshPoster(): void {
    const panel = this._panel;
    if (!panel) {
      if (this._poster) {
        clearWebviewNotificationPoster(this._poster);
        this._poster = undefined;
      }
      return;
    }
    if (this._poster) return;
    const poster = (message: NotificationMessage) => {
      void panel.webview.postMessage(message);
    };
    this._poster = poster;
    setWebviewNotificationPoster(poster);
  }
}
