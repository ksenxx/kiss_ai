// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

function isSilentDiscardMessage(message: string | undefined): boolean {
  return /^Discarded branch '[^']+'\.$/.test(message || '');
}

const NATIVE_VIEWER_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.bmp',
  '.ico',
  '.webp',
  '.tiff',
  '.tif',
  '.avif',
  '.heic',
  '.pdf',
  '.zip',
  '.tar',
  '.gz',
  '.tgz',
  '.bz2',
  '.xz',
  '.7z',
  '.rar',
  '.jar',
  '.war',
  '.doc',
  '.docx',
  '.xls',
  '.xlsx',
  '.ppt',
  '.pptx',
  '.odt',
  '.ods',
  '.odp',
  '.exe',
  '.dll',
  '.so',
  '.dylib',
  '.a',
  '.o',
  '.class',
  '.wasm',
  '.mp3',
  '.wav',
  '.ogg',
  '.flac',
  '.m4a',
  '.aac',
  '.mp4',
  '.m4v',
  '.mov',
  '.avi',
  '.mkv',
  '.webm',
  '.ttf',
  '.otf',
  '.woff',
  '.woff2',
  '.eot',
  '.pyc',
  '.pyo',
  '.bin',
  '.dat',
  '.db',
  '.sqlite',
  '.sqlite3',
]);

function isTextLikeExtension(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  if (!ext) return true;
  return !NATIVE_VIEWER_EXTENSIONS.has(ext);
}

/**
 * Whether clicking a link to *filePath* should render the file in a
 * webview tab instead of opening its source in a text editor — the VS
 * Code counterpart of the remote web app, which renders .html/.htm
 * files in an in-app tab (see renderContentView in media/main.js).
 */
function isRenderableHtmlExtension(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  return ext === '.html' || ext === '.htm';
}

/**
 * Whether clicking a link to *filePath* should render the file as
 * converted-to-HTML markdown in a tab (VS Code's built-in markdown
 * preview) — the counterpart of the remote web app, which converts
 * .md/.markdown content with marked and renders it in a content tab
 * (see renderContentView in media/main.js).
 */
function isRenderableMarkdownExtension(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  return ext === '.md' || ext === '.markdown';
}

function escapeHtmlAttr(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}

/**
 * Return *html* with a `<base>` tag pointing at *dirUri* injected, so
 * relative asset references in the document resolve through the
 * webview's resource scheme instead of the unreachable file scheme.
 *
 * Insertion point, in order of preference: right after the opening
 * `<head>` tag (located on a copy with comments blanked to the same
 * length, so a `<head>` inside a comment is never chosen, and scanned
 * to its closing `>` with quote awareness so a `>` inside an attribute
 * value does not end the tag early); otherwise right after the doctype,
 * so the injection never demotes the document to quirks mode; otherwise
 * at the very start.
 */
function injectHtmlBase(html: string, dirUri: string): string {
  const base = `<base href="${escapeHtmlAttr(dirUri)}/">`;
  const search = html.replace(/<!--[\s\S]*?-->/g, m => ' '.repeat(m.length));
  const head = /<head(?=[\s/>])/i.exec(search);
  if (head) {
    let i = head.index + head[0].length;
    let quote = '';
    for (; i < search.length; i++) {
      const ch = search[i];
      if (quote) {
        if (ch === quote) quote = '';
      } else if (ch === '"' || ch === "'") {
        quote = ch;
      } else if (ch === '>') {
        return html.slice(0, i + 1) + base + html.slice(i + 1);
      }
    }
  }
  const doctype = /<!doctype[^>]*>/i.exec(search);
  if (doctype) {
    const at = doctype.index + doctype[0].length;
    return html.slice(0, at) + base + html.slice(at);
  }
  return base + html;
}
import {AgentClient, DroppedCommandReason} from './AgentClient';
import {getGitApi} from './gitApi';
import {
  provisionalDefaultModel,
  resolveDefaultModel,
} from './DependencyInstaller';
import {buildChatHtml} from './SorcarTab';
import {PDF_DOWNLOAD_MESSAGE, buildPdfPreviewHtml} from './pdfPreview';
import {VoiceWakeService} from './voiceWake';
import {kissHomeDir} from './userAssets';
import {playVoiceAckClip} from './voiceAckPlayer';
import {bootstrapInstallUrl, findInstallScript} from './installerPath';
import {
  FromWebviewMessage,
  ToWebviewMessage,
  AgentCommand,
  MetaPanelValues,
  TaskUpdateState,
} from './types';
import {
  clearWebviewNotificationPoster,
  resolveWebviewNotificationAction,
  setWebviewNotificationPoster,
  showErrorNotification,
  showInformationNotification,
  showWarningNotification,
  withWebviewNotificationProgress,
} from './WebviewNotifications';
import {PRODUCT_NAME} from './brand';

/**
 * The webview surface a chat controller drives, abstracting over the
 * secondary-sidebar `WebviewView` and an editor tab's `WebviewPanel`
 * (editor-tabs mode): the handful of members the controller actually
 * uses, with `show` mapping to `WebviewView.show(preserveFocus)` /
 * `WebviewPanel.reveal(...)`.
 */
export interface ChatWebviewHost {
  readonly webview: vscode.Webview;
  readonly visible: boolean;
  show(): void;
  onDidChangeVisibility: vscode.Event<unknown>;
  onDidDispose: vscode.Event<void>;
}

/**
 * Editor-tabs-mode notifications a per-panel controller raises for its
 * panel manager: everything the webview asks of its hosting EDITOR TAB
 * rather than of the daemon.
 */
export type PanelEvent =
  // The root chat tab renamed itself or its task's status changed;
  // retitle the editor tab. `state` is '' (no task yet), 'running',
  // 'ok' or 'fail' — the internal tab strip's status icon.
  | {kind: 'title'; title: string; state?: string}
  // A task in the panel just finished; bring the editor tab forward.
  // `force`: a question is waiting, so reveal even while the user is
  // in a text editor.
  | {kind: 'reveal'; force?: boolean}
  // A task just started in the panel's chat tab: bring the Task Info
  // view up in the secondary side bar, keyboard focus kept in the chat.
  | {kind: 'taskStarted'}
  // Open another chat as a new editor tab (fresh when chatId is '').
  | {
      kind: 'openChat';
      chatId?: string;
      taskId?: string | number | null;
      title?: string;
      // Fresh chats: composer draft to seed the new panel's textarea.
      pendingText?: string;
      // Fresh chats: submit pendingText as the first task once ready.
      autoSubmit?: boolean;
    }
  // Close this panel. retire=true means the USER closed the root chat
  // inside the panel, so the host must also retire the tab from the
  // daemon registry; without it the registry already dropped the tab.
  | {kind: 'closeSelf'; retire?: boolean}
  // The root chat tab bound to a backend chat id (dedupe key for
  // history opens).
  | {kind: 'chatBound'; chatId: string}
  // The client gave up on the panel's own registry registration
  // (`openTab`/`resumeSession` dropped after an outage): the daemon
  // never saw it and the webview does not retry it, so the panel's
  // chat claim is void.
  | {kind: 'registrationDropped'}
  // The panel's live task-info values changed (tokens, cost, steps,
  // time, machine, workdir, max budget, task update): the panel
  // manager caches them and, when this is the ACTIVE panel, relays
  // them to the secondary sidebar's Task Info view.
  | {
      kind: 'metaUpdate';
      values: MetaPanelValues;
      taskUpdate: TaskUpdateState | null;
    };

/** One tab of the daemon's canonical `tabs_state` registry snapshot. */
export interface RegistryTabEntry {
  tabId: string;
  chatId: string;
  title: string;
  workDir: string;
}

/** One `tabs_state` snapshot as seen by the extension host (see
 * onRegistryTabsState). */
export interface RegistryTabsDelta {
  /**
   * Tabs newly listed — or first bound to a chat — relative to the
   * previous snapshot (ALL listed tabs when this is the first one).
   */
  added: RegistryTabEntry[];
  /**
   * Every entry the snapshot lists. The subscriber syncs open panels'
   * chat bindings from it, and can tell a chat whose old tab was
   * DISPLACED by the one-tab-per-chat invariant (old id no longer
   * listed) from an ordinary duplicate.
   */
  listed: RegistryTabEntry[];
  /**
   * True for the controller's first snapshot. Its tabs predate this
   * session, and a reloaded window may still hold some of them as
   * serialized editor-tab placeholders the panel manager has not
   * adopted yet — adopting is then unsafe (duplicate panels).
   */
  firstSnapshot: boolean;
}

/** Editor-tabs mode wiring handed to a per-panel chat controller. */
export interface PanelHooks {
  /** The panel's single root chat tab id (owned from the start). */
  rootTabId: string;
  /** Receives the panel-directed events listed in PanelEvent. */
  onEvent: (event: PanelEvent) => void;
  /**
   * `<body>` attributes for a surface attached through
   * resolveWebviewView (the primary-sidebar history panel, which is a
   * WebviewView rather than a WebviewPanel); editor-tab panels pass
   * theirs to attachWebviewHost directly.
   */
  bodyAttrs?: string;
}

const FORWARDED_COMMANDS: Record<string, readonly string[]> = {
  appendUserMessage: ['prompt', 'tabId'],
  // A tool-call panel's Stop button: the daemon interrupts just that
  // tool call on the tab's task and answers with a direct
  // `tool_interrupt_ack` that the client-listener relay passes back.
  interruptTool: ['tabId', 'toolName', 'callId'],
  getInputHistory: [],
  // The welcome page and the settings panel show how to reach the
  // remote webapp: the daemon owns the URL file and the tunnel, and
  // answers with the `remote_url` event it also broadcasts on every
  // URL change, so the host reads and polls nothing itself.
  getWelcomeInfo: [],
  // "Don't show tips again": the daemon writes the one opt-out marker
  // both surfaces honour.
  tipsOptOut: ['optOut'],
  newChat: ['tabId'],
  openTab: ['tabId', 'title', 'workDir'],
  getHistory: ['query', 'tag', 'offset', 'generation'],
  setFavorite: ['taskId', 'isFavorite'],
  // tabId must survive: the daemon echoes it on the `files` reply so the
  // webview can tell whether the @-mention picker still belongs to the
  // conversation on screen.
  getFiles: ['prefix', 'workDir', 'tabId'],
  // File links are resolved by the daemon for every surface (`~`, the
  // tab's work dir, its pending worktree — web_server.py
  // _resolve_tab_file). checkPaths is answered with `pathsExist`,
  // relayed to the webview; openFile is answered to this window with
  // `openResolvedFile` (handled in the client listener, opened in a
  // real editor tab) where a browser would get `fileContent`. workDir
  // is forwarded as sent: the daemon resolves an empty one against its
  // global working directory and echoes the client's value on
  // `pathsExist`, which the webview uses as a correlation key.
  openFile: ['path', 'line', 'workDir', 'tabId'],
  checkPaths: ['paths', 'workDir', 'tabId'],
  getAdjacentTask: ['tabId', 'taskId', 'direction'],
  getConfig: [],
  saveConfig: ['config', 'apiKeys'],
  // The settings panel's Custom Models subpanel: the daemon owns
  // ~/.kiss/MY_MODELS.json, so the CRUD travels to it whole and the
  // `myModelsData` replies come back through the client relay.
  getMyModels: [],
  saveMyModel: ['name', 'endpoint', 'apiKey', 'headers', 'originalName'],
  deleteMyModel: ['name'],
  // The Inject promptlet panel's Add button and per-row delete and edit
  // buttons: the daemon owns ~/.kiss/MY_INJECTION.md and answers with an
  // unstamped `tricksData` list that every window's panel repaints from.
  addTrick: ['text'],
  deleteTrick: ['text'],
  editTrick: ['text', 'newText'],
  // The daemon builds and writes the shared chat page for both the
  // extension and the remote webapp, so the webview's serialized
  // transcript travels through whole; the daemon answers with a
  // direct `share_done` that the client-listener relay above passes
  // straight back to the webview.
  shareChat: ['tabId', 'chatId', 'title', 'html', 'workDir'],
  // A share exports ALL of the chat's tasks; the daemon answers with a
  // direct `share_tasks` carrying every persisted transcript of the
  // chat, which the webview assembles into the page it then sends
  // back via `shareChat`. The optional taskId narrows the export to
  // one task and its sub-agents (a sub-agent tab's share).
  shareChatTasks: ['tabId', 'chatId', 'taskId'],
  // "Remind me later" on the webview's update toast: the daemon owns
  // the update_available broadcast, records the 24h snooze in the
  // update-check cache shared with this extension host, and
  // rebroadcasts so every window's toast disappears.
  snoozeUpdate: ['latest'],
  // "Update when idle" / "Cancel" on the same toast: the daemon arms a
  // poller that runs install.sh once no task is in flight, and
  // rebroadcasts update_available with `pendingIdle`.
  updateWhenIdle: ['cancel'],
  // The task-update poll of a RUNNING task (metainfo block in
  // main.js): the daemon resolves the tab's task, runs the task-update
  // agent when due (or when `refresh` is set) and answers with a
  // direct `taskUpdate` that the client-listener relay passes back.
  getTaskUpdate: ['tabId', 'knownSig', 'token', 'refresh'],
  // The right sidebar's Schedule / Apps / Spend subpanels; the direct
  // `cronJobs` / `appsStatus` / `spendReport` replies come back through
  // the client relay.
  getCronJobs: [],
  getAppsStatus: ['refresh'],
  getSpendReport: [],
  // The daemon owns the model-catalog refresh: it spawns
  // kiss.scripts.update_models against ~/.kiss/MODEL_INFO.json and
  // reports progress/failures back over the connection, so the settings
  // panel's "Update Models" button behaves identically in the webview
  // and in a remote browser window.
  updateModels: [],
  // In-page (browser-mic) capture fallback: when the host machine cannot
  // run the wake listener (hostMicUnavailable) but the webview's embedder
  // grants getUserMedia, voice.js records the utterance itself and ships
  // it here exactly like the remote webapp does over its WebSocket. The
  // daemon transcribes (`voice_transcribe`) and answers with a direct
  // `voiceSpeech` that the client-listener relay above passes straight
  // back to the webview.
  voiceTranscribe: ['audio', 'wakePrefixed', 'wakeSamples'],
  // The daemon machine's browser streamed as a tab (browserTab.js):
  // opening a page, closing it, address-bar actions, replayed input and
  // the size of the surface showing it.  `tab_id` is the browser page,
  // never a chat tab.
  browserOpen: ['url'],
  browserClose: ['tab_id'],
  browserNavigate: ['tab_id', 'action', 'url'],
  browserInput: ['tab_id', 'event'],
  browserViewport: ['tab_id', 'width', 'height', 'visible'],
  // The remote webapp's terminal tab (terminalTab.js).  Remote-only:
  // the daemon drops these from a VS Code window, which has its own
  // integrated terminal; listed so a forwarded message is well-formed.
  terminalOpen: ['tab_id', 'cols', 'rows'],
  terminalInput: ['tab_id', 'data'],
  terminalResize: ['tab_id', 'cols', 'rows'],
  terminalClose: ['tab_id'],
};

/**
 * The bash that runs the Update terminal: `/bin/bash` on POSIX; on Windows
 * the Git for Windows bash (`Git\\bin\\bash.exe` under Program Files, the
 * per-user Git install, or the MinGit the extension installed), never
 * `System32\\bash.exe`, which is the WSL launcher and would run install.sh
 * inside a different filesystem.  Returns null when no bash exists.
 */
export function updateShellPath(): string | null {
  if (process.platform !== 'win32') return '/bin/bash';
  const homeDir = process.env.USERPROFILE || os.homedir();
  const roots = [
    process.env.ProgramFiles,
    process.env['ProgramFiles(x86)'],
    process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'Programs'),
  ];
  const candidates = roots
    .filter((r): r is string => !!r)
    .map(r => path.join(r, 'Git', 'bin', 'bash.exe'));
  candidates.push(path.join(homeDir, '.local', 'git', 'bin', 'bash.exe'));
  for (const dir of (process.env.PATH || '').split(path.delimiter)) {
    if (dir && !/\\System32$/i.test(dir)) {
      candidates.push(path.join(dir, 'bash.exe'));
    }
  }
  return candidates.find(c => fs.existsSync(c)) ?? null;
}

/**
 * The real directory behind *p* (`..` segments and symlinks resolved),
 * or '' when *p* is empty or not an existing directory.
 */
function realDirectory(p: string): string {
  try {
    return p && fs.statSync(p).isDirectory() ? fs.realpathSync(p) : '';
  } catch {
    return '';
  }
}

export class SorcarSidebarView implements vscode.WebviewViewProvider {
  private _view?: ChatWebviewHost;
  private _panelHooks?: PanelHooks;
  // The notification poster this controller installed, if any, so
  // teardown clears only its own installation (see
  // clearWebviewNotificationPoster).
  private _installedPoster?: (message: ToWebviewMessage) => void;
  private _client: AgentClient | null = null;
  private _daemonConnected: boolean = false;
  private _activeTabId: string = '';
  // Task Info view only (meta-panel-mode): the last relayed metaState,
  // kept so a webview that resolves (or reloads) after the relay can
  // be brought up to date on its `ready`.
  private _lastMetaState?: Extract<ToWebviewMessage, {type: 'metaState'}>;
  // Task Info view only (meta-panel-mode): a run start (showForRun)
  // relayed before the webview loaded — the view is revealed for the
  // start and resolves after the relay — waits for its `ready`.
  private _showForRunPending = false;
  // History panel only (history-panel-mode): the last relayed
  // activeTask, replayed on `ready` for the same reason.
  private _lastActiveTask?: Extract<ToWebviewMessage, {type: 'activeTask'}>;
  /**
   * Called with the raw chat / task ids of the task this view's chat
   * webview shows whenever they change (its `activeTask` message). The
   * panel manager sets it on every editor panel's controller and
   * extension.ts on the sidebar chat view, so the ids of the surface
   * on screen reach the primary-sidebar history panel (postActiveTask).
   */
  public onActiveTask?: (chatId: string, taskId: string) => void;
  /**
   * Task Info view only (meta-panel-mode): called when the view's
   * task-update refresh button is pressed (its `metaRefresh` message).
   * extension.ts relays it to the active chat editor panel, which
   * polls the daemon with `refresh: true`.
   */
  public onMetaRefresh?: () => void;
  private _extensionUri: vscode.Uri;
  private _selectedModel: string;
  private _runningTabs: Set<string> = new Set();
  private _ownTabs: Set<string> = new Set();
  private _webviewHasFocus: boolean = false;
  private _webviewReady: boolean = false;
  // A prompt to submit as soon as the webview is ready (see
  // submitWhenReady); '' when none is waiting.
  private _pendingSubmit = '';

  private _voiceWake: VoiceWakeService | undefined;
  private _voiceSensitivity: number | undefined;
  // The user's last voiceToggle choice.  `_voiceWake.running` cannot
  // stand in for it: a stopped listener keeps counting as running while
  // it is dying (it holds the exclusive microphone until its exit event
  // fires), so "running" conflates "the user wants voice on" with "the
  // process has not exited yet".  Restart decisions — sensitivity
  // changes, hide/show suspension — follow this intent, never the
  // physical process state; otherwise a sensitivity value or a hide
  // arriving during that dying window turned the mic back ON after the
  // user had switched it off.
  private _voiceEnabled: boolean = false;
  private _voiceWakeSuspendedByHide: boolean = false;

  private _onCommitMessage = new vscode.EventEmitter<{
    message: string;
    error?: string;
    tabId?: string;
  }>();
  public readonly onCommitMessage = this._onCommitMessage.event;
  private _commitPendingTabs: Set<string> = new Set();
  private _worktreeDirs: Map<string, string> = new Map();
  // Tab ids listed by the daemon's last canonical `tabs_state`
  // snapshot. A tab that drops out of the snapshot was closed by
  // another client — that close never echoes back through this
  // webview as a `closeTab` message, so it is detected here and the
  // host releases the tab's resources (worktree fallback dir,
  // running/commit flags) instead of holding them forever.
  private _registryTabs: Set<string> = new Set();
  // The last canonical snapshot's full entries, so the extension can
  // materialize editor-tab panels for the registry's tabs when the
  // user switches editor-tabs mode on (see getRegistryTabEntries).
  private _registryEntries: Map<string, RegistryTabEntry> = new Map();
  // Whether a canonical `tabs_state` snapshot has been received yet.
  // The FIRST snapshot is a baseline only: its tabs predate this
  // session (a reloaded window may still hold them as serialized
  // editor-tab placeholders the panel manager has not adopted yet);
  // the delta's firstSnapshot flag tells the subscriber.
  private _seenTabsState: boolean = false;
  private _onRegistryTabsState = new vscode.EventEmitter<RegistryTabsDelta>();
  /**
   * Fires on every canonical `tabs_state` snapshot with the full list
   * and the delta of tabs another client created — or first bound to
   * a chat, which is how a task run in the remote web app surfaces on
   * an idle tab. Editor-tabs mode mirrors added tabs as editor tabs
   * the same way sidebar mode's webview adopts them into its internal
   * strip, and syncs open panels' chat bindings from the full list.
   */
  public readonly onRegistryTabsState = this._onRegistryTabsState.event;
  private _worktreeActionResolves: Map<string, () => void> = new Map();
  private _worktreeProgresses: Map<
    string,
    vscode.Progress<{message?: string}>
  > = new Map();
  private _disposed: boolean = false;
  // Set once by dispose() and never cleared: unlike _disposed (which tracks
  // the current webview's lifecycle and is reset by resolveWebviewView),
  // this flag marks terminal teardown of the whole provider.  After it is
  // set, nothing may reconnect a client or register new listeners.
  private _terminated: boolean = false;
  // The current webview's event registrations, retained so dispose() can
  // detach them; otherwise a late queued webview message could reach
  // _handleMessage() after terminal teardown.
  private _viewSubs: vscode.Disposable[] = [];
  private _lastSeenRemotePassword: string | undefined;
  private _configFileWatchTimer?: ReturnType<typeof setInterval>;
  private _onFirstResolve: (() => void) | undefined;
  private _sizeReportResolver:
    ((s: {inner: number; screen: number}) => void) | undefined;
  // The daemon's global working directory, as last reported by
  // `configData` / `workDirChanged`; every task runs there.
  private _daemonWorkDir = '';
  // One rendered-HTML or PDF tab per file path, mirroring the remote web
  // app's content tabs: a second click on the same link reveals (and
  // refreshes) the existing tab instead of stacking duplicates.
  private _htmlPreviewPanels: Map<string, vscode.WebviewPanel> = new Map();

  private _showActionProgress(
    title: string,
    tabId: string | undefined,
    progressMap: Map<string, vscode.Progress<{message?: string}>>,
    resolveMap: Map<string, () => void>,
    timeoutMs: number | undefined = 120_000,
  ): void {
    // An action sent without a tab id is registered under '' — the same
    // key the daemon normalizes an omitted command tabId to and echoes
    // on worktree_progress/worktree_result — so the terminal result,
    // replacement by a newer action, disconnect, and dispose() all reach
    // this resolver.  The timeout below is only a safety net.
    const key = tabId ?? '';
    const prev = resolveMap.get(key);
    if (prev) {
      resolveMap.delete(key);
      progressMap.delete(key);
      prev();
    }
    // Publish the lifecycle entry SYNCHRONOUSLY, before deferring into
    // the notification wrapper.  The production webview poster invokes
    // the task below only in a microtask (Promise.resolve().then(task)
    // in WebviewNotifications.ts), so a second same-key action — or a
    // terminal worktree_result / disconnect / dispose() — arriving in
    // the SAME turn must already find this resolver in the map.  If it
    // were registered only inside the task, both same-key calls would
    // pass the replacement check above, the second task would overwrite
    // the first resolver, and the first toast would stay open for ever
    // (its safety-net timer's identity guard no longer matching).
    let settled = false;
    let ownProgress: vscode.Progress<{message?: string}> | undefined;
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    let resolveDone: () => void = () => {};
    const done = new Promise<void>(resolve => {
      resolveDone = resolve;
    });
    const settle = (): void => {
      if (settled) return;
      settled = true;
      // Cancel the safety-net timer on EVERY settlement path
      // (replacement, terminal result, disconnect, tab cleanup,
      // dispose(), or the timer itself).  Otherwise each early-settled
      // action leaves a referenced 120s timer alive, and those obsolete
      // timers alone keep an otherwise idle extension-host process
      // running for up to two minutes.
      if (timeoutHandle !== undefined) {
        clearTimeout(timeoutHandle);
        timeoutHandle = undefined;
      }
      resolveDone();
    };
    resolveMap.set(key, settle);
    if (
      timeoutMs !== undefined &&
      Number.isFinite(timeoutMs) &&
      timeoutMs > 0
    ) {
      timeoutHandle = setTimeout(() => {
        if (resolveMap.get(key) === settle) {
          resolveMap.delete(key);
          // Drop the progress object too: a late progress event must
          // not report() into a toast that has already been closed.
          if (
            ownProgress !== undefined &&
            progressMap.get(key) === ownProgress
          ) {
            progressMap.delete(key);
          }
          settle();
        }
      }, timeoutMs);
    }
    withWebviewNotificationProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title,
        // The action belongs to one task's tab: its progress toast
        // must not pop over another task's chat.
        tabId,
      },
      progress => {
        // This may run a microtask after the entry was published.  If
        // the entry was superseded or settled meanwhile, do not touch
        // the maps: `done` is already resolved and closes this toast
        // at once.
        if (!settled && resolveMap.get(key) === settle) {
          progressMap.set(key, progress);
          ownProgress = progress;
        }
        return done;
      },
    );
  }

  private _resolveAllWorktreeActions(): void {
    for (const resolve of this._worktreeActionResolves.values()) resolve();
    this._worktreeActionResolves.clear();
    this._worktreeProgresses.clear();
  }

  public onFirstResolve(cb: () => void): void {
    this._onFirstResolve = cb;
  }

  public syncWorkDir(): void {
    this._getClient();
  }

  constructor(extensionUri: vscode.Uri, panelHooks?: PanelHooks) {
    this._extensionUri = extensionUri;
    this._panelHooks = panelHooks;
    if (panelHooks) {
      // The panel's root chat tab is this controller's own from the
      // start: broadcasts stamped with it (commit messages, worktree
      // results) must be treated as this window's even before the
      // webview has sent any message carrying the id.
      this._ownTabs.add(panelHooks.rootTabId);
    }
    const configured = vscode.workspace
      .getConfiguration('kissSorcar')
      .get<string>('defaultModel');
    if (configured) {
      this._selectedModel = configured;
    } else {
      // Never block on `uv run` here: this runs on the extension host's
      // event loop during activation (and for every new panel in
      // editor-tabs mode).  Start from a spawn-free guess and adopt the
      // real default when it arrives -- unless the daemon's `models`
      // reply or the user has picked a model meanwhile.
      const provisional = provisionalDefaultModel();
      this._selectedModel = provisional;
      resolveDefaultModel().then(
        model => {
          if (this._terminated || this._selectedModel !== provisional) return;
          this._selectedModel = model;
        },
        err => console.error('[SorcarSidebarView] default model lookup:', err),
      );
    }
  }

  /** Send *cmd* to the daemon (a no-op after dispose, see _getClient). */
  private _send(cmd: AgentCommand): void {
    this._getClient().sendCommand(cmd);
  }

  private _getClient(): AgentClient {
    if (this._client) return this._client;
    const client = new AgentClient();
    if (this._terminated) {
      // dispose() already ran.  Hand back an inert (disposed) client whose
      // connect() is a no-op so a stray late caller cannot resurrect the
      // daemon connection or register new listeners after teardown.
      client.dispose();
      return client;
    }
    this._client = client;
    this._installClientListener(client);
    // The working directory is ONE global value every task on every
    // surface runs in (config.json work_dir; changed from the "Working
    // directory" panel or the remote Explorer).  The window's workspace
    // folder only seeds it on a fresh install: `ifUnset` makes the
    // daemon ignore this once a directory is persisted, so opening a
    // window never overrides the user's pick.
    client.setPreamble({
      type: 'setWorkDir',
      workDir: this._workspaceDir(),
      ifUnset: true,
    });
    client.on('connect', () => {
      if (!this._panelHooks) {
        // The window's ONE long-lived controller (the sidebar view;
        // panels and the history panel carry panelHooks) asks for the
        // canonical registry snapshot on every (re)connect. The daemon
        // otherwise broadcasts one only after mutations and webview
        // `ready`s, so without this an editor-tabs window with no open
        // chat webview would have no baseline — the next remote task's
        // own mutation would be its FIRST snapshot — and a reconnect
        // would never learn of tabs created during the outage.
        client.sendCommand({type: 'getTabsState'});
      }
      this._daemonConnected = true;
      this._sendToWebview({type: 'daemonStatus', connected: true});
      if (this._view) {
        this._send({type: 'getModels'});
        this._send({type: 'getInputHistory'});
        this._send({type: 'getConfig'});
        // The settings panel's Custom Models list would otherwise stay
        // empty/stale in a panel left open across a daemon outage.
        this._send({type: 'getMyModels'});
      }
    });
    client.on('disconnect', () => {
      this._daemonConnected = false;
      this._sendToWebview({type: 'daemonStatus', connected: false});
      this._resolveAllWorktreeActions();
      // A commit-message request already sent to the daemon that died
      // gets no `commitMessage` or `status` back; settle it now rather
      // than after the 30 s safety timer in generateCommitMessage().
      // (`fire` settles the waiter, which removes the id from the set.)
      for (const tabId of [...this._commitPendingTabs]) {
        this._onCommitMessage.fire({
          message: '',
          error: 'The agent was unreachable',
          tabId,
        });
      }
    });
    client.on(
      'commandDropped',
      (cmd: AgentCommand, reason: DroppedCommandReason) => {
        this._handleDroppedCommand(cmd, reason);
      },
    );
    client.connect();
    return client;
  }

  /**
   * Settle the UI of a command the daemon never received.
   *
   * A submitted prompt waits for the daemon's `status` to show its tab
   * as running (or its `promptOpened` / follow-up handling); the
   * webview meanwhile holds it as the tab's unacknowledged draft.  Only
   * the daemon ever answers -- so a command the client gives up on
   * would leave the tab waiting for ever, with no agent behind it and
   * no word to the user.  A `status running:false` settles the tab and
   * a notification says what happened.
   *
   * @param cmd The command that was never delivered.
   * @param reason Why the client gave up on it.
   */
  private _handleDroppedCommand(
    cmd: AgentCommand,
    reason: DroppedCommandReason,
  ): void {
    const dropped = cmd as {type?: string; tabId?: string};
    const tabId = dropped.tabId;
    if (dropped.type === 'submit') {
      this._sendToWebview({type: 'status', running: false, tabId});
      const why =
        reason === 'expired'
          ? 'the agent was unreachable for too long'
          : 'too many requests were waiting';
      showWarningNotification(
        `Your request was not started because ${why}. Send it again.`,
        {tabId},
      );
      return;
    }
    if (dropped.type === 'generateCommitMessage') {
      // Its promise, its countdown and the SCM input box are all
      // waiting on an answer that is never coming.
      this._onCommitMessage.fire({
        message: '',
        error: 'The agent was unreachable',
        tabId: tabId ?? '',
      });
      return;
    }
    if (
      this._panelHooks &&
      tabId === this._panelHooks.rootTabId &&
      (dropped.type === 'openTab' || dropped.type === 'resumeSession')
    ) {
      // The panel's registry registration is gone for good: the daemon
      // never received it and the webview only re-sends `ready` (not
      // the resume) on reconnect. The panel manager must release the
      // panel's chat claim, or a registry tab another client bound to
      // the same chat would be skipped forever as a "duplicate" of a
      // panel that can never actually bind.
      this._panelHooks.onEvent({kind: 'registrationDropped'});
    }
  }

  private _installClientListener(client: AgentClient): void {
    client.on('message', (msg: ToWebviewMessage) => {
      if (msg.type === 'configData' && msg.config) {
        const wd = msg.config.work_dir;
        if (typeof wd === 'string' && wd) this._daemonWorkDir = wd;
      } else if (msg.type === 'workDirChanged' && msg.workDir) {
        this._daemonWorkDir = msg.workDir;
      }
      if (msg.type === 'commitMessage' && this._isOwnTab(msg.tabId)) {
        this._onCommitMessage.fire({
          message: msg.message,
          error: msg.error,
          tabId: msg.tabId ?? '',
        });
      }
      if (msg.type === 'models' && msg.selected) {
        this._selectedModel = msg.selected;
      }
      if (msg.type === 'openSubagentTab') {
        const subMsg = msg as {tab_id?: string; parent_tab_id?: string};
        if (
          subMsg.tab_id &&
          (subMsg.parent_tab_id
            ? this._ownTabs.has(subMsg.parent_tab_id)
            : this._ownTabs.has(subMsg.tab_id))
        ) {
          this._ownTabs.add(subMsg.tab_id);
        }
      }
      if (msg.type === 'closeSubagentTab' || msg.type === 'subagentDone') {
        // A sub-agent tab closes on every surface as soon as the
        // sub-agent finishes; it never appears in `tabs_state`, so this
        // is the only place its host-side bookkeeping is released
        // (otherwise stopTask() would later `stop` a tab that is gone).
        if (msg.tab_id) {
          this._ownTabs.delete(msg.tab_id);
          this._cleanupTabResources(msg.tab_id);
        }
      }
      if (msg.type === 'worktree_created' || msg.type === 'worktree_done') {
        // The directory is remembered only to close the worktree in
        // the SCM view when its branch is merged (worktree_result);
        // file-path resolution against a pending worktree is the
        // daemon's job (RemoteAccessServer._resolve_tab_file).
        const dir = msg.worktreeDir;
        if (dir && this._isOwnTab(msg.tabId)) {
          if (msg.tabId !== undefined) this._worktreeDirs.set(msg.tabId, dir);
          void this._openWorktreeInScm(dir);
        }
      }
      if (msg.type === 'openResolvedFile') {
        // The daemon's answer to this window's `openFile` (or path-only
        // `submit`): the path it resolved for the tab, which this host
        // opens in a real editor tab — the browser gets `fileContent`
        // instead. Direct reply, so no _isOwnTab gate is needed.
        if (msg.error) {
          console.warn(`[kissSorcar] ${msg.error}`);
        } else if (msg.path) {
          // The editor can still refuse a file the daemon resolved
          // (binary, oversized); keep that out of the socket listener.
          this._openResolvedFile(msg.path, msg.line).catch(err =>
            console.error('[kissSorcar] Failed to open file:', err),
          );
        }
        return;
      }
      if (msg.type === 'tabs_state' && Array.isArray(msg.tabs)) {
        // The snapshot is canonical and complete: a tab it no longer
        // lists was closed — possibly by another client, whose close
        // never reaches this host as a `closeTab` webview message. The
        // webview drops such tabs in reconcileTabs(); the host must
        // release its per-tab resources too, or a dead tab's worktree
        // fallback dir / running flags would linger for the session
        // (and could leak onto a later tab reusing the same id).
        // Sub-agent and other client-local tabs never appear in
        // snapshots, so only ids seen in a previous snapshot are
        // eligible for pruning.
        const listed = new Set<string>();
        const entries = new Map<string, RegistryTabEntry>();
        for (const t of msg.tabs) {
          if (t && t.tabId) {
            listed.add(t.tabId);
            entries.set(t.tabId, {
              tabId: t.tabId,
              chatId: t.chatId || '',
              title: t.title || '',
              workDir: t.workDir || '',
            });
          }
        }
        for (const staleId of this._registryTabs) {
          if (!listed.has(staleId)) {
            this._ownTabs.delete(staleId);
            this._cleanupTabResources(staleId);
          }
        }
        // Tabs another client created — or first bound to a chat (a
        // task run allocates the chat id) — since the previous
        // snapshot: e.g. a task run in the remote web app. On the
        // first snapshot every listed tab counts as added (the daemon
        // only broadcasts on mutations and webview `ready`s, so this
        // may itself be the remote run's mutation); the subscriber
        // uses the firstSnapshot flag to decide whether adopting is
        // safe (see RegistryTabsDelta).
        const added: RegistryTabEntry[] = [];
        const firstSnapshot = !this._seenTabsState;
        for (const [id, entry] of entries) {
          const prev = this._registryEntries.get(id);
          if (!prev || (entry.chatId && !prev.chatId)) added.push(entry);
        }
        this._seenTabsState = true;
        this._registryTabs = listed;
        this._registryEntries = entries;
        this._onRegistryTabsState.fire({
          added,
          listed: [...entries.values()],
          firstSnapshot,
        });
        if (this._panelHooks) {
          // Report the root tab's chat binding so the panel manager
          // can route a history open of the same chat to this panel
          // instead of stacking a second one.
          const own = entries.get(this._panelHooks.rootTabId);
          if (own?.chatId) {
            this._panelHooks.onEvent({kind: 'chatBound', chatId: own.chatId});
          }
        }
      }
      if (msg.type === 'worktree_progress') {
        const wpTabId = msg.tabId;
        const progress =
          wpTabId !== undefined
            ? this._worktreeProgresses.get(wpTabId)
            : this._worktreeProgresses.values().next().value;
        if (progress) {
          progress.report({message: msg.message});
        }
      }
      if (msg.type === 'worktree_result' && msg.success && !msg.kept) {
        // A merge/discard finished by any client retires the worktree
        // directory this window opened in the SCM view. A "Do nothing"
        // result (kept: true) leaves the worktree on disk, so it stays
        // open.
        const doneTabId = msg.tabId;
        if (doneTabId !== undefined) {
          const doneDir = this._worktreeDirs.get(doneTabId);
          if (doneDir) {
            void this._closeWorktreeInScm(doneDir);
            this._worktreeDirs.delete(doneTabId);
          }
        }
      }
      if (msg.type === 'worktree_result' && this._isOwnTab(msg.tabId)) {
        const wrTabId = msg.tabId;
        if (wrTabId !== undefined) {
          const resolve = this._worktreeActionResolves.get(wrTabId);
          if (resolve) {
            resolve();
            this._worktreeActionResolves.delete(wrTabId);
          }
          this._worktreeProgresses.delete(wrTabId);
        } else {
          this._resolveAllWorktreeActions();
        }
        // The outcome is one task's text, so the toast carries the
        // task's tab: the webview shows it only over that tab, and the
        // panel manager posts it to that tab's panel, never to whatever
        // chat the user is looking at now.
        const wrToast = {tabId: msg.tabId};
        if (msg.success) {
          if (!isSilentDiscardMessage(msg.message)) {
            showInformationNotification(
              msg.message || 'Worktree action completed.',
              wrToast,
            );
          }
        } else {
          showErrorNotification(
            msg.message || 'Worktree action failed.',
            wrToast,
          );
        }
      }
      if (msg.type === 'main_tree_result' && this._isOwnTab(msg.tabId)) {
        // Same toast rule as worktree_result above: the post-task
        // main-tree bar's Discard / Do nothing outcome is surfaced as
        // a notification (the webview renders the transcript line).
        const mtToast = {tabId: msg.tabId};
        if (msg.success) {
          showInformationNotification(
            msg.message || 'Main-tree action completed.',
            mtToast,
          );
        } else {
          showErrorNotification(
            msg.message || 'Main-tree action failed.',
            mtToast,
          );
        }
      }
      if (
        msg.type === 'autocommit_done' &&
        this._isOwnTab(msg.tabId) &&
        // A manual Git Commit already broadcasts its own toast
        // notification from the daemon; toasting here too would
        // show the same outcome twice.
        !msg.manual
      ) {
        const acToast = {tabId: msg.tabId};
        if (msg.success) {
          showInformationNotification(
            msg.message || 'Auto-commit completed.',
            acToast,
          );
        } else {
          showErrorNotification(msg.message || 'Auto-commit failed.', acToast);
        }
      }

      // A question raised by a still-running task must not steal focus: the
      // webview flags the waiting tab instead, so the user decides when to
      // answer it.

      this._sendToWebview(msg);
      if (msg.type === 'status') {
        const statusTabId = msg.tabId;
        if (msg.running) {
          if (statusTabId !== undefined && this._ownTabs.has(statusTabId)) {
            this._runningTabs.add(statusTabId);
          }
        } else {
          if (statusTabId !== undefined) this._runningTabs.delete(statusTabId);
          if (
            this._isOwnTab(statusTabId) &&
            this._commitPendingTabs.has(statusTabId ?? '')
          ) {
            this._onCommitMessage.fire({
              message: '',
              error: 'Process stopped',
              tabId: statusTabId ?? '',
            });
          }
        }
      }
    });
  }

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ): void {
    this.attachWebviewHost(
      {
        webview: webviewView.webview,
        get visible() {
          return webviewView.visible;
        },
        show: () => webviewView.show(true),
        onDidChangeVisibility: webviewView.onDidChangeVisibility,
        onDidDispose: webviewView.onDidDispose,
      },
      this._panelHooks?.bodyAttrs,
    );
  }

  /**
   * Bind this controller to a chat webview surface — the secondary
   * sidebar's view (via resolveWebviewView) or an editor tab's panel
   * (editor-tabs mode) — building the chat HTML and wiring message,
   * visibility and dispose handling.
   *
   * @param host The surface to drive.
   * @param bodyAttrs Extra `<body>` attributes for the chat HTML
   *     (editor-tabs mode's class and data attributes).
   */
  attachWebviewHost(host: ChatWebviewHost, bodyAttrs?: string): void {
    if (this._terminated) return;
    for (const sub of this._viewSubs) sub.dispose();
    this._viewSubs = [];
    const webviewView = host;
    this._view = webviewView;
    this._webviewReady = false;
    // A fresh webview has not reported focus yet; a stale true here
    // (left by a disposed webview) would make toggleFocus believe the
    // chat is focused and never focus it.
    this._webviewHasFocus = false;
    if (!this._panelHooks) {
      // In editor-tabs mode the panel manager owns the shared toast
      // poster (it routes to the active panel); a per-panel controller
      // installing its own would steal every other panel's toasts.
      const poster = (message: ToWebviewMessage) => {
        this._sendToWebview(message);
      };
      this._installedPoster = poster;
      setWebviewNotificationPoster(poster);
    }
    this._disposed = false;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(this._extensionUri, 'media'),
        vscode.Uri.joinPath(this._extensionUri, 'out'),
      ],
    };

    webviewView.webview.html = buildChatHtml(
      webviewView.webview,
      this._extensionUri,
      this._selectedModel,
      bodyAttrs,
    );

    this._viewSubs.push(
      webviewView.webview.onDidReceiveMessage((message: FromWebviewMessage) => {
        // _handleMessage is async and can reject (e.g. openTextDocument on
        // a binary or oversized file); an unhandled rejection here would
        // otherwise escape into the extension host.
        this._handleMessage(message).catch(err =>
          console.error('[SorcarSidebarView] message handling failed:', err),
        );
      }),
    );

    const visibilitySub = webviewView.onDidChangeVisibility(() => {
      if (this._view !== webviewView) return;
      // audit0903-coverage:start
      if (webviewView.visible) {
        this._send({type: 'getInputHistory'});
        if (this._voiceWakeSuspendedByHide) {
          this._voiceWakeSuspendedByHide = false;
          this._voiceWake?.start(this._voiceSensitivity);
        }
      } else if (this._voiceEnabled && this._voiceWake?.running) {
        // Gate on the user's intent as well: `running` alone stays true
        // while a listener the user just switched OFF is still dying,
        // and latching the suspend flag for it made the next show
        // restart it.
        this._voiceWakeSuspendedByHide = true;
        void this._voiceWake.stop();
      }
      // audit0903-coverage:end
    });
    this._viewSubs.push(visibilitySub);

    this._viewSubs.push(
      webviewView.onDidDispose(() => {
        if (this._view === webviewView) {
          this._view = undefined;
          this._disposed = true;
          this._webviewReady = false;
          // A disposed webview cannot hold focus; without this reset a
          // view disposed while focused leaves hasFocus stuck true and
          // the toggleFocus keybinding can never refocus the chat.
          this._webviewHasFocus = false;
          if (this._installedPoster) {
            clearWebviewNotificationPoster(this._installedPoster);
            this._installedPoster = undefined;
          }
          this._voiceWakeSuspendedByHide = false;
          // Voice stays off until the next webview toggles it back on:
          // without this a fresh webview's first voiceSensitivity would
          // start the microphone before any voiceToggle.
          // audit0903-coverage:start
          this._voiceEnabled = false;
          void this._voiceWake?.stop();
          // audit0903-coverage:end
        }
        this._resolveAllWorktreeActions();
      }),
    );

    if (this._onFirstResolve) {
      const cb = this._onFirstResolve;
      this._onFirstResolve = undefined;
      cb();
    }
  }

  get visible(): boolean {
    return this._view?.visible ?? false;
  }

  get hasFocus(): boolean {
    return this._webviewHasFocus;
  }

  /**
   * The directory a path-taking webview message resolves against when
   * it names none: the daemon's global working directory (where every
   * task runs), or the window's own folder before the daemon reported
   * one.
   */
  private _getWorkDir(): string {
    return this._daemonWorkDir || this._workspaceDir();
  }

  /** The window's workspace folder, or the host cwd unless that is a root. */
  private _workspaceDir(): string {
    const folders = vscode.workspace.workspaceFolders;
    if (folders && folders.length > 0) {
      return folders[0].uri.fsPath;
    }
    // No folder open: fall back to the extension host's cwd, which is
    // useful when VS Code was launched from a terminal (`code file.txt`
    // inherits the shell's directory).  A Dock/Finder-launched window
    // instead inherits the filesystem root ('/'), and reporting THAT as
    // the work dir would root every task started from this window — and
    // the daemon's @-mention file picker — at the whole disk.  Report
    // "no work dir" for a root cwd so the daemon falls back to its
    // configured folder instead.
    const cwd = process.cwd();
    return path.parse(cwd).root === cwd ? '' : cwd;
  }

  private _sendToWebview(message: ToWebviewMessage): void {
    if (!this._disposed && this._view) {
      this._view.webview.postMessage(message);
    }
  }

  private _watchConfigFile(): void {
    if (this._configFileWatchTimer) return;
    this._checkConfigFile();
    this._configFileWatchTimer = setInterval(
      () => this._checkConfigFile(),
      2_000,
    );
  }

  private _checkConfigFile(): void {
    const configFile = path.join(kissHomeDir(), 'config.json');
    let pw: string;
    try {
      const data = JSON.parse(fs.readFileSync(configFile, 'utf-8'));
      pw = typeof data.remote_password === 'string' ? data.remote_password : '';
    } catch {
      return;
    }
    const first = this._lastSeenRemotePassword === undefined;
    const changed = pw !== this._lastSeenRemotePassword;
    this._lastSeenRemotePassword = pw;
    if ((changed && !first) || (first && pw !== '')) {
      this._send({type: 'getConfig'});
    }
  }

  private _getVisibleEditorFile(): string {
    const activeEditor = vscode.window.activeTextEditor;
    if (activeEditor) {
      return activeEditor.document.uri.fsPath;
    }
    for (const group of vscode.window.tabGroups.all) {
      const activeTab = group.activeTab;
      if (activeTab && activeTab.input instanceof vscode.TabInputText) {
        return activeTab.input.uri.fsPath;
      }
    }
    return '';
  }

  private async _openWorktreeInScm(worktreeDir: string): Promise<void> {
    try {
      const api = await getGitApi();
      if (api?.openRepository) {
        await api.openRepository(vscode.Uri.file(worktreeDir));
      }
    } catch (err) {
      console.error('[kissSorcar] Failed to open worktree in SCM:', err);
    }
  }

  private async _closeWorktreeInScm(worktreeDir: string): Promise<void> {
    try {
      await vscode.commands.executeCommand(
        'git.close',
        vscode.Uri.file(worktreeDir),
      );
    } catch {}
  }

  private _isOwnTab(tabId: string | undefined): boolean {
    return !tabId || this._ownTabs.has(tabId);
  }

  /**
   * True when this webview holds *tabId*: its root tab, a chat tab it
   * opened, or a sub-agent tab nested under one of those.  The panel
   * manager uses it to post a task's toast to the panel that shows the
   * task, not to whichever panel is active.
   *
   * @param tabId The chat tab id to look up.
   */
  ownsTab(tabId: string): boolean {
    return this._ownTabs.has(tabId);
  }

  private async _handleMessage(message: FromWebviewMessage): Promise<void> {
    // A message already queued when dispose() ran must be dropped: handling
    // it could rebuild the daemon client and its listeners after teardown.
    if (this._terminated) return;
    const msgTabId = (message as {tabId?: string}).tabId;
    if (msgTabId) {
      if (message.type === 'closeTab') this._ownTabs.delete(msgTabId);
      else this._ownTabs.add(msgTabId);
    }
    if (message.type === 'ready' && message.restoredTabs) {
      for (const rt of message.restoredTabs) {
        if (rt.tabId) this._ownTabs.add(rt.tabId);
      }
    }
    const forwarded = FORWARDED_COMMANDS[message.type];
    if (forwarded) {
      const src = message as unknown as Record<string, unknown>;
      const cmd: Record<string, unknown> = {type: message.type};
      for (const field of forwarded) cmd[field] = src[field];
      this._send(cmd as unknown as AgentCommand);
      return;
    }
    switch (message.type) {
      case 'ready': {
        this._webviewReady = true;
        const readyTabId = message.tabId;
        if (readyTabId) this._activeTabId = readyTabId;
        this._sendToWebview({
          type: 'daemonStatus',
          connected: this._daemonConnected,
        });
        this._watchConfigFile();
        // The Task Info view (meta-panel-mode): a metaState relayed
        // before the webview loaded — or lost to a webview reload —
        // must not leave the panel on its placeholder dashes.
        if (this._lastMetaState) this._sendToWebview(this._lastMetaState);
        if (this._showForRunPending) {
          this._showForRunPending = false;
          this._sendToWebview({type: 'showForRun'});
        }
        if (this._lastActiveTask) this._sendToWebview(this._lastActiveTask);
        // The daemon owns the canonical tab registry, so `ready` is
        // forwarded whole: the daemon fans out the connId-scoped init
        // replies (models / input history / config), merges any legacy
        // restoredTabs into an empty registry, answers with the
        // canonical `tabs_state` snapshot, and replays every
        // chat-bound tab's transcript to this connection (only the
        // `singleTabId` one for an editor-tab panel).
        this._send({
          type: 'ready',
          tabId: message.tabId,
          restoredTabs: message.restoredTabs,
          singleTabId: message.singleTabId,
        } as AgentCommand);
        if (this._pendingSubmit) {
          this._sendToWebview({
            type: 'insertAndSubmit',
            text: this._pendingSubmit,
          });
          this._pendingSubmit = '';
        }
        break;
      }

      case 'submit':
        if (message.tabId) this._activeTabId = message.tabId;
        // The daemon owns the classification (path-only prompt, follow-up
        // or new run) and runs the task in the global working directory;
        // the host adds only its visible editor file.
        this._send({
          ...message,
          activeFile:
            this._getVisibleEditorFile() || message.activeFile || undefined,
        });
        break;

      case 'stop': {
        const stopTabId = message.tabId;
        if (stopTabId !== undefined) {
          this._send({type: 'stop', tabId: stopTabId});
        } else {
          for (const tab of this._runningTabs) {
            this._send({type: 'stop', tabId: tab});
          }
        }
        break;
      }

      case 'selectModel': {
        this._selectedModel = message.model;
        this._send({
          type: 'selectModel',
          model: message.model,
          tabId: message.tabId,
        });
        break;
      }

      case 'userAnswer': {
        const ansTabId = message.tabId;
        if (ansTabId !== undefined) {
          this._send({
            type: 'userAnswer',
            answer: message.answer,
            tabId: ansTabId,
          });
        }
        break;
      }

      case 'recordFileUsage':
        if (message.path) {
          this._send({
            type: 'recordFileUsage',
            path: message.path,
            workDir: message.workDir,
          });
        }
        break;

      case 'resumeSession': {
        this._send({
          type: 'resumeSession',
          chatId: message.chatId ?? message.id,
          taskId: message.taskId,
          tabId: message.tabId,
        });
        break;
      }

      case 'complete': {
        // The visible VS Code editor wins; without one the webview's
        // own file tab (a Monaco buffer, as in the remote webapp) is
        // the context.
        const editorFile = this._getVisibleEditorFile();
        const completeDoc = editorFile
          ? vscode.workspace.textDocuments.find(
              d => d.uri.fsPath === editorFile,
            )
          : undefined;
        this._send({
          type: 'complete',
          query: message.query,
          tabId: message.tabId || this._activeTabId || undefined,
          activeFile: editorFile || message.activeFile || undefined,
          activeFileContent: editorFile
            ? completeDoc?.getText()
            : message.activeFileContent,
        });
        break;
      }

      case 'worktreeAction': {
        const wtAction = message.action;
        const wtTabId = message.tabId;
        // A discard is instant; only a merge is worth a progress toast.
        if (wtAction === 'merge') {
          this._showActionProgress(
            'Committing and merging worktree…',
            wtTabId,
            this._worktreeProgresses,
            this._worktreeActionResolves,
          );
        }
        this._send({type: 'worktreeAction', action: wtAction, tabId: wtTabId});
        break;
      }

      // The post-task bar of a non-worktree manual-commit run: the
      // daemon discards the main tree's uncommitted changes (or, for
      // "nothing", just acknowledges) and reports the outcome through
      // a broadcast main_tree_result event that the webview renders on
      // its own. (The bar's Auto commit button sends autocommitAction.)
      case 'mainTreeAction':
        this._send({
          type: 'mainTreeAction',
          action: message.action,
          tabId: message.tabId,
          workDir: message.workDir || this._getWorkDir(),
        });
        break;

      // The settings panel's "Git Commit" button: the daemon commits
      // the tab's working tree and reports progress and the outcome
      // through broadcast autocommit_progress/autocommit_done events
      // that the webview renders on its own.
      case 'autocommitAction':
        this._send({
          type: 'autocommitAction',
          tabId: message.tabId,
          workDir: message.workDir || this._getWorkDir(),
        });
        break;

      case 'resolveDroppedPaths': {
        const workDir = message.workDir || this._getWorkDir();
        const paths = (message.uris || [])
          .map((uri: string) => {
            try {
              const absPath = vscode.Uri.parse(uri).fsPath;
              return path.relative(workDir, absPath);
            } catch {
              return '';
            }
          })
          // On Windows path.relative() across drives returns an ABSOLUTE
          // path that does not start with '..'; reject those too.
          .filter(
            (p: string) => p && !p.startsWith('..') && !path.isAbsolute(p),
          );
        this._sendToWebview({
          type: 'droppedPaths',
          paths,
          // Echo the owner: the webview rejects the reply if the user
          // switched tabs during this round trip.
          tabId: message.tabId,
        } as ToWebviewMessage);
        break;
      }

      case 'webviewFocusChanged':
        this._webviewHasFocus = message.focused;
        break;

      // Which chat tab the user is looking at.
      case 'activeTabChanged':
        this._activeTabId = message.tabId;
        break;

      case 'voiceToggle': {
        if (!this._voiceWake) {
          this._voiceWake = new VoiceWakeService(
            roundId => this._sendToWebview({type: 'voiceWake', roundId}),
            (listening, error, hostMicUnavailable) => {
              if (hostMicUnavailable) {
                // This machine cannot run the wake listener at all (it
                // died before READY — e.g. `OSError: PortAudio library
                // not found` on a mic-less remote host). That is a
                // property of the machine, not a runtime error: the
                // webview shows a calm "voice capture unavailable"
                // state instead of a red error, and the user's voice
                // intent is cleared so hide/show and sensitivity
                // changes never respawn the doomed listener. The
                // detail still goes to the extension-host log.
                console.warn('KISS voice listener unavailable:', error);
                this._voiceEnabled = false;
                this._sendToWebview({
                  type: 'voiceState',
                  listening: false,
                  hostMicUnavailable: true,
                });
                return;
              }
              this._sendToWebview({type: 'voiceState', listening, error});
            },
            (roundId, text, speaker, language) =>
              this._sendToWebview({
                type: 'voiceSpeech',
                roundId,
                text,
                speaker,
                language,
              }),
            () => this._sendToWebview({type: 'voiceTranscribing'}),
          );
        }
        if (typeof message.sensitivity === 'number') {
          this._voiceSensitivity = message.sensitivity;
        }
        // audit0903-coverage:start
        this._voiceWakeSuspendedByHide = false;
        this._voiceEnabled = !!message.enabled;
        if (this._voiceEnabled) this._voiceWake.start(this._voiceSensitivity);
        else void this._voiceWake.stop();
        // audit0903-coverage:end
        break;
      }

      case 'voiceAck': {
        playVoiceAckClip(
          path.join(this._extensionUri.fsPath, 'media', 'working-on-it.mp3'),
        );
        break;
      }

      // The user switched chat tabs while speaking, so the transcript was
      // never typed anywhere. Say so instead of losing the words silently.
      case 'voiceDropped': {
        if (typeof vscode.window.showWarningMessage !== 'function') break;
        void vscode.window.showWarningMessage(
          'Speech discarded because the chat tab changed while you spoke: ' +
            message.text,
        );
        break;
      }

      case 'voiceSensitivity': {
        // audit0902-coverage:start
        // audit0903-coverage:start
        if (typeof message.value !== 'number') break;
        this._voiceSensitivity = message.value;
        // Restart only when the user wants voice on AND the view is not
        // hidden.  The gate is _voiceEnabled, not _voiceWake.running: a
        // listener the user just switched off still counts as running
        // while it dies, and a running check here restarted it.  While
        // suspended by hide the value is only recorded; the show handler
        // starts the listener with it.
        if (
          this._voiceWake &&
          this._voiceEnabled &&
          !this._voiceWakeSuspendedByHide
        ) {
          // Restart the listener with the new sensitivity.  stop() is
          // asynchronous (on exclusive-capture audio backends the old
          // process holds the microphone until it has exited) and the
          // service queues a start() issued during the stop behind it,
          // so this cannot double-open the mic.  It is deliberately NOT
          // `await stop(); if (!running) start()`: a voiceToggle
          // {enabled:false} arriving during that await shares the same
          // stop promise, and once it settled the check passed and the
          // handler restarted the listener the user had just switched
          // off.  A queued start is cancelled by any later stop(), so
          // the off switch always wins.
          void this._voiceWake.stop();
          this._voiceWake.start(this._voiceSensitivity);
        }
        // audit0903-coverage:end
        // audit0902-coverage:end
        break;
      }

      case 'sizeReport': {
        const cb = this._sizeReportResolver;
        this._sizeReportResolver = undefined;
        if (cb) cb({inner: message.innerWidth, screen: message.screenWidth});
        break;
      }

      case 'focusEditor':
        // In editor-tabs mode the chat IS an editor in the first group,
        // so "back to the editor" means the previously used one.
        vscode.commands.executeCommand(
          this._panelHooks
            ? 'workbench.action.openPreviousRecentlyUsedEditor'
            : 'workbench.action.focusFirstEditorGroup',
        );
        break;

      case 'runUpdate':
        this.runUpdate();
        break;

      case 'serverReset':
        this._send({type: 'serverReset'});
        break;

      case 'notificationAction':
        resolveWebviewNotificationAction(message.id, message.action);
        break;

      case 'closeTab': {
        const closeTabId = message.tabId;
        if (closeTabId) {
          this._cleanupTabResources(closeTabId);
          this._send({type: 'closeTab', tabId: closeTabId});
        }
        break;
      }

      case 'panelTitle':
        this._panelHooks?.onEvent({
          kind: 'title',
          title: message.title,
          state: message.state,
        });
        break;

      case 'metaUpdate':
        this._panelHooks?.onEvent({
          kind: 'metaUpdate',
          values: message.values,
          taskUpdate: message.taskUpdate,
        });
        break;

      case 'metaRefresh':
        this.onMetaRefresh?.();
        break;

      case 'activeTask':
        this.onActiveTask?.(message.chatId, message.taskId);
        break;

      case 'revealPanel':
        this._panelHooks?.onEvent({kind: 'reveal'});
        break;

      case 'revealForQuestion':
        this._revealForQuestion();
        break;

      case 'taskStarted':
        this._showChatForRun();
        break;

      case 'openChatPanel':
        this._panelHooks?.onEvent({
          kind: 'openChat',
          chatId: message.chatId,
          taskId: message.taskId,
          title: message.title,
          pendingText: message.pendingText,
          autoSubmit: message.autoSubmit,
        });
        break;

      case 'closePanel':
        this._panelHooks?.onEvent({
          kind: 'closeSelf',
          retire: !!message.retire,
        });
        break;

      case 'setEditorTabsMode': {
        // The settings UI's editor-tabs toggle. Handled in BOTH modes
        // (the sidebar switches the mode on, a panel switches it off);
        // extension.ts reacts to the configuration change. Written to
        // the most specific scope that already holds a value, so a
        // workspace override cannot silently swallow the toggle.
        const cfg = vscode.workspace.getConfiguration('kissSorcar');
        const info = cfg.inspect?.<boolean>('editorTabsMode');
        const target =
          info?.workspaceFolderValue !== undefined
            ? vscode.ConfigurationTarget.WorkspaceFolder
            : info?.workspaceValue !== undefined
              ? vscode.ConfigurationTarget.Workspace
              : vscode.ConfigurationTarget.Global;
        await cfg.update('editorTabsMode', !!message.enabled, target);
        break;
      }

      case 'openWorkDir':
        this._openWorkDir(message.path);
        break;

      case 'pickWorkDir': {
        const wd = this._getWorkDir();
        const picked = await vscode.window.showOpenDialog({
          canSelectFolders: true,
          canSelectFiles: false,
          canSelectMany: false,
          openLabel: 'Use as Working Directory',
          defaultUri: wd ? vscode.Uri.file(wd) : undefined,
        });
        if (picked && picked[0]) {
          this._openWorkDir(picked[0].fsPath);
        }
        break;
      }
    }
  }

  /**
   * Make *dir*, chosen in the "Working directory" panel, the global
   * working directory every task on every surface runs in.
   *
   * The window keeps its workspace folder (no `vscode.openFolder`).  An
   * existing directory is sent to the daemon as `setWorkDir` (which
   * persists it and broadcasts `workDirChanged` to every client) and
   * answered with `workDirPicked` carrying its real path (`..` and
   * symlinks resolved); a path that is not a directory or is a
   * file-system root is reported back as `workDirError`.
   */
  private _openWorkDir(dir: string): void {
    const target = String(dir || '').trim();
    const real = realDirectory(target);
    if (!real) {
      this._sendToWebview({
        type: 'workDirError',
        text: 'Not a directory: ' + (target || '(empty path)'),
      });
      return;
    }
    if (path.dirname(real) === real) {
      // `/`, `C:\`, a UNC share root: the daemon never runs in one.
      this._sendToWebview({
        type: 'workDirError',
        text: 'A file-system root cannot be the working directory; pick a folder.',
      });
      return;
    }
    this._daemonWorkDir = real;
    this._send({type: 'setWorkDir', workDir: real});
    this._sendToWebview({type: 'workDirPicked', path: real});
  }

  /**
   * The daemon registry's last canonical tab snapshot, one entry per
   * chat tab; empty before the first `tabs_state` arrives.
   */
  public getRegistryTabEntries(): RegistryTabEntry[] {
    return [...this._registryEntries.values()];
  }

  /**
   * Retire *tabId* from the daemon's shared tab registry — the host
   * counterpart of the webview's own `closeTab` message, used when the
   * USER closes an editor-tab panel (the panel's webview is torn down
   * before it could send anything itself).
   */
  public closeChatTab(tabId: string): void {
    if (this._terminated || !tabId) return;
    this._cleanupTabResources(tabId);
    this._send({type: 'closeTab', tabId});
  }

  /** Release every host-side resource owned by a closed tab. */
  private _cleanupTabResources(tabId: string): void {
    this._runningTabs.delete(tabId);
    this._commitPendingTabs.delete(tabId);
    this._worktreeDirs.delete(tabId);
    const wtResolve = this._worktreeActionResolves.get(tabId);
    if (wtResolve) {
      this._worktreeActionResolves.delete(tabId);
      wtResolve();
    }
    this._worktreeProgresses.delete(tabId);
  }

  /**
   * Open a visible "KISS Sorcar Update" terminal whose terminal PROCESS is
   * `bash -c <command>` (see {@link updateShellPath}; plus a signal guard
   * and a hold-open tail),
   * instead of typing the command into an interactive shell with
   * `sendText`.
   *
   * Why not sendText: text sent to a shell prompt is just keystrokes, and
   * other extensions inject keystrokes into every new terminal.  The
   * Python / Python-Environments extension "activates" the workspace venv
   * in each new terminal, and that activation makes VS Code send Ctrl+C
   * first (core clears what it believes is leftover prompt input — see
   * microsoft/vscode#287139) followed by ` source .../activate`.  The
   * Ctrl+C cancelled the update command at the zsh prompt before it ever
   * ran (`^C%` then `source .../.venv/bin/activate` in the user's
   * transcript).  Running the installer AS the terminal process leaves no
   * prompt to stomp on and no shell integration for `executeCommand` to
   * send ^C through.
   *
   * The guard: `trap '' INT TERM HUP` makes the whole install pipeline
   * immune to a stray \x03 that an extension may still write into the PTY
   * (the line discipline turns it into SIGINT for the foreground process
   * group).  Signals ignored at entry stay ignored in non-interactive
   * children (bash, git, curl, uv all honour inherited SIG_IGN), and the
   * root install.sh additionally detaches into its own session.  Any
   * injected activation TEXT lands on the installer's stdin and is never
   * executed.
   *
   * The tail: a terminal with a custom `shellPath` is disposed the moment
   * its process exits, which would wipe the output — including the
   * cross-process lock's "another KISS update is already running (pid N)"
   * refusal and any install error.  On failure the tail holds the pane
   * open until the user presses Enter on an EMPTY line (or stdin hits
   * EOF, so a piped run can never hang): the venv-activation text other
   * extensions inject is a non-empty line and may arrive at any time, so
   * a plain `read` would let it close the pane before the user saw the
   * error.  On success the pane may close: install.sh writes the
   * .extension-updated marker and the window reloads anyway.
   */
  private _openUpdateTerminal(cwd: string, command: string): void {
    const shellPath = updateShellPath();
    if (shellPath === null) {
      vscode.window.showErrorMessage(
        `Updating ${PRODUCT_NAME} runs install.sh under bash, which was not found. ` +
          'Install Git for Windows (it ships bash.exe) and try again.',
      );
      return;
    }
    // audit0902-coverage:start
    const guarded =
      "trap '' INT TERM HUP; " +
      command +
      '; _kiss_rc=$?; ' +
      'if [ "$_kiss_rc" -ne 0 ]; then ' +
      `printf "\\n>>> ${PRODUCT_NAME} update exited with status %s. ` +
      'Press Enter to close this terminal.\\n" "$_kiss_rc"; ' +
      'while IFS= read -r _kiss_enter; do ' +
      'if [ -z "$_kiss_enter" ]; then break; fi; ' +
      'done; ' +
      'fi; exit "$_kiss_rc"';
    const terminal = vscode.window.createTerminal({
      name: `${PRODUCT_NAME} Update`,
      cwd,
      shellPath,
      shellArgs: ['-c', guarded],
    });
    terminal.show();
    // audit0902-coverage:end
  }

  /**
   * Ask the daemon to install the available update once no task is
   * running.  The daemon arms its idle poller and rebroadcasts
   * `update_available` with `pendingIdle`, so every chat window's toast
   * shows the armed state ("Update now" / "Cancel").
   */
  public updateWhenIdle(): void {
    this._send({type: 'updateWhenIdle'});
  }

  public runUpdate(): void {
    // Every click runs the installer; there is deliberately no
    // per-window "already running" guard here.  Whether another
    // installer is still running is known only to the cross-process
    // lock inside install.sh (two windows are two extension hosts, and
    // the daemon's update endpoint is a third caller), and the loser
    // prints "another KISS update is already running (pid N)" in this
    // terminal itself.  A guard keyed on the terminal's lifetime wrongly
    // refused every click after a finished update until the shell was
    // closed.
    // audit0902-coverage:start
    const scriptPath = findInstallScript();
    showInformationNotification(
      `An update of ${PRODUCT_NAME} is getting installed…`,
    );
    if (!scriptPath) {
      // No ~/.kiss/kiss_ai clone with an install.sh on this machine (the
      // extension was installed from a .vsix, or the clone was deleted).
      // Run the public curl bootstrap instead of refusing: it clones the
      // repo into ~/.kiss/kiss_ai (kissAiRoot()) and hands over to its
      // install.sh, holding the same cross-process update lock.  The URL
      // travels via the environment (like KISS_HOME below) so no shell
      // quoting of it is ever needed inside the nested bash -c string.
      const escHome = kissHomeDir().replace(/'/g, "'\\''");
      const escUrl = bootstrapInstallUrl().replace(/'/g, "'\\''");
      this._openUpdateTerminal(
        os.homedir(),
        `KISS_HOME='${escHome}' KISS_NONINTERACTIVE=1 ` +
          `KISS_BOOTSTRAP_URL='${escUrl}' bash -c ` +
          '\'set -o pipefail; curl -fsSL "$KISS_BOOTSTRAP_URL" | bash\'',
      );
      return;
    }
    // audit0902-coverage:end
    const escScript = scriptPath.replace(/'/g, "'\\''");
    const escDir = path.dirname(scriptPath).replace(/'/g, "'\\''");
    // Pin KISS_HOME to the value THIS extension host resolved: install.sh
    // writes the .extension-updated marker into $KISS_HOME, and the reload
    // watcher (extension.ts) watches the extension host's $KISS_HOME.  The
    // terminal process inherits the window's environment, which could
    // carry a different KISS_HOME, and then the marker would land where
    // no watcher looks — the update installs but this window never
    // reloads.
    const escKissHome = kissHomeDir().replace(/'/g, "'\\''");
    // audit0902-coverage:start
    // scripts/install.sh (the curl bootstrap) syncs the clone with origin
    // and hands over to ./install.sh -- exactly this preflight -- but
    // holds a cross-process lock for all of it, so two windows (or a
    // window and the daemon's update endpoint) cannot reset / uv sync /
    // restart the same tree at once: the loser prints "another KISS
    // update is already running (pid N)" and exits 1.  KISS_NONINTERACTIVE
    // is what ./install.sh reads in place of --non-interactive.  A clone
    // that predates the lock (no scripts/install.sh) gets the unlocked
    // preflight below.
    const bootstrap = path.join(
      path.dirname(scriptPath),
      'scripts',
      'install.sh',
    );
    if (fs.existsSync(bootstrap)) {
      const escBootstrap = bootstrap.replace(/'/g, "'\\''");
      this._openUpdateTerminal(
        path.dirname(scriptPath),
        `cd '${escDir}'; KISS_HOME='${escKissHome}' KISS_NONINTERACTIVE=1 ` +
          `bash '${escBootstrap}'`,
      );
      return;
    }
    // audit0902-coverage:end
    const preflight = [
      `cd '${escDir}'`,
      "echo '>>> Pre-flight: synchronizing repo with origin before install.sh...'",
      'git fetch --force --tags --prune origin 2>/dev/null || true',
      '_kiss_stashed=; if [ -n "$(git status --porcelain 2>/dev/null)" ]; then git stash push --include-untracked -m \'kiss-update-preflight\' >/dev/null 2>&1 && _kiss_stashed=1 || _kiss_stashed=; fi',
      "git reset --hard '@{upstream}' 2>/dev/null || git reset --hard origin/HEAD 2>/dev/null || true",
      'if [ -n "$_kiss_stashed" ]; then git stash pop >/dev/null 2>&1 || true; fi',
      // --non-interactive: the Update button is automation.  install.sh
      // would otherwise ask its [Y/n] questions (e.g. the Homebrew
      // install) in this terminal and skip the setsid detachment that
      // protects the install from the terminal-disposal ^C during step
      // [5/5].
      `KISS_HOME='${escKissHome}' bash '${escScript}' --non-interactive`,
    ].join('; ');
    this._openUpdateTerminal(path.dirname(scriptPath), preflight);
  }

  /**
   * Submit *prompt* as a task in this webview's chat once the webview
   * is ready, however long its first load takes: posted now when it is
   * ready, otherwise held and posted once on its `ready`.  The webview
   * owns the tab id every run needs, so the prompt always goes through
   * its composer (and the daemon's one submit path) rather than being
   * sent as a tab-less run the daemon would drop.
   *
   * @param prompt The task text; blank text is ignored.
   */
  public submitWhenReady(prompt: string): void {
    const text = prompt.trim();
    if (!text) return;
    if (this._webviewReady) {
      this._sendToWebview({type: 'insertAndSubmit', text});
    } else {
      this._pendingSubmit = text;
    }
  }

  /**
   * Reveal the chat, focus its composer and submit *prompt* there once
   * the webview is ready (see submitWhenReady).
   *
   * @param prompt The task text; blank text is ignored.
   */
  public async submitTask(prompt: string): Promise<void> {
    const text = prompt.trim();
    if (!text) return;
    await this.focusChatInput();
    this.submitWhenReady(text);
  }

  /**
   * Reveal the chat and open its settings panel — the editor-title
   * gear button's action in editor-tabs mode (also usable in sidebar
   * mode). Waits briefly for a freshly created webview to report
   * `ready` so the message is not dropped by a still-loading page.
   */
  public async openSettingsUI(): Promise<void> {
    await this.focusChatInput();
    await this._waitForWebviewReady();
    this._sendToWebview({type: 'openSettings'});
  }

  /**
   * Wait (at most 3 s) for a freshly created webview to report `ready`,
   * so a message posted next is not dropped by a still-loading page.
   * Returns at once without a webview.
   */
  private async _waitForWebviewReady(): Promise<void> {
    for (let i = 0; i < 15 && this._view && !this._webviewReady; i++) {
      await new Promise(r => setTimeout(r, 200));
    }
  }

  /**
   * Reveal the chat and run its manual Git Commit — the editor-title
   * git-commit button's action in editor-tabs mode (also usable in
   * sidebar mode). Same flow as the settings drawer's Git Commit
   * button: the webview asks the daemon to commit the active chat
   * tab's working tree (autocommitAction). Waits briefly for a
   * freshly created webview to report `ready` so the message is not
   * dropped by a still-loading page.
   */
  public async gitCommit(): Promise<void> {
    await this.focusChatInput();
    await this._waitForWebviewReady();
    this._sendToWebview({type: 'gitCommit'});
  }

  /**
   * Ask the webview to bring one of its chat's tasks on screen —
   * scroll to the task's transcript region or replay it (editor-tabs
   * mode: a history-panel click on a chat whose panel already exists).
   *
   * @param taskId The task's history-row id.
   */
  public showTask(taskId: string): void {
    this._sendToWebview({type: 'showTask', taskId});
  }

  /**
   * Open a chat in the sidebar chat view — a primary-sidebar history
   * panel click while editor-tabs mode is OFF. Reveals the view,
   * waits briefly for a freshly created webview to report `ready`,
   * then relays the click; the webview mirrors its own in-page
   * history rows (switch to the chat's tab, resume it in a fresh tab,
   * or show the task text read-only when there is nothing to resume).
   *
   * @param event The clicked chat/task: backend chat id ('' or absent
   *     when the task has nothing to resume), the task's id and the
   *     task text for the read-only fallback.
   */
  public async openChatFromHistory(event: {
    chatId?: string;
    taskId?: string | number | null;
    title?: string;
  }): Promise<void> {
    await this.focusChatInput();
    await this._waitForWebviewReady();
    // The reveal above can outlive the routing decision that chose
    // this surface: editor-tabs mode flipped ON mid-wait hides this
    // view (its `when` clause), and posting now would resume the chat
    // invisibly. The user's next click routes to the panel manager.
    const modeNow = vscode.workspace
      .getConfiguration('kissSorcar')
      .get<boolean>('editorTabsMode', false);
    if (modeNow) return;
    this._sendToWebview({
      type: 'openChatFromHistory',
      chatId: event.chatId ? String(event.chatId) : '',
      taskId: event.taskId === undefined ? null : event.taskId,
      title: event.title || '',
    });
  }

  public stopTask(): void {
    if (this._view && this._webviewReady) {
      this._sendToWebview({type: 'triggerStop'});
      return;
    }
    // No resolved webview to relay through — stop running tasks directly.
    for (const tab of this._runningTabs) {
      this._send({type: 'stop', tabId: tab});
    }
  }

  /**
   * An ask_user_question reached a tab of this webview: bring the
   * surface forward (the sidebar view, or the editor panel even while a
   * text editor is active) without taking keyboard focus. The Question
   * panel in the webview is the whole notice: no native notification.
   */
  private _revealForQuestion(): void {
    if (this._panelHooks)
      this._panelHooks.onEvent({kind: 'reveal', force: true});
    else this._view?.show();
  }

  /**
   * A task just started in the chat tab this webview shows: give the
   * run the window. The sidebar chat view, when it is on screen (a
   * task launched from another surface while the bar is closed must
   * not cover what the user is working on), maximizes the secondary
   * side bar: VS Code hides the editor area, the primary side bar and
   * the panel, and restores them when the user opens an editor or
   * presses the bar's restore button; a bar already maximized is left
   * alone. An editor-tab chat cannot lose the editor area it lives
   * in, so it brings the Task Info view up beside itself instead (the
   * panel manager, keyboard focus kept in the chat).
   */
  private _showChatForRun(): void {
    if (this._panelHooks) this._panelHooks.onEvent({kind: 'taskStarted'});
    else if (this._view?.visible)
      void vscode.commands
        .executeCommand('workbench.action.maximizeAuxiliaryBar')
        .then(undefined, () => undefined); // VS Code < 1.102: no such command
  }

  public async focusChatInput(): Promise<void> {
    if (!this._view && !this._panelHooks) {
      await vscode.commands.executeCommand(
        'kissSorcar.chatViewSecondary.focus',
      );
      for (let i = 0; i < 10 && !this._view; i++) {
        await new Promise(r => setTimeout(r, 200));
      }
    }
    if (this._view) {
      this._view.show();
      await new Promise(r => setTimeout(r, 150));
      this._sendToWebview({type: 'focusInput'});
    }
  }

  public async appendToInput(text: string): Promise<void> {
    // Resolve/show the view first so the command also works before the
    // sidebar has ever been opened (parity with submitTask).
    await this.focusChatInput();
    if (this._view) {
      this._sendToWebview({type: 'appendToInput', text});
    }
  }

  public newConversation(): void {
    this._sendToWebview({type: 'clearChat'});
  }

  /**
   * Task Info view only (meta-panel-mode): render the ACTIVE chat
   * editor panel's task-info values. The message is also remembered so
   * a webview that resolves after the relay catches up on `ready`.
   *
   * @param values The panel's #meta-list display strings, or null to
   *     show the placeholder dashes (no chat panel is reporting).
   * @param taskUpdate The running task's task-update report state,
   *     null to hide the info subpanel.
   */
  public postMetaState(
    values: MetaPanelValues | null,
    taskUpdate: TaskUpdateState | null,
  ): void {
    this._lastMetaState = {type: 'metaState', values, taskUpdate};
    this._sendToWebview(this._lastMetaState);
  }

  /**
   * Task Info view only (meta-panel-mode): a task just started in the
   * ACTIVE chat editor panel, and the view is being brought up for it
   * (SorcarPanelManager `taskStarted`): expand its Task Info and Task
   * update sections and collapse the rest. A webview still loading
   * (the view resolves after the reveal) gets the message on `ready`.
   */
  public postShowForRun(): void {
    if (this._webviewReady) this._sendToWebview({type: 'showForRun'});
    else this._showForRunPending = true;
  }

  /**
   * History panel (history-panel-mode): relay the chat / task ids of
   * the chat surface on screen, so the panel highlights that task's
   * row and scrolls it into view. Remembered so a webview that
   * resolves after the relay catches up on `ready`.
   *
   * @param chatId The chat's id, '' when no surface reports one.
   * @param taskId The task's id, '' when unknown.
   */
  public postActiveTask(chatId: string, taskId: string): void {
    this._lastActiveTask = {type: 'activeTask', chatId, taskId};
    this._sendToWebview(this._lastActiveTask);
  }

  private _measureSidebar(
    timeoutMs: number = 1500,
  ): Promise<{inner: number; screen: number} | undefined> {
    if (!this._view) return Promise.resolve(undefined);
    this._sizeReportResolver = undefined;
    return new Promise(resolve => {
      let done = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const finish = (v: {inner: number; screen: number} | undefined) => {
        if (done) return;
        done = true;
        // The safety-net timer must not outlive a prompt report: the
        // widening loop measures up to 31 times, and each leaked timer
        // kept the extension host alive for another 1.5 s.
        clearTimeout(timer);
        if (this._sizeReportResolver === inner) {
          this._sizeReportResolver = undefined;
        }
        resolve(v);
      };
      const inner = (s: {inner: number; screen: number}) => finish(s);
      this._sizeReportResolver = inner;
      this._sendToWebview({type: 'measureSize'});
      if (!done) timer = setTimeout(() => finish(undefined), timeoutMs);
    });
  }

  public async widenToOneThird(
    maxIterations: number = 30,
    tolerance: number = 0.06,
  ): Promise<void> {
    if (!this._view) return;
    const initial = await this._measureSidebar();
    if (!initial || initial.screen <= 0) return;
    const target = initial.screen / 3;
    let prev = initial.inner;
    let stuck = 0;
    for (let i = 0; i < maxIterations; i++) {
      const m = await this._measureSidebar();
      if (!m) return;
      const cur = m.inner;
      if (Math.abs(cur - target) <= target * tolerance) return;
      const cmd =
        cur < target
          ? 'workbench.action.increaseViewSize'
          : 'workbench.action.decreaseViewSize';
      await vscode.commands.executeCommand(cmd);
      await new Promise(r => setTimeout(r, 60));
      if (Math.abs(cur - prev) < 1) {
        stuck += 1;
        if (stuck >= 2) return;
      } else {
        stuck = 0;
      }
      prev = cur;
    }
  }

  /**
   * Ask the daemon for a commit message and wait for its answer.
   *
   * @param token Cancellation token of the invoking command, if any.
   * @param tabId Tab the generation belongs to.  The daemon stamps it
   *     on the answer and claims one generation per tab, so two
   *     repositories generating at once must pass different ids.
   * @param workDir Repository to diff.  Defaults to the window's
   *     working directory, which is only right when the request did not
   *     come from a specific repository: a workspace can hold several,
   *     and diffing the wrong one answers the wrong question.
   * @returns A promise resolved when the answer arrives, the token is
   *     cancelled, or the wait times out.
   */
  public generateCommitMessage(
    token?: vscode.CancellationToken,
    tabId: string = '',
    workDir?: string,
  ): Promise<void> {
    if (this._commitPendingTabs.has(tabId)) return Promise.resolve();
    this._commitPendingTabs.add(tabId);
    // The answer comes back stamped with this tab, and only messages
    // for tabs this window owns are forwarded on.
    if (tabId) this._ownTabs.add(tabId);
    this._send({
      type: 'generateCommitMessage',
      model: this._selectedModel,
      tabId,
      workDir: workDir || this._getWorkDir(),
    });

    return new Promise<void>(resolve => {
      let resolved = false;
      // eslint-disable-next-line prefer-const
      let cancelSub: vscode.Disposable | undefined;
      const done = () => {
        if (resolved) return;
        resolved = true;
        this._commitPendingTabs.delete(tabId);
        disposable.dispose();
        cancelSub?.dispose();
        clearTimeout(timer);
        resolve();
      };
      const disposable = this._onCommitMessage.event(ev => {
        if ((ev.tabId ?? '') === tabId) done();
      });
      cancelSub = token?.onCancellationRequested(() => done());
      const timer = setTimeout(done, 30_000);
    });
  }

  /**
   * Open an already-resolved, existing file the way a clicked file link
   * opens it: .html/.htm rendered in a webview tab, .md/.markdown
   * converted to HTML and rendered in a markdown-preview tab, .pdf drawn
   * in a pdf.js webview tab, text-like files in the text editor
   * (optionally revealing 1-indexed *line*), and everything else
   * (images, archives, ...) in VS Code's native viewer.
   * Both the 'openFile' message (a clicked link) and the path-only
   * 'submit' shortcut route through here so the two behave identically.
   */
  private async _openResolvedFile(
    filePath: string,
    line?: number,
  ): Promise<void> {
    let isDirectory = false;
    try {
      isDirectory = fs.statSync(filePath).isDirectory();
    } catch {
      // Deleted between the existence check and the click: fall through
      // to the file paths below, which surface their own errors.
    }
    if (isDirectory) {
      // A directory link cannot open in an editor; reveal it in the
      // Explorer instead so the user can browse its contents.
      await vscode.commands.executeCommand(
        'revealInExplorer',
        vscode.Uri.file(filePath),
      );
      return;
    }
    if (isRenderableHtmlExtension(filePath)) {
      // Render the page in a webview tab — like the remote web app
      // does — instead of showing its source in the editor.
      this._openHtmlPreviewTab(filePath);
      return;
    }
    if (path.extname(filePath).toLowerCase() === '.pdf') {
      // VS Code has no PDF editor (vscode.open shows a "binary file"
      // notice): draw the pages in a webview tab, like the remote web
      // app's content tab does.
      this._openPdfPreviewTab(filePath);
      return;
    }
    if (isRenderableMarkdownExtension(filePath)) {
      // Render the markdown converted to HTML in a tab — VS Code's
      // built-in markdown preview — like the remote web app does,
      // instead of showing the raw markdown source in the editor.
      // If the built-in markdown extension is unavailable the command
      // fails; fall through to the text editor then.
      try {
        await vscode.commands.executeCommand(
          'markdown.showPreview',
          vscode.Uri.file(filePath),
        );
        return;
      } catch (err) {
        console.warn(
          '[SorcarSidebarView] markdown preview unavailable, ' +
            'opening source instead:',
          err,
        );
      }
    }
    const uri = vscode.Uri.file(filePath);
    if (!isTextLikeExtension(filePath)) {
      await vscode.commands.executeCommand('vscode.open', uri);
      return;
    }
    const doc = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(doc, {
      preview: false,
      viewColumn: vscode.ViewColumn.One,
    });
    if (line !== undefined && line > 0) {
      const pos = new vscode.Position(line - 1, 0);
      editor.selection = new vscode.Selection(pos, pos);
      editor.revealRange(
        new vscode.Range(pos, pos),
        vscode.TextEditorRevealType.InCenter,
      );
    }
  }

  /**
   * Open *filePath* (a resolved, existing .html/.htm file) rendered in a
   * webview editor tab, the way the remote web app renders HTML files in
   * an in-app tab. One tab is kept per path: clicking the same link
   * again reveals the existing tab and re-reads the file so edits made
   * since the last click show up. Relative asset references work via an
   * injected <base> pointing at the file's directory (see
   * injectHtmlBase), with the file's directory and the workspace
   * folders allowed as local resource roots.
   */
  private _openHtmlPreviewTab(filePath: string): void {
    let html: string;
    try {
      html = fs.readFileSync(filePath, 'utf8');
    } catch (err) {
      showErrorNotification(`Failed to read ${filePath}: ${String(err)}`);
      return;
    }
    const dir = path.dirname(filePath);
    let panel = this._htmlPreviewPanels.get(filePath);
    if (!panel) {
      panel = vscode.window.createWebviewPanel(
        'kissSorcarHtmlPreview',
        path.basename(filePath),
        vscode.ViewColumn.One,
        {
          enableScripts: true,
          localResourceRoots: [
            vscode.Uri.file(dir),
            ...(vscode.workspace.workspaceFolders ?? []).map(f => f.uri),
          ],
        },
      );
      this._htmlPreviewPanels.set(filePath, panel);
      panel.onDidDispose(() => {
        this._htmlPreviewPanels.delete(filePath);
      });
    } else {
      panel.reveal(vscode.ViewColumn.One);
    }
    const dirUri = panel.webview.asWebviewUri(vscode.Uri.file(dir));
    panel.webview.html = injectHtmlBase(html, dirUri.toString());
  }

  /**
   * Open *filePath* (a resolved, existing .pdf file) in a webview editor
   * tab that draws its pages with pdf.js (see buildPdfPreviewHtml).  One
   * tab is kept per path: a second click reveals the existing tab and
   * reloads the file.  The page fetches the PDF through the webview
   * resource scheme, so the file's directory is a local resource root
   * next to the extension's media directory (pdfView.js, main.css).
   */
  private _openPdfPreviewTab(filePath: string): void {
    let panel = this._htmlPreviewPanels.get(filePath);
    if (!panel) {
      panel = vscode.window.createWebviewPanel(
        'kissSorcarPdfPreview',
        path.basename(filePath),
        vscode.ViewColumn.One,
        {
          enableScripts: true,
          retainContextWhenHidden: true,
          localResourceRoots: [
            vscode.Uri.file(path.dirname(filePath)),
            vscode.Uri.joinPath(this._extensionUri, 'media'),
          ],
        },
      );
      this._htmlPreviewPanels.set(filePath, panel);
      panel.onDidDispose(() => {
        this._htmlPreviewPanels.delete(filePath);
      });
      panel.webview.onDidReceiveMessage((msg: {type?: string}) => {
        if (msg && msg.type === PDF_DOWNLOAD_MESSAGE) {
          void this._savePdfCopy(filePath);
        }
      });
    } else {
      panel.reveal(vscode.ViewColumn.One);
    }
    panel.webview.html = buildPdfPreviewHtml(
      panel.webview,
      this._extensionUri,
      filePath,
    );
  }

  /**
   * The PDF preview's Download link: a webview cannot save a file, so
   * ask where to put a copy of *filePath* (the home directory and the
   * file's own name by default) and copy it there.
   */
  private async _savePdfCopy(filePath: string): Promise<void> {
    const target = await vscode.window.showSaveDialog({
      title: 'Download PDF',
      defaultUri: vscode.Uri.file(
        path.join(os.homedir(), path.basename(filePath)),
      ),
      filters: {PDF: ['pdf']},
    });
    if (!target) return;
    try {
      await vscode.workspace.fs.copy(vscode.Uri.file(filePath), target, {
        overwrite: true,
      });
    } catch (err) {
      showErrorNotification(`Failed to save ${target.fsPath}: ${String(err)}`);
    }
  }

  public dispose(): void {
    // Terminal: set first so any concurrently queued webview message or
    // late _getClient()/_send() call becomes a no-op and cannot
    // resurrect the daemon client or its listeners.
    this._terminated = true;
    this._disposed = true;
    for (const sub of this._viewSubs) sub.dispose();
    this._viewSubs = [];
    this._view = undefined;
    if (this._installedPoster) {
      clearWebviewNotificationPoster(this._installedPoster);
      this._installedPoster = undefined;
    }
    this._voiceWakeSuspendedByHide = false;
    // audit0903-coverage:start
    this._voiceEnabled = false;
    this._voiceWake?.dispose();
    // audit0903-coverage:end
    this._voiceWake = undefined;
    if (this._configFileWatchTimer) {
      clearInterval(this._configFileWatchTimer);
      this._configFileWatchTimer = undefined;
    }
    this._resolveAllWorktreeActions();
    if (this._client) {
      this._client.dispose();
      this._client = null;
    }
    // A commit-message request still in flight gets no answer from a
    // disposed client: settle its wait now (each `fire` resolves the
    // waiter and clears its tab) instead of after its 30 s safety timer.
    for (const tabId of [...this._commitPendingTabs]) {
      this._onCommitMessage.fire({message: '', tabId});
    }
    this._onCommitMessage.dispose();
    this._onRegistryTabsState.dispose();
    // Each panel's onDidDispose deletes its own map entry, so iterate a
    // snapshot and clear at the end.
    for (const panel of [...this._htmlPreviewPanels.values()]) {
      panel.dispose();
    }
    this._htmlPreviewPanels.clear();
  }
}
