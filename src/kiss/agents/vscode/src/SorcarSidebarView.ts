// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

function isPathInside(target: string, root: string): boolean {
  const rt = path.resolve(root);
  const tg = path.resolve(target);
  if (tg === rt) return true;
  const rel = path.relative(rt, tg);
  return rel.length > 0 && !rel.startsWith('..') && !path.isAbsolute(rel);
}

/**
 * Resolve *p* against *root* and return the resolved path only when it is
 * a real file or directory inside *root* — comparing REAL paths, so a
 * symlink inside the workspace cannot smuggle in a path that actually
 * lives outside it. Directories resolve too: clicking a directory link
 * reveals it in the Explorer (see _openResolvedFile). Pass
 * fileOnly=true to reject directories, so a caller that wants a file
 * can fall through to its next candidate (the pending worktree).
 */
function resolveWorkspaceFile(
  p: string,
  root: string,
  fileOnly = false,
): string | null {
  try {
    const resolved = path.resolve(root, p);
    if (!isPathInside(resolved, root)) return null;
    const real = fs.realpathSync(resolved);
    const realRoot = fs.realpathSync(root);
    if (!isPathInside(real, realRoot)) return null;
    const st = fs.statSync(real);
    if (!st.isFile() && (fileOnly || !st.isDirectory())) return null;
    return resolved;
  } catch {
    return null;
  }
}

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
import {SorcarApi} from './SorcarApi';
import {getGitApi} from './gitApi';
import {
  provisionalDefaultModel,
  resolveDefaultModel,
} from './DependencyInstaller';
import {buildChatHtml, readSampleTasks} from './SorcarTab';
import {VoiceWakeService} from './voiceWake';
import {kissHomeDir} from './userAssets';
import {playVoiceAckClip} from './voiceAckPlayer';
import {bootstrapInstallUrl, findInstallScript} from './installerPath';
import {
  FromWebviewMessage,
  ToWebviewMessage,
  Attachment,
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
  | {kind: 'reveal'}
  // Open another chat as a new editor tab (fresh when chatId is '').
  | {
      kind: 'openChat';
      chatId?: string;
      taskId?: string | number | null;
      title?: string;
      // Fresh chats: composer draft to seed the new panel's textarea.
      pendingText?: string;
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
  scopeWorkDir: string;
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
  newChat: ['tabId'],
  openTab: ['tabId', 'title', 'workDir'],
  getHistory: ['query', 'offset', 'generation'],
  getFrequentTasks: ['limit'],
  setFavorite: ['taskId', 'isFavorite'],
  deleteFrequentTask: ['task'],
  // tabId must survive: the daemon echoes it on the `files` reply so the
  // webview can tell whether the @-mention picker still belongs to the
  // conversation on screen.
  getFiles: ['prefix', 'workDir', 'tabId'],
  getAdjacentTask: ['tabId', 'taskId', 'direction'],
  getConfig: [],
  saveConfig: ['config', 'apiKeys'],
  // The settings panel's Custom Models subpanel: the daemon owns
  // ~/.kiss/MY_MODELS.json, so the CRUD travels to it whole and the
  // `myModelsData` replies come back through the client relay.
  getMyModels: [],
  saveMyModel: ['name', 'endpoint', 'apiKey', 'headers', 'originalName'],
  deleteMyModel: ['name'],
  // The Inject promptlet panel's Add button: the daemon owns
  // ~/.kiss/MY_INJECTION.md and answers with an unstamped `tricksData`
  // list that every window's panel repaints from.
  addTrick: ['text'],
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
  private _api: SorcarApi | null = null;
  private _daemonConnected: boolean = false;
  private _activeTabId: string = '';
  // Task Info view only (meta-panel-mode): the last relayed metaState,
  // kept so a webview that resolves (or reloads) after the relay can
  // be brought up to date on its `ready`.
  private _lastMetaState?: Extract<ToWebviewMessage, {type: 'metaState'}>;
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
  private _lastSentUrl: string = '';
  private _lastSeenRemotePassword: string | undefined;
  private _configFileWatchTimer?: ReturnType<typeof setInterval>;
  private _onFirstResolve: (() => void) | undefined;
  private _sizeReportResolver:
    ((s: {inner: number; screen: number}) => void) | undefined;
  private _workspaceFoldersSub: vscode.Disposable | undefined;
  // One rendered-HTML tab per file path, mirroring the remote web app's
  // content tabs: a second click on the same link reveals (and
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

  private _getApi(): SorcarApi {
    if (this._api) return this._api;
    const api = new SorcarApi(this._getClient());
    // After terminal dispose() the wrapper must not be re-cached: it wraps
    // an inert client and caching it would partially resurrect the view.
    if (!this._terminated) this._api = api;
    return api;
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
    client.on('connect', () => {
      this._getApi().setWorkDir(this._getWorkDir());
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
        this._getApi().getModels();
        this._getApi().getInputHistory();
        this._getApi().getConfig();
        // The settings panel's Custom Models list would otherwise stay
        // empty/stale in a panel left open across a daemon outage.
        this._getApi().forward({type: 'getMyModels'});
      }
    });
    client.on('disconnect', () => {
      this._daemonConnected = false;
      this._sendToWebview({type: 'daemonStatus', connected: false});
      this._resolveAllWorktreeActions();
    });
    client.on(
      'commandDropped',
      (cmd: AgentCommand, reason: DroppedCommandReason) => {
        this._handleDroppedCommand(cmd, reason);
      },
    );
    client.connect();
    this._workspaceFoldersSub = vscode.workspace.onDidChangeWorkspaceFolders(
      () => {
        const wd = this._getWorkDir();
        this._getApi().setWorkDir(wd);
        // The webview scopes its tab bar and history to the workspace
        // directory; tell it directly, because the daemon answers
        // `setWorkDir` with no `configData` the webview could learn
        // the change from.
        this._sendToWebview({type: 'workspaceWorkDir', workDir: wd});
      },
    );
    return client;
  }

  /**
   * Undo the optimistic UI of a command the daemon never received.
   *
   * A run is shown as started the instant the user sends it, long
   * before any daemon has confirmed it: the tab spins and the composer
   * locks.  Only a `status running:false` ever undoes that, and only
   * the daemon sends one -- so a command the client gives up on leaves
   * the tab running for ever, with no agent behind it and nothing the
   * user can do but reload the window.
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
    if (dropped.type === 'run') {
      if (tabId !== undefined) this._runningTabs.delete(tabId);
      this._sendToWebview({type: 'status', running: false, tabId});
      const why =
        reason === 'expired'
          ? 'the agent was unreachable for too long'
          : 'too many requests were waiting';
      showWarningNotification(
        `Your request was not started because ${why}. Send it again.`,
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
        // Show this window's workspace folder in the settings panel.
        // When the window has none (and the host cwd is a filesystem
        // root, so _getWorkDir reports nothing), keep the daemon's own
        // work_dir: that is where the window's tasks will actually run.
        const wd = this._getWorkDir();
        if (wd) msg.config.work_dir = wd;
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
      if (msg.type === 'worktree_created' || msg.type === 'worktree_done') {
        const dir = msg.worktreeDir;
        const wtTabId = msg.tabId;
        if (dir && wtTabId !== undefined) {
          // Not gated on _isOwnTab: a canonical tab created by another
          // client is adopted by this webview from `tabs_state` without
          // ever sending a message that would register it in _ownTabs,
          // yet its transcript (mirrored here) still needs the
          // pending-worktree fallback for _resolveTabFile(). Recording
          // a directory is side-effect free; only the SCM view below
          // stays scoped to tabs this window interacted with.
          // worktreeWorkDir (the task's cwd inside the worktree) wins
          // over the worktree root so relative paths from tasks
          // launched in a repo subdirectory resolve correctly.
          this._worktreeDirs.set(wtTabId, msg.worktreeWorkDir || dir);
        }
        if (dir && this._isOwnTab(wtTabId)) {
          void this._openWorktreeInScm(dir);
        }
      }
      if (msg.type === 'task_events' && Array.isArray(msg.events)) {
        // A session replay (reconnect, adopted canonical tab) reaches a
        // host that may have no _worktreeDirs entry for the tab: the
        // daemon dropped its own tracking in cleanup_tab() before the
        // replay, and while the task is still running nothing re-emits
        // worktree_done. The historical worktree events nested in the
        // replayed transcript are the only copy of the directory, so
        // scan them in order (a later successful worktree_result nets
        // out an earlier worktree_created).
        this._trackReplayedWorktreeEvents(msg.tabId, msg.events);
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
              scopeWorkDir: t.scopeWorkDir || '',
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
        // Mirrors the unconditional recording above: a merge/discard
        // finished by any client retires the worktree directory, so the
        // fallback entry must go even when this window never claimed
        // the tab. git.close on a repository that was never opened is
        // a harmless no-op. A "Do nothing" result (kept: true) leaves
        // the worktree on disk, so its fallback entry must survive for
        // transcript file links to keep resolving into it.
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
        if (msg.success) {
          if (!isSilentDiscardMessage(msg.message)) {
            showInformationNotification(
              msg.message || 'Worktree action completed.',
            );
          }
        } else {
          showErrorNotification(msg.message || 'Worktree action failed.');
        }
      }
      if (msg.type === 'main_tree_result' && this._isOwnTab(msg.tabId)) {
        // Same toast rule as worktree_result above: the post-task
        // main-tree bar's Discard / Do nothing outcome is surfaced as
        // a notification (the webview renders the transcript line).
        if (msg.success) {
          showInformationNotification(
            msg.message || 'Main-tree action completed.',
          );
        } else {
          showErrorNotification(msg.message || 'Main-tree action failed.');
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
        if (msg.success) {
          showInformationNotification(msg.message || 'Auto-commit completed.');
        } else {
          showErrorNotification(msg.message || 'Auto-commit failed.');
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
    this._lastSentUrl = '';

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
        this._getApi().getInputHistory();
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

  private _getWorkDir(): string {
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

  /**
   * Restore the pending-worktree fallback from a replayed transcript.
   *
   * Applies the same net effect as receiving the nested worktree
   * events live: `worktree_created` / `worktree_done` record the
   * directory (preferring the task's cwd inside the worktree), a
   * successful `worktree_result` retires it.
   */
  private _trackReplayedWorktreeEvents(
    tabId: string | undefined,
    events: unknown[],
  ): void {
    if (tabId === undefined) return;
    for (const raw of events) {
      if (!raw || typeof raw !== 'object') continue;
      const ev = raw as {
        type?: string;
        worktreeDir?: string;
        worktreeWorkDir?: string;
        success?: boolean;
      };
      if (ev.type === 'worktree_created' || ev.type === 'worktree_done') {
        const dir = ev.worktreeWorkDir || ev.worktreeDir;
        if (dir) this._worktreeDirs.set(tabId, dir);
      } else if (ev.type === 'worktree_result' && ev.success) {
        this._worktreeDirs.delete(tabId);
      }
    }
  }

  /**
   * Resolve *p* for a tab: against *wd* first, then against the tab's
   * pending worktree directory.
   *
   * A worktree task's committed artifacts live only on its un-merged
   * `kiss/wt-*` branch until the user merges (or the next run
   * auto-retires it), so a path printed in its result panel does not
   * exist under the workspace root yet and a plain
   * `resolveWorkspaceFile(p, wd)` reports it missing — leaving the
   * link permanently grey.  Falling back to the worktree dir recorded
   * for the tab (`worktree_created` / `worktree_done`) makes the path
   * resolvable the moment the result renders; after a merge or discard
   * the `worktree_result` handler drops the entry and the workspace
   * copy (or genuine absence) wins again.
   */
  private _resolveTabFile(
    p: string,
    wd: string,
    tabId: string | undefined,
    fileOnly = false,
  ): string | null {
    const resolved = resolveWorkspaceFile(p, wd, fileOnly);
    if (resolved) return resolved;
    const wtDir =
      tabId !== undefined ? this._worktreeDirs.get(tabId) : undefined;
    if (wtDir && wtDir !== wd) return resolveWorkspaceFile(p, wtDir, fileOnly);
    return null;
  }

  private _sendToWebview(message: ToWebviewMessage): void {
    if (!this._disposed && this._view) {
      this._view.webview.postMessage(message);
    }
  }

  private _sendWelcomeSuggestions(): void {
    this._sendToWebview({
      type: 'welcome_suggestions',
      suggestions: readSampleTasks(this._extensionUri.fsPath),
    } as ToWebviewMessage);
  }

  private _sendRemoteUrl(): void {
    const urlFile = path.join(kissHomeDir(), 'remote-url.json');
    this._tryReadAndSendUrl(urlFile);
    this._watchUrlFile(urlFile);
  }

  private _tryReadAndSendUrl(urlFile: string): void {
    let tunnel = '';
    let local = '';
    let loopback = '';
    let lanUrls: string[] = [];
    try {
      const data = JSON.parse(fs.readFileSync(urlFile, 'utf-8'));
      tunnel = data.tunnel || '';
      local = data.local || '';
      loopback = data.loopback || '';
      if (Array.isArray(data.lan)) {
        lanUrls = data.lan.filter((u: unknown) => typeof u === 'string');
      }
    } catch {}
    const tunnelActive = !!tunnel;
    const url = tunnel || local || '';
    const ntfyUrl = this._getNtfyUrl();
    const key =
      `${tunnelActive ? '1' : '0'}|${url}|${ntfyUrl}|` +
      `${loopback}|${lanUrls.join(',')}`;
    if (key === this._lastSentUrl) return;
    this._lastSentUrl = key;
    const msg: ToWebviewMessage = {type: 'remote_url', url, tunnelActive};
    if (ntfyUrl) {
      msg.ntfyUrl = ntfyUrl;
    }
    if (loopback) {
      msg.loopbackUrl = loopback;
    }
    if (lanUrls.length > 0) {
      msg.lanUrls = lanUrls;
    }
    this._sendToWebview(msg);
  }

  private _getNtfyUrl(): string {
    try {
      const topicFile = path.join(kissHomeDir(), 'ntfy_topic');
      const topic = fs.readFileSync(topicFile, 'utf-8').trim();
      if (topic) {
        return `https://ntfy.sh/${topic}`;
      }
    } catch {}
    return '';
  }

  private _urlFileWatchTimer?: ReturnType<typeof setInterval>;

  private _watchUrlFile(urlFile: string): void {
    if (this._urlFileWatchTimer) return;
    this._urlFileWatchTimer = setInterval(() => {
      this._tryReadAndSendUrl(urlFile);
    }, 10_000);
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
      this._getApi().getConfig();
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

  private _startTask(
    prompt: string,
    model: string,
    activeFile?: string,
    attachments?: Attachment[],
    useWorktree?: boolean,
    useParallel?: boolean,
    tabId?: string,
    workDir?: string,
    autoCommit?: boolean,
    webTools?: boolean,
  ): void {
    const effectiveWorkDir = workDir || this._getWorkDir();
    // No local setTaskText echo: the daemon's common run path
    // broadcasts it to EVERY client (this webview included), so the
    // task-panel text mirrors identically for all run origins.
    this._sendToWebview({type: 'status', running: true, tabId});
    this._getApi().run({
      prompt,
      model,
      workDir: effectiveWorkDir,
      activeFile,
      attachments,
      useWorktree,
      useParallel,
      autoCommit,
      webTools,
      tabId,
    });
  }

  private _isOwnTab(tabId: string | undefined): boolean {
    return !tabId || this._ownTabs.has(tabId);
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
      this._getApi().forward(cmd as unknown as AgentCommand);
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
        this._sendWelcomeSuggestions();
        this._sendRemoteUrl();
        this._watchConfigFile();
        // The Task Info view (meta-panel-mode): a metaState relayed
        // before the webview loaded — or lost to a webview reload —
        // must not leave the panel on its placeholder dashes.
        if (this._lastMetaState) this._sendToWebview(this._lastMetaState);
        if (this._lastActiveTask) this._sendToWebview(this._lastActiveTask);
        // The daemon owns the canonical tab registry, so `ready` is
        // forwarded whole: the daemon fans out the connId-scoped init
        // replies (models / input history / config), merges any legacy
        // restoredTabs into an empty registry, answers with the
        // canonical `tabs_state` snapshot, and replays every
        // chat-bound tab's transcript.
        this._getApi().forward({
          type: 'ready',
          tabId: message.tabId,
          restoredTabs: message.restoredTabs,
        } as AgentCommand);
        break;
      }

      case 'submit': {
        const tabId = message.tabId;
        if (tabId) this._activeTabId = tabId;
        if (tabId !== undefined && this._runningTabs.has(tabId)) {
          const followUp = message.prompt.trim();
          if (followUp) {
            this._getApi().appendUserMessage(message.prompt, tabId);
          }
          return;
        }

        const tabWorkDir = message.workDir;
        const effectiveWorkDir = tabWorkDir || this._getWorkDir();

        const trimmed = message.prompt.trim();
        if (trimmed && !trimmed.includes('\n')) {
          // _resolveTabFile (not plain resolveWorkspaceFile): a report
          // that lives only in the tab's pending worktree must open
          // like any other file link — falling through to _startTask
          // would launch an unintended agent run on a path-only prompt.
          // Regular files ONLY (fileOnly): _resolveTabFile also
          // resolves directories (for clickable directory links), but
          // a one-word prompt that happens to name a directory ("src",
          // "tmp", ...) must still start a task, not reveal the
          // directory in the Explorer. fileOnly also keeps a workspace
          // DIRECTORY from shadowing a pending-worktree FILE at the
          // same relative path: the directory candidate is skipped and
          // the worktree file still opens.
          const resolved = this._resolveTabFile(
            trimmed,
            effectiveWorkDir,
            tabId,
            true,
          );
          if (resolved) {
            await this._openResolvedFile(resolved);
            return;
          }
        }

        if (tabId !== undefined) this._runningTabs.add(tabId);
        this._startTask(
          message.prompt,
          message.model,
          this._getVisibleEditorFile() || undefined,
          message.attachments,
          message.useWorktree,
          message.useParallel,
          tabId,
          effectiveWorkDir,
          message.autoCommit,
          message.webTools,
        );
        break;
      }

      case 'stop': {
        const stopTabId = message.tabId;
        if (stopTabId !== undefined) {
          this._getApi().stop(stopTabId);
        } else {
          for (const tab of this._runningTabs) {
            this._getApi().stop(tab);
          }
        }
        break;
      }

      case 'selectModel': {
        this._selectedModel = message.model;
        const selTabId = message.tabId;
        this._getApi().selectModel(message.model, selTabId);
        break;
      }

      case 'userAnswer': {
        const ansTabId = message.tabId;
        if (ansTabId !== undefined) {
          this._getApi().userAnswer(message.answer, ansTabId);
        }
        break;
      }

      case 'recordFileUsage':
        if (message.path) {
          this._getApi().recordFileUsage(message.path, message.workDir);
        }
        break;

      case 'openFile':
        if (message.path) {
          const wd = message.workDir || this._getWorkDir();
          const filePath = this._resolveTabFile(
            message.path,
            wd,
            message.tabId,
          );
          if (!filePath) {
            console.warn(
              '[SorcarSidebarView] refusing to open file outside workspace:',
              message.path,
            );
            break;
          }
          await this._openResolvedFile(filePath, message.line);
        }
        break;

      case 'checkPaths': {
        // The chat webview linkifies file-path-looking strings in event
        // panel contents lazily: a path only becomes a clickable link
        // after this existence check confirms that clicking it would
        // actually open a file (same resolution rules as 'openFile').
        const wd = message.workDir || this._getWorkDir();
        const results: Record<string, boolean> = {};
        const paths = Array.isArray(message.paths) ? message.paths : [];
        for (const p of paths) {
          if (typeof p !== 'string' || !p) continue;
          results[p] = this._resolveTabFile(p, wd, message.tabId) !== null;
        }
        this._sendToWebview({
          type: 'pathsExist',
          results,
          workDir: message.workDir,
          tabId: message.tabId,
        });
        break;
      }

      case 'resumeSession': {
        const resumeTabId = message.tabId;
        this._getApi().resumeSession({
          chatId: message.chatId ?? message.id,
          taskId: message.taskId,
          tabId: resumeTabId,
        });
        break;
      }

      case 'getWelcomeSuggestions':
        this._sendWelcomeSuggestions();
        this._sendRemoteUrl();
        break;

      case 'complete': {
        const editorFile = this._getVisibleEditorFile();
        const completeDoc = editorFile
          ? vscode.workspace.textDocuments.find(
              d => d.uri.fsPath === editorFile,
            )
          : undefined;
        this._getApi().complete({
          query: message.query,
          tabId: message.tabId || this._activeTabId || undefined,
          activeFile: editorFile || undefined,
          activeFileContent: completeDoc?.getText(),
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
        this._getApi().worktreeAction(wtAction, wtTabId);
        break;
      }

      // The post-task bar of a non-worktree manual-commit run: the
      // daemon discards the main tree's uncommitted changes (or, for
      // "nothing", just acknowledges) and reports the outcome through
      // a broadcast main_tree_result event that the webview renders on
      // its own. (The bar's Auto commit button sends autocommitAction.)
      case 'mainTreeAction':
        this._getApi().mainTreeAction(
          message.action,
          message.tabId,
          message.workDir || this._getWorkDir(),
        );
        break;

      // The settings panel's "Git Commit" button: the daemon commits
      // the tab's working tree and reports progress and the outcome
      // through broadcast autocommit_progress/autocommit_done events
      // that the webview renders on its own.
      case 'autocommitAction':
        this._getApi().autocommitAction(
          message.tabId,
          message.workDir || this._getWorkDir(),
        );
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
        this._getApi().serverReset();
        break;

      case 'notificationAction':
        resolveWebviewNotificationAction(message.id, message.action);
        break;

      case 'closeTab': {
        const closeTabId = message.tabId;
        if (closeTabId) {
          this._cleanupTabResources(closeTabId);
          this._getApi().closeTab(closeTabId);
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

      case 'openChatPanel':
        this._panelHooks?.onEvent({
          kind: 'openChat',
          chatId: message.chatId,
          taskId: message.taskId,
          title: message.title,
          pendingText: message.pendingText,
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
        await this._openWorkDir(message.path);
        break;

      case 'pickWorkDir': {
        const wd = this._getWorkDir();
        const picked = await vscode.window.showOpenDialog({
          canSelectFolders: true,
          canSelectFiles: false,
          canSelectMany: false,
          openLabel: 'Open as Working Directory',
          defaultUri: wd ? vscode.Uri.file(wd) : undefined,
        });
        if (picked && picked[0]) await this._openWorkDir(picked[0].fsPath);
        break;
      }
    }
  }

  /**
   * Open *dir* as this window's folder.
   *
   * A VS Code window's working directory is its workspace folder, so a
   * directory chosen in the "Working directory" panel becomes a
   * `vscode.openFolder` in this window.  A path that is not a
   * directory, a file-system root, the folder this window already
   * shows, or an open VS Code refuses is reported back to the panel as
   * `workDirError` instead.
   */
  private async _openWorkDir(dir: string): Promise<void> {
    const target = String(dir || '').trim();
    const error = await this._openWorkDirError(target);
    if (error) this._sendToWebview({type: 'workDirError', text: error});
  }

  /** Open *target* in this window; the failure text, or '' on success. */
  private async _openWorkDirError(target: string): Promise<string> {
    const real = realDirectory(target);
    if (!real) return 'Not a directory: ' + (target || '(empty path)');
    if (path.dirname(real) === real) {
      // `/`, `C:\`, a UNC share root: the daemon never runs in one.
      return 'A file-system root cannot be the working directory; pick a folder.';
    }
    const current = this._getWorkDir();
    if (current && realDirectory(current) === real) {
      return target + ' is already the working directory of this window.';
    }
    try {
      await vscode.commands.executeCommand(
        'vscode.openFolder',
        vscode.Uri.file(target),
      );
    } catch (err) {
      return 'Could not open ' + target + ': ' + String(err);
    }
    return '';
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
    this._getApi().closeTab(tabId);
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
        'Updating KISS Sorcar runs install.sh under bash, which was not found. ' +
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
      'printf "\\n>>> KISS Sorcar update exited with status %s. ' +
      'Press Enter to close this terminal.\\n" "$_kiss_rc"; ' +
      'while IFS= read -r _kiss_enter; do ' +
      'if [ -z "$_kiss_enter" ]; then break; fi; ' +
      'done; ' +
      'fi; exit "$_kiss_rc"';
    const terminal = vscode.window.createTerminal({
      name: 'KISS Sorcar Update',
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
    this._getApi().forward({type: 'updateWhenIdle'});
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
      'An update of KISS Sorcar is getting installed…',
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

  public async submitTask(prompt: string): Promise<void> {
    const text = prompt.trim();
    if (!text) return;
    await this.focusChatInput();
    for (let i = 0; i < 15 && this._view && !this._webviewReady; i++) {
      await new Promise(r => setTimeout(r, 200));
    }
    if (this._view && this._webviewReady) {
      this._sendToWebview({type: 'insertAndSubmit', text});
      return;
    }
    this._startTask(
      text,
      this._selectedModel,
      this._getVisibleEditorFile() || undefined,
    );
  }

  /**
   * Reveal the chat and open its settings panel — the editor-title
   * gear button's action in editor-tabs mode (also usable in sidebar
   * mode). Waits briefly for a freshly created webview to report
   * `ready` so the message is not dropped by a still-loading page.
   */
  public async openSettingsUI(): Promise<void> {
    await this.focusChatInput();
    for (let i = 0; i < 15 && this._view && !this._webviewReady; i++) {
      await new Promise(r => setTimeout(r, 200));
    }
    this._sendToWebview({type: 'openSettings'});
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
    for (let i = 0; i < 15 && this._view && !this._webviewReady; i++) {
      await new Promise(r => setTimeout(r, 200));
    }
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
   *     when the task has nothing to resume), the task's id, and the
   *     task text for the read-only fallback.
   */
  public async openChatFromHistory(event: {
    chatId?: string;
    taskId?: string | number | null;
    title?: string;
  }): Promise<void> {
    await this.focusChatInput();
    for (let i = 0; i < 15 && this._view && !this._webviewReady; i++) {
      await new Promise(r => setTimeout(r, 200));
    }
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
      this._getApi().stop(tab);
    }
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
      const finish = (v: {inner: number; screen: number} | undefined) => {
        if (done) return;
        done = true;
        if (this._sizeReportResolver === inner) {
          this._sizeReportResolver = undefined;
        }
        resolve(v);
      };
      const inner = (s: {inner: number; screen: number}) => finish(s);
      this._sizeReportResolver = inner;
      this._sendToWebview({type: 'measureSize'});
      setTimeout(() => finish(undefined), timeoutMs);
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
    this._getApi().generateCommitMessage(
      this._selectedModel,
      tabId,
      workDir || this._getWorkDir(),
    );

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
   * converted to HTML and rendered in a markdown-preview tab, text-like
   * files in the text editor (optionally revealing 1-indexed *line*),
   * and everything else (images, pdf, ...) in VS Code's native viewer.
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

  public dispose(): void {
    // Terminal: set first so any concurrently queued webview message or
    // late _getClient()/_getApi() call becomes a no-op and cannot
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
    if (this._urlFileWatchTimer) {
      clearInterval(this._urlFileWatchTimer);
      this._urlFileWatchTimer = undefined;
    }
    if (this._configFileWatchTimer) {
      clearInterval(this._configFileWatchTimer);
      this._configFileWatchTimer = undefined;
    }
    this._resolveAllWorktreeActions();
    if (this._workspaceFoldersSub) {
      this._workspaceFoldersSub.dispose();
      this._workspaceFoldersSub = undefined;
    }
    if (this._client) {
      this._client.dispose();
      this._client = null;
    }
    // The API wrapper caches the client it was built around; keeping it
    // would hand out an object bound to the disposed client if the view
    // were ever used again, while a fresh _getClient() built a new one.
    this._api = null;
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
