// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Sidebar chat view for Sorcar.
 * Provides a WebviewViewProvider that renders the chat UI in the
 * VS Code secondary sidebar.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

/**
 * Return true when *target* (already resolved) is the same as *root* or
 * lives strictly inside *root* (after resolving symlinks).  Used to
 * defend against path-traversal in webview-supplied paths.
 *
 * H4 — guards every webview→extension file open call site.
 */
function isPathInside(target: string, root: string): boolean {
  const rt = path.resolve(root);
  const tg = path.resolve(target);
  if (tg === rt) return true;
  const rel = path.relative(rt, tg);
  return rel.length > 0 && !rel.startsWith('..') && !path.isAbsolute(rel);
}
import {AgentClient} from './AgentClient';
import {getGitApi} from './gitApi';
import {MergeManager} from './MergeManager';
import {getDefaultModel} from './DependencyInstaller';
import {buildChatHtml} from './SorcarTab';
import {findInstallScript, kissAiRoot} from './installerPath';
import {
  FromWebviewMessage,
  ToWebviewMessage,
  Attachment,
  AgentCommand,
} from './types';

/**
 * Webview messages forwarded verbatim to the daemon — message type →
 * the fields copied onto the outgoing ``AgentCommand``.  Messages that
 * need guards or extension-side side effects keep explicit ``case``
 * handlers in ``_handleMessage``.
 */
/**
 * Webview merge-action name → ``MergeManager`` method.  Also the single
 * source of truth for the ``kissSorcar.<method>`` merge keyboard
 * commands registered in ``extension.ts``.
 */
export const MERGE_ACTIONS = {
  accept: 'acceptChange',
  reject: 'rejectChange',
  prev: 'prevChange',
  next: 'nextChange',
  'accept-all': 'acceptAll',
  'reject-all': 'rejectAll',
  'accept-file': 'acceptFile',
  'reject-file': 'rejectFile',
} as const;

/** A MergeManager method name dispatchable via ``handleMergeCommand``. */
export type MergeCommand = (typeof MERGE_ACTIONS)[keyof typeof MERGE_ACTIONS];

const FORWARDED_COMMANDS: Record<string, readonly string[]> = {
  appendUserMessage: ['prompt', 'tabId'],
  getInputHistory: [],
  newChat: ['tabId'],
  getHistory: ['query', 'offset', 'generation'],
  getFrequentTasks: ['limit'],
  deleteTask: ['taskId'],
  setFavorite: ['taskId', 'isFavorite'],
  deleteFrequentTask: ['task'],
  getFiles: ['prefix', 'workDir'],
  getAdjacentTask: ['tabId', 'taskId', 'direction'],
  getConfig: [],
  saveConfig: ['config', 'apiKeys'],
};

/**
 * WebviewViewProvider for the KISS Sorcar chat in the secondary sidebar.
 *
 * Hosts the chat HTML/JS/CSS over a single AgentClient connection to
 * the kiss-web daemon (multiplexes every tab on one UDS socket).
 */
export class SorcarSidebarView implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  /**
   * Single persistent connection to the kiss-web daemon — multiplexes
   * every chat tab over one UDS socket.  Lazy-initialised on first use
   * via ``_getClient()``.  Reload survives running tasks: closing this
   * client only ends the socket; the daemon keeps every in-flight
   * ``_RunningAgentState`` alive for the deferred-close grace window so the next
   * activation can reconnect and re-subscribe.
   */
  private _client: AgentClient | null = null;
  /** The currently active tab ID (updated on every message with tabId). */
  private _activeTabId: string = '';
  private _extensionUri: vscode.Uri;
  private _selectedModel: string;
  private _runningTabs: Set<string> = new Set();
  private _webviewHasFocus: boolean = false;

  /** Per-tab MergeManager instances — each tab gets its own merge review. */
  private _mergeManagers: Map<string, MergeManager> = new Map();
  private _onCommitMessage = new vscode.EventEmitter<{
    message: string;
    error?: string;
  }>();
  public readonly onCommitMessage = this._onCommitMessage.event;
  private _commitPendingTabs: Set<string> = new Set();
  private _worktreeDirs: Map<string, string> = new Map();
  private _worktreeActionResolves: Map<string, () => void> = new Map();
  private _worktreeProgresses: Map<
    string,
    vscode.Progress<{message?: string}>
  > = new Map();
  private _autocommitActionResolves: Map<string, () => void> = new Map();
  private _autocommitProgresses: Map<
    string,
    vscode.Progress<{message?: string}>
  > = new Map();
  private _disposed: boolean = false;
  /** Last remote URL sent to the webview — avoids redundant messages. */
  private _lastSentUrl: string = '';
  /**
   * Last ``remote_password`` observed in ``~/.kiss/config.json`` by the
   * config-file watcher.  ``undefined`` until the first successful read
   * so the watcher can distinguish "never read" from "read as empty".
   */
  private _lastSeenRemotePassword: string | undefined;
  /** Poll timer for ``~/.kiss/config.json``; cleared on dispose(). */
  private _configFileWatchTimer?: ReturnType<typeof setInterval>;
  private _preMergeOpenFiles: Map<string, Set<string>> = new Map();
  private _restoreChain: Promise<void> = Promise.resolve();
  private _onFirstResolve: (() => void) | undefined;
  /** Pending resolver for the next ``sizeReport`` from the webview.
   *  Set by ``_measureSidebar`` and cleared by the ``sizeReport`` handler. */
  private _sizeReportResolver:
    | ((s: {inner: number; screen: number}) => void)
    | undefined;
  /** Subscription to workspace-folder changes; disposes on dispose(). */
  private _workspaceFoldersSub: vscode.Disposable | undefined;

  /**
   * Show a notification-progress dialog with a timeout-based auto-resolve.
   *
   * Stores the progress reporter and resolve callback in the given maps
   * so that incoming backend events can update the message or complete
   * the dialog.  If no completion event arrives within *timeoutMs* the
   * dialog is automatically dismissed.
   */
  private _showActionProgress(
    title: string,
    tabId: string | undefined,
    progressMap: Map<string, vscode.Progress<{message?: string}>>,
    resolveMap: Map<string, () => void>,
    timeoutMs: number = 120_000,
  ): void {
    vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title,
      },
      progress => {
        if (tabId !== undefined) {
          progressMap.set(tabId, progress);
        }
        return new Promise<void>(resolve => {
          if (tabId !== undefined) {
            resolveMap.set(tabId, resolve);
          }
          setTimeout(() => {
            if (tabId !== undefined && resolveMap.get(tabId) === resolve) {
              resolveMap.delete(tabId);
              resolve();
            }
          }, timeoutMs);
        });
      },
    );
  }

  /** Resolve all pending worktree/autocommit action promises and clear maps. */
  private _resolveAllWorktreeActions(): void {
    for (const resolve of this._worktreeActionResolves.values()) resolve();
    this._worktreeActionResolves.clear();
    this._worktreeProgresses.clear();
    for (const resolve of this._autocommitActionResolves.values()) resolve();
    this._autocommitActionResolves.clear();
    this._autocommitProgresses.clear();
  }

  /**
   * Register a one-time callback invoked when the webview view is first resolved.
   *
   * Used by the extension entry point to widen the secondary sidebar on
   * first activation.
   */
  public onFirstResolve(cb: () => void): void {
    this._onFirstResolve = cb;
  }

  /**
   * Eagerly push the current VS Code workspace folder to the daemon as
   * its ``work_dir``.
   *
   * The daemon caches ``self.work_dir`` once at process start from
   * ``KISS_WORKDIR``/``getcwd()``.  When VS Code launches and re-uses an
   * already-running daemon (the dependency-installer fast path), the
   * daemon retains the previous session's work_dir until the user
   * interacts with the sidebar (which lazily triggers ``_getClient()``).
   *
   * Calling this method from ``activate()`` ensures the daemon's
   * ``work_dir`` matches the open workspace folder *before* any backend
   * call (autocomplete file-list, commit-message generation, etc.) is
   * issued.  Safe to call repeatedly — the daemon ignores no-op
   * updates and the client queues the command if the socket is not yet
   * connected.
   */
  public syncWorkDir(): void {
    // _getClient() lazily creates the AgentClient and sends the
    // initial setWorkDir + installs the workspace-folders listener.
    this._getClient();
  }

  constructor(extensionUri: vscode.Uri) {
    this._extensionUri = extensionUri;
    this._selectedModel =
      vscode.workspace
        .getConfiguration('kissSorcar')
        .get<string>('defaultModel') || getDefaultModel();
  }

  /**
   * Get or create a MergeManager for the given tab.
   *
   * Each tab gets its own MergeManager so multiple tabs can show
   * their merge/diff UI concurrently without interfering.
   */
  private _getOrCreateMergeManager(tabId: string): MergeManager {
    const existing = this._mergeManagers.get(tabId);
    if (existing) return existing;
    const mgr = new MergeManager();
    this._mergeManagers.set(tabId, mgr);
    mgr.on('allDone', () => {
      this._mergeManagers.delete(tabId);
      mgr.dispose();
      this.sendMergeAllDone(tabId);
      this._restoreChain = this._restoreChain
        .then(() => this._restorePreMergeEditors(tabId))
        .catch(err => {
          console.error(
            '[SorcarSidebarView] restorePreMergeEditors failed:',
            err,
          );
        });
    });
    return mgr;
  }

  /**
   * Lazy-init the shared ``AgentClient`` connection to the kiss-web
   * daemon and install the message listener exactly once.
   *
   * The daemon already stamps ``tabId`` onto every outgoing event, so
   * the listener routes messages purely by ``msg.tabId``.  No per-tab
   * process bookkeeping is required on the extension side.
   */
  private _getClient(): AgentClient {
    if (this._client) return this._client;
    const client = new AgentClient();
    this._client = client;
    this._installClientListener(client);
    client.connect();
    // Sync the daemon's work_dir with the current workspace folder.
    // The daemon caches ``self.work_dir`` once at process start from
    // ``KISS_WORKDIR``/``getcwd()``; without this push, every backend
    // path that does not receive an explicit ``workDir`` per command
    // (autocomplete file-list, commit-message, worktree actions)
    // keeps using the stale init value after the user opens a
    // different folder in VS Code.
    client.sendCommand({type: 'setWorkDir', workDir: this._getWorkDir()});
    // Keep the daemon in sync whenever the workspace folder set
    // changes (e.g. user opens a different folder in this window).
    this._workspaceFoldersSub = vscode.workspace.onDidChangeWorkspaceFolders(
      () => {
        const wd = this._getWorkDir();
        this._getClient().sendCommand({type: 'setWorkDir', workDir: wd});
      },
    );
    return client;
  }

  /**
   * Install the unified message listener on the daemon client.
   *
   * Handles every message type (merge, worktree, status, models, etc.)
   * and forwards them to the webview.  ``msg.tabId`` is set by the
   * daemon for tab-scoped events; webview-side handlers route on it.
   */
  private _installClientListener(client: AgentClient): void {
    client.on('message', (msg: ToWebviewMessage) => {
      if (msg.type === 'commitMessage') {
        this._onCommitMessage.fire({message: msg.message, error: msg.error});
      }
      if (msg.type === 'models' && msg.selected) {
        this._selectedModel = msg.selected;
      }
      if (msg.type === 'merge_data') {
        const mergeTabId = msg.tabId;
        if (mergeTabId !== undefined) {
          const mgr = this._getOrCreateMergeManager(mergeTabId);
          this._restoreChain = this._restoreChain
            .then(async () => {
              if (!this._preMergeOpenFiles.has(mergeTabId)) {
                this._preMergeOpenFiles.set(
                  mergeTabId,
                  this._getOpenEditorFiles(),
                );
              }
              await mgr.openMerge(msg.data);
            })
            .catch(err => {
              console.error(
                '[SorcarSidebarView] openMerge failed for tab',
                mergeTabId,
                err,
              );
            });
        }
      }
      if (msg.type === 'worktree_created' || msg.type === 'worktree_done') {
        const dir = msg.worktreeDir;
        const wtTabId = msg.tabId;
        if (dir) {
          if (wtTabId !== undefined) {
            this._worktreeDirs.set(wtTabId, dir);
          }
          void this._openWorktreeInScm(dir);
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
      if (msg.type === 'worktree_result') {
        const wrTabId = msg.tabId;
        if (wrTabId !== undefined) {
          const resolve = this._worktreeActionResolves.get(wrTabId);
          if (resolve) {
            resolve();
            this._worktreeActionResolves.delete(wrTabId);
          }
          this._worktreeProgresses.delete(wrTabId);
        } else {
          // Fallback: resolve all pending
          this._resolveAllWorktreeActions();
        }
        if (msg.success) {
          vscode.window.showInformationMessage(
            msg.message || 'Worktree action completed.',
          );
        } else {
          vscode.window.showErrorMessage(
            msg.message || 'Worktree action failed.',
          );
        }
        if (msg.success && wrTabId !== undefined) {
          const wtDir = this._worktreeDirs.get(wrTabId);
          if (wtDir) {
            void this._closeWorktreeInScm(wtDir);
            this._worktreeDirs.delete(wrTabId);
          }
        }
      }
      if (msg.type === 'autocommit_progress') {
        const apTabId = msg.tabId;
        const progress =
          apTabId !== undefined
            ? this._autocommitProgresses.get(apTabId)
            : this._autocommitProgresses.values().next().value;
        if (progress) {
          progress.report({message: msg.message});
        }
      }
      if (msg.type === 'autocommit_done') {
        const adTabId = msg.tabId;
        if (adTabId !== undefined) {
          const resolve = this._autocommitActionResolves.get(adTabId);
          if (resolve) {
            resolve();
            this._autocommitActionResolves.delete(adTabId);
          }
          this._autocommitProgresses.delete(adTabId);
        }
        if (msg.success) {
          vscode.window.showInformationMessage(
            msg.message || 'Auto-commit completed.',
          );
        } else {
          vscode.window.showErrorMessage(msg.message || 'Auto-commit failed.');
        }
      }

      // Reveal the sidebar when the agent asks a question so the user
      // sees the modal even if they switched to another panel.
      if (msg.type === 'askUser' && this._view) {
        this._view.show(true);
      }

      // ``merge_data`` is handled exclusively by the native VS Code
      // ``MergeManager`` above.  The daemon's ``WebPrinter`` augments
      // every ``merge_data`` event with ``base_text``/``current_text``
      // so browser clients can render an inline diff in chat — but in
      // the extension we already paint the native merge editor, so
      // forwarding the augmented event to the webview would render the
      // diff twice (native merge editor + in-chat inline diff).  Drop
      // it.  ``merge_started`` / ``merge_ended`` / ``merge_nav`` still
      // reach the webview so the in-input merge toolbar (Prev / Next /
      // Accept / Reject buttons) keeps working.
      if (msg.type !== 'merge_data') {
        this._sendToWebview(msg);
      }
      if (msg.type === 'status') {
        const statusTabId = msg.tabId;
        if (msg.running) {
          if (statusTabId !== undefined) this._runningTabs.add(statusTabId);
        } else {
          if (statusTabId !== undefined) this._runningTabs.delete(statusTabId);
          if (this._commitPendingTabs.size > 0) {
            this._onCommitMessage.fire({message: '', error: 'Process stopped'});
          }
        }
      }
    });
  }

  /**
   * Called by VS Code when the sidebar view needs to be rendered.
   */
  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ): void {
    this._view = webviewView;

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
    );

    webviewView.webview.onDidReceiveMessage((message: FromWebviewMessage) =>
      this._handleMessage(message),
    );

    webviewView.onDidChangeVisibility(() => {
      if (webviewView.visible) {
        this._getClient().sendCommand({type: 'getInputHistory'});
      }
    });

    webviewView.onDidDispose(() => {
      this._disposed = true;
      this._resolveAllWorktreeActions();
    });

    if (this._onFirstResolve) {
      const cb = this._onFirstResolve;
      this._onFirstResolve = undefined;
      cb();
    }
  }

  /** Whether the underlying webview is currently visible. */
  get visible(): boolean {
    return this._view?.visible ?? false;
  }

  /** Whether the webview currently has input focus. */
  get hasFocus(): boolean {
    return this._webviewHasFocus;
  }

  /**
   * Snapshot the file paths of all currently open editor tabs.
   *
   * Used before the merge UI opens so we can later close any
   * tabs that were only opened for the merge review.
   */
  private _getOpenEditorFiles(): Set<string> {
    const files = new Set<string>();
    for (const group of vscode.window.tabGroups.all) {
      for (const tab of group.tabs) {
        if (tab.input instanceof vscode.TabInputText) {
          files.add(tab.input.uri.fsPath);
        }
      }
    }
    return files;
  }

  /**
   * Close editor tabs that were not open before the merge started.
   *
   * Reads the snapshot from ``_preMergeOpenFiles``, compares it
   * against the currently open tabs, closes extras, and clears
   * the snapshot.
   */
  private async _restorePreMergeEditors(tabId: string): Promise<void> {
    const snapshot = this._preMergeOpenFiles.get(tabId);
    this._preMergeOpenFiles.delete(tabId);
    if (!snapshot) return;
    const tabsToClose: vscode.Tab[] = [];
    for (const group of vscode.window.tabGroups.all) {
      for (const tab of group.tabs) {
        if (tab.input instanceof vscode.TabInputText) {
          if (!snapshot.has(tab.input.uri.fsPath)) {
            tabsToClose.push(tab);
          }
        }
      }
    }
    if (tabsToClose.length > 0) {
      await vscode.window.tabGroups.close(tabsToClose);
    }
  }

  private _getWorkDir(): string {
    const folders = vscode.workspace.workspaceFolders;
    if (folders && folders.length > 0) {
      return folders[0].uri.fsPath;
    }
    return process.cwd();
  }

  private _sendToWebview(message: ToWebviewMessage): void {
    if (!this._disposed && this._view) {
      this._view.webview.postMessage(message);
    }
  }

  private _sendWelcomeSuggestions(): void {
    const jsonPath = path.join(this._extensionUri.fsPath, 'SAMPLE_TASKS.json');
    try {
      const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
      this._sendToWebview({
        type: 'welcome_suggestions',
        suggestions: data,
      } as ToWebviewMessage);
    } catch {
      this._sendToWebview({
        type: 'welcome_suggestions',
        suggestions: [],
      } as ToWebviewMessage);
    }
  }

  /**
   * Read ``~/.kiss/remote-url.json`` and post the tunnel/local URL to
   * the webview.  Also starts a persistent file watcher that re-sends
   * the URL whenever the file changes (e.g. after a daemon restart
   * creates a new tunnel with a different URL).
   */
  private _sendRemoteUrl(): void {
    const urlFile = path.join(os.homedir(), '.kiss', 'remote-url.json');
    this._tryReadAndSendUrl(urlFile);
    this._watchUrlFile(urlFile);
  }

  /**
   * Try to read the URL file and post to the webview.
   *
   * Always sends a ``remote_url`` message (even on a missing/empty
   * file) so the webview can hide the welcome-page remote-password
   * panel when no Cloudflare tunnel is active.  ``tunnelActive`` is
   * True only when ``data.tunnel`` is present (a real tunnel URL).
   * Dedups against ``_lastSentUrl`` using a key that combines the
   * URL and the active flag so transitions between "no tunnel" and
   * "tunnel established" always reach the webview.
   */
  private _tryReadAndSendUrl(urlFile: string): void {
    let tunnel = '';
    let local = '';
    try {
      const data = JSON.parse(fs.readFileSync(urlFile, 'utf-8'));
      tunnel = data.tunnel || '';
      local = data.local || '';
    } catch {
      /* file missing or malformed — fall through with empty values */
    }
    const tunnelActive = !!tunnel;
    const url = tunnel || local || '';
    const key = `${tunnelActive ? '1' : '0'}|${url}`;
    if (key === this._lastSentUrl) return;
    this._lastSentUrl = key;
    const msg: ToWebviewMessage = {type: 'remote_url', url, tunnelActive};
    const ntfyUrl = this._getNtfyUrl();
    if (ntfyUrl) {
      msg.ntfyUrl = ntfyUrl;
    }
    this._sendToWebview(msg);
  }

  /**
   * Build the ``https://ntfy.sh/{topic}`` URL from ``~/.kiss/ntfy_topic``.
   * Returns an empty string if the file is missing or empty.
   */
  private _getNtfyUrl(): string {
    try {
      const topicFile = path.join(os.homedir(), '.kiss', 'ntfy_topic');
      const topic = fs.readFileSync(topicFile, 'utf-8').trim();
      if (topic) {
        return `https://ntfy.sh/${topic}`;
      }
    } catch {
      /* file missing */
    }
    return '';
  }

  private _urlFileWatchTimer?: ReturnType<typeof setInterval>;

  /**
   * Persistently poll ``~/.kiss/remote-url.json`` every 10 seconds.
   * Re-sends the URL to the webview whenever the file content changes
   * (e.g. after a daemon restart assigns a new tunnel URL).
   */
  private _watchUrlFile(urlFile: string): void {
    if (this._urlFileWatchTimer) return;
    this._urlFileWatchTimer = setInterval(() => {
      this._tryReadAndSendUrl(urlFile);
    }, 10_000);
  }

  /**
   * Poll ``~/.kiss/config.json`` and re-request ``getConfig`` from the
   * daemon whenever its ``remote_password`` changes.
   *
   * On VS Code launch the daemon may still be (re)starting and the
   * remote password may be empty or written only after the activation
   * prompt — so the webview's initial ``getConfig`` can return a blank
   * password, leaving the welcome-page remote-password panel empty
   * until the user opens the Settings panel.  This watcher closes that
   * gap: when the persisted password first becomes non-empty (or later
   * changes), it re-issues ``getConfig`` so the daemon broadcasts a
   * fresh ``configData`` and the webview repopulates both the welcome
   * and settings password fields automatically.
   */
  private _watchConfigFile(): void {
    if (this._configFileWatchTimer) return;
    this._checkConfigFile();
    this._configFileWatchTimer = setInterval(
      () => this._checkConfigFile(),
      2_000,
    );
  }

  /**
   * Read ``~/.kiss/config.json`` once and re-request ``getConfig`` from
   * the daemon when the persisted ``remote_password`` first becomes
   * non-empty or later changes.  A missing / mid-write file is ignored
   * and retried on the next poll tick.
   */
  private _checkConfigFile(): void {
    const configFile = path.join(os.homedir(), '.kiss', 'config.json');
    let pw: string;
    try {
      const data = JSON.parse(fs.readFileSync(configFile, 'utf-8'));
      pw = typeof data.remote_password === 'string' ? data.remote_password : '';
    } catch {
      // File missing or mid-write — retry on the next tick.
      return;
    }
    const first = this._lastSeenRemotePassword === undefined;
    const changed = pw !== this._lastSeenRemotePassword;
    this._lastSeenRemotePassword = pw;
    // Re-fetch when the password changed, and also on the first
    // successful read of a non-empty password (the webview's init
    // getConfig may have raced a transient empty/truncated config).
    if ((changed && !first) || (first && pw !== '')) {
      this._getClient().sendCommand({type: 'getConfig'});
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
    } catch {
      /* ignored */
    }
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
  ): void {
    const effectiveWorkDir = workDir || this._getWorkDir();
    this._sendToWebview({type: 'setTaskText', text: prompt, tabId});
    this._sendToWebview({type: 'status', running: true, tabId});
    this._getClient().sendCommand({
      type: 'run',
      prompt,
      model,
      workDir: effectiveWorkDir,
      activeFile,
      attachments,
      useWorktree,
      useParallel,
      autoCommit,
      tabId,
    });
  }

  private async _handleMessage(message: FromWebviewMessage): Promise<void> {
    const forwarded = FORWARDED_COMMANDS[message.type];
    if (forwarded) {
      const src = message as unknown as Record<string, unknown>;
      const cmd: Record<string, unknown> = {type: message.type};
      for (const field of forwarded) cmd[field] = src[field];
      this._getClient().sendCommand(cmd as unknown as AgentCommand);
      return;
    }
    switch (message.type) {
      case 'ready': {
        const readyTabId = message.tabId;
        if (readyTabId) this._activeTabId = readyTabId;
        const client = this._getClient();
        client.sendCommand({type: 'getModels'});
        this._sendWelcomeSuggestions();
        this._sendRemoteUrl();
        client.sendCommand({type: 'getInputHistory'});
        // Request the current config so the welcome-page remote-password
        // panel is populated from the established connection (the webview's
        // own init getConfig can race the daemon restart / password write).
        client.sendCommand({type: 'getConfig'});
        // Re-fetch config whenever ~/.kiss/config.json changes so a
        // remote_password set after launch (e.g. via the activation prompt
        // or a daemon restart) reaches the welcome panel without the user
        // having to open the Settings panel.
        this._watchConfigFile();
        this._sendToWebview({type: 'focusInput'} as ToWebviewMessage);
        // Auto-reload events for restored tabs that had active sessions.
        // The daemon's _RunningAgentState retains state across reloads, so resumeSession
        // either replays persisted events or re-subscribes to a still-running
        // task via the printer's subscriber map.
        const restoredTabs = message.restoredTabs;
        if (restoredTabs && restoredTabs.length > 0) {
          for (const rt of restoredTabs) {
            client.sendCommand({
              type: 'resumeSession',
              chatId: rt.chatId,
              tabId: rt.tabId,
            });
          }
        }
        break;
      }

      case 'submit': {
        const tabId = message.tabId;
        if (tabId) this._activeTabId = tabId;
        if (tabId !== undefined && this._runningTabs.has(tabId)) return;

        const tabWorkDir = message.workDir;
        const effectiveWorkDir = tabWorkDir || this._getWorkDir();

        const trimmed = message.prompt.trim();
        if (trimmed && !trimmed.includes('\n')) {
          const bare = trimmed.replace(/^PWD[/\\]/, '');
          const resolved = path.resolve(effectiveWorkDir, bare);
          // H4 — only treat as a file shortcut when the resolved path is
          // strictly inside the work dir; otherwise fall through and let
          // the prompt run as a normal task.
          if (
            isPathInside(resolved, effectiveWorkDir) &&
            fs.existsSync(resolved) &&
            fs.statSync(resolved).isFile()
          ) {
            const uri = vscode.Uri.file(resolved);
            const doc = await vscode.workspace.openTextDocument(uri);
            await vscode.window.showTextDocument(doc, {
              preview: false,
              viewColumn: vscode.ViewColumn.One,
            });
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
        );
        break;
      }

      case 'stop': {
        const stopTabId = message.tabId;
        const client = this._getClient();
        if (stopTabId !== undefined) {
          client.sendCommand({type: 'stop', tabId: stopTabId});
        } else {
          // Stop every running tab on this connection.
          for (const tab of this._runningTabs) {
            client.sendCommand({type: 'stop', tabId: tab});
          }
        }
        break;
      }

      case 'selectModel': {
        this._selectedModel = message.model;
        const selTabId = message.tabId;
        this._getClient().sendCommand({
          type: 'selectModel',
          model: message.model,
          tabId: selTabId,
        });
        break;
      }

      case 'userAnswer': {
        const ansTabId = message.tabId;
        if (ansTabId !== undefined) {
          this._getClient().sendCommand({
            type: 'userAnswer',
            answer: message.answer,
            tabId: ansTabId,
          });
        }
        break;
      }

      case 'recordFileUsage':
        if (message.path) {
          this._getClient().sendCommand({
            type: 'recordFileUsage',
            path: message.path,
            workDir: message.workDir,
          });
        }
        break;

      case 'openFile':
        if (message.path) {
          const wd = this._getWorkDir();
          const filePath = path.resolve(wd, message.path);
          // H4 — refuse to open files outside the workspace.  Use
          // path.relative so symlinks/normalised paths are compared
          // properly; isPathInside() avoids prefix-match false
          // positives like "/wd-evil" matching "/wd".
          if (
            !isPathInside(filePath, wd) ||
            !fs.existsSync(filePath) ||
            !fs.statSync(filePath).isFile()
          ) {
            console.warn(
              '[SorcarSidebarView] refusing to open file outside workspace:',
              message.path,
            );
            break;
          }
          {
            const uri = vscode.Uri.file(filePath);
            const doc = await vscode.workspace.openTextDocument(uri);
            const editor = await vscode.window.showTextDocument(doc, {
              preview: false,
              viewColumn: vscode.ViewColumn.One,
            });
            if (message.line !== undefined && message.line > 0) {
              const pos = new vscode.Position(message.line - 1, 0);
              editor.selection = new vscode.Selection(pos, pos);
              editor.revealRange(
                new vscode.Range(pos, pos),
                vscode.TextEditorRevealType.InCenter,
              );
            }
          }
        }
        break;

      case 'resumeSession': {
        const resumeTabId = message.tabId;
        // The daemon's _replay_session re-subscribes a still-running
        // chat via the printer's subscriber map when chatId belongs to
        // a live _RunningAgentState, otherwise replays persisted events.  No
        // process-level reattachment is required on the extension
        // side anymore.
        this._getClient().sendCommand({
          type: 'resumeSession',
          chatId: message.id,
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
        this._getClient().sendCommand({
          type: 'complete',
          query: message.query,
          tabId: this._activeTabId || undefined,
          activeFile: editorFile || undefined,
          activeFileContent: completeDoc?.getText(),
        });
        break;
      }

      case 'mergeAction': {
        const mTabId = message.tabId || this._activeTabId;
        const mgr = this._mergeManagers.get(mTabId);
        if (!mgr) {
          if (message.action === 'all-done') {
            this.sendMergeAllDone(mTabId);
          }
          break;
        }
        const mAction = message.action;
        const method = MERGE_ACTIONS[mAction as keyof typeof MERGE_ACTIONS];
        if (method) void mgr[method]();
        else if (mAction === 'all-done') {
          this.sendMergeAllDone(mTabId);
        }
        break;
      }

      case 'worktreeAction': {
        const wtAction = message.action;
        const wtTabId = message.tabId;
        const progressTitle =
          wtAction === 'merge'
            ? 'Committing and merging worktree…'
            : wtAction === 'discard'
              ? 'Discarding worktree…'
              : 'Processing worktree action…';
        this._showActionProgress(
          progressTitle,
          wtTabId,
          this._worktreeProgresses,
          this._worktreeActionResolves,
        );
        this._getClient().sendCommand({
          type: 'worktreeAction',
          action: wtAction,
          tabId: wtTabId,
        });
        break;
      }

      case 'autocommitAction': {
        const acAction = message.action;
        const acTabId = message.tabId;
        if (acAction === 'commit') {
          this._showActionProgress(
            'Auto-committing…',
            acTabId,
            this._autocommitProgresses,
            this._autocommitActionResolves,
          );
        }
        this._getClient().sendCommand({
          type: 'autocommitAction',
          action: acAction,
          tabId: acTabId,
          workDir: message.workDir,
        });
        break;
      }

      case 'resolveDroppedPaths': {
        const workDir = this._getWorkDir();
        const paths = (message.uris || [])
          .map((uri: string) => {
            try {
              const absPath = vscode.Uri.parse(uri).fsPath;
              return path.relative(workDir, absPath);
            } catch {
              return '';
            }
          })
          .filter((p: string) => p && !p.startsWith('..'));
        this._sendToWebview({type: 'droppedPaths', paths} as ToWebviewMessage);
        break;
      }

      case 'webviewFocusChanged':
        this._webviewHasFocus = message.focused;
        break;

      case 'sizeReport': {
        const cb = this._sizeReportResolver;
        this._sizeReportResolver = undefined;
        if (cb) cb({inner: message.innerWidth, screen: message.screenWidth});
        break;
      }

      case 'focusEditor':
        vscode.commands.executeCommand(
          'workbench.action.focusFirstEditorGroup',
        );
        break;

      case 'runUpdate':
        this._runUpdate();
        break;

      case 'closeTab': {
        const closeTabId = message.tabId;
        if (closeTabId) {
          this._getClient().sendCommand({
            type: 'closeTab',
            tabId: closeTabId,
          });
        }
        break;
      }
    }
  }

  /**
   * Run ``install.sh`` from the KISS Sorcar source checkout to update
   * the extension and notify the user that an update is being
   * installed.
   *
   * The script always lives at ``~/kiss_ai/install.sh`` because the
   * curl-piped bootstrapper (``scripts/install.sh``) clones the repo
   * to that fixed path — see :mod:`installerPath` for the rationale.
   * It is executed in a dedicated integrated terminal so its progress
   * is visible.  When the script is missing an error message points
   * the user at the canonical install root rather than their current
   * workspace.
   */
  private _runUpdate(): void {
    const scriptPath = findInstallScript();
    if (!scriptPath) {
      vscode.window.showErrorMessage(
        `Cannot update KISS Sorcar: install.sh not found in ${kissAiRoot()}.`,
      );
      return;
    }
    vscode.window.showInformationMessage(
      'An update of KISS Sorcar is getting installed…',
    );
    const terminal = vscode.window.createTerminal({
      name: 'KISS Sorcar Update',
      cwd: path.dirname(scriptPath),
    });
    terminal.show();
    terminal.sendText(`bash '${scriptPath.replace(/'/g, "'\\''")}'`);
  }

  /**
   * Dispatch a merge command to the active tab's MergeManager.
   *
   * Used by extension.ts keyboard shortcuts that don't know the tab ID.
   * Routes to the MergeManager of ``_activeTabId``.
   */
  public handleMergeCommand(cmd: MergeCommand): void {
    const mgr = this._mergeManagers.get(this._activeTabId);
    if (mgr) void mgr[cmd]();
  }

  /** Notify the agent that all merge changes have been reviewed. */
  public sendMergeAllDone(tabId?: string): void {
    this._getClient().sendCommand({
      type: 'mergeAction',
      action: 'all-done',
      tabId,
      workDir: this._getWorkDir(),
    });
  }

  /** Submit a task programmatically (e.g. from runSelection command). */
  public submitTask(prompt: string): void {
    if (!prompt.trim()) return;
    this._startTask(
      prompt.trim(),
      this._selectedModel,
      this._getVisibleEditorFile() || undefined,
    );
  }

  /** Stop the currently running task in the active tab. */
  public stopTask(): void {
    this._sendToWebview({type: 'triggerStop'} as ToWebviewMessage);
  }

  /** Focus the chat input in the sidebar. */
  public async focusChatInput(): Promise<void> {
    if (!this._view) {
      // Webview not yet resolved — trigger resolution by focusing the view.
      // The .focus command opens the secondary sidebar and resolves the
      // webview, but resolution can be slow on first launch.  Poll up to
      // 2 seconds (10 × 200ms) so we don't miss it.
      await vscode.commands.executeCommand(
        'kissSorcar.chatViewSecondary.focus',
      );
      for (let i = 0; i < 10 && !this._view; i++) {
        await new Promise(r => setTimeout(r, 200));
      }
    }
    if (this._view) {
      this._view.show(true);
      await new Promise(r => setTimeout(r, 150));
      this._sendToWebview({type: 'focusInput'});
    }
  }

  /** Append text to the chat input and focus it. */
  public async appendToInput(text: string): Promise<void> {
    if (this._view) {
      this._view.show(true);
      await new Promise(r => setTimeout(r, 150));
      this._sendToWebview({type: 'appendToInput', text});
    }
  }

  /** Start a new conversation in a new tab (without affecting running tabs). */
  public newConversation(): void {
    this._sendToWebview({type: 'clearChat'});
  }

  /**
   * Ask the webview to report its current sidebar width (``window.innerWidth``)
   * and the host screen width (``screen.availWidth``).  Returns ``undefined``
   * if the webview does not respond within ``timeoutMs`` (default 1500ms).
   *
   * The host VS Code window width is not exposed by the extension API, so
   * ``screen.availWidth`` is used as the closest proxy — it equals the VS
   * Code window width when the window is maximized (the typical case on
   * first install).
   */
  private _measureSidebar(
    timeoutMs: number = 1500,
  ): Promise<{inner: number; screen: number} | undefined> {
    if (!this._view) return Promise.resolve(undefined);
    // Drop any previous pending resolver — we only want the latest.
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
      this._sendToWebview({type: 'measureSize'} as ToWebviewMessage);
      setTimeout(() => finish(undefined), timeoutMs);
    });
  }

  /**
   * Iteratively resize the secondary side bar so its width is approximately
   * one-third of the VS Code window width.
   *
   * Algorithm: measure the current sidebar width via the webview, compare to
   * ``screenWidth / 3``, then call ``workbench.action.increaseViewSize`` or
   * ``workbench.action.decreaseViewSize`` (each adjusts by a fixed amount)
   * and re-measure.  Stops when the width is within ``tolerance`` (default
   * 6 % of target) or after ``maxIterations`` attempts (default 30).
   *
   * Used on first activation so the chat panel has enough room without
   * requiring the user to drag the splitter.  The webview needs to be
   * focused for the increase/decrease commands to apply to the secondary
   * side bar — callers must ensure ``focusAuxiliaryBar`` is invoked first.
   */
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
      // Give VS Code a moment to apply the resize before measuring again.
      await new Promise(r => setTimeout(r, 60));
      // Bail out if the resize command had no effect for two consecutive
      // iterations (e.g. we hit the min/max sidebar size).
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
   * Generate a commit message using this view's agent process.
   *
   * @param token Optional cancellation token.
   * @param tabId Optional tab ID — each tab can independently request a
   *              commit message without blocking other tabs.
   */
  public generateCommitMessage(
    token?: vscode.CancellationToken,
    tabId: string = '',
  ): Promise<void> {
    if (this._commitPendingTabs.has(tabId)) return Promise.resolve();
    this._commitPendingTabs.add(tabId);
    this._getClient().sendCommand({
      type: 'generateCommitMessage',
      model: this._selectedModel,
      tabId,
      workDir: this._getWorkDir(),
    });

    return new Promise<void>(resolve => {
      let resolved = false;
      const done = () => {
        if (resolved) return;
        resolved = true;
        this._commitPendingTabs.delete(tabId);
        disposable.dispose();
        clearTimeout(timer);
        resolve();
      };
      const disposable = this._onCommitMessage.event(() => done());
      token?.onCancellationRequested(() => done());
      const timer = setTimeout(done, 30_000);
    });
  }

  /**
   * Cleanup: dispose listeners and close the daemon connection.
   *
   * Closing the UDS socket only ends this client's connection — the
   * daemon's ``_RunningAgentState`` lives on through the deferred-close grace
   * window so in-flight agent tasks survive an extension reload.  A
   * fresh activation re-connects and re-subscribes via ``ready`` /
   * ``resumeSession`` exactly as a browser refresh does.
   */
  public dispose(): void {
    this._disposed = true;
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
    for (const mgr of this._mergeManagers.values()) mgr.dispose();
    this._mergeManagers.clear();
    if (this._client) {
      this._client.dispose();
      this._client = null;
    }
    this._onCommitMessage.dispose();
  }
}
