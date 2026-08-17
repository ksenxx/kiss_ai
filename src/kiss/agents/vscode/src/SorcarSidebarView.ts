// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
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
 * a real file inside *root* — comparing REAL paths, so a symlink inside
 * the workspace cannot smuggle in a file that actually lives outside it.
 */
function resolveWorkspaceFile(p: string, root: string): string | null {
  try {
    const resolved = path.resolve(root, p);
    if (!isPathInside(resolved, root)) return null;
    const real = fs.realpathSync(resolved);
    const realRoot = fs.realpathSync(root);
    if (!isPathInside(real, realRoot)) return null;
    if (!fs.statSync(real).isFile()) return null;
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
import {getDefaultModel} from './DependencyInstaller';
import {buildChatHtml, readSampleTasks} from './SorcarTab';
import {VoiceWakeService} from './voiceWake';
import {kissHomeDir} from './userAssets';
import {playVoiceAckClip} from './voiceAckPlayer';
import {findInstallScript, kissAiRoot} from './installerPath';
import {
  FromWebviewMessage,
  ToWebviewMessage,
  Attachment,
  AgentCommand,
} from './types';
import {
  resolveWebviewNotificationAction,
  setWebviewNotificationPoster,
  showErrorNotification,
  showInformationNotification,
  showWarningNotification,
  withWebviewNotificationProgress,
} from './WebviewNotifications';

const FORWARDED_COMMANDS: Record<string, readonly string[]> = {
  appendUserMessage: ['prompt', 'tabId'],
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
};

export class SorcarSidebarView implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  private _client: AgentClient | null = null;
  private _api: SorcarApi | null = null;
  private _daemonConnected: boolean = false;
  private _activeTabId: string = '';
  private _extensionUri: vscode.Uri;
  private _selectedModel: string;
  private _runningTabs: Set<string> = new Set();
  private _ownTabs: Set<string> = new Set();
  private _webviewHasFocus: boolean = false;
  private _webviewReady: boolean = false;

  private _voiceWake: VoiceWakeService | undefined;
  private _voiceSensitivity: number | undefined;
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
    if (tabId !== undefined) {
      const prev = resolveMap.get(tabId);
      if (prev) {
        resolveMap.delete(tabId);
        progressMap.delete(tabId);
        prev();
      }
    }
    withWebviewNotificationProgress(
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
          if (
            timeoutMs !== undefined &&
            Number.isFinite(timeoutMs) &&
            timeoutMs > 0
          ) {
            setTimeout(() => {
              if (tabId !== undefined && resolveMap.get(tabId) === resolve) {
                resolveMap.delete(tabId);
                // Drop the progress object too: a late progress event must
                // not report() into a toast that has already been closed.
                if (progressMap.get(tabId) === progress) {
                  progressMap.delete(tabId);
                }
                resolve();
              }
            }, timeoutMs);
          }
        });
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

  constructor(extensionUri: vscode.Uri) {
    this._extensionUri = extensionUri;
    this._selectedModel =
      vscode.workspace
        .getConfiguration('kissSorcar')
        .get<string>('defaultModel') || getDefaultModel();
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
      this._daemonConnected = true;
      this._sendToWebview({type: 'daemonStatus', connected: true});
      if (this._view) {
        this._getApi().getModels();
        this._getApi().getInputHistory();
        this._getApi().getConfig();
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
    }
  }

  private _installClientListener(client: AgentClient): void {
    client.on('message', (msg: ToWebviewMessage) => {
      if (msg.type === 'configData' && msg.config) {
        msg.config.work_dir = this._getWorkDir();
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
        for (const t of msg.tabs) {
          if (t && t.tabId) listed.add(t.tabId);
        }
        for (const staleId of this._registryTabs) {
          if (!listed.has(staleId)) {
            this._ownTabs.delete(staleId);
            this._cleanupTabResources(staleId);
          }
        }
        this._registryTabs = listed;
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
      if (msg.type === 'worktree_result' && msg.success) {
        // Mirrors the unconditional recording above: a merge/discard
        // finished by any client retires the worktree directory, so the
        // fallback entry must go even when this window never claimed
        // the tab. git.close on a repository that was never opened is
        // a harmless no-op.
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
    if (this._terminated) return;
    for (const sub of this._viewSubs) sub.dispose();
    this._viewSubs = [];
    this._view = webviewView;
    this._webviewReady = false;
    setWebviewNotificationPoster(message =>
      this._sendToWebview(message as ToWebviewMessage),
    );
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
      if (webviewView.visible) {
        this._getApi().getInputHistory();
        if (this._voiceWakeSuspendedByHide) {
          this._voiceWakeSuspendedByHide = false;
          this._voiceWake?.start(this._voiceSensitivity);
        }
      } else if (this._voiceWake?.running) {
        this._voiceWakeSuspendedByHide = true;
        this._voiceWake.stop();
      }
    });
    this._viewSubs.push(visibilitySub);

    this._viewSubs.push(
      webviewView.onDidDispose(() => {
        if (this._view === webviewView) {
          this._view = undefined;
          this._disposed = true;
          this._webviewReady = false;
          setWebviewNotificationPoster(undefined);
          this._voiceWakeSuspendedByHide = false;
          this._voiceWake?.stop();
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
    return process.cwd();
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

  private _resolveTabFile(
    p: string,
    wd: string,
    tabId: string | undefined,
  ): string | null {
    const resolved = resolveWorkspaceFile(p, wd);
    if (resolved) return resolved;
    const wtDir =
      tabId !== undefined ? this._worktreeDirs.get(tabId) : undefined;
    if (wtDir && wtDir !== wd) return resolveWorkspaceFile(p, wtDir);
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
    try {
      const data = JSON.parse(fs.readFileSync(urlFile, 'utf-8'));
      tunnel = data.tunnel || '';
      local = data.local || '';
    } catch {}
    const tunnelActive = !!tunnel;
    const url = tunnel || local || '';
    const ntfyUrl = this._getNtfyUrl();
    const key = `${tunnelActive ? '1' : '0'}|${url}|${ntfyUrl}`;
    if (key === this._lastSentUrl) return;
    this._lastSentUrl = key;
    const msg: ToWebviewMessage = {type: 'remote_url', url, tunnelActive};
    if (ntfyUrl) {
      msg.ntfyUrl = ntfyUrl;
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
          const resolved = this._resolveTabFile(
            trimmed,
            effectiveWorkDir,
            tabId,
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
        this._sendToWebview({type: 'droppedPaths', paths} as ToWebviewMessage);
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
            (listening, error) =>
              this._sendToWebview({type: 'voiceState', listening, error}),
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
        this._voiceWakeSuspendedByHide = false;
        if (message.enabled) this._voiceWake.start(this._voiceSensitivity);
        else this._voiceWake.stop();
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
        if (typeof message.value !== 'number') break;
        this._voiceSensitivity = message.value;
        if (this._voiceWake?.running) {
          this._voiceWake.stop();
          this._voiceWake.start(this._voiceSensitivity);
        }
        break;
      }

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
    }
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

  public runUpdate(): void {
    const scriptPath = findInstallScript();
    if (!scriptPath) {
      showErrorNotification(
        `Cannot update KISS Sorcar: install.sh not found in ${kissAiRoot()}.`,
      );
      return;
    }
    showInformationNotification(
      'An update of KISS Sorcar is getting installed…',
    );
    const terminal = vscode.window.createTerminal({
      name: 'KISS Sorcar Update',
      cwd: path.dirname(scriptPath),
    });
    terminal.show();
    const escScript = scriptPath.replace(/'/g, "'\\''");
    const escDir = path.dirname(scriptPath).replace(/'/g, "'\\''");
    const preflight = [
      `cd '${escDir}'`,
      "echo '>>> Pre-flight: synchronizing repo with origin before install.sh...'",
      'git fetch --force --tags --prune origin 2>/dev/null || true',
      '_kiss_stashed=; if [ -n "$(git status --porcelain 2>/dev/null)" ]; then git stash push --include-untracked -m \'kiss-update-preflight\' >/dev/null 2>&1 && _kiss_stashed=1 || _kiss_stashed=; fi',
      "git reset --hard '@{upstream}' 2>/dev/null || git reset --hard origin/HEAD 2>/dev/null || true",
      'if [ -n "$_kiss_stashed" ]; then git stash pop >/dev/null 2>&1 || true; fi',
      `bash '${escScript}'`,
    ].join('; ');
    terminal.sendText(preflight);
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
    if (!this._view) {
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
    setWebviewNotificationPoster(undefined);
    this._voiceWakeSuspendedByHide = false;
    this._voiceWake?.dispose();
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
    // Each panel's onDidDispose deletes its own map entry, so iterate a
    // snapshot and clear at the end.
    for (const panel of [...this._htmlPreviewPanels.values()]) {
      panel.dispose();
    }
    this._htmlPreviewPanels.clear();
  }
}
