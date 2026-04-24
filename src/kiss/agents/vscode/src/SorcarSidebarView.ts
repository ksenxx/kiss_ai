/**
 * Sidebar chat view for Sorcar.
 * Provides a WebviewViewProvider that renders the chat UI in the
 * VS Code secondary sidebar.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import {AgentProcess} from './AgentProcess';
import {MergeManager} from './MergeManager';
import {getDefaultModel} from './DependencyInstaller';
import {buildChatHtml} from './SorcarTab';
import {
  FromWebviewMessage,
  ToWebviewMessage,
  Attachment,
  AgentCommand,
} from './types';

/**
 * WebviewViewProvider for the KISS Sorcar chat in the secondary sidebar.
 *
 * Hosts the chat HTML/JS/CSS interface with its own AgentProcess.
 */
export class SorcarSidebarView implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  /** Per-tab task processes. Created on submit, disposed on next submit. */
  private _taskProcesses: Map<string, AgentProcess> = new Map();
  /** Shared service process for non-task commands (getModels, etc.). */
  private _serviceProcess: AgentProcess | null = null;
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
  private _preMergeOpenFiles: Map<string, Set<string>> = new Map();
  private _restoreChain: Promise<void> = Promise.resolve();

  /** Resolve all pending worktree/autocommit action promises and clear maps. */
  private _resolveAllWorktreeActions(): void {
    for (const resolve of this._worktreeActionResolves.values()) resolve();
    this._worktreeActionResolves.clear();
    this._worktreeProgresses.clear();
    for (const resolve of this._autocommitActionResolves.values()) resolve();
    this._autocommitActionResolves.clear();
    this._autocommitProgresses.clear();
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
   * Get or create the shared service process for non-task commands.
   *
   * The service process handles global commands (getModels, getHistory,
   * getFiles, complete, etc.) and per-tab state commands (resumeSession,
   * getAdjacentTask) when no task process exists for that tab.
   */
  private _getServiceProcess(): AgentProcess {
    if (this._serviceProcess) return this._serviceProcess;
    this._serviceProcess = new AgentProcess('__service__');
    this._setupProcessListeners(this._serviceProcess, '');
    this._serviceProcess.start(this._getWorkDir());
    return this._serviceProcess;
  }

  /**
   * Get the best process for a specific tab.
   *
   * Returns the tab's task process if one exists (it has the tab's
   * per-tab agent state from the most recent task), otherwise falls
   * back to the shared service process.
   */
  private _getTabProcess(tabId: string): AgentProcess {
    return this._taskProcesses.get(tabId) || this._getServiceProcess();
  }

  /**
   * Create a fresh task process for a new task in the given tab.
   *
   * Disposes any existing task process for that tab first, ensuring
   * each task runs in a clean, isolated Python subprocess.
   */
  private _createTaskProcess(tabId: string): AgentProcess {
    const old = this._taskProcesses.get(tabId);
    if (old) {
      old.dispose();
      this._taskProcesses.delete(tabId);
    }
    const proc = new AgentProcess(tabId);
    this._taskProcesses.set(tabId, proc);
    this._setupProcessListeners(proc, tabId);
    return proc;
  }

  /**
   * Set up event listeners on a per-tab AgentProcess.
   *
   * Handles all message types (merge, worktree, status, etc.) and
   * forwards them to the webview. Injects tabId into messages that
   * don't already have one.
   */
  private _setupProcessListeners(proc: AgentProcess, tabId: string): void {
    proc.on('message', (msg: ToWebviewMessage) => {
      // Inject tabId if the Python side didn't set it
      if (msg.tabId === undefined && tabId) {
        msg.tabId = tabId;
      }

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

      this._sendToWebview(msg);
      if (msg.type === 'status') {
        const statusTabId = msg.tabId;
        if (msg.running) {
          if (statusTabId !== undefined) this._runningTabs.add(statusTabId);
        } else {
          if (statusTabId !== undefined) this._runningTabs.delete(statusTabId);
          this._sendActiveFileInfo();
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
        const proc = this._getServiceProcess();
        proc.sendCommand({type: 'getInputHistory'});
        this._sendActiveFileInfo();
      }
    });

    webviewView.onDidDispose(() => {
      this._disposed = true;
      this._resolveAllWorktreeActions();
    });
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

  private _sendActiveFileInfo(): void {
    const fpath = this._getVisibleEditorFile();
    const isPrompt = !!fpath && fpath.toLowerCase().endsWith('.md');
    this._sendToWebview({
      type: 'activeFileInfo',
      isPrompt,
      filename: isPrompt ? path.basename(fpath) : '',
      path: fpath,
    } as ToWebviewMessage);
  }

  private async _openWorktreeInScm(worktreeDir: string): Promise<void> {
    try {
      const gitExt = vscode.extensions.getExtension('vscode.git');
      if (!gitExt) return;
      const git = gitExt.isActive ? gitExt.exports : await gitExt.activate();
      const api = git.getAPI(1);
      if (api.openRepository) {
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
  ): void {
    const effectiveWorkDir = workDir || this._getWorkDir();
    const proc = tabId
      ? this._createTaskProcess(tabId)
      : this._createTaskProcess(this._activeTabId || '__default__');
    const started = proc.start(effectiveWorkDir);
    if (!started) {
      if (tabId !== undefined) this._runningTabs.delete(tabId);
      this._sendToWebview({type: 'status', running: false, tabId});
      return;
    }
    this._sendToWebview({type: 'setTaskText', text: prompt, tabId});
    this._sendToWebview({type: 'status', running: true, tabId});
    proc.sendCommand({
      type: 'run',
      prompt,
      model,
      workDir: effectiveWorkDir,
      activeFile,
      attachments,
      useWorktree,
      useParallel,
      tabId,
    });
  }

  private async _handleMessage(message: FromWebviewMessage): Promise<void> {
    switch (message.type) {
      case 'ready': {
        const readyTabId = message.tabId;
        if (readyTabId) this._activeTabId = readyTabId;
        // Use the tab's task process if it exists, otherwise the service process
        const readyProc = readyTabId
          ? this._getTabProcess(readyTabId)
          : this._getServiceProcess();
        readyProc.sendCommand({type: 'getModels'});
        this._sendWelcomeSuggestions();
        readyProc.sendCommand({type: 'getInputHistory'});
        this._sendActiveFileInfo();
        this._sendToWebview({type: 'focusInput'} as ToWebviewMessage);
        // Auto-reload events for restored tabs that had active sessions
        const restoredTabs = message.restoredTabs;
        if (restoredTabs && restoredTabs.length > 0) {
          const svc = this._getServiceProcess();
          for (const rt of restoredTabs) {
            svc.sendCommand({
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
          if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) {
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
        );
        break;
      }

      case 'stop': {
        const stopTabId = message.tabId;
        if (stopTabId !== undefined) {
          const stopProc = this._taskProcesses.get(stopTabId);
          if (stopProc) stopProc.sendCommand({type: 'stop', tabId: stopTabId});
        } else {
          // Stop all running task processes
          for (const proc of this._taskProcesses.values()) proc.stop();
        }
        break;
      }

      case 'selectModel': {
        this._selectedModel = message.model;
        const selTabId = message.tabId;
        // Persist model selection via service process
        this._getServiceProcess().sendCommand({
          type: 'selectModel',
          model: message.model,
          tabId: selTabId,
        });
        break;
      }

      case 'getModels':
      case 'getInputHistory':
        this._getServiceProcess().sendCommand({
          type: message.type,
        } as AgentCommand);
        break;

      case 'newChat': {
        const newChatTabId = message.tabId;
        const newChatProc = newChatTabId
          ? this._getTabProcess(newChatTabId)
          : this._getServiceProcess();
        newChatProc.sendCommand({type: 'newChat', tabId: newChatTabId});
        break;
      }

      case 'getHistory':
        this._getServiceProcess().sendCommand({
          type: 'getHistory',
          query: message.query,
          offset: message.offset,
          generation: message.generation,
        });
        break;

      case 'getFiles':
        this._getServiceProcess().sendCommand({
          type: 'getFiles',
          prefix: message.prefix,
        });
        break;

      case 'userAnswer': {
        const ansTabId = message.tabId;
        const ansProc = ansTabId
          ? this._taskProcesses.get(ansTabId)
          : undefined;
        if (ansProc)
          ansProc.sendCommand({
            type: 'userAnswer',
            answer: message.answer,
            tabId: ansTabId,
          });
        break;
      }

      case 'userActionDone':
        this._getServiceProcess().sendCommand({
          type: 'userAnswer',
          answer: 'done',
        });
        break;

      case 'recordFileUsage':
        if (message.path) {
          this._getServiceProcess().sendCommand({
            type: 'recordFileUsage',
            path: message.path,
          });
        }
        break;

      case 'openFile':
        if (message.path) {
          const filePath = path.resolve(this._getWorkDir(), message.path);
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
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
        const resumeProc = resumeTabId
          ? this._getTabProcess(resumeTabId)
          : this._getServiceProcess();
        resumeProc.sendCommand({
          type: 'resumeSession',
          chatId: message.id,
          tabId: resumeTabId,
        });
        break;
      }

      case 'getAdjacentTask': {
        const adjTabId = message.tabId;
        const adjProc = adjTabId
          ? this._getTabProcess(adjTabId)
          : this._getServiceProcess();
        adjProc.sendCommand({
          type: 'getAdjacentTask',
          tabId: adjTabId,
          task: message.task,
          direction: message.direction,
        });
        break;
      }

      case 'getWelcomeSuggestions':
        this._sendWelcomeSuggestions();
        break;

      case 'complete': {
        const editorFile = this._getVisibleEditorFile();
        const completeDoc = editorFile
          ? vscode.workspace.textDocuments.find(
              d => d.uri.fsPath === editorFile,
            )
          : undefined;
        this._getServiceProcess().sendCommand({
          type: 'complete',
          query: message.query,
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
        const mergeDispatch: Record<string, () => void> = {
          accept: () => mgr.acceptChange(),
          reject: () => mgr.rejectChange(),
          prev: () => mgr.prevChange(),
          next: () => mgr.nextChange(),
          'accept-all': () => mgr.acceptAll(),
          'reject-all': () => mgr.rejectAll(),
          'accept-file': () => mgr.acceptFile(),
          'reject-file': () => mgr.rejectFile(),
        };
        const mAction = message.action;
        const handler = mergeDispatch[mAction];
        if (handler) handler();
        else if (mAction === 'all-done') {
          this.sendMergeAllDone(mTabId);
        }
        break;
      }

      case 'generateCommitMessage':
        void this.generateCommitMessage();
        break;

      case 'runPrompt': {
        // runPrompt doesn't have a tabId from the webview; allow if any tab is free
        if (this._runningTabs.size > 0) return;
        const promptPath = this._getVisibleEditorFile();
        if (!promptPath || !promptPath.toLowerCase().endsWith('.md')) return;
        const promptDoc = vscode.workspace.textDocuments.find(
          d => d.uri.fsPath === promptPath,
        );
        if (!promptDoc) return;
        const content = promptDoc.getText();
        if (!content.trim()) return;
        this._startTask(
          content,
          this._selectedModel,
          promptPath,
          undefined,
          undefined,
          undefined,
          this._activeTabId || undefined,
        );
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
        const worktreeTimeout = 120_000;
        vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: progressTitle,
          },
          progress => {
            if (wtTabId !== undefined) {
              this._worktreeProgresses.set(wtTabId, progress);
            }
            return new Promise<void>(resolve => {
              if (wtTabId !== undefined) {
                this._worktreeActionResolves.set(wtTabId, resolve);
              }
              setTimeout(() => {
                if (
                  wtTabId !== undefined &&
                  this._worktreeActionResolves.get(wtTabId) === resolve
                ) {
                  this._worktreeActionResolves.delete(wtTabId);
                  resolve();
                }
              }, worktreeTimeout);
            });
          },
        );
        const wtProc = wtTabId
          ? this._getTabProcess(wtTabId)
          : this._getServiceProcess();
        wtProc.sendCommand({
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
          const autocommitTimeout = 120_000;
          vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: 'Auto-committing…',
            },
            progress => {
              if (acTabId !== undefined) {
                this._autocommitProgresses.set(acTabId, progress);
              }
              return new Promise<void>(resolve => {
                if (acTabId !== undefined) {
                  this._autocommitActionResolves.set(acTabId, resolve);
                }
                setTimeout(() => {
                  if (
                    acTabId !== undefined &&
                    this._autocommitActionResolves.get(acTabId) === resolve
                  ) {
                    this._autocommitActionResolves.delete(acTabId);
                    resolve();
                  }
                }, autocommitTimeout);
              });
            },
          );
        }
        const acProc = acTabId
          ? this._getTabProcess(acTabId)
          : this._getServiceProcess();
        acProc.sendCommand({
          type: 'autocommitAction',
          action: acAction,
          tabId: acTabId,
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

      case 'focusEditor':
        vscode.commands.executeCommand(
          'workbench.action.focusFirstEditorGroup',
        );
        break;

      case 'closeTab': {
        const closeTabId = message.tabId;
        if (closeTabId) {
          const closeProc = this._taskProcesses.get(closeTabId);
          if (closeProc) {
            closeProc.sendCommand({type: 'closeTab', tabId: closeTabId});
          } else {
            this._getServiceProcess().sendCommand({
              type: 'closeTab',
              tabId: closeTabId,
            });
          }
        }
        break;
      }

      case 'closeSecondaryBar':
        vscode.commands.executeCommand('workbench.action.closeAuxiliaryBar');
        break;
    }
  }

  /**
   * Dispatch a merge command to the active tab's MergeManager.
   *
   * Used by extension.ts keyboard shortcuts that don't know the tab ID.
   * Routes to the MergeManager of ``_activeTabId``.
   */
  public handleMergeCommand(
    cmd:
      | 'acceptChange'
      | 'rejectChange'
      | 'prevChange'
      | 'nextChange'
      | 'acceptAll'
      | 'rejectAll'
      | 'acceptFile'
      | 'rejectFile',
  ): void {
    const mgr = this._mergeManagers.get(this._activeTabId);
    if (mgr) void mgr[cmd]();
  }

  /** Notify the agent that all merge changes have been reviewed. */
  public sendMergeAllDone(tabId?: string): void {
    const proc = tabId ? this._taskProcesses.get(tabId) : undefined;
    (proc || this._getServiceProcess()).sendCommand({
      type: 'mergeAction',
      action: 'all-done',
      tabId,
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
      // Webview not yet resolved — trigger resolution by focusing the view
      await vscode.commands.executeCommand(
        'kissSorcar.chatViewSecondary.focus',
      );
      await new Promise(r => setTimeout(r, 200));
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

  /** Ensure at least one chat tab exists; creates one only if there are none. */
  public ensureChat(): void {
    this._sendToWebview({type: 'ensureChat'});
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
    const proc = this._getServiceProcess();
    proc.start(this._getWorkDir());
    proc.sendCommand({
      type: 'generateCommitMessage',
      model: this._selectedModel,
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

  /** Cleanup: kill all agent processes and dispose listeners. */
  public dispose(): void {
    this._disposed = true;
    this._resolveAllWorktreeActions();
    for (const proc of this._taskProcesses.values()) proc.dispose();
    this._taskProcesses.clear();
    if (this._serviceProcess) {
      this._serviceProcess.dispose();
      this._serviceProcess = null;
    }
    for (const mgr of this._mergeManagers.values()) mgr.dispose();
    this._mergeManagers.clear();
    this._onCommitMessage.dispose();
  }
}
