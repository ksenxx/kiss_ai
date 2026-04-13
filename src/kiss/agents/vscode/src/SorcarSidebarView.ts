/**
 * Secondary sidebar chat view for Sorcar.
 * Provides a WebviewViewProvider that renders the same chat UI as SorcarTab
 * but embedded in the VS Code secondary sidebar instead of an editor tab.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { AgentProcess } from './AgentProcess';
import { MergeManager } from './MergeManager';
import { getDefaultModel } from './DependencyInstaller';
import { buildChatHtml } from './SorcarTab';
import { FromWebviewMessage, ToWebviewMessage, Attachment, AgentCommand } from './types';

/**
 * WebviewViewProvider for the KISS Sorcar chat in the secondary sidebar.
 *
 * Hosts the same HTML/JS/CSS chat interface as the editor-area SorcarTab,
 * but in a sidebar panel. Has its own AgentProcess for independent operation.
 */
export class SorcarSidebarView implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  private _agentProcess: AgentProcess;
  private _extensionUri: vscode.Uri;
  private _selectedModel: string;
  private _isRunning: boolean = false;
  private _pendingNewChat: boolean = false;
  private _mergeManager: MergeManager;
  private _mergeOwner: boolean = false;
  private _onCommitMessage = new vscode.EventEmitter<{ message: string; error?: string }>();
  public readonly onCommitMessage = this._onCommitMessage.event;
  private _commitPending: boolean = false;
  private _worktreeDir: string = '';
  private _worktreeActionResolve: (() => void) | null = null;
  private _worktreeProgress: vscode.Progress<{ message?: string }> | null = null;
  private _disposed: boolean = false;

  /** Callback to set this view as the merge owner in the extension. */
  public mergeOwnerCallback?: (view: SorcarSidebarView) => void;

  constructor(extensionUri: vscode.Uri, mergeManager: MergeManager) {
    this._extensionUri = extensionUri;
    this._mergeManager = mergeManager;
    this._agentProcess = new AgentProcess();
    this._selectedModel = vscode.workspace.getConfiguration('kissSorcar').get<string>('defaultModel') || getDefaultModel();
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
      webviewView.webview, this._extensionUri, this._selectedModel,
    );

    webviewView.webview.onDidReceiveMessage(
      (message: FromWebviewMessage) => this._handleMessage(message),
    );

    webviewView.onDidChangeVisibility(() => {
      if (webviewView.visible) {
        this._agentProcess.sendCommand({ type: 'getInputHistory' });
        this._sendActiveFileInfo();
      }
    });

    webviewView.onDidDispose(() => {
      this._disposed = true;
      if (this._worktreeActionResolve) {
        this._worktreeActionResolve();
        this._worktreeActionResolve = null;
      }
    });

    this._agentProcess.on('message', (msg: ToWebviewMessage) => {
      if (msg.type === 'commitMessage') {
        this._onCommitMessage.fire(msg as any);
      }
      if (msg.type === 'models' && (msg as any).selected) {
        this._selectedModel = (msg as any).selected;
      }
      if (msg.type === 'merge_data') {
        this._mergeOwner = true;
        this.mergeOwnerCallback?.(this);
        this._mergeManager.openMerge((msg as any).data);
      }
      if (msg.type === 'worktree_created' || msg.type === 'worktree_done') {
        const dir = (msg as any).worktreeDir;
        if (dir) {
          this._worktreeDir = dir;
          this._openWorktreeInScm(dir);
        }
      }
      if (msg.type === 'worktree_progress') {
        if (this._worktreeProgress) {
          this._worktreeProgress.report({ message: (msg as any).message });
        }
      }
      if (msg.type === 'worktree_result') {
        if (this._worktreeActionResolve) {
          this._worktreeActionResolve();
          this._worktreeActionResolve = null;
        }
        this._worktreeProgress = null;
        const result = msg as any;
        if (result.success) {
          vscode.window.showInformationMessage(result.message || 'Worktree action completed.');
        } else {
          vscode.window.showErrorMessage(result.message || 'Worktree action failed.');
        }
        if (result.success && this._worktreeDir) {
          this._closeWorktreeInScm(this._worktreeDir);
          this._worktreeDir = '';
        }
      }

      this._sendToWebview(msg);
      if (msg.type === 'status') {
        this._isRunning = msg.running;
        if (!msg.running) {
          this._sendActiveFileInfo();
          if (this._pendingNewChat) {
            this._pendingNewChat = false;
            this._agentProcess.sendCommand({ type: 'newChat' });
            this._sendToWebview({ type: 'clearChat' });
          }
          if (this._commitPending) {
            this._onCommitMessage.fire({ message: '', error: 'Process stopped' });
          }
        }
      }
    });

    const workDir = this._getWorkDir();
    this._agentProcess.start(workDir);
  }

  /** Whether the underlying webview is currently visible. */
  get visible(): boolean {
    return this._view?.visible ?? false;
  }

  /** Whether this view is the merge session owner. */
  get isMergeOwner(): boolean { return this._mergeOwner; }
  set isMergeOwner(v: boolean) { this._mergeOwner = v; }

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
      this._sendToWebview({ type: 'welcome_suggestions', suggestions: data } as ToWebviewMessage);
    } catch {
      this._sendToWebview({ type: 'welcome_suggestions', suggestions: [] } as ToWebviewMessage);
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
      await vscode.commands.executeCommand('git.close', vscode.Uri.file(worktreeDir));
    } catch { /* ignored */ }
  }

  private _startTask(prompt: string, model: string, activeFile?: string, attachments?: Attachment[], useWorktree?: boolean, useParallel?: boolean): void {
    const workDir = this._getWorkDir();
    const started = this._agentProcess.start(workDir);
    if (!started) {
      this._isRunning = false;
      this._sendToWebview({ type: 'status', running: false });
      return;
    }
    this._sendToWebview({ type: 'setTaskText', text: prompt } as ToWebviewMessage);
    this._sendToWebview({ type: 'status', running: true });
    this._agentProcess.sendCommand({
      type: 'run',
      prompt,
      model,
      workDir,
      activeFile,
      attachments,
      useWorktree,
      useParallel,
    });
  }

  private async _handleMessage(message: FromWebviewMessage): Promise<void> {
    switch (message.type) {
      case 'ready':
        this._sendToWebview({ type: 'status', running: this._isRunning });
        this._agentProcess.sendCommand({ type: 'getModels' });
        this._sendWelcomeSuggestions();
        this._agentProcess.sendCommand({ type: 'getInputHistory' });
        this._agentProcess.sendCommand({ type: 'getLastSession' });
        this._sendActiveFileInfo();
        this._sendToWebview({ type: 'focusInput' } as ToWebviewMessage);
        break;

      case 'submit': {
        if (this._isRunning) return;
        this._isRunning = true;

        const trimmed = message.prompt.trim();
        if (trimmed && !trimmed.includes('\n')) {
          const bare = trimmed.replace(/^WORK_DIR[/\\]/, '');
          const resolved = path.resolve(this._getWorkDir(), bare);
          if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) {
            this._isRunning = false;
            const uri = vscode.Uri.file(resolved);
            const doc = await vscode.workspace.openTextDocument(uri);
            await vscode.window.showTextDocument(doc, {
              preview: false,
              viewColumn: vscode.ViewColumn.One,
            });
            return;
          }
        }

        this._startTask(
          message.prompt,
          message.model,
          this._getVisibleEditorFile() || undefined,
          message.attachments,
          message.useWorktree,
          message.useParallel,
        );
        break;
      }

      case 'stop':
        this._agentProcess.stop();
        break;

      case 'selectModel':
        this._selectedModel = message.model;
        this._agentProcess.sendCommand({ type: 'selectModel', model: message.model });
        break;

      case 'getModels':
      case 'newChat':
      case 'getInputHistory':
        this._agentProcess.sendCommand({ type: message.type } as AgentCommand);
        break;

      case 'getHistory':
        this._agentProcess.sendCommand({ type: 'getHistory', query: message.query, offset: message.offset, generation: message.generation });
        break;

      case 'getFiles':
        this._agentProcess.sendCommand({ type: 'getFiles', prefix: message.prefix });
        break;

      case 'userAnswer':
        this._agentProcess.sendCommand({ type: 'userAnswer', answer: message.answer });
        break;

      case 'userActionDone':
        this._agentProcess.sendCommand({ type: 'userAnswer', answer: 'done' });
        break;

      case 'recordFileUsage':
        if (message.path) {
          this._agentProcess.sendCommand({ type: 'recordFileUsage', path: message.path });
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
              editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
            }
          }
        }
        break;

      case 'resumeSession':
        this._agentProcess.sendCommand({ type: 'resumeSession', sessionId: message.id });
        break;

      case 'getAdjacentTask':
        this._agentProcess.sendCommand({ type: 'getAdjacentTask', task: (message as any).task, direction: (message as any).direction });
        break;

      case 'getWelcomeSuggestions':
        this._sendWelcomeSuggestions();
        break;

      case 'complete': {
        const editorFile = this._getVisibleEditorFile();
        const completeDoc = editorFile
          ? vscode.workspace.textDocuments.find(d => d.uri.fsPath === editorFile)
          : undefined;
        this._agentProcess.sendCommand({
          type: 'complete',
          query: message.query,
          activeFile: editorFile || undefined,
          activeFileContent: completeDoc?.getText(),
        });
        break;
      }

      case 'mergeAction': {
        const mergeDispatch: Record<string, () => void> = {
          'accept': () => this._mergeManager.acceptChange(),
          'reject': () => this._mergeManager.rejectChange(),
          'prev': () => this._mergeManager.prevChange(),
          'next': () => this._mergeManager.nextChange(),
          'accept-all': () => this._mergeManager.acceptAll(),
          'reject-all': () => this._mergeManager.rejectAll(),
          'accept-file': () => this._mergeManager.acceptFile(),
          'reject-file': () => this._mergeManager.rejectFile(),
        };
        const mAction = (message as any).action;
        const handler = mergeDispatch[mAction];
        if (handler) handler();
        else if (mAction === 'all-done') {
          this._agentProcess.sendCommand({ type: 'mergeAction', action: 'all-done' });
        }
        break;
      }

      case 'generateCommitMessage':
        this.generateCommitMessage();
        break;

      case 'runPrompt': {
        if (this._isRunning) return;
        const promptPath = this._getVisibleEditorFile();
        if (!promptPath || !promptPath.toLowerCase().endsWith('.md')) return;
        const promptDoc = vscode.workspace.textDocuments.find(d => d.uri.fsPath === promptPath);
        if (!promptDoc) return;
        const content = promptDoc.getText();
        if (!content.trim()) return;
        this._isRunning = true;
        this._startTask(content, this._selectedModel, promptPath);
        break;
      }

      case 'worktreeAction': {
        const wtAction = (message as any).action;
        const progressTitle = wtAction === 'merge'
          ? 'Committing and merging worktree…'
          : wtAction === 'discard'
            ? 'Discarding worktree…'
            : wtAction === 'do_nothing'
              ? 'Finishing up…'
              : 'Processing worktree action…';
        const worktreeTimeout = 120_000;
        vscode.window.withProgress(
          { location: vscode.ProgressLocation.Notification, title: progressTitle },
          (progress) => {
            this._worktreeProgress = progress;
            return new Promise<void>((resolve) => {
              this._worktreeActionResolve = resolve;
              setTimeout(() => {
                if (this._worktreeActionResolve === resolve) {
                  this._worktreeActionResolve = null;
                  resolve();
                }
              }, worktreeTimeout);
            });
          },
        );
        this._agentProcess.sendCommand({
          type: 'worktreeAction',
          action: wtAction,
        });
        break;
      }

      case 'resolveDroppedPaths': {
        const workDir = this._getWorkDir();
        const paths = (message.uris || []).map((uri: string) => {
          try {
            const absPath = vscode.Uri.parse(uri).fsPath;
            return path.relative(workDir, absPath);
          } catch {
            return '';
          }
        }).filter((p: string) => p && !p.startsWith('..'));
        this._sendToWebview({ type: 'droppedPaths', paths } as ToWebviewMessage);
        break;
      }

      case 'focusEditor':
        vscode.commands.executeCommand('workbench.action.focusFirstEditorGroup');
        break;
    }
  }

  /** Notify the agent that all merge changes have been reviewed. */
  public sendMergeAllDone(): void {
    this._agentProcess.sendCommand({ type: 'mergeAction', action: 'all-done' });
  }

  /** Submit a task programmatically (e.g. from runSelection command). */
  public submitTask(prompt: string): void {
    if (this._isRunning || !prompt.trim()) return;
    this._isRunning = true;
    this._startTask(
      prompt.trim(),
      this._selectedModel,
      this._getVisibleEditorFile() || undefined,
    );
  }

  /** Stop the currently running task. */
  public stopTask(): void {
    this._agentProcess.stop();
  }

  /** Focus the chat input in the sidebar. */
  public async focusChatInput(): Promise<void> {
    if (this._view) {
      this._view.show(true);
      await new Promise(r => setTimeout(r, 150));
      this._sendToWebview({ type: 'focusInput' });
    }
  }

  /** Start a new conversation, stopping any running task first. */
  public newConversation(): void {
    if (this._isRunning) {
      this._pendingNewChat = true;
      this._agentProcess.stop();
    } else {
      this._agentProcess.sendCommand({ type: 'newChat' });
      this._sendToWebview({ type: 'clearChat' });
    }
  }

  /**
   * Generate a commit message using this view's agent process.
   */
  public generateCommitMessage(token?: vscode.CancellationToken): Promise<void> {
    if (this._commitPending) return Promise.resolve();
    this._commitPending = true;
    this._agentProcess.start(this._getWorkDir());
    this._agentProcess.sendCommand({ type: 'generateCommitMessage', model: this._selectedModel });

    return new Promise<void>((resolve) => {
      let resolved = false;
      const done = () => {
        if (resolved) return;
        resolved = true;
        this._commitPending = false;
        disposable.dispose();
        clearTimeout(timer);
        resolve();
      };
      const disposable = this._onCommitMessage.event(() => done());
      token?.onCancellationRequested(() => done());
      const timer = setTimeout(done, 30_000);
    });
  }

  /** Cleanup: kill agent process and dispose listeners. */
  public dispose(): void {
    this._disposed = true;
    if (this._worktreeActionResolve) {
      this._worktreeActionResolve();
      this._worktreeActionResolve = null;
    }
    this._agentProcess.dispose();
    this._onCommitMessage.dispose();
  }
}
