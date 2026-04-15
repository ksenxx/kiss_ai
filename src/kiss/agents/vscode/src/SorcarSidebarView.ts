/**
 * Sidebar chat view for Sorcar.
 * Provides a WebviewViewProvider that renders the chat UI in the
 * VS Code secondary sidebar.
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
 * Hosts the chat HTML/JS/CSS interface with its own AgentProcess.
 */
export class SorcarSidebarView implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  private _agentProcess: AgentProcess;
  private _extensionUri: vscode.Uri;
  private _selectedModel: string;
  private _runningTabs: Set<string> = new Set();

  private _mergeManager: MergeManager;
  private _mergeOwnerTabIdQueue: string[] = [];
  private _onCommitMessage = new vscode.EventEmitter<{ message: string; error?: string }>();
  public readonly onCommitMessage = this._onCommitMessage.event;
  private _commitPendingTabs: Set<string> = new Set();
  private _worktreeDirs: Map<string, string> = new Map();
  private _worktreeActionResolves: Map<string, () => void> = new Map();
  private _worktreeProgresses: Map<string, vscode.Progress<{ message?: string }>> = new Map();
  private _disposed: boolean = false;

  constructor(extensionUri: vscode.Uri, mergeManager: MergeManager) {
    this._extensionUri = extensionUri;
    this._mergeManager = mergeManager;
    this._agentProcess = new AgentProcess();
    this._selectedModel = vscode.workspace.getConfiguration('kissSorcar').get<string>('defaultModel') || getDefaultModel();
    this._mergeManager.on('allDone', () => {
      const tabId = this._mergeOwnerTabIdQueue.shift();
      if (tabId !== undefined) {
        this.sendMergeAllDone(tabId);
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
      for (const resolve of this._worktreeActionResolves.values()) resolve();
      this._worktreeActionResolves.clear();
      this._worktreeProgresses.clear();
    });

    this._agentProcess.on('message', (msg: ToWebviewMessage) => {
      if (msg.type === 'commitMessage') {
        this._onCommitMessage.fire(msg as any);
      }
      if (msg.type === 'models' && (msg as any).selected) {
        this._selectedModel = (msg as any).selected;
      }
      if (msg.type === 'merge_data') {
        const mergeTabId = (msg as any).tabId as string | undefined;
        if (mergeTabId !== undefined) {
          this._mergeOwnerTabIdQueue.push(mergeTabId);
        }
        this._mergeManager.openMerge((msg as any).data);
      }
      if (msg.type === 'worktree_created' || msg.type === 'worktree_done') {
        const dir = (msg as any).worktreeDir;
        const wtTabId = (msg as any).tabId as string | undefined;
        if (dir) {
          if (wtTabId !== undefined) {
            this._worktreeDirs.set(wtTabId, dir);
          }
          this._openWorktreeInScm(dir);
        }
      }
      if (msg.type === 'worktree_progress') {
        const wpTabId = (msg as any).tabId as string | undefined;
        const progress = wpTabId !== undefined
          ? this._worktreeProgresses.get(wpTabId)
          : this._worktreeProgresses.values().next().value;
        if (progress) {
          progress.report({ message: (msg as any).message });
        }
      }
      if (msg.type === 'worktree_result') {
        const wrTabId = (msg as any).tabId as string | undefined;
        if (wrTabId !== undefined) {
          const resolve = this._worktreeActionResolves.get(wrTabId);
          if (resolve) {
            resolve();
            this._worktreeActionResolves.delete(wrTabId);
          }
          this._worktreeProgresses.delete(wrTabId);
        } else {
          // Fallback: resolve all pending
          for (const resolve of this._worktreeActionResolves.values()) resolve();
          this._worktreeActionResolves.clear();
          this._worktreeProgresses.clear();
        }
        const result = msg as any;
        if (result.success) {
          vscode.window.showInformationMessage(result.message || 'Worktree action completed.');
        } else {
          vscode.window.showErrorMessage(result.message || 'Worktree action failed.');
        }
        if (result.success && wrTabId !== undefined) {
          const wtDir = this._worktreeDirs.get(wrTabId);
          if (wtDir) {
            this._closeWorktreeInScm(wtDir);
            this._worktreeDirs.delete(wrTabId);
          }
        }
      }

      this._sendToWebview(msg);
      if (msg.type === 'status') {
        const tabId = (msg as any).tabId as string | undefined;
        if (msg.running) {
          if (tabId !== undefined) this._runningTabs.add(tabId);
        } else {
          if (tabId !== undefined) this._runningTabs.delete(tabId);
          else this._runningTabs.clear();
          this._sendActiveFileInfo();
          if (this._commitPendingTabs.size > 0) {
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

  private _startTask(prompt: string, model: string, activeFile?: string, attachments?: Attachment[], useWorktree?: boolean, useParallel?: boolean, tabId?: string, workDir?: string): void {
    const effectiveWorkDir = workDir || this._getWorkDir();
    const started = this._agentProcess.start(effectiveWorkDir);
    if (!started) {
      if (tabId !== undefined) this._runningTabs.delete(tabId);
      this._sendToWebview({ type: 'status', running: false, tabId } as any);
      return;
    }
    this._sendToWebview({ type: 'setTaskText', text: prompt, tabId } as any);
    this._sendToWebview({ type: 'status', running: true, tabId } as any);
    this._agentProcess.sendCommand({
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
      case 'ready':
        // Send running state for each tab that has a running task
        for (const tabId of this._runningTabs) {
          this._sendToWebview({ type: 'status', running: true, tabId } as any);
        }
        this._agentProcess.sendCommand({ type: 'getModels' });
        this._sendWelcomeSuggestions();
        this._agentProcess.sendCommand({ type: 'getInputHistory' });
        this._agentProcess.sendCommand({ type: 'getLastSession', tabId: (message as any).tabId });
        this._sendActiveFileInfo();
        this._sendToWebview({ type: 'focusInput' } as ToWebviewMessage);
        break;

      case 'submit': {
        const tabId = (message as any).tabId as string | undefined;
        if (tabId !== undefined && this._runningTabs.has(tabId)) return;

        const tabWorkDir = (message as any).workDir as string | undefined;
        const effectiveWorkDir = tabWorkDir || this._getWorkDir();

        const trimmed = message.prompt.trim();
        if (trimmed && !trimmed.includes('\n')) {
          const bare = trimmed.replace(/^WORK_DIR[/\\]/, '');
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
        const stopTabId = (message as any).tabId as string | undefined;
        if (stopTabId !== undefined) {
          this._agentProcess.sendCommand({ type: 'stop', tabId: stopTabId });
        } else {
          this._agentProcess.stop();
        }
        break;
      }

      case 'selectModel':
        this._selectedModel = message.model;
        this._agentProcess.sendCommand({ type: 'selectModel', model: message.model, tabId: (message as any).tabId });
        break;

      case 'getModels':
      case 'getInputHistory':
        this._agentProcess.sendCommand({ type: message.type } as AgentCommand);
        break;

      case 'newChat':
        this._agentProcess.sendCommand({ type: 'newChat', tabId: (message as any).tabId });
        break;

      case 'getHistory':
        this._agentProcess.sendCommand({ type: 'getHistory', query: message.query, offset: message.offset, generation: message.generation });
        break;

      case 'getFiles':
        this._agentProcess.sendCommand({ type: 'getFiles', prefix: message.prefix });
        break;

      case 'userAnswer':
        this._agentProcess.sendCommand({ type: 'userAnswer', answer: message.answer, tabId: message.tabId });
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
        this._agentProcess.sendCommand({ type: 'resumeSession', sessionId: message.id, tabId: (message as any).tabId });
        break;

      case 'getAdjacentTask':
        this._agentProcess.sendCommand({ type: 'getAdjacentTask', tabId: (message as any).tabId, task: (message as any).task, direction: (message as any).direction });
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
        // runPrompt doesn't have a tabId from the webview; allow if any tab is free
        if (this._runningTabs.size > 0) return;
        const promptPath = this._getVisibleEditorFile();
        if (!promptPath || !promptPath.toLowerCase().endsWith('.md')) return;
        const promptDoc = vscode.workspace.textDocuments.find(d => d.uri.fsPath === promptPath);
        if (!promptDoc) return;
        const content = promptDoc.getText();
        if (!content.trim()) return;
        this._startTask(content, this._selectedModel, promptPath);
        break;
      }

      case 'worktreeAction': {
        const wtAction = (message as any).action;
        const wtTabId = (message as any).tabId as string | undefined;
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
            if (wtTabId !== undefined) {
              this._worktreeProgresses.set(wtTabId, progress);
            }
            return new Promise<void>((resolve) => {
              if (wtTabId !== undefined) {
                this._worktreeActionResolves.set(wtTabId, resolve);
              }
              setTimeout(() => {
                if (wtTabId !== undefined && this._worktreeActionResolves.get(wtTabId) === resolve) {
                  this._worktreeActionResolves.delete(wtTabId);
                  resolve();
                }
              }, worktreeTimeout);
            });
          },
        );
        this._agentProcess.sendCommand({
          type: 'worktreeAction',
          action: wtAction,
          tabId: wtTabId,
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
  public sendMergeAllDone(tabId?: string): void {
    this._agentProcess.sendCommand({ type: 'mergeAction', action: 'all-done', tabId });
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
    this._sendToWebview({ type: 'triggerStop' } as ToWebviewMessage);
  }

  /** Focus the chat input in the sidebar. */
  public async focusChatInput(): Promise<void> {
    if (!this._view) {
      // Webview not yet resolved — trigger resolution by focusing the view
      await vscode.commands.executeCommand('kissSorcar.chatViewSecondary.focus');
      await new Promise(r => setTimeout(r, 200));
    }
    if (this._view) {
      this._view.show(true);
      await new Promise(r => setTimeout(r, 150));
      this._sendToWebview({ type: 'focusInput' });
    }
  }

  /** Append text to the chat input and focus it. */
  public async appendToInput(text: string): Promise<void> {
    if (this._view) {
      this._view.show(true);
      await new Promise(r => setTimeout(r, 150));
      this._sendToWebview({ type: 'appendToInput', text });
    }
  }

  /** Start a new conversation in a new tab (without affecting running tabs). */
  public newConversation(): void {
    this._sendToWebview({ type: 'clearChat' });
  }

  /**
   * Generate a commit message using this view's agent process.
   *
   * @param token Optional cancellation token.
   * @param tabId Optional tab ID — each tab can independently request a
   *              commit message without blocking other tabs.
   */
  public generateCommitMessage(token?: vscode.CancellationToken, tabId: string = ''): Promise<void> {
    if (this._commitPendingTabs.has(tabId)) return Promise.resolve();
    this._commitPendingTabs.add(tabId);
    this._agentProcess.start(this._getWorkDir());
    this._agentProcess.sendCommand({ type: 'generateCommitMessage', model: this._selectedModel });

    return new Promise<void>((resolve) => {
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

  /** Cleanup: kill agent process and dispose listeners. */
  public dispose(): void {
    this._disposed = true;
    for (const resolve of this._worktreeActionResolves.values()) resolve();
    this._worktreeActionResolves.clear();
    this._worktreeProgresses.clear();
    this._agentProcess.dispose();
    this._onCommitMessage.dispose();
  }
}
