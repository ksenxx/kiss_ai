/**
 * Editor-tab-based chat windows for Sorcar.
 * Each SorcarTab wraps a WebviewPanel + AgentProcess in the editor area.
 * TabManager tracks open tabs and routes commands to the active one.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { AgentProcess, findKissProject } from './AgentProcess';
import { getDefaultModel } from './DependencyInstaller';
import { FromWebviewMessage, ToWebviewMessage, Attachment, AgentCommand } from './types';

/**
 * Return the ViewColumn for the rightmost editor split.
 * If the editor is not yet split, returns ``ViewColumn.Two`` which
 * creates a vertical split automatically.
 */
function rightmostColumn(): vscode.ViewColumn {
  const groups = vscode.window.tabGroups.all;
  if (groups.length <= 1) return vscode.ViewColumn.Two;
  let max = vscode.ViewColumn.One;
  for (const g of groups) {
    if (g.viewColumn > max) max = g.viewColumn;
  }
  return max;
}

/** Read the KISS project version from ``_version.py`` on disk. */
function getVersion(): string {
  try {
    const kissRoot = findKissProject();
    if (kissRoot) {
      const versionFile = path.join(kissRoot, 'src', 'kiss', '_version.py');
      const content = fs.readFileSync(versionFile, 'utf-8');
      const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
      if (match) return match[1];
    }
  } catch { /* ignore */ }
  return '';
}

/**
 * A single chat tab in the editor area.
 * Wraps a WebviewPanel and its own AgentProcess (Python subprocess).
 */
export class SorcarTab {
  private _panel: vscode.WebviewPanel;
  private _agentProcess: AgentProcess;
  private _extensionUri: vscode.Uri;
  private _selectedModel: string;
  private _isRunning: boolean = false;
  private _onCommitMessage = new vscode.EventEmitter<{ message: string; error?: string }>();
  public readonly onCommitMessage = this._onCommitMessage.event;
  private _commitPending: boolean = false;
  private _pendingNewChat: boolean = false;
  private _disposed: boolean = false;
  private _loadLastSession: boolean;
  private _lastTask: string = '';

  /** The underlying WebviewPanel (for reveal/focus tracking). */
  get panel(): vscode.WebviewPanel { return this._panel; }

  /**
   * @param extensionUri - Extension root URI for resolving media assets.
   * @param loadLastSession - If true, restore the last chat session on ready.
   *   If false, start a fresh conversation with welcome suggestions.
   * @param _onDispose - Callback invoked when the tab is disposed.
   */
  constructor(extensionUri: vscode.Uri, loadLastSession: boolean, private _onDispose: (tab: SorcarTab) => void) {
    this._extensionUri = extensionUri;
    this._loadLastSession = loadLastSession;
    this._agentProcess = new AgentProcess();
    this._selectedModel = vscode.workspace.getConfiguration('kissSorcar').get<string>('defaultModel') || getDefaultModel();

    // Create editor-area WebviewPanel
    this._panel = vscode.window.createWebviewPanel(
      'kissSorcar.chat',
      'KS: new chat',
      rightmostColumn(),
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(this._extensionUri, 'media'),
          vscode.Uri.joinPath(this._extensionUri, 'out'),
        ],
      }
    );

    this._panel.iconPath = vscode.Uri.joinPath(this._extensionUri, 'media', 'kiss-icon.svg');
    this._panel.webview.html = this._getHtmlContent(this._panel.webview);

    // Handle messages from webview
    this._panel.webview.onDidReceiveMessage(
      (message: FromWebviewMessage) => this._handleMessage(message),
      undefined,
      []
    );

    // Handle panel disposal
    this._panel.onDidDispose(() => {
      this.dispose();
      this._onDispose(this);
    });

    // Listen for agent events
    this._agentProcess.on('message', (msg: ToWebviewMessage) => {
      if (msg.type === 'commitMessage') {
        this._onCommitMessage.fire(msg as any);
      }
      if (msg.type === 'models' && (msg as any).selected) {
        this._selectedModel = (msg as any).selected;
      }
      if (msg.type === 'task_events' && (msg as any).task) {
        this._updateTabTitle((msg as any).task);
      }
      if (msg.type === 'worktree_done') {
        this._handleWorktreeDone(msg as any);
      }
      this.sendToWebview(msg);
      if (msg.type === 'status') {
        this._isRunning = msg.running;
        if (!msg.running) {
          this._sendActiveFileInfo();
          if (this._pendingNewChat) {
            this._pendingNewChat = false;
            this._agentProcess.sendCommand({ type: 'newChat' });
            this.sendToWebview({ type: 'clearChat' });
            this._updateTabTitle('');
          }
          if (this._commitPending) {
            this._onCommitMessage.fire({ message: '', error: 'Process stopped' });
          }
        }
      }
    });

    // Start agent process
    const workDir = this._getWorkDir();
    this._agentProcess.start(workDir);
  }

  private _getWorkDir(): string {
    const folders = vscode.workspace.workspaceFolders;
    if (folders && folders.length > 0) {
      return folders[0].uri.fsPath;
    }
    return process.cwd();
  }

  private _sendWelcomeSuggestions(): void {
    const jsonPath = path.join(this._extensionUri.fsPath, 'SAMPLE_TASKS.json');
    try {
      const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
      this.sendToWebview({ type: 'welcome_suggestions', suggestions: data } as ToWebviewMessage);
    } catch {
      this.sendToWebview({ type: 'welcome_suggestions', suggestions: [] } as ToWebviewMessage);
    }
  }

  /**
   * Dynamically find the file path of the visible text editor tab.
   * Checks the active text editor first; if unavailable (e.g. when the
   * Sorcar webview panel has focus), scans all tab groups for an active
   * tab whose input is a text file.
   */
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
    this.sendToWebview({
      type: 'activeFileInfo',
      isPrompt,
      filename: isPrompt ? path.basename(fpath) : '',
      path: fpath,
    } as ToWebviewMessage);
  }

  /**
   * Handle worktree task completion: open diff editor and show quick pick
   * for merge/discard/manual actions.
   */
  private async _handleWorktreeDone(msg: {
    branch: string;
    worktreeDir: string;
    originalBranch: string;
    changedFiles: string[];
  }): Promise<void> {
    // Open diff editor for the first changed file
    if (msg.changedFiles.length > 0) {
      const file = msg.changedFiles[0];
      const origUri = vscode.Uri.file(path.join(this._getWorkDir(), file));
      const wtUri = vscode.Uri.file(path.join(msg.worktreeDir, file));
      await vscode.commands.executeCommand('vscode.diff',
        origUri, wtUri,
        `${file} (original ↔ agent changes)`);
    }

    // Show quick pick for action
    const choice = await vscode.window.showQuickPick([
      { label: '$(check) Merge', description: 'Auto-merge branch into ' + msg.originalBranch, action: 'merge' as const },
      { label: '$(git-merge) Review & Merge Manually', description: 'Open instructions for manual merge', action: 'manual' as const },
      { label: '$(trash) Discard', description: 'Delete the branch and worktree', action: 'discard' as const },
    ], {
      placeHolder: `Task completed on branch ${msg.branch}. What would you like to do?`,
      ignoreFocusOut: true,
    });

    if (choice) {
      this._agentProcess.sendCommand({
        type: 'worktreeAction',
        action: choice.action,
      });
    }
    // If dismissed (Escape), branch stays pending — next task in same chat
    // will be blocked with merge instructions (existing behavior).
  }

  /**
   * Update the tab title to reflect the current task.
   * The title is truncated to keep the tab compact while the tooltip
   * (set via ``WebviewPanel.description``) retains the full text.
   */
  private _updateTabTitle(task: string): void {
    this._lastTask = task;
    if (!task.trim()) {
      this._panel.title = 'KS: new chat';
      return;
    }
    const firstLine = task.split('\n')[0].trim();
    const maxLen = 30;
    const truncated = firstLine.length > maxLen
      ? firstLine.slice(0, maxLen) + '…'
      : firstLine;
    this._panel.title = 'KS: ' + truncated;
  }

  private _startTask(prompt: string, model: string, activeFile?: string, attachments?: Attachment[]): void {
    const workDir = this._getWorkDir();
    const started = this._agentProcess.start(workDir);
    if (!started) {
      this._isRunning = false;
      this.sendToWebview({ type: 'status', running: false });
      return;
    }
    this._updateTabTitle(prompt);
    this.sendToWebview({ type: 'setTaskText', text: prompt } as ToWebviewMessage);
    this.sendToWebview({ type: 'status', running: true });
    this._agentProcess.sendCommand({
      type: 'run',
      prompt,
      model,
      workDir,
      activeFile,
      attachments,
    });
  }

  private async _handleMessage(message: FromWebviewMessage): Promise<void> {
    switch (message.type) {
      case 'ready':
        this.sendToWebview({ type: 'status', running: this._isRunning });
        this._agentProcess.sendCommand({ type: 'getModels' });
        this._sendWelcomeSuggestions();
        this._agentProcess.sendCommand({ type: 'getInputHistory' });
        if (this._loadLastSession) {
          this._agentProcess.sendCommand({ type: 'getLastSession' });
        }
        this._sendActiveFileInfo();
        this.sendToWebview({ type: 'focusInput' } as ToWebviewMessage);
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
            await vscode.window.showTextDocument(doc);
            return;
          }
        }

        this._startTask(
          message.prompt,
          message.model,
          this._getVisibleEditorFile() || undefined,
          message.attachments,
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
            const editor = await vscode.window.showTextDocument(doc);
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

      case 'focusEditor':
        vscode.commands.executeCommand('workbench.action.focusFirstEditorGroup');
        break;
    }
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

  /** Reveal and focus the chat input. */
  public async focusChatInput(): Promise<void> {
    this._panel.reveal();
    await new Promise(r => setTimeout(r, 150));
    this.sendToWebview({ type: 'focusInput' });
  }

  /** Start a new conversation, stopping any running task first. */
  public newConversation(): void {
    if (this._isRunning) {
      this._pendingNewChat = true;
      this._agentProcess.stop();
    } else {
      this._agentProcess.sendCommand({ type: 'newChat' });
      this.sendToWebview({ type: 'clearChat' });
      this._updateTabTitle('');
    }
  }

  /** Send a message to the webview. */
  public sendToWebview(message: ToWebviewMessage): void {
    if (!this._disposed) {
      this._panel.webview.postMessage(message);
    }
  }

  /**
   * Generate a commit message. Returns a Promise that resolves when the message
   * is received (or an error occurs). Accepts a CancellationToken so the VS Code
   * SCM sparkle button can show a stop/cancel button while generation is in progress.
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
    if (this._disposed) return;
    this._disposed = true;
    this._agentProcess.dispose();
    this._onCommitMessage.dispose();
  }

  private _getHtmlContent(webview: vscode.Webview): string {
    const nonce = this._getNonce();

    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'main.css')
    );
    const hljsCssUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'highlight-github-dark.min.css')
    );
    const hljsUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'highlight.min.js')
    );
    const markedUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'marked.min.js')
    );
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js')
    );

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} data: https:; font-src ${webview.cspSource};">
  <link href="${styleUri}" rel="stylesheet">
  <link href="${hljsCssUri}" rel="stylesheet">
  <title>KISS Sorcar</title>
</head>
<body>
  <div id="app">
    <header>
      <div class="header-left">
        <span class="logo">\u2731 KISS Sorcar <span class="version">${getVersion()}</span></span>
      </div>
      <div class="status">
        <span class="dot" id="status-dot"></span>
        <span id="status-text">Ready</span>
      </div>
    </header>

    <div id="task-panel"></div>

    <div id="output">
      <div id="welcome">
        <h2>Welcome to KISS Sorcar</h2>
        <p>Your AI assistant. Ask me anything!</p>
        <div id="suggestions"></div>
      </div>
    </div>

    <div id="input-area">
      <div id="autocomplete"></div>
      <div id="input-container">
        <div id="file-chips"></div>
        <div id="input-wrap">
          <div id="input-text-wrap">
            <div id="ghost-overlay"></div>
            <textarea id="task-input" placeholder="Ask anything... (@ for files, ${process.platform === 'darwin' ? '⌘' : 'Ctrl+'}D toggle between editor and chat, ${process.platform === 'darwin' ? '⌘' : 'Ctrl+'}T new chat, ${process.platform === 'darwin' ? '⌘' : 'Ctrl+'}L run selected text in editor as task)" rows="1"></textarea>
            <button id="input-clear-btn" style="display:none;">&times;</button>
          </div>
        </div>
        <div id="input-footer">
          <div id="model-picker">
            <button id="model-btn" data-tooltip="Select model">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z"/></svg>
              <span id="model-name">${this._selectedModel}</span>
            </button>
            <button id="upload-btn" data-tooltip="Attach files">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
            </button>
            <button id="history-btn" data-tooltip="Task history">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
              </svg>
            </button>
            <button id="clear-btn" data-tooltip="New chat">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </button>
            <button id="run-prompt-btn" data-tooltip="Run current file as prompt" disabled>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                <polygon points="5,3 19,12 5,21"/>
              </svg>
            </button>
            <div id="model-dropdown">
              <input type="text" id="model-search" placeholder="Search models...">
              <div id="model-list"></div>
            </div>
          </div>
          <div id="input-actions">
            <span id="wait-spinner"></span>
            <button id="send-btn" data-tooltip="Send message">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
            </button>
            <button id="stop-btn" data-tooltip="Stop agent" style="display:none;">
              <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <div id="sidebar">
      <button id="sidebar-close" data-tooltip="Close sidebar">&times;</button>
      <div class="sidebar-section">
        <div class="sidebar-hdr">Recent Conversations</div>
        <input type="text" id="history-search" placeholder="Search history...">
        <div id="history-list">
          <div class="sidebar-empty">No conversations yet</div>
        </div>
      </div>
    </div>
    <div id="sidebar-overlay"></div>

    <div id="ask-user-modal" style="display:none;">
      <div class="modal-content">
        <div class="modal-title">Agent needs your input</div>
        <div id="ask-user-question"></div>
        <textarea id="ask-user-input" placeholder="Your answer..."></textarea>
        <button id="ask-user-submit" data-tooltip="Submit answer">Submit</button>
      </div>
    </div>
  </div>

  <script nonce="${nonce}" src="${hljsUri}"></script>
  <script nonce="${nonce}" src="${markedUri}"></script>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }

  private _getNonce(): string {
    let text = '';
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
      text += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return text;
  }
}

/**
 * Manages multiple chat tabs. Tracks the active tab and routes commands.
 */
export class TabManager {
  private _tabs: SorcarTab[] = [];
  private _activeTab: SorcarTab | undefined;
  private _extensionUri: vscode.Uri;
  private _onCommitMessage = new vscode.EventEmitter<{ message: string; error?: string }>();
  /** Aggregated commit message events from all tabs. */
  public readonly onCommitMessage = this._onCommitMessage.event;

  constructor(extensionUri: vscode.Uri) {
    this._extensionUri = extensionUri;
  }

  /**
   * Create a new chat tab, set it as active, and return it.
   * @param loadLastSession - If true, restore the last session on ready.
   *   Pass true for the auto-open tab at activation, false for Cmd+T new chats.
   */
  createTab(loadLastSession: boolean = false): SorcarTab {
    const tab = new SorcarTab(this._extensionUri, loadLastSession, (disposed) => {
      this._tabs = this._tabs.filter(t => t !== disposed);
      if (this._activeTab === disposed) {
        this._activeTab = this._tabs.length > 0 ? this._tabs[this._tabs.length - 1] : undefined;
      }
    });

    this._tabs.push(tab);
    this._activeTab = tab;

    // Track active tab via panel focus
    tab.panel.onDidChangeViewState(() => {
      if (tab.panel.active) {
        this._activeTab = tab;
      }
    });

    // Forward commit messages
    tab.onCommitMessage((ev) => this._onCommitMessage.fire(ev));

    return tab;
  }

  /** Get the currently active (last focused) tab, or undefined. */
  getActiveTab(): SorcarTab | undefined {
    return this._activeTab;
  }

  /** Dispose all tabs and clean up. */
  dispose(): void {
    for (const tab of [...this._tabs]) {
      tab.panel.dispose();
    }
    this._tabs = [];
    this._activeTab = undefined;
    this._onCommitMessage.dispose();
  }
}
