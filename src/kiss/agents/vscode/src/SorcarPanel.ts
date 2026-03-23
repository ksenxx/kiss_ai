/**
 * Webview panel manager for Sorcar chat interface.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { AgentProcess } from './AgentProcess';
import { MergeManager } from './MergeManager';
import { FromWebviewMessage, ToWebviewMessage, Attachment, AgentCommand } from './types';

export class SorcarViewProvider implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  private _agentProcess: AgentProcess;
  private _extensionUri: vscode.Uri;
  private _selectedModel: string;
  private _isRunning: boolean = false;
  private _mergeManager: MergeManager;
  private _onCommitMessage = new vscode.EventEmitter<{ message: string; error?: string }>();
  public readonly onCommitMessage = this._onCommitMessage.event;
  private _activeEditorDisposable?: vscode.Disposable;

  constructor(extensionUri: vscode.Uri, mergeManager?: MergeManager) {
    this._extensionUri = extensionUri;
    this._agentProcess = new AgentProcess();
    this._mergeManager = mergeManager || new MergeManager();
    this._selectedModel = vscode.workspace.getConfiguration('kissSorcar').get<string>('defaultModel') || 'claude-opus-4-6';

    this._mergeManager.on('allDone', () => {
      this._agentProcess.sendCommand({ type: 'mergeAction', action: 'all-done' });
      this.sendToWebview({ type: 'merge_ended' } as ToWebviewMessage);
    });
    this._mergeManager.on('hunkProcessed', () => {
      this._agentProcess.sendCommand({ type: 'mergeAction', action: 'accept' });
    });

    // Track active editor changes to update run-prompt button
    this._activeEditorDisposable = vscode.window.onDidChangeActiveTextEditor(() => {
      this._sendActiveFileInfo();
    });

    // Listen for agent events
    this._agentProcess.on('message', (msg: ToWebviewMessage) => {
      if (msg.type === 'merge_data') {
        this._mergeManager.openMerge((msg as any).data).catch((err) => {
          console.error('[SorcarPanel] merge open failed:', err);
        });
      }
      if (msg.type === 'commitMessage') {
        this._onCommitMessage.fire(msg as any);
      }
      this.sendToWebview(msg);
      if (msg.type === 'status') {
        this._isRunning = msg.running;
        if (!msg.running) {
          this._sendActiveFileInfo();
        }
      }
    });
  }

  get mergeManager(): MergeManager {
    return this._mergeManager;
  }

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(this._extensionUri, 'media'),
        vscode.Uri.joinPath(this._extensionUri, 'out'),
      ],
    };

    webviewView.webview.html = this._getHtmlContent(webviewView.webview);

    // Handle messages from webview
    webviewView.webview.onDidReceiveMessage(
      (message: FromWebviewMessage) => this._handleMessage(message),
      undefined,
      []
    );

    // Start the agent process
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

  private _sendActiveFileInfo(): void {
    const editor = vscode.window.activeTextEditor;
    const fpath = editor?.document.uri.fsPath || '';
    const isPrompt = !!fpath && fpath.toLowerCase().endsWith('.md');
    this.sendToWebview({
      type: 'activeFileInfo',
      isPrompt,
      filename: isPrompt ? fpath.split('/').pop() || '' : '',
      path: fpath,
    } as ToWebviewMessage);
  }

  private async _handleMessage(message: FromWebviewMessage): Promise<void> {
    switch (message.type) {
      case 'ready':
        this.sendToWebview({ type: 'status', running: this._isRunning });
        this._agentProcess.sendCommand({ type: 'getModels' });
        this._agentProcess.sendCommand({ type: 'getWelcomeSuggestions' });
        this._sendActiveFileInfo();
        break;

      case 'submit': {
        if (this._isRunning) return;

        // If the prompt is just a file path that exists, open it in the editor
        const trimmed = message.prompt.trim();
        if (trimmed && !trimmed.includes('\n')) {
          const resolved = trimmed.startsWith('~')
            ? path.join(process.env.HOME || '', trimmed.slice(1))
            : path.isAbsolute(trimmed)
              ? trimmed
              : path.join(this._getWorkDir(), trimmed);
          if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) {
            const uri = vscode.Uri.file(resolved);
            const doc = await vscode.workspace.openTextDocument(uri);
            await vscode.window.showTextDocument(doc);
            return;
          }
        }

        this._isRunning = true;
        this.sendToWebview({ type: 'status', running: true });

        const activeFile = vscode.window.activeTextEditor?.document.uri.fsPath;

        const cmd: AgentCommand = {
          type: 'run',
          prompt: message.prompt,
          model: message.model,
          workDir: this._getWorkDir(),
          activeFile: activeFile,
          attachments: message.attachments,
        };
        this._agentProcess.sendCommand(cmd);
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
        this._agentProcess.sendCommand({ type: 'getModels' });
        break;

      case 'getHistory':
        this._agentProcess.sendCommand({ type: 'getHistory', query: message.query });
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
          const uri = vscode.Uri.file(message.path);
          const doc = await vscode.workspace.openTextDocument(uri);
          const editor = await vscode.window.showTextDocument(doc);
          if (message.line !== undefined && message.line > 0) {
            const pos = new vscode.Position(message.line - 1, 0);
            editor.selection = new vscode.Selection(pos, pos);
            editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
          }
        }
        break;

      case 'clearChat':
        this.sendToWebview({ type: 'clearChat' });
        break;

      case 'newChat':
        this._agentProcess.sendCommand({ type: 'newChat' });
        break;

      case 'resumeSession':
        this._agentProcess.sendCommand({ type: 'resumeSession', sessionId: message.id });
        break;

      case 'getWelcomeSuggestions':
        this._agentProcess.sendCommand({ type: 'getWelcomeSuggestions' });
        break;

      case 'complete':
        this._agentProcess.sendCommand({ type: 'complete', query: message.query });
        break;

      case 'generateCommitMessage':
        this.generateCommitMessage();
        break;

      case 'runPrompt': {
        if (this._isRunning) return;
        const editor = vscode.window.activeTextEditor;
        if (!editor || !editor.document.uri.fsPath.toLowerCase().endsWith('.md')) return;
        const content = editor.document.getText();
        if (!content.trim()) return;
        this._isRunning = true;
        this.sendToWebview({ type: 'status', running: true });
        const promptCmd: AgentCommand = {
          type: 'run',
          prompt: content,
          model: this._selectedModel,
          workDir: this._getWorkDir(),
          activeFile: editor.document.uri.fsPath,
        };
        this._agentProcess.sendCommand(promptCmd);
        break;
      }

      case 'focusEditor':
        vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
        break;

      case 'mergeAction': {
        const action = message.action;
        if (action === 'accept') {
          this._mergeManager.acceptChange();
        } else if (action === 'reject') {
          this._mergeManager.rejectChange();
        } else if (action === 'accept-all') {
          this._mergeManager.acceptAll();
          this._agentProcess.sendCommand({ type: 'mergeAction', action: 'accept-all' });
        } else if (action === 'reject-all') {
          this._mergeManager.rejectAll();
          this._agentProcess.sendCommand({ type: 'mergeAction', action: 'reject-all' });
        } else if (action === 'next') {
          this._mergeManager.nextChange();
        } else if (action === 'prev') {
          this._mergeManager.prevChange();
        }
        break;
      }
    }
  }

  public submitTask(prompt: string): void {
    if (this._isRunning || !prompt.trim()) return;
    this._agentProcess.start(this._getWorkDir());
    this._isRunning = true;
    this.sendToWebview({ type: 'status', running: true });
    const activeFile = vscode.window.activeTextEditor?.document.uri.fsPath;
    const cmd: AgentCommand = {
      type: 'run',
      prompt: prompt.trim(),
      model: this._selectedModel,
      workDir: this._getWorkDir(),
      activeFile,
    };
    this._agentProcess.sendCommand(cmd);
  }

  public async focusChatInput(): Promise<void> {
    if (!this._view) return;
    this._view.show(false);
    await new Promise(r => setTimeout(r, 150));
    this._view?.webview.postMessage({ type: 'focusInput' });
  }

  public sendToWebview(message: ToWebviewMessage): void {
    if (this._view) {
      this._view.webview.postMessage(message);
    }
  }

  public newConversation(): void {
    this._isRunning = false;
    this._agentProcess.sendCommand({ type: 'newChat' });
    this.sendToWebview({ type: 'status', running: false });
    this.sendToWebview({ type: 'clearChat' });
  }

  public stopTask(): void {
    this._agentProcess.stop();
  }

  public generateCommitMessage(): void {
    this._agentProcess.start(this._getWorkDir());
    this._agentProcess.sendCommand({ type: 'generateCommitMessage' });
  }

  public dispose(): void {
    this._activeEditorDisposable?.dispose();
    this._agentProcess.dispose();
    this._mergeManager.dispose();
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
        <span class="logo">\u2731 KISS Sorcar <span class="version">${this._getVersion()}</span></span>
      </div>
      <div class="status">
        <span class="dot" id="status-dot"></span>
        <span id="status-text">Ready</span>
      </div>
    </header>

    <div id="output">
      <div id="welcome">
        <h2>Welcome to KISS Sorcar</h2>
        <p>Your AI coding assistant. Ask me anything about your code!</p>
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
            <textarea id="task-input" placeholder="Ask anything... (@ for files, ⌘D toggle between editor and chat, ⌘T new chat, ⌘L run selected text in editor as task)" rows="1"></textarea>
            <button id="input-clear-btn" style="display:none;">&times;</button>
          </div>
        </div>
        <div id="input-footer">
          <div id="model-picker">
            <button id="model-btn" data-tooltip="Select model">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z"/></svg>
              <span id="model-name">claude-opus-4-6</span>
            </button>
            <button id="upload-btn" data-tooltip="Attach files">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
            </button>
            <button id="clear-btn" data-tooltip="New chat">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </button>
            <button id="history-btn" data-tooltip="Task history">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
              </svg>
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

  private _getVersion(): string {
    try {
      const kissRoot = this._agentProcess.findKissProject();
      if (kissRoot) {
        const versionFile = path.join(kissRoot, 'src', 'kiss', '_version.py');
        const content = fs.readFileSync(versionFile, 'utf-8');
        const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
        if (match) return match[1];
      }
    } catch { /* ignore */ }
    return '';
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
