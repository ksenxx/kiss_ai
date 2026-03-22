"use strict";
/**
 * Webview panel manager for Sorcar chat interface.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.SorcarViewProvider = void 0;
const vscode = __importStar(require("vscode"));
const AgentProcess_1 = require("./AgentProcess");
const MergeManager_1 = require("./MergeManager");
class SorcarViewProvider {
    _view;
    _agentProcess;
    _extensionUri;
    _selectedModel;
    _isRunning = false;
    _mergeManager;
    constructor(extensionUri, mergeManager) {
        this._extensionUri = extensionUri;
        this._agentProcess = new AgentProcess_1.AgentProcess();
        this._mergeManager = mergeManager || new MergeManager_1.MergeManager();
        this._selectedModel = vscode.workspace.getConfiguration('kissSorcar').get('defaultModel') || 'claude-opus-4-6';
        this._mergeManager.on('allDone', () => {
            this._agentProcess.sendCommand({ type: 'mergeAction', action: 'all-done' });
            this.sendToWebview({ type: 'merge_ended' });
        });
        this._mergeManager.on('hunkProcessed', () => {
            this._agentProcess.sendCommand({ type: 'mergeAction', action: 'accept' });
        });
        // Listen for agent events
        this._agentProcess.on('message', (msg) => {
            if (msg.type === 'merge_data') {
                this._mergeManager.openMerge(msg.data).catch((err) => {
                    console.error('[SorcarPanel] merge open failed:', err);
                });
            }
            this.sendToWebview(msg);
            if (msg.type === 'status') {
                this._isRunning = msg.running;
            }
        });
    }
    get mergeManager() {
        return this._mergeManager;
    }
    resolveWebviewView(webviewView, _context, _token) {
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
        webviewView.webview.onDidReceiveMessage((message) => this._handleMessage(message), undefined, []);
        // Start the agent process
        const workDir = this._getWorkDir();
        this._agentProcess.start(workDir);
    }
    _getWorkDir() {
        const folders = vscode.workspace.workspaceFolders;
        if (folders && folders.length > 0) {
            return folders[0].uri.fsPath;
        }
        return process.cwd();
    }
    async _handleMessage(message) {
        switch (message.type) {
            case 'ready':
                this.sendToWebview({ type: 'status', running: this._isRunning });
                this._agentProcess.sendCommand({ type: 'getModels' });
                this._agentProcess.sendCommand({ type: 'getWelcomeSuggestions' });
                break;
            case 'submit':
                if (this._isRunning)
                    return;
                this._isRunning = true;
                this.sendToWebview({ type: 'status', running: true });
                const activeFile = vscode.window.activeTextEditor?.document.uri.fsPath;
                const cmd = {
                    type: 'run',
                    prompt: message.prompt,
                    model: message.model,
                    workDir: this._getWorkDir(),
                    activeFile: activeFile,
                    attachments: message.attachments,
                };
                this._agentProcess.sendCommand(cmd);
                break;
            case 'stop':
                this._agentProcess.stop();
                break;
            case 'selectModel':
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
            case 'resumeSession':
                this._agentProcess.sendCommand({ type: 'resumeSession', sessionId: message.id });
                break;
            case 'getWelcomeSuggestions':
                this._agentProcess.sendCommand({ type: 'getWelcomeSuggestions' });
                break;
            case 'complete':
                this._agentProcess.sendCommand({ type: 'complete', query: message.query });
                break;
            case 'mergeAction': {
                const action = message.action;
                if (action === 'accept') {
                    this._mergeManager.acceptChange();
                }
                else if (action === 'reject') {
                    this._mergeManager.rejectChange();
                }
                else if (action === 'accept-all') {
                    this._mergeManager.acceptAll();
                    this._agentProcess.sendCommand({ type: 'mergeAction', action: 'accept-all' });
                }
                else if (action === 'reject-all') {
                    this._mergeManager.rejectAll();
                    this._agentProcess.sendCommand({ type: 'mergeAction', action: 'reject-all' });
                }
                else if (action === 'next') {
                    this._mergeManager.nextChange();
                }
                else if (action === 'prev') {
                    this._mergeManager.prevChange();
                }
                break;
            }
        }
    }
    sendToWebview(message) {
        if (this._view) {
            this._view.webview.postMessage(message);
        }
    }
    newConversation() {
        this._isRunning = false;
        this.sendToWebview({ type: 'status', running: false });
        this.sendToWebview({ type: 'clearChat' });
    }
    stopTask() {
        this._agentProcess.stop();
    }
    dispose() {
        this._agentProcess.dispose();
        this._mergeManager.dispose();
    }
    _getHtmlContent(webview) {
        const nonce = this._getNonce();
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'main.css'));
        const hljsCssUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'highlight-github-dark.min.css'));
        const hljsUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'highlight.min.js'));
        const markedUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'marked.min.js'));
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js'));
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
        <span class="logo">\u2731 KISS Sorcar</span>
        <div class="status">
          <span class="dot" id="status-dot"></span>
          <span id="status-text">Ready</span>
        </div>
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
            <textarea id="task-input" placeholder="Ask anything... (@ to mention files)" rows="1"></textarea>
          </div>
        </div>
        <div id="input-footer">
          <div id="model-picker">
            <button id="model-btn">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z"/></svg>
              <span id="model-name">claude-opus-4-6</span>
            </button>
            <button id="upload-btn" title="Attach files">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
            </button>
            <button id="clear-btn" title="Clear chat">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
            </button>
            <button id="history-btn" title="Task history">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
              </svg>
            </button>
            <div id="model-dropdown">
              <input type="text" id="model-search" placeholder="Search models...">
              <div id="model-list"></div>
            </div>
          </div>
          <div id="input-actions">
            <span id="wait-spinner"></span>
            <button id="send-btn" title="Send">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
            </button>
            <button id="stop-btn" title="Stop" style="display:none;">
              <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <div id="sidebar">
      <button id="sidebar-close">&times;</button>
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
        <button id="ask-user-submit">Submit</button>
      </div>
    </div>
  </div>

  <script nonce="${nonce}" src="${hljsUri}"></script>
  <script nonce="${nonce}" src="${markedUri}"></script>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
    }
    _getNonce() {
        let text = '';
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        for (let i = 0; i < 32; i++) {
            text += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return text;
    }
}
exports.SorcarViewProvider = SorcarViewProvider;
//# sourceMappingURL=SorcarPanel.js.map