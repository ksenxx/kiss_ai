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
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const AgentProcess_1 = require("./AgentProcess");
class SorcarViewProvider {
    static viewType = 'kissSorcar.chatView';
    _view;
    _agentProcess;
    _extensionUri;
    _selectedModel;
    _isRunning = false;
    /** Maps file path -> original content saved before a Write/Edit tool call. */
    _originals = new Map();
    /** The file path from the most recent file-modifying tool_call, if any. */
    _pendingDiffPath = null;
    /** Directory for storing original file snapshots for diff view. */
    _tmpDir;
    constructor(extensionUri) {
        this._extensionUri = extensionUri;
        this._agentProcess = new AgentProcess_1.AgentProcess();
        this._selectedModel = vscode.workspace.getConfiguration('kissSorcar').get('defaultModel') || 'claude-opus-4-6';
        this._tmpDir = path.join(extensionUri.fsPath, '.diff-originals');
        // Listen for agent events
        this._agentProcess.on('message', (msg) => {
            this._handleAgentEvent(msg);
            this.sendToWebview(msg);
            if (msg.type === 'status') {
                this._isRunning = msg.running;
            }
        });
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
                // Webview is ready, send initial state
                this.sendToWebview({ type: 'status', running: this._isRunning });
                this._agentProcess.sendCommand({ type: 'getModels' });
                break;
            case 'submit':
                if (this._isRunning)
                    return;
                this._isRunning = true;
                this.sendToWebview({ type: 'status', running: true });
                // Get active editor file
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
                this._selectedModel = message.model;
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
        }
    }
    sendToWebview(message) {
        if (this._view) {
            this._view.webview.postMessage(message);
        }
    }
    /**
     * Handle agent events to detect file modifications and show merge/diff views.
     *
     * On tool_call for Write/Edit: saves the original file content.
     * On tool_result (success) after a file-modifying tool: opens VS Code diff editor.
     * On task_done/task_error/task_stopped: cleans up temporary files.
     */
    _handleAgentEvent(msg) {
        if (msg.type === 'tool_call') {
            const isFileModify = msg.name === 'Write' || msg.name === 'Edit';
            if (isFileModify && msg.path) {
                this._saveOriginal(msg.path);
                this._pendingDiffPath = msg.path;
            }
            else {
                this._pendingDiffPath = null;
            }
        }
        else if (msg.type === 'tool_result' && this._pendingDiffPath) {
            const filePath = this._pendingDiffPath;
            this._pendingDiffPath = null;
            if (!msg.is_error) {
                this._showDiff(filePath);
            }
        }
        else if (msg.type === 'task_done' || msg.type === 'task_error' || msg.type === 'task_stopped') {
            this._cleanup();
        }
    }
    /**
     * Save a snapshot of the file's current content before modification.
     */
    _saveOriginal(filePath) {
        try {
            if (this._originals.has(filePath)) {
                return; // Already have an original for this file in this session
            }
            if (fs.existsSync(filePath)) {
                this._originals.set(filePath, fs.readFileSync(filePath, 'utf-8'));
            }
            else {
                // New file - original is empty
                this._originals.set(filePath, '');
            }
        }
        catch {
            // Ignore errors reading file
        }
    }
    /**
     * Show VS Code diff editor comparing the original file content to the current modified version.
     */
    async _showDiff(filePath) {
        const original = this._originals.get(filePath);
        if (original === undefined) {
            return;
        }
        try {
            // Write original content to a temp file for the diff left side
            if (!fs.existsSync(this._tmpDir)) {
                fs.mkdirSync(this._tmpDir, { recursive: true });
            }
            const fileName = path.basename(filePath);
            const tmpFile = path.join(this._tmpDir, `original-${fileName}`);
            fs.writeFileSync(tmpFile, original, 'utf-8');
            const originalUri = vscode.Uri.file(tmpFile);
            const modifiedUri = vscode.Uri.file(filePath);
            const title = `${fileName} (Sorcar Changes)`;
            await vscode.commands.executeCommand('vscode.diff', originalUri, modifiedUri, title);
        }
        catch {
            // Ignore errors showing diff
        }
    }
    /**
     * Clean up temporary diff files and reset state.
     */
    _cleanup() {
        this._originals.clear();
        this._pendingDiffPath = null;
        try {
            if (fs.existsSync(this._tmpDir)) {
                const files = fs.readdirSync(this._tmpDir);
                for (const file of files) {
                    fs.unlinkSync(path.join(this._tmpDir, file));
                }
                fs.rmdirSync(this._tmpDir);
            }
        }
        catch {
            // Ignore cleanup errors
        }
    }
    newConversation() {
        // Clear the chat and start fresh
        this.sendToWebview({ type: 'status', running: false });
        // Webview will handle clearing its state
    }
    stopTask() {
        this._agentProcess.stop();
    }
    dispose() {
        this._agentProcess.dispose();
        this._cleanup();
    }
    _getHtmlContent(webview) {
        const nonce = this._getNonce();
        // Get resource URIs
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'main.css'));
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js'));
        return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} data: https:; font-src ${webview.cspSource};">
  <link href="${styleUri}" rel="stylesheet">
  <title>KISS Sorcar</title>
</head>
<body>
  <div id="app">
    <header>
      <div class="header-left">
        <span class="logo">✱ KISS Sorcar</span>
        <div class="status">
          <span class="dot" id="status-dot"></span>
          <span id="status-text">Ready</span>
        </div>
      </div>
      <div class="header-right">
        <button id="history-btn" title="History">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
        </button>
      </div>
    </header>

    <div id="output">
      <div id="welcome">
        <h2>Welcome to KISS Sorcar</h2>
        <p>Your AI coding assistant. Ask me anything about your code!</p>
        <div id="suggestions">
          <div class="suggestion-chip" data-prompt="Explain this codebase structure">
            <span class="chip-label">Quick Start</span>
            Explain this codebase structure
          </div>
          <div class="suggestion-chip" data-prompt="Find and fix bugs in this file">
            <span class="chip-label">Quick Start</span>
            Find and fix bugs in this file
          </div>
          <div class="suggestion-chip" data-prompt="Write tests for the current file">
            <span class="chip-label">Quick Start</span>
            Write tests for the current file
          </div>
          <div class="suggestion-chip" data-prompt="Optimize this code for performance">
            <span class="chip-label">Quick Start</span>
            Optimize this code for performance
          </div>
        </div>
      </div>
    </div>

    <div id="input-area">
      <div id="autocomplete"></div>
      <div id="file-chips"></div>
      <div id="input-container">
        <div id="input-wrap">
          <div id="input-text-wrap">
            <textarea id="task-input" placeholder="Ask anything... (@ to mention files)" rows="1"></textarea>
          </div>
          <div id="input-actions">
            <button id="upload-btn" title="Attach file">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
              </svg>
            </button>
            <button id="send-btn" title="Send">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
            <button id="stop-btn" title="Stop" style="display:none;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2"></rect>
              </svg>
            </button>
          </div>
        </div>
        <div id="input-footer">
          <div id="model-picker">
            <button id="model-btn">
              <span id="model-name">claude-opus-4-6</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </button>
            <div id="model-dropdown">
              <input type="text" id="model-search" placeholder="Search models...">
              <div id="model-list"></div>
            </div>
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