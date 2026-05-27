/**
 * Shared utilities for Sorcar webview HTML rendering.
 * Used by SorcarSidebarView.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import {findKissProject} from './kissPaths';

/** Read the KISS project version from ``_version.py`` on disk. */
export function getVersion(): string {
  try {
    const kissRoot = findKissProject();
    if (kissRoot) {
      const versionFile = path.join(kissRoot, 'src', 'kiss', '_version.py');
      const content = fs.readFileSync(versionFile, 'utf-8');
      const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
      if (match) return match[1];
    }
  } catch {
    /* ignore */
  }
  return '';
}

/**
 * Read ``src/kiss/INJECTIONS.md`` and return one entry per ``## Trick``
 * section.  Returns an empty list if the file is missing — the Tricks
 * button still renders, just with an empty list.
 */
export function getTricks(): string[] {
  try {
    const kissRoot = findKissProject();
    if (!kissRoot) return [];
    const tricksFile = path.join(kissRoot, 'src', 'kiss', 'INJECTIONS.md');
    const text = fs.readFileSync(tricksFile, 'utf-8');
    const tricks: string[] = [];
    const sections = text.split(/^##\s+/m);
    for (let i = 1; i < sections.length; i++) {
      const section = sections[i];
      const newline = section.indexOf('\n');
      if (newline < 0) continue;
      const title = section.slice(0, newline).trim();
      if (title !== 'Trick') continue;
      const body = section.slice(newline + 1).trim();
      if (body) tricks.push(body);
    }
    return tricks;
  } catch {
    return [];
  }
}

/**
 * Generate a cryptographically random nonce string for Content Security
 * Policy.  Uses Node's CSPRNG (``crypto.randomBytes``) — never
 * ``Math.random``, which is predictable.
 */
export function getNonce(): string {
  // 24 random bytes -> 32 base64url chars; restrict to [A-Za-z0-9] for
  // CSP-safe nonce values.
  return crypto
    .randomBytes(24)
    .toString('base64')
    .replace(/[^A-Za-z0-9]/g, '')
    .slice(0, 32);
}

/**
 * Build the full chat HTML for a Sorcar webview.
 * Used by the sidebar view (SorcarSidebarView).
 */
export function buildChatHtml(
  webview: vscode.Webview,
  extensionUri: vscode.Uri,
  selectedModel: string,
): string {
  const nonce = getNonce();
  const version = getVersion();
  const tricksJson = JSON.stringify(getTricks());

  const styleUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, 'media', 'main.css'),
  );
  const hljsCssUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, 'media', 'highlight-github-dark.min.css'),
  );
  const hljsUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, 'media', 'highlight.min.js'),
  );
  const markedUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, 'media', 'marked.min.js'),
  );
  const scriptUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, 'media', 'main.js'),
  );
  const demoScriptUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, 'media', 'demo.js'),
  );

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} data: https:; font-src ${webview.cspSource}; form-action 'none'; frame-src 'none'; object-src 'none'; base-uri 'none';">
  <link href="${styleUri}" rel="stylesheet">
  <link href="${hljsCssUri}" rel="stylesheet">
  <title>KISS Sorcar</title>
</head>
<body>
  <div id="app">
    <div id="tab-bar"><div id="tab-list"></div></div>

    <div id="tab-status-bar">
      <div class="status">
        <span id="status-text">Ready</span>
        <span id="status-tokens" class="status-metric"></span>
        <span id="status-budget" class="status-metric"></span>
        <span id="status-steps" class="status-metric"></span>
      </div>
    </div>

    <div id="task-panel"><button id="task-panel-chevron" type="button" aria-label="Toggle panel visibility"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg></button><div id="task-panel-text"></div><button id="task-panel-copy" type="button" aria-label="Copy task to clipboard"><svg class="icon-copy" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><svg class="icon-check" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="display:none;"><polyline points="20 6 9 17 4 12"/></svg></button></div>

    <div id="output">
      <div id="welcome">
        <h2>Welcome to KISS Sorcar</h2>
        <p>Your AI assistant. Ask me anything!</p>
        <div id="welcome-config" class="welcome-config" style="display:none;">
          <div id="welcome-remote-url"></div>
          <label class="config-label welcome-config-label">Remote password
            <div class="config-password-wrap">
              <input type="password" id="welcome-cfg-remote-password" placeholder="Remote access password">
              <button type="button" id="welcome-cfg-remote-password-toggle" class="config-password-toggle" aria-label="Show password" aria-pressed="false" title="Show password">
                <svg class="icon-eye" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>
                <svg class="icon-eye-off" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="display:none;"><path d="M17.94 17.94A10.94 10.94 0 0 1 12 19c-7 0-11-7-11-7a19.7 19.7 0 0 1 4.22-5.06"/><path d="M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 7 11 7a19.7 19.7 0 0 1-3.16 4.19"/><path d="M14.12 14.12A3 3 0 0 1 9.88 9.88"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
              </button>
            </div>
          </label>
        </div>
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
            <textarea id="task-input" placeholder="Ask anything... (@ for files, ${process.platform === 'darwin' ? '⌘' : 'Ctrl+'}D toggle between editor and chat, ${process.platform === 'darwin' ? '⌘' : 'Ctrl+'}T new chat, ${process.platform === 'darwin' ? '⌘' : 'Ctrl+'}E run selected text as task, ${process.platform === 'darwin' ? '⌘' : 'Ctrl+'}L copy text to chat)" rows="1"></textarea>
            <button id="input-clear-btn" style="display:none;">&times;</button>
          </div>
        </div>
        <div id="input-footer">
          <div id="model-picker">
            <button id="menu-btn">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
            </button>
            <button id="model-btn" data-tooltip="Select model">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z"/></svg>
              <span id="model-name">${selectedModel}</span>
            </button>
            <button id="upload-btn" data-tooltip="Attach files">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
            </button>
            <button id="frequent-tasks-btn" class="toggle-btn" data-tooltip="Frequent tasks">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="20" x2="6" y2="14"/><line x1="12" y1="20" x2="12" y2="9"/><line x1="18" y1="20" x2="18" y2="4"/></svg>
            </button>
            <button id="tricks-btn" class="toggle-btn" data-tooltip="Inject instruction">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 00-4 12.7c.7.6 1 1.4 1 2.3h6c0-.9.3-1.7 1-2.3A7 7 0 0012 2z"/></svg>
            </button>
            <div id="model-dropdown">
              <div class="search-wrap">
                <input type="text" id="model-search" placeholder="Search models...">
                <button class="search-clear-btn" id="model-search-clear" style="display:none;">&times;</button>
              </div>
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
      <button id="sidebar-close">&times;</button>
      <div id="sidebar-tab-history-panel" class="sidebar-section sidebar-tab-panel">
        <div class="sidebar-hdr">History</div>
        <div class="search-wrap">
          <input type="text" id="history-search" placeholder="Search history...">
          <button class="search-clear-btn" id="history-search-clear" style="display:none;">&times;</button>
        </div>
        <div class="history-filter-bar">
          <label class="history-filter-chk" title="Show running tasks">
            <input type="checkbox" id="hf-running" checked>Running
          </label>
          <label class="history-filter-chk" title="Show tasks that finished with error">
            <input type="checkbox" id="hf-errors" checked>Errored
          </label>
          <label class="history-filter-chk" title="Show successfully completed tasks">
            <input type="checkbox" id="hf-completed" checked>Succeeded
          </label>
          <label class="history-filter-chk" title="Show only tasks marked as favourite">
            <input type="checkbox" id="hf-favorite">Favorites
          </label>
          <label for="hf-from" class="history-filter-date-lbl">From:</label>
          <input type="date" id="hf-from" class="history-filter-date" title="From date" aria-label="From date">
          <button type="button" id="hf-from-btn" class="history-filter-date-btn" title="Pick From date" aria-label="Pick From date">📅</button>
          <label for="hf-to" class="history-filter-date-lbl">To:</label>
          <input type="date" id="hf-to" class="history-filter-date" title="To date" aria-label="To date">
          <button type="button" id="hf-to-btn" class="history-filter-date-btn" title="Pick To date" aria-label="Pick To date">📅</button>
        </div>
        <div id="history-list">
          <div class="sidebar-empty">No conversations yet</div>
        </div>
      </div>
    </div>
    <div id="sidebar-overlay"></div>

    <div id="frequent-panel">
      <button id="frequent-panel-close">&times;</button>
      <div class="sidebar-section sidebar-tab-panel">
        <div class="sidebar-hdr">Frequent tasks</div>
        <div id="frequent-list">
          <div class="sidebar-empty">No tasks yet</div>
        </div>
      </div>
    </div>
    <div id="frequent-overlay"></div>

    <div id="tricks-panel">
      <button id="tricks-panel-close">&times;</button>
      <div class="sidebar-section sidebar-tab-panel">
        <div class="sidebar-hdr">Inject</div>
        <div id="tricks-list">
          <div class="sidebar-empty">No tricks available</div>
        </div>
      </div>
    </div>
    <div id="tricks-overlay"></div>

    <div id="settings-panel">
      <button id="settings-panel-close">&times;</button>
      <div class="sidebar-section">
        <div class="sidebar-hdr">Sorcar Configuration${version ? ' ' + version : ''}</div>
        <div id="remote-url"></div>
        <div id="config-form">
          <label class="config-label">Remote password
            <div class="config-password-wrap">
              <input type="password" id="cfg-remote-password" placeholder="Remote access password">
              <button type="button" id="cfg-remote-password-toggle" class="config-password-toggle" aria-label="Show password" aria-pressed="false" title="Show password">
                <svg class="icon-eye" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>
                <svg class="icon-eye-off" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="display:none;"><path d="M17.94 17.94A10.94 10.94 0 0 1 12 19c-7 0-11-7-11-7a19.7 19.7 0 0 1 4.22-5.06"/><path d="M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 7 11 7a19.7 19.7 0 0 1-3.16 4.19"/><path d="M14.12 14.12A3 3 0 0 1 9.88 9.88"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
              </button>
            </div>
          </label>
          <label class="config-label">Max budget per task ($)
            <input type="number" id="cfg-max-budget" min="0" step="1" value="100">
          </label>
          <label class="config-label config-checkbox">
            <input type="checkbox" id="cfg-use-web-browser" checked>
            Use web browser
          </label>
          <label class="config-label config-checkbox">
            <input type="checkbox" id="cfg-auto-commit" checked>
            Auto commit
            <button id="autocommit-btn" type="button" data-tooltip="git commit">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><line x1="1.05" y1="12" x2="7" y2="12"/><line x1="17.01" y1="12" x2="22.96" y2="12"/><line x1="12" y1="1.05" x2="12" y2="7"/><line x1="12" y1="17.01" x2="12" y2="22.96"/></svg>
            </button>
          </label>
          <label class="config-label config-checkbox">
            <input type="checkbox" id="cfg-use-parallel" checked>
            Use parallel agents
          </label>
          <label class="config-label config-checkbox">
            <input type="checkbox" id="cfg-use-worktree" checked>
            Use worktree
          </label>
          <label class="config-label config-checkbox">
            <input type="checkbox" id="cfg-demo-mode">
            Demo mode
          </label>

          <div class="config-divider"></div>
          <div class="sidebar-hdr" style="margin-top:8px;">API Keys</div>
          <label class="config-label">Gemini API Key
            <input type="text" id="cfg-key-GEMINI_API_KEY" placeholder="Enter Gemini API key">
          </label>
          <label class="config-label">OpenAI API Key
            <input type="text" id="cfg-key-OPENAI_API_KEY" placeholder="Enter OpenAI API key">
          </label>
          <label class="config-label">Anthropic API Key
            <input type="text" id="cfg-key-ANTHROPIC_API_KEY" placeholder="Enter Anthropic API key">
          </label>
          <label class="config-label">Together API Key
            <input type="text" id="cfg-key-TOGETHER_API_KEY" placeholder="Enter Together API key">
          </label>
          <label class="config-label">OpenRouter API Key
            <input type="text" id="cfg-key-OPENROUTER_API_KEY" placeholder="Enter OpenRouter API key">
          </label>
          <label class="config-label">MiniMax API Key
            <input type="text" id="cfg-key-MINIMAX_API_KEY" placeholder="Enter MiniMax API key">
          </label>
          <label class="config-label">Custom endpoint (local model)
            <input type="text" id="cfg-custom-endpoint" placeholder="http://localhost:8080/v1">
          </label>
          <label class="config-label">Custom API key
            <input type="text" id="cfg-custom-api-key" placeholder="Optional API key for custom endpoint">
          </label>
          <label class="config-label">Custom headers
            <textarea id="cfg-custom-headers" rows="2" placeholder="Key:Value (one per line)"></textarea>
          </label>
        </div>
      </div>
    </div>
    <div id="settings-overlay"></div>

    <div id="ask-user-modal" style="display:none;">
      <div class="modal-content">
        <div class="modal-title">Agent needs your input</div>
        <div id="ask-user-slot"></div>
      </div>
    </div>
  </div>

  <script nonce="${nonce}" src="${hljsUri}"></script>
  <script nonce="${nonce}" src="${markedUri}"></script>
  <script nonce="${nonce}">window.__TRICKS__ = ${tricksJson};</script>
  <script nonce="${nonce}" src="${scriptUri}"></script>
  <script nonce="${nonce}" src="${demoScriptUri}"></script>
</body>
</html>`;
}
