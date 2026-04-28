"""Standalone web server for remote KISS Sorcar access.

Provides HTTP + WebSocket access to the Sorcar chat interface from any
browser, including mobile devices.  Uses the ``websockets`` library to
serve both HTTP (for the HTML page and static media assets) and
WebSocket (for bidirectional command/event communication) on a single
port.

Authentication uses the ``remote_password`` setting from
``~/.kiss/config.json``.  An optional ``cloudflared`` quick-tunnel can
expose the server through Cloudflare so devices outside the LAN can
connect without manual port-forwarding.

Usage::

    from kiss.agents.vscode.web_server import RemoteAccessServer

    server = RemoteAccessServer(port=8787, use_tunnel=True)
    server.start()   # blocks until Ctrl-C
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from kiss.agents.sorcar.persistence import _append_chat_event
from kiss.agents.vscode.browser_ui import _DISPLAY_EVENT_TYPES, BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.vscode_config import load_config

__all__ = ["RemoteAccessServer", "WebPrinter"]

logger = logging.getLogger(__name__)

MEDIA_DIR = Path(__file__).parent / "media"

#: Commands that are VS-Code-UI-specific and have no backend handler.
_VSCODE_ONLY_COMMANDS = frozenset({
    "closeSecondaryBar",
    "focusEditor",
    "webviewFocusChanged",
    "openFile",
    "resolveDroppedPaths",
    "runPrompt",
    "getWelcomeSuggestions",
})


# ---------------------------------------------------------------------------
# WebPrinter — sends events over WebSocket instead of stdout
# ---------------------------------------------------------------------------


class WebPrinter(BaseBrowserPrinter):
    """Printer that broadcasts JSON events to connected WebSocket clients.

    Thread-safe: ``broadcast()`` is called from agent task-runner threads
    and the asyncio event loop.  A lock protects the client set, and
    ``asyncio.run_coroutine_threadsafe`` is used to schedule sends on
    the event loop from non-async threads.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ws_clients: set[ServerConnection] = set()
        self._ws_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._persist_agents: dict[str, Any] = {}

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send *event* to every connected WebSocket client.

        Injects ``tabId`` from thread-local storage (same as
        ``VSCodePrinter``), records the event for replay, and persists
        display events to the database.

        Args:
            event: The event dictionary to emit.
        """
        tab_id = getattr(self._thread_local, "tab_id", None)
        if tab_id is not None and "tabId" not in event:
            event = {**event, "tabId": tab_id}

        with self._lock:
            self._record_event(event)

        # Persist display events to the database
        if event.get("type") in _DISPLAY_EVENT_TYPES:
            evt_tab = event.get("tabId")
            if evt_tab is not None:
                agent = self._persist_agents.get(evt_tab)
                if agent is not None:
                    task_id = agent._last_task_id
                    if task_id is not None:
                        _append_chat_event(event, task_id=task_id)

        data = json.dumps(event)
        with self._ws_lock:
            clients = list(self._ws_clients)
        loop = self._loop
        for ws in clients:
            if loop is not None and loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(ws.send(data), loop)
                except Exception:
                    logger.debug("Failed to send to WS client", exc_info=True)

    def add_client(self, ws: ServerConnection) -> None:
        """Register a WebSocket client for event broadcasting.

        Args:
            ws: The WebSocket server connection to add.
        """
        with self._ws_lock:
            self._ws_clients.add(ws)

    def remove_client(self, ws: ServerConnection) -> None:
        """Remove a WebSocket client from event broadcasting.

        Args:
            ws: The WebSocket server connection to remove.
        """
        with self._ws_lock:
            self._ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# HTML template with WebSocket shim
# ---------------------------------------------------------------------------


def _build_html() -> str:
    """Build the standalone HTML page for remote Sorcar access.

    Produces HTML equivalent to ``SorcarTab.buildChatHtml`` but uses
    plain ``/media/`` URLs for assets and injects a WebSocket shim
    script that provides ``acquireVsCodeApi()`` for ``main.js``.

    Returns:
        The complete HTML string.
    """
    version = _read_version()
    shim = _WS_SHIM_JS
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="/media/main.css" rel="stylesheet">
  <link href="/media/highlight-github-dark.min.css" rel="stylesheet">
  <title>KISS Sorcar</title>
  <style>
    html, body {{ height: 100%; margin: 0; padding: 0; overflow: hidden; }}
    body {{ background: var(--vscode-editor-background, #1e1e1e);
            color: var(--vscode-editor-foreground, #cccccc); }}
    :root {{
      --vscode-editor-background: #1e1e1e;
      --vscode-editor-foreground: #cccccc;
      --vscode-input-background: #3c3c3c;
      --vscode-input-foreground: #cccccc;
      --vscode-input-border: #3c3c3c;
      --vscode-focusBorder: #007acc;
      --vscode-button-background: #0e639c;
      --vscode-button-foreground: #ffffff;
      --vscode-button-hoverBackground: #1177bb;
      --vscode-sideBar-background: #252526;
      --vscode-list-hoverBackground: #2a2d2e;
      --vscode-badge-background: #4d4d4d;
      --vscode-badge-foreground: #ffffff;
      --vscode-textLink-foreground: #3794ff;
      --vscode-descriptionForeground: #8b8b8b;
      --vscode-editorWidget-background: #252526;
      --vscode-editorWidget-border: #454545;
      --vscode-panel-border: #80808059;
    }}
  </style>
</head>
<body>
  <div id="app">
    <div id="tab-bar"><div id="tab-list"></div><button id="config-btn" title="Configuration">
              <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"/>
                <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 \
2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 \
2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 \
01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 \
010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 \
012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 \
0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 \
2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 \
4h-.09a1.65 1.65 0 00-1.51 1z"/>
              </svg>
            </button><button id="history-btn">
              <svg class="history-chevron" width="1em" height="1em" viewBox="0 0 24 24"
               fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"
               stroke-linejoin="round">
                <polyline points="15 18 9 12 15 6"/>
              </svg>
              <span>History</span>
            </button></div>

    <div id="tab-status-bar">
      <div class="status">
        <span id="status-text">Ready</span>
        <span id="status-tokens" class="status-metric"></span>
        <span id="status-budget" class="status-metric"></span>
        <span id="status-steps" class="status-metric"></span>
      </div>
    </div>

    <div id="task-panel">
      <button id="task-panel-chevron" type="button" aria-label="Toggle panel visibility">
        <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="9 18 15 12 9 6"/>
        </svg>
      </button>
      <div id="task-panel-text"></div>
    </div>

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
            <textarea id="task-input"
             placeholder="Ask anything... (@ for files)" rows="1"></textarea>
            <button id="input-clear-btn" style="display:none;">&times;</button>
          </div>
        </div>
        <div id="input-footer">
          <div id="model-picker">
            <button id="model-btn" data-tooltip="Select model">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2">
                <path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z"/>
              </svg>
              <span id="model-name">loading...</span>
            </button>
            <button id="upload-btn" data-tooltip="Attach files">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19\
a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/>
              </svg>
            </button>
            <button id="worktree-toggle-btn" data-tooltip="Use worktree">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round">
                <line x1="6" y1="3" x2="6" y2="15"/>
                <circle cx="18" cy="6" r="3"/>
                <circle cx="6" cy="18" r="3"/>
                <path d="M18 9a9 9 0 01-9 9"/>
              </svg>
            </button>
            <button id="parallel-toggle-btn" data-tooltip="Use parallelism">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round">
                <line x1="6" y1="4" x2="6" y2="20"/>
                <line x1="12" y1="4" x2="12" y2="20"/>
                <line x1="18" y1="4" x2="18" y2="20"/>
              </svg>
            </button>
            <button id="run-prompt-btn" data-tooltip="Run current file as prompt"
             disabled>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"
               stroke="none">
                <polygon points="5,3 19,12 5,21"/>
              </svg>
            </button>
            <button id="demo-toggle-btn" data-tooltip="Toggle demo mode">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
                <line x1="8" y1="21" x2="16" y2="21"/>
                <line x1="12" y1="17" x2="12" y2="21"/>
              </svg>
            </button>
            <div id="model-dropdown">
              <div class="search-wrap">
                <input type="text" id="model-search" placeholder="Search models...">
                <button class="search-clear-btn" id="model-search-clear"
                 style="display:none;">&times;</button>
              </div>
              <div id="model-list"></div>
            </div>
          </div>
          <div id="input-actions">
            <button id="autocommit-btn" data-tooltip="Auto commit changes">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round">
                <circle cx="12" cy="12" r="4"/>
                <line x1="1.05" y1="12" x2="7" y2="12"/>
                <line x1="17.01" y1="12" x2="22.96" y2="12"/>
                <line x1="12" y1="1.05" x2="12" y2="7"/>
                <line x1="12" y1="17.01" x2="12" y2="22.96"/>
              </svg>
            </button>
            <span id="wait-spinner"></span>
            <button id="send-btn" data-tooltip="Send message">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            </button>
            <button id="stop-btn" data-tooltip="Stop agent" style="display:none;">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <div id="sidebar">
      <button id="sidebar-close">&times;</button>
      <div class="sidebar-section">
        <div class="sidebar-hdr">Recent Conversations</div>
        <div class="search-wrap">
          <input type="text" id="history-search" placeholder="Search history...">
          <button class="search-clear-btn" id="history-search-clear"
           style="display:none;">&times;</button>
        </div>
        <div id="history-list">
          <div class="sidebar-empty">No conversations yet</div>
        </div>
      </div>
    </div>
    <div id="sidebar-overlay"></div>

    <div id="config-sidebar">
      <button id="config-sidebar-close">&times;</button>
      <div class="sidebar-section">
        <div class="sidebar-hdr">Sorcar Configuration{' ' + version if version else ''}</div>
        <div id="config-form">
          <label class="config-label">Max budget per task ($)
            <input type="number" id="cfg-max-budget" min="0" step="1" value="100">
          </label>
          <label class="config-label">Custom endpoint (local model)
            <input type="text" id="cfg-custom-endpoint"
             placeholder="http://localhost:8080/v1">
          </label>
          <label class="config-label">Custom API key
            <input type="password" id="cfg-custom-api-key"
             placeholder="Optional API key for custom endpoint">
          </label>
          <label class="config-label config-checkbox">
            <input type="checkbox" id="cfg-use-web-browser" checked>
            Use web browser
          </label>
          <label class="config-label">Remote password
            <input type="password" id="cfg-remote-password"
             placeholder="Remote access password">
          </label>
          <div class="config-divider"></div>
          <div class="sidebar-hdr" style="margin-top:8px;">API Keys</div>
          <label class="config-label">Gemini API Key
            <input type="password" id="cfg-key-GEMINI_API_KEY"
             placeholder="Enter Gemini API key">
          </label>
          <label class="config-label">OpenAI API Key
            <input type="password" id="cfg-key-OPENAI_API_KEY"
             placeholder="Enter OpenAI API key">
          </label>
          <label class="config-label">Anthropic API Key
            <input type="password" id="cfg-key-ANTHROPIC_API_KEY"
             placeholder="Enter Anthropic API key">
          </label>
          <label class="config-label">Together API Key
            <input type="password" id="cfg-key-TOGETHER_API_KEY"
             placeholder="Enter Together API key">
          </label>
          <label class="config-label">OpenRouter API Key
            <input type="password" id="cfg-key-OPENROUTER_API_KEY"
             placeholder="Enter OpenRouter API key">
          </label>
          <label class="config-label">MiniMax API Key
            <input type="password" id="cfg-key-MINIMAX_API_KEY"
             placeholder="Enter MiniMax API key">
          </label>
          <button id="cfg-save-btn" class="config-save-btn">Save Configuration</button>
        </div>
      </div>
    </div>
    <div id="config-sidebar-overlay"></div>

    <div id="ask-user-modal" style="display:none;">
      <div class="modal-content">
        <div class="modal-title">Agent needs your input</div>
        <div id="ask-user-slot"></div>
      </div>
    </div>
  </div>

  <script src="/media/highlight.min.js"></script>
  <script src="/media/marked.min.js"></script>
  <script>{shim}</script>
  <script src="/media/main.js"></script>
  <script src="/media/demo.js"></script>
</body>
</html>"""


def _read_version() -> str:
    """Read the KISS project version from ``_version.py``.

    Returns:
        The version string, or ``""`` if it cannot be read.
    """
    try:
        vfile = Path(__file__).parent.parent.parent / "_version.py"
        text = vfile.read_text()
        for line in text.splitlines():
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("\"'")
    except Exception:
        pass
    return ""


#: JavaScript shim injected before ``main.js`` to replace the VS Code
#: webview API with WebSocket-based communication.
_WS_SHIM_JS = r"""
(function() {
  var _state = null;
  try { _state = JSON.parse(sessionStorage.getItem('sorcar-state')); } catch(e) {}
  var _ws = null;
  var _pending = [];
  var _authenticated = false;
  var _needsPassword = false;

  window.acquireVsCodeApi = function() {
    return {
      postMessage: function(msg) {
        var data = JSON.stringify(msg);
        if (_ws && _ws.readyState === WebSocket.OPEN && _authenticated) {
          _ws.send(data);
        } else {
          _pending.push(data);
        }
      },
      getState: function() { return _state; },
      setState: function(s) {
        _state = s;
        try { sessionStorage.setItem('sorcar-state', JSON.stringify(s)); } catch(e) {}
      }
    };
  };

  function connect() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    _ws = new WebSocket(proto + '//' + location.host + '/ws');
    _authenticated = false;

    _ws.onopen = function() {
      var pwd = sessionStorage.getItem('sorcar-remote-pwd') || '';
      _ws.send(JSON.stringify({type: 'auth', password: pwd}));
    };

    _ws.onmessage = function(event) {
      var msg = JSON.parse(event.data);
      if (msg.type === 'auth_ok') {
        _authenticated = true;
        _needsPassword = false;
        for (var i = 0; i < _pending.length; i++) _ws.send(_pending[i]);
        _pending = [];
        return;
      }
      if (msg.type === 'auth_required') {
        _needsPassword = true;
        var pwd = prompt('Enter remote access password:');
        if (pwd !== null) {
          sessionStorage.setItem('sorcar-remote-pwd', pwd);
          _ws.send(JSON.stringify({type: 'auth', password: pwd}));
        }
        return;
      }
      window.dispatchEvent(new MessageEvent('message', {data: msg}));
    };

    _ws.onclose = function() {
      _authenticated = false;
      setTimeout(connect, 3000);
    };

    _ws.onerror = function() {};
  }

  connect();
})();
"""


def _http_response(status: int, content_type: str, body: bytes) -> Response:
    """Build a proper HTTP/1.1 Response for the websockets server.

    Args:
        status: HTTP status code (e.g. 200, 404).
        content_type: MIME type for the Content-Type header.
        body: Response body bytes.

    Returns:
        A websockets ``Response`` with Content-Length and Connection headers.
    """
    reason = "OK" if status == 200 else "Not Found"
    return Response(
        status,
        reason,
        Headers([
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Connection", "close"),
        ]),
        body,
    )


# ---------------------------------------------------------------------------
# RemoteAccessServer
# ---------------------------------------------------------------------------


class RemoteAccessServer:
    """Web server providing remote browser access to KISS Sorcar.

    Serves the Sorcar chat webview over HTTP and bridges commands/events
    over WebSocket.  Optionally starts a ``cloudflared`` quick-tunnel
    so the server is reachable from the public internet without manual
    port-forwarding or DNS setup.

    Args:
        host: Bind address (default ``"0.0.0.0"`` for all interfaces).
        port: TCP port for both HTTP and WebSocket (default ``8787``).
        use_tunnel: If True, start a ``cloudflared`` tunnel on launch.
        work_dir: Working directory for the agent (default cwd).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8787,
        use_tunnel: bool = False,
        work_dir: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.use_tunnel = use_tunnel

        if work_dir:
            os.environ["KISS_WORKDIR"] = work_dir

        self._vscode_server = VSCodeServer()
        self._printer = WebPrinter()
        self._vscode_server.printer = self._printer  # type: ignore[assignment]

        self._html_bytes = _build_html().encode("utf-8")
        self._tunnel_proc: subprocess.Popen[str] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_server: Any = None

    # -- HTTP handler -------------------------------------------------------

    async def _process_request(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        """Serve HTTP requests for the HTML page and static assets.

        Returns a :class:`Response` for regular HTTP requests, or
        ``None`` to let the WebSocket handshake proceed for ``/ws``.

        Args:
            connection: The server connection (unused for HTTP).
            request: The incoming HTTP request.

        Returns:
            An HTTP response, or ``None`` for WebSocket upgrade.
        """
        path = request.path
        if path == "/" or path == "":
            return _http_response(200, "text/html; charset=utf-8", self._html_bytes)
        if path.startswith("/media/"):
            filename = path[7:]
            filepath = MEDIA_DIR / filename
            if (
                filepath.resolve().is_relative_to(MEDIA_DIR.resolve())
                and filepath.is_file()
            ):
                ctype = mimetypes.guess_type(str(filepath))[0] or "application/octet-stream"
                return _http_response(200, ctype, filepath.read_bytes())
            return _http_response(404, "text/plain", b"Not Found")
        if path == "/ws":
            return None  # proceed to WebSocket
        return _http_response(404, "text/plain", b"Not Found")

    # -- WebSocket handler --------------------------------------------------

    async def _ws_handler(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket client connection.

        Performs password authentication, then relays messages between
        the browser and the ``VSCodeServer`` command dispatcher.

        Args:
            websocket: The WebSocket server connection.
        """
        cfg = load_config()
        password = cfg.get("remote_password", "")

        # Authentication
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("type") != "auth":
                await websocket.close()
                return
            if password and msg.get("password") != password:
                await websocket.send(json.dumps({"type": "auth_required"}))
                # Give one more chance
                raw2 = await asyncio.wait_for(websocket.recv(), timeout=60)
                msg2 = json.loads(raw2)
                if msg2.get("type") != "auth" or msg2.get("password") != password:
                    await websocket.send(
                        json.dumps({"type": "error", "text": "Authentication failed"})
                    )
                    await websocket.close()
                    return
            await websocket.send(json.dumps({"type": "auth_ok"}))
        except Exception:
            logger.debug("WS auth failed", exc_info=True)
            return

        self._printer.add_client(websocket)
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                except json.JSONDecodeError:
                    continue
                cmd_type = cmd.get("type", "")
                if cmd_type in _VSCODE_ONLY_COMMANDS:
                    continue  # silently ignore VS Code-only commands
                if cmd_type == "ready":
                    await self._handle_ready(cmd, websocket)
                    continue
                if cmd_type == "submit":
                    await self._handle_submit(cmd)
                    continue
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._vscode_server._handle_command, cmd
                )
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logger.debug("WS handler error", exc_info=True)
        finally:
            self._printer.remove_client(websocket)

    # -- Webview command translators -----------------------------------------

    async def _handle_ready(
        self, cmd: dict[str, Any], websocket: ServerConnection,
    ) -> None:
        """Translate the webview ``ready`` command into backend commands.

        The VS Code TypeScript extension intercepts ``ready`` and fans
        it out into ``getModels``, ``getInputHistory``, ``getConfig``,
        plus session replay for restored tabs.  The web server must do
        the same translation since there is no TypeScript middleman.

        Args:
            cmd: The ``ready`` message from the browser.
            websocket: The client connection (for direct replies).
        """
        tab_id = cmd.get("tabId", "")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._vscode_server._handle_command,
            {"type": "getModels"},
        )
        await loop.run_in_executor(
            None,
            self._vscode_server._handle_command,
            {"type": "getInputHistory"},
        )
        await loop.run_in_executor(
            None,
            self._vscode_server._handle_command,
            {"type": "getConfig"},
        )
        # Send focusInput event back to the client
        try:
            await websocket.send(
                json.dumps({"type": "focusInput", "tabId": tab_id})
            )
        except Exception:
            pass
        # Replay restored tabs
        restored = cmd.get("restoredTabs") or []
        for rt in restored:
            chat_id = rt.get("chatId", "")
            rt_tab = rt.get("tabId", "")
            if chat_id:
                await loop.run_in_executor(
                    None,
                    self._vscode_server._handle_command,
                    {"type": "resumeSession", "chatId": chat_id, "tabId": rt_tab},
                )

    async def _handle_submit(self, cmd: dict[str, Any]) -> None:
        """Translate the webview ``submit`` command into a backend ``run``.

        The VS Code TypeScript extension transforms ``submit`` into a
        ``run`` command after resolving paths and tracking running tabs.
        The web server performs the same translation.

        Args:
            cmd: The ``submit`` message from the browser.
        """
        tab_id = cmd.get("tabId", "")
        prompt = cmd.get("prompt", "")
        # Emit status events that the TypeScript extension normally sends
        self._printer.broadcast({"type": "setTaskText", "text": prompt, "tabId": tab_id})
        self._printer.broadcast({"type": "status", "running": True, "tabId": tab_id})
        # Translate submit → run
        run_cmd: dict[str, Any] = {
            "type": "run",
            "prompt": prompt,
            "model": cmd.get("model", ""),
            "workDir": cmd.get("workDir", self._vscode_server.work_dir),
            "tabId": tab_id,
            "attachments": cmd.get("attachments"),
            "useWorktree": cmd.get("useWorktree", False),
            "useParallel": cmd.get("useParallel", False),
        }
        if "skipMerge" in cmd:
            run_cmd["skipMerge"] = cmd["skipMerge"]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._vscode_server._handle_command, run_cmd,
        )

    # -- Tunnel management --------------------------------------------------

    def _start_tunnel(self) -> str | None:
        """Start a ``cloudflared`` quick-tunnel and return the public URL.

        The tunnel process is stored in ``_tunnel_proc`` and must be
        terminated via :meth:`_stop_tunnel`.

        Returns:
            The public ``https://`` URL, or None if tunnel start fails.
        """
        try:
            self._tunnel_proc = subprocess.Popen(
                [
                    "cloudflared",
                    "tunnel",
                    "--url",
                    f"http://localhost:{self.port}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # cloudflared prints the URL to stderr
            import re

            for line in iter(self._tunnel_proc.stderr.readline, ""):  # type: ignore[union-attr]
                match = re.search(r"(https://[^\s]+\.trycloudflare\.com)", line)
                if match:
                    return match.group(1)
                # Stop waiting after the process exits
                if self._tunnel_proc.poll() is not None:
                    break
        except FileNotFoundError:
            logger.warning("cloudflared not found — tunnel not started")
        except Exception:
            logger.debug("Failed to start tunnel", exc_info=True)
        return None

    def _stop_tunnel(self) -> None:
        """Terminate the ``cloudflared`` tunnel process if running."""
        if self._tunnel_proc is not None:
            self._tunnel_proc.terminate()
            try:
                self._tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._tunnel_proc.kill()
            self._tunnel_proc = None

    # -- Server lifecycle ---------------------------------------------------

    async def _serve_async(self) -> None:
        """Internal async entry point for the server."""
        self._loop = asyncio.get_event_loop()
        self._printer._loop = self._loop

        self._ws_server = await serve(
            self._ws_handler,
            self.host,
            self.port,
            process_request=self._process_request,
        )

        local_url = f"http://localhost:{self.port}"
        print(f"KISS Sorcar remote access: {local_url}", file=sys.stderr)

        tunnel_url: str | None = None
        if self.use_tunnel:
            tunnel_url = await asyncio.get_event_loop().run_in_executor(
                None, self._start_tunnel
            )
            if tunnel_url:
                print(f"Cloudflare tunnel:         {tunnel_url}", file=sys.stderr)
            else:
                print(
                    "Warning: cloudflared tunnel failed to start",
                    file=sys.stderr,
                )

        await self._ws_server.serve_forever()

    def start(self) -> None:
        """Start the server (blocks until interrupted).

        Call this from the main thread.  Press Ctrl-C to stop.
        """
        try:
            asyncio.run(self._serve_async())
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_tunnel()

    async def start_async(self) -> None:
        """Start the server asynchronously (for use in existing event loops).

        Returns after the server is listening.  The caller must keep
        the event loop running.
        """
        self._loop = asyncio.get_event_loop()
        self._printer._loop = self._loop

        self._ws_server = await serve(
            self._ws_handler,
            self.host,
            self.port,
            process_request=self._process_request,
        )

    async def stop_async(self) -> None:
        """Stop the server gracefully."""
        if self._ws_server is not None:
            self._ws_server.close()
            try:
                await asyncio.wait_for(self._ws_server.wait_closed(), timeout=2)
            except TimeoutError:
                pass
        self._stop_tunnel()


def main() -> None:  # pragma: no cover — CLI entry point
    """CLI entry point for the remote access server."""
    import argparse

    parser = argparse.ArgumentParser(description="KISS Sorcar Remote Access Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8787, help="Port number")
    parser.add_argument("--tunnel", action="store_true", help="Start cloudflared tunnel")
    parser.add_argument("--workdir", default=None, help="Working directory")
    args = parser.parse_args()

    server = RemoteAccessServer(
        host=args.host,
        port=args.port,
        use_tunnel=args.tunnel,
        work_dir=args.workdir,
    )
    server.start()


if __name__ == "__main__":
    main()
