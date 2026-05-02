"""Standalone web server for remote KISS Sorcar access.

Provides HTTPS + WSS access to the Sorcar chat interface from any
browser, including mobile devices.  Uses the ``websockets`` library to
serve both HTTPS (for the HTML page and static media assets) and
WSS (for bidirectional command/event communication) on a single port.
TLS is always enabled; a self-signed certificate is auto-generated in
``~/.kiss/tls/`` when no explicit certificate is provided.

Authentication uses the ``remote_password`` setting from
``~/.kiss/config.json``.  An optional ``cloudflared`` tunnel can
expose the server through Cloudflare so devices outside the LAN can
connect without manual port-forwarding.

By default (no token), a **quick-tunnel** is used, which assigns a
random ``*.trycloudflare.com`` URL that changes on every restart.  To
get a **fixed** (non-dynamic) URL, create a named tunnel in the
`Cloudflare Zero Trust dashboard <https://one.dash.cloudflare.com/>`_,
copy its token, and set it via the ``CLOUDFLARE_TUNNEL_TOKEN``
environment variable or the ``tunnel_token`` key in
``~/.kiss/config.json``.

Usage::

    # Quick tunnel (random URL, changes on restart):
    server = RemoteAccessServer(port=8787, use_tunnel=True)
    server.start()

    # Named tunnel (fixed URL):
    server = RemoteAccessServer(port=8787, use_tunnel=True,
                                tunnel_token="eyJ...")
    server.start()
"""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import json
import logging
import mimetypes
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.vscode_config import load_config, source_shell_env

__all__ = ["RemoteAccessServer", "WebPrinter"]

logger = logging.getLogger(__name__)

MEDIA_DIR = Path(__file__).parent / "media"

#: How often (in seconds) the unified watchdog runs.  Each tick checks
#: tunnel health, IP address changes, and pings WebSocket clients.
TUNNEL_CHECK_INTERVAL = 30

#: WebSocket pong timeout in seconds.  Connections that fail to respond
#: to a ping within this window are closed as stale.
_WS_PING_TIMEOUT = 10

#: Number of consecutive watchdog ticks the ``cloudflared`` tunnel may
#: have zero ready edge connections before the watchdog force-restarts
#: it.  Cloudflare occasionally drops a quick-tunnel's registration
#: (the local subprocess stays alive but the public hostname stops
#: resolving).  At ``TUNNEL_CHECK_INTERVAL=30`` this allows ~90s of
#: zero edge connections before forcing a restart, which is generous
#: enough to ride out transient connection blips during initial
#: startup or network flaps.
_TUNNEL_UNHEALTHY_LIMIT = 3

#: HTTP 200 response for HEAD health checks from cloudflared.
_HEAD_200 = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Length: 0\r\n"
    b"Connection: close\r\n"
    b"\r\n"
)


class _HeadAwareServerConnection(ServerConnection):
    """``ServerConnection`` subclass that handles HEAD health checks.

    The ``websockets`` library only accepts GET requests (for WebSocket
    upgrade handshakes).  Cloudflare tunnels send HEAD requests to check
    origin health.  Without this handler, those HEAD requests cause
    parse errors, Cloudflare marks the tunnel as unhealthy, and the
    tunnel URL stops resolving (NXDOMAIN).

    Intercepts incoming data before the websockets parser sees it.  If
    the first HTTP request line is ``HEAD …``, responds with 200 OK and
    closes the connection.  All other requests pass through normally.
    """

    def __init__(
        self,
        protocol: Any,
        server: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(protocol, server, **kwargs)
        self._head_buffer: bytes = b""
        self._head_checked: bool = False

    def data_received(self, data: bytes) -> None:
        """Intercept HEAD requests before the websockets parser.

        Buffers incoming bytes until the first HTTP request line is
        complete.  If it starts with ``HEAD ``, writes a 200 OK and
        closes.  Otherwise, feeds all buffered data to the normal
        websockets pipeline.

        Args:
            data: Raw bytes from the transport.
        """
        if self._head_checked:
            super().data_received(data)
            return
        self._head_buffer += data
        idx = self._head_buffer.find(b"\r\n")
        if idx == -1:
            return  # first line not yet complete
        self._head_checked = True
        first_line = self._head_buffer[:idx]
        if first_line.startswith(b"HEAD "):
            transport = self.transport
            if transport is not None:
                transport.write(_HEAD_200)
                transport.close()
            return
        # Not a HEAD request — replay buffered data through normal path
        buffered = self._head_buffer
        self._head_buffer = b""
        super().data_received(buffered)


# ---------------------------------------------------------------------------
# Server-side merge state for web clients
# ---------------------------------------------------------------------------


class _WebMergeState:
    """Tracks merge review state for a single tab in the web server.

    In VS Code, the TypeScript ``MergeManager`` handles per-hunk
    accept/reject by modifying files through the editor API.  Since the
    standalone web server has no editor, this class provides equivalent
    functionality by tracking hunk resolution state and modifying files
    on disk directly.

    Args:
        merge_data: The ``data`` dict from a ``merge_data`` event,
            containing a ``files`` list with ``name``, ``base``,
            ``current``, and ``hunks`` entries.
    """

    def __init__(self, merge_data: dict[str, Any]) -> None:
        self.files: list[dict[str, Any]] = merge_data.get("files", [])
        self._all_hunks: list[tuple[int, int]] = [
            (fi, hi)
            for fi, f in enumerate(self.files)
            for hi in range(len(f.get("hunks", [])))
        ]
        self._pos = 0  # index into _all_hunks
        self._resolved: set[tuple[int, int]] = set()

    @property
    def total_hunks(self) -> int:
        """Total number of hunks across all files."""
        return len(self._all_hunks)

    @property
    def remaining(self) -> int:
        """Number of unresolved hunks."""
        return self.total_hunks - len(self._resolved)

    def current(self) -> tuple[int, int] | None:
        """Return (file_idx, hunk_idx) for the current position, or None."""
        if not self._all_hunks:
            return None
        if self._pos >= len(self._all_hunks):
            self._pos = len(self._all_hunks) - 1
        return self._all_hunks[self._pos]

    def mark_resolved(self, fi: int, hi: int) -> None:
        """Mark a hunk as resolved."""
        self._resolved.add((fi, hi))

    def _seek(self, step: int) -> None:
        """Move *step* (+1 or -1) to the next unresolved hunk."""
        if not self.remaining:
            return
        for _ in range(len(self._all_hunks)):
            self._pos = (self._pos + step) % len(self._all_hunks)
            if self._all_hunks[self._pos] not in self._resolved:
                return

    def advance(self) -> None:
        """Move to the next unresolved hunk."""
        self._seek(1)

    def go_prev(self) -> None:
        """Move to the previous unresolved hunk."""
        self._seek(-1)

    def unresolved_in_file(self, fi: int) -> list[int]:
        """Return hunk indices not yet resolved for file *fi*."""
        return [
            hi
            for ffi, hi in self._all_hunks
            if ffi == fi and (ffi, hi) not in self._resolved
        ]

    def all_unresolved(self) -> list[tuple[int, int]]:
        """Return all (file_idx, hunk_idx) pairs not yet resolved."""
        return [
            (fi, hi) for fi, hi in self._all_hunks if (fi, hi) not in self._resolved
        ]


def _reject_hunk_in_file(
    current_path: str,
    base_path: str,
    hunk: dict[str, int],
) -> None:
    """Revert a single hunk in the current file to the base version.

    Reads both files, replaces the hunk's lines in the current file
    with the corresponding lines from the base file, and writes the
    result back.

    Args:
        current_path: Path to the file with agent changes.
        base_path: Path to the pre-task base copy.
        hunk: Hunk dict with keys ``bs``, ``bc``, ``cs``, ``cc``
            (0-based line positions).
    """
    try:
        cur_lines = Path(current_path).read_text().splitlines(keepends=True)
    except OSError:
        cur_lines = []
    try:
        base_lines = Path(base_path).read_text().splitlines(keepends=True)
    except OSError:
        base_lines = []

    new_lines = (
        cur_lines[: hunk["cs"]]
        + base_lines[hunk["bs"] : hunk["bs"] + hunk["bc"]]
        + cur_lines[hunk["cs"] + hunk["cc"] :]
    )
    Path(current_path).write_text("".join(new_lines))


def _reject_all_hunks_in_file(file_data: dict[str, Any]) -> None:
    """Revert an entire file to its base version.

    Simply copies the base file content over the current file.

    Args:
        file_data: File entry from merge data with ``base`` and
            ``current`` path strings.
    """
    if Path(file_data["base"]).is_file():
        shutil.copy2(file_data["base"], file_data["current"])

#: Commands that are VS-Code-UI-specific and have no backend handler.
_VSCODE_ONLY_COMMANDS = frozenset({

    "focusEditor",
    "webviewFocusChanged",
    "openFile",
    "resolveDroppedPaths",
})

_TLS_DIR = Path.home() / ".kiss" / "tls"
_URL_FILE = Path.home() / ".kiss" / "remote-url.json"


def _discover_tunnel_url_from_metrics() -> str | None:
    """Try to discover the quick-tunnel URL from a running ``cloudflared``.

    Scans running ``cloudflared`` processes for their metrics port, then
    queries the ``/quicktunnel`` endpoint to get the assigned hostname.
    This is a fallback for when ``~/.kiss/remote-url.json`` does not
    exist (e.g. because ``_start_quick_tunnel`` failed to capture the
    URL from stderr).

    Returns:
        The ``https://`` tunnel URL, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-a", "cloudflared"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None

    # Look for --metrics port in command lines, then try default ports.
    # Use dict.fromkeys for insertion-ordered dedup.
    parsed: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        for i, p in enumerate(parts):
            if p == "--metrics" and i + 1 < len(parts):
                try:
                    parsed.append(int(parts[i + 1].rsplit(":", 1)[-1]))
                except (ValueError, IndexError):
                    pass
    metrics_ports = list(dict.fromkeys(parsed + list(range(20240, 20260))))

    for port in metrics_ports:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/quicktunnel",
                headers={"User-Agent": "kiss-web"},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                hostname = data.get("hostname", "")
                if hostname and not hostname.startswith("api."):
                    return f"https://{hostname}"
        except Exception:
            continue
    return None


def _pick_free_local_port() -> int:
    """Return a currently free TCP port on 127.0.0.1.

    Used to pre-assign a fixed ``--metrics`` port to ``cloudflared``
    so the watchdog can probe the same port reliably across restarts.
    There is a small TOCTOU window between releasing the socket and
    cloudflared binding it, but the only consequence is that
    cloudflared may fail to bind, which the watchdog will detect via
    the missing metrics endpoint and recover from on the next cycle.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return port


def _probe_tunnel_ready(metrics_port: int) -> bool:
    """Return True if ``cloudflared`` has at least one live edge connection.

    Queries the ``cloudflared`` ``/ready`` metrics endpoint and parses
    the JSON ``readyConnections`` field.  Cloudflare's edge can
    deregister a quick-tunnel while the local ``cloudflared``
    subprocess is still alive (e.g. after the laptop sleeps for a long
    time, or when Cloudflare rotates a flaky quick-tunnel).  When that
    happens the subprocess keeps retrying ``register_connection`` and
    never reaches a ready state, so the public ``*.trycloudflare.com``
    hostname stops resolving (NXDOMAIN) but the watchdog's
    ``proc.poll()`` check still reports the tunnel as alive.  A zero
    ``readyConnections`` reading is the canonical signal for this
    "process alive but tunnel deregistered" failure mode.

    Args:
        metrics_port: The port on which ``cloudflared`` is serving its
            metrics HTTP endpoint (passed via ``--metrics``).

    Returns:
        True if the endpoint reports at least one ready connection.
        False on any error (timeout, connection refused, parse error,
        zero connections), so the caller can treat it as "unhealthy"
        and increment its consecutive-failure counter.
    """
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{metrics_port}/ready",
            headers={"User-Agent": "kiss-web"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
    except Exception:
        return False
    try:
        return int(data.get("readyConnections", 0)) > 0
    except (TypeError, ValueError):
        return False


def _save_url_file(local_url: str, tunnel_url: str | None = None) -> None:
    """Write the active server URLs to ``~/.kiss/remote-url.json``.

    Creates the parent directory if needed.  The file is read by
    ``kiss-web --url`` so users can discover the remote URL without
    digging through log files.

    Args:
        local_url: The local ``https://localhost:PORT`` URL.
        tunnel_url: The Cloudflare tunnel URL, or None.
    """
    data: dict[str, str] = {"local": local_url}
    if tunnel_url:
        data["tunnel"] = tunnel_url
    _URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _URL_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _remove_url_file() -> None:
    """Delete ``~/.kiss/remote-url.json`` if it exists."""
    try:
        _URL_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _get_local_ips() -> frozenset[str]:
    """Return the current non-loopback IPv4 addresses of the host machine.

    Uses a UDP connect to ``8.8.8.8`` (no packet is actually sent) to
    discover the default-route IP, plus :func:`socket.getaddrinfo` on
    the hostname for any additional addresses.

    Returns:
        A frozen set of IPv4 address strings (e.g.
        ``frozenset({"192.168.1.42"})``).  Returns an empty set when
        no non-loopback addresses are found.
    """
    ips: set[str] = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(1)
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = str(info[4][0])
            if not addr.startswith("127."):
                ips.add(addr)
    except Exception:
        pass
    return frozenset(ips)


def _print_url() -> None:
    """Print the active remote URL from ``~/.kiss/remote-url.json``.

    Prints the tunnel URL if available, otherwise the local URL.
    Exits with code 1 if the server is not running or the file is
    missing.
    """
    try:
        data = json.loads(_URL_FILE.read_text()) if _URL_FILE.is_file() else {}
    except (json.JSONDecodeError, OSError):
        data = {}
    url = data.get("tunnel") or data.get("local")
    if url:
        print(url)
    else:
        print("KISS Sorcar web server is not running.", file=sys.stderr)
        sys.exit(1)


def _generate_self_signed_cert(
    cert_path: Path,
    key_path: Path,
) -> None:
    """Generate a self-signed TLS certificate and private key.

    Creates an RSA 2048-bit key and a self-signed X.509 certificate
    valid for 365 days, covering ``localhost``, ``127.0.0.1``, ``::1``,
    and all ``*.local`` names.  Parent directories are created as needed.

    Args:
        cert_path: Where to write the PEM-encoded certificate.
        key_path: Where to write the PEM-encoded private key.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "KISS Sorcar"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "KISS Sorcar"),
    ])

    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("*.local"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv6Address("::1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    for d in {cert_path.parent, key_path.parent}:
        d.mkdir(parents=True, exist_ok=True)

    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _create_ssl_context(
    certfile: str | None = None,
    keyfile: str | None = None,
) -> ssl.SSLContext:
    """Create an SSL context for the HTTPS/WSS server.

    If *certfile* and *keyfile* are provided, loads them directly.
    Otherwise auto-generates a self-signed certificate in
    ``~/.kiss/tls/`` and uses that.

    Args:
        certfile: Path to PEM certificate file, or None for auto-gen.
        keyfile: Path to PEM private key file, or None for auto-gen.

    Returns:
        A configured ``ssl.SSLContext`` ready for ``websockets.serve()``.
    """
    if certfile and keyfile:
        cert_path = Path(certfile)
        key_path = Path(keyfile)
    else:
        cert_path = _TLS_DIR / "cert.pem"
        key_path = _TLS_DIR / "key.pem"
        if not cert_path.is_file() or not key_path.is_file():
            logger.info("Generating self-signed TLS certificate in %s", _TLS_DIR)
            _generate_self_signed_cert(cert_path, key_path)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert_path), str(key_path))
    return ctx


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
        self._merge_state_callback: (
            Callable[[str, dict[str, Any]], None] | None
        ) = None

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send *event* to every connected WebSocket client.

        Injects ``tabId`` from thread-local storage (via
        ``_inject_tab_id``), records the event for replay, persists
        display events to the database (via ``_persist_event``), and
        augments ``merge_data`` events with file contents for web-based
        diff rendering.

        Args:
            event: The event dictionary to emit.
        """
        event = self._inject_tab_id(event)

        # Augment merge_data with file contents for web clients
        if event.get("type") == "merge_data":
            event = _augment_merge_data(event)
            # Register merge state for the web merge manager
            evt_tab = event.get("tabId", "")
            if evt_tab and self._merge_state_callback is not None:
                self._merge_state_callback(evt_tab, event.get("data", {}))

        with self._lock:
            self._record_event(event)

        self._persist_event(event)

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
  <meta name="viewport" content="\
width=device-width,initial-scale=1,maximum-scale=1">
  <link href="/media/main.css" rel="stylesheet">
  <link href="/media/highlight-github-dark.min.css" rel="stylesheet">
  <title>KISS Sorcar</title>
  <style>
    html, body {{ height: 100%; margin: 0; padding: 0; overflow: hidden; }}
    body {{ background: var(--vscode-editor-background, #1e1e1e);
            color: var(--vscode-editor-foreground, #cccccc); }}
    :root {{
      --vscode-font-size: 13px;
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
      --vscode-terminal-ansiRed: #f44747;
      --vscode-terminal-ansiGreen: #6a9955;
      --vscode-terminal-ansiYellow: #d7ba7d;
      --vscode-terminal-ansiBlue: #569cd6;
      --vscode-terminal-ansiMagenta: #c586c0;
      --vscode-terminal-ansiCyan: #4ec9b0;
    }}
  </style>
</head>
<body class="remote-chat">
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
             placeholder="Ask anything... (@ for files)" rows="1"
             enterkeyhint="send"></textarea>
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
            <button id="menu-btn" data-tooltip="Advanced options">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round">
                <line x1="3" y1="6" x2="21" y2="6"/>
                <line x1="3" y1="12" x2="21" y2="12"/>
                <line x1="3" y1="18" x2="21" y2="18"/>
              </svg>
            </button>
            <div id="menu-dropdown">
              <button id="worktree-toggle-btn" class="menu-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round"
                 stroke-linejoin="round">
                  <line x1="6" y1="3" x2="6" y2="15"/>
                  <circle cx="18" cy="6" r="3"/>
                  <circle cx="6" cy="18" r="3"/>
                  <path d="M18 9a9 9 0 01-9 9"/>
                </svg>
                <span>Use worktree</span>
              </button>
              <button id="parallel-toggle-btn" class="menu-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round"
                 stroke-linejoin="round">
                  <line x1="6" y1="4" x2="6" y2="20"/>
                  <line x1="12" y1="4" x2="12" y2="20"/>
                  <line x1="18" y1="4" x2="18" y2="20"/>
                </svg>
                <span>Use parallelism</span>
              </button>
              <button id="autocommit-btn" class="menu-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round"
                 stroke-linejoin="round">
                  <circle cx="12" cy="12" r="4"/>
                  <line x1="1.05" y1="12" x2="7" y2="12"/>
                  <line x1="17.01" y1="12" x2="22.96" y2="12"/>
                  <line x1="12" y1="1.05" x2="12" y2="7"/>
                  <line x1="12" y1="17.01" x2="12" y2="22.96"/>
                </svg>
                <span>Auto commit</span>
              </button>
              <button id="demo-toggle-btn" class="menu-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round"
                 stroke-linejoin="round">
                  <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
                  <line x1="8" y1="21" x2="16" y2="21"/>
                  <line x1="12" y1="17" x2="12" y2="21"/>
                </svg>
                <span>Demo mode</span>
              </button>
            </div>
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
        <div id="remote-url"></div>
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
    """Read the KISS project version from ``_version.py``."""
    try:
        vfile = Path(__file__).parent.parent.parent / "_version.py"
        for line in vfile.read_text().splitlines():
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
    _ws = new WebSocket('wss://' + location.host + '/ws');
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
    return Response(
        status,
        "OK" if status == 200 else "Not Found",
        Headers([
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Connection", "close"),
        ]),
        body,
    )


def _augment_merge_data(event: dict[str, Any]) -> dict[str, Any]:
    """Add ``base_text`` and ``current_text`` to each file in a ``merge_data`` event.

    The browser needs file contents to render diffs.  In VS Code, the
    ``MergeManager`` reads files through the editor API; in the web
    server we read them from disk and include the text in the event.

    Args:
        event: A ``merge_data`` event dict.

    Returns:
        A copy of the event with file contents added.
    """
    event = {**event}
    data = {**event.get("data", {})}
    files = []
    for f in data.get("files", []):
        f = {**f}
        try:
            f["base_text"] = Path(f["base"]).read_text()
        except (OSError, KeyError):
            f["base_text"] = ""
        try:
            f["current_text"] = Path(f["current"]).read_text()
        except (OSError, KeyError):
            f["current_text"] = ""
        files.append(f)
    data["files"] = files
    event["data"] = data
    return event


def _translate_webview_command(cmd: dict[str, Any]) -> dict[str, Any]:
    """Translate a webview message into a backend command.

    The VS Code TypeScript extension (``SorcarSidebarView``) intercepts
    messages from the webview and rewrites several of them before
    forwarding to the Python backend.  This function performs the same
    translations so the standalone web server can relay messages
    directly.

    Translations applied:

    * ``userActionDone`` → ``userAnswer`` with ``answer="done"``
    * ``resumeSession`` → renames ``id`` field to ``chatId``

    Args:
        cmd: Raw command dictionary from the browser WebSocket.

    Returns:
        The (possibly modified) command dictionary ready for
        ``VSCodeServer._handle_command``.
    """
    cmd_type = cmd.get("type", "")
    if cmd_type == "userActionDone":
        return {"type": "userAnswer", "answer": "done", "tabId": cmd.get("tabId", "")}
    if cmd_type == "resumeSession" and "id" in cmd and "chatId" not in cmd:
        out = dict(cmd)
        out["chatId"] = out.pop("id")
        return out
    return cmd


# ---------------------------------------------------------------------------
# RemoteAccessServer
# ---------------------------------------------------------------------------


class RemoteAccessServer:
    """Web server providing remote browser access to KISS Sorcar.

    Serves the Sorcar chat webview over HTTPS and bridges commands/events
    over WSS.  TLS is always enabled; a self-signed certificate is
    auto-generated in ``~/.kiss/tls/`` when *certfile*/*keyfile* are not
    provided.  Optionally starts a ``cloudflared`` tunnel so the server
    is reachable from the public internet without manual port-forwarding
    or DNS setup.

    When *tunnel_token* is provided, a **named tunnel** is used, giving
    a fixed URL that persists across restarts.  Without a token, a
    quick-tunnel is created with a random ``*.trycloudflare.com`` URL.

    Args:
        host: Bind address (default ``"0.0.0.0"`` for all interfaces).
        port: TCP port for both HTTPS and WSS (default ``8787``).
        use_tunnel: If True, start a ``cloudflared`` tunnel on launch.
        tunnel_token: Cloudflare named-tunnel token for a fixed URL.
            When set, ``cloudflared tunnel run --token <TOKEN>`` is
            used instead of a quick-tunnel.
        work_dir: Working directory for the agent (default cwd).
        certfile: Path to a PEM certificate file for TLS.
        keyfile: Path to a PEM private key file for TLS.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8787,
        use_tunnel: bool = False,
        tunnel_token: str | None = None,
        work_dir: str | None = None,
        certfile: str | None = None,
        keyfile: str | None = None,
    ) -> None:
        source_shell_env()

        self.host = host
        self.port = port
        self.use_tunnel = use_tunnel
        self.tunnel_token = tunnel_token
        self._ssl_context: ssl.SSLContext = _create_ssl_context(certfile, keyfile)

        if work_dir:
            os.environ["KISS_WORKDIR"] = work_dir

        self._vscode_server = VSCodeServer()
        self._printer = WebPrinter()
        self._vscode_server.printer = self._printer  # type: ignore[assignment]

        self._html_bytes = _build_html().encode("utf-8")
        self._tunnel_proc: subprocess.Popen[str] | None = None
        #: Port assigned to ``cloudflared --metrics`` so the watchdog
        #: can probe ``/ready`` and detect edge deregistrations.
        self._tunnel_metrics_port: int | None = None
        #: Number of consecutive watchdog ticks the tunnel has had
        #: zero ready edge connections.  When this reaches
        #: :data:`_TUNNEL_UNHEALTHY_LIMIT`, the watchdog force-restarts
        #: the ``cloudflared`` subprocess.
        self._tunnel_unhealthy_ticks = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_server: Any = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._local_url = f"https://localhost:{self.port}"
        self._merge_states: dict[str, _WebMergeState] = {}
        self._printer._merge_state_callback = self._register_merge_state
        self._active_url: str | None = None
        self._last_ips: frozenset[str] = frozenset()

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
        if path == "/ws":
            return None  # proceed to WebSocket
        if path.startswith("/media/"):
            filepath = MEDIA_DIR / path[7:]
            if (
                filepath.resolve().is_relative_to(MEDIA_DIR.resolve())
                and filepath.is_file()
            ):
                ctype = mimetypes.guess_type(str(filepath))[0] or "application/octet-stream"
                return _http_response(200, ctype, filepath.read_bytes())
        return _http_response(404, "text/plain", b"Not Found")

    # -- WebSocket handler --------------------------------------------------

    async def _authenticate_ws(self, websocket: ServerConnection) -> bool:
        """Authenticate a WebSocket client using the configured password.

        Returns True on success, False (and closes the socket) on failure.
        """
        password = load_config().get("remote_password", "")
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("type") != "auth":
                await websocket.close()
                return False
            if password and msg.get("password") != password:
                await websocket.send(json.dumps({"type": "auth_required"}))
                raw2 = await asyncio.wait_for(websocket.recv(), timeout=60)
                msg2 = json.loads(raw2)
                if msg2.get("type") != "auth" or msg2.get("password") != password:
                    await websocket.send(
                        json.dumps({"type": "error", "text": "Authentication failed"})
                    )
                    await websocket.close()
                    return False
            await websocket.send(json.dumps({"type": "auth_ok"}))
            return True
        except Exception:
            logger.debug("WS auth failed", exc_info=True)
            return False

    async def _run_cmd(self, cmd: dict[str, Any]) -> None:
        """Run a backend command in the thread-pool executor."""
        assert self._loop is not None
        await self._loop.run_in_executor(
            None, self._vscode_server._handle_command, cmd,
        )

    async def _ws_handler(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket client connection.

        Performs password authentication, then relays messages between
        the browser and the ``VSCodeServer`` command dispatcher.

        Args:
            websocket: The WebSocket server connection.
        """
        if not await self._authenticate_ws(websocket):
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
                if cmd_type == "getWelcomeSuggestions":
                    self._send_welcome_info()
                    continue
                if cmd_type == "mergeAction":
                    action = cmd.get("action", "")
                    if action != "all-done":
                        await self._handle_web_merge_action(cmd)
                        continue
                    # "all-done" falls through to backend
                # Translate webview-only fields that the TypeScript
                # extension normally rewrites before they reach the
                # Python backend.
                cmd = _translate_webview_command(cmd)
                await self._run_cmd(cmd)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logger.debug("WS handler error", exc_info=True)
        finally:
            self._printer.remove_client(websocket)

    # -- Webview command translators -----------------------------------------

    def _send_welcome_info(self) -> None:
        """Broadcast welcome suggestions and the active remote URL.

        Broadcasts a ``welcome_suggestions`` event with an empty list
        because the remote chat webview deliberately suppresses sample
        task suggestions on the welcome page (the centered input
        textbox is the only welcome-page UI).  Then broadcasts the
        ``remote_url`` event using the in-memory URL, the URL file, or
        the ``cloudflared`` metrics API as successive fallbacks.
        """
        self._printer.broadcast({
            "type": "welcome_suggestions", "suggestions": [],
        })

        url: str | None = self._active_url
        if not url:
            try:
                data = json.loads(_URL_FILE.read_text())
                url = data.get("tunnel") or data.get("local", "")
            except Exception:
                pass
        if not url:
            url = _discover_tunnel_url_from_metrics()
            if url:
                _save_url_file(self._local_url, url)
                self._active_url = url
        if url:
            self._printer.broadcast({"type": "remote_url", "url": url})

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
        for init_cmd in ("getModels", "getInputHistory", "getConfig"):
            await self._run_cmd({"type": init_cmd})
        self._send_welcome_info()
        try:
            await websocket.send(
                json.dumps({"type": "focusInput", "tabId": tab_id})
            )
        except Exception:
            pass
        for rt in cmd.get("restoredTabs") or []:
            chat_id = rt.get("chatId", "")
            if chat_id:
                await self._run_cmd(
                    {"type": "resumeSession", "chatId": chat_id,
                     "tabId": rt.get("tabId", "")},
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
        await self._run_cmd(run_cmd)

    def _register_merge_state(
        self, tab_id: str, merge_data: dict[str, Any],
    ) -> None:
        """Register a merge state when a merge_data event is broadcast.

        Called from ``WebPrinter.broadcast()`` so the web server can
        track active merge sessions and handle ``mergeAction`` commands.

        Args:
            tab_id: The tab that started the merge.
            merge_data: The ``data`` field from the ``merge_data`` event.
        """
        self._merge_states[tab_id] = _WebMergeState(merge_data)

    # -- Merge action handling for web clients --------------------------------

    async def _handle_web_merge_action(self, cmd: dict[str, Any]) -> None:
        """Handle merge toolbar actions (accept/reject/navigate) server-side.

        In VS Code, the TypeScript ``MergeManager`` processes these
        actions.  In the standalone web server, this method provides
        equivalent functionality by tracking hunk state and modifying
        files on disk.

        Args:
            cmd: The ``mergeAction`` command from the browser, with
                ``action`` and ``tabId`` fields.
        """
        action = cmd.get("action", "")
        tab_id = cmd.get("tabId", "")
        state = self._merge_states.get(tab_id)
        if state is None:
            return  # no active merge for this tab

        assert self._loop is not None
        cur = state.current()
        if action == "accept":
            if cur is not None:
                state.mark_resolved(*cur)
                state.advance()
        elif action == "reject":
            if cur is not None:
                fi, hi = cur
                fd = state.files[fi]
                hunk = fd["hunks"][hi]
                await self._loop.run_in_executor(
                    None, _reject_hunk_in_file, fd["current"], fd["base"], hunk,
                )
                # Adjust subsequent hunks' cs offsets in the same file
                delta = hunk["bc"] - hunk["cc"]
                for later_hi in range(hi + 1, len(fd["hunks"])):
                    if (fi, later_hi) not in state._resolved:
                        fd["hunks"][later_hi]["cs"] += delta
                state.mark_resolved(fi, hi)
                state.advance()
        elif action == "prev":
            state.go_prev()
        elif action == "next":
            state.advance()
        elif action in ("accept-file", "reject-file"):
            if cur is not None:
                fi = cur[0]
                fd = state.files[fi]
                if action == "reject-file":
                    await self._loop.run_in_executor(
                        None, _reject_all_hunks_in_file, fd,
                    )
                for hi in state.unresolved_in_file(fi):
                    state.mark_resolved(fi, hi)
                state.advance()
        elif action == "accept-all":
            for fi, hi in state.all_unresolved():
                state.mark_resolved(fi, hi)
        elif action == "reject-all":
            # Group unresolved hunks by file and reject whole files
            unresolved_files: set[int] = set()
            for fi, hi in state.all_unresolved():
                unresolved_files.add(fi)
                state.mark_resolved(fi, hi)
            for fi in unresolved_files:
                fd = state.files[fi]
                await self._loop.run_in_executor(
                    None, _reject_all_hunks_in_file, fd,
                )

        # Broadcast navigation update
        self._printer.broadcast({
            "type": "merge_nav",
            "tabId": tab_id,
            "remaining": state.remaining,
            "total": state.total_hunks,
        })

        # When all hunks resolved, finish the merge via backend
        if not state.remaining:
            del self._merge_states[tab_id]
            await self._run_cmd(
                {"type": "mergeAction", "action": "all-done", "tabId": tab_id},
            )

    # -- Tunnel management --------------------------------------------------

    def _start_tunnel(self) -> str | None:
        """Start a ``cloudflared`` tunnel and return the public URL.

        When :attr:`tunnel_token` is set, a **named tunnel** is started
        with ``cloudflared tunnel run --token <TOKEN>``.  The URL is
        pre-configured in the Cloudflare Zero Trust dashboard so it
        stays fixed across restarts.  The method reads the connector ID
        from the process output and returns the configured hostname (or
        a confirmation string when the hostname cannot be determined
        from logs).

        When no token is set, a **quick-tunnel** is started with
        ``cloudflared tunnel --url``, which assigns a random
        ``*.trycloudflare.com`` URL.

        The tunnel process is stored in ``_tunnel_proc`` and must be
        terminated via :meth:`_stop_tunnel`.

        Returns:
            The public ``https://`` URL, or None if tunnel start fails.
        """
        try:
            if self.tunnel_token:
                return self._start_named_tunnel()
            return self._start_quick_tunnel()
        except FileNotFoundError:
            logger.warning("cloudflared not found — tunnel not started")
        except Exception:
            logger.debug("Failed to start tunnel", exc_info=True)
        return None

    def _start_quick_tunnel(self) -> str | None:
        """Start a quick-tunnel (random ``*.trycloudflare.com`` URL).

        Starts ``cloudflared tunnel --url`` and attempts to capture the
        assigned URL.  First tries parsing stderr for up to 30 seconds.
        If that fails (e.g. output format changed), falls back to
        querying the ``cloudflared`` metrics API ``/quicktunnel``
        endpoint.

        Returns:
            The public ``https://`` URL, or None on failure.
        """
        self._tunnel_metrics_port = _pick_free_local_port()
        self._tunnel_proc = subprocess.Popen(
            [
                "cloudflared",
                "tunnel",
                "--metrics",
                f"127.0.0.1:{self._tunnel_metrics_port}",
                "--url",
                self._local_url,
                "--no-tls-verify",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Try to capture the URL from stderr with a timeout.
        # cloudflared prints the URL during startup, but the format
        # can vary across versions.  After it finishes printing
        # startup messages, readline() blocks forever.
        stderr_fd = self._tunnel_proc.stderr
        assert stderr_fd is not None  # guaranteed by PIPE
        result_box: list[str | None] = [None]

        def _reader_target() -> None:
            for line in iter(stderr_fd.readline, ""):
                match = re.search(
                    r"(https://(?!api\.)[^\s]+\.trycloudflare\.com)", line,
                )
                if match:
                    result_box[0] = match.group(1)
                    return
                if self._tunnel_proc is None:
                    break
                if self._tunnel_proc.poll() is not None:
                    break

        reader = threading.Thread(target=_reader_target, daemon=True)
        reader.start()
        reader.join(timeout=30)

        url = result_box[0]
        if url:
            return url

        # Fallback: poll the cloudflared metrics API
        for _ in range(20):
            if self._tunnel_proc.poll() is not None:
                break
            url = _discover_tunnel_url_from_metrics()
            if url:
                return url
            time.sleep(1)

        return None

    def _start_named_tunnel(self) -> str | None:
        """Start a named tunnel using :attr:`tunnel_token`.

        The tunnel hostname is configured in the Cloudflare Zero Trust
        dashboard.  ``cloudflared`` logs the registered hostname(s) to
        stderr during startup; this method captures that output to
        return the URL.

        Returns:
            The public ``https://`` URL, or None on failure.
        """
        self._tunnel_metrics_port = _pick_free_local_port()
        self._tunnel_proc = subprocess.Popen(
            [
                "cloudflared",
                "tunnel",
                "--metrics",
                f"127.0.0.1:{self._tunnel_metrics_port}",
                "run",
                "--token",
                self.tunnel_token or "",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for line in iter(self._tunnel_proc.stderr.readline, ""):  # type: ignore[union-attr]
            # Named tunnels log the ingress hostname, e.g.:
            #   "... Connection ... registered connIndex=0 ..."
            #   "... config ... hostname=myapp.example.com ..."
            match = re.search(r"https?://([^\s/]+)", line)
            if match:
                hostname = match.group(1)
                # Ignore localhost/internal URLs
                if "localhost" not in hostname and "127.0.0.1" not in hostname:
                    url = f"https://{hostname}"
                    return url
            # Also look for explicit "Registered tunnel connection" as
            # a sign the tunnel is live (hostname may not appear in logs
            # for connector-protocol tunnels).
            if "Registered tunnel connection" in line or "Connection registered" in line:
                # Tunnel is running; the hostname is configured in the
                # dashboard, not echoed.  Return a sentinel so the
                # caller knows the tunnel started successfully.
                return "(named tunnel running — URL configured in Cloudflare dashboard)"
            if self._tunnel_proc.poll() is not None:
                break
        return None

    async def _check_and_restart_tunnel(self) -> None:
        """Check tunnel health and restart if dead or deregistered.

        Called periodically by :meth:`_watchdog`.  Detects two failure
        modes:

        1. **Process dead** — ``cloudflared`` exited (e.g. because
           macOS killed it during sleep).  Detected via ``poll()``.
        2. **Process alive but tunnel deregistered** — Cloudflare's
           edge has dropped this quick-tunnel's registration so the
           public hostname stops resolving (NXDOMAIN), but the local
           subprocess keeps retrying ``register_connection`` and
           never reconnects.  Detected by polling the ``cloudflared``
           ``/ready`` metrics endpoint for ``readyConnections > 0``.
           After :data:`_TUNNEL_UNHEALTHY_LIMIT` consecutive ticks of
           zero ready connections, the subprocess is force-terminated
           and a fresh tunnel is started, which triggers a new
           ``*.trycloudflare.com`` URL.
        """
        if self._tunnel_proc is None:
            return
        if self._tunnel_proc.poll() is not None:
            rc = self._tunnel_proc.returncode
            logger.info(
                "cloudflared tunnel process died (rc=%s), restarting…", rc,
            )
            self._tunnel_proc = None
            self._tunnel_metrics_port = None
            self._tunnel_unhealthy_ticks = 0
            await self._restart_tunnel_url()
            return
        # Process is alive.  Probe the metrics endpoint to confirm the
        # tunnel is actually registered with Cloudflare's edge.  Skip
        # this check if no metrics port has been recorded (e.g. legacy
        # subprocess started without --metrics, or the tunnel is still
        # in its very first startup window where readyConnections may
        # be 0 transiently).
        if self._tunnel_metrics_port is None:
            return
        assert self._loop is not None
        healthy = await self._loop.run_in_executor(
            None, _probe_tunnel_ready, self._tunnel_metrics_port,
        )
        if healthy:
            self._tunnel_unhealthy_ticks = 0
            return
        self._tunnel_unhealthy_ticks += 1
        logger.info(
            "cloudflared tunnel reports zero ready edge connections "
            "(tick %d/%d on metrics port %d)",
            self._tunnel_unhealthy_ticks,
            _TUNNEL_UNHEALTHY_LIMIT,
            self._tunnel_metrics_port,
        )
        if self._tunnel_unhealthy_ticks < _TUNNEL_UNHEALTHY_LIMIT:
            return
        logger.warning(
            "cloudflared tunnel appears deregistered from Cloudflare's "
            "edge (readyConnections=0 for %d ticks); force-restarting",
            self._tunnel_unhealthy_ticks,
        )
        self._tunnel_unhealthy_ticks = 0
        self._terminate_tunnel_proc()
        await self._restart_tunnel_url()

    async def _restart_tunnel_url(self) -> None:
        """Start a fresh tunnel and refresh ``~/.kiss/remote-url.json``.

        Helper shared by both restart paths in
        :meth:`_check_and_restart_tunnel` (process-died and
        edge-deregistered).  Always rewrites the URL file even when
        the new tunnel fails to start, so stale data does not linger.
        """
        assert self._loop is not None
        tunnel_url = await self._loop.run_in_executor(
            None, self._start_tunnel,
        )
        if tunnel_url:
            logger.info("Tunnel restarted: %s", tunnel_url)
        else:
            logger.warning("Failed to restart tunnel")
        _save_url_file(self._local_url, tunnel_url)
        self._active_url = tunnel_url or self._local_url
        self._printer.broadcast(
            {"type": "remote_url", "url": self._active_url},
        )

    def _terminate_tunnel_proc(self) -> None:
        """Terminate ``self._tunnel_proc`` without touching the URL file.

        Used by :meth:`_check_and_restart_tunnel` when the subprocess
        is alive but has been deregistered by Cloudflare's edge.
        Unlike :meth:`_stop_tunnel`, this leaves ``_active_url`` in
        place so that ``~/.kiss/remote-url.json`` is not removed before
        the replacement tunnel writes its own URL.
        """
        proc = self._tunnel_proc
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        self._tunnel_proc = None
        self._tunnel_metrics_port = None

    async def _ping_one_ws(self, ws: Any) -> None:
        """Send a ping to a single WebSocket client, closing if stale."""
        try:
            pong = await ws.ping()
            await asyncio.wait_for(pong, timeout=_WS_PING_TIMEOUT)
        except Exception:
            try:
                await ws.close()
            except Exception:
                pass

    async def _watchdog(self) -> None:
        """Unified periodic watchdog (runs every :data:`TUNNEL_CHECK_INTERVAL`).

        Each tick performs three checks:

        1. **Tunnel health** — if the ``cloudflared`` process died
           (e.g. macOS killed it during sleep), restart it.
        2. **IP change** — if the host's network addresses changed
           (WiFi switch, DHCP renewal, VPN), initiate a graceful
           shutdown so the daemon manager restarts the process.
        3. **WebSocket ping** — send a ping to every connected client
           and close connections that fail to respond within
           :data:`_WS_PING_TIMEOUT` seconds.
        """
        while True:
            await asyncio.sleep(TUNNEL_CHECK_INTERVAL)
            # 1. Tunnel health
            if self.use_tunnel:
                try:
                    await self._check_and_restart_tunnel()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("Watchdog tunnel check error", exc_info=True)
            # 2. IP change
            try:
                current_ips = _get_local_ips()
                if current_ips != self._last_ips:
                    logger.info(
                        "IP address changed: %s → %s, restarting server…",
                        self._last_ips,
                        current_ips,
                    )
                    self._last_ips = current_ips
                    if self._ws_server is not None:
                        self._ws_server.close()
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog IP check error", exc_info=True)
            # 3. WebSocket ping
            try:
                if self._ws_server is not None:
                    connections = list(self._ws_server.connections)
                    if connections:
                        await asyncio.gather(
                            *[self._ping_one_ws(ws) for ws in connections],
                            return_exceptions=True,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog WS ping error", exc_info=True)

    def _stop_tunnel(self) -> None:
        """Terminate the ``cloudflared`` tunnel process if running."""
        self._terminate_tunnel_proc()
        self._tunnel_unhealthy_ticks = 0
        # Do NOT delete the URL file on shutdown.  A new daemon
        # instance may have already overwritten it, and deleting it
        # would race with the new instance's _save_url_file().  Stale
        # data in the file is harmless — the next instance will
        # overwrite it — while a missing file causes the VS Code
        # sidebar to show no URL at all.
        self._active_url = None

    # -- Server lifecycle ---------------------------------------------------

    async def _setup_server(self) -> None:
        """Shared setup for both blocking and async server start.

        Binds the WebSocket server, starts the tunnel (if enabled),
        saves the URL file, and starts watchdog tasks.
        """
        self._loop = asyncio.get_event_loop()
        self._printer._loop = self._loop

        self._ws_server = await serve(
            self._ws_handler,
            self.host,
            self.port,
            process_request=self._process_request,
            ssl=self._ssl_context,
            ping_interval=None,
            ping_timeout=None,
            create_connection=_HeadAwareServerConnection,
        )

        tunnel_url: str | None = None
        if self.use_tunnel:
            tunnel_url = await self._loop.run_in_executor(  # type: ignore[union-attr]
                None, self._start_tunnel,
            )

        _save_url_file(self._local_url, tunnel_url)
        self._active_url = tunnel_url or self._local_url

        self._last_ips = _get_local_ips()
        self._watchdog_task = asyncio.create_task(self._watchdog())

    async def _serve_async(self) -> None:
        """Internal async entry point for the server."""
        await self._setup_server()
        print(f"KISS Sorcar remote access: {self._local_url}", file=sys.stderr)
        if self.use_tunnel and self._active_url != self._local_url:
            print(f"Cloudflare tunnel:         {self._active_url}", file=sys.stderr)
        elif self.use_tunnel:
            print("Warning: cloudflared tunnel failed to start", file=sys.stderr)
        await self._ws_server.serve_forever()  # type: ignore[union-attr]

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
        await self._setup_server()

    async def stop_async(self) -> None:
        """Stop the server gracefully."""
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None
        if self._ws_server is not None:
            self._ws_server.close()
            try:
                await asyncio.wait_for(self._ws_server.wait_closed(), timeout=2)
            except TimeoutError:
                pass
        self._stop_tunnel()
        # Graceful shutdown: remove URL marker file so kiss-web detection
        # correctly reports the daemon as stopped.  Crash paths (where
        # stop_async is not invoked) leave the file in place — this is
        # handled by _stop_tunnel's "do not delete" policy.
        _remove_url_file()


def main() -> None:  # pragma: no cover — CLI entry point
    """CLI entry point for the remote access server."""
    import argparse

    parser = argparse.ArgumentParser(description="KISS Sorcar Remote Access Server")
    parser.add_argument(
        "--url", action="store_true",
        help="Print the active remote URL and exit",
    )
    parser.add_argument("--workdir", default=None, help="Working directory")
    args = parser.parse_args()

    if args.url:
        _print_url()
        return

    # Resolve tunnel token: env var > config file
    tunnel_token = os.environ.get("CLOUDFLARE_TUNNEL_TOKEN")
    if not tunnel_token:
        cfg = load_config()
        tunnel_token = cfg.get("tunnel_token")

    server = RemoteAccessServer(
        use_tunnel=True,
        tunnel_token=tunnel_token or None,
        work_dir=args.workdir,
    )
    server.start()


if __name__ == "__main__":
    main()
