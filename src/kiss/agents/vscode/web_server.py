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
import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable
from concurrent.futures import Future as ConcurrentFuture
from functools import partial
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

TUNNEL_CHECK_INTERVAL = 30

_WS_PING_TIMEOUT = 10

_TUNNEL_UNHEALTHY_LIMIT_NAMED = 3

# Quick tunnels get a new random URL on every restart, so be much more
# conservative before force-restarting.  20 ticks × 30s = 10 minutes of
# readyConnections=0 before the watchdog kills cloudflared.  This gives
# cloudflared ample time to re-register a temporarily-dropped tunnel
# without burning through dozens of distinct *.trycloudflare.com URLs.
_TUNNEL_UNHEALTHY_LIMIT_QUICK = 20

_TUNNEL_STARTUP_GRACE = 120

_TUNNEL_BACKOFF_INITIAL = 60

_TUNNEL_BACKOFF_MAX = 1800

# When cloudflared's stderr reports HTTP 429 / Cloudflare error code
# 1015 ("rate-limited") on the trycloudflare.com quick-tunnel API, the
# normal exponential backoff (60s → 120s → ...) is far too aggressive:
# every retry within the cooldown window resets Cloudflare's per-IP
# clock, so the tunnel can stay unreachable for *hours* while burning
# through dozens of distinct *.trycloudflare.com URLs.  When a
# rate-limit signal is detected we use a much longer baseline plus a
# random jitter so a fleet of restarts (e.g. across machines on the
# same egress IP) does not synchronise into another rate-limit burst.
_TUNNEL_RATE_LIMIT_BACKOFF = 900  # 15 minutes

_TUNNEL_RATE_LIMIT_JITTER = 300  # 0..5 minutes additional jitter

# Substrings (case-insensitive) in cloudflared's stderr that indicate
# trycloudflare.com is rate-limiting the local egress IP.  Matching is
# done line-by-line via :func:`_is_rate_limit_line`.
_RATE_LIMIT_INDICATORS = (
    "error code: 1015",
    "error code 1015",
    "429 too many requests",
    'status_code="429',
    "status_code=429",
    "rate-limited",
    "rate limited",
)

# Auth rate-limiting (per source IP).  After _AUTH_FAIL_MAX failures
# within _AUTH_FAIL_WINDOW seconds, new connections from that IP are
# refused for _AUTH_LOCKOUT seconds.
_AUTH_FAIL_MAX = 5

_AUTH_FAIL_WINDOW = 60.0

_AUTH_LOCKOUT = 60.0

# M7: client-supplied lists are clamped to these sizes to bound the
# work the server does per command.  An authenticated-but-malicious or
# buggy client cannot make the server resume thousands of sessions or
# attach thousands of files in a single submit.
_MAX_RESTORED_TABS = 32

_MAX_ATTACHMENTS = 32

# M7: cap the prompt size echoed back via ``setTaskText`` so a giant
# JSON payload cannot push tens of MB through the broadcast pipeline.
_MAX_PROMPT_BYTES = 1_000_000

# Maximum length of a single newline-delimited JSON command line read
# from a UDS or WSS client.  The default ``asyncio.StreamReader`` limit
# is 64 KiB which is smaller than even a modest base64-encoded image
# attachment — a single ``submit`` with an attached PNG/PDF would
# overflow the reader, raise ``LimitOverrunError`` and silently drop
# the user's task.  Bumped to 64 MiB so attachments up to a few MB
# each (capped by ``_MAX_ATTACHMENTS``) can be delivered as one line.
_MAX_LINE_BYTES = 64 * 1024 * 1024

# Grace period (seconds) between a WebSocket connection dropping and
# the deferred ``closeTab`` actually firing for every tab seen on
# that connection.  Browsers do not reliably send a "closeTab" before
# the WSS shuts down (and ``beforeunload``/``pagehide`` WebSocket
# writes are routinely dropped), so a brief grace window is the
# canonical way to distinguish a transient reconnect/reload from an
# intentional browser close.  A fresh ``ready`` message that
# re-claims a tab id (current ``tabId`` or any entry in
# ``restoredTabs``) cancels the pending close before it fires.  10s
# is long enough to cover typical reloads on a slow link without
# letting an orphaned ``_RunningAgentState`` linger meaningfully.
_TAB_CLOSE_GRACE = 10.0

# Path to the localhost Unix-domain socket exposed by
# :class:`RemoteAccessServer` in addition to the public WSS port.
# Local clients (the VS Code extension) connect to this socket over
# the SAME newline-delimited JSON protocol that browsers speak over
# WSS — no password challenge is performed because POSIX filesystem
# permissions (mode 0o600) restrict access to the owning user.  A
# fresh ``RemoteAccessServer(uds_path=...)`` argument overrides this
# location for tests so multiple instances do not race on the same
# socket file.
_UDS_PATH = Path.home() / ".kiss" / "sorcar.sock"


def _tunnel_backoff_delay(failure_count: int) -> int:
    """Return the backoff delay for *failure_count* consecutive failures.

    The first failure delays by :data:`_TUNNEL_BACKOFF_INITIAL` seconds
    and each additional failure doubles the delay, capped at
    :data:`_TUNNEL_BACKOFF_MAX`.  A *failure_count* of zero returns
    zero (no backoff).

    Args:
        failure_count: Number of consecutive failures observed.

    Returns:
        Seconds to wait before the next restart attempt.
    """
    if failure_count <= 0:
        return 0
    delay: int = _TUNNEL_BACKOFF_INITIAL * (2 ** (failure_count - 1))
    return min(delay, _TUNNEL_BACKOFF_MAX)


def _is_rate_limit_line(line: str) -> bool:
    """Return True if *line* indicates Cloudflare rate-limiting.

    Matches the substrings in :data:`_RATE_LIMIT_INDICATORS`
    case-insensitively.  A typical rate-limited cloudflared stderr
    line looks like::

        ERR Error unmarshaling QuickTunnel response: error code: 1015
            error="invalid character 'e' ..."  status_code="429 Too Many Requests"

    Args:
        line: A single line of cloudflared stderr.

    Returns:
        True when the line names HTTP 429 or Cloudflare error 1015.
    """
    low = line.lower()
    return any(ind in low for ind in _RATE_LIMIT_INDICATORS)


def _rate_limit_backoff_seconds() -> int:
    """Return the backoff to apply after a rate-limited tunnel attempt.

    Uses :data:`_TUNNEL_RATE_LIMIT_BACKOFF` as the floor and adds a
    cryptographically-random jitter of up to
    :data:`_TUNNEL_RATE_LIMIT_JITTER` seconds so concurrent daemons on
    the same egress IP do not synchronise into the same retry window
    and re-trigger the rate-limit cooldown.
    """
    jitter = secrets.randbelow(_TUNNEL_RATE_LIMIT_JITTER + 1)
    return _TUNNEL_RATE_LIMIT_BACKOFF + jitter

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
            return
        self._head_checked = True
        first_line = self._head_buffer[:idx]
        if first_line.startswith(b"HEAD "):
            transport = self.transport
            if transport is not None:
                transport.write(_HEAD_200)
                transport.close()
            return
        buffered = self._head_buffer
        self._head_buffer = b""
        super().data_received(buffered)


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
        self._pos = 0
        # Maps (file_idx, hunk_idx) -> resolution status ("accepted" or
        # "rejected"); used so the browser can render accepted hunks
        # dimmed and rejected hunks struck-through.
        self._resolved: dict[tuple[int, int], str] = {}

    @property
    def total_hunks(self) -> int:
        """Total number of hunks across all files."""
        return len(self._all_hunks)

    @property
    def remaining(self) -> int:
        """Number of unresolved hunks."""
        return self.total_hunks - len(self._resolved)

    def current(self) -> tuple[int, int] | None:
        """Return (file_idx, hunk_idx) for the current position, or None.

        M11: returns ``None`` when every hunk has been resolved, so a
        post-``accept-all``/``reject-all`` ``current()`` consultation
        is unambiguously empty rather than silently pointing at the
        last (now-resolved) hunk.
        """
        if not self._all_hunks:
            return None
        if not self.remaining:
            return None
        if self._pos >= len(self._all_hunks):
            self._pos = len(self._all_hunks) - 1
        return self._all_hunks[self._pos]

    def mark_resolved(self, fi: int, hi: int, status: str = "accepted") -> None:
        """Mark a hunk as resolved with the given *status*.

        Args:
            fi: File index in :attr:`files`.
            hi: Hunk index within the file.
            status: ``"accepted"`` or ``"rejected"``.  Used by the web
                frontend to render accepted hunks dimmed and rejected
                hunks struck-through after the user acts on them.
        """
        self._resolved[(fi, hi)] = status

    def is_resolved(self, fi: int, hi: int) -> bool:
        """Return True if hunk ``(fi, hi)`` has been marked resolved.

        M9: prefer this method over poking at ``_resolved`` directly so
        the resolution-tracking representation can change without
        breaking callers.
        """
        return (fi, hi) in self._resolved

    def resolutions(self) -> list[dict[str, Any]]:
        """Return the full list of resolved hunks for the browser.

        Each entry is a dict ``{"fi": ..., "hi": ..., "status": ...}``
        suitable for inclusion in a ``merge_nav`` broadcast so the
        webview can visually mark every resolved hunk.
        """
        return [
            {"fi": fi, "hi": hi, "status": status}
            for (fi, hi), status in self._resolved.items()
        ]

    def _seek(self, step: int) -> None:
        """Move *step* (+1 or -1) to the next unresolved hunk."""
        if not self.remaining:
            return
        for _ in range(len(self._all_hunks)):
            self._pos = (self._pos + step) % len(self._all_hunks)
            if not self.is_resolved(*self._all_hunks[self._pos]):
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
            if ffi == fi and not self.is_resolved(ffi, hi)
        ]

    def all_unresolved(self) -> list[tuple[int, int]]:
        """Return all (file_idx, hunk_idx) pairs not yet resolved."""
        return [
            (fi, hi)
            for fi, hi in self._all_hunks
            if not self.is_resolved(fi, hi)
        ]


def _reject_hunk_in_file(
    current_path: str,
    base_path: str,
    hunk: dict[str, int],
    target_path: str | None = None,
) -> None:
    """Revert a single hunk in the current file to the base version.

    Reads both files, replaces the hunk's lines in the current file
    with the corresponding lines from the base file, and writes the
    result back.

    When *target_path* differs from *current_path* — which happens when
    the agent deleted a tracked file and the merge view uses a
    ``.deleted`` placeholder as the visible "current" — the rejected
    content is written to *target_path* (the real workspace location),
    so the file is actually restored on disk.  Subsequent hunks read
    from *target_path* too, so partial rejections accumulate
    correctly.

    Args:
        current_path: Path to the file with agent changes (may be a
            display placeholder for deleted files).
        base_path: Path to the pre-task base copy.
        hunk: Hunk dict with keys ``bs``, ``bc``, ``cs``, ``cc``
            (0-based line positions).
        target_path: Real workspace path to write the rejection to.
            Defaults to *current_path* for backwards compatibility.
    """
    write_to = target_path or current_path
    # Read from *write_to* (the real workspace target) when it exists
    # so that successive partial rejections accumulate against the
    # restored content rather than the (now-stale) placeholder.
    try:
        cur_lines = Path(write_to).read_text().splitlines(keepends=True)
    except OSError:
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
    Path(write_to).parent.mkdir(parents=True, exist_ok=True)
    Path(write_to).write_text("".join(new_lines))


def _reject_all_hunks_in_file(file_data: dict[str, Any]) -> None:
    """Revert an entire file to its base version.

    Copies the base file content over the real workspace path
    (``file_data["target"]`` when present, otherwise
    ``file_data["current"]``).  Using ``target`` ensures that when the
    agent deleted a tracked file and the user rejects all hunks, the
    workspace file is actually restored on disk instead of being left
    behind as a placeholder.

    Args:
        file_data: File entry from merge data with ``base``,
            ``current`` and (optionally) ``target`` path strings.
    """
    write_to = file_data.get("target") or file_data["current"]
    if Path(file_data["base"]).is_file():
        Path(write_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_data["base"], write_to)


_VSCODE_ONLY_COMMANDS = frozenset({
    "focusEditor",
    "webviewFocusChanged",
    "openFile",
    "resolveDroppedPaths",
    "pickFolder",
})

_KISS_HOME = Path(os.environ.get("KISS_HOME") or (Path.home() / ".kiss"))
_TLS_DIR = _KISS_HOME / "tls"
_URL_FILE = _KISS_HOME / "remote-url.json"


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


_CLOUDFLARED_PIDFILE = Path.home() / ".kiss" / "cloudflared.pid"


def _is_pid_alive(pid: int) -> bool:
    """Return True iff a process with *pid* currently exists.

    Uses ``os.kill(pid, 0)`` which sends signal 0 (no-op) and either
    succeeds (process exists and we have permission), raises
    :class:`ProcessLookupError` (process is gone — return False),
    or raises :class:`PermissionError` (process exists but is owned
    by another user — still alive, so return True).
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _save_cloudflared_pidfile(
    pid: int, metrics_port: int, url: str | None,
) -> None:
    """Persist cloudflared's pid + metrics port + URL to disk.

    Written atomically via tmp + ``Path.replace`` so concurrent readers
    (a sibling ``kiss-web`` restarted by ``launchd``) never observe a
    partially-written file.  Best-effort: write failures are logged at
    DEBUG and do not propagate, since the worst case is that the next
    ``kiss-web`` startup falls back to spawning a fresh cloudflared.
    """
    data: dict[str, Any] = {"pid": pid, "metrics_port": metrics_port}
    if url:
        data["url"] = url
    try:
        _CLOUDFLARED_PIDFILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CLOUDFLARED_PIDFILE.with_name(
            f"cloudflared.pid.tmp.{os.getpid()}",
        )
        tmp.write_text(json.dumps(data) + "\n")
        tmp.replace(_CLOUDFLARED_PIDFILE)
    except OSError as exc:
        logger.debug("Failed to write cloudflared pidfile: %s", exc)


def _load_cloudflared_pidfile() -> dict[str, Any] | None:
    """Read and validate the cloudflared pidfile.

    Returns the parsed dict (with at least an integer ``pid`` key) or
    ``None`` if the file is missing, malformed, or invalid.
    """
    try:
        raw = _CLOUDFLARED_PIDFILE.read_text()
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pid"), int):
        return None
    return data


def _try_adopt_existing_cloudflared() -> tuple[int, int, str] | None:
    """Look for a healthy cloudflared started by a previous kiss-web.

    Reads ``~/.kiss/cloudflared.pid``, verifies the pid is alive and
    the metrics ``/ready`` endpoint reports ``readyConnections > 0``,
    and re-discovers the public URL via the ``/quicktunnel`` endpoint
    (falling back to the URL recorded in the pidfile when the metrics
    endpoint doesn't expose one — e.g. named tunnels).

    This is how the daemon preserves a single quick-tunnel URL across
    its own restarts: ``cloudflared`` is spawned in its own process
    group (``start_new_session=True``) so it survives ``kiss-web``'s
    SIGTERM, the VS Code extension's ``pkill kiss-web`` no longer
    targets it, and the next ``kiss-web`` startup adopts it here
    instead of spawning a fresh quick-tunnel with a new hostname.

    Returns:
        ``(pid, metrics_port, url)`` if adoption succeeded, else
        ``None`` (caller spawns a fresh cloudflared).
    """
    data = _load_cloudflared_pidfile()
    if data is None:
        return None
    pid = int(data["pid"])
    metrics_port = data.get("metrics_port")
    if not isinstance(metrics_port, int):
        return None
    if not _is_pid_alive(pid):
        logger.info(
            "cloudflared pidfile points to dead pid %d; ignoring", pid,
        )
        return None
    if not _probe_tunnel_ready(metrics_port):
        logger.info(
            "cloudflared pid %d alive but metrics port %d reports "
            "no ready connections; not adopting",
            pid, metrics_port,
        )
        return None
    # Prefer a freshly-probed URL so we recover from URL rotation
    # between adoption attempts; fall back to the saved one.
    url: str | None = None
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{metrics_port}/quicktunnel",
            headers={"User-Agent": "kiss-web"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            doc = json.loads(resp.read())
            hostname = doc.get("hostname", "")
            if hostname and not hostname.startswith("api."):
                url = f"https://{hostname}"
    except Exception:
        url = None
    if url is None:
        saved = data.get("url")
        if isinstance(saved, str) and saved.startswith("https://"):
            url = saved
    if url is None:
        return None
    logger.info(
        "Adopted existing cloudflared pid=%d metrics_port=%d url=%s",
        pid, metrics_port, url,
    )
    return pid, metrics_port, url


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


def _stderr_reader_loop(
    stderr: Any,
    parse: Callable[[str], str | None],
    result: list[str | None],
    proc: subprocess.Popen[str],
    stop_event: threading.Event | None = None,
    rate_limit_flag: list[bool] | None = None,
    url_found_event: threading.Event | None = None,
) -> None:
    """Read *stderr* lines, parse for a URL, and keep draining until EOF.

    Stores the discovered URL in ``result[0]``.  Top-level helper so
    :func:`_read_url_from_stderr` does not need a closure.

    The loop relies on ``iter(stderr.readline, "")`` so it terminates
    naturally when the subprocess closes its stderr (which happens on
    exit).  ``proc.poll()`` is intentionally **not** checked between
    reads: doing so introduces a race where the subprocess can finish
    writing all its output and exit before the reader has drained the
    pipe, causing the reader to bail out with stderr buffered data
    unread (and the URL therefore missed).

    **Critically**, after finding the URL the loop does **not** return.
    It continues draining stderr so the pipe buffer never fills up.
    If the buffer were to fill (~64 KiB), ``cloudflared`` would block
    on its next stderr write, which in Go deadlocks the whole process
    (the logging mutex prevents any goroutine from making progress).
    The result is an unhealthy tunnel that the watchdog force-restarts,
    giving a new URL every few minutes.

    Args:
        stderr: A line-buffered text-mode file-like object.
        parse: Callback invoked on each line; returns a URL string when
            recognised, otherwise ``None``.
        result: Single-element list used to communicate the URL back
            to the caller across the thread boundary.
        proc: The subprocess being read; retained for API symmetry —
            timeouts are enforced by :func:`_read_url_from_stderr`.
        stop_event: When set by the caller (after a timeout) the loop
            exits at its next iteration.  Used by H6 to bound the
            reader-thread lifetime: once a single additional line is
            consumed (or the subprocess dies) the daemon thread exits
            instead of running until process shutdown.
        rate_limit_flag: Optional single-element list set to ``True``
            on the first stderr line matching
            :func:`_is_rate_limit_line`.  Lets callers distinguish a
            rate-limited tunnel start (HTTP 429 / Cloudflare error
            1015) from a generic failure so the watchdog can apply a
            much longer backoff.
        url_found_event: Optional event set when a URL is first
            discovered.  Signals :func:`_read_url_from_stderr` to
            return the URL immediately while this thread keeps
            draining.
    """
    del proc
    found = False
    for line in iter(stderr.readline, ""):
        if (
            rate_limit_flag is not None
            and not rate_limit_flag[0]
            and _is_rate_limit_line(line)
        ):
            rate_limit_flag[0] = True
        if not found:
            url = parse(line)
            if url is not None:
                result[0] = url
                found = True
                if url_found_event is not None:
                    url_found_event.set()
                # Keep draining stderr — do NOT return.
                continue
        if stop_event is not None and stop_event.is_set():
            return


def _read_url_from_stderr(
    proc: subprocess.Popen[str],
    parse: Callable[[str], str | None],
    timeout: float = 30.0,
    rate_limit_flag: list[bool] | None = None,
) -> str | None:
    """Read *proc*'s stderr until *parse* finds a URL or *timeout* elapses.

    The reader runs in a daemon thread so this call is bounded even
    when ``cloudflared`` keeps streaming non-matching log lines after
    startup.

    Args:
        proc: A subprocess started with ``stderr=subprocess.PIPE`` and
            ``text=True``.
        parse: Per-line URL extractor; returns the URL string or
            ``None``.
        timeout: Maximum seconds to wait before giving up.
        rate_limit_flag: Optional single-element list forwarded to
            :func:`_stderr_reader_loop`; set to ``True`` if any
            consumed stderr line matches a Cloudflare rate-limit
            indicator (HTTP 429 / error 1015).  Lets the caller
            apply a different backoff for rate-limited failures.

    Returns:
        The first URL returned by *parse*, or ``None`` if *proc* exits
        or the timeout elapses without a match.
    """
    stderr = proc.stderr
    assert stderr is not None
    result: list[str | None] = [None]
    stop_event = threading.Event()
    url_found_event = threading.Event()
    reader = threading.Thread(
        target=_stderr_reader_loop,
        args=(
            stderr, parse, result, proc, stop_event, rate_limit_flag,
            url_found_event,
        ),
        daemon=True,
    )
    reader.start()
    # Wait until the reader finds a URL *or* the timeout elapses.
    # Unlike the previous reader.join(timeout), this unblocks as soon
    # as the URL is discovered (the reader thread keeps running in the
    # background to drain stderr so the pipe buffer never fills).
    url_found_event.wait(timeout=timeout)
    if result[0] is not None:
        # URL found.  The reader daemon thread continues draining
        # stderr in the background until cloudflared exits.  This is
        # essential: if nobody reads stderr, the ~64 KiB OS pipe
        # buffer fills, cloudflared blocks on write(), and in Go the
        # logging mutex deadlocks the whole process — making the
        # tunnel unresponsive and triggering a watchdog restart (with
        # a new URL) every few minutes.
        return result[0]
    # H6: No URL found within *timeout*.  Signal the reader to exit on
    # its next iteration.  The reader may still be blocked inside
    # readline() (closing stderr from another thread does not unblock
    # the in-flight read on every platform), but the moment the next
    # log line arrives — or the subprocess dies and readline() returns
    # "" — the loop will observe stop_event and exit, instead of
    # running forever.  This bounds the leak to "one extra line
    # consumed after timeout" rather than "thread leaks on every
    # timed-out restart".
    stop_event.set()
    return None


def _parse_quick_tunnel_url(line: str) -> str | None:
    """Return the ``*.trycloudflare.com`` URL from a quick-tunnel log line.

    Skips ``api.trycloudflare.com`` (Cloudflare's API endpoint, which
    cloudflared logs before the real tunnel URL).
    """
    match = re.search(
        r"(https://(?!api\.)[^\s]+\.trycloudflare\.com)", line,
    )
    return match.group(1) if match else None


def _parse_named_tunnel_url(line: str, configured_url: str | None) -> str | None:
    """Return the public URL of a named tunnel from a log *line*.

    Returns any non-local ``https?://…`` hostname directly.  When a
    "Registered tunnel connection"/"Connection registered" line
    appears, returns *configured_url* (or a sentinel string when no
    URL was pre-configured).  Returns ``None`` on lines that do not
    match either pattern.
    """
    match = re.search(r"https?://([^\s/]+)", line)
    if match:
        host = match.group(1)
        if "localhost" not in host and "127.0.0.1" not in host:
            return f"https://{host}"
    if (
        "Registered tunnel connection" in line
        or "Connection registered" in line
    ):
        return configured_url or (
            "(named tunnel running — URL configured in Cloudflare "
            "dashboard)"
        )
    return None


def _wait_for_remote_password(timeout: float = 30.0) -> str:
    """Block up to *timeout* seconds for ``remote_password`` to appear.

    Polls ``~/.kiss/config.json`` every 500 ms.  This eliminates the
    boot-time race where ``kiss-web`` is restarted by the VS Code
    extension *before* the extension's ``ensureRemotePassword`` flow
    has written the password back to disk: instead of refusing to start
    the tunnel and exiting (which causes ``launchd`` to respawn the
    daemon and mint a brand-new ``*.trycloudflare.com`` URL), the
    daemon waits patiently for the password to arrive.

    Args:
        timeout: Maximum seconds to wait for a non-empty password.

    Returns:
        The non-empty ``remote_password`` value, or ``""`` if the
        timeout elapses without one appearing.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        pw = str(load_config().get("remote_password", "") or "")
        if pw:
            return pw
        if time.monotonic() >= deadline:
            return ""
        time.sleep(0.5)


def _save_url_file(
    url_file: Path, local_url: str, tunnel_url: str | None = None,
) -> None:
    """Write the active server URLs to ``url_file``.

    Creates the parent directory if needed.  The default file location
    is ``~/.kiss/remote-url.json``, which is read by ``kiss-web --url``
    so users can discover the remote URL without digging through log
    files.  Tests inject a temporary path to avoid touching the live
    file that the VS Code extension and the ``kiss-web`` daemon watch.

    Args:
        url_file: Path to the JSON file to write.
        local_url: The local ``https://localhost:PORT`` URL.
        tunnel_url: The Cloudflare tunnel URL, or None.
    """
    data: dict[str, str] = {"local": local_url}
    if tunnel_url:
        data["tunnel"] = tunnel_url
    url_file.parent.mkdir(parents=True, exist_ok=True)
    url_file.write_text(json.dumps(data, indent=2) + "\n")


def _remove_url_file(url_file: Path) -> None:
    """Delete ``url_file`` if it exists."""
    try:
        url_file.unlink(missing_ok=True)
    except OSError:
        pass


def _read_url_from_file(url_file: Path) -> str | None:
    """Read the active remote URL from ``url_file``.

    Synchronous helper invoked from
    :meth:`RemoteAccessServer._send_welcome_info` via
    ``run_in_executor`` so the disk read does not block the asyncio
    event loop.  Returns ``None`` on missing file, parse error, or
    empty content.
    """
    try:
        data = json.loads(url_file.read_text())
    except Exception:
        return None
    url = data.get("tunnel") or data.get("local", "")
    return url or None


def _get_machine_topic() -> str:
    """Return a deterministic ntfy.sh topic derived from machine identity.

    Combines the hostname and MAC address into a SHA-256 hash so the
    topic stays the same across process restarts on the same machine
    but is not guessable by outsiders.

    Returns:
        A hex string suitable for use as an ntfy.sh topic name.
    """
    topic_file = _KISS_HOME / "ntfy_topic"
    if topic_file.is_file():
        stored = topic_file.read_text().strip()
        if stored:
            return stored
    identity = f"{platform.node()}:{uuid.getnode()}"
    topic = "kiss-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
    topic_file.parent.mkdir(parents=True, exist_ok=True)
    topic_file.write_text(topic + "\n")
    return topic


def _get_ntfy_url() -> str:
    """Return the ``https://ntfy.sh/{topic}`` URL for this machine.

    Returns an empty string if the topic cannot be determined.
    """
    try:
        topic = _get_machine_topic()
        if topic:
            return f"https://ntfy.sh/{topic}"
    except Exception:
        logger.debug("Failed to build ntfy URL", exc_info=True)
    return ""


def _post_url_to_message_board(url: str) -> None:
    """Post the active Cloudflare URL to ntfy.sh as a private message.

    Uses the machine-stable topic from :func:`_get_machine_topic` so
    the URL can be retrieved by subscribing to the same topic.  The
    message is posted with a title indicating it is a KISS Sorcar
    remote URL update.  Failures are logged but never raised.

    Args:
        url: The ``https://`` URL to publish.
    """
    if not url or url.startswith("https://localhost"):
        return
    try:
        topic = _get_machine_topic()
        data = url.encode("utf-8")
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=data,
            method="POST",
            headers={
                "Title": "KISS Sorcar Remote URL",
                "Tags": "link,kiss-sorcar",
                "User-Agent": "kiss-web",
            },
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        logger.info("Posted remote URL to ntfy.sh/%s", topic)
    except Exception:
        logger.debug("Failed to post URL to ntfy.sh", exc_info=True)


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
    valid for 10 years, covering ``localhost``, ``127.0.0.1``, ``::1``,
    and all ``*.local`` names.  Parent directories are created as needed.

    M4: the validity is intentionally long-lived (10 years) so the
    auto-generated developer cert does not silently start failing
    after a year.  :func:`_create_ssl_context` also regenerates an
    expiring/expired cert, so even if the validity changes again the
    auto-renewal path will rescue it.

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
        .not_valid_after(now + datetime.timedelta(days=3650))
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
        try:
            os.chmod(d, 0o700)
        except OSError:
            logger.debug("Could not chmod 0700 on %s", d, exc_info=True)

    key_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    # Create the key file with mode 0600 *before* writing, so the
    # private key bytes are never on disk world-readable even briefly.
    if key_path.exists():
        key_path.unlink()
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key_bytes)
    finally:
        os.close(fd)
    # Defensive: also chmod afterwards in case umask interfered.
    os.chmod(key_path, 0o600)
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
        # M4: regenerate a self-signed cert that is already expired or
        # within 30 days of expiry.  Without this the server would
        # silently start serving an expired cert after ~1 year (with
        # the historical 365-day validity) and every browser would
        # refuse to connect.  Only the auto-generated path is
        # regenerated; user-supplied cert/key paths are never touched.
        elif _self_signed_cert_needs_renewal(cert_path):
            logger.info(
                "Self-signed TLS certificate %s is expired or "
                "expiring within 30 days; regenerating",
                cert_path,
            )
            _generate_self_signed_cert(cert_path, key_path)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # M3: pin a minimum TLS version so this hardened context cannot be
    # downgraded to SSLv3 / TLS 1.0 / 1.1 by a hostile client even on
    # Python builds that allow older protocols by default.
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert_path), str(key_path))
    return ctx


def _self_signed_cert_needs_renewal(
    cert_path: Path, threshold_days: int = 30,
) -> bool:
    """Return True if *cert_path* is expired or expires within *threshold_days*.

    Helper for M4 — the auto-generated TLS cert is regenerated when it
    is close to (or past) its ``not_valid_after`` date.  Returns True
    on parse errors so a corrupt cert is also regenerated rather than
    crashing the server at ``load_cert_chain``.
    """
    try:
        from cryptography import x509

        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        not_after = cert.not_valid_after_utc
    except Exception:
        return True
    return not_after - datetime.datetime.now(datetime.UTC) <= datetime.timedelta(
        days=threshold_days,
    )


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
        # Unix-domain socket writers (local extension clients).  Same
        # newline-delimited JSON protocol as WSS clients; broadcasts
        # fan out to both sets in lockstep so a browser viewer and a
        # VS Code extension viewer of the same chat tab observe an
        # identical event stream.
        self._uds_writers: set[asyncio.StreamWriter] = set()
        self._ws_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._merge_state_callback: (
            Callable[[str, dict[str, Any]], None] | None
        ) = None
        # M2: per-instance work_dir avoids mutating ``os.environ``
        # ("KISS_WORKDIR") across instances.  Set by
        # :class:`RemoteAccessServer.__init__`.  Empty string falls
        # back to the env var or cwd.
        self.work_dir: str = ""
        # M8: Pending ``run_coroutine_threadsafe`` futures per client.
        # Tracked so :meth:`remove_client` can cancel pending sends to
        # a slow / dead peer instead of leaking them.
        self._pending_sends: dict[Any, set[ConcurrentFuture[None]]] = {}

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send *event* to every connected WebSocket client.

        Two code paths:

        * Events that already carry an explicit ``tabId`` (status,
          askUser, commitMessage, merge_data, etc.) are treated as
          targeted "system" events: sent verbatim to all connected
          clients (which filter by ``tabId``), but **not** recorded
          or persisted.
        * Events with no ``tabId`` but a thread-local ``task_id`` are
          task events: ``taskId`` is injected, the event is recorded
          under the task and queued for persistence, and one stamped
          copy per subscribed tab is sent to clients.  When no tab is
          currently subscribed the event is recorded / persisted but
          no copy is sent over the wire.
        * Events with neither ``tabId`` nor a resolvable ``taskId``
          are global system events (``configData``, ``models``,
          ``history``, ``inputHistory``, ``welcome_suggestions``,
          ``error``, etc.) and are broadcast verbatim to every
          connected client.

        Args:
            event: The event dictionary to emit.
        """
        if event.get("type") == "configData":
            cfg = event.get("config")
            if isinstance(cfg, dict) and not cfg.get("work_dir"):
                # M2: prefer per-instance work_dir over the global env var.
                cfg["work_dir"] = (
                    self.work_dir
                    or os.environ.get("KISS_WORKDIR", "")
                    or os.getcwd()
                )

        if event.get("type") == "merge_data":
            event = _augment_merge_data(event)
            evt_tab = event.get("tabId", "")
            if evt_tab and self._merge_state_callback is not None:
                self._merge_state_callback(evt_tab, event.get("data", {}))

        if "tabId" in event:
            # Targeted "system" event — forward verbatim, never record
            # or persist (recording / persistence is per-task and is
            # owned by the agent thread).
            self._send_to_ws_clients(json.dumps(event))
            return

        event = self._inject_task_id(event)

        if not event.get("taskId"):
            # Global system event with no task context (configData,
            # models, history, error, etc.) — broadcast verbatim to
            # every connected client.  Not recorded, not persisted.
            self._send_to_ws_clients(json.dumps(event))
            return

        with self._lock:
            self._record_event(event)

        self._persist_event(event)

        # Fan out one stamped copy per subscribed tab.  The frontend
        # filters incoming events by ``tabId``; an event with no
        # subscriber is silently swallowed (which is correct: no UI
        # is currently watching this task).
        for tab_id in self._fanout_targets(event.get("taskId")):
            self._send_to_ws_clients(json.dumps({**event, "tabId": tab_id}))

    def _send_to_ws_clients(self, data: str) -> None:
        """Send a pre-serialised JSON payload to every connected client.

        Factored out of :meth:`broadcast` so fan-out copies for
        subscribed viewer tab ids reuse the same dispatch and pending-
        future tracking as the primary broadcast.  Fans out to BOTH
        WSS clients and local Unix-domain socket writers in lockstep.

        Args:
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        with self._ws_lock:
            clients = list(self._ws_clients)
            writers = list(self._uds_writers)
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        for ws in clients:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    ws.send(data), loop,
                )
            except Exception:
                logger.debug("Failed to send to WS client", exc_info=True)
                continue
            # M8: track the future so a stuck/slow peer's pending
            # sends can be cancelled when the client disconnects.
            with self._ws_lock:
                pending = self._pending_sends.get(ws)
                if pending is not None:
                    pending.add(fut)
            fut.add_done_callback(
                partial(self._discard_pending_send, ws),
            )
        for writer in writers:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._uds_send(writer, data), loop,
                )
            except Exception:
                logger.debug("Failed to send to UDS client", exc_info=True)
                continue
            with self._ws_lock:
                pending = self._pending_sends.get(writer)
                if pending is not None:
                    pending.add(fut)
            fut.add_done_callback(
                partial(self._discard_pending_send, writer),
            )

    async def _uds_send(
        self, writer: asyncio.StreamWriter, data: str,
    ) -> None:
        """Write a newline-delimited JSON payload to a UDS client.

        Mirrors ``ServerConnection.send`` for Unix-domain socket
        peers.  On any write failure, the writer is removed from the
        active set so subsequent broadcasts skip it.

        Args:
            writer: The asyncio stream writer for the UDS connection.
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        try:
            writer.write(data.encode("utf-8") + b"\n")
            await writer.drain()
        except Exception:
            logger.debug("Failed to write to UDS client", exc_info=True)
            self.remove_uds_writer(writer)

    def add_client(self, ws: ServerConnection) -> None:
        """Register a WebSocket client for event broadcasting.

        Args:
            ws: The WebSocket server connection to add.
        """
        with self._ws_lock:
            self._ws_clients.add(ws)
            self._pending_sends.setdefault(ws, set())

    def remove_client(self, ws: ServerConnection) -> None:
        """Remove a WebSocket client from event broadcasting.

        Cancels any pending ``run_coroutine_threadsafe`` futures for
        this client (M8) so a permanently stuck send queue cannot
        keep the underlying coroutine alive after the peer is gone.

        Args:
            ws: The WebSocket server connection to remove.
        """
        with self._ws_lock:
            self._ws_clients.discard(ws)
            pending = self._pending_sends.pop(ws, set())
        for fut in pending:
            try:
                fut.cancel()
            except Exception:
                logger.debug("Failed to cancel pending send", exc_info=True)

    def add_uds_writer(self, writer: asyncio.StreamWriter) -> None:
        """Register a Unix-domain socket writer for event broadcasting.

        Args:
            writer: The asyncio stream writer to add.
        """
        with self._ws_lock:
            self._uds_writers.add(writer)
            self._pending_sends.setdefault(writer, set())

    def remove_uds_writer(self, writer: asyncio.StreamWriter) -> None:
        """Remove a Unix-domain socket writer from event broadcasting.

        Cancels any pending ``run_coroutine_threadsafe`` futures for
        this writer so a permanently stuck send queue cannot keep the
        underlying coroutine alive after the peer is gone.

        Args:
            writer: The asyncio stream writer to remove.
        """
        with self._ws_lock:
            self._uds_writers.discard(writer)
            pending = self._pending_sends.pop(writer, set())
        for fut in pending:
            try:
                fut.cancel()
            except Exception:
                logger.debug(
                    "Failed to cancel pending UDS send", exc_info=True,
                )

    def _discard_pending_send(self, client: Any, fut: Any) -> None:
        """Remove a completed send future from the per-client pending set.

        Called via :meth:`concurrent.futures.Future.add_done_callback`
        once the wrapped coroutine finishes (or errors).  Keeps the
        :attr:`_pending_sends` set bounded so it does not grow without
        limit on a long-running healthy connection.

        ``client`` may be a :class:`ServerConnection` (WSS) or an
        :class:`asyncio.StreamWriter` (UDS).
        """
        with self._ws_lock:
            pending = self._pending_sends.get(client)
            if pending is not None:
                pending.discard(fut)


def _build_html() -> str:
    """Build the standalone HTML page for remote Sorcar access.

    Produces HTML equivalent to ``SorcarTab.buildChatHtml`` but uses
    plain ``/media/`` URLs for assets and injects a WebSocket shim
    script that provides ``acquireVsCodeApi()`` for ``main.js``.

    Returns:
        The complete HTML string.
    """
    version = _read_version()
    tricks_json = json.dumps(_read_tricks())
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
      --vscode-font-size: 16px;
      --vscode-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
        Roboto, 'Helvetica Neue', Arial, sans-serif;
      --vscode-editor-font-size: 16px;
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
    <div id="tab-bar"><div id="tab-list"></div></div>

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
        <div id="welcome-config" class="welcome-config">
          <div id="welcome-remote-url"></div>
          <label class="config-label welcome-config-label">Remote password
            <div class="config-password-wrap">
              <input type="password" id="welcome-cfg-remote-password"
               placeholder="Remote access password">
              <button type="button" id="welcome-cfg-remote-password-toggle"
               class="config-password-toggle"
               aria-label="Show password" aria-pressed="false"
               title="Show password">
                <svg class="icon-eye" width="16" height="16" viewBox="0 0 24 24"
                 fill="none" stroke="currentColor" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/>
                  <circle cx="12" cy="12" r="3"/>
                </svg>
                <svg class="icon-eye-off" width="16" height="16"
                 viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                 aria-hidden="true" style="display:none;">
                  <path d="M17.94 17.94A10.94 10.94 0 0 1 12 19c-7 0-11-7-11-7\
a19.7 19.7 0 0 1 4.22-5.06"/>
                  <path d="M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 7 11 7\
a19.7 19.7 0 0 1-3.16 4.19"/>
                  <path d="M14.12 14.12A3 3 0 0 1 9.88 9.88"/>
                  <line x1="1" y1="1" x2="23" y2="23"/>
                </svg>
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
            <textarea id="task-input"
             placeholder="Ask anything... (@ for files)" rows="1"
             enterkeyhint="send"></textarea>
            <button id="input-clear-btn" style="display:none;">&times;</button>
          </div>
        </div>
        <div id="input-footer">
          <div id="model-picker">
            <button id="menu-btn">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round">
                <line x1="3" y1="6" x2="21" y2="6"/>
                <line x1="3" y1="12" x2="21" y2="12"/>
                <line x1="3" y1="18" x2="21" y2="18"/>
              </svg>
            </button>
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
            <button id="worktree-toggle-btn" class="toggle-btn"
             data-tooltip="Use worktree">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round">
                <line x1="6" y1="3" x2="6" y2="15"/>
                <circle cx="18" cy="6" r="3"/>
                <circle cx="6" cy="18" r="3"/>
                <path d="M18 9a9 9 0 01-9 9"/>
              </svg>
            </button>
            <button id="frequent-tasks-btn" class="toggle-btn"
             data-tooltip="Frequent tasks">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
               stroke-linejoin="round">
                <line x1="6" y1="20" x2="6" y2="14"/>
                <line x1="12" y1="20" x2="12" y2="9"/>
                <line x1="18" y1="20" x2="18" y2="4"/>
              </svg>
            </button>
            <button id="tricks-btn" class="toggle-btn"
             data-tooltip="Inject instruction">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
               stroke-linejoin="round">
                <path d="M9 18h6"/>
                <path d="M10 22h4"/>
                <path d="M12 2a7 7 0 00-4 12.7c.7.6 1 1.4 1 2.3h6c0-.9.3-1.7 1-2.3A7 7 0 0012 2z"/>
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
      <div id="sidebar-tab-history-panel"
       class="sidebar-section sidebar-tab-panel">
        <div class="sidebar-hdr">History</div>
        <div class="search-wrap">
          <input type="text" id="history-search" placeholder="Search history...">
          <button class="search-clear-btn" id="history-search-clear"
           style="display:none;">&times;</button>
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
          <label class="history-filter-chk"
           title="Show only tasks marked as favourite">
            <input type="checkbox" id="hf-favorite">Favorites
          </label>
          <label for="hf-from" class="history-filter-date-lbl">From:</label>
          <input type="date" id="hf-from" class="history-filter-date"
           title="From date" aria-label="From date">
          <button type="button" id="hf-from-btn"
           class="history-filter-date-btn"
           title="Pick From date" aria-label="Pick From date">📅</button>
          <label for="hf-to" class="history-filter-date-lbl">To:</label>
          <input type="date" id="hf-to" class="history-filter-date"
           title="To date" aria-label="To date">
          <button type="button" id="hf-to-btn"
           class="history-filter-date-btn"
           title="Pick To date" aria-label="Pick To date">📅</button>
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
        <div class="sidebar-hdr">Tricks</div>
        <div id="tricks-list">
          <div class="sidebar-empty">No tricks available</div>
        </div>
      </div>
    </div>
    <div id="tricks-overlay"></div>

    <div id="settings-panel">
      <button id="settings-panel-close">&times;</button>
      <div class="sidebar-section">
        <div class="sidebar-hdr">Sorcar Configuration{' ' + version if version else ''}</div>
        <div id="remote-url"></div>
        <div id="config-form">
          <label class="config-label">Remote password
            <div class="config-password-wrap">
              <input type="password" id="cfg-remote-password"
               placeholder="Remote access password">
              <button type="button" id="cfg-remote-password-toggle"
               class="config-password-toggle"
               aria-label="Show password" aria-pressed="false"
               title="Show password">
                <svg class="icon-eye" width="16" height="16" viewBox="0 0 24 24"
                 fill="none" stroke="currentColor" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/>
                  <circle cx="12" cy="12" r="3"/>
                </svg>
                <svg class="icon-eye-off" width="16" height="16"
                 viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                 aria-hidden="true" style="display:none;">
                  <path d="M17.94 17.94A10.94 10.94 0 0 1 12 19c-7 0-11-7-11-7\
a19.7 19.7 0 0 1 4.22-5.06"/>
                  <path d="M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 7 11 7\
a19.7 19.7 0 0 1-3.16 4.19"/>
                  <path d="M14.12 14.12A3 3 0 0 1 9.88 9.88"/>
                  <line x1="1" y1="1" x2="23" y2="23"/>
                </svg>
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
          </label>
          <label class="config-label config-checkbox">
            <input type="checkbox" id="cfg-use-parallel" checked>
            Use parallel agents
          </label>
          <label class="config-label config-checkbox">
            <input type="checkbox" id="cfg-demo-mode">
            Demo mode
          </label>

          <div class="config-divider"></div>
          <div class="sidebar-hdr" style="margin-top:8px;">API Keys</div>
          <label class="config-label">Gemini API Key
            <input type="text" id="cfg-key-GEMINI_API_KEY"
             placeholder="Enter Gemini API key">
          </label>
          <label class="config-label">OpenAI API Key
            <input type="text" id="cfg-key-OPENAI_API_KEY"
             placeholder="Enter OpenAI API key">
          </label>
          <label class="config-label">Anthropic API Key
            <input type="text" id="cfg-key-ANTHROPIC_API_KEY"
             placeholder="Enter Anthropic API key">
          </label>
          <label class="config-label">Together API Key
            <input type="text" id="cfg-key-TOGETHER_API_KEY"
             placeholder="Enter Together API key">
          </label>
          <label class="config-label">OpenRouter API Key
            <input type="text" id="cfg-key-OPENROUTER_API_KEY"
             placeholder="Enter OpenRouter API key">
          </label>
          <label class="config-label">MiniMax API Key
            <input type="text" id="cfg-key-MINIMAX_API_KEY"
             placeholder="Enter MiniMax API key">
          </label>
          <label class="config-label">Custom endpoint (local model)
            <input type="text" id="cfg-custom-endpoint"
             placeholder="http://localhost:8080/v1">
          </label>
          <label class="config-label">Custom API key
            <input type="text" id="cfg-custom-api-key"
             placeholder="Optional API key for custom endpoint">
          </label>
          <label class="config-label">Custom headers
            <textarea id="cfg-custom-headers" rows="2"
             placeholder="Key:Value (one per line)"></textarea>
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

    <div id="auth-modal" style="display:none;">
      <div class="auth-modal-content">
        <div class="auth-modal-title">Remote access password</div>
        <input type="password" id="auth-modal-input" class="auth-modal-input"
               autocomplete="current-password" placeholder="Enter password">
        <div class="auth-modal-actions">
          <button id="auth-modal-cancel" class="auth-modal-btn auth-modal-cancel"
                  type="button">Cancel</button>
          <button id="auth-modal-ok" class="auth-modal-btn auth-modal-ok"
                  type="button">OK</button>
        </div>
      </div>
    </div>
  </div>

  <script src="/media/highlight.min.js"></script>
  <script src="/media/marked.min.js"></script>
  <script>{shim}</script>
  <script>window.__TRICKS__ = {tricks_json};</script>
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


def _read_tricks() -> list[str]:
    """Parse ``src/kiss/INJECTIONS.md`` and return the trick texts.

    The file contains a series of ``## Trick`` sections, each followed
    by a blank line and a one-line trick.  Returns an empty list if the
    file is missing or unparseable, so a deployment without
    INJECTIONS.md still renders the button (with an empty list).
    """
    try:
        tfile = Path(__file__).parent.parent.parent / "INJECTIONS.md"
        text = tfile.read_text()
    except Exception:
        return []
    tricks: list[str] = []
    # Split on H2 headings, then keep only sections whose title is
    # "Trick".  ``re.split`` with a capturing group preserves the
    # heading so we can identify each section.
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for section in sections[1:]:
        lines = section.splitlines()
        if not lines or lines[0].strip() != "Trick":
            continue
        body = "\n".join(lines[1:]).strip()
        if body:
            tricks.append(body)
    return tricks


_WS_SHIM_JS = r"""
(function() {
  var _state = null;
  try { _state = JSON.parse(sessionStorage.getItem('sorcar-state')); } catch(e) {}
  var _ws = null;
  var _pending = [];
  var _authenticated = false;
  var _needsPassword = false;

  // Custom auth modal — replaces the browser-native prompt(), which is
  // rendered tall with wasted space below its buttons on most desktop
  // browsers.  Falls back to prompt() when the modal nodes are not in
  // the DOM (e.g. unit tests that load the shim in isolation).
  function _showAuthModal() {
    return new Promise(function(resolve) {
      var modal  = document.getElementById('auth-modal');
      var input  = document.getElementById('auth-modal-input');
      var okBtn  = document.getElementById('auth-modal-ok');
      var cnclBtn = document.getElementById('auth-modal-cancel');
      if (!modal || !input || !okBtn || !cnclBtn) {
        resolve(prompt('Enter remote access password:'));
        return;
      }
      input.value = '';
      modal.style.display = 'flex';
      setTimeout(function() { try { input.focus(); } catch(e) {} }, 0);

      function cleanup() {
        modal.style.display = 'none';
        okBtn.removeEventListener('click', onOk);
        cnclBtn.removeEventListener('click', onCancel);
        input.removeEventListener('keydown', onKey);
      }
      function onOk()     { var v = input.value; cleanup(); resolve(v); }
      function onCancel() { cleanup(); resolve(null); }
      function onKey(e) {
        if (e.key === 'Enter')        { e.preventDefault(); onOk();     }
        else if (e.key === 'Escape')  { e.preventDefault(); onCancel(); }
      }
      okBtn.addEventListener('click', onOk);
      cnclBtn.addEventListener('click', onCancel);
      input.addEventListener('keydown', onKey);
    });
  }

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
      var pwd = '';
      try { pwd = localStorage.getItem('sorcar-remote-pwd') || ''; } catch(e) {}
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
        // Stored password (if any) was rejected; drop it so a refresh
        // re-prompts instead of silently retrying the bad value.
        try { localStorage.removeItem('sorcar-remote-pwd'); } catch(e) {}
        _showAuthModal().then(function(pwd) {
          if (pwd === null || pwd === undefined) return;
          try { localStorage.setItem('sorcar-remote-pwd', pwd); } catch(e) {}
          if (_ws && _ws.readyState === WebSocket.OPEN) {
            _ws.send(JSON.stringify({type: 'auth', password: pwd}));
          }
        });
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
        # Binary files (PDFs, images, etc.) have no meaningful text
        # representation; the MergeManager / web client open them via
        # the native viewer.  Attempting ``read_text()`` on them would
        # raise ``UnicodeDecodeError`` (a ``ValueError``, not
        # ``OSError``), which previously aborted the entire
        # ``merge_data`` broadcast and prevented the diff/merge UI
        # from appearing for binary-only changes.
        if f.get("binary"):
            f["base_text"] = ""
            f["current_text"] = ""
            files.append(f)
            continue
        try:
            f["base_text"] = Path(f["base"]).read_text()
        except (OSError, KeyError, UnicodeDecodeError):
            f["base_text"] = ""
        try:
            f["current_text"] = Path(f["current"]).read_text()
        except (OSError, KeyError, UnicodeDecodeError):
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

    A named tunnel's public hostname is configured in the Cloudflare
    Zero Trust dashboard and is **not** embedded in the token, nor
    echoed by ``cloudflared`` in a parseable form.  To advertise the
    public URL to clients (in ``~/.kiss/remote-url.json`` and via the
    ``remote_url`` WebSocket broadcast), the user must supply that URL
    via *tunnel_url*, the ``CLOUDFLARE_TUNNEL_URL`` env var, or the
    ``tunnel_url`` key in ``~/.kiss/config.json``.

    Args:
        host: Bind address (default ``"0.0.0.0"`` for all interfaces).
        port: TCP port for both HTTPS and WSS (default ``8787``).
        use_tunnel: If True, start a ``cloudflared`` tunnel on launch.
        tunnel_token: Cloudflare named-tunnel token for a fixed URL.
            When set, ``cloudflared tunnel run --token <TOKEN>`` is
            used instead of a quick-tunnel.
        tunnel_url: Public ``https://`` URL of the named tunnel as
            configured in the Cloudflare dashboard.  Only meaningful
            when *tunnel_token* is set.  When provided, this URL is
            returned to clients once the tunnel registers a connection.
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
        tunnel_url: str | None = None,
        work_dir: str | None = None,
        certfile: str | None = None,
        keyfile: str | None = None,
        url_file: str | Path | None = None,
        uds_path: str | Path | None = None,
    ) -> None:
        source_shell_env()

        self.host = host
        self.port = port
        self.use_tunnel = use_tunnel
        self.tunnel_token = tunnel_token
        self.tunnel_url = tunnel_url
        self._ssl_context: ssl.SSLContext = _create_ssl_context(certfile, keyfile)
        # Path to the JSON file used to publish the active URL.  Tests
        # override this to a per-test temporary path to avoid racing
        # the live ``~/.kiss/remote-url.json`` watched by the VS Code
        # extension and the ``kiss-web`` daemon.
        self._url_file: Path = Path(url_file) if url_file else _URL_FILE

        if not work_dir:
            from kiss.agents.vscode.vscode_config import load_config

            work_dir = load_config().get("work_dir", "") or None
        # M2: store work_dir on the instance instead of mutating the
        # global ``os.environ["KISS_WORKDIR"]`` (which would stomp on
        # other concurrent ``RemoteAccessServer`` instances and leak
        # process-wide state into tests).  ``self.work_dir`` is the
        # canonical resolved value, ``""`` when neither argument nor
        # config provides one.
        self.work_dir: str = work_dir or ""

        self._printer = WebPrinter()
        self._printer.work_dir = self.work_dir
        self._vscode_server = VSCodeServer(printer=self._printer)
        if self.work_dir:
            self._vscode_server.work_dir = self.work_dir

        self._html_bytes = _build_html().encode("utf-8")
        self._tunnel_proc: subprocess.Popen[str] | None = None
        self._tunnel_metrics_port: int | None = None
        self._tunnel_unhealthy_ticks = 0
        self._tunnel_started_at: float | None = None
        self._tunnel_failure_count = 0
        self._tunnel_next_retry = 0.0
        # Pid of a cloudflared spawned by a *previous* ``kiss-web``
        # process that this instance adopted on startup (see
        # :func:`_try_adopt_existing_cloudflared`).  When non-None,
        # ``self._tunnel_proc`` is None (we don't own the process) but
        # ``self._tunnel_metrics_port`` and the URL file are populated
        # as if we had spawned it ourselves.  The watchdog treats an
        # adopted pid as equivalent to a self-spawned proc for
        # health-check purposes.
        self._tunnel_adopted_pid: int | None = None
        # Set by ``_start_quick_tunnel`` when cloudflared's stderr
        # reports a Cloudflare quick-tunnel rate-limit (HTTP 429 /
        # error 1015).  ``_restart_tunnel_url`` reads (and clears) it
        # to apply a much longer backoff than the regular exponential
        # one so the per-IP cooldown actually has time to clear.
        self._tunnel_rate_limited = False
        # Last Cloudflare tunnel URL posted to the ntfy.sh message
        # board.  Tracked so a watchdog restart that yields the *same*
        # public hostname (e.g. an adopted cloudflared, or a named
        # tunnel re-registering) does not re-publish the unchanged URL
        # to subscribers.
        self._last_posted_url: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_server: Any = None
        # Unix-domain socket listener for local clients (the VS Code
        # extension).  Bound in :meth:`_setup_server` and torn down
        # in :meth:`stop_async`.  Tests pass an explicit
        # ``uds_path`` so concurrent instances do not race on the
        # shared default ``~/.kiss/sorcar.sock``.
        self._uds_path: Path = (
            Path(uds_path) if uds_path else _UDS_PATH
        )
        self._uds_server: asyncio.Server | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._local_url = f"https://localhost:{self.port}"
        self._merge_states: dict[str, _WebMergeState] = {}
        # M6: a small lock guards _merge_states so the agent
        # task-runner thread (via WebPrinter.broadcast →
        # _register_merge_state) and the asyncio thread (via
        # _handle_web_merge_action / _ws_handler cleanup) cannot lose
        # each other's mutations.  CPython's GIL makes individual
        # dict ops atomic, but a sequence like ``del[k]`` racing a
        # concurrent ``[k] = v`` can still drop a registration.
        self._merge_states_lock = threading.Lock()
        # M6: per-WebSocket set of tab IDs ever seen on this
        # connection.  When the WS closes we delete merge states
        # belonging to those tabs so the dict does not grow unbounded
        # over many short-lived sessions.
        self._ws_tabs: dict[ServerConnection, set[str]] = {}
        # Deferred-disposal state for tabs whose WebSocket connection
        # has dropped but whose backend ``_RunningAgentState`` should survive a
        # short grace window so a reload / transient reconnect can
        # re-claim it.  See :meth:`_schedule_tab_close`.
        self._pending_tab_closes: dict[str, asyncio.TimerHandle] = {}
        self._pending_tab_closes_lock = threading.Lock()
        # Strong references for in-flight ``closeTab`` tasks dispatched
        # by :meth:`_fire_pending_tab_close`.  Without this set the
        # asyncio runtime can GC the task while it is still pending.
        self._pending_close_tasks: set[asyncio.Task[None]] = set()
        self._printer._merge_state_callback = self._register_merge_state
        self._active_url: str | None = None
        self._last_ips: frozenset[str] = frozenset()
        # Per-IP auth failure timestamps, used by _is_auth_locked.
        self._auth_failures: dict[str, list[float]] = {}


    async def _process_request(
        self, _connection: ServerConnection, request: Request
    ) -> Response | None:
        """Serve HTTP requests for the HTML page and static assets.

        Returns a :class:`Response` for regular HTTP requests, or
        ``None`` to let the WebSocket handshake proceed for ``/ws``.

        Args:
            _connection: The server connection (unused for HTTP).
            request: The incoming HTTP request.

        Returns:
            An HTTP response, or ``None`` for WebSocket upgrade.
        """
        path = request.path
        if path == "/" or path == "":
            return _http_response(200, "text/html; charset=utf-8", self._html_bytes)
        if path == "/ws":
            return None
        if path.startswith("/media/"):
            filepath = MEDIA_DIR / path[7:]
            if (
                filepath.resolve().is_relative_to(MEDIA_DIR.resolve())
                and filepath.is_file()
            ):
                ctype = mimetypes.guess_type(str(filepath))[0] or "application/octet-stream"
                return _http_response(200, ctype, filepath.read_bytes())
        return _http_response(404, "text/plain", b"Not Found")


    @staticmethod
    def _passwords_equal(a: str, b: str) -> bool:
        """Constant-time string compare to defeat timing attacks.

        Encodes to bytes and delegates to :func:`secrets.compare_digest`.
        """
        return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

    def _client_ip(self, websocket: ServerConnection) -> str:
        """Return the source IP of *websocket*, or ``"?"`` if unknown."""
        addr = getattr(websocket, "remote_address", None)
        if addr and len(addr) >= 1:
            return str(addr[0])
        return "?"

    def _is_auth_locked(self, ip: str) -> bool:
        """Return True if *ip* is currently rate-limited.

        An IP becomes locked once it has accumulated
        :data:`_AUTH_FAIL_MAX` failures within the most recent
        :data:`_AUTH_FAIL_WINDOW` seconds.  The lock persists until
        :data:`_AUTH_LOCKOUT` seconds have elapsed since the last
        recorded failure.
        """
        now = time.monotonic()
        fails = self._auth_failures.get(ip, [])
        # Drop entries outside the window.
        fails = [t for t in fails if now - t <= _AUTH_FAIL_WINDOW]
        self._auth_failures[ip] = fails
        if len(fails) < _AUTH_FAIL_MAX:
            return False
        return (now - fails[-1]) <= _AUTH_LOCKOUT

    def _record_auth_failure(self, ip: str) -> None:
        """Record a failed authentication attempt from *ip*."""
        now = time.monotonic()
        self._auth_failures.setdefault(ip, []).append(now)

    async def _authenticate_ws(self, websocket: ServerConnection) -> bool:
        """Authenticate a WebSocket client using the configured password.

        Returns True on success, False (and closes the socket) on failure.

        When the configured ``remote_password`` is empty, all clients
        are still required to send an empty-password ``auth`` message
        (using a constant-time compare).  See also
        :meth:`_setup_server` which refuses to advertise the public
        cloudflared tunnel when no password is configured.
        """
        ip = self._client_ip(websocket)
        if self._is_auth_locked(ip):
            logger.warning("Auth rate-limit hit for %s; closing socket", ip)
            try:
                await websocket.close()
            except Exception:
                pass
            return False
        password = load_config().get("remote_password", "")
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("type") != "auth":
                await websocket.close()
                return False
            client_pw = msg.get("password", "")
            if not isinstance(client_pw, str):
                client_pw = ""
            if not self._passwords_equal(password, client_pw):
                self._record_auth_failure(ip)
                await websocket.send(json.dumps({"type": "auth_required"}))
                raw2 = await asyncio.wait_for(websocket.recv(), timeout=60)
                msg2 = json.loads(raw2)
                client_pw2 = msg2.get("password", "")
                if not isinstance(client_pw2, str):
                    client_pw2 = ""
                if msg2.get("type") != "auth" or not self._passwords_equal(
                    password, client_pw2,
                ):
                    self._record_auth_failure(ip)
                    await websocket.send(
                        json.dumps({"type": "error", "text": "Authentication failed"})
                    )
                    await websocket.close()
                    return False
            await websocket.send(json.dumps({"type": "auth_ok"}))
            return True
        except Exception:
            logger.debug("WS auth failed", exc_info=True)
            try:
                await websocket.close()
            except Exception:
                pass
            return False

    async def _run_cmd(self, cmd: dict[str, Any]) -> None:
        """Run a backend command in the thread-pool executor."""
        assert self._loop is not None
        await self._loop.run_in_executor(
            None, self._vscode_server._handle_command, cmd,
        )

    def _schedule_tab_close(self, tab_id: str) -> None:
        """Schedule a deferred ``closeTab`` for *tab_id* after a grace period.

        Called from :meth:`_ws_handler`'s ``finally`` block whenever a
        WebSocket connection drops, for every tab id that was seen on
        that connection.  Because browsers cannot reliably emit a
        ``closeTab`` before the WSS shuts down (``beforeunload`` /
        ``pagehide`` WebSocket writes are commonly buffered then
        dropped), the grace timer is the canonical way to detect that
        the frontend tab is truly gone — a reload / transient
        reconnect that re-claims the same tab id within
        :data:`_TAB_CLOSE_GRACE` seconds calls
        :meth:`_cancel_pending_tab_close` to abort the disposal.

        If a previous timer was already armed for *tab_id*, it is
        cancelled and replaced (extending the grace window each time
        the same tab id appears on a new dropped connection).

        Args:
            tab_id: The frontend tab identifier whose backend state
                should be torn down once the grace period elapses,
                unless cancelled by a reconnect.
        """
        if not tab_id:
            return
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        with self._pending_tab_closes_lock:
            existing = self._pending_tab_closes.pop(tab_id, None)
            if existing is not None:
                try:
                    existing.cancel()
                except Exception:
                    logger.debug(
                        "Cancel of stale pending close failed",
                        exc_info=True,
                    )
            handle = loop.call_later(
                _TAB_CLOSE_GRACE,
                self._fire_pending_tab_close,
                tab_id,
            )
            self._pending_tab_closes[tab_id] = handle

    def _cancel_pending_tab_close(self, tab_id: str) -> None:
        """Cancel a pending deferred ``closeTab`` for *tab_id*.

        Called from :meth:`_handle_ready` when a fresh WebSocket
        connection re-claims a tab id (either as the current
        ``tabId`` or as an entry in ``restoredTabs``).  Idempotent and
        safe for unknown tab ids — if no timer was armed for *tab_id*
        the call is a no-op.

        Args:
            tab_id: The frontend tab identifier being re-claimed.
        """
        if not tab_id:
            return
        with self._pending_tab_closes_lock:
            handle = self._pending_tab_closes.pop(tab_id, None)
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                logger.debug(
                    "Cancel of pending tab close failed", exc_info=True,
                )

    def _fire_pending_tab_close(self, tab_id: str) -> None:
        """Execute the deferred ``closeTab`` for *tab_id*.

        Runs on the asyncio event loop after :data:`_TAB_CLOSE_GRACE`
        seconds elapse without a reconnect cancelling the timer.
        Pops the merge state (if any) for *tab_id* and dispatches the
        ``closeTab`` command through :meth:`_run_cmd`, which routes
        through :class:`VSCodeServer._close_tab`.  When the tab is
        idle, ``_close_tab`` disposes the ``_RunningAgentState`` immediately;
        when a task or merge review is still in flight, it flips
        ``frontend_closed=True`` and lets the existing deferred-
        disposal hook (:meth:`VSCodeServer._dispose_if_closed`) tear
        down the state once the lifecycle ends — never interrupting
        the running agent.

        Args:
            tab_id: The frontend tab identifier whose grace window
                has elapsed.
        """
        with self._pending_tab_closes_lock:
            self._pending_tab_closes.pop(tab_id, None)
        with self._merge_states_lock:
            self._merge_states.pop(tab_id, None)
        if self._loop is None or not self._loop.is_running():
            return
        task = asyncio.ensure_future(
            self._run_cmd({"type": "closeTab", "tabId": tab_id}),
            loop=self._loop,
        )
        self._pending_close_tasks.add(task)
        task.add_done_callback(self._pending_close_tasks.discard)

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
        # M6: track tab_ids seen on this connection so we can clean
        # up associated merge state when the connection drops.
        self._ws_tabs[websocket] = set()
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                except json.JSONDecodeError:
                    continue
                tab_id = cmd.get("tabId", "")
                if isinstance(tab_id, str) and tab_id:
                    self._ws_tabs[websocket].add(tab_id)
                cmd_type = cmd.get("type", "")
                if cmd_type in _VSCODE_ONLY_COMMANDS:
                    continue
                if cmd_type == "ready":
                    await self._handle_ready(cmd, websocket)
                    continue
                if cmd_type == "submit":
                    await self._handle_submit(cmd)
                    continue
                if cmd_type == "getWelcomeSuggestions":
                    await self._send_welcome_info()
                    continue
                if cmd_type == "mergeAction":
                    action = cmd.get("action", "")
                    if action != "all-done":
                        await self._handle_web_merge_action(cmd)
                        continue
                cmd = _translate_webview_command(cmd)
                await self._run_cmd(cmd)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logger.debug("WS handler error", exc_info=True)
        finally:
            # Deferred disposal: schedule a ``closeTab`` for every
            # tab id this connection touched, with a short grace
            # window so a reload / transient reconnect can re-claim
            # the same tab ids and cancel the pending close.  Merge
            # state is also popped lazily inside
            # :meth:`_fire_pending_tab_close` so a reconnect within
            # the grace window keeps the merge review intact.
            tabs = self._ws_tabs.pop(websocket, set())
            for tab in tabs:
                self._schedule_tab_close(tab)
            self._printer.remove_client(websocket)

    async def _uds_handler(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a local Unix-domain socket client connection.

        Speaks the SAME newline-delimited JSON protocol as
        :meth:`_ws_handler` but skips the password challenge — POSIX
        filesystem permissions (mode 0o600 on the socket file) gate
        access to the owning user.  Used by the VS Code extension so
        the same :class:`VSCodeServer` instance serves both local and
        remote clients out of one process, eliminating the per-tab
        Python subprocess plumbing.

        Args:
            reader: Asyncio stream reader for the connection.
            writer: Asyncio stream writer for the connection.  Also
                registered with :class:`WebPrinter` so backend
                broadcasts reach this peer.
        """
        self._printer.add_uds_writer(writer)
        tabs_seen: set[str] = set()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(cmd, dict):
                    continue
                tab_id = cmd.get("tabId", "")
                if isinstance(tab_id, str) and tab_id:
                    tabs_seen.add(tab_id)
                cmd_type = cmd.get("type", "")
                if cmd_type in _VSCODE_ONLY_COMMANDS:
                    continue
                if cmd_type == "ready":
                    await self._handle_ready(cmd, writer)
                    continue
                if cmd_type == "submit":
                    await self._handle_submit(cmd)
                    continue
                if cmd_type == "getWelcomeSuggestions":
                    await self._send_welcome_info()
                    continue
                if cmd_type == "mergeAction":
                    action = cmd.get("action", "")
                    if action != "all-done":
                        await self._handle_web_merge_action(cmd)
                        continue
                cmd = _translate_webview_command(cmd)
                await self._run_cmd(cmd)
        except Exception:
            logger.debug("UDS handler error", exc_info=True)
        finally:
            for tab in tabs_seen:
                self._schedule_tab_close(tab)
            self._printer.remove_uds_writer(writer)
            try:
                writer.close()
            except Exception:
                logger.debug("UDS writer close failed", exc_info=True)


    async def _send_welcome_info(self) -> None:
        """Broadcast welcome suggestions and the active remote URL.

        Broadcasts a ``welcome_suggestions`` event with an empty list
        because the remote chat webview deliberately suppresses sample
        task suggestions on the welcome page (the centered input
        textbox is the only welcome-page UI).  Then broadcasts the
        ``remote_url`` event using the in-memory URL, the URL file, or
        the ``cloudflared`` metrics API as successive fallbacks.

        M10: the URL-file read and the ``_discover_tunnel_url_from_metrics``
        call (which spawns ``pgrep`` and does HTTP requests) are
        blocking I/O.  They run in :meth:`asyncio.AbstractEventLoop.run_in_executor`
        so a slow ``pgrep`` or unreachable cloudflared metrics
        endpoint cannot stall the asyncio event loop.
        """
        self._printer.broadcast({
            "type": "welcome_suggestions", "suggestions": [],
        })

        url: str | None = self._active_url
        loop = self._loop
        assert loop is not None
        if not url:
            url = await loop.run_in_executor(
                None, _read_url_from_file, self._url_file,
            )
        if not url:
            discovered = await loop.run_in_executor(
                None, _discover_tunnel_url_from_metrics,
            )
            if discovered:
                await loop.run_in_executor(
                    None, _save_url_file,
                    self._url_file, self._local_url, discovered,
                )
                self._active_url = discovered
                url = discovered
        if url:
            ntfy_url = _get_ntfy_url()
            msg: dict[str, object] = {"type": "remote_url", "url": url}
            if ntfy_url:
                msg["ntfyUrl"] = ntfy_url
            self._printer.broadcast(msg)

    @staticmethod
    async def _endpoint_send(endpoint: Any, data: str) -> None:
        """Send ``data`` to either a WSS or a UDS endpoint.

        ``endpoint`` is either a :class:`ServerConnection` (WSS) or
        an :class:`asyncio.StreamWriter` (UDS).  This helper hides
        the protocol difference so :meth:`_handle_ready` and
        :meth:`_uds_handler` share a single dispatch path.

        Args:
            endpoint: The connection to send to.
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        if isinstance(endpoint, asyncio.StreamWriter):
            endpoint.write(data.encode("utf-8") + b"\n")
            await endpoint.drain()
        else:
            await endpoint.send(data)

    async def _handle_ready(
        self, cmd: dict[str, Any], websocket: Any,
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
        # A fresh ``ready`` is the unambiguous signal that the
        # frontend has reconnected and is re-claiming whatever tab
        # ids it carries.  Cancel the deferred ``closeTab`` for the
        # current tab id (and every restored tab below) so a reload
        # within :data:`_TAB_CLOSE_GRACE` keeps the backend state.
        self._cancel_pending_tab_close(tab_id)
        for init_cmd in ("getModels", "getInputHistory", "getConfig"):
            await self._run_cmd({"type": init_cmd})
        await self._send_welcome_info()
        try:
            await self._endpoint_send(
                websocket,
                json.dumps({"type": "focusInput", "tabId": tab_id}),
            )
        except Exception:
            pass
        # M7: cap the number of restored tabs a single client can ask
        # the server to resume so an authenticated-but-malicious or
        # buggy client cannot flood the executor with thousands of
        # ``resumeSession`` jobs.
        restored = cmd.get("restoredTabs") or []
        if not isinstance(restored, list):
            restored = []
        if len(restored) > _MAX_RESTORED_TABS:
            logger.warning(
                "restoredTabs count %d exceeds cap %d; truncating",
                len(restored), _MAX_RESTORED_TABS,
            )
            restored = restored[:_MAX_RESTORED_TABS]
        for rt in restored:
            rt_id = rt.get("tabId", "")
            if rt_id:
                self._cancel_pending_tab_close(rt_id)
            chat_id = rt.get("chatId", "")
            if chat_id:
                await self._run_cmd(
                    {"type": "resumeSession", "chatId": chat_id,
                     "tabId": rt_id},
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
        # M7: clamp the prompt size so a giant payload cannot push
        # tens of MB through the broadcast pipeline (every connected
        # client receives a ``setTaskText`` echo of this string).
        if isinstance(prompt, str) and len(prompt) > _MAX_PROMPT_BYTES:
            logger.warning(
                "prompt size %d exceeds cap %d bytes; truncating",
                len(prompt), _MAX_PROMPT_BYTES,
            )
            prompt = prompt[:_MAX_PROMPT_BYTES]
        # M7: clamp the attachments list size symmetrically.
        attachments = cmd.get("attachments")
        if isinstance(attachments, list) and len(attachments) > _MAX_ATTACHMENTS:
            logger.warning(
                "attachments count %d exceeds cap %d; truncating",
                len(attachments), _MAX_ATTACHMENTS,
            )
            attachments = attachments[:_MAX_ATTACHMENTS]
        self._printer.broadcast({"type": "setTaskText", "text": prompt, "tabId": tab_id})
        self._printer.broadcast({"type": "status", "running": True, "tabId": tab_id})
        run_cmd: dict[str, Any] = {
            "type": "run",
            "prompt": prompt,
            "model": cmd.get("model", ""),
            "workDir": cmd.get("workDir", self._vscode_server.work_dir),
            "tabId": tab_id,
            "attachments": attachments,
            "useWorktree": cmd.get("useWorktree", False),
            "useParallel": cmd.get("useParallel", False),
        }
        await self._run_cmd(run_cmd)

    def _register_merge_state(
        self, tab_id: str, merge_data: dict[str, Any],
    ) -> None:
        """Register a merge state when a merge_data event is broadcast.

        Called from ``WebPrinter.broadcast()`` so the web server can
        track active merge sessions and handle ``mergeAction`` commands.

        M6: takes :attr:`_merge_states_lock` because this runs on the
        agent task-runner thread while ``_handle_web_merge_action``
        and the ``_ws_handler`` cleanup mutate the same dict on the
        asyncio thread.

        Args:
            tab_id: The tab that started the merge.
            merge_data: The ``data`` field from the ``merge_data`` event.
        """
        with self._merge_states_lock:
            self._merge_states[tab_id] = _WebMergeState(merge_data)


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
        with self._merge_states_lock:
            state = self._merge_states.get(tab_id)
        if state is None:
            return

        assert self._loop is not None
        cur = state.current()
        if action == "accept":
            if cur is not None:
                state.mark_resolved(*cur, "accepted")
                state.advance()
        elif action == "reject":
            if cur is not None:
                fi, hi = cur
                fd = state.files[fi]
                hunk = fd["hunks"][hi]
                await self._loop.run_in_executor(
                    None,
                    _reject_hunk_in_file,
                    fd["current"],
                    fd["base"],
                    hunk,
                    fd.get("target"),
                )
                delta = hunk["bc"] - hunk["cc"]
                for later_hi in range(hi + 1, len(fd["hunks"])):
                    if not state.is_resolved(fi, later_hi):
                        fd["hunks"][later_hi]["cs"] += delta
                state.mark_resolved(fi, hi, "rejected")
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
                file_status = "rejected" if action == "reject-file" else "accepted"
                for hi in state.unresolved_in_file(fi):
                    state.mark_resolved(fi, hi, file_status)
                state.advance()
        elif action == "accept-all":
            for fi, hi in state.all_unresolved():
                state.mark_resolved(fi, hi, "accepted")
        elif action == "reject-all":
            unresolved_files: set[int] = set()
            for fi, hi in state.all_unresolved():
                unresolved_files.add(fi)
                state.mark_resolved(fi, hi, "rejected")
            for fi in unresolved_files:
                fd = state.files[fi]
                await self._loop.run_in_executor(
                    None, _reject_all_hunks_in_file, fd,
                )

        cur_after = state.current()
        self._printer.broadcast({
            "type": "merge_nav",
            "tabId": tab_id,
            "remaining": state.remaining,
            "total": state.total_hunks,
            "cur": (
                {"fi": cur_after[0], "hi": cur_after[1]}
                if cur_after is not None
                else None
            ),
            "resolved": state.resolutions(),
        })

        if not state.remaining:
            with self._merge_states_lock:
                self._merge_states.pop(tab_id, None)
            await self._run_cmd(
                {"type": "mergeAction", "action": "all-done", "tabId": tab_id},
            )


    def _spawn_cloudflared(self, args: list[str], retries: int = 3) -> None:
        """Spawn ``cloudflared`` with *args* and a free ``--metrics`` port.

        Records the subprocess in :attr:`_tunnel_proc`, the metrics
        port in :attr:`_tunnel_metrics_port`, and the start time in
        :attr:`_tunnel_started_at`.  The full argv is
        ``cloudflared tunnel --metrics 127.0.0.1:PORT`` followed by
        *args* (e.g. ``["--url", LOCAL, "--no-tls-verify"]`` for a
        quick tunnel or ``["run", "--token", TOKEN]`` for a named
        tunnel).

        M5: there is a small TOCTOU window between
        :func:`_pick_free_local_port` releasing its probe socket and
        ``cloudflared`` binding the same port — another process could
        grab the port in between, causing ``cloudflared`` to exit
        immediately.  When that happens the spawn is retried up to
        *retries* times with a freshly-picked port.

        Args:
            args: Extra arguments after ``--metrics 127.0.0.1:PORT``.
            retries: Maximum number of bind-failure retries.
        """
        last_proc: subprocess.Popen[str] | None = None
        for attempt in range(max(1, retries)):
            self._tunnel_metrics_port = _pick_free_local_port()
            # stdout is DEVNULL (not PIPE) because nothing reads it.
            # An un-drained PIPE buffer fills after ~64 KiB of
            # cloudflared log output and causes the subprocess to
            # block on write().
            proc = subprocess.Popen(
                [
                    "cloudflared", "tunnel",
                    "--metrics",
                    f"127.0.0.1:{self._tunnel_metrics_port}",
                    *args,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                # Detach cloudflared into its own process group / session
                # so it survives ``kiss-web``'s SIGTERM / SIGINT.  Combined
                # with the pidfile written below and adoption logic in
                # :func:`_try_adopt_existing_cloudflared`, this keeps the
                # public quick-tunnel URL stable across ``kiss-web``
                # restarts that would otherwise mint a brand-new
                # ``*.trycloudflare.com`` hostname every time.
                start_new_session=True,
            )
            # Give cloudflared a brief moment to fail-fast on a bind
            # collision.  Healthy cloudflared takes seconds to start
            # so it will not have exited within 250ms.
            time.sleep(0.25)
            if proc.poll() is None:
                self._tunnel_proc = proc
                self._tunnel_started_at = time.monotonic()
                self._tunnel_adopted_pid = None
                # Best-effort pidfile (URL added later once known).
                _save_cloudflared_pidfile(
                    proc.pid, self._tunnel_metrics_port, None,
                )
                return
            last_proc = proc
            logger.info(
                "cloudflared exited immediately on metrics port %d "
                "(attempt %d/%d, rc=%s); retrying with fresh port",
                self._tunnel_metrics_port, attempt + 1, retries,
                proc.returncode,
            )
        # All retries exhausted — keep the most recent (already-dead)
        # process so the caller's stderr-parsing path can still run
        # (it will time out and return None as before).
        self._tunnel_proc = last_proc
        self._tunnel_started_at = time.monotonic()

    def _start_tunnel(self) -> str | None:
        """Start a ``cloudflared`` tunnel and return the public URL.

        When :attr:`tunnel_token` is set, a **named tunnel** is started
        (fixed URL configured in the Cloudflare Zero Trust dashboard).
        Otherwise a **quick-tunnel** is started with a random
        ``*.trycloudflare.com`` URL.  The subprocess is stored in
        :attr:`_tunnel_proc` and must be terminated via
        :meth:`_stop_tunnel`.

        Returns:
            The public ``https://`` URL, or ``None`` if tunnel start
            fails (e.g. cloudflared missing, rate-limited, exited
            before registering).
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

        Spawns ``cloudflared tunnel --url`` and parses its stderr for
        the assigned URL.  If the URL never appears in stderr (e.g.
        log format changed across cloudflared versions), falls back to
        the cloudflared metrics ``/quicktunnel`` endpoint.

        Returns:
            The public ``https://`` URL, or ``None`` on failure.
        """
        self._spawn_cloudflared(
            ["--url", self._local_url, "--no-tls-verify"],
        )
        assert self._tunnel_proc is not None
        rate_limit_flag = [False]
        url = _read_url_from_stderr(
            self._tunnel_proc, _parse_quick_tunnel_url, timeout=30,
            rate_limit_flag=rate_limit_flag,
        )
        if url:
            assert self._tunnel_metrics_port is not None
            _save_cloudflared_pidfile(
                self._tunnel_proc.pid, self._tunnel_metrics_port, url,
            )
            return url
        for _ in range(20):
            if self._tunnel_proc.poll() is not None:
                break
            url = _discover_tunnel_url_from_metrics()
            if url:
                assert self._tunnel_metrics_port is not None
                _save_cloudflared_pidfile(
                    self._tunnel_proc.pid, self._tunnel_metrics_port, url,
                )
                return url
            time.sleep(1)
        # Rate-limit detection: if cloudflared's stderr named HTTP
        # 429 / error 1015 (and we never got a URL out), tell the
        # restart machinery to use a longer cooldown.  Setting this
        # only when ``url is None`` prevents a stray mid-line match
        # in healthy logs from poisoning the next backoff.
        if rate_limit_flag[0]:
            self._tunnel_rate_limited = True
            logger.warning(
                "cloudflared reported HTTP 429 / Cloudflare error "
                "1015 — Cloudflare is rate-limiting "
                "trycloudflare.com quick-tunnels for this egress IP",
            )
        return None

    def _start_named_tunnel(self) -> str | None:
        """Start a named tunnel using :attr:`tunnel_token`.

        The tunnel hostname is configured in the Cloudflare Zero Trust
        dashboard separately from the token.  Some ``cloudflared``
        builds echo the public hostname during startup (which
        :func:`_parse_named_tunnel_url` extracts) and some do not.
        When no hostname appears in logs but the tunnel reports a
        registered connection, :attr:`tunnel_url` is returned (or a
        sentinel string when no URL was pre-configured).

        Returns:
            The discovered or configured ``https://`` URL, the legacy
            sentinel string, or ``None`` if the subprocess exits
            before registering.
        """
        self._spawn_cloudflared(["run", "--token", self.tunnel_token or ""])
        assert self._tunnel_proc is not None
        url = _read_url_from_stderr(
            self._tunnel_proc,
            partial(_parse_named_tunnel_url, configured_url=self.tunnel_url),
            timeout=30,
        )
        if url and self._tunnel_metrics_port is not None:
            _save_cloudflared_pidfile(
                self._tunnel_proc.pid, self._tunnel_metrics_port, url,
            )
        return url

    async def _check_and_restart_tunnel(self) -> None:
        """Check tunnel health and restart if dead or deregistered.

        Called periodically by :meth:`_watchdog`.  Detects two failure
        modes:

        1. **Process dead** — ``cloudflared`` exited (e.g. macOS
           killed it during sleep).  Detected via ``poll()``.
        2. **Process alive but tunnel deregistered** — Cloudflare's
           edge dropped this tunnel's registration so the public
           hostname stops resolving (NXDOMAIN), but the local
           subprocess keeps retrying ``register_connection``.
           Detected by polling the ``/ready`` metrics endpoint for
           ``readyConnections > 0``; after
           :data:`_TUNNEL_UNHEALTHY_LIMIT_NAMED` (named tunnel) or
           :data:`_TUNNEL_UNHEALTHY_LIMIT_QUICK` (quick tunnel)
           consecutive zero-ticks the subprocess is force-terminated.

        During the first :data:`_TUNNEL_STARTUP_GRACE` seconds the
        metrics check is skipped (``readyConnections=0`` is expected
        while the tunnel is registering).  Failed (re)starts schedule
        an exponentially-growing backoff via :attr:`_tunnel_next_retry`
        so the watchdog stops hammering Cloudflare when rate-limited.
        """
        now = time.monotonic()
        proc = self._tunnel_proc
        adopted_pid = self._tunnel_adopted_pid

        if proc is not None and proc.poll() is not None:
            logger.info(
                "cloudflared tunnel process died (rc=%s), restarting…",
                proc.returncode,
            )
            self._terminate_tunnel_proc()
            proc = None

        # When we adopted an externally-owned cloudflared, treat a
        # vanished pid the same as a self-spawned proc that exited:
        # clear the adopted-pid bookkeeping so the no-process branch
        # below kicks in and a fresh cloudflared is spawned.
        if adopted_pid is not None and not _is_pid_alive(adopted_pid):
            logger.info(
                "Adopted cloudflared (pid=%d) is gone; restarting…",
                adopted_pid,
            )
            self._tunnel_adopted_pid = None
            self._tunnel_metrics_port = None
            self._tunnel_started_at = None
            self._tunnel_unhealthy_ticks = 0
            adopted_pid = None

        if proc is None and adopted_pid is None:
            # Don't (re)start the tunnel when the auth password is
            # unset — see H1 in the security review.  The watchdog
            # mirrors the guard in _setup_server so a runtime config
            # change doesn't accidentally bring up an open tunnel.
            if not load_config().get("remote_password", ""):
                return
            if now >= self._tunnel_next_retry:
                await self._restart_tunnel_url()
            return

        if (
            self._tunnel_started_at is not None
            and now - self._tunnel_started_at < _TUNNEL_STARTUP_GRACE
        ):
            return
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
        unhealthy_limit = (
            _TUNNEL_UNHEALTHY_LIMIT_NAMED
            if self.tunnel_token
            else _TUNNEL_UNHEALTHY_LIMIT_QUICK
        )
        logger.info(
            "cloudflared tunnel reports zero ready edge connections "
            "(tick %d/%d on metrics port %d)",
            self._tunnel_unhealthy_ticks,
            unhealthy_limit,
            self._tunnel_metrics_port,
        )
        if self._tunnel_unhealthy_ticks < unhealthy_limit:
            return

        logger.warning(
            "cloudflared tunnel appears deregistered from Cloudflare's "
            "edge (readyConnections=0 for %d ticks); force-restarting",
            self._tunnel_unhealthy_ticks,
        )
        # Kill the adopted cloudflared too — it is the unhealthy one
        # we are replacing; leaving it running would leak metrics
        # ports and confuse the next adoption attempt.
        self._terminate_tunnel_proc(kill_adopted=True)
        if now >= self._tunnel_next_retry:
            await self._restart_tunnel_url()

    async def _restart_tunnel_url(self) -> None:
        """Start a fresh tunnel and refresh ``~/.kiss/remote-url.json``.

        Always rewrites the URL file (even on failure, so stale data
        does not linger), updates :attr:`_active_url`, and broadcasts
        ``remote_url`` to connected clients.  On failure schedules an
        exponential backoff via :attr:`_tunnel_next_retry`.
        """
        assert self._loop is not None
        tunnel_url = await self._loop.run_in_executor(
            None, self._start_tunnel,
        )
        if tunnel_url:
            logger.info("Tunnel restarted: %s", tunnel_url)
            self._tunnel_failure_count = 0
            self._tunnel_next_retry = 0.0
            self._tunnel_rate_limited = False
        else:
            self._tunnel_failure_count += 1
            # When cloudflared reported a rate-limit on this attempt,
            # ignore the regular exponential schedule (60s, 120s, ...)
            # and wait at least _TUNNEL_RATE_LIMIT_BACKOFF + jitter
            # seconds: shorter waits keep resetting Cloudflare's per-IP
            # cooldown clock and burn through dozens of distinct
            # *.trycloudflare.com URLs without ever recovering.
            if self._tunnel_rate_limited:
                delay = _rate_limit_backoff_seconds()
                self._tunnel_rate_limited = False
                logger.warning(
                    "cloudflared rate-limited (HTTP 429 / error 1015) "
                    "on attempt %d; backing off %ds (long cooldown) "
                    "to let Cloudflare's per-IP quota clear",
                    self._tunnel_failure_count,
                    delay,
                )
            else:
                delay = _tunnel_backoff_delay(self._tunnel_failure_count)
                logger.warning(
                    "Failed to restart tunnel (attempt %d); "
                    "backing off %ds",
                    self._tunnel_failure_count,
                    delay,
                )
            self._tunnel_next_retry = time.monotonic() + delay
        _save_url_file(self._url_file, self._local_url, tunnel_url)
        self._active_url = tunnel_url or self._local_url
        ntfy_url = _get_ntfy_url()
        msg: dict[str, object] = {
            "type": "remote_url", "url": self._active_url,
        }
        if ntfy_url:
            msg["ntfyUrl"] = ntfy_url
        self._printer.broadcast(msg)
        if self.use_tunnel and self._active_url != self._last_posted_url:
            assert self._loop is not None
            await self._loop.run_in_executor(
                None, _post_url_to_message_board, self._active_url,
            )
            self._last_posted_url = self._active_url

    def _terminate_tunnel_proc(self, kill_adopted: bool = False) -> None:
        """Terminate ``_tunnel_proc`` and reset per-process state.

        Resets :attr:`_tunnel_proc`, :attr:`_tunnel_metrics_port`,
        :attr:`_tunnel_started_at`, and :attr:`_tunnel_unhealthy_ticks`
        so the next restart starts cleanly.  Leaves
        :attr:`_active_url` and the URL file alone so the file is not
        removed before a replacement tunnel writes its own URL.

        When *kill_adopted* is False (default — used on graceful
        kiss-web shutdown) and the current tunnel was *adopted* from a
        previous kiss-web, this method leaves the adopted cloudflared
        running so the next kiss-web can re-adopt it (this is the core
        of how the public URL survives kiss-web restarts).

        When *kill_adopted* is True (used by the unhealthy-tunnel
        watchdog before respawning a replacement), the adopted pid is
        sent SIGTERM, then SIGKILL after a short grace period if it is
        still alive, and the pidfile is removed.
        """
        proc = self._tunnel_proc
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            # Pidfile is stale now that we killed our own cloudflared.
            try:
                _CLOUDFLARED_PIDFILE.unlink(missing_ok=True)
            except OSError:
                pass
        elif kill_adopted and self._tunnel_adopted_pid is not None:
            adopted_pid = self._tunnel_adopted_pid
            try:
                os.kill(adopted_pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            else:
                for _ in range(50):
                    if not _is_pid_alive(adopted_pid):
                        break
                    time.sleep(0.1)
                else:
                    try:
                        os.kill(adopted_pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
            try:
                _CLOUDFLARED_PIDFILE.unlink(missing_ok=True)
            except OSError:
                pass
        self._tunnel_proc = None
        self._tunnel_adopted_pid = None
        self._tunnel_metrics_port = None
        self._tunnel_started_at = None
        self._tunnel_unhealthy_ticks = 0

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

        Each tick performs four checks:

        1. **Tunnel health** — if the ``cloudflared`` process died
           (e.g. macOS killed it during sleep), restart it.
        2. **URL-file presence** — re-write ``~/.kiss/remote-url.json``
           if it has been removed (e.g. by a developer's pytest run
           that touches the real file, or by an unrelated cleanup).
           Without this the VS Code settings panel's 10-second poller
           cannot discover the active URL.
        3. **IP change** — if the host's network addresses changed
           (WiFi switch, DHCP renewal, VPN), initiate a graceful
           shutdown so the daemon manager restarts the process.
        4. **WebSocket ping** — send a ping to every connected client
           and close connections that fail to respond within
           :data:`_WS_PING_TIMEOUT` seconds.
        """
        while True:
            await asyncio.sleep(TUNNEL_CHECK_INTERVAL)
            if self.use_tunnel:
                try:
                    await self._check_and_restart_tunnel()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("Watchdog tunnel check error", exc_info=True)
            try:
                if not self._url_file.is_file():
                    tunnel_url = (
                        self._active_url
                        if self._active_url and self._active_url != self._local_url
                        else None
                    )
                    _save_url_file(self._url_file, self._local_url, tunnel_url)
                    logger.info(
                        "Re-wrote missing URL file %s (tunnel=%s)",
                        self._url_file, tunnel_url,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog URL-file check error", exc_info=True)
            try:
                current_ips = _get_local_ips()
                if current_ips != self._last_ips:
                    self._last_ips = current_ips
                    if self.use_tunnel:
                        # In tunnel mode, cloudflared handles edge
                        # reconnection automatically.  Restarting the
                        # entire daemon would assign a new random
                        # *.trycloudflare.com URL — avoid that.  Just
                        # log and let cloudflared recover on its own.
                        # N.B. Check use_tunnel only — NOT
                        # _tunnel_proc, which can be None during
                        # startup, when remote_password is empty, or
                        # between a force-restart.
                        logger.info(
                            "IP address changed: %s → %s; tunnel mode "
                            "— cloudflared will re-register "
                            "automatically",
                            self._last_ips,
                            current_ips,
                        )
                    else:
                        logger.info(
                            "IP address changed: %s → %s, "
                            "restarting server…",
                            self._last_ips,
                            current_ips,
                        )
                        if self._ws_server is not None:
                            self._ws_server.close()
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog IP check error", exc_info=True)
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
        """Terminate the tunnel process and reset all tunnel state.

        Calls :meth:`_terminate_tunnel_proc` (which resets per-process
        state), then clears the backoff counters and active URL.  Does
        not delete ``~/.kiss/remote-url.json`` because a replacement
        daemon may have already overwritten it; removing it would
        race with the new instance's ``_save_url_file`` and cause the
        VS Code sidebar to show no URL.
        """
        self._terminate_tunnel_proc()
        self._tunnel_failure_count = 0
        self._tunnel_next_retry = 0.0
        self._tunnel_rate_limited = False
        self._active_url = None


    async def _setup_server(self) -> None:
        """Shared setup for both blocking and async server start.

        Binds the WebSocket server, starts the tunnel (if enabled),
        saves the URL file, and starts watchdog tasks.
        """
        # M1: ``asyncio.get_event_loop()`` is deprecated when there is a
        # running loop on Python 3.12+.  ``_setup_server`` only ever
        # runs from inside a coroutine driven by ``asyncio.run`` (or
        # an externally-managed loop), so the running loop is always
        # available here — use it directly.
        self._loop = asyncio.get_running_loop()
        self._printer._loop = self._loop

        self._ws_server = await serve(
            self._ws_handler,
            self.host,
            self.port,
            process_request=self._process_request,
            ssl=self._ssl_context,
            ping_interval=None,
            ping_timeout=None,
            max_size=_MAX_LINE_BYTES,
            create_connection=_HeadAwareServerConnection,
        )

        # Bind the local Unix-domain socket for the VS Code extension.
        # File permissions (mode 0o600) restrict access to the owning
        # user, replacing the WSS password challenge for local peers.
        try:
            self._uds_path.parent.mkdir(parents=True, exist_ok=True)
            if self._uds_path.exists() or self._uds_path.is_symlink():
                try:
                    self._uds_path.unlink()
                except OSError:
                    logger.debug(
                        "Could not unlink stale UDS socket at %s",
                        self._uds_path, exc_info=True,
                    )
            self._uds_server = await asyncio.start_unix_server(
                self._uds_handler, path=str(self._uds_path),
                limit=_MAX_LINE_BYTES,
            )
            os.chmod(self._uds_path, 0o600)
        except Exception:
            logger.warning(
                "Failed to bind UDS at %s; local extension clients "
                "will fall back to WSS",
                self._uds_path, exc_info=True,
            )
            self._uds_server = None

        tunnel_url: str | None = None
        if self.use_tunnel:
            # Refuse to expose the server to the public internet
            # (cloudflared tunnel) when no authentication password is
            # configured.  Without a password every WS client is
            # admitted, which would let anyone on the internet drive
            # the local agent.  Local connections still work.
            #
            # Wait up to 30 s for the password to be written.  When the
            # VS Code extension restarts ``kiss-web``, its
            # ``ensureRemotePassword`` prompt may not finish until a
            # few seconds after the daemon starts; polling here avoids
            # a launchd respawn that would mint a new tunnel URL.
            password = await self._loop.run_in_executor(  # type: ignore[union-attr]
                None, _wait_for_remote_password, 30.0,
            )
            if password:
                # Before spawning a fresh cloudflared, try to adopt one
                # left running by a previous ``kiss-web`` (detached via
                # ``start_new_session=True``).  Adoption keeps the
                # same public URL across kiss-web restarts.
                adopted = await self._loop.run_in_executor(  # type: ignore[union-attr]
                    None, _try_adopt_existing_cloudflared,
                )
                if adopted is not None:
                    adopted_pid, adopted_port, adopted_url = adopted
                    self._tunnel_adopted_pid = adopted_pid
                    self._tunnel_metrics_port = adopted_port
                    self._tunnel_started_at = time.monotonic()
                    tunnel_url = adopted_url
                    # Refresh pidfile with the freshly-probed URL so
                    # subsequent adoption attempts see the current
                    # hostname even when cloudflared has rotated it.
                    _save_cloudflared_pidfile(
                        adopted_pid, adopted_port, adopted_url,
                    )
            if not password:
                logger.warning(
                    "remote_password is not set in ~/.kiss/config.json; "
                    "refusing to start the cloudflared tunnel.  "
                    "Set a password in the config panel to enable "
                    "remote access.",
                )
                print(
                    "Warning: remote_password is empty; cloudflared "
                    "tunnel disabled.  Set a password to enable "
                    "remote access.",
                    file=sys.stderr,
                )
            elif tunnel_url is None:
                # No existing cloudflared was adopted — spawn a fresh
                # one.  When ``tunnel_url`` was already set by the
                # adoption branch above we deliberately skip this
                # call to avoid starting a duplicate cloudflared.
                tunnel_url = await self._loop.run_in_executor(  # type: ignore[union-attr]
                    None, self._start_tunnel,
                )

        _save_url_file(self._url_file, self._local_url, tunnel_url)
        self._active_url = tunnel_url or self._local_url
        if self.use_tunnel and self._active_url != self._last_posted_url:
            await self._loop.run_in_executor(
                None, _post_url_to_message_board, self._active_url,
            )
            self._last_posted_url = self._active_url

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
        # Cancel any armed deferred-close timers so a server shutdown
        # does not leave dangling ``call_later`` handles in the loop.
        with self._pending_tab_closes_lock:
            handles = list(self._pending_tab_closes.values())
            self._pending_tab_closes.clear()
        for handle in handles:
            try:
                handle.cancel()
            except Exception:
                logger.debug(
                    "Cancel of pending tab close on shutdown failed",
                    exc_info=True,
                )
        if self._ws_server is not None:
            self._ws_server.close()
            try:
                await asyncio.wait_for(self._ws_server.wait_closed(), timeout=2)
            except TimeoutError:
                pass
        if self._uds_server is not None:
            self._uds_server.close()
            try:
                await asyncio.wait_for(
                    self._uds_server.wait_closed(), timeout=2,
                )
            except TimeoutError:
                pass
            self._uds_server = None
            try:
                self._uds_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug(
                    "UDS unlink on shutdown failed", exc_info=True,
                )
        self._stop_tunnel()
        _remove_url_file(self._url_file)


def _resolve_tunnel_settings() -> tuple[str | None, str | None]:
    """Resolve the named-tunnel token and public URL.

    Reads the Cloudflare tunnel token from the
    ``CLOUDFLARE_TUNNEL_TOKEN`` env var first, falling back to the
    ``tunnel_token`` key in ``~/.kiss/config.json``.  The public URL
    is resolved the same way from ``CLOUDFLARE_TUNNEL_URL`` /
    ``tunnel_url``.  An env-var value takes precedence over the config
    value independently for each setting.

    Returns:
        A ``(token, url)`` pair where each element is the resolved
        string or ``None`` when neither env var nor config provides
        that setting.
    """
    token = os.environ.get("CLOUDFLARE_TUNNEL_TOKEN") or None
    url = os.environ.get("CLOUDFLARE_TUNNEL_URL") or None
    if token and url:
        return token, url
    cfg = load_config()
    if not token:
        token = cfg.get("tunnel_token") or None
    if not url:
        url = cfg.get("tunnel_url") or None
    return token, url


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

    tunnel_token, tunnel_url = _resolve_tunnel_settings()

    server = RemoteAccessServer(
        use_tunnel=True,
        tunnel_token=tunnel_token,
        tunnel_url=tunnel_url,
        work_dir=args.workdir,
    )
    server.start()


if __name__ == "__main__":
    main()
