# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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
import errno
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
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.vscode_config import load_config, source_shell_env
from kiss.core.config import get_jobs_root
from kiss.viz_trajectory.server import find_job_dir, list_jobs, load_job_trajectories

__all__ = ["RemoteAccessServer", "WebPrinter"]

logger = logging.getLogger(__name__)

MEDIA_DIR = Path(__file__).parent / "media"

# HTML page for the agent-trajectory visualizer, served at ``/trajectories/``.
TRAJECTORY_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "viz_trajectory"
    / "templates"
    / "index.html"
)

TUNNEL_CHECK_INTERVAL = 30

# Number of consecutive watchdog ticks that must observe the *same*
# new non-empty set of local IPs before the watchdog will treat it as
# a genuine network change and restart the server.  Without this
# debounce a single transient flake from :func:`_get_local_ips`
# (briefly empty result, DHCP renewal, VPN flap, post-sleep DNS hiccup)
# would force a spurious daemon restart on LAN-only deployments.  Four
# ticks at :data:`TUNNEL_CHECK_INTERVAL` = 120 s of sustained change.
# Earlier code used 2 ticks (60 s) which still let a real-but-brief
# VPN-connect / Ethernet↔WiFi handover that holds a new consistent
# IP set for ≥60 s trigger a daemon restart even if the address
# reverted seconds later.  Tests override both knobs to drive the
# loop faster.
_IP_CHANGE_DEBOUNCE_TICKS = 4

# Maximum number of attempts :meth:`RemoteAccessServer._setup_server`
# will make to bind the WSS listener before giving up.  Each transient
# ``OSError`` (most often ``EADDRINUSE`` lingering from a previous
# instance still in ``TIME_WAIT``, or ``EADDRNOTAVAIL`` while an
# interface is still coming up at boot / post-resume) is backed off
# via :data:`_BIND_RETRY_BACKOFF` between attempts.  After exhausting
# all retries the server exits with a structured ``SystemExit`` and a
# single-line error message — *no* traceback — so that a supervisor
# (launchd, systemd, the VS Code extension's respawn loop) sees a
# clean non-zero exit and backs off naturally, rather than respawning
# straight into the same OSError traceback in a tight flap loop.
_BIND_RETRY_ATTEMPTS = 5
# Per-attempt backoff in seconds; the value at index ``attempt`` is
# slept *before* attempt ``attempt + 2`` (i.e. between attempts).  The
# tuple is indexed by ``min(attempt, len(_BIND_RETRY_BACKOFF) - 1)``
# so a shorter override (in tests) still terminates the loop.
_BIND_RETRY_BACKOFF: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
# ``OSError.errno`` values worth retrying.  ``EADDRINUSE`` covers the
# common "previous kiss-web instance just SIGTERM'd, its port is in
# TIME_WAIT" case; ``EADDRNOTAVAIL`` covers a slow interface coming up
# at boot or after a network state change.  Other errnos (``EACCES``
# = privileged port without permission, ``ENOENT`` = bad UDS path,
# TLS load errors which surface as :class:`ssl.SSLError`, ...) are
# *not* retried — they will not be fixed by waiting and would only
# delay the inevitable failure exit.
_BIND_RETRYABLE_ERRNOS: frozenset[int] = frozenset({
    errno.EADDRINUSE, errno.EADDRNOTAVAIL,
})

# How often the server polls PyPI to learn whether a newer
# ``kiss-agent-framework`` release is available.  3600 s = once per
# hour, matching the "every hour" requirement.  Tests override this
# to a sub-second value to exercise the periodic loop quickly.
_VERSION_CHECK_INTERVAL: float = 3600

# PyPI JSON endpoint that reports the latest release of the project.
# Tests override this to point at a local stub HTTP server so no
# real network access is required.
_PYPI_LATEST_URL = "https://pypi.org/pypi/kiss-agent-framework/json"

# Timeout for the PyPI HTTP request — kept small so a slow PyPI
# response cannot stall the periodic loop for long.
_PYPI_FETCH_TIMEOUT = 5.0

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

# After the watchdog force-restarts cloudflared because it observed
# ``readyConnections == 0`` for the unhealthy-tick limit, wait at
# least this many seconds before allowing another force-restart even
# if the new cloudflared instance is *also* immediately unhealthy.
# Without this cool-down a chronically-flaky cloudflared metrics
# endpoint (or a Cloudflare edge that briefly drops every fresh
# quick-tunnel registration) rotates the public ``*.trycloudflare.com``
# URL every ~10 minutes (= quick-tunnel limit × tick interval) forever,
# breaking long-lived browser sessions and pushing a steady stream of
# stale URLs to the message board.  Each consecutive force-restart
# without a sustained healthy period in between doubles the wait, up
# to :data:`_TUNNEL_FORCE_RESTART_COOLDOWN_MAX`.
_TUNNEL_FORCE_RESTART_COOLDOWN_INITIAL = 60

_TUNNEL_FORCE_RESTART_COOLDOWN_MAX = 3600

# Once a freshly-restarted cloudflared has been continuously healthy
# for this many seconds since its ``_tunnel_started_at`` we consider
# the chronic-flake episode over and reset the consecutive
# force-restart counter so a future, *unrelated* event starts at the
# minimum cool-down.
_TUNNEL_FORCE_RESTART_RESET_AFTER_HEALTHY = 600

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

_KISS_HOME = Path(os.environ.get("KISS_HOME") or (Path.home() / ".kiss"))
_TLS_DIR = _KISS_HOME / "tls"
_URL_FILE = _KISS_HOME / "remote-url.json"

# Path to the localhost Unix-domain socket exposed by
# :class:`RemoteAccessServer` in addition to the public WSS port.
# Local clients (the VS Code extension) connect to this socket over
# the SAME newline-delimited JSON protocol that browsers speak over
# WSS — no password challenge is performed because POSIX filesystem
# permissions (mode 0o600) restrict access to the owning user.  A
# fresh ``RemoteAccessServer(uds_path=...)`` argument overrides this
# location for tests so multiple instances do not race on the same
# socket file.
_UDS_PATH = _KISS_HOME / "sorcar.sock"


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
        # The tab's repository (or worktree) directory, stamped by the
        # backend ``_start_merge_session``.  Echoed back on the
        # ``all-done`` ``mergeAction`` so the post-merge autocommit scan
        # runs against the tab's own repo rather than the daemon-wide
        # ``self.work_dir`` (which may be a different, non-git folder).
        self.work_dir: str = merge_data.get("work_dir", "")
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
    # ``sizeReport`` is the webview's reply to the extension-only
    # ``measureSize`` request; it never has meaning for the web
    # server but must not surface as "Unknown command" if a client
    # ever emits it.
    "sizeReport",
})

# Canonical KISS Sorcar source-checkout root.  The curl-piped
# bootstrapper (``scripts/install.sh``) clones the GitHub repo to this
# fixed location and ``install.sh`` — which the Update button re-runs —
# lives at its root.  Mirrors ``kissAiRoot()`` in the VS Code
# extension's ``installerPath.js`` so the extension and the remote
# webapp resolve the updater identically.
_KISS_AI_ROOT = Path.home() / "kiss_ai"


def _find_install_script(root: Path) -> Path | None:
    """Return ``install.sh`` inside *root* if it exists, else ``None``.

    Python twin of ``findInstallScript()`` in the extension's
    ``installerPath.js`` so the remote webapp's Update button probes
    the exact same location as the VS Code extension.

    Args:
        root: Directory expected to contain ``install.sh`` (production
            callers pass :data:`_KISS_AI_ROOT`; tests pass a temp dir).

    Returns:
        The absolute script path, or ``None`` when missing/unreadable.
    """
    candidate = root / "install.sh"
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


def _query_quicktunnel_hostname(metrics_port: int) -> str | None:
    """Ask a cloudflared metrics endpoint for its quick-tunnel URL.

    Queries ``http://127.0.0.1:{metrics_port}/quicktunnel`` and returns
    the public ``https://`` URL built from the reported hostname, or
    ``None`` when the endpoint is unreachable, the response is
    malformed, or the hostname is empty / Cloudflare's ``api.``
    endpoint (which cloudflared reports before the real tunnel URL).

    Args:
        metrics_port: Port of cloudflared's ``--metrics`` endpoint.

    Returns:
        The ``https://`` tunnel URL, or ``None`` if unavailable.
    """
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{metrics_port}/quicktunnel",
            headers={"User-Agent": "kiss-web"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            hostname = data.get("hostname", "")
            if hostname and not hostname.startswith("api."):
                return f"https://{hostname}"
    except Exception:
        return None
    return None


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
            encoding="utf-8",
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
        url = _query_quicktunnel_hostname(port)
        if url:
            return url
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


_CLOUDFLARED_PIDFILE = _KISS_HOME / "cloudflared.pid"


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


def _unlink_cloudflared_pidfile() -> None:
    """Best-effort removal of the cloudflared pidfile.

    Used once the recorded cloudflared process is known to be dead so
    a later kiss-web does not try to adopt a stale pid.  Failures are
    ignored — the worst case is a stale pidfile that the next adoption
    attempt rejects via its pid-liveness check.
    """
    try:
        _CLOUDFLARED_PIDFILE.unlink(missing_ok=True)
    except OSError:
        pass


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
    url = _query_quicktunnel_hostname(metrics_port)
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


def _probe_tunnel_ready(metrics_port: int) -> bool | None:
    """Return tunnel readiness as a 3-valued result.

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

    The previous version of this helper returned a plain ``bool`` and
    folded every error (connection refused, timeout, parse error,
    schema change) into ``False`` — which the watchdog then counted
    as "unhealthy" and used to force-restart cloudflared.  On a slow
    CPU after wake, during a post-sleep socket-rebind window, or just
    a momentary 127.0.0.1 loopback hiccup, this conflated "endpoint
    unreachable" with "tunnel deregistered" and was the single
    biggest source of spurious quick-tunnel URL rotation.  Returning
    ``None`` for "no information" lets callers skip the tick entirely
    instead of incrementing their unhealthy-streak counter.

    Args:
        metrics_port: The port on which ``cloudflared`` is serving its
            metrics HTTP endpoint (passed via ``--metrics``).

    Returns:
        ``True`` if the endpoint reports ``readyConnections > 0``.
        ``False`` if the endpoint *successfully* reports
        ``readyConnections == 0`` (confirmed deregistration).
        ``None`` if the endpoint is unreachable, the response is not
        valid JSON, or the value is non-numeric — callers should treat
        this as "no information" and *not* count it toward an unhealthy
        streak.
    """
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{metrics_port}/ready",
            headers={"User-Agent": "kiss-web"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    try:
        return int(data.get("readyConnections", 0)) > 0
    except (TypeError, ValueError):
        return None


def _stderr_reader_loop(
    stderr: Any,
    parse: Callable[[str], str | None],
    result: list[str | None],
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
            stderr, parse, result, stop_event, rate_limit_flag,
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


_NTFY_BASE_URL = "https://ntfy.sh"


def _fetch_last_ntfy_message(
    topic: str, base_url: str = _NTFY_BASE_URL,
) -> str | None:
    """Return the most recent message body posted to ``{base_url}/{topic}``.

    Queries ntfy.sh's poll endpoint (``/{topic}/json?poll=1``) which
    returns cached messages (default retention 12h) as newline-
    delimited JSON.  Only entries with ``event == "message"`` are
    considered; the last one wins because the server returns events
    in chronological order.

    Args:
        topic: ntfy.sh topic name (without leading slash).
        base_url: Override the ntfy server URL (used by tests).

    Returns:
        The body (``message`` field) of the most recent cached
        message, or ``None`` if the topic has no cached messages or
        the request fails.
    """
    try:
        req = urllib.request.Request(
            f"{base_url}/{topic}/json?poll=1",
            headers={"User-Agent": "kiss-web"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception:
        logger.debug("Failed to fetch last ntfy message", exc_info=True)
        return None
    last: str | None = None
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") != "message":
            continue
        msg = obj.get("message")
        if isinstance(msg, str):
            last = msg
    return last


def _post_url_to_message_board(
    url: str, base_url: str = _NTFY_BASE_URL,
) -> None:
    """Post the active Cloudflare URL to ntfy.sh as a private message.

    Uses the machine-stable topic from :func:`_get_machine_topic` so
    the URL can be retrieved by subscribing to the same topic.  The
    message is posted with a title indicating it is a KISS Sorcar
    remote URL update.  Before posting, the most recent cached
    message on the topic is fetched via :func:`_fetch_last_ntfy_message`;
    if it already matches ``url`` the post is skipped so subscribers
    are not woken up by duplicate notifications when a watchdog
    restart or named-tunnel re-registration produces the same public
    hostname.  Failures are logged but never raised.

    Args:
        url: The ``https://`` URL to publish.
        base_url: Override the ntfy server URL (used by tests).
    """
    if not url or url.startswith("https://localhost"):
        return
    try:
        topic = _get_machine_topic()
        last = _fetch_last_ntfy_message(topic, base_url=base_url)
        if last is not None and last.strip() == url.strip():
            logger.info(
                "Skipping ntfy.sh post for %s; last message on "
                "topic %s already has the same URL", url, topic,
            )
            return
        data = url.encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/{topic}",
            data=data,
            method="POST",
            headers={
                "Title": "KISS Sorcar Remote URL",
                "Tags": "link,kiss-sorcar",
                # Make the notification itself clickable: tapping the
                # notification (mobile) or the message card (web UI)
                # opens the URL in a browser.  Without this header
                # ntfy renders the body as plain text that is not
                # auto-linked, so users cannot navigate to the site
                # by clicking it.
                "Click": url,
                "User-Agent": "kiss-web",
            },
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        logger.info("Posted remote URL to ntfy.sh/%s", topic)
    except Exception:
        logger.debug("Failed to post URL to ntfy.sh", exc_info=True)


def _get_local_ips() -> frozenset[str]:
    """Return the current routable IPv4 addresses of the host machine.

    Uses a UDP connect to ``8.8.8.8`` (no packet is actually sent) to
    discover the default-route IP, plus :func:`socket.getaddrinfo` on
    the hostname for any additional addresses.  The raw discovery is
    then filtered to drop addresses that should never trigger a
    server restart:

    *   ``127.0.0.0/8`` loopback — never a useful LAN address.
    *   ``169.254.0.0/16`` link-local — auto-assigned when DHCP fails
        or while an interface is still negotiating.  These addresses
        come and go during boot, sleep/wake, captive portals and
        VPN flaps, which used to surface as spurious "IP changed"
        events from :meth:`RemoteAccessServer._watchdog`.
    *   IPv4-mapped IPv6 addresses in dotted form (e.g.
        ``"::ffff:1.2.3.4"``) — returned by :func:`socket.getaddrinfo`
        on dual-stack hosts as the same underlying IPv4 address; the
        ``::ffff:`` prefix would make them look like a *new* address
        each time the family-preference oscillated, again causing
        spurious change events.

    Returns:
        A frozen set of routable IPv4 address strings (e.g.
        ``frozenset({"192.168.1.42"})``).  Returns an empty set when
        discovery failed or all discovered addresses were filtered.
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
            ips.add(addr)
    except Exception:
        pass
    filtered: set[str] = set()
    for addr in ips:
        if addr.startswith("127."):
            continue
        if addr.startswith("169.254."):
            continue
        if addr.startswith("::ffff:"):
            continue
        filtered.add(addr)
    return frozenset(filtered)


def _print_url() -> None:
    """Print the active remote URL from ``~/.kiss/remote-url.json``.

    Prints the tunnel URL if available, otherwise the local URL.
    Exits with code 1 if the server is not running or the file is
    missing.
    """
    url = _read_url_from_file(_URL_FILE)
    if url:
        print(url)
    else:
        print("KISS Sorcar web server is not running.", file=sys.stderr)
        sys.exit(1)


def _snapshot_active_tabs() -> list[str]:
    """Return ``"<tabId>(task=<task_id>)"`` strings for active tabs.

    Snapshots the running-agent registry under its lock before
    iterating so a concurrent worker thread mutating
    ``running_agent_states`` (registering a fresh tab, disposing a
    finished one) cannot race the iterator and raise ``RuntimeError:
    dictionary changed size during iteration``.  ``_registry_lock`` is
    a :class:`threading.RLock`, so re-entry from the same thread is
    safe even when called from a signal handler that interrupted a
    lock holder.  Falls back to a best-effort unlocked snapshot if the
    lock itself is unusable (e.g. during interpreter shutdown), and
    skips malformed tab entries rather than propagating, so callers —
    the shutdown-signal logger and the ``activeTasksQuery`` handler —
    always get a usable (possibly partial) report.
    """
    from kiss.agents.sorcar.running_agent_state import _RunningAgentState

    try:
        with _RunningAgentState._registry_lock:
            items = list(_RunningAgentState.running_agent_states.items())
    except Exception:
        items = list(_RunningAgentState.running_agent_states.items())
    active_tabs: list[str] = []
    for tab_id, tab in items:
        try:
            if tab.is_task_active:
                task_id = tab.task_history_id or tab.last_task_id
                active_tabs.append(f"{tab_id}(task={task_id})")
        except Exception:
            logger.debug(
                "skipping malformed tab entry in active-task snapshot",
                exc_info=True,
            )
    return active_tabs


def _rss_mb() -> float:
    """Return this process's peak RSS in megabytes, or ``-1.0`` on failure.

    ``ru_maxrss`` is reported in bytes on macOS and in kilobytes on
    Linux; both are normalised to MB.
    """
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return -1.0


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


class WebPrinter(JsonPrinter):
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
          ``history``, ``inputHistory``, ``error``, etc.) and are
          broadcast verbatim to every connected client.

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

    Loads ``media/chat.html`` — the exact same template the VS Code
    extension's ``SorcarTab.buildChatHtml`` reads — and substitutes
    remote-mode values (no CSP, plain ``/media/`` URLs, ``loading...``
    model name, the auth-modal block, and the WebSocket shim that
    provides ``acquireVsCodeApi()`` for ``main.js``).

    Sharing the markup with the extension guarantees the two HTML
    pages cannot drift in script ordering or DOM ids — the bug that
    previously broke the tab bar, the ``+`` button and the send-task
    flow on the remote webapp.

    Returns:
        The complete HTML string.
    """
    version = _read_version()
    tricks_json = json.dumps(_read_tricks())
    head_style = (
        "<style>\n"
        "    html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; }\n"
        "    body { background: var(--vscode-editor-background, #1e1e1e);\n"
        "            color: var(--vscode-editor-foreground, #cccccc); }\n"
        "    :root {\n"
        "      --vscode-font-size: 16px;\n"
        "      --vscode-font-family: -apple-system, BlinkMacSystemFont, "
        "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;\n"
        "      --vscode-editor-font-size: 16px;\n"
        "      --vscode-editor-background: #1e1e1e;\n"
        "      --vscode-editor-foreground: #cccccc;\n"
        "      --vscode-input-background: #3c3c3c;\n"
        "      --vscode-input-foreground: #cccccc;\n"
        "      --vscode-input-border: #3c3c3c;\n"
        "      --vscode-focusBorder: #007acc;\n"
        "      --vscode-button-background: #0e639c;\n"
        "      --vscode-button-foreground: #ffffff;\n"
        "      --vscode-button-hoverBackground: #1177bb;\n"
        "      --vscode-sideBar-background: #252526;\n"
        "      --vscode-list-hoverBackground: #2a2d2e;\n"
        "      --vscode-badge-background: #4d4d4d;\n"
        "      --vscode-badge-foreground: #ffffff;\n"
        "      --vscode-textLink-foreground: #3794ff;\n"
        "      --vscode-descriptionForeground: #8b8b8b;\n"
        "      --vscode-editorWidget-background: #252526;\n"
        "      --vscode-editorWidget-border: #454545;\n"
        "      --vscode-panel-border: #80808059;\n"
        "      --vscode-terminal-ansiRed: #f44747;\n"
        "      --vscode-terminal-ansiGreen: #6a9955;\n"
        "      --vscode-terminal-ansiYellow: #d7ba7d;\n"
        "      --vscode-terminal-ansiBlue: #569cd6;\n"
        "      --vscode-terminal-ansiMagenta: #c586c0;\n"
        "      --vscode-terminal-ansiCyan: #4ec9b0;\n"
        "    }\n"
        "  </style>"
    )
    auth_modal = (
        '    <div id="auth-modal" style="display:none;">\n'
        '      <div class="auth-modal-content">\n'
        '        <div class="auth-modal-title">Remote access password</div>\n'
        '        <input type="password" id="auth-modal-input" '
        'class="auth-modal-input"\n'
        '               autocomplete="current-password" '
        'placeholder="Enter password">\n'
        '        <div class="auth-modal-actions">\n'
        '          <button id="auth-modal-cancel" '
        'class="auth-modal-btn auth-modal-cancel"\n'
        '                  type="button">Cancel</button>\n'
        '          <button id="auth-modal-ok" '
        'class="auth-modal-btn auth-modal-ok"\n'
        '                  type="button">OK</button>\n'
        '        </div>\n'
        '      </div>\n'
        '    </div>\n'
    )
    subs = {
        "VIEWPORT": "width=device-width,initial-scale=1,maximum-scale=1",
        "CSP_META": "",
        "STYLE_HREF": "/media/main.css",
        "HLJS_CSS_HREF": "/media/highlight-github-dark.min.css",
        "HEAD_STYLE": head_style,
        "BODY_CLASS_ATTR": ' class="remote-chat"',
        "INPUT_PLACEHOLDER": "Ask anything... (@ for files)",
        "ENTERKEYHINT": ' enterkeyhint="send"',
        "MODEL_NAME": "loading...",
        "VERSION_SUFFIX": f" {version}" if version else "",
        "AUTH_MODAL": auth_modal,
        "NONCE_ATTR": "",
        "HLJS_SRC": "/media/highlight.min.js",
        "MARKED_SRC": "/media/marked.min.js",
        "PANEL_COPY_SRC": "/media/panelCopy.js",
        "MAIN_SRC": "/media/main.js",
        "DEMO_SRC": "/media/demo.js",
        "SHIM_SCRIPT": f"<script>{_WS_SHIM_JS}</script>\n  ",
        "TRICKS_JSON": tricks_json,
    }
    tpl = (MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
    for key, value in subs.items():
        tpl = tpl.replace("{{" + key + "}}", value)
    return tpl


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


def _version_tuple(v: str) -> tuple[int, ...] | None:
    """Return ``v`` parsed as an int-tuple, or ``None`` on failure.

    ``kiss-agent-framework`` uses CalVer ``YYYY.M.P`` so a simple
    ``int`` split on ``.`` is sufficient; ``None`` is returned for
    anything that cannot be parsed so a malformed PyPI payload never
    triggers a false "update available" notification.
    """
    try:
        parts = [int(p) for p in v.strip().split(".") if p != ""]
    except (ValueError, AttributeError):
        return None
    return tuple(parts) if parts else None


def _compare_versions(a: str, b: str) -> int:
    """Compare two CalVer/SemVer-ish version strings.

    Returns ``1`` when *a* > *b*, ``-1`` when *a* < *b*, ``0`` when
    they compare equal (including the case where either is
    unparseable — see :func:`_version_tuple`).  Shorter tuples are
    right-padded with zeros so ``"2026.6"`` and ``"2026.6.0"`` are
    equal.
    """
    ta, tb = _version_tuple(a), _version_tuple(b)
    if ta is None or tb is None:
        return 0
    n = max(len(ta), len(tb))
    ta = ta + (0,) * (n - len(ta))
    tb = tb + (0,) * (n - len(tb))
    if ta > tb:
        return 1
    if ta < tb:
        return -1
    return 0


def _fetch_latest_version() -> str | None:
    """Fetch the latest ``kiss-agent-framework`` version from PyPI.

    Returns the version string on success, ``None`` on any error
    (network failure, malformed JSON, missing key).  Callers must
    treat ``None`` as "no information" — never as "no update".
    """
    try:
        req = urllib.request.Request(
            _PYPI_LATEST_URL,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(  # noqa: S310 — fixed PyPI URL
            req, timeout=_PYPI_FETCH_TIMEOUT,
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        logger.debug("PyPI version fetch failed", exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    info = data.get("info")
    if not isinstance(info, dict):
        return None
    version = info.get("version")
    if not isinstance(version, str) or not version.strip():
        return None
    return version.strip()


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
        HTTPStatus(status).phrase,
        Headers([
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Connection", "close"),
        ]),
        body,
    )


def _trajectory_jobs_response() -> Response:
    """Return a JSON HTTP response listing all trajectory jobs.

    Mirrors the ``/api/jobs`` endpoint of the standalone trajectory
    visualizer (:mod:`kiss.viz_trajectory.server`).

    Returns:
        A 200 ``application/json`` response with the job list.
    """
    body = json.dumps(list_jobs(get_jobs_root())).encode("utf-8")
    return _http_response(200, "application/json", body)


def _trajectory_job_response(path: str) -> Response:
    """Return a JSON HTTP response with the trajectories for one job.

    Mirrors the ``/api/jobs/<job_name>/trajectories`` endpoint of the
    standalone trajectory visualizer.

    Args:
        path: Request path of the form ``/api/jobs/<job_name>/trajectories``.

    Returns:
        A 200 ``application/json`` response with the trajectory list, a 400
        response for an invalid job name, or a 404 response when the job
        directory does not exist.
    """
    job_name = unquote(path[len("/api/jobs/") : -len("/trajectories")])
    if "/" in job_name or "\\" in job_name or ".." in job_name:
        return _http_response(
            400, "application/json", b'{"error": "Invalid job name"}'
        )
    jobs_root = get_jobs_root()
    if find_job_dir(jobs_root, job_name) is None:
        body = json.dumps({"error": f"Job '{job_name}' not found"}).encode("utf-8")
        return _http_response(404, "application/json", body)
    body = json.dumps(load_job_trajectories(jobs_root, job_name)).encode("utf-8")
    return _http_response(200, "application/json", body)


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
        # Counts consecutive force-restarts of cloudflared driven by
        # the unhealthy-tick watchdog without a sustained healthy
        # period in between.  Used together with
        # ``_tunnel_force_restart_next_allowed`` to apply an
        # exponentially-growing cool-down so a chronically-flaky
        # metrics endpoint (or an edge that keeps dropping fresh
        # quick-tunnel registrations) cannot rotate
        # ``*.trycloudflare.com`` URLs every ~10 minutes forever.
        # The counter is reset to 0 by the next ``healthy`` probe
        # observed after ``_TUNNEL_FORCE_RESTART_RESET_AFTER_HEALTHY``
        # seconds of post-restart uptime.
        self._tunnel_force_restart_count = 0
        self._tunnel_force_restart_next_allowed = 0.0
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
        # Cached PyPI version learned by :meth:`_check_for_update`.
        # ``None`` until the first poll completes; ``""`` if PyPI is
        # unreachable.  ``_send_welcome_info`` replays the cached
        # ``update_available`` event so a client that connects between
        # polls still learns the current state without waiting for
        # the next hour's tick.
        self._latest_version: str | None = None
        self._version_check_task: asyncio.Task[None] | None = None
        # Set True by :meth:`_handle_shutdown_signal` the first time a
        # SIGTERM is caught.  Guards against a *second* SIGTERM (e.g.
        # an impatient ``pkill`` loop) re-raising ``KeyboardInterrupt``
        # while :meth:`start` is already in its ``finally`` cleanup —
        # which would escape the cleanup uncaught, abort
        # ``subprocess.wait`` mid-sleep, and crash the process with an
        # unhandled traceback (killing any running agent task).
        self._shutdown_initiated = False
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
        # Per-tab asyncio lock serialising :meth:`_handle_web_merge_action`.
        # Both the local UDS handler (VS Code extension) and the remote
        # WebSocket handler dispatch ``mergeAction`` commands on the SAME
        # event loop.  When two clients act on the *same* tab's merge
        # review concurrently, the two coroutines would otherwise
        # interleave at the ``run_in_executor`` await inside the reject
        # branches — both reading the same ``current()`` hunk, both
        # rejecting it, and leaving a later hunk permanently unresolved
        # (a lost update).  Holding this per-tab lock for the whole
        # action body makes the read-modify-write atomic per tab.
        # Lazily created in :meth:`_merge_action_lock`; the dict itself
        # is guarded by ``self._merge_states_lock`` above.
        self._merge_action_locks: dict[str, asyncio.Lock] = {}
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
        # Debounce state for the IP-change watchdog.  ``_pending_ip_change``
        # holds the most recent *candidate* new IP set observed by the
        # watchdog (``None`` when no change is pending), and
        # ``_pending_ip_change_count`` counts how many consecutive ticks
        # have observed that same candidate.  Only when the count reaches
        # :data:`_IP_CHANGE_DEBOUNCE_TICKS` does the watchdog accept the
        # candidate as the new baseline and (in LAN mode) restart the
        # server.  See :meth:`_watchdog` for the full state machine.
        self._pending_ip_change: frozenset[str] | None = None
        self._pending_ip_change_count: int = 0
        # Per-IP auth failure timestamps, used by _is_auth_locked.
        self._auth_failures: dict[str, list[float]] = {}
        # Where the ``runUpdate`` command looks for ``install.sh`` and
        # where it appends the updater's output.  Mirrors the VS Code
        # extension's ``installerPath.js`` lookup; tests override both
        # to temp paths.
        self._install_root: Path = _KISS_AI_ROOT
        self._update_log_path: Path = _KISS_HOME / "update.log"


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
        if path == "/trajectories" or path == "/trajectories/":
            return _http_response(
                200,
                "text/html; charset=utf-8",
                TRAJECTORY_TEMPLATE.read_bytes(),
            )
        if path == "/api/jobs":
            return _trajectory_jobs_response()
        if path.startswith("/api/jobs/") and path.endswith("/trajectories"):
            return _trajectory_job_response(path)
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
            # Two attempts: the first wrong password elicits an
            # ``auth_required`` retry prompt; the second failure (or a
            # non-auth message on the retry) closes the connection.
            for is_retry, timeout in ((False, 30), (True, 60)):
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                msg = json.loads(raw)
                client_pw = msg.get("password", "")
                if not isinstance(client_pw, str):
                    client_pw = ""
                if msg.get("type") == "auth" and self._passwords_equal(
                    password, client_pw,
                ):
                    await websocket.send(json.dumps({"type": "auth_ok"}))
                    return True
                if not is_retry and msg.get("type") != "auth":
                    # First message was not an auth attempt at all:
                    # close without counting it as a failed login.
                    await websocket.close()
                    return False
                self._record_auth_failure(ip)
                if not is_retry:
                    await websocket.send(json.dumps({"type": "auth_required"}))
            await websocket.send(
                json.dumps({"type": "error", "text": "Authentication failed"})
            )
            await websocket.close()
            return False
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
            self._merge_action_locks.pop(tab_id, None)
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
        tabs_seen: set[str] = set()
        # Per-connection work_dir (see _dispatch_client_command).
        conn_state: dict[str, str] = {"work_dir": ""}
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if not isinstance(cmd, dict):
                    continue
                await self._dispatch_client_command(
                    cmd, websocket, tabs_seen, conn_state,
                )
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
            for tab in tabs_seen:
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
        # Per-connection work_dir: each VS Code window owns exactly one
        # UDS connection, so recording the window's ``setWorkDir`` here
        # (instead of only on the daemon-global fallback) is what keeps
        # every window's work_dir == its own workspace folder even when
        # several windows share this one daemon process.
        conn_state: dict[str, str] = {"work_dir": ""}
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
                await self._dispatch_client_command(
                    cmd, writer, tabs_seen, conn_state,
                )
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


    async def _dispatch_client_command(
        self,
        cmd: dict[str, Any],
        endpoint: Any,
        tabs_seen: set[str],
        conn_state: dict[str, str],
    ) -> None:
        """Dispatch one parsed client command from a WSS or UDS peer.

        Single shared per-message dispatch body for
        :meth:`_ws_handler` (remote browsers) and :meth:`_uds_handler`
        (the local VS Code extension), so the two transports cannot
        drift in behaviour.  Records the command's ``tabId`` in
        *tabs_seen* (used by the callers' ``finally`` blocks to arm
        deferred ``closeTab`` timers), drops VS Code-only webview
        messages, special-cases the commands the TypeScript extension
        would otherwise translate (``ready``, ``submit``,
        ``getWelcomeSuggestions``, ``runUpdate``, ``mergeAction``,
        ``activeTasksQuery``), and forwards everything else through
        :func:`_translate_webview_command` to
        :class:`VSCodeServer._handle_command`.

        Args:
            cmd: The parsed JSON command dictionary.
            endpoint: The client connection — a
                :class:`ServerConnection` (WSS) or an
                :class:`asyncio.StreamWriter` (UDS).  Used for direct
                replies via :meth:`_endpoint_send`.
            tabs_seen: Per-connection set of tab ids, mutated in place.
            conn_state: Per-connection mutable state holding the
                connection's own ``work_dir``.  Each VS Code window
                owns exactly one connection and announces its
                workspace folder via ``setWorkDir``; every later
                command from the same connection that does not carry
                an explicit ``workDir`` is stamped with it here.  This
                is what guarantees the per-window work_dir invariant:
                two windows sharing this daemon can never observe each
                other's folder through the daemon-global fallback,
                because their commands always arrive pre-stamped with
                their own connection's work_dir.
        """
        tab_id = cmd.get("tabId", "")
        if isinstance(tab_id, str) and tab_id:
            tabs_seen.add(tab_id)
        cmd_type = cmd.get("type", "")
        if cmd_type == "setWorkDir":
            new_wd = cmd.get("workDir", "")
            if isinstance(new_wd, str) and new_wd:
                conn_state["work_dir"] = new_wd
        elif conn_state["work_dir"] and not cmd.get("workDir"):
            cmd["workDir"] = conn_state["work_dir"]
        if cmd_type in _VSCODE_ONLY_COMMANDS:
            return
        if cmd_type == "activeTasksQuery":
            await self._handle_active_tasks_query(endpoint)
            return
        if cmd_type == "ready":
            await self._handle_ready(cmd, endpoint)
            return
        if cmd_type == "submit":
            await self._handle_submit(cmd)
            return
        if cmd_type == "getWelcomeSuggestions":
            await self._send_welcome_info()
            return
        if cmd_type == "runUpdate":
            await self._handle_run_update()
            return
        if cmd_type == "mergeAction" and cmd.get("action", "") != "all-done":
            await self._handle_web_merge_action(cmd)
            return
        cmd = _translate_webview_command(cmd)
        await self._run_cmd(cmd)

    async def _handle_run_update(self) -> None:
        """Run ``~/kiss_ai/install.sh`` to update KISS Sorcar.

        Server-side twin of the VS Code extension's
        ``SorcarSidebarView._runUpdate()``: the extension locates the
        installer via ``installerPath.js`` and runs it in an integrated
        terminal; the web server locates it via
        :func:`_find_install_script` and runs it as a detached
        subprocess (output appended to ``~/.kiss/update.log``) since a
        remote browser has no terminal.  Error/info wording matches
        the extension's ``showErrorMessage``/``showInformationMessage``
        so both frontends behave the same.
        """
        loop = self._loop
        assert loop is not None
        script = await loop.run_in_executor(
            None, _find_install_script, self._install_root,
        )
        if script is None:
            self._printer.broadcast({
                "type": "error",
                "text": (
                    "Cannot update KISS Sorcar: install.sh not found "
                    f"in {self._install_root}."
                ),
            })
            return
        self._printer.broadcast({
            "type": "notice",
            "text": (
                "An update of KISS Sorcar is getting installed… "
                f"(output: {self._update_log_path})"
            ),
        })
        await loop.run_in_executor(None, self._spawn_update_script, script)

    def _spawn_update_script(self, script: Path) -> None:
        """Start ``install.sh`` detached, logging to the update log.

        Runs in the executor so file I/O and process spawn never block
        the event loop.  ``start_new_session=True`` keeps the updater
        alive when ``install.sh`` restarts this very daemon.  Failures
        are broadcast as ``error`` events instead of raised.

        Args:
            script: Absolute path of the ``install.sh`` to execute.
        """
        try:
            self._update_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._update_log_path, "ab") as log:
                subprocess.Popen(
                    ["bash", str(script)],
                    cwd=str(script.parent),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            self._printer.broadcast({
                "type": "error",
                "text": f"Failed to start KISS Sorcar update: {exc}",
            })

    async def _handle_active_tasks_query(self, endpoint: Any) -> None:
        """Report in-flight agent tasks back to a single client.

        Used by the VS Code extension's dependency installer before it
        considers SIGTERMing the daemon: when any task is still active,
        the extension must defer the restart so that an in-progress
        agent run is not interrupted by ``ensureDependencies()`` on a
        spurious re-activation of the extension.

        The response is a single JSON object sent directly to the
        requesting *endpoint* — a UDS writer or a WSS connection, via
        :meth:`_endpoint_send` — i.e. not broadcast to other clients.
        It has the shape::

            {"type": "activeTasksResponse",
             "count": <int>,
             "tabs": ["<tabId>(task=<task_id>)", ...]}

        Inactive tabs are filtered out; ``count`` is the length of the
        ``tabs`` list, matching the format emitted by the signal-
        handler log line above.
        """
        active_tabs = _snapshot_active_tabs()
        payload = json.dumps({
            "type": "activeTasksResponse",
            "count": len(active_tabs),
            "tabs": active_tabs,
        })
        try:
            await self._endpoint_send(endpoint, payload)
        except Exception:
            logger.debug(
                "activeTasksQuery: failed to write response", exc_info=True,
            )


    def _broadcast_remote_url(self, url: str, tunnel_active: bool) -> None:
        """Broadcast a ``remote_url`` event to every connected client.

        Includes the ``ntfyUrl`` field only when both *url* is
        non-empty and an ntfy topic is configured, matching the
        contract pinned by the welcome-info and tunnel-restart tests.

        Args:
            url: The active URL (``""`` when none is known).
            tunnel_active: True only when a real Cloudflare tunnel
                URL is in effect (not the local fallback).
        """
        ntfy_url = _get_ntfy_url() if url else ""
        msg: dict[str, object] = {
            "type": "remote_url",
            "url": url or "",
            "tunnelActive": tunnel_active,
        }
        if ntfy_url:
            msg["ntfyUrl"] = ntfy_url
        self._printer.broadcast(msg)

    def _broadcast_update_available(self) -> None:
        """Broadcast the cached PyPI ``update_available`` state.

        No-op until :meth:`_check_for_update` has cached a latest
        version on :attr:`_latest_version`.
        """
        latest = self._latest_version
        if not latest:
            return
        current = _read_version()
        available = bool(current) and _compare_versions(latest, current) > 0
        self._printer.broadcast({
            "type": "update_available",
            "available": available,
            "latest": latest,
            "current": current,
        })

    async def _post_url_if_changed(self) -> None:
        """Post :attr:`_active_url` to the ntfy message board once.

        Skips the post when tunneling is disabled or the URL is
        unchanged since the last post, so a watchdog restart that
        yields the same public hostname does not re-notify
        subscribers.
        """
        url = self._active_url
        if self.use_tunnel and url is not None and url != self._last_posted_url:
            assert self._loop is not None
            await self._loop.run_in_executor(
                None, _post_url_to_message_board, url,
            )
            self._last_posted_url = url

    async def _send_welcome_info(self) -> None:
        """Broadcast the active remote URL to all connected clients.

        Broadcasts the ``remote_url`` event using the in-memory URL,
        the URL file, or the ``cloudflared`` metrics API as successive
        fallbacks.

        Historically this method also broadcast a
        ``welcome_suggestions`` event with an empty list because the
        remote-chat webview hides the sample-task suggestions panel
        via CSS (``body.remote-chat #welcome > #suggestions { display:
        none }``).  That broadcast was redundant for the webapp and
        actively harmful for the VS Code extension: the extension is
        a *second* client of the same broadcaster (over its UDS
        connection), and it populates its own ``#suggestions``
        container locally from ``SAMPLE_TASKS.json``.  The empty-list
        broadcast was forwarded to the extension's webview and
        cleared every chip on the welcome page whenever any webapp
        client opened a new chat tab — see
        ``test_welcome_suggestions_not_broadcast.py``.

        M10: the URL-file read and the ``_discover_tunnel_url_from_metrics``
        call (which spawns ``pgrep`` and does HTTP requests) are
        blocking I/O.  They run in :meth:`asyncio.AbstractEventLoop.run_in_executor`
        so a slow ``pgrep`` or unreachable cloudflared metrics
        endpoint cannot stall the asyncio event loop.
        """
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
        # ``tunnelActive`` is True only when a real Cloudflare tunnel
        # URL is in effect (not the local fallback).  The frontend hides
        # the welcome-page remote-password panel when this is False so
        # users are not shown a password field for a tunnel that does
        # not exist.
        tunnel_active = bool(
            self.use_tunnel and url and url != self._local_url
        )
        self._broadcast_remote_url(url or "", tunnel_active)
        # Replay the cached PyPI update-available state so a client
        # that just (re)connected sees the green download badge on
        # the Update button without having to wait for the next
        # hourly poll.
        self._broadcast_update_available()

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
            # ``or`` (not ``dict.get`` default) so an explicit empty
            # ``workDir`` also falls back to the daemon-wide default.
            # Commands from VS Code windows arrive pre-stamped with the
            # window's own work_dir by ``_dispatch_client_command``.
            "workDir": cmd.get("workDir") or self._vscode_server.work_dir,
            "tabId": tab_id,
            "attachments": attachments,
            "useWorktree": cmd.get("useWorktree", False),
            "useParallel": cmd.get("useParallel", False),
            # Mirror the extension's ``_startTask``: the webview's
            # "Auto commit" toggle must survive the submit → run
            # translation or remote submits silently lose the mode.
            "autoCommit": cmd.get("autoCommit", False),
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


    def _merge_action_lock(self, tab_id: str) -> asyncio.Lock:
        """Return the per-tab :class:`asyncio.Lock` serialising merge actions.

        Lazily creates a lock for *tab_id* on first use.  Creation is
        guarded by :attr:`_merge_states_lock` (a threading lock) so two
        coroutines that request the lock for the same tab in the same
        event-loop tick still share one lock instance.

        Args:
            tab_id: The frontend tab id whose merge review is acted on.

        Returns:
            The shared :class:`asyncio.Lock` for *tab_id*.
        """
        with self._merge_states_lock:
            lock = self._merge_action_locks.get(tab_id)
            if lock is None:
                lock = asyncio.Lock()
                self._merge_action_locks[tab_id] = lock
            return lock

    async def _handle_web_merge_action(self, cmd: dict[str, Any]) -> None:
        """Handle merge toolbar actions (accept/reject/navigate) server-side.

        In VS Code, the TypeScript ``MergeManager`` processes these
        actions.  In the standalone web server, this method provides
        equivalent functionality by tracking hunk state and modifying
        files on disk.

        Serialised per tab via :meth:`_merge_action_lock` so two clients
        (the local UDS VS Code extension and a remote WebSocket browser)
        acting on the *same* tab's merge review cannot interleave at the
        ``run_in_executor`` await inside the reject branches and drop a
        hunk resolution.

        Args:
            cmd: The ``mergeAction`` command from the browser, with
                ``action`` and ``tabId`` fields.
        """
        tab_id = cmd.get("tabId", "")
        async with self._merge_action_lock(tab_id):
            await self._apply_web_merge_action(cmd)

    async def _apply_web_merge_action(self, cmd: dict[str, Any]) -> None:
        """Apply a single merge action while holding the per-tab lock.

        Performs the read-modify-write on the per-tab
        :class:`_WebMergeState`.  Must be called by
        :meth:`_handle_web_merge_action` with the tab's
        :meth:`_merge_action_lock` held so the whole sequence (including
        the ``run_in_executor`` file rewrites) is atomic per tab.

        Args:
            cmd: The ``mergeAction`` command, with ``action`` and
                ``tabId`` fields.
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
                {
                    "type": "mergeAction",
                    "action": "all-done",
                    "tabId": tab_id,
                    "workDir": state.work_dir,
                },
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
                encoding="utf-8",
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
        if healthy is None:
            # Metrics endpoint unreachable, response malformed, or
            # schema changed.  This is "no information" — do NOT
            # increment ``_tunnel_unhealthy_ticks`` (which would
            # eventually force a brand-new ``*.trycloudflare.com``
            # URL on every flake of the loopback metrics socket).
            return
        if healthy:
            self._tunnel_unhealthy_ticks = 0
            # After enough sustained healthy uptime, treat the prior
            # chronic-flake episode as resolved and let a future,
            # unrelated event start the cool-down ladder from rung 1.
            if (
                self._tunnel_force_restart_count > 0
                and self._tunnel_started_at is not None
                and now - self._tunnel_started_at
                    > _TUNNEL_FORCE_RESTART_RESET_AFTER_HEALTHY
            ):
                self._tunnel_force_restart_count = 0
                self._tunnel_force_restart_next_allowed = 0.0
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

        if now < self._tunnel_force_restart_next_allowed:
            # The watchdog already force-restarted cloudflared recently
            # and the replacement is *still* unhealthy.  Skip the
            # force-restart this tick: rotating the public URL every
            # ~10 minutes for a problem that has not gone away just
            # breaks every existing browser session without recovering
            # the tunnel.  The cool-down grows exponentially so a
            # genuinely-dead edge will eventually be force-restarted.
            remaining = int(self._tunnel_force_restart_next_allowed - now)
            logger.info(
                "cloudflared tunnel still reports zero ready edge "
                "connections, but a force-restart was attempted "
                "recently; deferring the next force-restart for "
                "~%ds (consecutive force-restarts: %d)",
                remaining,
                self._tunnel_force_restart_count,
            )
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
        self._tunnel_force_restart_count += 1
        cooldown = min(
            _TUNNEL_FORCE_RESTART_COOLDOWN_INITIAL
                * (2 ** (self._tunnel_force_restart_count - 1)),
            _TUNNEL_FORCE_RESTART_COOLDOWN_MAX,
        )
        self._tunnel_force_restart_next_allowed = now + cooldown
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
        self._broadcast_remote_url(self._active_url, bool(tunnel_url))
        await self._post_url_if_changed()

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
            _unlink_cloudflared_pidfile()
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
            _unlink_cloudflared_pidfile()
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

    async def _check_for_update(self) -> None:
        """Poll PyPI and broadcast an ``update_available`` event.

        Fetches the latest ``kiss-agent-framework`` version (in a
        background executor so the blocking ``urllib`` call cannot
        stall the asyncio loop), caches it on ``self._latest_version``,
        and broadcasts an ``update_available`` event of the form
        ``{"type": "update_available", "available": bool,
            "latest": str, "current": str}`` to every connected client.

        Called both at startup and periodically by
        :meth:`_version_check_loop`.
        """
        loop = self._loop
        assert loop is not None
        latest = await loop.run_in_executor(None, _fetch_latest_version)
        if not latest:
            # Network error or malformed payload — keep the previous
            # cached state (if any) and do nothing.  Re-broadcasting
            # a stale ``False`` would mask a genuine update.
            return
        self._latest_version = latest
        self._broadcast_update_available()

    async def _version_check_loop(self) -> None:
        """Run :meth:`_check_for_update` every hour.

        The very first check runs immediately so clients learn about
        a pending upgrade as soon as the daemon starts, instead of
        waiting an entire hour for the first tick.
        """
        try:
            await self._check_for_update()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Initial version check failed", exc_info=True)
        while True:
            await asyncio.sleep(_VERSION_CHECK_INTERVAL)
            try:
                await self._check_for_update()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Periodic version check failed", exc_info=True)

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
                self._watchdog_check_url_file()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog URL-file check error", exc_info=True)
            try:
                if self._watchdog_check_ip_change():
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog IP check error", exc_info=True)
            try:
                await self._watchdog_ping_clients()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog WS ping error", exc_info=True)

    def _watchdog_check_url_file(self) -> None:
        """Re-write ``~/.kiss/remote-url.json`` if it went missing.

        A developer's pytest run that touches the real file, or an
        unrelated cleanup, can remove it; without a re-write the VS
        Code settings panel's 10-second poller cannot discover the
        active URL.
        """
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

    def _watchdog_check_ip_change(self) -> bool:
        """Detect a debounced local-IP change and initiate a restart.

        Compares the current :func:`_get_local_ips` result against the
        established baseline in :attr:`_last_ips`, requiring
        :data:`_IP_CHANGE_DEBOUNCE_TICKS` consecutive ticks observing
        the *same* new non-empty set before acting (see the module
        docstring of ``test_web_server_ip_watchdog_debounce.py``).

        Returns:
            True when a restart was initiated (the WSS listener has
            been closed and the watchdog loop must exit); False
            otherwise.
        """
        current_ips = _get_local_ips()
        if not current_ips:
            # IP discovery failed (transient WiFi roam, DHCP
            # renewal, VPN flap, slow resolver, post-sleep DNS
            # hiccup).  Bare ``try/except: pass`` inside
            # :func:`_get_local_ips` swallows any error and
            # returns ``frozenset()``.  Treat this as "no
            # information" — do NOT compare it against
            # ``_last_ips`` (which would falsely look like a
            # network change and trigger a spurious restart),
            # and drop any in-flight debounce candidate so a
            # later real change starts its count from scratch.
            self._pending_ip_change = None
            self._pending_ip_change_count = 0
        elif current_ips == self._last_ips:
            # IPs match the established baseline — clear any
            # in-flight candidate (the host's network briefly
            # diverged and recovered before reaching the
            # debounce threshold).
            self._pending_ip_change = None
            self._pending_ip_change_count = 0
        elif not self._last_ips:
            # Initial discovery (or recovery after a sustained
            # failure) — adopt the first non-empty result as
            # the baseline without restarting.  Without this
            # branch a server that started before the network
            # came up would always trigger one spurious restart
            # on the first successful poll.
            self._last_ips = current_ips
            self._pending_ip_change = None
            self._pending_ip_change_count = 0
        else:
            # Real divergence from the established baseline.
            # Require :data:`_IP_CHANGE_DEBOUNCE_TICKS`
            # consecutive ticks observing the *same* candidate
            # set before acting.  A single divergent tick (the
            # most common spurious-restart trigger) is no
            # longer enough.
            if current_ips == self._pending_ip_change:
                self._pending_ip_change_count += 1
            else:
                self._pending_ip_change = current_ips
                self._pending_ip_change_count = 1
            if self._pending_ip_change_count >= _IP_CHANGE_DEBOUNCE_TICKS:
                prev_ips = self._last_ips
                self._last_ips = current_ips
                self._pending_ip_change = None
                self._pending_ip_change_count = 0
                if self.use_tunnel:
                    # In tunnel mode, cloudflared handles edge
                    # reconnection automatically.  Restarting
                    # the entire daemon would assign a new
                    # random *.trycloudflare.com URL — avoid
                    # that.  Just log and let cloudflared
                    # recover on its own.
                    # N.B. Check use_tunnel only — NOT
                    # _tunnel_proc, which can be None during
                    # startup, when remote_password is empty,
                    # or between a force-restart.
                    logger.info(
                        "IP address changed: %s → %s; tunnel "
                        "mode — cloudflared will re-register "
                        "automatically",
                        prev_ips,
                        current_ips,
                    )
                else:
                    logger.info(
                        "IP address changed: %s → %s, "
                        "restarting server…",
                        prev_ips,
                        current_ips,
                    )
                    if self._ws_server is not None:
                        self._ws_server.close()
                    return True
        return False

    async def _watchdog_ping_clients(self) -> None:
        """Ping every connected WSS client, closing unresponsive ones.

        Delegates the per-connection timeout/close logic to
        :meth:`_ping_one_ws`; exceptions from individual pings are
        collected via ``return_exceptions`` so one bad client cannot
        skip the rest.
        """
        if self._ws_server is not None:
            connections = list(self._ws_server.connections)
            if connections:
                await asyncio.gather(
                    *[self._ping_one_ws(ws) for ws in connections],
                    return_exceptions=True,
                )

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

        # Bind the WSS listener with bounded retry on transient
        # ``OSError`` (most often ``EADDRINUSE`` lingering from a
        # previous instance still in ``TIME_WAIT`` during a fast
        # launchd / extension respawn, or ``EADDRNOTAVAIL`` while the
        # network interface is still coming up post-boot / post-
        # resume).  Without retry, the first crash propagates an
        # OSError traceback out of ``asyncio.run`` and the supervisor
        # immediately respawns kiss-web into the same OSError, visible
        # to the user as a flap loop.  After ``_BIND_RETRY_ATTEMPTS``
        # we raise :class:`SystemExit` with a clean error message and
        # no traceback so the supervisor backs off naturally.
        last_err: OSError | None = None
        for attempt in range(_BIND_RETRY_ATTEMPTS):
            try:
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
                break
            except OSError as exc:
                # Permission errors (EACCES on a privileged port) and
                # other non-transient errnos are not worth retrying —
                # fail fast so the supervisor sees the error promptly.
                if exc.errno not in _BIND_RETRYABLE_ERRNOS:
                    logger.error(
                        "WSS bind to %s:%d failed with non-retryable "
                        "errno %s: %s",
                        self.host, self.port, exc.errno, exc,
                    )
                    print(
                        f"Error: cannot bind to {self.host}:{self.port}: "
                        f"{exc}",
                        file=sys.stderr,
                    )
                    raise SystemExit(2) from exc
                last_err = exc
                # Was this the final attempt?  Skip the sleep and
                # fall through to the SystemExit below.
                if attempt + 1 >= _BIND_RETRY_ATTEMPTS:
                    break
                delay = _BIND_RETRY_BACKOFF[
                    min(attempt, len(_BIND_RETRY_BACKOFF) - 1)
                ]
                logger.warning(
                    "WSS bind to %s:%d failed (attempt %d/%d, "
                    "errno=%s): %s — retrying in %.1fs",
                    self.host, self.port, attempt + 1,
                    _BIND_RETRY_ATTEMPTS, exc.errno, exc, delay,
                )
                await asyncio.sleep(delay)
        if self._ws_server is None:
            # All retryable attempts exhausted.  Exit cleanly so the
            # supervisor backs off naturally instead of respawning
            # into the same OSError traceback in a flap loop.
            logger.error(
                "WSS bind to %s:%d failed after %d attempts: %s — exiting",
                self.host, self.port, _BIND_RETRY_ATTEMPTS, last_err,
            )
            print(
                f"Error: cannot bind to {self.host}:{self.port} after "
                f"{_BIND_RETRY_ATTEMPTS} attempts: {last_err}",
                file=sys.stderr,
            )
            raise SystemExit(2)

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
        await self._post_url_if_changed()

        self._last_ips = _get_local_ips()
        self._watchdog_task = asyncio.create_task(self._watchdog())
        self._version_check_task = asyncio.create_task(
            self._version_check_loop(),
        )

    async def _serve_async(self) -> None:
        """Internal async entry point for the server."""
        await self._setup_server()
        print(f"KISS Sorcar remote access: {self._local_url}", file=sys.stderr)
        if self.use_tunnel and self._active_url != self._local_url:
            print(f"Cloudflare tunnel:         {self._active_url}", file=sys.stderr)
        elif self.use_tunnel:
            print("Warning: cloudflared tunnel failed to start", file=sys.stderr)
        await self._ws_server.serve_forever()  # type: ignore[union-attr]

    def _handle_shutdown_signal(
        self, signum: int, _frame: Any = None,
    ) -> None:
        """React to a catchable termination signal (SIGTERM / SIGHUP).

        Logs the signal alongside a snapshot of in-flight agent tasks
        (via :func:`_snapshot_active_tabs`, which is signal-safe) and
        current memory.  For ``SIGTERM`` the *first* invocation raises
        :class:`KeyboardInterrupt` so the ``asyncio.run`` loop in
        :meth:`start` unwinds through its existing
        ``except KeyboardInterrupt`` handler and runs its cleanup.

        A subsequent SIGTERM that arrives *while shutdown is already in
        progress* must NOT raise again.  During the ``finally`` cleanup,
        :meth:`_stop_tunnel` blocks in ``subprocess.wait`` (a
        ``time.sleep`` loop).  A second SIGTERM delivered then — e.g. by
        an impatient ``pkill``/supervisor restart loop — would otherwise
        re-raise ``KeyboardInterrupt`` inside that sleep, escape the
        ``finally`` block uncaught, and crash the process with an
        unhandled traceback (abruptly killing any running agent task).
        Once :attr:`_shutdown_initiated` is set we therefore only log
        and return so the cleanup runs to completion.

        Args:
            signum: The signal number delivered by the OS.
            _frame: The interrupted stack frame (unused; present so
                the method can be registered with ``signal.signal``
                directly).
        """
        sig_name = signal.Signals(signum).name
        active_tabs = _snapshot_active_tabs()
        logger.warning(
            "Signal %s received: pid=%d active_tasks=[%s] rss=%.1fMB",
            sig_name,
            os.getpid(),
            ", ".join(active_tabs) if active_tabs else "none",
            _rss_mb(),
        )
        # For SIGTERM: raise KeyboardInterrupt so the asyncio loop can
        # unwind cleanly through the existing try/except — but only on
        # the first signal.  Re-raising during the cleanup phase would
        # crash the process (see docstring).
        if signum == signal.SIGTERM:
            if self._shutdown_initiated:
                logger.info(
                    "SIGTERM during shutdown ignored: pid=%d "
                    "(cleanup already in progress)",
                    os.getpid(),
                )
                return
            self._shutdown_initiated = True
            raise KeyboardInterrupt(f"Received {sig_name}")

    def _stop_active_agent_tasks(self, timeout: float = 12.0) -> None:
        """Stop in-flight agent worker threads so they unwind cleanly.

        Each task runs in a daemon worker thread spawned by
        :meth:`VSCodeServer._run_task`.  On process exit those daemon
        threads are killed abruptly, skipping the cleanup ``finally``
        that persists a meaningful ``task_history.result`` and
        broadcasts the outcome.  The row is then left at the
        ``"Agent Failed Abruptly"`` sentinel and the next startup's
        orphan sweep rewrites it to ``"Task terminated unexpectedly
        (process killed)"`` — a *silent* failure the user never sees in
        real time.

        This method reproduces what the user-facing "stop" button does
        (set the cooperative stop event, then inject a
        ``KeyboardInterrupt`` into the worker thread via
        ``PyThreadState_SetAsyncExc``) but, crucially, **joins** each
        worker synchronously so its cleanup ``finally`` runs to
        completion (persisting ``"Task stopped by user"`` and
        broadcasting a final result) before the process exits.

        The total time spent is bounded by *timeout* seconds across all
        workers so a thread wedged in uninterruptible C code cannot hang
        shutdown indefinitely.

        Args:
            timeout: Maximum wall-clock seconds to wait, in aggregate,
                for all active worker threads to unwind.
        """
        import ctypes

        # Declare the C signature once so the ``PyThreadState_SetAsyncExc``
        # calls below marshal arguments correctly.
        ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
            ctypes.c_ulong,
            ctypes.py_object,
        ]

        from kiss.agents.sorcar.running_agent_state import _RunningAgentState

        active: list[tuple[str, threading.Event | None, threading.Thread]] = []
        active_task_history_ids: set[int] = set()
        with _RunningAgentState._registry_lock:
            for tab_id, tab in _RunningAgentState.running_agent_states.items():
                thread = tab.task_thread
                if tab.is_task_active and thread is not None and thread.is_alive():
                    # Mark BEFORE the cooperative stop / injected
                    # KeyboardInterrupt so the worker's
                    # ``except KeyboardInterrupt`` handler reliably
                    # observes that this cancellation is a graceful
                    # server shutdown (SIGTERM / daemon restart), not a
                    # user "Stop" click.  This is the sole signal that
                    # lets the task-runner persist
                    # "Task interrupted by server restart/shutdown"
                    # (event ``task_interrupted``) instead of
                    # "Task stopped by user" (event ``task_stopped``).
                    tab.interrupted_by_shutdown = True
                    active.append((tab_id, tab.stop_event, thread))
                    # Capture the in-flight task row id so the helper
                    # below can pre-emptively rewrite its sentinel
                    # ``"Agent Failed Abruptly"`` ``result`` column to
                    # ``"Task interrupted by server restart/shutdown"``.
                    # ``tab.task_history_id`` is only set in the
                    # task_runner's inner finally *after* the agent's
                    # ``run()`` returns — but the agent itself sets
                    # ``self._last_task_id`` early in ``run()`` (see
                    # ``ChatSorcarAgent.run``) so a worker wedged
                    # mid-``run()`` is still recoverable via the
                    # agent's own attribute.
                    th_id = tab.task_history_id
                    if th_id is None and tab.agent is not None:
                        th_id = getattr(tab.agent, "_last_task_id", None)
                    if th_id is not None:
                        active_task_history_ids.add(int(th_id))

        if not active:
            return

        # Pre-emptive persistence safety net.  Done BEFORE we start
        # signalling workers so that — even if a worker is wedged in
        # uninterruptible C code (a blocking LLM API call) and never
        # reaches its cleanup ``finally`` within *timeout* — the
        # task_history row already carries a meaningful result rather
        # than the sentinel.  Without this net, the next startup's
        # orphan sweep would rewrite the surviving sentinel to
        # ``"Task terminated unexpectedly (process killed)"`` — the
        # intermittent "agent was killed" symptom users observe.
        # Workers that *do* finish unwinding will overwrite this
        # placeholder with a more detailed result via
        # ``_save_task_result``.
        if active_task_history_ids:
            try:
                from kiss.agents.sorcar.persistence import (
                    _shutdown_persist_in_flight_results,
                )

                _shutdown_persist_in_flight_results(active_task_history_ids)
            except Exception:  # noqa: BLE001 — best-effort, must not block shutdown
                logger.debug(
                    "Pre-emptive shutdown persistence failed",
                    exc_info=True,
                )

        logger.warning(
            "Shutdown: stopping %d in-flight agent task(s) before exit: %s",
            len(active),
            ", ".join(tab_id for tab_id, _, _ in active),
        )

        # Phase 1: cooperative stop signal for every worker first.
        for _tab_id, stop_event, _thread in active:
            if stop_event is not None:
                stop_event.set()

        # Phase 2: force a KeyboardInterrupt in any worker that did not
        # exit on its own, then join (bounded by the shared deadline) so
        # the worker's cleanup finally runs before we return.
        deadline = time.monotonic() + timeout
        for tab_id, _stop_event, thread in active:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=min(1.0, remaining))
            if thread.is_alive():
                tid = thread.ident
                if tid is not None:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(tid),
                        ctypes.py_object(KeyboardInterrupt),
                    )
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                logger.warning(
                    "Shutdown: agent task %s did not stop within timeout; "
                    "it may be persisted as a process-killed task",
                    tab_id,
                )

    def _install_signal_handlers(self) -> None:
        """Register handlers for catchable termination signals.

        SIGKILL cannot be caught, but SIGTERM (``pkill``, ``systemd
        stop``) and SIGHUP (terminal closed) can — and are the most
        common non-OOM kill causes.  Both are routed through
        :meth:`_handle_shutdown_signal`.  Registration is best-effort:
        it silently no-ops when not on the main thread or when the
        signal is unsupported on the current platform.
        """
        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, self._handle_shutdown_signal)
            except (OSError, ValueError):
                pass  # e.g. not main thread, or unsupported on this OS

    def start(self) -> None:
        """Start the server (blocks until interrupted).

        Call this from the main thread.  Press Ctrl-C to stop.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        pid = os.getpid()
        logger.info(
            "Server starting: pid=%d python=%s platform=%s "
            "work_dir=%s host=%s port=%d",
            pid,
            sys.version.split()[0],
            platform.platform(),
            self.work_dir,
            self.host,
            self.port,
        )
        logger.info("Initial memory: rss=%.1fMB pid=%d", _rss_mb(), pid)

        self._install_signal_handlers()

        try:
            asyncio.run(self._serve_async())
        except KeyboardInterrupt:
            logger.info("Server shutting down: pid=%d (KeyboardInterrupt)", pid)
        finally:
            # Gracefully stop any in-flight agent worker thread BEFORE
            # the process exits.  Without this the daemon worker thread
            # is killed abruptly when ``start()`` returns, its cleanup
            # ``finally`` (which persists a real result + broadcasts the
            # outcome) never runs, and the task_history row is left at
            # the ``"Agent Failed Abruptly"`` sentinel — later rewritten
            # by the orphan sweep to "Task terminated unexpectedly
            # (process killed)".  That is the silent-failure mode this
            # fix eliminates (see tasks 2968 in the bundled sorcar.db).
            self._stop_active_agent_tasks()
            logger.info("Server stopped: pid=%d", pid)
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
        if self._version_check_task is not None:
            self._version_check_task.cancel()
            try:
                await self._version_check_task
            except asyncio.CancelledError:
                pass
            self._version_check_task = None
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
