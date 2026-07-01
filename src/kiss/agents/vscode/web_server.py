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
from typing import Any, cast
from urllib.parse import unquote, urlsplit

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from kiss.agents.vscode.diff_merge import _read_lines_preserved
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.vscode_config import load_config, source_shell_env
from kiss.core.config import get_jobs_root
from kiss.viz_trajectory.server import find_job_dir, list_jobs, load_job_trajectories

__all__ = ["RemoteAccessServer", "WebPrinter"]

logger = logging.getLogger(__name__)

MEDIA_DIR = Path(__file__).parent / "media"
_MEDIA_VERSION_CACHE: dict[str, str] = {}

# HTML page for the agent-trajectory visualizer, served at ``/trajectories/``.
TRAJECTORY_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "viz_trajectory"
    / "templates"
    / "index.html"
)

TUNNEL_CHECK_INTERVAL = 15

# Number of consecutive watchdog ticks that must observe the *same*
# new non-empty set of local IPs before the watchdog will treat it as
# a genuine network change and restart the server.  Without this
# debounce a single transient flake from :func:`_get_local_ips`
# (briefly empty result, DHCP renewal, VPN flap, post-sleep DNS hiccup)
# would force a spurious daemon restart on LAN-only deployments.  Four
# ticks at :data:`TUNNEL_CHECK_INTERVAL` = 60 s of sustained change.
# Earlier code used 2 ticks (30 s) which still let a real-but-brief
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

# Seconds to wait after acknowledging a "Server reset" request before
# SIGTERMing this daemon, so the ``notification`` event flushes to the
# clicking window before its socket drops on shutdown.
_SERVER_RESET_DELAY = 0.4

# Seconds after the freshly-restarted daemon binds its listeners
# before it broadcasts the "Server restart complete" notification.
# Long enough for the VS Code extension's ``AgentClient`` and any
# reconnecting browser webview to reattach to the UDS / WSS so the
# toast is delivered into a live socket instead of being dropped on
# the floor.  Tests monkey-patch this constant to a tiny value so
# they do not have to wait the full window.
_SERVER_RESET_COMPLETE_DELAY = 3.0

# File name of the pending-reset flag dropped by
# :meth:`RemoteAccessServer._handle_server_reset` in the same
# directory as ``remote-url.json`` (``~/.kiss`` in production, a
# tmpdir in tests via ``url_file=``).  Its presence at daemon
# startup means the previous instance SIGTERMed itself in response
# to a user-initiated "Server reset" — the freshly-restarted daemon
# deletes the flag and schedules a "Server restart complete"
# notification to reconnecting clients.  Absent flag ⇒ this is a
# regular launch / crash restart / install respawn and the
# completion toast is intentionally suppressed.
_SERVER_RESET_FLAG_NAME = "server-reset-pending.json"

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
        # Full ``data`` payload of the opening ``merge_data`` event,
        # kept so an in-flight review can be replayed verbatim to a
        # client that reconnects mid-review (browser reload).  The
        # ``hunks`` dicts inside are shared (not copied): reject
        # actions adjust their ``cs`` offsets in place, so a replay
        # always reflects the current on-disk line numbers.
        self.data: dict[str, Any] = merge_data
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


def _apply_exec_bit(path: str) -> None:
    """Re-apply the executable bit to a restored file.

    Mirrors git's checkout behavior for mode-``100755`` entries:
    execute permission is granted wherever read permission is present.

    Args:
        path: File whose mode should gain exec bits.
    """
    mode = os.stat(path).st_mode
    os.chmod(path, mode | ((mode & 0o444) >> 2))


def _restore_base_bytes(
    base_path: str,
    write_to: str,
    link_target: str | None = None,
    *,
    make_executable: bool = False,
) -> None:
    """Restore *write_to* to the exact bytes of *base_path*.

    Used to reject changes to binary (or undecodable) files, whose
    merge entries carry a single whole-file pseudo-hunk: line-based
    splicing is meaningless for them, so the base copy is restored
    wholesale.  A missing/unreadable base restores an empty file
    (mirroring ``_write_base_copy``'s empty-base convention).

    When *link_target* is given, the base of the entry is a SYMLINK
    blob (the agent retargeted, deleted, or replaced a tracked
    symlink): the link itself is recreated instead of writing the
    blob's target string as regular-file content.

    Args:
        base_path: Path to the pre-task base copy.
        write_to: Real workspace path to restore.
        link_target: Target string of the base symlink, or ``None``
            for regular content.
        make_executable: True when the base mode is ``100755`` — the
            restored file gets its exec bit back so rejecting a
            deleted script leaves a clean tree and a runnable file.
    """
    dest = Path(write_to)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if link_target is not None:
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        os.symlink(link_target, write_to)
        return
    try:
        data = Path(base_path).read_bytes()
    except OSError:
        data = b""
    # Never write THROUGH a symlink: git tracks the link itself, not
    # its target, and the target may be a precious file (possibly
    # outside the repo) whose truncation would be silent data loss.
    if dest.is_symlink():
        dest.unlink()
    dest.write_bytes(data)
    if make_executable:
        _apply_exec_bit(write_to)


def _reject_hunk_in_file(
    current_path: str,
    base_path: str,
    hunk: dict[str, int],
    target_path: str | None = None,
    *,
    binary: bool = False,
    link_target: str | None = None,
    make_executable: bool = False,
) -> None:
    """Revert a single hunk in the current file to the base version.

    Reads both files, replaces the hunk's lines in the current file
    with the corresponding lines from the base file, and writes the
    result back.  Files are read and written WITHOUT newline
    translation (and split on ``\\n`` only) so rejecting one hunk of a
    CRLF file does not silently rewrite every other line with LF
    endings, and line numbering matches git's ``\\n``-based hunks.

    When *binary* is true — or either file turns out not to be
    decodable text — the base bytes are restored wholesale instead:
    binary merge entries carry a single whole-file pseudo-hunk, so
    line splicing does not apply (and used to raise
    ``UnicodeDecodeError``, crashing the merge action).

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
        binary: True when the merge entry is flagged binary; restores
            the base bytes wholesale.
        link_target: Target string when the base is a symlink blob;
            the link itself is restored (see ``_restore_base_bytes``).
        make_executable: True when the base mode is ``100755``; the
            rewritten file gets its exec bit re-applied.
    """
    write_to = target_path or current_path
    if binary or link_target is not None:
        _restore_base_bytes(
            base_path, write_to, link_target,
            make_executable=make_executable,
        )
        return
    # Read from *write_to* (the real workspace target) when it exists
    # so that successive partial rejections accumulate against the
    # restored content rather than the (now-stale) placeholder.
    cur_lines: list[str] = []
    try:
        try:
            cur_lines = _read_lines_preserved(write_to)
        except OSError:
            try:
                cur_lines = _read_lines_preserved(current_path)
            except OSError:
                cur_lines = []
        base_lines = _read_lines_preserved(base_path)
    except UnicodeDecodeError:
        # Undecodable content that slipped past the binary sniff
        # (e.g. UTF-16 / latin-1 without NUL bytes in the first 8 KiB).
        # Restoring the base bytes wholesale beats crashing the merge
        # action with an exception.
        _restore_base_bytes(
            base_path, write_to, make_executable=make_executable,
        )
        return
    except OSError:
        base_lines = []

    new_lines = (
        cur_lines[: hunk["cs"]]
        + base_lines[hunk["bs"] : hunk["bs"] + hunk["bc"]]
        + cur_lines[hunk["cs"] + hunk["cc"] :]
    )
    dest = Path(write_to)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Replace a symlink instead of writing THROUGH it — writing through
    # would clobber the pointed-to file (which may live outside the
    # repo) while leaving the rejected link itself untouched.
    if dest.is_symlink():
        dest.unlink()
    with open(write_to, "w", newline="") as f:
        f.write("".join(new_lines))
    if make_executable:
        _apply_exec_bit(write_to)


def _reject_all_hunks_in_file(
    file_data: dict[str, Any], hunk_indices: list[int] | None = None,
) -> None:
    """Surgically revert the given hunks of a file to the base version.

    Reverts only the hunks named by *hunk_indices* (all hunks when
    ``None``) via :func:`_reject_hunk_in_file`, applying the same
    ``cs`` offset fix-ups to later pending hunks as the per-hunk
    ``reject`` action.  Callers (``reject-file`` / ``reject-all``)
    pass the file's UNRESOLVED hunk indices so content the user
    already ACCEPTED stays on disk — a whole-file base copy here
    would silently wipe accepted hunks while ``resolutions()`` still
    reported them ``"accepted"``.

    Args:
        file_data: File entry from merge data with ``base``,
            ``current``, ``hunks`` and (optionally) ``target`` path
            strings.
        hunk_indices: Indices into ``file_data["hunks"]`` to revert;
            ``None`` means every hunk.
    """
    hunks = file_data.get("hunks", [])
    if hunk_indices is None:
        hunk_indices = list(range(len(hunks)))
    if file_data.get("binary"):
        # Binary entries carry a single whole-file pseudo-hunk; restore
        # the base bytes wholesale (line splicing does not apply).
        # ``link_target`` marks a symlink-base entry whose reject must
        # recreate the link itself.
        if hunk_indices:
            _restore_base_bytes(
                file_data["base"],
                file_data.get("target") or file_data["current"],
                file_data.get("link_target"),
                make_executable=bool(file_data.get("exec")),
            )
        return
    pending = set(hunk_indices)
    for hi in sorted(hunk_indices):
        hunk = hunks[hi]
        _reject_hunk_in_file(
            file_data["current"], file_data["base"], hunk,
            file_data.get("target"),
            make_executable=bool(file_data.get("exec")),
        )
        pending.discard(hi)
        delta = hunk["bc"] - hunk["cc"]
        for later_hi in range(hi + 1, len(hunks)):
            if later_hi in pending:
                hunks[later_hi]["cs"] += delta


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
    if not isinstance(data, dict):
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
        # Per-connection endpoint registry, keyed by the ``conn_id``
        # that the handlers stamp (as ``connId``) on every client
        # command.  Request/reply events (``models``, ``history``,
        # ``files``, ``ghost``, ``configData``, ...) carry the
        # requesting connection's id back on the event so
        # :meth:`broadcast` can deliver them ONLY to the window that
        # asked — one VS Code window's webview activity must never
        # change the UI of another window's webview.
        self._conn_endpoints: dict[str, Any] = {}
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
          are global system events (``tasks_updated``, ``remote_url``,
          ``update_available``, etc.) and are broadcast verbatim to
          every connected client.
        * Events stamped with a non-empty ``connId`` are request/reply
          events (``models``, ``history``, ``frequentTasks``,
          ``inputHistory``, ``files``, ``ghost``, ``configData``,
          unknown-command ``error``): the stamp is stripped and the
          event is sent ONLY to the connection (= VS Code window /
          browser tab) that issued the request, so one window's
          webview activity can never change another window's UI.

        Args:
            event: The event dictionary to emit.
        """
        conn_id = event.pop("connId", "")
        if event.get("type") == "configData":
            cfg = event.get("config")
            if isinstance(cfg, dict) and not cfg.get("work_dir"):
                # M2: prefer per-instance work_dir over the global env var.
                cfg["work_dir"] = (
                    self.work_dir
                    or os.environ.get("KISS_WORKDIR", "")
                    or os.getcwd()
                )

        if conn_id:
            # Targeted request/reply event — deliver only to the
            # requesting connection.  Never recorded or persisted
            # (these are transient UI replies, not task events).  A
            # vanished endpoint (client disconnected before the reply
            # was ready) silently drops the event.
            self._send_to_conn(conn_id, json.dumps(event))
            return

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
        # is currently watching this task).  The event is serialised
        # ONCE and the per-tab ``tabId`` stamp is spliced into the
        # JSON string, instead of re-encoding the whole event for
        # every subscribed tab — this path runs once per streamed
        # token, so avoiding the redundant ``json.dumps`` calls keeps
        # multi-viewer streaming cheap.
        targets = self._fanout_targets(event.get("taskId"))
        if not targets:
            return
        # ``event`` always carries at least ``type`` and ``taskId``
        # here, so the serialised form ends with ``...}`` and the
        # splice below produces exactly ``json.dumps({**event,
        # "tabId": tab_id})`` (sans key ordering).
        base = json.dumps(event)[:-1]
        for tab_id in targets:
            self._send_to_ws_clients(
                f'{base}, "tabId": {json.dumps(tab_id)}}}'
            )

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
            endpoints = list(self._ws_clients) + list(self._uds_writers)
        for endpoint in endpoints:
            self._schedule_send(endpoint, data)

    def _send_to_conn(self, conn_id: str, data: str) -> None:
        """Send a pre-serialised JSON payload to ONE connection.

        Used by :meth:`broadcast` for request/reply events stamped
        with the requesting connection's ``connId``: the reply must
        reach only the VS Code window (or browser tab) that issued
        the request, never its sibling windows.

        Args:
            conn_id: The connection id registered via :meth:`bind_conn`.
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        with self._ws_lock:
            endpoint = self._conn_endpoints.get(conn_id)
        if endpoint is None:
            return
        self._schedule_send(endpoint, data)

    def _schedule_send(self, endpoint: Any, data: str) -> None:
        """Schedule one payload send to one endpoint on the event loop.

        Shared by :meth:`_send_to_ws_clients` (fan-out) and
        :meth:`_send_to_conn` (targeted reply).  ``endpoint`` is a
        :class:`ServerConnection` (WSS) or an
        :class:`asyncio.StreamWriter` (UDS).  The resulting future is
        tracked in ``_pending_sends`` (M8) so a stuck/slow peer's
        pending sends can be cancelled when the client disconnects.

        Args:
            endpoint: The client connection to write to.
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            if isinstance(endpoint, asyncio.StreamWriter):
                fut = asyncio.run_coroutine_threadsafe(
                    self._uds_send(endpoint, data), loop,
                )
            else:
                fut = asyncio.run_coroutine_threadsafe(
                    endpoint.send(data), loop,
                )
        except Exception:
            logger.debug("Failed to send to client", exc_info=True)
            return
        with self._ws_lock:
            pending = self._pending_sends.get(endpoint)
            if pending is not None:
                pending.add(fut)
        fut.add_done_callback(
            partial(self._discard_pending_send, endpoint),
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

    def bind_conn(self, conn_id: str, endpoint: Any) -> None:
        """Associate a connection id with its transport endpoint.

        Called by the WSS / UDS handlers when a client connects, so
        :meth:`broadcast` can route request/reply events (stamped
        with ``connId``) back to ONLY the requesting connection.

        Args:
            conn_id: The unique id stamped (as ``connId``) on every
                command from this connection.
            endpoint: The :class:`ServerConnection` (WSS) or
                :class:`asyncio.StreamWriter` (UDS) for the connection.
        """
        with self._ws_lock:
            self._conn_endpoints[conn_id] = endpoint

    def unbind_conn(self, conn_id: str) -> None:
        """Drop the connection-id → endpoint binding for a closed peer.

        Args:
            conn_id: The connection id registered via :meth:`bind_conn`.
        """
        with self._ws_lock:
            self._conn_endpoints.pop(conn_id, None)

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


def _media_url(name: str) -> str:
    """Return a cache-busted URL for a packaged web media asset."""
    ver = _MEDIA_VERSION_CACHE.get(name)
    if ver is None:
        data = (MEDIA_DIR / name).read_bytes()
        ver = hashlib.sha256(data).hexdigest()[:16]
        _MEDIA_VERSION_CACHE[name] = ver
    return f"/media/{name}?v={ver}"


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
        "STYLE_HREF": _media_url("main.css"),
        "HLJS_CSS_HREF": _media_url("highlight-github-dark.min.css"),
        "HEAD_STYLE": head_style,
        "BODY_CLASS_ATTR": ' class="remote-chat"',
        "INPUT_PLACEHOLDER": "Ask anything... (@ for files)",
        "ENTERKEYHINT": ' enterkeyhint="send"',
        "MODEL_NAME": "loading...",
        "VERSION_SUFFIX": f" {version}" if version else "",
        "AUTH_MODAL": auth_modal,
        "NONCE_ATTR": "",
        "HLJS_SRC": _media_url("highlight.min.js"),
        "MARKED_SRC": _media_url("marked.min.js"),
        "PANEL_COPY_SRC": _media_url("panelCopy.js"),
        "MAIN_SRC": _media_url("main.js"),
        "DEMO_SRC": _media_url("demo.js"),
        "SHIM_SCRIPT": f"<script>{_WS_SHIM_JS}</script>\n  ",
        "TRICKS_JSON": tricks_json,
    }
    tpl = (MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
    # Single-pass substitution: injected values (e.g. the JS shim's
    # documentation comment that mentions ``{{AUTH_MODAL}}`` by name)
    # must NOT be re-scanned, otherwise stray placeholder-shaped tokens
    # inside substituted JS/HTML would either be wrongly replaced or
    # (when their key happens to be processed earlier in the dict)
    # survive into the served page as unsubstituted placeholders.
    return re.sub(
        r"\{\{([A-Z_]+)\}\}",
        lambda m: subs.get(m.group(1), m.group(0)),
        tpl,
    )


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
    """Parse ``~/.kiss/INJECTIONS.md`` and return the trick texts.

    Thin wrapper around :func:`kiss.agents.vscode.tricks.read_tricks`
    kept for backward compatibility with existing call sites in
    :mod:`web_server`.  The user-local copy at
    ``~/.kiss/INJECTIONS.md`` is the runtime source of truth — the
    sidebar "Inject" panel and the ghost-text fast-complete pipeline
    share that same file via the helper module.

    Returns:
        Ordered list of trick text strings (one per ``## Trick``
        section), or an empty list when INJECTIONS.md is missing or
        unparseable so the button still renders.
    """
    from kiss.agents.vscode.tricks import read_tricks
    return read_tricks()


_WS_SHIM_JS = r"""
(function() {
  var _state = null;
  try { _state = JSON.parse(sessionStorage.getItem('sorcar-state')); } catch(e) {}
  var _ws = null;
  var _pending = [];
  var _authenticated = false;
  var _needsPassword = false;
  // Tracks whether this client has previously completed a full
  // auth handshake.  Once true, the next successful ``auth_ok``
  // after an ``onclose`` (i.e. a server restart or network blip)
  // means the page state is stale relative to the freshly booted
  // backend and we must reload the page so the normal load
  // pipeline replays history, restored tabs, in-flight merges,
  // etc.  Without this the page only re-binds the socket and the
  // user is left staring at the "KISS Sorcar Server is starting
  // ..." overlay (or stale UI) until they manually refresh.
  var _hadAuthThenClosed = false;
  // Reconnect backoff attempt count — reset to 0 after a successful
  // ``auth_ok`` so a fresh disconnect tries again almost immediately.
  var _reconnectAttempt = 0;
  // Pending reconnect timer id, used so visibilitychange / pageshow /
  // online wake-ups can short-circuit the scheduled delay.
  var _reconnectTimer = null;

  // ``sessionStorage`` persists across the ``window.location.reload()``
  // performed inside the ``_hadAuthThenClosed`` branch of ``auth_ok``,
  // which lets the freshly-loaded page detect "this load is actually
  // a reconnect from a previously-authenticated session" and label the
  // loading overlay accordingly.
  var _RECONNECT_FLAG = 'sorcar-reconnect-pending';

  function _readReconnectingFlag() {
    try { return sessionStorage.getItem(_RECONNECT_FLAG) === '1'; }
    catch (e) { return false; }
  }
  function _setReconnectingFlag(on) {
    try {
      if (on) sessionStorage.setItem(_RECONNECT_FLAG, '1');
      else sessionStorage.removeItem(_RECONNECT_FLAG);
    } catch (e) {}
  }

  /**
   * Replace the overlay text so the user sees an accurate status.
   *
   * On a brand-new tab the message is "KISS Sorcar Server is starting
   * ..." because the server may legitimately not be up yet.  Once we
   * have proven the server is reachable (a previous ``auth_ok`` came
   * through, then the socket later closed) every subsequent display of
   * the overlay represents a RECONNECT, not a cold start — say so.
   */
  function _updateLoadingMsg(reconnecting) {
    var msg = document.getElementById('kiss-server-loading-msg');
    if (!msg) return;
    msg.textContent = reconnecting
      ? 'Reconnecting to KISS Sorcar Server ...'
      : 'KISS Sorcar Server is starting ...';
  }

  // Apply the reconnect label immediately on script start when the
  // sessionStorage flag survives from the prior page instance.  Without
  // this the user would briefly see "Server is starting ..." after
  // backgrounding Safari and returning, even though we know the server
  // is up and we are merely re-establishing the WebSocket.
  if (_readReconnectingFlag()) {
    _updateLoadingMsg(true);
  }

  function _scheduleReconnect() {
    if (_reconnectTimer !== null) return;
    // Aggressive backoff: 250ms, 500ms, 1s, 2s, 4s, capped at 5s.
    // The old 3000ms fixed delay made reconnects feel sluggish on
    // mobile Safari, which already pauses JS in backgrounded tabs.
    var delay = Math.min(5000, 250 * Math.pow(2, _reconnectAttempt));
    _reconnectAttempt++;
    _reconnectTimer = setTimeout(function () {
      _reconnectTimer = null;
      connect();
    }, delay);
  }

  function _reconnectNowIfNeeded() {
    // Called from visibilitychange / pageshow / online handlers so a
    // user who left Safari for another app does not wait the full
    // backoff after returning.  We treat CONNECTING as "in flight,
    // don't disturb"; CLOSED / CLOSING / null all warrant an
    // immediate attempt.
    if (_ws && (_ws.readyState === WebSocket.OPEN ||
                _ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    if (_reconnectTimer !== null) {
      try { clearTimeout(_reconnectTimer); } catch (e) {}
      _reconnectTimer = null;
    }
    _reconnectAttempt = 0;
    connect();
  }

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
        if (msg && msg.type === 'setWorkDir') {
          // Pin this webapp instance's work_dir.  sessionStorage is
          // scoped per browser tab, so each tab (= one webapp
          // instance) keeps its own value across reloads, and the
          // auth_ok handler below replays it on every reconnect —
          // mirroring how each VS Code window re-announces its
          // workspace folder on every UDS (re)connect.
          try {
            sessionStorage.setItem('sorcar-work-dir', msg.workDir || '');
          } catch(e) {}
        }
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
    // Neutralise the previous socket BEFORE we install a fresh one.
    // On iOS Safari the OS may kill the underlying WebSocket while
    // the tab is backgrounded; when JS resumes, the wake-up listeners
    // (visibilitychange / focus / pageshow) frequently fire BEFORE
    // the queued ``onclose`` of the dead socket.  If we don't clear
    // the old handlers, that late ``onclose`` will run against the
    // module-level ``_ws`` we just replaced -- it would call
    // ``_scheduleReconnect()`` (overwriting the in-flight new socket
    // after the backoff fires) and any late ``onopen``/``onmessage``
    // on the old socket would ``_ws.send(...)`` on the new one.
    // Nulling the handlers and closing the old socket here makes the
    // replacement atomic from the rest of the shim's perspective.
    if (_ws) {
      try {
        _ws.onopen = null;
        _ws.onmessage = null;
        _ws.onclose = null;
        _ws.onerror = null;
      } catch (e) {}
      try { _ws.close(); } catch (e) {}
    }
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
        // Recover from a server restart / network blip: if we had
        // already authenticated at least once and the WS later
        // closed, the page JS state is stale relative to the
        // freshly booted backend.  Reload so the normal page-load
        // pipeline (history replay, restored tabs, in-flight
        // merge replay, ...) runs against the new server state.
        // The reload is gated by ``_hadAuthThenClosed`` so the
        // very first authentication on a fresh page load does NOT
        // reload (otherwise we would loop forever).
        if (_hadAuthThenClosed) {
          try { window.location.reload(); } catch (e) {}
          return;
        }
        _authenticated = true;
        _needsPassword = false;
        // We have a live, authenticated socket — any future
        // disconnect IS a reconnect, but the just-completed
        // handshake is not.  Drop the sessionStorage flag so a
        // subsequent fresh tab (different browsing session, same
        // device) doesn't mislabel its first overlay.  The
        // _hadAuthThenClosed branch above keeps the flag intact
        // during the reload it triggers.
        _setReconnectingFlag(false);
        _reconnectAttempt = 0;
        // Re-establish this instance's pinned work_dir BEFORE flushing
        // any queued commands: the server stamps each connection's
        // work_dir onto later commands, so the pin must arrive first.
        // Every reconnect creates a fresh server-side connection state
        // with an empty work_dir; without this replay a reload or a
        // dropped WebSocket would silently fall back to the
        // daemon-global work_dir (possibly another instance's folder).
        var _wd = '';
        try { _wd = sessionStorage.getItem('sorcar-work-dir') || ''; } catch(e) {}
        if (_wd) {
          _ws.send(JSON.stringify({type: 'setWorkDir', workDir: _wd}));
        }
        for (var i = 0; i < _pending.length; i++) _ws.send(_pending[i]);
        _pending = [];
        // Hide the "KISS Sorcar Server is starting ..." overlay now
        // that the WebSocket is authenticated.  The remote webapp has
        // no equivalent of the VS Code extension host's daemonStatus
        // posts (the daemon == this WSS server), so we synthesise the
        // same window ``message`` event ``media/main.js`` listens for.
        // Without this the overlay covers ``#app`` forever and the
        // user only ever sees "KISS Sorcar Server is starting ...".
        window.dispatchEvent(new MessageEvent('message', {
          data: {type: 'daemonStatus', connected: true}
        }));
        return;
      }
      if (msg.type === 'auth_required') {
        _needsPassword = true;
        // Stored password (if any) was rejected; drop it so a refresh
        // re-prompts instead of silently retrying the bad value.
        try { localStorage.removeItem('sorcar-remote-pwd'); } catch(e) {}
        // Reveal ``#app`` so the auth modal (which lives INSIDE #app
        // in the chat.html template — see the ``AUTH_MODAL`` template
        // placeholder substituted by ``_build_html``) is no longer
        // hidden by its display:none parent.  Without this
        // dispatch a password-protected webapp shows the loading
        // overlay forever and the user can never enter their
        // password.  Symmetric to the auth_ok dispatch above — both
        // states prove the server is reachable.
        window.dispatchEvent(new MessageEvent('message', {
          data: {type: 'daemonStatus', connected: true}
        }));
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
      // Latch "we had a real session and then lost it" so the next
      // successful ``auth_ok`` reloads the page.  We only set the
      // flag when the prior socket had completed its auth handshake
      // — a fresh page that has not yet authenticated must NOT
      // trigger a reload on its first ``auth_ok``.
      if (_authenticated) {
        _hadAuthThenClosed = true;
        // Persist the reconnect-state across the ``location.reload()``
        // that ``auth_ok`` will trigger so the freshly-loaded page
        // labels its overlay "Reconnecting ..." instead of the
        // misleading "KISS Sorcar Server is starting ...".  Mobile
        // Safari frequently kills the WebSocket whenever the user
        // switches apps, so this is the common case, not an edge
        // case.
        _setReconnectingFlag(true);
      }
      _authenticated = false;
      // Switch the overlay text BEFORE re-revealing it: once we have
      // had at least one successful handshake (current page or any
      // previous one, latched via sessionStorage) every overlay
      // appearance is a reconnect from the user's perspective.
      _updateLoadingMsg(_hadAuthThenClosed || _readReconnectingFlag());
      // Re-show the loading overlay while the socket is down so the
      // user knows actions will not reach the backend.  Symmetric to
      // the ``auth_ok`` dispatch above and to
      // ``SorcarSidebarView.ts``'s disconnect handler in the VS Code
      // path.
      window.dispatchEvent(new MessageEvent('message', {
        data: {type: 'daemonStatus', connected: false}
      }));
      _scheduleReconnect();
    };

    _ws.onerror = function() {};
  }

  // Wake-up listeners — mobile Safari pauses JS in backgrounded tabs,
  // so a scheduled ``setTimeout(connect, ...)`` may not fire until the
  // user returns.  These events fire AS SOON AS the user comes back,
  // triggering an immediate reconnect instead of waiting for the
  // backoff timer.  Without them the user would stare at the loading
  // overlay for the remainder of the (paused) backoff after every
  // app-switch round-trip.
  if (typeof document !== 'undefined' &&
      typeof document.addEventListener === 'function') {
    document.addEventListener('visibilitychange', function () {
      if (document.visibilityState === 'visible') {
        _reconnectNowIfNeeded();
      }
    });
  }
  if (typeof window !== 'undefined' &&
      typeof window.addEventListener === 'function') {
    // ``pageshow`` covers Safari's bfcache restore, which does not
    // fire ``visibilitychange``.
    window.addEventListener('pageshow', function () {
      _reconnectNowIfNeeded();
    });
    window.addEventListener('online', function () {
      _reconnectNowIfNeeded();
    });
    // ``focus`` is the universal fallback for older mobile browsers
    // that ignore visibilitychange/pageshow under certain conditions.
    window.addEventListener('focus', function () {
      _reconnectNowIfNeeded();
    });
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
            ("Cache-Control", "no-cache, no-store, must-revalidate"),
            ("Pragma", "no-cache"),
            ("Expires", "0"),
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
        # Read WITHOUT newline translation: hunk coordinates are
        # computed by splitting the on-disk bytes on "\n" only (see
        # ``diff_merge._read_lines_preserved``), so a universal-newline
        # read would hand the browser text whose CRLF endings are
        # silently rewritten — and whose line COUNT differs for
        # lone-"\r" content — misaligning hunk highlighting.
        try:
            with open(f["base"], newline="") as bfh:
                f["base_text"] = bfh.read()
        except (OSError, KeyError, UnicodeDecodeError):
            f["base_text"] = ""
        try:
            with open(f["current"], newline="") as cfh:
                f["current_text"] = cfh.read()
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
        # Copy the command (like the resumeSession branch below) so the
        # ``connId``/``workDir`` stamps from ``_dispatch_client_command``
        # survive the translation.
        out = dict(cmd)
        out["type"] = "userAnswer"
        out["answer"] = "done"
        out["tabId"] = cmd.get("tabId", "")
        return out
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
        # Task ids the local ``sorcar`` CLI has announced as running
        # via ``cliTaskStart`` envelopes (see
        # :meth:`_handle_cli_task_start`).  Consulted by
        # :meth:`VSCodeServer._replay_session` (read through the
        # ``_is_cli_task_running`` hook) so a chat webview tab that
        # later resumes a CLI-launched task from the history sidebar
        # can be subscribed to the live event stream and shown the
        # blinking-green-circle "running" indicator in its tab title.
        # Guarded by ``_cli_running_lock`` because the UDS handler
        # mutates it from the asyncio thread and ``_replay_session``
        # reads it from agent / handler threads.
        self._cli_running_tasks: set[str] = set()
        self._cli_running_lock = threading.Lock()
        # Expose the running-set lookup to ``VSCodeServer`` so its
        # ``_replay_session`` can subscribe a freshly opened webview
        # tab to a CLI-launched task that is still running.  The
        # closure captures both the set and its lock so the read is
        # atomic with respect to mutations from the UDS handler.
        self._vscode_server.set_cli_running_lookup(self._is_cli_task_running)
        # Companion snapshot hook used by
        # :meth:`VSCodeServer._get_running_task_ids` to UNION CLI-
        # launched running task ids into the per-row ``is_running``
        # flag the History panel uses to render the pulsing-green-dot
        # indicator.
        self._vscode_server.set_cli_running_task_ids_lookup(
            self._snapshot_cli_running_task_ids,
        )

    def _snapshot_cli_running_task_ids(self) -> set[str]:
        """Return a thread-safe copy of the CLI-running task id set.

        Returned set is a fresh copy so callers can iterate / mutate
        without taking ``_cli_running_lock`` (or racing the UDS
        handler that mutates the underlying set).
        """
        with self._cli_running_lock:
            return set(self._cli_running_tasks)

    def _is_cli_task_running(self, task_id: str) -> bool:
        """Return ``True`` when *task_id* is being run by the CLI.

        Used by :meth:`VSCodeServer._replay_session` to decide whether
        to subscribe a freshly opened webview tab to a CLI-launched
        task's live event stream and broadcast a ``status:running``
        event so the tab title shows the blinking-green-circle
        indicator.
        """
        with self._cli_running_lock:
            return task_id in self._cli_running_tasks

    def _handle_cli_task_start(self, task_id: str, conn_state: dict[str, Any]) -> None:
        """Record *task_id* as a CLI-launched running task.

        Also stamps the task id into the UDS connection's per-conn
        ``cli_tasks`` set so :meth:`_uds_handler` can clean it up if
        the CLI process disconnects without sending a matching
        ``cliTaskEnd`` (Ctrl+C, crash, abrupt termination).
        """
        with self._cli_running_lock:
            self._cli_running_tasks.add(task_id)
        cli_tasks = conn_state.setdefault("cli_tasks", set())
        if isinstance(cli_tasks, set):
            cli_tasks.add(task_id)

    def _handle_cli_task_end(self, task_id: str, conn_state: dict[str, Any]) -> None:
        """Mark *task_id* as no longer running and stop the indicator.

        Drops the task id from :attr:`_cli_running_tasks` and from
        the connection's per-conn ``cli_tasks`` set, then broadcasts
        a ``status:running=false`` event to every webview tab
        currently subscribed to the task id so the
        blinking-green-circle "running" indicator stops on the tab
        title.
        """
        with self._cli_running_lock:
            self._cli_running_tasks.discard(task_id)
        cli_tasks = conn_state.get("cli_tasks")
        if isinstance(cli_tasks, set):
            cli_tasks.discard(task_id)
        self._fanout_cli_status(task_id, running=False)

    def _fanout_cli_status(self, task_id: str, *, running: bool) -> None:
        """Send ``status:running`` to every tab subscribed to *task_id*.

        Mirrors the per-tab fan-out idiom from
        :meth:`WebPrinter.broadcast` /
        :meth:`_relay_cli_event`: looks up the viewer tabs currently
        subscribed to the task id, then for each one writes a
        ``status`` event with that tab's ``tabId`` spliced in.  Used
        when a CLI task ends so the blinking-green-circle indicator
        clears on every webview that was watching it.
        """
        targets = self._printer._fanout_targets(task_id)
        if not targets:
            return
        for tab_id in targets:
            payload = json.dumps({
                "type": "status",
                "running": running,
                "tabId": tab_id,
            })
            self._printer._send_to_ws_clients(payload)

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
        request_path = urlsplit(request.path).path
        path = unquote(request_path)
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
        if path == "/favicon.ico":
            favicon = MEDIA_DIR / "kiss-icon.png"
            if favicon.is_file():
                return _http_response(200, "image/png", favicon.read_bytes())
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
        # Drop entries outside the window.  Only write the pruned list
        # back for IPs that already have an entry (i.e. recorded at
        # least one real failure) — unconditionally assigning here
        # used to create a permanent empty-list entry for EVERY source
        # IP that ever connected (including ones that always
        # authenticated successfully), growing ``_auth_failures``
        # without bound over the daemon's lifetime.
        fails = [t for t in fails if now - t <= _AUTH_FAIL_WINDOW]
        if fails or ip in self._auth_failures:
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

        When the configured ``remote_password`` is empty, authentication
        is skipped entirely for local connections — the public
        cloudflared tunnel is already disabled in :meth:`_setup_server`
        when no password is set, so no unauthenticated client can
        reach the daemon from the public internet.  This makes the
        webapp immediately usable on ``localhost`` without any
        configuration.
        """
        password = load_config().get("remote_password", "")
        if not password:
            logger.info("No remote_password configured; skipping auth")
            await websocket.send(json.dumps({"type": "auth_ok"}))
            return True
        ip = self._client_ip(websocket)
        if self._is_auth_locked(ip):
            logger.warning("Auth rate-limit hit for %s; closing socket", ip)
            try:
                await websocket.close()
            except Exception:
                pass
            return False
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
            merge_state = self._merge_states.pop(tab_id, None)
            self._merge_action_locks.pop(tab_id, None)
        if self._loop is None or not self._loop.is_running():
            return
        task = asyncio.ensure_future(
            self._finish_merge_and_close_tab(tab_id, merge_state),
            loop=self._loop,
        )
        self._pending_close_tasks.add(task)
        task.add_done_callback(self._pending_close_tasks.discard)

    async def _finish_merge_and_close_tab(
        self, tab_id: str, merge_state: _WebMergeState | None,
    ) -> None:
        """End an in-flight merge review (if any) and close *tab_id*.

        Companion of :meth:`_fire_pending_tab_close`.  When the close
        grace elapsed while a merge review was still in flight, the
        popped :class:`_WebMergeState` is the ONLY thing that could
        ever drive the review to ``all-done`` — the backend
        ``_close_tab`` sees ``is_merging=True`` (a busy lifecycle
        flag), merely flips ``frontend_closed=True`` and waits for
        the merge to end, which would now never happen.  Dispatching
        ``all-done`` here treats the close as "accept the remaining
        hunks" (no disk writes — the workspace already holds the
        agent's content): ``_finish_merge`` clears ``is_merging``,
        cleans the per-tab merge artifacts, presents any pending
        worktree, and the subsequent ``closeTab`` disposes the
        backend tab instead of leaking it forever.

        Args:
            tab_id: The frontend tab identifier being closed.
            merge_state: The web-side merge state popped for the tab,
                or ``None`` when no review was in flight.
        """
        if merge_state is not None:
            await self._run_cmd({
                "type": "mergeAction",
                "action": "all-done",
                "tabId": tab_id,
                "workDir": merge_state.work_dir,
            })
        await self._run_cmd({"type": "closeTab", "tabId": tab_id})

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
        # Per-connection work_dir + unique connection id (see
        # _dispatch_client_command).
        conn_state: dict[str, str] = {
            "work_dir": "", "conn_id": uuid.uuid4().hex,
        }
        self._printer.bind_conn(conn_state["conn_id"], websocket)
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if not isinstance(cmd, dict):
                    continue
                try:
                    await self._dispatch_client_command(
                        cmd, websocket, tabs_seen, conn_state,
                    )
                except websockets.exceptions.ConnectionClosed:
                    raise
                except Exception:
                    # Contain per-command failures (malformed fields,
                    # unexpected I/O errors in handlers): one bad
                    # message must not tear down the authenticated
                    # connection — the ``finally`` below would arm
                    # deferred closeTab timers for EVERY tab this
                    # client touched, force-finishing in-flight merge
                    # reviews as "accept remaining".
                    logger.warning(
                        "Error handling client command %r; "
                        "connection kept",
                        cmd.get("type", ""), exc_info=True,
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
            self._vscode_server.drop_connection_state(conn_state["conn_id"])
            self._printer.unbind_conn(conn_state["conn_id"])
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
        # several windows share this one daemon process.  The unique
        # ``conn_id`` is stamped (as ``connId``) on every command so
        # ``VSCodeServer`` can key its per-connection autocomplete
        # state (active-file snapshot, request staleness) by window.
        conn_state: dict[str, str] = {
            "work_dir": "", "conn_id": uuid.uuid4().hex,
        }
        self._printer.bind_conn(conn_state["conn_id"], writer)
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
                try:
                    await self._dispatch_client_command(
                        cmd, writer, tabs_seen, conn_state,
                    )
                except (ConnectionError, asyncio.IncompleteReadError):
                    raise
                except Exception:
                    # Same per-command containment as ``_ws_handler``:
                    # one bad message (malformed field, handler I/O
                    # error) must not drop the VS Code extension's
                    # UDS connection and force-close its tabs.
                    logger.warning(
                        "Error handling UDS command %r; connection kept",
                        cmd.get("type", ""), exc_info=True,
                    )
        except Exception:
            logger.debug("UDS handler error", exc_info=True)
        finally:
            for tab in tabs_seen:
                self._schedule_tab_close(tab)
            # Clean up any CLI task ids announced on this connection
            # but never closed (CLI crash, Ctrl+C, abrupt SIGKILL):
            # without this the daemon would keep them in
            # ``_cli_running_tasks`` forever and a webview later
            # resuming the task would mis-display the
            # blinking-green-circle "running" indicator for a task
            # that is no longer actually running anywhere.
            stale_cli_tasks = cast("Any", conn_state.get("cli_tasks"))
            if isinstance(stale_cli_tasks, set):
                for task_id in list(stale_cli_tasks):
                    if isinstance(task_id, str) and task_id:
                        self._handle_cli_task_end(
                            task_id, cast("dict[str, Any]", conn_state),
                        )
            self._vscode_server.drop_connection_state(conn_state["conn_id"])
            self._printer.unbind_conn(conn_state["conn_id"])
            self._printer.remove_uds_writer(writer)
            try:
                writer.close()
            except Exception:
                logger.debug("UDS writer close failed", exc_info=True)


    def _relay_cli_event(self, ev: dict[str, Any]) -> None:
        """Fan a CLI-originated event out to subscribed webview tabs.

        The ``sorcar`` CLI's :class:`RecordingConsolePrinter` ships
        every display event over the daemon's UDS endpoint wrapped in
        a ``cliEvent`` envelope (see
        :mod:`kiss.agents.sorcar.cli_daemon_bridge`).  The CLI process
        has ALREADY recorded the event into its per-task recording
        and persisted it to the chat DB via
        :meth:`JsonPrinter.broadcast`, so this method must NOT call
        ``_record_event`` or ``_persist_event`` again — doing so would
        produce duplicate rows in the ``events`` table.  It only
        mirrors the tail of :meth:`WebPrinter.broadcast`: look up the
        viewer tabs currently subscribed to the event's task id and
        splice each ``tabId`` into the pre-serialised JSON, then push
        to every WSS / UDS client in lockstep.  Any chat webview
        that opened the task's chat id therefore receives the event
        live, without waiting for a page reload to replay it from
        the database.

        Args:
            ev: The event dictionary the CLI emitted; expected to
                carry at least ``type`` and ``taskId``.
        """
        targets = self._printer._fanout_targets(ev.get("taskId"))
        if not targets:
            return
        base = json.dumps(ev)[:-1]
        for tab_id in targets:
            self._printer._send_to_ws_clients(
                f'{base}, "tabId": {json.dumps(tab_id)}}}'
            )

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
                connection's own ``work_dir`` and unique ``conn_id``.
                Each VS Code window owns exactly one connection and
                announces its workspace folder via ``setWorkDir``;
                every later command from the same connection that does
                not carry an explicit ``workDir`` is stamped with it
                here.  This is what guarantees the per-window work_dir
                invariant: two windows sharing this daemon can never
                observe each other's folder through the daemon-global
                fallback, because their commands always arrive
                pre-stamped with their own connection's work_dir.  The
                ``conn_id`` is stamped (as ``connId``) on EVERY command
                — overwriting any client-supplied value so it cannot be
                spoofed — and keys ``VSCodeServer``'s per-connection
                autocomplete state (active-file snapshot and request
                staleness), giving each window the same isolation for
                ghost-text completions as for its work_dir.
        """
        tab_id = cmd.get("tabId", "")
        if isinstance(tab_id, str) and tab_id:
            tabs_seen.add(tab_id)
        cmd["connId"] = conn_state["conn_id"]
        cmd_type = cmd.get("type", "")
        if cmd_type == "cliEvent":
            # CLI -> daemon live-stream bridge.  The sorcar CLI
            # forwards every display event here so any chat webview
            # subscribed to the task's chat id sees the event
            # immediately instead of having to reload to replay it
            # from the events DB.  See ``_relay_cli_event``.
            ev = cmd.get("event")
            if isinstance(ev, dict):
                self._relay_cli_event(ev)
            return
        if cmd_type == "cliTaskStart":
            # CLI announces a fresh task is now running so a webview
            # tab later resuming the task from the history sidebar
            # can be subscribed to the live stream and shown the
            # blinking-green-circle "running" indicator in its tab
            # title.  See ``_handle_cli_task_start``.
            raw_id = cmd.get("taskId")
            task_id_str = (
                raw_id if isinstance(raw_id, str) and raw_id else ""
            )
            if not task_id_str:
                logger.debug("cliTaskStart with bad taskId %r", raw_id)
                return
            self._handle_cli_task_start(task_id_str, conn_state)
            return
        if cmd_type == "cliTaskEnd":
            # CLI announces a previously-running task has finished
            # so the daemon stops the blinking-green-circle indicator
            # on every subscribed webview tab.  See
            # ``_handle_cli_task_end``.
            raw_id = cmd.get("taskId")
            task_id_str = (
                raw_id if isinstance(raw_id, str) and raw_id else ""
            )
            if not task_id_str:
                logger.debug("cliTaskEnd with bad taskId %r", raw_id)
                return
            self._handle_cli_task_end(task_id_str, conn_state)
            return
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
            # The deferred-close contract (see ``_ws_handler``'s
            # ``finally``) is "schedule a closeTab for every tab id
            # this connection touched".  ``_handle_ready`` re-claims
            # (cancels the pending close of, and resumes) every
            # ``restoredTabs`` entry, so those tab ids are touched by
            # this connection too — record them in ``tabs_seen`` or a
            # later disconnect would never re-arm their deferred
            # close, leaking the restored backend state forever.
            restored = cmd.get("restoredTabs")
            if isinstance(restored, list):
                for rt in restored[:_MAX_RESTORED_TABS]:
                    if isinstance(rt, dict):
                        rt_id = rt.get("tabId", "")
                        if isinstance(rt_id, str) and rt_id:
                            tabs_seen.add(rt_id)
            await self._handle_ready(cmd, endpoint)
            return
        if cmd_type == "submit":
            await self._handle_submit(cmd)
            return
        if cmd_type == "getWelcomeSuggestions":
            await self._send_welcome_info()
            return
        if cmd_type == "runUpdate":
            await self._handle_run_update(conn_state["conn_id"])
            return
        if cmd_type == "serverReset":
            await self._handle_server_reset(conn_state["conn_id"])
            return
        if cmd_type == "mergeAction":
            if cmd.get("action", "") != "all-done":
                await self._handle_web_merge_action(cmd)
                return
            # An ``all-done`` arriving FROM a client is the VS Code
            # extension's TS MergeManager finishing its editor-managed
            # review (its per-hunk actions never reach the backend —
            # see ``SorcarSidebarView.sendMergeAllDone``).  Drop the
            # server-side shadow ``_WebMergeState`` registered when the
            # ``merge_data`` event was broadcast: leaving it would
            # replay a ZOMBIE review on the next webview reload
            # (``ready`` → ``_replay_merge_review``), fire a spurious
            # second all-done from the deferred-close path, and leak
            # one state (with full file payloads) per finished review
            # in the meantime.  The command still falls through to the
            # backend ``_cmd_merge_action`` → ``_finish_merge`` below.
            if isinstance(tab_id, str) and tab_id:
                with self._merge_states_lock:
                    self._merge_states.pop(tab_id, None)
                    self._merge_action_locks.pop(tab_id, None)
        if (
            cmd_type == "closeTab"
            and isinstance(tab_id, str)
            and tab_id
            and not isinstance(endpoint, asyncio.StreamWriter)
        ):
            # A WEB client closing its chat tab destroys the only UI
            # that could ever finish an in-flight (server-tracked)
            # merge review for that tab: the backend ``_close_tab``
            # would see ``is_merging=True``, flip ``frontend_closed``
            # and wait forever for an ``all-done`` that no client can
            # send any more.  End the review first (close = accept the
            # remaining hunks; no disk writes) so the tab is disposed
            # instead of leaking in ``is_merging`` limbo.  UDS (VS
            # Code) clients are exempt: their TypeScript MergeManager
            # owns the review in real editor tabs that survive the
            # chat tab's closure and will still send ``all-done``.
            with self._merge_states_lock:
                merge_state = self._merge_states.pop(tab_id, None)
                self._merge_action_locks.pop(tab_id, None)
            await self._finish_merge_and_close_tab(tab_id, merge_state)
            return
        cmd = _translate_webview_command(cmd)
        await self._run_cmd(cmd)

    async def _handle_server_reset(self, conn_id: str = "") -> None:
        """Restart the ``kiss-web`` daemon at the user's request.

        Server-side handler for the settings-panel "Server reset"
        button.  Broadcasts an acknowledgement ``notification`` to the
        requesting window (stamped with its ``connId`` so siblings do
        not pop a banner), then schedules a ``SIGTERM`` to this very
        process after a short delay so the notification flushes to the client
        before its socket drops.  The ``SIGTERM`` is caught by
        :meth:`_handle_shutdown_signal`, which raises
        :class:`KeyboardInterrupt` to unwind the ``asyncio.run`` loop in
        :meth:`start` through its existing graceful-shutdown path
        (stopping in-flight agent tasks and the tunnel).  The process
        then exits and the supervising macOS LaunchAgent (``KeepAlive``)
        / Linux systemd unit (``Restart=always``) respawns a fresh
        ``kiss-web`` that re-adopts the same port and ``cloudflared``
        tunnel — so the public URL is preserved across the reset.

        Args:
            conn_id: Requesting connection id (``""`` to broadcast).
        """
        loop = self._loop
        assert loop is not None
        notification: dict[str, Any] = {
            "type": "notification",
            "id": "server-reset-restarting",
            "severity": "info",
            "message": "Restarting the KISS Sorcar web server…",
        }
        if conn_id:
            notification["connId"] = conn_id
        self._printer.broadcast(notification)
        # Drop a pending-reset flag so the freshly-respawned daemon
        # knows to broadcast a paired "Server restart complete"
        # notification once it is listening again — see
        # :meth:`_setup_server`.  Written atomically (tmp + replace)
        # so a crash mid-write never leaves a half-baked file the
        # next daemon would read.  Best-effort: a filesystem error
        # here only suppresses the post-restart toast, never blocks
        # the actual restart.
        self._write_server_reset_flag(conn_id)
        loop.call_later(_SERVER_RESET_DELAY, self._trigger_server_reset)

    def _server_reset_flag_path(self) -> Path:
        """Path of the pending-reset flag file.

        Lives next to ``remote-url.json`` so tests that supply a
        custom ``url_file=`` automatically get an isolated flag
        location and never touch the user's real ``~/.kiss``.
        """
        return self._url_file.parent / _SERVER_RESET_FLAG_NAME

    def _write_server_reset_flag(self, conn_id: str) -> None:
        """Persist a pending-reset marker before the daemon SIGTERMs.

        Args:
            conn_id: Requesting connection id (kept for diagnostics
                only — the connection itself cannot survive the
                SIGTERM, so the post-restart notification is
                broadcast to all reconnecting clients).
        """
        flag_path = self._server_reset_flag_path()
        try:
            flag_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = flag_path.with_suffix(flag_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {"requested_at": time.time(), "conn_id": conn_id},
                ),
                encoding="utf-8",
            )
            os.replace(tmp, flag_path)
        except OSError:
            logger.debug(
                "Could not write server-reset pending flag at %s",
                flag_path, exc_info=True,
            )

    def _maybe_schedule_server_reset_complete(self) -> None:
        """Schedule the post-restart broadcast iff a pending flag exists.

        Called once from :meth:`_setup_server` after the WSS / UDS
        listeners are bound and the watchdog tasks are armed.  When
        the flag file written by :meth:`_write_server_reset_flag`
        in the previous daemon instance is found, it is removed
        eagerly (so the toast fires at most once per user-initiated
        reset, even if the daemon restarts again before the timer
        runs) and a delayed callback is queued to broadcast the
        "Server restart complete" notification.
        """
        flag_path = self._server_reset_flag_path()
        if not flag_path.exists():
            return
        try:
            flag_path.unlink()
        except OSError:
            logger.debug(
                "Could not remove server-reset pending flag at %s",
                flag_path, exc_info=True,
            )
        loop = self._loop
        assert loop is not None
        loop.call_later(
            _SERVER_RESET_COMPLETE_DELAY,
            self._broadcast_server_reset_complete,
        )

    def _broadcast_server_reset_complete(self) -> None:
        """Broadcast the "Server restart complete" notification.

        Pair to the "Restarting the KISS Sorcar web server…" toast
        sent by :meth:`_handle_server_reset` in the *previous*
        daemon instance.  Scheduled from :meth:`_setup_server` when
        a pending-reset flag file is found, and delivered to every
        currently-connected client — the requesting connection
        died with the previous daemon so ``connId`` cannot be
        preserved across the restart, but every webview that was
        disconnected by the SIGTERM benefits from the same
        confirmation.  The stable ``id`` lets the existing webview
        dedup (``data-notification-id`` in ``showNotification``)
        replace any stale "restarting" toast in-place instead of
        stacking a duplicate.
        """
        self._printer.broadcast(
            {
                "type": "notification",
                "id": "server-reset-complete",
                "severity": "info",
                "message": "KISS Sorcar web server restart complete.",
            },
        )

    def _trigger_server_reset(self) -> None:
        """Send ``SIGTERM`` to this process to trigger a clean restart.

        Runs as a delayed event-loop callback on the main thread (see
        :meth:`_handle_server_reset`).  Delivering ``SIGTERM`` to the
        daemon's own pid routes through :meth:`_handle_shutdown_signal`
        exactly like an external ``pkill``/supervisor stop, so the
        established graceful-shutdown path runs and the supervisor
        respawns a fresh daemon.
        """
        logger.warning(
            "Server reset requested: pid=%d sending SIGTERM to self",
            os.getpid(),
        )
        os.kill(os.getpid(), signal.SIGTERM)

    async def _handle_run_update(self, conn_id: str = "") -> None:
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

        The acknowledgement ``notice`` / ``error`` events are stamped
        with the requesting connection's ``connId`` (when non-empty)
        so they reach ONLY the window whose user clicked "Update" —
        the extension's twin shows its messages only in the clicking
        window, and clicking Update in one browser window must not
        pop a banner in every sibling window.

        Args:
            conn_id: Requesting connection id (``""`` to broadcast).
        """
        loop = self._loop
        assert loop is not None
        script = await loop.run_in_executor(
            None, _find_install_script, self._install_root,
        )
        if script is None:
            event: dict[str, Any] = {
                "type": "error",
                "text": (
                    "Cannot update KISS Sorcar: install.sh not found "
                    f"in {self._install_root}."
                ),
            }
            if conn_id:
                event["connId"] = conn_id
            self._printer.broadcast(event)
            return
        notice: dict[str, Any] = {
            "type": "notice",
            "text": (
                "An update of KISS Sorcar is getting installed… "
                f"(output: {self._update_log_path})"
            ),
        }
        if conn_id:
            notice["connId"] = conn_id
        self._printer.broadcast(notice)
        await loop.run_in_executor(
            None, self._spawn_update_script, script, conn_id,
        )

    def _spawn_update_script(self, script: Path, conn_id: str = "") -> None:
        """Start ``install.sh`` detached, logging to the update log.

        Runs in the executor so file I/O and process spawn never block
        the event loop.  ``start_new_session=True`` keeps the updater
        alive when ``install.sh`` restarts this very daemon.
        ``stdin=DEVNULL`` detaches the script from the daemon's stdin so
        its interactive prompts (e.g. the git-upgrade question) fall
        back to their non-interactive defaults instead of failing a
        ``read`` on a dead descriptor.  Failures are emitted as
        ``error`` events instead of raised, stamped with the
        requesting connection's ``connId`` (when non-empty) so only
        the window that clicked "Update" renders the error banner.

        Args:
            script: Absolute path of the ``install.sh`` to execute.
            conn_id: Requesting connection id (``""`` to broadcast).
        """
        try:
            self._update_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._update_log_path, "ab") as log:
                subprocess.Popen(
                    ["bash", str(script)],
                    cwd=str(script.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            event: dict[str, Any] = {
                "type": "error",
                "text": f"Failed to start KISS Sorcar update: {exc}",
            }
            if conn_id:
                event["connId"] = conn_id
            self._printer.broadcast(event)

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
        container locally from ``~/.kiss/MY_TASK_TEMPLATES.md`` plus
        the bundled ``src/kiss/SAMPLE_TASKS.md``.  The empty-list
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

        The three fanned-out init commands carry the ``ready``
        sender's ``connId`` so their replies (``models``,
        ``inputHistory``, ``configData``) reach ONLY the window that
        just (re)connected.  Without the stamp the replies would be
        broadcast to every connected client — opening or reloading
        one browser window would repaint every sibling window's model
        picker (resetting its selected model to the default), clobber
        any open settings form, and reset its input-history cache.

        Args:
            cmd: The ``ready`` message from the browser (already
                stamped with the connection's ``connId`` by
                :meth:`_dispatch_client_command`).
            websocket: The client connection (for direct replies).
        """
        tab_id = cmd.get("tabId", "")
        conn_id = cmd.get("connId", "")
        # A fresh ``ready`` is the unambiguous signal that the
        # frontend has reconnected and is re-claiming whatever tab
        # ids it carries.  Cancel the deferred ``closeTab`` for the
        # current tab id (and every restored tab below) so a reload
        # within :data:`_TAB_CLOSE_GRACE` keeps the backend state.
        self._cancel_pending_tab_close(tab_id)
        # Propagate the connection's stamped ``workDir`` (if any) onto
        # the fanned-out init commands.  ``_cmd_get_config`` reports
        # the command's ``workDir`` as ``config.work_dir`` so the
        # settings panel shows the folder THIS instance actually uses;
        # dropping the stamp here made every page-load ``configData``
        # fall back to the daemon-global work_dir instead of the
        # connection's pinned folder.
        work_dir = cmd.get("workDir", "")
        for init_cmd in ("getModels", "getInputHistory", "getConfig"):
            init: dict[str, Any] = {"type": init_cmd, "connId": conn_id}
            if work_dir:
                init["workDir"] = work_dir
            await self._run_cmd(init)
        await self._send_welcome_info()
        try:
            await self._endpoint_send(
                websocket,
                json.dumps({"type": "focusInput", "tabId": tab_id}),
            )
        except Exception:
            pass
        # Replay in-flight merge reviews after restored-tab session
        # replays below.  ``merge_data`` events are tab-stamped and
        # never persisted, so without this a page reload mid-review
        # loses the merge UI forever while the server-side
        # ``_WebMergeState`` (and the backend tab's ``is_merging``
        # flag) stay stuck.  However, when a refreshed tab has a
        # backend ``chatId``, ``resumeSession`` emits ``task_events``;
        # the frontend handles that by clearing ``#output`` before it
        # renders the replayed history.  Sending ``merge_data`` before
        # that history replay therefore reconstructs the diff panel
        # and then immediately erases it.  Collect unique merge tabs
        # during ready handling and replay them only after all
        # ``resumeSession`` calls have completed.  The set also avoids
        # duplicate active/restored replays; duplicate ``merge_data``
        # panels leave the first diff panel stale in the remote webapp
        # because later ``merge_nav`` updates only the most recent
        # panel.
        merge_tabs_to_replay: list[str] = []
        seen_merge_tabs: set[str] = set()
        if isinstance(tab_id, str) and tab_id:
            merge_tabs_to_replay.append(tab_id)
            seen_merge_tabs.add(tab_id)
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
            # M7: the list itself is type-checked above, but a
            # malformed (non-dict) ELEMENT must be skipped too — an
            # ``AttributeError`` here would propagate out of
            # ``_dispatch_client_command`` and tear down the whole
            # authenticated connection over one bad field.
            if not isinstance(rt, dict):
                logger.warning("ignoring non-dict restoredTabs entry: %r", rt)
                continue
            rt_id = rt.get("tabId", "")
            if rt_id:
                self._cancel_pending_tab_close(rt_id)
            chat_id = rt.get("chatId", "")
            if chat_id:
                await self._run_cmd(
                    {"type": "resumeSession", "chatId": chat_id,
                     "tabId": rt_id},
                )
            if (
                isinstance(rt_id, str)
                and rt_id
                and rt_id not in seen_merge_tabs
            ):
                merge_tabs_to_replay.append(rt_id)
                seen_merge_tabs.add(rt_id)
        for merge_tab_id in merge_tabs_to_replay:
            await self._replay_merge_review(merge_tab_id, websocket)

    async def _replay_merge_review(self, tab_id: str, websocket: Any) -> None:
        """Re-send an in-flight merge review to a reconnecting client.

        ``merge_data`` events are tab-stamped, so ``WebPrinter.broadcast``
        forwards them to currently-connected clients only — they are
        never recorded or persisted.  A browser that reloads mid-review
        would therefore never see the merge UI again even though the
        server still holds the unresolved :class:`_WebMergeState` (and
        the backend tab stays ``is_merging``).  The VS Code extension's
        ``MergeManager`` survives webview reloads in the extension
        host; for the web client the server is the only source of
        truth, so it replays ``merge_data`` + ``merge_started`` +
        ``merge_nav`` (resolutions included) to the reconnecting
        endpoint.  Sends are targeted at *websocket* only — sibling
        windows already received the original broadcast.

        Args:
            tab_id: The tab the connection (re-)claimed.
            websocket: The reconnecting client connection.
        """
        if not tab_id:
            return
        with self._merge_states_lock:
            if tab_id not in self._merge_states:
                return
        # Serialise with in-flight merge actions: the reject branches
        # of ``_apply_web_merge_action`` rewrite the reviewed files on
        # disk (truncate + write) and mutate the shared hunk ``cs``
        # offsets while holding this per-tab lock.  Reading the files
        # (``_augment_merge_data``) and hunk dicts without it could
        # hand the reconnecting client a torn ``current_text`` and
        # mid-mutation offsets that no later ``merge_nav`` broadcast
        # can repair (``merge_nav`` carries no file text).
        async with self._merge_action_lock(tab_id):
            # Re-check under the lock: the action we waited for may
            # have resolved the final hunk and finished the review.
            with self._merge_states_lock:
                state = self._merge_states.get(tab_id)
            if state is None or not state.remaining:
                return
            assert self._loop is not None
            # ``_augment_merge_data`` reads every reviewed file from
            # disk; do it off the event loop like the broadcast path's
            # callers.
            event = await self._loop.run_in_executor(
                None,
                _augment_merge_data,
                {
                    "type": "merge_data",
                    "tabId": tab_id,
                    "data": state.data,
                    "hunk_count": state.total_hunks,
                },
            )
            cur = state.current()
            nav = {
                "type": "merge_nav",
                "tabId": tab_id,
                "remaining": state.remaining,
                "total": state.total_hunks,
                "cur": (
                    {"fi": cur[0], "hi": cur[1]} if cur is not None else None
                ),
                "resolved": state.resolutions(),
            }
            try:
                await self._endpoint_send(websocket, json.dumps(event))
                await self._endpoint_send(
                    websocket,
                    json.dumps({"type": "merge_started", "tabId": tab_id}),
                )
                await self._endpoint_send(websocket, json.dumps(nav))
            except Exception:
                logger.debug("Merge review replay failed", exc_info=True)

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

    def _broadcast_reject_failure(
        self, tab_id: str, file_data: dict[str, Any], exc: OSError,
    ) -> None:
        """Report a failed hunk-rejection write to every client.

        Called from the reject branches of
        :meth:`_apply_web_merge_action` when restoring a file's base
        content fails on disk (canonical trigger: the agent deleted a
        tracked file and created a directory at the same path, so the
        restore write raises ``IsADirectoryError``).  The hunks of the
        affected file remain unresolved, the review stays open, and
        the user sees an ``error`` chat event instead of a silently
        dropped (or connection-killing) rejection.

        Args:
            tab_id: The tab whose merge review the action targeted.
            file_data: The merge-data file entry whose restore failed.
            exc: The ``OSError`` raised by the restore write.
        """
        fname = file_data.get("name") or file_data.get("target") or "file"
        logger.warning("Merge reject failed for %s: %s", fname, exc)
        self._printer.broadcast({
            "type": "error",
            "text": f"Failed to reject changes in {fname}: {exc}",
            "tabId": tab_id,
        })

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
                try:
                    await self._loop.run_in_executor(
                        None,
                        partial(
                            _reject_hunk_in_file,
                            fd["current"],
                            fd["base"],
                            hunk,
                            fd.get("target"),
                            binary=bool(fd.get("binary")),
                            link_target=fd.get("link_target"),
                            make_executable=bool(fd.get("exec")),
                        ),
                    )
                except OSError as exc:
                    # The restore write failed (e.g. the agent replaced
                    # the deleted file with a DIRECTORY of the same
                    # name → IsADirectoryError, or the target is not
                    # writable).  Nothing landed on disk, so the hunk
                    # must stay UNRESOLVED — and the failure must not
                    # propagate, or the transport loop would tear down
                    # the whole client connection over one bad hunk.
                    self._broadcast_reject_failure(tab_id, fd, exc)
                else:
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
                resolve_file = True
                if action == "reject-file":
                    try:
                        await self._loop.run_in_executor(
                            None, _reject_all_hunks_in_file, fd,
                            state.unresolved_in_file(fi),
                        )
                    except OSError as exc:
                        # See the per-hunk ``reject`` branch: the
                        # restore write failed, so the file's hunks
                        # stay unresolved and the review survives.
                        self._broadcast_reject_failure(tab_id, fd, exc)
                        resolve_file = False
                if resolve_file:
                    file_status = (
                        "rejected" if action == "reject-file" else "accepted"
                    )
                    for hi in state.unresolved_in_file(fi):
                        state.mark_resolved(fi, hi, file_status)
                    state.advance()
        elif action == "accept-all":
            for fi, hi in state.all_unresolved():
                state.mark_resolved(fi, hi, "accepted")
        elif action == "reject-all":
            unresolved_by_file: dict[int, list[int]] = {}
            for fi, hi in state.all_unresolved():
                unresolved_by_file.setdefault(fi, []).append(hi)
            for fi, his in unresolved_by_file.items():
                fd = state.files[fi]
                # Mark hunks resolved only AFTER their file's restore
                # write succeeded: marking up-front would zombify the
                # review on failure (remaining == 0 with the state
                # never popped and all-done never dispatched), while a
                # propagating exception killed the client connection
                # AND left sibling files unrestored.
                try:
                    await self._loop.run_in_executor(
                        None, _reject_all_hunks_in_file, fd, his,
                    )
                except OSError as exc:
                    self._broadcast_reject_failure(tab_id, fd, exc)
                    continue
                for hi in his:
                    state.mark_resolved(fi, hi, "rejected")

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

    def _detach_tunnel(self) -> None:
        """Reset tunnel bookkeeping without killing ``cloudflared``.

        Used by :meth:`start`'s shutdown ``finally`` so that a
        ``kiss-web`` exit (SIGTERM / KeyboardInterrupt / launchd
        restart / VS Code extension's ``pkill kiss-web``) does **not**
        take the public Cloudflare tunnel down with it.  The spawned
        ``cloudflared`` was launched with ``start_new_session=True``
        and its pid + metrics port were persisted to
        ``~/.kiss/cloudflared.pid`` by :meth:`_spawn_cloudflared`, so
        the next ``kiss-web`` instance re-adopts it via
        :func:`_try_adopt_existing_cloudflared` and keeps serving on
        the same ``*.trycloudflare.com`` (or named-tunnel) hostname.

        This is the difference between :meth:`_stop_tunnel` (kills
        the spawned ``cloudflared`` immediately — used by the
        watchdog when the tunnel is unhealthy and must be replaced)
        and :meth:`_detach_tunnel` (leaves the spawned ``cloudflared``
        running — used on graceful kiss-web shutdown so the public
        URL survives the restart).

        Critical detail: ``cloudflared`` was spawned with
        ``stderr=PIPE``.  When this ``kiss-web`` process exits, the
        pipe's read end (held only by this process) is closed by the
        kernel; ``cloudflared``'s next stderr write then returns
        ``EPIPE``, which the Go runtime turns into a fatal
        ``SIGPIPE`` for writes to fd 1/2.  Without a workaround, the
        spawned ``cloudflared`` would therefore die within seconds of
        this ``kiss-web`` exit — defeating the whole adoption design.
        To prevent that, ``_detach_tunnel`` hands the pipe's read end
        off to a tiny detached ``cat`` shim (its own session via
        ``start_new_session=True``) that drains the pipe forever.
        The shim survives this ``kiss-web``'s exit, so the read end
        stays open and ``cloudflared`` keeps writing happily until
        the next ``kiss-web`` adopts it or it is intentionally
        replaced.

        Like :meth:`_stop_tunnel`, this method does not delete
        ``~/.kiss/remote-url.json``: a sibling kiss-web that has
        already taken over may have overwritten it, and removing it
        would briefly blank the VS Code sidebar URL.
        """
        proc = self._tunnel_proc
        if proc is not None and proc.poll() is None:
            self._spawn_stderr_drain_shim(proc)
        self._tunnel_proc = None
        self._tunnel_adopted_pid = None
        self._tunnel_metrics_port = None
        self._tunnel_started_at = None
        self._tunnel_unhealthy_ticks = 0
        self._tunnel_failure_count = 0
        self._tunnel_next_retry = 0.0
        self._tunnel_rate_limited = False
        self._active_url = None

    @staticmethod
    def _spawn_stderr_drain_shim(
        proc: subprocess.Popen[str],
    ) -> subprocess.Popen[bytes] | None:
        """Hand off *proc*'s stderr pipe to a detached drain shim.

        Spawns ``cat`` with ``proc.stderr`` as its stdin and detaches
        it into its own session so it survives the current
        ``kiss-web`` exit.  The shim continuously reads (and
        discards) every byte ``cloudflared`` writes to its stderr,
        keeping the pipe's read end open and preventing the
        ``SIGPIPE``-on-next-write that would otherwise kill the
        adopted ``cloudflared`` shortly after this ``kiss-web``
        exits.  When ``cloudflared`` itself eventually dies, the
        pipe closes from the write side and ``cat`` exits cleanly.

        Best-effort: a ``cat`` spawn failure (missing binary, EMFILE,
        permission error) is logged at DEBUG and otherwise ignored.
        The worst case is a return to the pre-fix behaviour for that
        particular shutdown — ``cloudflared`` may die from
        ``SIGPIPE`` and the next ``kiss-web`` will mint a fresh
        public URL — which is still no worse than no detach at all.

        Args:
            proc: The ``cloudflared`` subprocess; must have been
                started with ``stderr=PIPE``.

        Returns:
            The detached shim's ``Popen`` handle on success, or
            ``None`` if no stderr pipe was available or the shim
            spawn failed.
        """
        stderr = proc.stderr
        if stderr is None:
            return None
        try:
            shim = subprocess.Popen(
                ["cat"],
                stdin=stderr.fileno(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # New session so the shim survives this kiss-web's
                # exit *and* is not killed by a process-group signal
                # sent to kiss-web (e.g. ``pkill -g``).
                start_new_session=True,
                close_fds=True,
            )
        except (OSError, ValueError):
            logger.debug(
                "Failed to spawn stderr drain shim for cloudflared",
                exc_info=True,
            )
            return None
        return shim


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
        #
        # Windows does not support Unix-domain sockets; skip the bind
        # attempt so the ``except Exception`` below is not used for
        # platform detection.
        if sys.platform == "win32":
            logger.info(
                "UDS not supported on Windows; local extension "
                "clients will fall back to WSS",
            )
            self._uds_server = None
        else:
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
        # Skip the tunnel startup when ``cloudflared`` is not installed
        # — without this the spawn raises ``FileNotFoundError`` deep
        # inside ``_start_tunnel`` and we log a misleading warning.
        if self.use_tunnel and not shutil.which("cloudflared"):
            logger.warning(
                "cloudflared not found in PATH; tunnel disabled",
            )
            self.use_tunnel = False
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

        # If the previous instance of this daemon SIGTERMed itself in
        # response to a user-initiated "Server reset", drop a delayed
        # broadcast announcing the restart completed so reconnecting
        # clients see a confirmation toast (paired with the
        # "Restarting the KISS Sorcar web server…" toast emitted by
        # the previous instance).  The flag file is removed eagerly
        # so a *crash* respawn or a routine launchd kick — neither
        # of which writes the flag — never replays an obsolete
        # notification on the next launch.  The delay gives the VS
        # Code extension's ``AgentClient`` and any reconnecting
        # browser webview time to reattach to the freshly-bound UDS
        # / WSS before the broadcast goes out.
        self._maybe_schedule_server_reset_complete()

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
        active_task_history_ids: set[str] = set()
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
                    if th_id:
                        active_task_history_ids.add(str(th_id))

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
            # Leave the spawned ``cloudflared`` running so the next
            # ``kiss-web`` adopts it (preserving the public tunnel
            # URL across restarts) — see :meth:`_detach_tunnel`.
            self._detach_tunnel()

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
