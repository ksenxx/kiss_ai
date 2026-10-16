# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Standalone web server for remote KISS Sorcar access.

Provides HTTPS + WSS access to the Sorcar chat interface from any
browser, including mobile devices.  Uses the ``websockets`` library to
serve both HTTPS (for the HTML page and static media assets) and
WSS (for bidirectional command/event communication) on a single port.
TLS is always enabled; when no explicit certificate is provided a
machine-local CA and a server certificate signed by it are auto-generated
in ``~/.kiss/tls/`` (:mod:`kiss.core.tls_certs`).  Trusting the CA once
(``kiss-web --trust-ca`` on this machine, the ``/ca.crt`` download on a
phone) removes the browser warning on the Local and LAN URLs.

Authentication uses the ``remote_password`` setting from
``~/.kiss/config.json`` (the password ``install.sh`` printed; every
frame of a password-authenticated client is checked against it).  A
browser on this machine itself (a loopback peer with no forwarded
client IP) is admitted without it.  While no password is set, the server is
localhost-only: non-loopback peers are refused with 403 (HTTP and the
WebSocket upgrade alike) and no tunnel is started, so neither the LAN
nor the internet can reach the app.  An optional ``cloudflared`` tunnel can
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
import base64
import binascii
import collections
import contextlib
import errno
import functools
import hashlib
import html
import json
import logging
import mimetypes
import os
import platform
import re
import secrets
import select
import shutil
import signal
import socket
import ssl
import stat as stat_module
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from asyncio import sslproto
from collections.abc import Callable, Coroutine, Iterable
from concurrent.futures import Future as ConcurrentFuture
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit

import websockets
from websockets.asyncio.server import Server as WebSocketServer
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from kiss.agents.sorcar import cron_agent, local_endpoint
from kiss.agents.sorcar._concurrency import locked_call
from kiss.agents.sorcar.persistence import (
    _load_all_chat_events_by_chat_id,
    _load_chat_events_by_task_id,
    _load_subagent_rows_by_parent_task_id,
    _queue_chat_event,
)
from kiss.core import tls_certs
from kiss.core.brand import BRAND, HOME_DIR, PRODUCT_NAME
from kiss.core.browser_handoff import set_browser_tab_opener
from kiss.core.config import get_jobs_root as get_jobs_root
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.processes import find_bash, suppress_error_dialogs
from kiss.core.processes import pid_alive as _is_pid_alive
from kiss.core.processes import process_identity as _process_identity
from kiss.core.remote_password import password_accepted
from kiss.core.user_config import (
    apply_config_to_env,
    load_api_keys,
    load_config,
)
from kiss.core.utils import (
    is_root_dir,
    read_bytes_waiting_for_writer,
    replace_waiting_for_readers,
)
from kiss.server import agent_state
from kiss.server import sorcar as sorcar_api
from kiss.server.commands import broadcast_to_conn
from kiss.server.json_printer import (
    JsonPrinter,
    stamp_event_ts,
    with_synthesized_events,
)
from kiss.server.server import AgentServer
from kiss.server.speech import (
    DEFAULT_AUDIO_MODEL,
    MODEL_NAME,
    SpeakerIdentifier,
    _download_url_to_file,
    default_models_dir,
    transcribe_pcm,
)
from kiss.server.stall_watchdog import start_stall_watchdog
from kiss.server.task_update import TaskUpdateRunner
from kiss.server.tips import tips_data
from kiss.server.tricks import read_tricks_data
from kiss.viz_trajectory.server import find_job_dir as find_job_dir
from kiss.viz_trajectory.server import list_jobs as list_jobs
from kiss.viz_trajectory.server import (
    load_job_trajectories as load_job_trajectories,
)

__all__ = ["RemoteAccessServer", "WebPrinter"]

logger = logging.getLogger(__name__)

# An outbound payload: serialised JSON, or a replay slot reserved by
# ``JsonPrinter.replay_snapshot`` whose JSON is supplied later.
Payload = str | ConcurrentFuture[str]

MEDIA_DIR = Path(__file__).resolve().parent.parent / "webapp" / "media"

VOICE_MODEL_URL = (
    "https://ccoreilly.github.io/vosk-browser/models/"
    "vosk-model-small-en-us-0.15.tar.gz"
)
_voice_model_lock = threading.Lock()


def _voice_model_cache_path() -> Path:
    """Return the wake-word archive path: override or lazy default.

    Resolved on every call so a ``KISS_HOME`` set after this module
    was imported is honoured — freezing it at import time made the
    browser wake-word pipeline re-download the 40MB archive into an
    empty test home instead of reusing ``~/.kiss/models``.  Assigning
    ``web_server.VOICE_MODEL_CACHE`` remains a supported test
    override, matching the lazy ``_URL_FILE`` attribute below.
    """
    override = globals().get("VOICE_MODEL_CACHE")
    if isinstance(override, Path):
        return override
    return default_models_dir() / f"{MODEL_NAME}.tar.gz"


def _atomic_publish(target: Path, write_tmp: Callable[[Path], object]) -> None:
    """Atomically publish *target* via a pid-unique temp + ``Path.replace``.

    Creates the parent directory, calls *write_tmp* with a pid-unique
    temporary path in the same directory, then atomically renames it
    onto *target* so a concurrent reader can never observe a torn
    write.  The pid suffix matters: an in-process lock cannot
    serialize a SIBLING process (a second kiss-web daemon, a test
    running next to a live daemon), and a shared fixed temp name would
    let two processes interleave writes into the same file and then
    publish the corrupted result — or race ``replace`` so the loser
    raised ``FileNotFoundError``.  On failure the temp file is removed
    and the exception propagates to the caller.

    Args:
        target: Final path to publish.
        write_tmp: Callable that writes the content to the temp path.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(
        f"{target.name}.{os.getpid()}.{threading.get_ident()}."
        f"{uuid.uuid4().hex[:8]}.tmp",
    )
    try:
        write_tmp(tmp)
        replace_waiting_for_readers(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_text_to(tmp: Path, text: str) -> None:
    """Write *text* to *tmp* as UTF-8 (writer for :func:`_atomic_publish`)."""
    tmp.write_text(text, encoding="utf-8")


_SAVE_FILE_LOCK = threading.Lock()
"""Serializes ``saveFile``'s version check + publish across worker threads.

One lock for every path, deliberately: saves are rare, short, and
already run off the event loop in :func:`asyncio.to_thread`, so the
simplicity beats a per-path lock table that would need its own
housekeeping.  (Sibling daemons are not covered — see
``RemoteAccessServer._handle_save_file``.)
"""


def _file_version(st: os.stat_result) -> str:
    """Return the ``"<st_mtime_ns>:<st_size>"`` stamp of a file's state.

    Sent with ``fileContent`` and echoed by ``saveFile`` so
    ``RemoteAccessServer._handle_save_file`` can tell whether the file
    changed on disk while it was open in the remote editor.  A string,
    deliberately: ``st_mtime_ns`` (about 1.8e18) does not survive the
    round trip through a JavaScript number (2^53 ≈ 9e15), so an
    integer stamp compared back on the server would never match.
    """
    return f"{st.st_mtime_ns}:{st.st_size}"


def _write_bytes_with_mode(tmp: Path, data: bytes, st: os.stat_result) -> None:
    """Write *data* to *tmp* and copy *st*'s permission bits onto it.

    Writer for :func:`_atomic_publish` used by
    ``RemoteAccessServer._handle_save_file``: ``Path.replace`` swaps
    the inode, so without this an executable script saved from the
    web editor would come back non-executable.
    """
    tmp.write_bytes(data)
    os.chmod(tmp, stat_module.S_IMODE(st.st_mode))


def _atomic_write_text(target: Path, text: str) -> None:
    """Atomically write *text* (UTF-8) to *target*.

    Thin text convenience over :func:`_atomic_publish`; see it for the
    pid-unique-temp + ``Path.replace`` rationale.

    Args:
        target: Final path to publish.
        text: Full file content to write.
    """
    _atomic_publish(target, partial(_write_text_to, text=text))


def _download_voice_model_to(tmp: Path) -> None:
    """Download the browser voice-model archive to *tmp*.

    Writer callback for :func:`_atomic_publish` used by
    :func:`_ensure_voice_model`.  Uses the timeout-bounded
    :func:`kiss.server.speech._download_url_to_file` rather than
    ``urllib.request.urlretrieve``, which accepts no timeout: a
    black-holed connection blocked forever while the caller held
    ``_voice_model_lock`` on a default-executor thread, and every
    retrying ``/voice-model.tar.gz`` request then parked another
    shared executor worker on the lock until the daemon stopped
    dispatching commands entirely.
    """
    _download_url_to_file(VOICE_MODEL_URL, tmp)


def _ensure_voice_model() -> Path | None:
    """Return the cached wake-word model archive, downloading on first use.

    Serializes concurrent downloads with a lock and writes through a
    pid-unique temporary file (atomically published via
    ``Path.replace``) so a partially-downloaded archive is never
    served.  The pid suffix matters: the in-process lock cannot
    serialize a SIBLING process (a second kiss-web daemon, a test
    running next to a live daemon), and a shared fixed temp name let
    two processes interleave writes into the same file and then
    publish the corrupted result — or race ``replace`` so the loser
    raised ``FileNotFoundError`` and returned ``None``.

    Returns:
        Path to the cached ``.tar.gz`` archive, or ``None`` when the
        download failed (e.g. no network).
    """
    with _voice_model_lock:
        cache = _voice_model_cache_path()
        if cache.is_file() and cache.stat().st_size > 0:
            return cache
        try:
            _atomic_publish(cache, _download_voice_model_to)
            return cache
        except Exception:
            logger.exception("voice model download failed: %s", VOICE_MODEL_URL)
            return None
# Per media asset: (stat fingerprint, sha256 prefix) of the bytes last
# hashed by _media_url.  The fingerprint (see _media_fingerprint) is
# checked on every call so the hash is recomputed after the file
# changes on disk.
_MEDIA_VERSION_CACHE: dict[str, tuple[tuple[int, int, int, int], str]] = {}

TRAJECTORY_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "viz_trajectory"
    / "templates"
    / "index.html"
)

TUNNEL_CHECK_INTERVAL = 15
TRUSTED_CLIENTS_POLL = 2.0
"""Seconds between the daemon's looks at the enrolled machine CAs' directory."""
MACHINES_REGISTRY_POLL_SECONDS = 2.0
"""Seconds between the daemon's looks at ``machines.json`` (see ``_machines_registry_watch``)."""


def _file_stamp(path: Path) -> tuple[int, int, int] | None:
    """Return ``(st_mtime_ns, st_size, st_ino)`` of *path*, ``None`` when it is absent.

    Enough to notice a rewrite of a small registry file: an atomic
    rename-in-place moves the inode, an in-place write the mtime.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)

_IP_CHANGE_DEBOUNCE_TICKS = 4

_BIND_RETRY_ATTEMPTS = 5
_BIND_RETRY_BACKOFF: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
_BIND_RETRYABLE_ERRNOS: frozenset[int] = frozenset({
    errno.EADDRINUSE, errno.EADDRNOTAVAIL,
})

_VERSION_CHECK_INTERVAL: float = 3600
# How often an armed "Update when idle" re-checks the agent registry for
# in-flight tasks before launching the installer.
_IDLE_UPDATE_POLL_S: float = 5.0

_PYPI_LATEST_URL = "https://pypi.org/pypi/kiss-agent-framework/json"

_PYPI_FETCH_TIMEOUT = 5.0

_WS_PING_TIMEOUT = 10

_WS_HEARTBEAT_FRAME = json.dumps({"type": "heartbeat"})
"""Proof-of-life frame the watchdog sends every WSS client after each
successful keep-alive ping (see :meth:`RemoteAccessServer._ping_one_ws`)."""
# Catchable termination signals routed through
# ``RemoteAccessServer._handle_shutdown_signal``.  SIGHUP (terminal
# closed) does not exist on Windows, where only SIGTERM is available.
_SHUTDOWN_SIGNALS: tuple[int, ...] = tuple(
    sig
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None))
    if sig is not None
)
_SEND_TIMEOUT = 30.0
"""Seconds a client may leave its socket unread before it is dropped.

The watchdog's 15 s ping only detects a dead peer, not one that merely
stops reading, so such a peer would otherwise hold its send lock forever
while every later broadcast queues another pending send for it without
bound.
"""

_TUNNEL_UNHEALTHY_LIMIT_NAMED = 3

_TUNNEL_UNHEALTHY_LIMIT_QUICK = 40

_TUNNEL_STARTUP_GRACE = 120

_TUNNEL_BACKOFF_INITIAL = 60

_TUNNEL_BACKOFF_MAX = 1800

_TUNNEL_RATE_LIMIT_BACKOFF = 900

_TUNNEL_RATE_LIMIT_JITTER = 300

_TUNNEL_FORCE_RESTART_COOLDOWN_INITIAL = 60

_TUNNEL_FORCE_RESTART_COOLDOWN_MAX = 3600

_TUNNEL_FORCE_RESTART_RESET_AFTER_HEALTHY = 600

_SPAWN_FAILFAST_WINDOW = 1.0

_RATE_LIMIT_INDICATORS = (
    "error code: 1015",
    "error code 1015",
    "429 too many requests",
    'status_code="429',
    "status_code=429",
    "rate-limited",
    "rate limited",
)

_AUTH_FAIL_MAX = 5

_AUTH_FAIL_WINDOW = 60.0

_AUTH_LOCKOUT = 60.0

_MAX_RESTORED_TABS = 32

_MAX_ATTACHMENTS = 32

_SERVER_RESET_DELAY = 0.4

_SERVER_RESET_COMPLETE_DELAY = 3.0

_SERVER_RESET_FLAG_NAME = "server-reset-pending.json"

_SHUTDOWN_EXIT_FAILSAFE = 30.0

_MAX_PROMPT_BYTES = 1_000_000

_MAX_LINE_BYTES = 64 * 1024 * 1024

# Byte budget for the JSON tasks of one ``share_tasks`` reply, kept
# well under the 64 MiB frame limit (``_MAX_LINE_BYTES``) so the reply
# envelope and the frame's UTF-8 / escaping overhead never push a
# browser's share export past it.  Older tasks beyond the budget are
# dropped and the reply is flagged ``truncated`` (see
# _handle_share_chat_tasks).
_SHARE_TASKS_MAX_REPLY_BYTES = 24 * 1024 * 1024

# Echoed identifiers ride in every ``share_tasks`` reply; a client
# cannot make the reply overflow its frame by inflating them.
_SHARE_TASKS_MAX_ID_CHARS = 256


def _merging_tab_ids() -> list[str]:
    """Return the tabs whose merge/discard claim is still live (see ``_await_active_merges``)."""
    with agent_state.STATE_LOCK:
        return [
            state.tab_id
            for state in agent_state.agent_states.values()
            if state.merge_in_progress()
        ]


def _share_subagent_entries(
    root_task_id: str,
    max_bytes: int | None = None,
    peek: Callable[[str], list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Return every sub-agent transcript below *root_task_id*.

    Feeds the ``subagents`` list of one task in a ``share_tasks``
    reply (see :meth:`RemoteAccessServer._handle_share_chat_tasks`):
    the chat page's share export renders each sub-agent's
    transcript into a hidden section of the shared page, which the
    page's tab strip (``media/share.js``) opens and closes exactly
    like the live chat page's sub-agent tabs.

    Walks the ``parent_task_id`` tree breadth-first starting at the
    direct children of *root_task_id*
    (:func:`~kiss.agents.sorcar.persistence._load_subagent_rows_by_parent_task_id`,
    rowid order — the order the parent enqueued its sub-agents), so a
    sub-agent's own fan-outs (grandchildren, ...) ride along too.  A
    ``seen`` set guards against cycles and duplicate rows.

    A STILL-RUNNING sub-agent's events reach the database through an
    asynchronous writer, so its persisted transcript can lag the live
    run; when *peek* (the printer's ``peek_recording_for_task``)
    returns a live recording for a sub-agent, that recording replaces
    the persisted events — the same safeguard the session replay's
    ``_open_persisted_subagent_tabs`` applies.

    Args:
        root_task_id: ``task_history`` row id of the chat task.
        max_bytes: Optional byte budget; each entry is charged the
            UTF-8 length of its JSON encoding and the walk stops (the
            overflow flag set) at the first entry that does not fit —
            BEFORE loading the rest, so an oversized chat can never
            make the daemon materialize transcripts it is going to
            drop anyway.  ``None`` loads all.
        peek: Optional live-recording lookup by task id.

    Returns:
        ``(entries, bytes_used, overflowed)``.  *entries* is a flat
        list, parents before their children, each
        ``{"task", "task_id", "parent_task_id", "events"}`` with the
        events led by the ensured ``task_settings`` event; empty when
        the task fanned out no sub-agents.  *bytes_used* is the JSON
        byte total already charged against *max_bytes*.  *overflowed*
        is True when the budget cut the walk short.
    """
    out: list[dict[str, Any]] = []
    used = 0
    seen = {str(root_task_id)}
    frontier = [str(root_task_id)]
    while frontier:
        next_frontier: list[str] = []
        for parent_id in frontier:
            for row in _load_subagent_rows_by_parent_task_id(parent_id):
                sub_id = str(row.get("task_id") or "")
                if not sub_id or sub_id in seen:
                    continue
                seen.add(sub_id)
                next_frontier.append(sub_id)
                events = cast(
                    "list[dict[str, Any]]", row.get("events") or [],
                )
                if peek is not None:
                    live_events = peek(sub_id)
                    if live_events:
                        events = live_events
                entry = {
                    "task": row.get("task", ""),
                    "task_id": sub_id,
                    "parent_task_id": parent_id,
                    "events": with_synthesized_events(events, row),
                }
                if max_bytes is not None:
                    used += len(json.dumps(entry).encode("utf-8"))
                    if used > max_bytes:
                        return out, used, True
                out.append(entry)
        frontier = next_frontier
    return out, used, False

# ``websockets`` wraps the whole opening handshake - including a plain
# HTTP reply produced by ``process_request`` - in ``open_timeout``.  Its
# 10s default silently guillotines large downloads such as the 40MB
# wake-word model, which reaches the browser as ERR_EMPTY_RESPONSE.
_OPEN_TIMEOUT_SECONDS = 300.0

# Upper bound on waiting for a sibling daemon's ``.tls.lock`` (a
# self-signed cert generation takes well under a second).
_TLS_LOCK_TIMEOUT_S = 30.0

_MAX_VOICE_AUDIO_B64 = 4 * 1024 * 1024

_TLS_DIR: Path | None = None


def _tls_dir() -> Path:
    """Return the directory holding the self-signed TLS cert/key pair."""
    return _TLS_DIR if _TLS_DIR is not None else kiss_home() / "tls"


def _url_file_path() -> Path:
    """Return the persisted remote-URL path: override or lazy default.

    Assigning ``web_server._URL_FILE`` is a supported test override,
    matching the lazy ``CONFIG_DIR`` / ``CONFIG_PATH`` attributes in
    :mod:`kiss.core.user_config`.  The override must be
    consulted here (rather than only exposed through ``__getattr__``)
    because production consumers such as :class:`RemoteAccessServer`
    call this accessor directly.
    """
    override = globals().get("_URL_FILE")
    return override if override is not None else kiss_home() / "remote-url.json"


if TYPE_CHECKING:
    _URL_FILE: Path
    VOICE_MODEL_CACHE: Path


def __getattr__(name: str) -> Path:
    """Resolve ``_URL_FILE`` / ``VOICE_MODEL_CACHE`` lazily (PEP 562).

    Several test modules import these names by value; resolving them
    at access time keeps that import surface working while honoring a
    ``KISS_HOME`` set after this module was first imported.
    """
    if name == "_URL_FILE":
        return _url_file_path()
    if name == "VOICE_MODEL_CACHE":
        return _voice_model_cache_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _bound_loopback(*ws_servers: Any) -> tuple[str, int]:
    """Return the ``(host, port)`` a same-machine client should dial.

    A listener bound to ``localhost`` or a wildcard may hold several
    sockets (IPv4 and IPv6, each with its own ephemeral port when the
    daemon asked for port 0).  Prefers an IPv4 socket reachable at
    ``127.0.0.1``, then an IPv6 one reachable at ``::1``, over the
    given servers in order (``None`` entries are skipped); when no
    socket is reachable over loopback, the first bound address.

    Args:
        ws_servers: ``websockets`` servers whose ``sockets`` are bound.

    Returns:
        The URL host (IPv6 in brackets) and the port of the chosen socket.
    """
    fallback: tuple[str, int] | None = None
    v6: tuple[str, int] | None = None
    sockets = [
        sock for server in ws_servers if server is not None for sock in server.sockets
    ]
    for sock in sockets:
        addr, port = sock.getsockname()[:2]
        if sock.family == socket.AF_INET:
            if addr in ("127.0.0.1", "0.0.0.0"):
                return "127.0.0.1", port
        elif sock.family == socket.AF_INET6 and addr in ("::1", "::") and v6 is None:
            v6 = ("[::1]", port)
        if fallback is None:
            fallback = (f"[{addr}]" if sock.family == socket.AF_INET6 else addr, port)
    if v6 is not None:
        return v6
    if fallback is None:
        raise RuntimeError("the WSS listener has no bound socket")
    return fallback


def _forwarded_client_ip(websocket: ServerConnection) -> str:
    """Return the real client IP forwarded by cloudflared, or ``""``.

    cloudflared sets ``Cf-Connecting-Ip`` to the origin visitor's IP on
    the WebSocket upgrade request; ``X-Forwarded-For`` (whose first hop
    is the original client) is used as a fallback.  The returned value
    must be treated as trusted **only** for loopback TCP peers (see
    :meth:`RemoteAccessServer._client_ip`).

    Args:
        websocket: The server connection whose upgrade request headers
            are inspected.

    Returns:
        The forwarded client IP as a string, or ``""`` when no usable
        header is present.
    """
    request = getattr(websocket, "request", None)
    headers = getattr(request, "headers", None)
    if headers is None:
        return ""
    cf_ip = str(headers.get("Cf-Connecting-Ip") or "").strip()
    if cf_ip:
        return cf_ip
    xff = str(headers.get("X-Forwarded-For") or "")
    first = xff.split(",")[0].strip()
    if first:
        return first
    return ""


_HEAD_200 = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Length: 0\r\n"
    b"Connection: close\r\n"
    b"\r\n"
)

_HEAD_403 = (
    b"HTTP/1.1 403 Forbidden\r\n"
    b"Content-Length: 0\r\n"
    b"Connection: close\r\n"
    b"\r\n"
)


def _connection_peer_is_loopback(connection: Any) -> bool:
    """Return True when *connection*'s raw TCP peer is loopback.

    Deliberately ignores the forwarded-client headers that
    :meth:`RemoteAccessServer._client_ip` trusts for loopback peers:
    this check gates ACCESS (the empty-password localhost-only
    lockdown), so only the unforgeable transport-level peer address
    may be consulted.  An unknown peer address counts as NOT loopback
    (fail closed).

    Args:
        connection: A WebSocket server connection (or any object with
            a ``remote_address`` tuple).

    Returns:
        True when the TCP peer is an IPv4/IPv6 loopback address.
    """
    addr = getattr(connection, "remote_address", None)
    peer_ip = str(addr[0]) if addr and len(addr) >= 1 else ""
    return sorcar_api.is_loopback_ip(peer_ip)


def _head_health_response(connection: Any) -> bytes:
    """Return the reply for a HEAD health check from *connection*.

    Cloudflare's origin health checks arrive from the local
    cloudflared (a loopback peer) and must keep getting 200 so the
    tunnel stays registered.  A NON-loopback HEAD probe is answered
    403 while the configured ``remote_password`` is empty, matching
    the localhost-only lockdown that
    :meth:`RemoteAccessServer._process_request` enforces for every
    parsed request — without this, a LAN peer's HEAD would bypass the
    gate (it is answered before the websockets HTTP parser runs).
    The config file is read only on that rare non-loopback path, so
    the hot loopback health checks never pay the disk read.

    Args:
        connection: The server connection that received the HEAD.

    Returns:
        The raw HTTP response bytes to write to the transport.
    """
    if _connection_peer_is_loopback(connection):
        return _HEAD_200
    if not str(load_config().get("remote_password", "") or ""):
        return _HEAD_403
    return _HEAD_200

# Cap on the bytes buffered while waiting for the first CRLF of an
# incoming request line.  Matches the conventional HTTP request-line
# limit; anything longer is fed to the websockets parser (which
# rejects it) instead of being buffered without bound (F4-08).
_MAX_HEAD_LINE_BYTES = 8192


# Seconds a freshly accepted peer gets to send its first byte (a TLS
# ClientHello or an HTTP request line) before the socket is dropped.
_SNIFF_TIMEOUT_SECONDS = 10.0

_PLAIN_HTTP_REFUSAL = (
    b"HTTP/1.1 400 Bad Request\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"Connection: close\r\n"
    b"Content-Length: 58\r\n\r\n"
    b"Plain HTTP is served to this computer only; use https://.\n"
)


async def _peek_first_byte(sock: Any) -> bytes:
    """Return the first byte a peer sent on *sock* without consuming it.

    *sock* is the transport's ``TransportSocket``, whose reading is
    paused.  A duplicate descriptor shares its open file, so waiting
    for the duplicate to become readable (the loop refuses a reader on
    the transport's own descriptor) and ``MSG_PEEK``-ing it leaves the
    bytes in the kernel buffer for the transport to read afterwards.
    ``b""`` means the peer closed the connection first.
    """
    with sock.dup() as peek_sock:
        while True:
            await _wait_readable(peek_sock.fileno())
            try:
                return bytes(peek_sock.recv(1, socket.MSG_PEEK))
            except BlockingIOError:
                continue


async def _wait_readable(fd: int) -> None:
    """Wait until *fd* is readable.

    Through the loop's reader callbacks where they exist (the selector
    loops of Linux and macOS); Windows' proactor loop has none, so there
    a worker thread blocks in ``select`` for at most the sniff timeout.
    """
    loop = asyncio.get_running_loop()
    readable: asyncio.Future[None] = loop.create_future()
    try:
        loop.add_reader(fd, _settle, readable)
    except NotImplementedError:
        await asyncio.to_thread(select.select, [fd], [], [], _SNIFF_TIMEOUT_SECONDS)
        return
    try:
        await readable
    finally:
        loop.remove_reader(fd)


def _settle(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _consume_handshake_result(handshake: asyncio.Future[None]) -> None:
    """Read a failed TLS handshake's exception (it is logged by the TLS layer)."""
    if not handshake.cancelled():
        handshake.exception()


class _HeadAwareServerConnection(ServerConnection):
    """``ServerConnection`` that speaks TLS or plain HTTP as the peer does.

    One port serves both: the first byte of a connection is peeked at
    before ``websockets`` sees the transport.  A TLS ClientHello
    (``0x16``) wraps the transport in TLS with the server's certificate,
    as every LAN browser, tunnel edge and mTLS machine link sends; any
    other first byte is plain HTTP, served to peers on this machine's
    loopback (``http://127.0.0.1:8787``, cloudflared's origin) and
    refused with 400 from anywhere else, so a password never crosses a
    network in the clear.

    It also answers the HEAD health checks Cloudflare tunnels send
    (``websockets`` accepts only GET and would otherwise mark the
    origin unhealthy): a first request line starting with ``HEAD ``
    gets a 200 and the connection is closed.
    """

    def __init__(
        self,
        protocol: Any,
        server: Any,
        ssl_context: ssl.SSLContext | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(protocol, server, **kwargs)
        self._ssl_context = ssl_context
        self._sniff_task: asyncio.Task[None] | None = None
        # Whether ``websockets`` has been given the transport (plain
        # right after the sniff; TLS once the handshake is done).
        self._attached: bool = False
        self._head_buffer: bytes = b""
        self._head_checked: bool = False

    def connection_made(self, transport: Any) -> None:
        """Hold a fresh transport back until its first byte says TLS or plain.

        Called twice for a TLS connection: with the raw transport (which
        is only sniffed) and, once the handshake is done, by the TLS
        layer with the TLS transport, which ``websockets`` gets.
        """
        if self._ssl_context is None or self._attached:
            self._attach(transport)
            return
        transport.pause_reading()
        self._sniff_task = asyncio.get_running_loop().create_task(self._sniff(transport))

    def _attach(self, transport: Any) -> None:
        self._attached = True
        super().connection_made(transport)

    async def _sniff(self, transport: Any) -> None:
        try:
            await self._sniff_or_raise(transport)
        except Exception:
            # Whatever went wrong, a paused transport must not linger.
            logger.debug("sniffing a connection failed", exc_info=True)
            transport.close()

    async def _sniff_or_raise(self, transport: Any) -> None:
        try:
            first = await asyncio.wait_for(
                _peek_first_byte(transport.get_extra_info("socket")),
                _SNIFF_TIMEOUT_SECONDS,
            )
        except (TimeoutError, OSError):
            first = b""
        if not first:
            transport.close()
            return
        if first == b"\x16" and self._ssl_context is not None:
            # What ``loop.start_tls`` does, except that the TLS layer
            # calls ``connection_made`` itself when the handshake
            # completes: the request a TLS 1.3 client sends right behind
            # its Finished (and a close_notify after it) then reaches an
            # attached connection, in order, before the TLS layer acts
            # on the EOF.  ``start_tls`` would hand the transport back
            # only later, after those bytes were delivered.
            self._attached = True
            loop = asyncio.get_running_loop()
            handshake: asyncio.Future[None] = loop.create_future()
            handshake.add_done_callback(_consume_handshake_result)
            tls = sslproto.SSLProtocol(
                loop, self, self._ssl_context, handshake, server_side=True,
            )
            transport.set_protocol(tls)
            tls.connection_made(transport)
            transport.resume_reading()
            return
        peer = transport.get_extra_info("peername")
        if peer and sorcar_api.is_loopback_ip(str(peer[0])):
            self._attach(transport)
            transport.resume_reading()
            return
        transport.write(_PLAIN_HTTP_REFUSAL)
        transport.close()

    def connection_lost(self, exc: Exception | None) -> None:
        """Stop a pending sniff; hand the loss to ``websockets`` once it knows the transport."""
        if self._sniff_task is not None and not self._sniff_task.done():
            self._sniff_task.cancel()
        if hasattr(self, "transport"):
            super().connection_lost(exc)

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
            if len(self._head_buffer) > _MAX_HEAD_LINE_BYTES:
                # An unauthenticated peer sent an over-long first
                # request line; stop buffering (which would otherwise
                # grow without bound) and hand everything to the
                # websockets HTTP parser, whose own limits reject it.
                self._head_checked = True
                buffered = self._head_buffer
                self._head_buffer = b""
                super().data_received(buffered)
            return
        self._head_checked = True
        first_line = self._head_buffer[:idx]
        if first_line.startswith(b"HEAD "):
            transport = self.transport
            if transport is not None:
                transport.write(_head_health_response(self))
                transport.close()
            return
        buffered = self._head_buffer
        self._head_buffer = b""
        super().data_received(buffered)


_OPEN_FILE_MAX_BYTES = 2_000_000

# Binary files the remote webapp shows in a tab instead of refusing: a
# PDF opens in the browser's built-in viewer, an image as a picture.
# Their bytes travel base64-encoded inside the fileContent reply, so
# the cap keeps one reply well under the 64 MiB WebSocket frame limit.
_OPEN_BINARY_MAX_BYTES = 24 * 1024 * 1024
_INLINE_BINARY_MIMES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/x-icon",
        "image/vnd.microsoft.icon",
        "image/avif",
    }
)


def _inline_binary_mime(path: Path) -> str:
    """The MIME type *path* is served inline as, or ``""`` for text/other."""
    mime = mimetypes.guess_type(path.name, strict=False)[0] or ""
    return mime if mime in _INLINE_BINARY_MIMES else ""

# Caps on an openFile directory-listing reply, so one click on a huge
# directory (node_modules, .git/objects, ...) cannot produce a
# multi-megabyte WSS message: at most this many entries, and at most
# this many characters of entry lines (deep absolute prefixes repeat on
# every line, so an entry cap alone does not bound the reply size); the
# header and truncation-note lines are the only text beyond that.
_DIR_LISTING_MAX_ENTRIES = 2_000
_DIR_LISTING_MAX_CHARS = 512_000


def _directory_listing_text(directory: Path) -> str:
    """Render *directory* as the plain-text listing served for a dir click.

    The remote-web client shows ``openFile`` replies in a text content
    tab, so a clicked directory link is answered with this listing
    instead of file bytes: one absolute path per line, directories
    first (marked with a trailing ``/``), each group sorted by name.
    Unreadable directories raise ``OSError`` for the caller's existing
    error handling; listings longer than
    :data:`_DIR_LISTING_MAX_ENTRIES` entries or whose entry lines would
    exceed :data:`_DIR_LISTING_MAX_CHARS` characters are truncated with
    a trailing note.

    Args:
        directory: The resolved, existing directory to list.

    Returns:
        The listing text, starting with a ``<path>:`` header line.
    """
    dirs: list[str] = []
    files: list[str] = []
    for entry in sorted(directory.iterdir(), key=lambda p: p.name):
        try:
            is_dir = entry.is_dir()
        except OSError:
            is_dir = False
        if is_dir:
            dirs.append(f"{entry}/")
        else:
            files.append(str(entry))
    entries = dirs + files
    lines = [f"{directory}:", ""]
    shown = 0
    used_chars = 0
    for entry_line in entries:
        if shown >= _DIR_LISTING_MAX_ENTRIES:
            break
        if used_chars + len(entry_line) + 1 > _DIR_LISTING_MAX_CHARS:
            break
        lines.append(entry_line)
        used_chars += len(entry_line) + 1
        shown += 1
    omitted = len(entries) - shown
    if omitted:
        lines.append(f"... {omitted} more entries not shown")
    if not entries:
        lines.append("(empty directory)")
    return "\n".join(lines) + "\n"

# The curl installer (scripts/install.sh) clones the public repo into
# ~/.kiss/kiss_ai; the Update button runs the install.sh of that clone.  Kept
# literal (not $KISS_HOME-relative) to match the installer exactly.
_KISS_AI_ROOT = Path.home() / ".kiss" / "kiss_ai"

# The public curl bootstrap (README's install one-liner).  When
# ~/.kiss/kiss_ai/install.sh is missing (the clone was deleted) the
# Update button falls back to this script, which clones ~/.kiss/kiss_ai
# and hands over to its install.sh.
_DEFAULT_BOOTSTRAP_INSTALL_URL = (
    "https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh"
)


def _bootstrap_install_url() -> str:
    """Return the URL of the curl bootstrap installer.

    Honours ``$KISS_UPDATE_BOOTSTRAP_URL`` (forks, tests — ``curl``
    accepts ``file://`` URLs) and otherwise returns the public
    ``scripts/install.sh`` raw URL from the README.

    Returns:
        The bootstrap installer URL.
    """
    return (
        os.environ.get("KISS_UPDATE_BOOTSTRAP_URL")
        or _DEFAULT_BOOTSTRAP_INSTALL_URL
    )


def _find_install_script(root: Path) -> Path | None:
    """Return ``install.sh`` inside *root* if it exists, else ``None``.

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


def _reap_proc(proc: subprocess.Popen[str], kill: bool = False) -> None:
    """Close *proc*'s stderr pipe and wait for it (killing it first if asked).

    The tunnel spawn retains a failed ``cloudflared`` child between
    attempts; releasing it here keeps its pipe from lingering until GC.

    Args:
        proc: The child process to reap.
        kill: Kill a still-running child before waiting for it.
    """
    if kill:
        proc.kill()
    if proc.stderr is not None:
        proc.stderr.close()
    proc.wait()


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


_CLOUDFLARED_PIDFILE: Path | None = None


def _cloudflared_pidfile() -> Path:
    """Return the path of the persisted cloudflared PID file."""
    if _CLOUDFLARED_PIDFILE is not None:
        return _CLOUDFLARED_PIDFILE
    return kiss_home() / "cloudflared.pid"


_SYSTEMD_RUN_SCOPE_PREFIX = (
    "systemd-run", "--user", "--scope", "--collect", "--quiet", "--",
)


def _current_cgroup() -> str:
    """Return this process's ``/proc/self/cgroup`` content ('' off-Linux).

    The read only fails where the file does not exist (macOS, BSD) or
    procfs is unmounted — environments without systemd cgroups, where
    returning ``''`` correctly reports "not in a systemd service".
    """
    try:
        return Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return ""


def _cloudflared_launch_prefix(cgroup: str | None = None) -> list[str]:
    """Return an argv prefix that detaches cloudflared from a service cgroup.

    When ``kiss-web`` runs as a systemd service (the ``kiss-web.service``
    user unit installed by the installer), ``systemctl restart``
    SIGTERMs **every** process in the service's control group — systemd's
    default ``KillMode=control-group`` — so the ``start_new_session=True``
    cloudflared dies together with the daemon.  ``start_new_session``
    creates a new process *session* but cannot leave the *cgroup*, which
    is why every production restart logged "cloudflared pidfile points to
    dead pid; ignoring" and minted a fresh ``*.trycloudflare.com``
    hostname, defeating :func:`_try_adopt_existing_cloudflared`.

    Launching cloudflared through ``systemd-run --user --scope`` places it
    in its own transient ``run-*.scope`` unit *outside* the service
    cgroup, out of ``systemctl restart``'s reach.  With ``--scope`` the
    systemd-run process registers the scope and then ``exec``s the payload
    in-place, so the :class:`subprocess.Popen` pid IS cloudflared's pid
    (the pidfile and adoption logic keep working unchanged) and the
    stderr pipe used to parse the tunnel URL is preserved.  ``--collect``
    garbage-collects the scope even when cloudflared exits non-zero.

    Args:
        cgroup: ``/proc/self/cgroup`` content to inspect; ``None`` reads
            the real file.  Only the *final* path component is matched
            against ``.service`` — every user process lives under
            ``user@UID.service``, and transient scopes end in ``.scope``,
            so a substring match would misfire.

    Returns:
        The ``systemd-run`` argv prefix when this process is inside a
        systemd service cgroup and ``systemd-run`` is installed,
        otherwise ``[]`` (spawn cloudflared directly).
    """
    text = _current_cgroup() if cgroup is None else cgroup
    in_service = False
    for line in text.splitlines():
        unit = line.rsplit(":", 1)[-1].rsplit("/", 1)[-1].strip()
        if unit.endswith(".service"):
            in_service = True
            break
    if not in_service:
        return []
    if shutil.which("systemd-run") is None:
        return []
    return list(_SYSTEMD_RUN_SCOPE_PREFIX)


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
        _atomic_write_text(_cloudflared_pidfile(), json.dumps(data) + "\n")
    except OSError as exc:
        logger.debug("Failed to write cloudflared pidfile: %s", exc)


def _load_cloudflared_pidfile() -> dict[str, Any] | None:
    """Read and validate the cloudflared pidfile.

    Returns the parsed dict (with at least an integer ``pid`` key) or
    ``None`` if the file is missing, malformed, or invalid.
    """
    try:
        raw = _cloudflared_pidfile().read_text(encoding="utf-8")
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
        _cloudflared_pidfile().unlink(missing_ok=True)
    except OSError:
        pass


def _looks_like_cloudflared(pid: int) -> bool:
    """Return True iff the process behind *pid* appears to be cloudflared.

    The pidfile records only a bare integer PID, so after an unclean
    kiss-web shutdown the OS may have recycled that PID for an
    UNRELATED process.  Any code about to *signal* the recorded PID
    must therefore confirm the process identity first.  Uses
    ``ps -o comm=`` (portable across macOS and Linux) or, on Windows,
    the image path from :func:`kiss.core.processes.process_identity`,
    and matches the executable basename against ``cloudflared`` — the
    name of the binary both the spawn path and the adoption path launch.

    Returns:
        True when the command basename is exactly ``cloudflared`` (or
        ``cloudflared.exe``); False for any other process, a dead PID,
        or a ``ps`` failure (fail-safe: an unverifiable process is
        never signalled).  The match is exact — a prefix match would
        also kill an unrelated ``cloudflared-helper``-style process
        that inherited a recycled PID.
    """
    if sys.platform == "win32":  # pragma: no cover — Windows only
        # "<creation-time> <image path>"; the time carries no spaces.
        identity = _process_identity(pid) or ""
        comm = identity.split(" ", 1)[1] if " " in identity else ""
    else:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm="],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
            )
        except Exception:
            return False
        comm = result.stdout.strip()
    if not comm:
        return False
    return Path(comm).name.lower() in ("cloudflared", "cloudflared.exe")


def _terminate_declined_cloudflared(pid: int) -> None:
    """Terminate a pidfile-recorded cloudflared we declined to adopt.

    A declined-but-alive cloudflared would otherwise be orphaned
    forever (the caller spawns a fresh one next), leaking a process
    and a metrics port and confusing the next adoption attempt.  Only
    pids recorded in our own pidfile are ever passed here — but the
    pidfile can be STALE: after an unclean shutdown the OS may have
    recycled the recorded PID for an unrelated process, so the
    process identity is verified (:func:`_looks_like_cloudflared`)
    before every signal; a mismatch only unlinks the stale pidfile.
    Sends SIGTERM, waits up to ~2s, re-verifies identity (the PID can
    be recycled inside the wait window too), escalates to SIGKILL if
    still alive, then unlinks the pidfile.
    """
    _terminate_cloudflared_pid(pid)
    _unlink_cloudflared_pidfile()


def _terminate_cloudflared_pid(pid: int) -> None:
    """Terminate the cloudflared process *pid* without touching the pidfile.

    Verifies the process identity (:func:`_looks_like_cloudflared`)
    before EVERY signal because *pid* may come from a stale pidfile or
    a process listing and could have been recycled for an unrelated
    process.  Sends SIGTERM, waits up to ~2s, re-verifies, escalates to
    SIGKILL if still alive.  Used both for pidfile-recorded processes
    (via :func:`_terminate_declined_cloudflared`) and for stray tunnels
    found by :func:`_find_cloudflared_forwarding_to`, where unlinking
    the pidfile would wrongly discard the daemon's OWN tunnel record.
    """
    if not _looks_like_cloudflared(pid):
        logger.info(
            "pid %d is not a cloudflared process (pid recycled); "
            "not signalling it",
            pid,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    for _ in range(20):
        if not _is_pid_alive(pid):
            break
        time.sleep(0.1)
    if _is_pid_alive(pid) and _looks_like_cloudflared(pid):
        try:
            # Windows has no SIGKILL; ``os.kill`` there terminates the
            # process outright for any signal number.
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except (ProcessLookupError, PermissionError, OSError):
            pass


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _find_cloudflared_forwarding_to(local_port: int) -> list[tuple[int, int]]:
    """Find every cloudflared of this user that forwards to *local_port*.

    The pidfile is the daemon's only link to its tunnel, and that link
    breaks in practice: the file is lost, or the daemon's home directory
    changes (a brand switch between ``~/.kiss`` and ``~/.s10s`` keeps a
    separate pidfile per home).  The tunnel itself keeps running — it was
    detached on purpose so its public URL survives restarts — and keeps
    relaying internet visitors to ``--url http://localhost:<port>``.
    This scan rediscovers such tunnels from the process table so the
    caller can adopt one (preserving its URL) or terminate the rest
    (no unmanaged public URL may reach this server).

    Only processes that look like a tunnel THIS daemon could have spawned
    (:meth:`RemoteAccessServer._spawn_cloudflared`) count: owned by the
    current user, executable basename ``cloudflared``, quick-tunnel
    argv (``tunnel ...`` without the named-tunnel ``run`` subcommand),
    ``--metrics 127.0.0.1:<port>`` (the probes connect to 127.0.0.1, so
    a tunnel exposing metrics elsewhere could never be monitored) and a
    ``--url`` whose host is loopback and whose port is *local_port*.  A
    user's own named tunnel or a tunnel to another machine's port is
    never matched, so it is neither adopted nor killed.

    Returns:
        ``(pid, metrics_port)`` pairs, in process-table order; empty on
        Windows (no ``ps``) or when ``ps`` fails.
    """
    if sys.platform == "win32":  # pragma: no cover — Windows only
        return []
    try:
        result = subprocess.run(
            ["ps", "-ww", "-eo", "pid=,uid=,args="],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )
    except Exception:
        return []
    uid = str(os.getuid())
    found: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1] != uid:
            continue
        if Path(parts[2]).name.lower() != "cloudflared":
            continue
        args = parts[3:]
        if args[0] != "tunnel" or "run" in args:
            continue
        try:
            metrics_port = _quick_tunnel_argv_ports(args, local_port)
        except ValueError:
            # Unparsable ``--metrics``/``--url`` value (not an int, bad
            # IPv6 literal, ...): not a tunnel this daemon spawned.
            continue
        if metrics_port is not None:
            found.append((int(parts[0]), metrics_port))
    return found


def _quick_tunnel_argv_ports(args: list[str], local_port: int) -> int | None:
    """Return the ``--metrics`` port of a quick-tunnel argv aimed at *local_port*.

    *args* are the arguments after the ``cloudflared`` executable.
    Returns ``None`` when the metrics endpoint is not on 127.0.0.1 or
    the ``--url`` host is not loopback or its port is not *local_port*.
    Raises ``ValueError`` for values that do not parse at all.
    """
    metrics_port: int | None = None
    url_port: int | None = None
    for i, arg in enumerate(args[:-1]):
        if arg == "--metrics":
            host, _, port = args[i + 1].rpartition(":")
            if host == "127.0.0.1":
                metrics_port = int(port)
        elif arg == "--url":
            target = urlsplit(args[i + 1])
            if target.hostname in _LOOPBACK_HOSTS:
                url_port = target.port
    if metrics_port is None or url_port != local_port:
        return None
    return metrics_port


def _terminate_stray_cloudflared(local_port: int, keep_pid: int | None) -> None:
    """Terminate every cloudflared forwarding to *local_port* except *keep_pid*.

    Called once the daemon's own tunnel is settled (adopted or freshly
    spawned, *keep_pid*) or when no tunnel may exist at all
    (``keep_pid=None``, empty password).  Any other cloudflared aimed at
    this port is a leftover of an earlier daemon that lost its pidfile
    or ran from another home directory: nobody monitors it, yet its
    public URL still reaches this server.
    """
    for pid, metrics_port in _find_cloudflared_forwarding_to(local_port):
        if pid == keep_pid:
            continue
        logger.warning(
            "Terminating stray cloudflared pid=%d metrics_port=%d: it "
            "forwards to local port %d but is not this server's tunnel",
            pid, metrics_port, local_port,
        )
        _terminate_cloudflared_pid(pid)


def _terminate_orphan_cloudflared(local_port: int | None = None) -> None:
    """Kill a previous kiss-web's surviving cloudflared, if any.

    Called at startup when the ``remote_password`` is EMPTY: a tunnel
    deliberately left alive by the previous instance (so its public
    URL survives restarts) would keep forwarding internet traffic to
    this server over loopback, where the empty password authenticates.
    With no password configured there must be no public tunnel at all,
    so instead of adopting the orphan it is terminated.

    The signalling (identity check before EVERY signal, SIGTERM,
    bounded wait, re-verified SIGKILL escalation, pidfile unlink) is
    delegated to :func:`_terminate_declined_cloudflared`, which exists
    for exactly this "recorded pid we must not adopt" situation.  When
    *local_port* is given, cloudflared processes forwarding to it that
    the pidfile does not know about (lost pidfile, other home directory)
    are terminated as well — they expose this server just the same.
    """
    data = _load_cloudflared_pidfile()
    if data is not None:
        pid = int(data["pid"])
        if _is_pid_alive(pid) and _looks_like_cloudflared(pid):
            logger.warning(
                "no remote password is set; terminating the cloudflared "
                "tunnel (pid=%d) left by a previous kiss-web so its "
                "public URL stops reaching this server.", pid,
            )
        _terminate_declined_cloudflared(pid)
    if local_port is not None:
        _terminate_stray_cloudflared(local_port, None)


def _try_adopt_existing_cloudflared(
    local_port: int | None = None,
) -> tuple[int, int, str] | None:
    """Look for a healthy cloudflared started by a previous kiss-web.

    Reads ``~/.kiss/cloudflared.pid``, verifies the pid is alive and
    the metrics ``/ready`` endpoint reports ``readyConnections > 0``,
    and re-discovers the public URL via the ``/quicktunnel`` endpoint
    (falling back to the URL recorded in the pidfile when the metrics
    endpoint doesn't expose one — e.g. named tunnels).

    When the pidfile yields nothing (missing, malformed, dead pid, or a
    live process that had to be declined) and *local_port* is given,
    the process table is scanned for a cloudflared that forwards to
    that port anyway (:func:`_find_cloudflared_forwarding_to`) — the
    tunnel of a daemon that lost its pidfile or ran from another home
    directory — and the first healthy one is adopted the same way.
    Without this the public URL rotated on every such restart while the
    old tunnel lived on unmanaged.

    A live cloudflared whose reachable metrics endpoint reports zero
    ready connections (mid-reconnect after a network switch or wake
    from sleep) is still adopted — *tentatively* — as long as its
    public URL is known: the watchdog's startup grace and
    unhealthy-tick budget then decide between recovery and
    replacement, exactly as they would have for the previous owner.
    Declining (and killing) such a tunnel here would rotate a
    recoverable public URL on every kiss-web restart that happens to
    land mid-reconnect.

    This is how the daemon preserves a single quick-tunnel URL across
    its own restarts: ``cloudflared`` is spawned in its own process
    group (``start_new_session=True``) so it survives ``kiss-web``'s
    SIGTERM and the installer's ``pkill kiss-web``, and — when
    ``kiss-web`` runs as a systemd service — in its own transient
    scope unit (see :func:`_cloudflared_launch_prefix`) so it also
    survives ``systemctl restart kiss-web``'s cgroup-wide kill.  The
    next ``kiss-web`` startup then adopts it here instead of spawning
    a fresh quick-tunnel with a new hostname.

    Returns:
        ``(pid, metrics_port, url)`` if adoption succeeded, else
        ``None`` (caller spawns a fresh cloudflared).
    """
    recorded_pid: int | None = None
    data = _load_cloudflared_pidfile()
    if data is not None:
        pid = int(data["pid"])
        metrics_port = data.get("metrics_port")
        if isinstance(metrics_port, int):
            recorded_pid = pid
            if _is_pid_alive(pid):
                adopted = _adopt_cloudflared_candidate(
                    pid, metrics_port, data.get("url"),
                )
                if adopted is not None:
                    return adopted
            else:
                logger.info(
                    "cloudflared pidfile points to dead pid %d; ignoring",
                    pid,
                )
    if local_port is None:
        return None
    for pid, metrics_port in _find_cloudflared_forwarding_to(local_port):
        if pid == recorded_pid:
            continue
        logger.info(
            "Found cloudflared pid=%d metrics_port=%d forwarding to local "
            "port %d that the pidfile does not record; trying to adopt it",
            pid, metrics_port, local_port,
        )
        adopted = _adopt_cloudflared_candidate(pid, metrics_port, None)
        if adopted is not None:
            return adopted
    return None


def _adopt_cloudflared_candidate(
    pid: int, metrics_port: int, saved_url: object,
) -> tuple[int, int, str] | None:
    """Probe one live cloudflared and adopt it or terminate it.

    Shared by the pidfile path and the process-table fallback of
    :func:`_try_adopt_existing_cloudflared`; see there for the
    adoption rules.  *saved_url* is the URL recorded in the pidfile
    (``None`` for a process found in the process table), used when the
    metrics endpoint exposes no ``/quicktunnel`` hostname.

    Returns:
        ``(pid, metrics_port, url)`` on adoption; ``None`` after the
        process was declined and terminated.
    """
    ready = _probe_tunnel_ready(metrics_port)
    if ready is not True:
        # Re-probe before declining: ``None`` (endpoint unreachable —
        # e.g. metrics socket still binding after wake) and ``False``
        # (HTTP 503 — "zero ready connections *right now*", which a
        # tunnel mid-reconnect reports briefly) are both potentially
        # transient.  Terminating on the first such reading would
        # needlessly rotate a recoverable quick-tunnel URL.
        for _ in range(4):
            time.sleep(0.5)
            ready = _probe_tunnel_ready(metrics_port)
            if ready is True:
                break
    url = _query_quicktunnel_hostname(metrics_port)
    if url is None and isinstance(saved_url, str) and saved_url.startswith("https://"):
        url = saved_url
    if ready is not True:
        if ready is False and url is not None and _looks_like_cloudflared(pid):
            # The metrics endpoint is REACHABLE but reports zero ready
            # edge connections — the canonical mid-reconnect signature
            # (network switch, wake from sleep).  The watchdog tolerates
            # exactly this state for ``_TUNNEL_STARTUP_GRACE`` plus a
            # full unhealthy-tick budget (minutes) before rotating the
            # URL, so the adoption path must not be stricter: killing a
            # recovering cloudflared here is what rotated the public URL
            # on every install-triggered kiss-web restart that landed
            # mid-reconnect.  Adopt it tentatively — the caller marks it
            # freshly started, and the watchdog's existing grace/tick
            # machinery rotates it only if it never recovers.
            logger.info(
                "cloudflared pid=%d metrics_port=%d reports zero ready "
                "connections (likely mid-reconnect); adopting "
                "tentatively with url=%s — the watchdog will replace it "
                "only if it never recovers",
                pid, metrics_port, url,
            )
            return pid, metrics_port, url
        if ready is None:
            reason = f"metrics port {metrics_port} is unreachable"
        elif url is None:
            reason = (
                f"metrics port {metrics_port} reports no ready "
                "connections and no URL is known"
            )
        else:
            reason = "the process no longer looks like cloudflared"
        logger.info(
            "cloudflared pid %d alive but %s; not adopting", pid, reason,
        )
        _terminate_declined_cloudflared(pid)
        return None
    if url is None:
        _terminate_declined_cloudflared(pid)
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
        ``readyConnections == 0`` (confirmed deregistration).  Real
        ``cloudflared`` sends this as **HTTP 503** with a JSON body
        (``{"status":503,"readyConnections":0,...}``) — ``urlopen``
        raises :class:`urllib.error.HTTPError` for it, so the 503
        reply is parsed from the error object; a 503 whose body
        cannot be parsed still counts as ``False`` because a 503
        from ``/ready`` is by definition "not ready".
        ``None`` if the endpoint is unreachable, replies with a
        non-503 HTTP error, the response is not valid JSON, or the
        value is non-numeric — callers should treat this as "no
        information" and *not* count it toward an unhealthy streak.
    """
    not_ready_status = 503
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{metrics_port}/ready",
            headers={"User-Agent": "kiss-web"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # cloudflared's /ready replies 503 (with a JSON body) while
        # the tunnel has zero ready edge connections — the canonical
        # "deregistered, public hostname is NXDOMAIN" signal.
        if exc.code != not_ready_status:
            return None
        try:
            data = json.loads(exc.read())
            return int(data.get("readyConnections", 0)) > 0
        except Exception:
            return False
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

    **Critically**, the loop never returns while the subprocess is
    alive — neither after finding the URL nor after the caller's URL
    wait timed out.  It is the ONLY reader of the pipe for the whole
    life of the ``cloudflared`` process (the callers keep the process
    when the URL is discovered later through the metrics endpoint, or
    unconditionally for a named tunnel).  If the pipe buffer were to
    fill (~64 KiB), ``cloudflared`` would block on its next stderr
    write, which in Go deadlocks the whole process (the logging mutex
    prevents any goroutine from making progress); its metrics endpoint
    then stops answering too, so the watchdog can no longer even tell
    that the tunnel is unhealthy.  A stop-on-timeout early exit used
    to exist here and produced exactly that hang.

    Args:
        stderr: A line-buffered text-mode file-like object.
        parse: Callback invoked on each line; returns a URL string when
            recognised, otherwise ``None``.
        result: Single-element list used to communicate the URL back
            to the caller across the thread boundary.
        rate_limit_flag: Optional single-element list set to ``True``
            on the first stderr line matching
            :func:`_is_rate_limit_line`.  Lets callers distinguish a
            rate-limited tunnel start (HTTP 429 / Cloudflare error
            1015) from a generic failure so the watchdog can apply a
            much longer backoff.
        url_found_event: Optional event set when a URL is first
            discovered and again at EOF.  Signals
            :func:`_read_url_from_stderr` to return the URL
            immediately while this thread keeps draining, or to
            report the failure as soon as the process has exited
            instead of sitting out the whole URL timeout.
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
    # EOF: the process closed its stderr, i.e. it exited.  Every line
    # has been consumed, so ``result`` and ``rate_limit_flag`` are final.
    if url_found_event is not None:
        url_found_event.set()


def _read_url_from_stderr(
    proc: subprocess.Popen[str],
    parse: Callable[[str], str | None],
    timeout: float = 30.0,
    rate_limit_flag: list[bool] | None = None,
) -> str | None:
    """Read *proc*'s stderr until *parse* finds a URL or *timeout* elapses.

    The reader runs in a daemon thread so this call is bounded even
    when ``cloudflared`` keeps streaming non-matching log lines after
    startup.  The thread itself is NOT bounded by *timeout*: it keeps
    draining the pipe until EOF, i.e. until *proc* exits (see
    :func:`_stderr_reader_loop` for why a live ``cloudflared`` must
    never be left without a stderr reader).  One drain thread per live
    tunnel process is therefore expected; it ends with the process.

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
    url_found_event = threading.Event()
    reader = threading.Thread(
        target=_stderr_reader_loop,
        args=(stderr, parse, result, rate_limit_flag, url_found_event),
        name=f"cloudflared-stderr-drain-{proc.pid}",
        daemon=True,
    )
    try:
        reader.start()
    except Exception:
        # No drain thread (``RuntimeError: can't start new thread``
        # under thread exhaustion): a live *proc* would block on its
        # first full stderr pipe — for cloudflared a whole-process
        # deadlock.  Kill and reap it before reporting the failure so
        # the caller never keeps a reader-less process.
        proc.kill()
        proc.wait()
        stderr.close()
        raise
    url_found_event.wait(timeout=timeout)
    return result[0]


def _parse_quick_tunnel_url(line: str) -> str | None:
    """Return the ``*.trycloudflare.com`` URL from a quick-tunnel log line.

    Skips ``api.trycloudflare.com`` (Cloudflare's API endpoint, which
    cloudflared logs before the real tunnel URL).
    """
    match = re.search(
        r"(https://(?!api\.)[^\s]+\.trycloudflare\.com)", line,
    )
    return match.group(1) if match else None


_NON_TUNNEL_LOG_HOSTS = frozenset({
    "developers.cloudflare.com",
    "github.com",
    "www.cloudflare.com",
    "cloudflare.com",
})


def _parse_named_tunnel_url(line: str, configured_url: str | None) -> str | None:
    """Return the public URL of a named tunnel from a log *line*.

    Returns any non-local ``https?://…`` hostname directly, except
    known documentation/banner hosts (:data:`_NON_TUNNEL_LOG_HOSTS`)
    that cloudflared prints in update notices and doc links — those
    must never be published as the tunnel URL.  When a
    "Registered tunnel connection"/"Connection registered" line
    appears, returns *configured_url* (or a sentinel string when no
    URL was pre-configured).  Returns ``None`` on lines that do not
    match either pattern.
    """
    match = re.search(r"https?://([^\s/]+)", line)
    if match:
        host = match.group(1)
        if (
            "localhost" not in host
            and "127.0.0.1" not in host
            and host not in _NON_TUNNEL_LOG_HOSTS
        ):
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
    boot-time race where ``kiss-web`` is (re)started *before* the
    installer or the settings page has written the password to disk:
    instead of refusing to start the tunnel and exiting (which causes
    ``launchd`` to respawn the daemon and mint a brand-new
    ``*.trycloudflare.com`` URL), the daemon waits patiently for the
    password to arrive.

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
    loopback_url: str | None = None, lan_urls: list[str] | None = None,
    local_ca: bool = False,
) -> None:
    """Write the active server URLs to ``url_file``.

    Creates the parent directory if needed.  The default file location
    is ``~/.kiss/remote-url.json``, which is read by ``kiss-web --url``
    so users can discover the remote URL without digging through log
    files.  Tests inject a temporary path to avoid touching the live
    file the ``kiss-web`` daemon maintains.

    Args:
        url_file: Path to the JSON file to write.
        local_url: The local ``http://localhost:PORT`` URL.
        tunnel_url: The Cloudflare tunnel URL, or None.
        loopback_url: The ``http://127.0.0.1:PORT`` URL, or None.
        lan_urls: ``https://<lan-ip>:PORT`` URLs for the host's
            routable LAN addresses, or None.
        local_ca: True when the daemon serves the auto-generated,
            locally-signed certificate, so the ``/ca.crt`` download and
            ``kiss-web --trust-ca`` apply (the chat page shows the
            trust hint only then).
    """
    data: dict[str, object] = {"local": local_url}
    if tunnel_url:
        data["tunnel"] = tunnel_url
    if loopback_url:
        data["loopback"] = loopback_url
    if lan_urls:
        data["lan"] = list(lan_urls)
    if local_ca:
        data["localCa"] = True
    _atomic_write_text(url_file, json.dumps(data, indent=2) + "\n")


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
        data = json.loads(url_file.read_text(encoding="utf-8"))
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
    but is not guessable by outsiders.  When ``KISS_HOME`` is not the
    default ``~/.kiss``, the home path is mixed into the hash as well:
    processes running against an isolated home (tests, secondary smoke
    servers) can then never compute — and thus never pollute — the
    production daemon's discovery topic, while every existing
    default-home install keeps its topic byte-identical.

    The stored topic is read directly (an ``OSError`` — e.g. the file
    vanishing between an existence check and the read — falls through
    to recomputation instead of propagating) and persisted atomically
    via a pid-unique temp file + ``Path.replace`` so a concurrent
    reader in a sibling process can never observe a torn write.
    Persistence failures are non-fatal: the topic is deterministic,
    so the freshly computed value is returned regardless.

    Returns:
        A hex string suitable for use as an ntfy.sh topic name.
    """
    kiss_home_path = kiss_home()
    topic_file = kiss_home_path / "ntfy_topic"
    try:
        stored = topic_file.read_text(encoding="utf-8").strip()
    except OSError:
        stored = ""
    if stored:
        return stored
    # Only the literal stock ``~/.kiss`` keeps the historical unsalted topic
    # (phones subscribed before this code existed); any other home — a
    # custom KISS_HOME or a white-label brand's directory — is salted with
    # its path, so two installs on one machine never share a topic.
    default_home = Path.home() / ".kiss"
    if kiss_home_path.expanduser().resolve() == default_home.resolve():
        identity = f"{platform.node()}:{uuid.getnode()}"
    else:
        identity = f"{platform.node()}:{uuid.getnode()}:{kiss_home_path}"
    topic = "kiss-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
    try:
        _atomic_write_text(topic_file, topic + "\n")
    except OSError:
        logger.debug("Failed to persist ntfy topic", exc_info=True)
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

# A same-URL ntfy message younger than this many seconds suppresses a
# repost (anti-spam for watchdog restarts and named-tunnel
# re-registrations).  An *older* same-URL message is reposted anyway so
# a daemon restart bumps the URL back to the top of the subscriber's
# feed and restarts ntfy.sh's 12h message-cache clock.
_NTFY_REPOST_MAX_AGE = 3600.0


def _fetch_last_ntfy_message(
    topic: str, base_url: str = _NTFY_BASE_URL,
) -> tuple[str, float] | None:
    """Return the most recent message posted to ``{base_url}/{topic}``.

    Queries ntfy.sh's poll endpoint (``/{topic}/json?poll=1``) which
    returns cached messages (default retention 12h) as newline-
    delimited JSON.  Only entries with ``event == "message"`` are
    considered; the last one wins because the server returns events
    in chronological order.

    Args:
        topic: ntfy.sh topic name (without leading slash).
        base_url: Override the ntfy server URL (used by tests).

    Returns:
        A ``(message, time)`` tuple for the most recent cached
        message, where ``time`` is the message's publish time in epoch
        seconds (``0.0`` when the server omits or mangles the field),
        or ``None`` if the topic has no cached messages or the request
        fails.
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
    last: tuple[str, float] | None = None
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
            raw_time = obj.get("time")
            posted_at = (
                float(raw_time)
                if isinstance(raw_time, (int, float)) else 0.0
            )
            last = (msg, posted_at)
    return last


def _ntfy_message(url: str) -> str:
    """Return the ntfy.sh message body for *url*: ``<url> (<machine name>)``.

    The machine name is ``platform.node()``, the same host name the
    remote page title and the task settings report, so a user who
    subscribes to several KISS machines can tell the posts apart.
    The bare URL is returned when the host name is unknown.
    """
    node = platform.node().strip()
    return f"{url} ({node})" if node else url


def _post_url_to_message_board(
    url: str, base_url: str = _NTFY_BASE_URL,
) -> None:
    """Post the active Cloudflare URL to ntfy.sh as a private message.

    Uses the machine-stable topic from :func:`_get_machine_topic` so
    the URL can be retrieved by subscribing to the same topic.  The
    message body is :func:`_ntfy_message` (the URL followed by the
    machine name) and the title indicates it is a KISS Sorcar remote
    URL update.  Before posting, the most recent cached message on
    the topic is fetched via :func:`_fetch_last_ntfy_message`; if it
    already matches the new body *and* is younger than
    :data:`_NTFY_REPOST_MAX_AGE`, the post is skipped so subscribers
    are not woken up by duplicate notifications when a watchdog
    restart or named-tunnel re-registration produces the same public
    hostname.  An older same-URL message no longer suppresses the
    post: reposting bumps the URL back to the top of the subscriber's
    ntfy feed after a daemon restart and restarts ntfy.sh's 12h
    message-cache clock.  Failures are logged but never raised.

    Args:
        url: The ``https://`` URL to publish.
        base_url: Override the ntfy server URL (used by tests).
    """
    if not url or url.startswith(("https://localhost", "http://localhost")):
        return
    try:
        topic = _get_machine_topic()
        message = _ntfy_message(url)
        last = _fetch_last_ntfy_message(topic, base_url=base_url)
        if last is not None and last[0].strip() == message:
            age = time.time() - last[1]
            if age < _NTFY_REPOST_MAX_AGE:
                logger.info(
                    "Skipping ntfy.sh post for %s; last message on "
                    "topic %s already has the same URL "
                    "(posted %.0fs ago)", url, topic, age,
                )
                return
            logger.info(
                "Reposting %s to ntfy.sh topic %s; last same-URL "
                "message is stale (posted %.0fs ago)", url, topic, age,
            )
        req = urllib.request.Request(
            f"{base_url}/{topic}",
            data=message.encode("utf-8"),
            method="POST",
            headers={
                "Title": f"{PRODUCT_NAME} Remote URL",
                "Tags": "link,kiss-sorcar",
                "Click": url,
                "User-Agent": "kiss-web",
            },
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        logger.info("Posted remote URL to ntfy.sh/%s", topic)
    except Exception:
        logger.debug("Failed to post URL to ntfy.sh", exc_info=True)


def _describe_port_listeners(port: int) -> str:
    """Name the other processes listening on TCP *port*, via ``lsof``.

    Returns e.g. ``"Code Helper (Plugin)[90174]"`` (several are
    comma-separated); the calling process is left out.  Empty when
    ``lsof`` is unavailable or reports nothing.
    """
    try:
        proc = subprocess.run(
            ["lsof", "-nP", "+c", "0", "-Fpc", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    names: list[str] = []
    pid = ""
    for line in proc.stdout.splitlines():
        if line.startswith("p"):
            pid = line[1:]
        elif line.startswith("c") and pid != str(os.getpid()):
            names.append(f"{line[1:]}[{pid}]")
    return ", ".join(names)


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
    return frozenset(
        addr for addr in ips
        if not addr.startswith(("127.", "169.254.", "::ffff:"))
    )


def _trust_local_ca() -> None:
    """``kiss-web --trust-ca``: trust the local CA in this user's browsers.

    Creates the CA first when no daemon has run yet, then installs
    ``~/.kiss/tls/ca.pem`` into every trust store found
    (:func:`kiss.server.tls_trust.trust_local_ca`) and prints one line
    per store plus the phone instructions.
    """
    from kiss.server.tls_trust import trust_local_ca

    _refresh_local_tls_pair(_get_local_ips())
    for line in trust_local_ca(_tls_dir() / tls_certs.CA_CERT_FILE):
        print(line)


def _print_url() -> None:
    """Print the active remote URL from ``~/.kiss/remote-url.json``.

    Prints the tunnel URL if available, otherwise the local URL.
    Exits with code 1 if the server is not running or the file is
    missing.
    """
    url = _read_url_from_file(_url_file_path())
    if url:
        print(url)
    else:
        print(f"{PRODUCT_NAME} web server is not running.", file=sys.stderr)
        sys.exit(1)


def _snapshot_active_tabs() -> list[str]:
    """Return ``"<tabId>(task=<task_id>)"`` strings for active tasks.

    Snapshots the agent-state registry under its lock before iterating
    so a concurrent worker thread mutating it cannot race the iterator.
    The lock is a :class:`threading.RLock`, so re-entry from the same
    thread is safe even when called from a signal handler that
    interrupted a lock holder.  Falls back to a best-effort unlocked
    snapshot if the lock itself is unusable (e.g. during interpreter
    shutdown), and skips malformed entries rather than propagating, so
    callers — the shutdown-signal logger and the ``activeTasksQuery``
    handler — always get a usable (possibly partial) report.

    Liveness is :meth:`AgentState.busy`, not ``is_task_active`` alone:
    a worker that ``_cmd_run`` has started but that has not yet raised
    the flag owns a real task, and answering ``count: 0`` for it lets
    the installer (``scripts/check-kiss-web-active-tasks.py``) SIGTERM
    the daemon on top of a just-launched run (F08-2).
    """
    from kiss.server import agent_state

    try:
        states = agent_state.snapshot()
    except Exception:
        try:
            states = list(agent_state.agent_states.values())
        except Exception:
            logger.debug(
                "unlocked registry snapshot failed", exc_info=True,
            )
            states = []
    active_tabs: list[str] = []
    for state in states:
        try:
            if state.busy():
                active_tabs.append(f"{state.tab_id}(task={state.task_id})")
        except Exception:
            logger.debug(
                "skipping malformed entry in active-task snapshot",
                exc_info=True,
            )
    return active_tabs


def _shutdown_state_owns_thread(state: Any, thread: threading.Thread) -> bool:
    """True while the swept *state* still owns its unstarted *thread*.

    Ownership guard for the graceful-shutdown sweep's pre-start wait
    (:func:`kiss.server.task_runner.wait_for_thread_start`), evaluated
    under ``agent_state.STATE_LOCK``.  Unlike the Stop watchdog's
    :func:`~kiss.server.task_runner._state_owns_thread` it must NOT
    treat an acknowledged stop as lost ownership: ``_cmd_run``'s
    pre-cancel path acknowledges the stop first and only then clears
    ``state.task_thread``, and the sweep may only stop waiting once
    the thread can no longer be started (audit0903 F1).

    Args:
        state: The :class:`~kiss.server.agent_state.AgentState`
            selected by the sweep.
        thread: The worker thread captured with it.

    Returns:
        ``True`` while ``state.task_thread`` is still *thread*.
    """
    return state.task_thread is thread


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


def _raise_open_file_limit() -> None:
    """Raise the soft ``RLIMIT_NOFILE`` toward the hard limit.

    macOS defaults the soft limit to 256 open files, which a
    long-running kiss-web daemon (websocket connections, agent
    subprocesses, log and trajectory files) exhausts under load,
    surfacing as ``OSError: [Errno 24] Too many open files`` when e.g.
    saving a trajectory YAML.  Raising the soft limit up to the hard
    limit needs no privileges.  macOS rejects soft values above
    ``kern.maxfilesperproc`` (and ``RLIM_INFINITY``) with
    ``ValueError``, so descending candidates are tried until one
    sticks.  Child agent subprocesses inherit the raised limit.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover — non-POSIX platform
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):  # pragma: no cover
        return
    for target in (1048576, 262144, 65536, 10240, 4096, 1024):
        if target <= soft:
            break
        if hard != resource.RLIM_INFINITY and target > hard:
            continue
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.info(
                "Raised RLIMIT_NOFILE soft limit from %d to %d", soft, target
            )
            break
        except (ValueError, OSError):
            continue


def _generate_self_signed_cert(
    cert_path: Path,
    key_path: Path,
) -> None:
    """Generate a locally-signed TLS cert/key pair at *cert_path*/*key_path*.

    Creates (or reuses) the machine-local CA beside the certificate
    (``ca.pem`` / ``ca-key.pem`` in ``cert_path.parent``) and issues a
    server certificate signed by it covering ``localhost``, the
    hostname, ``127.0.0.1`` and ``::1`` (see
    :mod:`kiss.core.tls_certs`).  Kept under its historical name for
    the test fixtures that build throwaway servers; the daemon itself
    goes through :func:`_create_ssl_context`, which also adds the LAN IPs.

    Args:
        cert_path: Where to write the PEM-encoded certificate.
        key_path: Where to write the PEM-encoded private key.
    """
    ca_cert_path = cert_path.parent / tls_certs.CA_CERT_FILE
    ca_key_path = cert_path.parent / tls_certs.CA_KEY_FILE
    if not tls_certs._ca_pair_is_usable(ca_cert_path, ca_key_path):
        tls_certs.generate_local_ca(ca_cert_path, ca_key_path)
    tls_certs.issue_server_cert(cert_path, key_path, ca_cert_path, ca_key_path)


def _tls_lock_path() -> Path:
    """Return the lock file serialising sibling daemons' access to :func:`_tls_dir`.

    Held through :func:`kiss.core.file_lock.exclusive_file_lock` with
    :data:`_TLS_LOCK_TIMEOUT_S`: the bounded wait polls the non-blocking
    lock, since a blocking ``flock`` offloaded to the executor cannot be
    interrupted by cancelling its coroutine, and a sibling that wedged
    while holding the lock would otherwise stall startup forever.
    """
    return _tls_dir() / tls_certs.TLS_LOCK_FILE


def _load_local_tls_pair(ctx: ssl.SSLContext, lan_ips: Iterable[str]) -> bytes:
    """Under the TLS lock, (re)issue the auto-generated pair for *lan_ips* and load it into *ctx*.

    Serialises sibling daemons with an exclusive file lock held from the
    check through ``load_cert_chain``: the check-then-generate sequence
    and the pair publication are not atomic, so two concurrent processes
    could otherwise publish (or load) a mismatched cert/key pair (F4-10).
    The lock is bounded like every other sidecar lock in this module: a
    blocking ``LOCK_EX`` behind a wedged sibling would stall startup
    forever, and cancelling the ``to_thread`` caller cannot interrupt
    the executor syscall.

    Self-heals a pair that OpenSSL rejects (a daemon that died between
    writing the key and the certificate leaves a mismatched pair every
    future load would refuse, F4-10 residual): the server certificate
    is re-issued under the same lock and loaded again.

    Args:
        ctx: The server context to load; must not be serving yet (see
            :meth:`RemoteAccessServer._refresh_tls_cert` for a live one).
        lan_ips: The host's current LAN IP addresses; each must be in
            the certificate's SAN or the ``https://<lan-ip>:PORT`` URL
            fails hostname verification even on a browser that trusts
            the local CA.

    Returns:
        The PEM bytes of the certificate that was loaded.
    """
    tls_dir = _tls_dir()
    with exclusive_file_lock(_tls_lock_path(), timeout=_TLS_LOCK_TIMEOUT_S):
        cert_path, key_path = tls_certs.ensure_local_tls_pair(tls_dir, lan_ips)
        try:
            ctx.load_cert_chain(str(cert_path), str(key_path))
        except ssl.SSLError:
            logger.warning(
                "Auto-generated TLS cert/key pair in %s is mismatched or "
                "corrupt; re-issuing", tls_dir,
            )
            key_path.unlink(missing_ok=True)
            cert_path, key_path = tls_certs.ensure_local_tls_pair(tls_dir, lan_ips)
            ctx.load_cert_chain(str(cert_path), str(key_path))
        return cert_path.read_bytes()


def _refresh_local_tls_pair(lan_ips: Iterable[str]) -> bytes:
    """Under the TLS lock, (re)issue the auto-generated pair for *lan_ips* and return its cert PEM.

    Executor half of :meth:`RemoteAccessServer._refresh_tls_cert`: the
    generation (key, signing, file writes) runs off the event loop, and the
    returned bytes tell the caller whether the certificate on disk
    differs from the one its live context is serving.

    Args:
        lan_ips: See :func:`_load_local_tls_pair`.
    """
    with exclusive_file_lock(_tls_lock_path(), timeout=_TLS_LOCK_TIMEOUT_S):
        cert_path, _key_path = tls_certs.ensure_local_tls_pair(_tls_dir(), lan_ips)
        return cert_path.read_bytes()


def _reload_local_tls_pair_if_unlocked(ctx: ssl.SSLContext, cert_pem: bytes) -> bool:
    """Load the on-disk pair into the live *ctx* if it is still *cert_pem* and the lock is free.

    Event-loop half of :meth:`RemoteAccessServer._refresh_tls_cert`.
    ``load_cert_chain`` must run on the event-loop thread, where every
    ``SSL_new`` for accepted connections also runs, so the context is
    never mutated concurrently with a handshake setup.  The lock is
    therefore only tried, never waited for: a busy sibling or a pair a
    sibling has meanwhile replaced makes this a no-op and the next
    watchdog tick retries.

    Args:
        ctx: The live server context.
        cert_pem: The certificate bytes the caller observed under the
            lock in :func:`_refresh_local_tls_pair`.

    Returns:
        True when the pair was loaded into *ctx*.
    """
    tls_dir = _tls_dir()
    cert_path = tls_dir / tls_certs.SERVER_CERT_FILE
    key_path = tls_dir / tls_certs.SERVER_KEY_FILE
    with exclusive_file_lock(_tls_lock_path(), blocking=False) as locked:
        if not locked or cert_path.read_bytes() != cert_pem:
            return False
        ctx.load_cert_chain(str(cert_path), str(key_path))
        return True


def _create_ssl_context(
    certfile: str | None = None,
    keyfile: str | None = None,
    lan_ips: Iterable[str] | None = None,
) -> ssl.SSLContext:
    """Create an SSL context for the HTTPS/WSS server.

    If *certfile* and *keyfile* are provided, loads them directly.
    Otherwise uses the machine-local CA in ``~/.kiss/tls/`` to issue a
    server certificate covering ``localhost``, the hostname, the
    loopback addresses and *lan_ips*, re-issuing it when it is missing,
    expiring, signed by a different CA or lacking one of the IPs.

    Args:
        certfile: Path to PEM certificate file, or None for auto-gen.
        keyfile: Path to PEM private key file, or None for auto-gen.
        lan_ips: LAN IP addresses the auto-generated certificate must
            cover; probed with :func:`_get_local_ips` when ``None``.

    Returns:
        A configured ``ssl.SSLContext`` ready for ``websockets.serve()``.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    # No TLS 1.3 session tickets: the synchronous ``websockets`` client
    # (daemon_client, cron, run_agent dispatch) reads post-handshake
    # NewSessionTicket records on its receive thread while its main thread
    # writes the HTTP upgrade on the same SSL object, which intermittently
    # loses the upgrade request and stalls the connection.  Local clients
    # never resume sessions, so the tickets buy nothing.
    ctx.num_tickets = 0
    if certfile and keyfile:
        ctx.load_cert_chain(certfile, keyfile)
        return ctx
    _load_local_tls_pair(ctx, _get_local_ips() if lan_ips is None else lan_ips)
    return ctx


def _trusted_clients_dir() -> Path:
    """Return the directory of enrolled machine-link CAs (``<tls dir>/trusted-clients``)."""
    return _tls_dir() / tls_certs.TRUSTED_CLIENTS_DIR


def _load_trusted_client_cas(ctx: ssl.SSLContext, loaded: set[bytes]) -> list[str]:
    """Load the enrolled client CAs not yet in *loaded* into *ctx*; return their names.

    Every ``*.pem`` of :func:`_trusted_clients_dir` (created when
    missing: its presence tells an enrolling machine that this daemon
    understands enrollment) is added to the context's trust store and
    recorded in *loaded* (by content, so a file re-read on every call
    is loaded once).  Once any CA is loaded the context requests a
    client certificate (``CERT_OPTIONAL``: a browser without one still
    connects, a certificate that chains to no loaded CA fails the
    handshake).  A CA cannot be unloaded from a live context, so
    revocation is :func:`tls_certs.enrolled_client_ca`'s re-check
    against the files on disk at every handshake.  A file that is not
    a certificate is logged once and skipped.
    """
    trusted_dir = _trusted_clients_dir()
    trusted_dir.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    for name, pem in tls_certs.trusted_client_cas(trusted_dir):
        if pem in loaded:
            continue
        loaded.add(pem)
        try:
            ctx.load_verify_locations(cadata=pem.decode("ascii"))
        except (ssl.SSLError, ValueError, UnicodeDecodeError):
            logger.warning("Ignoring %s/%s.pem: not a PEM certificate", trusted_dir, name)
            continue
        added.append(name)
        ctx.verify_mode = ssl.CERT_OPTIONAL
    return added


def _local_ca_cert_bytes() -> bytes | None:
    """Return the PEM bytes of the auto-generated CA, or None when absent."""
    ca_path = _tls_dir() / tls_certs.CA_CERT_FILE
    try:
        return ca_path.read_bytes()
    except OSError:
        return None


class FifoSendLock:
    """Per-endpoint FIFO lock whose waiters cost O(1) each.

    Every outbound payload is its own event-loop task queued on the
    endpoint's send lock (:meth:`WebPrinter._locked_send`).  With many
    concurrent streaming tasks that queue grows to thousands of
    waiters, and :class:`asyncio.Lock` removes each woken waiter from
    its deque with ``deque.remove`` (O(n)), so a backlog of *n* sends
    costs O(n^2) loop time and starves the event loop (observed with
    60 daemon tasks: clients dropped by the drain timeout, results
    never delivered).  This lock hands ownership to the leftmost live
    waiter on :meth:`release` and never scans the queue.
    """

    def __init__(self) -> None:
        self._locked = False
        self._waiters: collections.deque[asyncio.Future[None]] = collections.deque()

    def locked(self) -> bool:
        """Return True while some task holds the lock."""
        return self._locked

    async def acquire(self) -> bool:
        """Wait in FIFO order until the lock is owned by the caller."""
        if not self._locked:
            self._locked = True
            return True
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # release() already handed the lock to us; pass it on.
                self.release()
            raise
        return True

    def release(self) -> None:
        """Hand the lock to the next live waiter, or unlock."""
        while self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(None)  # ownership transfers; stays locked
                return
        self._locked = False

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc: object) -> None:
        self.release()


class WebPrinter(JsonPrinter):
    """Printer that broadcasts JSON events to connected WebSocket clients.

    Thread-safe: ``broadcast()`` is called from agent task-runner threads
    and the asyncio event loop.  A lock protects the client set, and
    ``asyncio.run_coroutine_threadsafe`` is used to schedule sends on
    the event loop from non-async threads.
    """

    def __init__(self) -> None:
        super().__init__()
        # Remote peers: password-authenticated browsers on other devices.
        self._remote_clients: set[ServerConnection] = set()
        # Local peers (token-authenticated loopback connections:
        # ``daemon_client`` runs) are kept apart from
        # ``_remote_clients`` because task-event copies reach only
        # the local peers interested in the stamped tab (see
        # ``_local_clients_for_tab``).
        self._local_clients: set[ServerConnection] = set()
        # Daemon-link peers: another daemon's ``MachineLink`` request
        # socket (``"link": true`` in its ``auth`` frame).  Such a peer
        # only ever wants the replies to its own requests, so it is
        # kept out of both sets above: it gets connection-scoped
        # replies and global events, never the tab-stamped task copies
        # (streamed tokens, status, a whole ``task_events`` transcript
        # replayed by someone's history click) that ``broadcast`` fans
        # out to every other peer.
        self._link_clients: set[ServerConnection] = set()
        # Which local connection addressed which tab ids (keyed by
        # connection id); see ``register_local_tab``.
        self._local_tab_sets: dict[str, set[str]] = {}
        self._conn_endpoints: dict[str, ServerConnection] = {}
        self._ws_lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.work_dir: str = ""
        self._pending_sends: dict[Any, set[ConcurrentFuture[None]]] = {}
        self._send_locks: dict[Any, FifoSendLock] = {}
        self._send_timeout: float = _SEND_TIMEOUT
        # tabId -> pending worktree dir of that tab's finished (or
        # running) worktree task; see _track_worktree_event().
        self._tab_worktree_dirs: dict[str, str] = {}

    def _track_worktree_event(
        self, event: dict[str, Any], tab_id: Any, task_id: Any = None,
    ) -> None:
        """Track *tab_id*'s pending worktree directory from *event*.

        A worktree task's committed artifacts live only in its worktree
        until the branch is merged, so ``checkPaths``/``openFile``
        requests from remote clients need the tab's worktree dir as a
        resolution fallback (:meth:`RemoteAccessServer._resolve_tab_file`).
        ``worktree_created`` / ``worktree_done`` record the directory
        (preferring ``worktreeWorkDir`` — the task's cwd inside the
        worktree — over the worktree root, so relative paths from tasks
        launched in a repo subdirectory resolve correctly); a
        successful ``worktree_result`` (merge or discard finished)
        drops it, so the main checkout wins again.

        A ``task_events`` replay envelope is scanned too, in event
        order: session replay first runs :meth:`cleanup_tab` (dropping
        the entry) and, while the task is still running, nothing else
        re-presents the worktree — the historical ``worktree_created``
        nested in the replayed transcript is the only copy of the
        directory, so it must restore the tracking.

        Args:
            event: The event being broadcast.
            tab_id: The tab the event copy is addressed to.
            task_id: For a fan-out copy, the task whose subscriber
                list named *tab_id*; the directory is recorded only if
                the tab is STILL subscribed to it (see
                :meth:`_record_tab_worktree_dir`).  ``None`` for an
                event addressed to the tab directly.
        """
        if not isinstance(tab_id, str) or not tab_id:
            return
        etype = event.get("type")
        if etype == "task_events":
            nested = event.get("events")
            if isinstance(nested, list):
                for sub in nested:
                    if isinstance(sub, dict):
                        self._track_worktree_event(sub, tab_id, task_id)
            return
        if etype in ("worktree_created", "worktree_done"):
            wt_work_dir = event.get("worktreeWorkDir")
            wt_dir = (
                wt_work_dir
                if isinstance(wt_work_dir, str) and wt_work_dir
                else event.get("worktreeDir")
            )
            # A worktree of another repository (no ``worktreeWorkDir``:
            # the task's work dir did not move into it) never replaces
            # the work dir's worktree, which relative file links need.
            keep_existing = not (isinstance(wt_work_dir, str) and wt_work_dir)
            if isinstance(wt_dir, str) and wt_dir:
                self._record_tab_worktree_dir(tab_id, wt_dir, task_id, keep_existing)
        elif (
            etype == "worktree_result"
            and event.get("success")
            and not event.get("kept")
        ):
            # A "Do nothing" result (kept: true) leaves the worktree on
            # disk, so the remote client's openFile/checkPaths fallback
            # must survive for transcript file links to keep resolving
            # into it.  A result naming another repository's worktree
            # (one per repository from commit_and_merge) leaves the
            # recorded one, still pending, alone.
            locked_call(self._lock, self._forget_tab_worktree_dir, tab_id, event.get("worktreeDir"))

    def _forget_tab_worktree_dir(self, tab_id: str, result_dir: Any) -> None:
        """Under ``_lock``: drop *tab_id*'s worktree entry once *result_dir* merged it."""
        recorded = self._tab_worktree_dirs.get(tab_id, "")
        if (
            recorded and isinstance(result_dir, str) and result_dir
            and not Path(recorded).is_relative_to(result_dir)
        ):
            return
        self._tab_worktree_dirs.pop(tab_id, None)

    def _record_tab_worktree_dir(
        self, tab_id: str, wt_dir: str, task_id: Any, keep_existing: bool = False,
    ) -> None:
        """Record *wt_dir* as *tab_id*'s worktree, unless the copy is stale.

        A fan-out copy is addressed from a subscriber snapshot taken
        before this call; a concurrent rebind (``cleanup_tab`` then a
        subscription to another task) can land in between, and
        recording the OLD task's worktree for the rebound tab would
        make its ``openFile``/``checkPaths`` resolve into the wrong
        repository.  The subscription is therefore re-checked and the
        entry written in one ``_lock`` critical section — the same
        lock :meth:`cleanup_tab` unsubscribes and drops the entry
        under — so every interleaving converges on the serial outcome.

        Args:
            tab_id: The tab the event copy is addressed to.
            wt_dir: The worktree directory to record.
            task_id: The task the fan-out copy belongs to, or ``None``
                for an event addressed to the tab directly (always
                recorded).
            keep_existing: Record *wt_dir* only when the tab has no
                worktree directory yet (a second repository's worktree
                must not displace the work dir's).
        """
        locked_call(
            self._lock, self._store_tab_worktree_dir, tab_id, wt_dir, task_id, keep_existing,
        )

    def _store_tab_worktree_dir(
        self, tab_id: str, wt_dir: str, task_id: Any, keep_existing: bool,
    ) -> None:
        """Under ``_lock``: the subscription re-check and write of ``_record_tab_worktree_dir``."""
        if task_id is not None:
            key = self._coerce_task_id(task_id)
            viewers = self._subscribers.get(key) if key else None
            if not viewers or tab_id not in viewers:
                return
        if keep_existing and self._tab_worktree_dirs.get(tab_id):
            return
        self._tab_worktree_dirs[tab_id] = wt_dir

    def worktree_dir_for_tab(self, tab_id: str) -> str:
        """Return the pending worktree dir recorded for *tab_id*.

        Args:
            tab_id: The requesting client's tab id.

        Returns:
            The worktree directory path, or ``""`` when the tab has no
            pending worktree.
        """
        return self._tab_worktree_dirs.get(tab_id, "")

    def cleanup_tab(self, tab_id: str, keep_task_id: Any = None) -> None:
        """Drop *tab_id*'s per-tab state, including its worktree dir.

        A tab closed without merging (no successful ``worktree_result``
        ever arrives for it) would otherwise leave its
        ``_tab_worktree_dirs`` entry behind for the daemon lifetime —
        one dead key per closed pending-worktree tab, and a stale
        resolution fallback should a later client reuse the tab id.

        Also runs when a live tab merely re-subscribes (session
        replay, new chat).  That is safe: after rebinding a chat the
        server re-presents a finished pending worktree
        (``_emit_pending_worktree`` broadcasts ``worktree_done``),
        and for a still-running task the replayed ``task_events``
        transcript carries the historical ``worktree_created`` — either
        way :meth:`_track_worktree_event` re-records the entry before
        any file link is checked.

        The subscriptions go first, then the entry is dropped under the
        same ``_lock`` :meth:`_record_tab_worktree_dir` writes under.
        A stale fan-out copy that passed its subscription re-check
        before the unsubscribe has already written its entry by the
        time this pop runs (so the pop removes it); one that re-checks
        afterwards writes nothing.  Either way the entry is gone.

        Args:
            tab_id: The frontend tab identifier to drop.
            keep_task_id: See :meth:`JsonPrinter.cleanup_tab`.
        """
        super().cleanup_tab(tab_id, keep_task_id)
        if tab_id:
            locked_call(self._lock, self._tab_worktree_dirs.pop, tab_id, None)

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send *event* to every connected WebSocket client.

        Two code paths:

        * Events that already carry an explicit ``tabId`` (status,
          askUser, gitStatus, etc.) are treated as
          targeted "system" events: sent verbatim to all connected
          clients (which filter by ``tabId``), but **not** recorded
          or persisted — except the ``TAB_STAMPED_TASK_EVENT_TYPES``
          (``prompt`` echoes, ``ask_answer`` replies, ``result``) that
          ALSO carry a ``taskId``, whose tabId-stripped copy is
          recorded and persisted under that task (see the tabId branch
          below).
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
          events (``models``, ``history``, ``inputHistory``,
          ``files``, ``ghost``, ``configData``,
          unknown-command ``error``): the stamp is stripped and the
          event is sent ONLY to the connection (= browser tab) that
          issued the request, so one tab's activity can never change
          another tab's UI.
        * Events stamped ``recordOnly`` (the drain hook's durable copy
          of a prompt echo that was already rendered live at queueing
          time — see ``SorcarAgent._drain_pending_user_messages``) are
          recorded and persisted under the thread-local task id but
          never sent to clients; with no resolvable task id they are
          dropped.

        Args:
            event: The event dictionary to emit.
        """
        stamp_event_ts(event)
        conn_id = event.pop("connId", "")
        record_only = bool(event.pop("recordOnly", False))
        if event.get("type") == "configData":
            cfg = event.get("config")
            if isinstance(cfg, dict) and not cfg.get("work_dir"):
                cfg["work_dir"] = (
                    self.work_dir
                    or os.environ.get("KISS_WORKDIR", "")
                    or os.getcwd()
                )

        slot = self._take_replay_slot(event)
        if slot is not None:
            # A running task's replay: its sends were reserved under
            # ``delivery_lock`` by ``replay_snapshot``; supply the payload.
            if "tabId" in event:
                self._track_worktree_event(event, event.get("tabId"))
            slot.set_result(json.dumps(event))
            return

        if conn_id:
            # A ``ready``-driven replay is delivered to one connection
            # but still re-presents the tab's worktree (a running
            # worktree task's ``worktree_created`` nested in the
            # transcript), which the daemon tracks for every client.
            if "tabId" in event:
                self._track_worktree_event(event, event.get("tabId"))
            self._send_to_conn(conn_id, json.dumps(event))
            return

        if "tabId" in event:
            # Recording and sending under ``delivery_lock``: a replay
            # snapshot of the task either has this event and precedes
            # it, or lacks it and follows it (see ``delivery_lock``).
            # Encoded first: a transcript payload can be large, and the
            # lock serializes every task's deliveries.
            kept = self._tab_stamped_task_record(event)
            data = "" if record_only else json.dumps(event)
            locked_call(self.delivery_lock, self._deliver_tab_stamped, event, kept, data)
            return

        event = self._inject_task_id(event)

        if not event.get("taskId"):
            if record_only:
                return
            data = json.dumps(event)
            self._send_to_ws_clients(data)
            self._send_to_link_clients(data)
            return

        # A ``talk`` event is never replayed (not a display event), and
        # deciding whether this machine plays it takes the agent-state
        # lock and may start a local player, so it fans out after the lock.
        talk = event.get("type") == "talk"
        # Resolved before ``delivery_lock`` (it takes ``STATE_LOCK`` and
        # the agent's lock); queued under it so the events table keeps
        # the order the clients saw.
        persist_id = self._persistence_task_id(event)
        # Encoded once, before the lock (a tool result can embed
        # images): persisted as is, and spliced per subscribed tab.
        # The tabId branch above returned, so ``event`` has no tabId.
        data = "" if talk else json.dumps(event)
        locked_call(
            self.delivery_lock, self._deliver_task_event, event, data, persist_id,
            record_only or talk,
        )
        if talk and not record_only:
            self._fanout_talk(event)

    def _fanout_talk(self, event: dict[str, Any]) -> None:
        """Fan out a ``talk`` event, playing its clip here for headless clients.

        A browser plays the synthesized clip itself.  A local client that
        addressed the event's tab (a ``sorcar`` CLI or ``daemon_client``
        run on this machine) has no player of its own, so the daemon
        plays the clip once on this machine's speakers
        (:mod:`kiss.server.talk_player`) before every client receives
        the event as usual.

        Args:
            event: The ``talk`` event (no ``tabId`` stamp).
        """
        targets = set(self._fanout_targets(event.get("taskId")))
        if targets and self._local_client_addressed(targets):
            self._play_talk_clip_locally(event)
        self._fanout_stamped(event)

    def _local_client_addressed(self, tab_ids: set[str]) -> bool:
        """Return True when a local client addressed one of *tab_ids*.

        Runs on the task thread of the ``talk`` tool, so the lock goes
        through ``locked_call`` like every other printer lock a Stop
        injection can reach.
        """
        return locked_call(self._ws_lock, self._local_client_addressed_locked, tab_ids)

    def _local_client_addressed_locked(self, tab_ids: set[str]) -> bool:
        """``_local_client_addressed`` under ``_ws_lock``."""
        return any(tabs & tab_ids for tabs in self._local_tab_sets.values())

    @staticmethod
    def _play_talk_clip_locally(event: dict[str, Any]) -> bool:
        """Play a talk event's synthesized clip on this machine's speakers.

        Uses the :class:`~kiss.server.talk_player.TalkPlayer` singleton,
        a real audio-player child process fed from a serialising queue
        with ``talkId`` dedupe, so playback never blocks the event loop.

        Args:
            event: The ``talk`` broadcast event.

        Returns:
            ``True`` when the machine has an audio player and the event
            carries a clip, so playback was enqueued; ``False`` otherwise.
        """
        if not event.get("audioB64"):
            return False
        try:
            from kiss.server import talk_player

            if talk_player.player_command() is None:
                return False
            talk_player.shared_player().play(dict(event))
            return True
        except Exception:
            logger.exception("daemon-side talk clip playback failed")
            return False

    def _deliver_tab_stamped(
        self, event: dict[str, Any], kept: tuple[dict[str, Any], str] | None, data: str,
    ) -> None:
        """Under ``delivery_lock``: record and send a ``tabId``-stamped event."""
        self._track_worktree_event(event, event.get("tabId"))
        if kept is not None:
            self._keep_tab_stamped_task_event(*kept)
        if data:
            self._send_to_ws_clients(data)
            # An explicit empty stamp is a global event sent
            # verbatim (``tabs_state``, ``tasks_updated``), so
            # daemon-link peers get it; a tab's event is a
            # task copy they never want.
            if not event["tabId"]:
                self._send_to_link_clients(data)

    def _deliver_task_event(
        self, event: dict[str, Any], data: str, persist_id: str, record_only: bool,
    ) -> None:
        """Under ``delivery_lock``: record, persist and fan out a task event."""
        self._record_task_event(event)
        if persist_id:
            _queue_chat_event(data, task_id=persist_id)
        if not record_only:
            self._fanout_stamped(event, data)

    def _fanout_stamped(self, event: dict[str, Any], encoded: str = "") -> None:
        """Send one ``tabId``-stamped copy of *event* per subscribed tab.

        The frontend filters incoming events by ``tabId``; an event
        with no subscriber is silently swallowed.  The event is
        serialised ONCE and the per-tab stamp spliced into the JSON
        string — this path runs once per streamed token, so avoiding
        redundant ``json.dumps`` calls keeps multi-viewer streaming
        cheap.  ``event`` always carries at least ``type`` and
        ``taskId``, so the splice below produces exactly
        ``json.dumps({**event, "tabId": tab_id})`` (sans ordering).

        Any ``tabId`` already present on *event* is stripped first:
        events can reach this fan-out still carrying a stale stamp
        (e.g. a ``subagentDone`` broadcast with its ``tabId: ""``
        marker), and splicing a second ``"tabId"`` member would
        produce ambiguous JSON with duplicate keys — routed correctly
        today only because parsers happen to keep the last member.

        Args:
            event: The task event (carrying ``taskId``).
            encoded: ``json.dumps(event)`` from a caller that encodes
                before taking ``delivery_lock``, or ``""``.  Ignored
                when *event* carries a ``tabId``.
        """
        targets = self._fanout_targets(event.get("taskId"))
        if not targets:
            return
        if "tabId" in event:
            event = {k: v for k, v in event.items() if k != "tabId"}
            encoded = ""
        base = (encoded or json.dumps(event))[:-1]
        for tab_id in targets:
            self._track_worktree_event(event, tab_id, event.get("taskId"))
            self._send_to_ws_clients(
                f'{base}, "tabId": {json.dumps(tab_id)}}}', tab_id,
            )

    def _local_clients_for_tab(self, tab_id: str) -> list[ServerConnection]:
        """Local clients that receive a task event copy stamped *tab_id*.

        Peers that never addressed a tab keep receiving every copy.  A
        headless peer that addressed only OTHER tabs (a ``run_agent``
        or benchmark client driving its own ``api-…`` tab) is skipped:
        sending every streamed token of every task to dozens of such
        clients multiplied the event loop's work by the number of
        clients and starved it (drain timeouts dropped clients).

        Args:
            tab_id: The frontend tab id the copy is stamped with.

        Returns:
            The connections to schedule the copy on.
        """
        return locked_call(self._ws_lock, self._local_clients_for_tab_locked, tab_id)

    def _local_clients_for_tab_locked(self, tab_id: str) -> list[ServerConnection]:
        """``_local_clients_for_tab`` under ``_ws_lock``."""
        skip: set[Any] = set()
        for conn_id, tabs in self._local_tab_sets.items():
            if tabs and tab_id not in tabs:
                endpoint = self._conn_endpoints.get(conn_id)
                if endpoint is not None:
                    skip.add(endpoint)
        return [w for w in self._local_clients if w not in skip]

    def register_local_tab(self, conn_id: str, tab_id: str) -> None:
        """Record the local connection *conn_id*'s interest in *tab_id*.

        Interest is what a local command's ``tabId`` proves: this peer
        addressed the tab.  Task-event copies stamped with other tabs
        are then withheld from the peer (see
        :meth:`_local_clients_for_tab`).  The per-connection set lives
        only here, in ``_local_tab_sets``, under the printer lock.

        Args:
            conn_id: The local connection's id.
            tab_id: The frontend tab id seen on a local command.
        """
        with self._ws_lock:
            self._local_tab_sets.setdefault(conn_id, set()).add(tab_id)

    def unregister_local_tabs(self, conn_id: str) -> None:
        """Drop a disconnected local connection's local-tab registrations.

        Args:
            conn_id: The local connection's id.
        """
        with self._ws_lock:
            self._local_tab_sets.pop(conn_id, None)

    def _send_to_remote_clients(self, data: Payload) -> None:
        """Send a pre-serialised JSON payload to remote clients only.

        Remote peers (password-authenticated browsers) receive every
        copy: each browser tab mirrors the whole tab registry.

        Args:
            data: The JSON payload (already encoded with ``json.dumps``)
                or a reserved replay slot that resolves to it.
        """
        for endpoint in locked_call(self._ws_lock, self._remote_clients.copy):
            self._schedule_send(endpoint, data)

    def _send_to_local_clients(self, data: Payload, tab_id: str = "") -> None:
        """Send a pre-serialised JSON payload to local peers only.

        Local peers are Python clients that authenticated with the
        local token (``daemon_client`` runs).

        Args:
            data: The JSON payload (already encoded with ``json.dumps``)
                or a reserved replay slot that resolves to it.
            tab_id: The tab the payload is stamped with, when it is a
                task-event copy; only the peers that can show that tab
                receive it (see :meth:`_local_clients_for_tab`).  Empty
                for global events, which reach every peer.
        """
        endpoints: Iterable[ServerConnection]
        if tab_id:
            endpoints = self._local_clients_for_tab(tab_id)
        else:
            endpoints = locked_call(self._ws_lock, self._local_clients.copy)
        for endpoint in endpoints:
            self._schedule_send(endpoint, data)

    def _send_to_ws_clients(self, data: Payload, tab_id: str = "") -> None:
        """Send a pre-serialised JSON payload to every connected client.

        Factored out of :meth:`broadcast` so fan-out copies for
        subscribed viewer tab ids reuse the same dispatch and pending-
        future tracking as the primary broadcast.  Fans out to BOTH
        remote and local clients in lockstep by
        delegating to :meth:`_send_to_remote_clients` and
        :meth:`_send_to_local_clients` (per-endpoint FIFO order is
        preserved by each endpoint's ``send_lock``).  Daemon-link peers
        are not clients in this sense: :meth:`broadcast` sends them
        global events through :meth:`_send_to_link_clients` and
        nothing else.

        Args:
            data: The JSON payload (already encoded with ``json.dumps``)
                or a reserved replay slot that resolves to it.
            tab_id: The stamped tab of a task-event copy (empty for
                global events); local delivery is narrowed to the
                peers that can show it.
        """
        self._send_to_remote_clients(data)
        self._send_to_local_clients(data, tab_id)

    def _send_to_link_clients(self, data: str) -> None:
        """Send a global event to every daemon-link peer.

        Called by :meth:`broadcast` for an event with no task id or
        with an explicit empty ``tabId`` stamp only; a task-event copy
        never reaches a link peer, which wants just the replies to its
        own requests (and the global events some of those replies are,
        such as ``tabs_state``).

        Args:
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        for endpoint in locked_call(self._ws_lock, self._link_clients.copy):
            self._schedule_send(endpoint, data)

    def _reserve_replay_send(
        self, slot: ConcurrentFuture[str], conn_id: str,
    ) -> None:
        """Schedule a replay's sends now, with the payload supplied later.

        Each recipient's send takes its FIFO place at once and waits
        for *slot* (see :meth:`_locked_send`), so sends scheduled
        afterwards follow the replay while the payload is built
        outside ``delivery_lock``.  Recipients are those
        :meth:`broadcast` picks for a ``task_events`` replay: the
        requesting connection, or every client.  The caller owns
        *slot* and cancels it if the payload never comes.

        Args:
            slot: The future the serialised replay is set on.
            conn_id: Connection the replay is scoped to, or ``""``.
        """
        if conn_id:
            self._send_to_conn(conn_id, slot)
        else:
            self._send_to_ws_clients(slot)

    def _send_to_conn(self, conn_id: str, data: Payload) -> None:
        """Send a pre-serialised JSON payload to ONE connection.

        Used by :meth:`broadcast` for request/reply events stamped
        with the requesting connection's ``connId``: the reply must
        reach only the browser tab that issued the request, never its
        sibling tabs.

        Args:
            conn_id: The connection id registered via :meth:`bind_conn`.
            data: The JSON payload (already encoded with ``json.dumps``)
                or a reserved replay slot that resolves to it.
        """
        endpoint = locked_call(self._ws_lock, self._conn_endpoints.get, conn_id)
        if endpoint is None:
            return
        self._schedule_send(endpoint, data)

    def send_lock(self, endpoint: Any) -> FifoSendLock:
        """Return the per-endpoint lock serialising outbound sends.

        Every code path that writes to *endpoint* — the broadcast
        fan-out in :meth:`_schedule_send` and the direct replies in
        :meth:`RemoteAccessServer._endpoint_send` — must acquire this
        lock so payloads reach the wire in send-start order even when
        an earlier ``send()`` is suspended on write backpressure.

        Args:
            endpoint: The client connection the payload targets.

        Returns:
            The (lazily created) :class:`FifoSendLock` for *endpoint*.
        """
        return locked_call(self._ws_lock, self._send_lock_locked, endpoint)

    def _send_lock_locked(self, endpoint: Any) -> FifoSendLock:
        """``send_lock`` under ``_ws_lock``: the lock is kept only for a registered endpoint."""
        lock = self._send_locks.get(endpoint)
        if lock is None:
            lock = FifoSendLock()
            if endpoint in self._pending_sends:
                self._send_locks[endpoint] = lock
        return lock

    async def _locked_send(
        self, endpoint: ServerConnection, data: Payload,
    ) -> None:
        """Send one payload to one endpoint under its FIFO send lock.

        Args:
            endpoint: The client connection to write to.
            data: The JSON payload (already encoded with ``json.dumps``)
                or a reserved replay slot that resolves to it.
        """
        async with self.send_lock(endpoint):
            if isinstance(data, ConcurrentFuture):
                # A reserved replay slot (``replay_snapshot``): hold this
                # endpoint's FIFO place until the payload is set (or the
                # slot is cancelled, which raises ``CancelledError``).
                # Shielded so a disconnect cancelling this send cannot
                # cancel the slot shared with the other recipients.
                text = await asyncio.shield(asyncio.wrap_future(data))
            else:
                text = data
            await self._timed_send(endpoint, text)

    def _schedule_send(self, endpoint: ServerConnection, data: Payload) -> None:
        """Schedule one payload send to one endpoint on the event loop.

        Shared by :meth:`_send_to_ws_clients` (fan-out) and
        :meth:`_send_to_conn` (targeted reply).  ``endpoint`` is the
        client's :class:`ServerConnection`.  The resulting future is
        tracked in ``_pending_sends`` (M8) so a stuck/slow peer's
        pending sends can be cancelled when the client disconnects.

        Args:
            endpoint: The client connection to write to.
            data: The JSON payload (already encoded with ``json.dumps``)
                or a reserved replay slot that resolves to it.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._locked_send(endpoint, data), loop,
            )
        except Exception:
            logger.debug("Failed to send to client", exc_info=True)
            return
        if not locked_call(self._ws_lock, self._track_pending_send, endpoint, fut):
            fut.cancel()
            return
        fut.add_done_callback(
            partial(self._discard_pending_send, endpoint),
        )

    async def _timed_send(self, endpoint: ServerConnection, data: str) -> None:
        """Send *data* to *endpoint*, dropping a peer that stops reading.

        ``ServerConnection.send`` waits for the transport's write
        buffer to drain, so a peer that stops reading would otherwise
        hold its send lock forever while every later broadcast queued
        one more pending send for it without bound (the watchdog's
        15 s ping only detects a dead peer, not one that merely stops
        reading).  A send that does not complete within
        :attr:`_send_timeout` removes the peer from both client sets
        and aborts its transport, which also ends its handler.

        Args:
            endpoint: The client connection to write to.
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        try:
            await asyncio.wait_for(endpoint.send(data), self._send_timeout)
        except Exception:
            logger.debug("Failed to write to client", exc_info=True)
            self.remove_client(endpoint)
            transport = getattr(endpoint, "transport", None)
            if transport is not None:
                transport.abort()

    def add_client(
        self, ws: ServerConnection, local: bool = False, link: bool = False,
    ) -> None:
        """Register a WebSocket client for event broadcasting.

        Args:
            ws: The WebSocket server connection to add.
            local: ``True`` for a token-authenticated local peer (a
                ``daemon_client`` run), which joins
                ``_local_clients``; ``False`` for a remote,
                password-authenticated browser.
            link: ``True`` for another daemon's ``MachineLink`` request
                socket, whatever *local* says: it joins
                ``_link_clients`` instead and receives only the replies
                to its own requests and global events, never a
                task-event copy (a transcript replayed by a history
                click on this daemon used to close such a socket with
                1009 and show this machine as unreachable).
        """
        with self._ws_lock:
            if link:
                self._link_clients.add(ws)
            else:
                (self._local_clients if local else self._remote_clients).add(ws)
            self._pending_sends.setdefault(ws, set())

    def remove_client(self, ws: ServerConnection) -> None:
        """Remove a client (local or remote) and cancel its pending sends.

        Cancelling the pending ``run_coroutine_threadsafe`` futures
        (M8) ensures a permanently stuck send queue cannot keep the
        underlying coroutine alive after the peer is gone.  Safe to
        call for an already removed connection.

        Args:
            ws: The WebSocket server connection to remove.
        """
        with self._ws_lock:
            self._remote_clients.discard(ws)
            self._local_clients.discard(ws)
            self._link_clients.discard(ws)
            pending = self._pending_sends.pop(ws, set())
            self._send_locks.pop(ws, None)
        for fut in pending:
            try:
                fut.cancel()
            except Exception:
                logger.debug("Failed to cancel pending send", exc_info=True)

    def bind_conn(self, conn_id: str, endpoint: ServerConnection) -> None:
        """Associate a connection id with its transport endpoint.

        Called by the connection handler when a client connects, so
        :meth:`broadcast` can route request/reply events (stamped
        with ``connId``) back to ONLY the requesting connection.

        Args:
            conn_id: The unique id stamped (as ``connId``) on every
                command from this connection.
            endpoint: The connection's :class:`ServerConnection`.
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

    def _discard_pending_send(
        self, client: ServerConnection, fut: ConcurrentFuture[None],
    ) -> None:
        """Remove a completed send future from the per-client pending set.

        Called via :meth:`concurrent.futures.Future.add_done_callback`
        once the wrapped coroutine finishes (or errors).  Keeps the
        :attr:`_pending_sends` set bounded so it does not grow without
        limit on a long-running healthy connection.

        """
        locked_call(self._ws_lock, self._untrack_pending_send, client, fut)

    def _track_pending_send(self, endpoint: Any, fut: ConcurrentFuture[None]) -> bool:
        """Under ``_ws_lock``: add *fut* to *endpoint*'s pending sends; False once it
        unregistered."""
        pending = self._pending_sends.get(endpoint)
        if pending is None:
            return False
        pending.add(fut)
        return True

    def _untrack_pending_send(self, client: Any, fut: ConcurrentFuture[None]) -> None:
        """Under ``_ws_lock``: forget *client*'s completed send *fut*."""
        pending = self._pending_sends.get(client)
        if pending is not None:
            pending.discard(fut)


def _media_url(name: str) -> str:
    """Return a cache-busted URL for a packaged web media asset.

    The ``?v=`` value is a prefix of the sha256 of the file's CURRENT
    bytes: the file is stat'ed on every call and re-hashed whenever its
    :func:`_media_fingerprint` differs from the one last hashed.  A
    daemon keeps running across an in-place upgrade of the media files
    (the installer replaces them under the same paths), and
    ``chat.html`` is re-read on every request; a hash frozen at first
    use would pair the new page with the OLD ``main.css`` / ``main.js``
    on every browser whose service worker (media/sw.js, cache first
    for ``/media``) still holds the old URL — on a phone that showed up
    as the unstyled "Working directory" sheet stuck below the chat.
    """
    path = MEDIA_DIR / name
    fingerprint = _media_fingerprint(path)
    cached = _MEDIA_VERSION_CACHE.get(name)
    if cached is None or cached[0] != fingerprint:
        ver = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        cached = (fingerprint, ver)
        _MEDIA_VERSION_CACHE[name] = cached
    return f"/media/{name}?v={cached[1]}"


def _media_fingerprint(path: Path) -> tuple[int, int, int, int]:
    """Return a stat-based change fingerprint of a media file.

    ``(st_mtime_ns, st_ctime_ns, st_size, st_ino)``: a copy that
    preserves the source's mtime and size (``cp -p``, archive
    extraction) still moves ctime — which userspace cannot set — and
    an atomic rename-in-place replacement changes the inode.
    """
    st = path.stat()
    return (st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino)


_FONT_VARS_CSS = (
    # VS Code's workbench font (src/vs/base/browser/fonts.ts
    # DEFAULT_FONT_FAMILY) is chosen per platform; the browser cannot
    # know which VS Code build the user runs, so the stack lists the
    # macOS, Windows and Linux choices in turn.  Same for the editor
    # font (src/vs/editor/common/config/fontInfo.ts
    # EDITOR_FONT_DEFAULTS).  Both sizes are 13px: the workbench size
    # VS Code gives its webviews and the base of Cursor's UI type scale
    # (Cursor 3.23 workbench.glass.main.css: 13px / 18px line height),
    # which main.css builds on; the editor size matches it because an
    # earlier request made the chat text match the task panel, which
    # sizes itself with --theme-editor-font-size.
    "      --theme-font-size: 13px;\n"
    "      --theme-font-family: -apple-system, BlinkMacSystemFont, "
    '"Segoe WPC", "Segoe UI", system-ui, "Ubuntu", "Droid Sans", '
    "sans-serif;\n"
    "      --theme-font-weight: normal;\n"
    "      --theme-editor-font-size: 13px;\n"
    '      --theme-editor-font-family: Menlo, Monaco, Consolas, '
    '"Droid Sans Mono", "Courier New", monospace;\n'
    "      --theme-editor-font-weight: normal;\n"
)
"""VS Code's default fonts as ``--theme-*`` variables."""

_DARK_MODERN_CSS = (
    # extensions/theme-defaults/themes/dark_modern.json, plus the
    # colour-registry defaults it inherits (list.*, terminal.ansi*,
    # widget.shadow, toolbar.hoverBackground, editorWarning.foreground,
    # charts.*: platform/theme/common/colors/chartsColors.ts).
    "      --theme-foreground: #cccccc;\n"
    "      --theme-descriptionForeground: #9d9d9d;\n"
    "      --theme-errorForeground: #f85149;\n"
    "      --theme-icon-foreground: #cccccc;\n"
    "      --theme-focusBorder: #0078d4;\n"
    "      --theme-editor-background: #1f1f1f;\n"
    "      --theme-editor-foreground: #cccccc;\n"
    "      --theme-editor-selectionBackground: #264f78;\n"
    "      --theme-editor-inactiveSelectionBackground: #3a3d41;\n"
    "      --theme-editor-selectionHighlightBackground: #add6ff26;\n"
    "      --theme-editorLineNumber-foreground: #6e7681;\n"
    "      --theme-editorLineNumber-activeForeground: #cccccc;\n"
    "      --theme-editorWidget-background: #202020;\n"
    "      --theme-editorWarning-foreground: #cca700;\n"
    "      --theme-editorGutter-addedBackground: #2ea043;\n"
    "      --theme-editorGutter-deletedBackground: #f85149;\n"
    "      --theme-editorGutter-modifiedBackground: #0078d4;\n"
    "      --theme-sideBar-background: #181818;\n"
    "      --theme-sideBar-border: #2b2b2b;\n"
    "      --theme-panel-border: #2b2b2b;\n"
    "      --theme-widget-border: #313131;\n"
    "      --theme-widget-shadow: #0000005c;\n"
    "      --theme-editorGroupHeader-tabsBackground: #2b2b2b;\n"
    "      --theme-tab-activeBackground: #1f1f1f;\n"
    "      --theme-tab-inactiveBackground: #2b2b2b;\n"
    "      --theme-tab-activeForeground: #ffffff;\n"
    "      --theme-tab-inactiveForeground: #9d9d9d;\n"
    "      --theme-tab-activeBorderTop: #0078d4;\n"
    "      --theme-activityBar-foreground: #d7d7d7;\n"
    "      --theme-activityBar-inactiveForeground: #868686;\n"
    "      --theme-activityBar-activeBorder: #0078d4;\n"
    "      --theme-list-hoverBackground: #2a2d2e;\n"
    "      --theme-list-activeSelectionBackground: #04395e;\n"
    "      --theme-list-inactiveSelectionBackground: #37373d;\n"
    "      --theme-input-background: #313131;\n"
    "      --theme-input-foreground: #cccccc;\n"
    "      --theme-input-border: #3c3c3c;\n"
    "      --theme-input-placeholderForeground: #989898;\n"
    "      --theme-dropdown-background: #313131;\n"
    "      --theme-dropdown-border: #3c3c3c;\n"
    "      --theme-button-background: #0078d4;\n"
    "      --theme-button-foreground: #ffffff;\n"
    "      --theme-button-hoverBackground: #026ec1;\n"
    "      --theme-button-border: #ffffff1a;\n"
    "      --theme-button-secondaryBackground: #00000000;\n"
    "      --theme-button-secondaryForeground: #cccccc;\n"
    "      --theme-button-secondaryHoverBackground: #2b2b2b;\n"
    "      --theme-badge-background: #616161;\n"
    "      --theme-badge-foreground: #f8f8f8;\n"
    "      --theme-textLink-foreground: #4daafc;\n"
    "      --theme-textLink-activeForeground: #4daafc;\n"
    "      --theme-textCodeBlock-background: #2b2b2b;\n"
    "      --theme-textBlockQuote-background: #2b2b2b;\n"
    "      --theme-textBlockQuote-border: #616161;\n"
    "      --theme-textPreformat-foreground: #d0d0d0;\n"
    "      --theme-textPreformat-background: #3c3c3c;\n"
    "      --theme-menu-background: #1f1f1f;\n"
    "      --theme-menu-selectionBackground: #0078d4;\n"
    "      --theme-notifications-background: #1f1f1f;\n"
    "      --theme-notifications-border: #2b2b2b;\n"
    "      --theme-toolbar-hoverBackground: #5a5d5e50;\n"
    "      --theme-scrollbarSlider-background: #79797966;\n"
    "      --theme-terminal-foreground: #cccccc;\n"
    "      --theme-terminal-ansiBlack: #000000;\n"
    "      --theme-terminal-ansiRed: #cd3131;\n"
    "      --theme-terminal-ansiGreen: #0dbc79;\n"
    "      --theme-terminal-ansiYellow: #e5e510;\n"
    "      --theme-terminal-ansiBlue: #2472c8;\n"
    "      --theme-terminal-ansiMagenta: #bc3fbc;\n"
    "      --theme-terminal-ansiCyan: #11a8cd;\n"
    "      --theme-terminal-ansiWhite: #e5e5e5;\n"
    "      --theme-terminal-ansiBrightBlack: #666666;\n"
    "      --theme-terminal-ansiBrightRed: #f14c4c;\n"
    "      --theme-terminal-ansiBrightGreen: #23d18b;\n"
    "      --theme-terminal-ansiBrightYellow: #f5f543;\n"
    "      --theme-terminal-ansiBrightBlue: #3b8eea;\n"
    "      --theme-terminal-ansiBrightMagenta: #d670d6;\n"
    "      --theme-terminal-ansiBrightCyan: #29b8db;\n"
    "      --theme-terminal-ansiBrightWhite: #e5e5e5;\n"
    "      --theme-charts-green: #89d185;\n"
    "      --theme-charts-red: #f14c4c;\n"
    "      --theme-charts-yellow: #cca700;\n"
    "      --theme-charts-purple: #b180d7;\n"
)
"""VS Code's "Dark Modern" theme as ``--theme-*`` variables."""

_LIGHT_MODERN_CSS = (
    # extensions/theme-defaults/themes/light_modern.json and the
    # colour-registry light defaults, one line per Dark Modern line.
    "      --theme-foreground: #3b3b3b;\n"
    "      --theme-descriptionForeground: #3b3b3b;\n"
    "      --theme-errorForeground: #f85149;\n"
    "      --theme-icon-foreground: #3b3b3b;\n"
    "      --theme-focusBorder: #005fb8;\n"
    "      --theme-editor-background: #ffffff;\n"
    "      --theme-editor-foreground: #3b3b3b;\n"
    "      --theme-editor-selectionBackground: #add6ff;\n"
    "      --theme-editor-inactiveSelectionBackground: #e5ebf1;\n"
    "      --theme-editor-selectionHighlightBackground: #add6ff80;\n"
    "      --theme-editorLineNumber-foreground: #6e7681;\n"
    "      --theme-editorLineNumber-activeForeground: #171184;\n"
    "      --theme-editorWidget-background: #f8f8f8;\n"
    "      --theme-editorWarning-foreground: #bf8803;\n"
    "      --theme-editorGutter-addedBackground: #2ea043;\n"
    "      --theme-editorGutter-deletedBackground: #f85149;\n"
    "      --theme-editorGutter-modifiedBackground: #005fb8;\n"
    "      --theme-sideBar-background: #f8f8f8;\n"
    "      --theme-sideBar-border: #e5e5e5;\n"
    "      --theme-panel-border: #e5e5e5;\n"
    "      --theme-widget-border: #e5e5e5;\n"
    "      --theme-widget-shadow: #00000029;\n"
    "      --theme-editorGroupHeader-tabsBackground: #e5e5e5;\n"
    "      --theme-tab-activeBackground: #ffffff;\n"
    "      --theme-tab-inactiveBackground: #e5e5e5;\n"
    "      --theme-tab-activeForeground: #3b3b3b;\n"
    "      --theme-tab-inactiveForeground: #616161;\n"
    "      --theme-tab-activeBorderTop: #005fb8;\n"
    "      --theme-activityBar-foreground: #1f1f1f;\n"
    "      --theme-activityBar-inactiveForeground: #616161;\n"
    "      --theme-activityBar-activeBorder: #005fb8;\n"
    "      --theme-list-hoverBackground: #f2f2f2;\n"
    "      --theme-list-activeSelectionBackground: #e8e8e8;\n"
    "      --theme-list-inactiveSelectionBackground: #e4e6f1;\n"
    "      --theme-input-background: #ffffff;\n"
    "      --theme-input-foreground: #3b3b3b;\n"
    "      --theme-input-border: #cecece;\n"
    "      --theme-input-placeholderForeground: #767676;\n"
    "      --theme-dropdown-background: #ffffff;\n"
    "      --theme-dropdown-border: #cecece;\n"
    "      --theme-button-background: #005fb8;\n"
    "      --theme-button-foreground: #ffffff;\n"
    "      --theme-button-hoverBackground: #0258a8;\n"
    "      --theme-button-border: #0000001a;\n"
    "      --theme-button-secondaryBackground: #e5e5e5;\n"
    "      --theme-button-secondaryForeground: #3b3b3b;\n"
    "      --theme-button-secondaryHoverBackground: #cccccc;\n"
    "      --theme-badge-background: #cccccc;\n"
    "      --theme-badge-foreground: #3b3b3b;\n"
    "      --theme-textLink-foreground: #005fb8;\n"
    "      --theme-textLink-activeForeground: #005fb8;\n"
    "      --theme-textCodeBlock-background: #f8f8f8;\n"
    "      --theme-textBlockQuote-background: #f8f8f8;\n"
    "      --theme-textBlockQuote-border: #e5e5e5;\n"
    "      --theme-textPreformat-foreground: #3b3b3b;\n"
    "      --theme-textPreformat-background: #0000001f;\n"
    "      --theme-menu-background: #ffffff;\n"
    "      --theme-menu-selectionBackground: #005fb8;\n"
    "      --theme-notifications-background: #ffffff;\n"
    "      --theme-notifications-border: #e5e5e5;\n"
    "      --theme-toolbar-hoverBackground: #b8b8b850;\n"
    "      --theme-scrollbarSlider-background: #64646466;\n"
    "      --theme-terminal-foreground: #3b3b3b;\n"
    "      --theme-terminal-ansiBlack: #000000;\n"
    "      --theme-terminal-ansiRed: #cd3131;\n"
    "      --theme-terminal-ansiGreen: #107c10;\n"
    "      --theme-terminal-ansiYellow: #949800;\n"
    "      --theme-terminal-ansiBlue: #0451a5;\n"
    "      --theme-terminal-ansiMagenta: #bc05bc;\n"
    "      --theme-terminal-ansiCyan: #0598bc;\n"
    "      --theme-terminal-ansiWhite: #555555;\n"
    "      --theme-terminal-ansiBrightBlack: #666666;\n"
    "      --theme-terminal-ansiBrightRed: #f14c4c;\n"
    "      --theme-terminal-ansiBrightGreen: #14ce14;\n"
    "      --theme-terminal-ansiBrightYellow: #b5ba00;\n"
    "      --theme-terminal-ansiBrightBlue: #3b8eea;\n"
    "      --theme-terminal-ansiBrightMagenta: #d670d6;\n"
    "      --theme-terminal-ansiBrightCyan: #29b8db;\n"
    "      --theme-terminal-ansiBrightWhite: #a5a5a5;\n"
    "      --theme-charts-green: #388a34;\n"
    "      --theme-charts-red: #e51400;\n"
    "      --theme-charts-yellow: #bf8803;\n"
    "      --theme-charts-purple: #652d90;\n"
)
"""VS Code's "Light Modern" theme as ``--theme-*`` variables."""

_THEME_VARS_CSS = (
    ":root {\n" + _FONT_VARS_CSS + _DARK_MODERN_CSS + "    }\n"
)
"""The theme variables main.css derives its palette from.

The chat page (:func:`_build_html`) and the shared chat pages
(:func:`_build_share_page`) both inline this block — one copy, so the
two pages can never disagree on the palette.  Dark Modern is the
default theme; the light theme swaps in
:data:`_SHARE_PAGE_LIGHT_VARS_CSS`.
"""

_SHARE_PAGE_LIGHT_VARS_CSS = (
    "html.light-theme,\n"
    "    body.remote-chat.light-theme {\n" + _LIGHT_MODERN_CSS + "    }\n"
)
"""Light Modern overrides for the light/dark theme toggles.

Both pages inline this block after :data:`_THEME_VARS_CSS`.
``share.js`` toggles the ``light-theme`` class on ``<html>``, the
same element the ``:root`` block targets, so ``main.css``'s
``:root``-level derived variables (``--bg``, ``--fg``, ...) pick the
overrides up.  ``main.js`` toggles it on ``<body>`` instead, where
``media/remote-codex.css`` re-derives every semantic variable from
the ``--theme-*`` names, so the remote page follows too.
"""

_SHARE_PAGE_CSS = """\
/* A shared chat page is a normal scrolling document, not the
   fixed-height app shell that main.css lays out. */
html, body { height: auto; overflow: auto; }
#app { height: auto; display: block; }
#output { overflow: visible; }
/* Chrome that only works inside the live chat page. */
.panel-copy-btn, .panel-stop-btn { display: none !important; }
/* The sub-agent tab strip (created and driven by share.js, styled by
   the inlined main.css's #tab-bar / .chat-tab rules). It rides along
   the top of the scrolling document, with room on the right for the
   floating theme toggle. */
#tab-bar {
  position: sticky;
  top: 0;
  z-index: var(--z-docked);
  padding-right: 56px;
}
/* A hidden section (a sub-agent transcript whose tab is not open, or
   the chat itself while a sub-agent tab is selected) must stay hidden
   whatever display value other rules give it. */
.share-task[hidden] { display: none !important; }
/* The floating light/dark theme toggle (wired up by share.js). */
#share-theme-btn {
  position: fixed;
  top: 10px;
  right: 14px;
  z-index: var(--z-drawer);
  display: flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: var(--radius-round);
  border: 1px solid var(--border);
  background: var(--bg2);
  color: var(--fg);
  cursor: pointer;
}
#share-theme-btn:hover { border-color: var(--accent); color: var(--accent); }
/* One .share-task section per task of the chat, oldest first. */
.share-task + .share-task {
  margin-top: var(--space-3);
  border-top: 1px solid var(--border);
}
"""
"""Layout overrides appended after main.css on a shared chat page."""


def _share_page_filename(title: str, chat_id: str) -> str:
    """Return the file name for a shared chat page.

    ``chat-<title-slug>-<chat-id>.html`` when the chat has a title,
    so the file is recognisable in a folder listing (a slug of up to
    60 characters: lower-case letters, digits and hyphens); the
    filename-safe chat id (up to 80 characters) is always the suffix,
    which keeps the name unique per chat and makes re-sharing the
    same chat overwrite its previous page.  A chat without a usable
    title gets the previous ``chat-<chat-id>.html`` name.

    Args:
        title: The chat title as shown in the tab (may be empty).
        chat_id: The chat's id.

    Returns:
        The bare file name (no directory).
    """
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", chat_id)
    safe_id = safe_id.strip("-.")[:80] or "chat"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60].strip("-")
    if slug:
        return f"chat-{slug}-{safe_id}.html"
    return f"chat-{safe_id}.html"


def _build_share_page(title: str, body_html: str) -> str:
    """Build one standalone, self-contained shared chat page.

    Wraps *body_html* — the chat page's serialized chat, one
    ``.share-task`` section (task panel + transcript) per task
    of the chat, plus one hidden ``.share-task.share-subagent``
    section per sub-agent the chat fanned out (see
    ``buildShareableHtml`` in ``media/main.js``) —
    in a complete HTML document that needs no
    server: ``media/main.css`` (the exact stylesheet the chat page
    uses), both highlight.js themes, the palette variables
    (dark plus the light-mode overrides behind the page's theme
    toggle) and ``media/share.js`` (collapse / expand behaviour for
    the event panels, the task panel, the sub-agent tab strip
    that opens and closes the sub-agent sections like the live
    chat page's tabs, and the light/dark toggle) are all inlined.

    Args:
        title: Page title; falls back to "KISS Sorcar chat".
        body_html: The serialized ``#output`` markup (one
            ``.share-task`` section per task), placed verbatim inside
            the page's ``#app``.

    Returns:
        The complete HTML document string.
    """
    main_css = (MEDIA_DIR / "main.css").read_text(encoding="utf-8")
    brand_css = (MEDIA_DIR / "brand.css").read_text(encoding="utf-8")
    hljs_dark_css = (MEDIA_DIR / "highlight-dark.css").read_text(
        encoding="utf-8",
    )
    hljs_light_css = (MEDIA_DIR / "highlight-light.css").read_text(
        encoding="utf-8",
    )
    share_js = (MEDIA_DIR / "share.js").read_text(encoding="utf-8")
    page_title = html.escape(title.strip() or f"{PRODUCT_NAME} chat")
    # Both highlight.js themes ship inline; share.js's theme toggle
    # flips which one applies through the style elements' media
    # attribute (dark is the default).
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, '
        'initial-scale=1.0">\n'
        f"<title>{page_title}</title>\n"
        "<style>\n" + _THEME_VARS_CSS + "</style>\n"
        "<style>\n" + _SHARE_PAGE_LIGHT_VARS_CSS + "</style>\n"
        '<style id="hljs-style-dark">\n' + hljs_dark_css + "\n</style>\n"
        '<style id="hljs-style-light" media="not all">\n'
        + hljs_light_css + "\n</style>\n"
        "<style>\n" + main_css + "\n</style>\n"
        "<style>\n" + brand_css + "\n</style>\n"
        "<style>\n" + _SHARE_PAGE_CSS + "</style>\n"
        "</head>\n"
        "<body>\n"
        '<button id="share-theme-btn" type="button" '
        'title="Switch to light mode" '
        'aria-label="Switch to light mode"></button>\n'
        '<div id="app">\n' + body_html + "\n</div>\n"
        "<script>\n" + share_js + "</script>\n"
        "</body>\n"
        "</html>\n"
    )


def _page_title() -> str:
    """Return the remote page's ``<title>``: ``KISS Sorcar: <machine name>``.

    The machine name is ``platform.node()`` (the same host name the
    task settings report as "Machine info"), so a user with several
    KISS servers open can tell the browser tabs apart.  Falls back to
    the bare product name when the host name is unknown.
    """
    node = platform.node().strip()
    return f"{PRODUCT_NAME}: {node}" if node else PRODUCT_NAME


def _build_html() -> str:
    """Build the webapp's HTML page.

    Fills the ``{{PLACEHOLDER}}`` slots of ``media/chat.html``:
    cache-busted ``/media/`` URLs of the stylesheets and scripts
    (``transport.js`` is the WebSocket transport ``main.js`` talks
    to the daemon through), the inline brand / tricks / tips / voice
    JSON, the ``loading...`` model name and the auth-modal block.
    ``media/assetLoadGuard.js`` is inlined as the first script of
    ``<head>`` so it is installed before any asset can fail to load.

    Returns:
        The complete HTML string.
    """
    version = _read_version()
    tricks_data = read_tricks_data()
    tricks_json = json.dumps(tricks_data["tricks"]).replace("</", "<\\/")
    tips_json = json.dumps(tips_data(version)).replace("</", "<\\/")
    head_style = (
        f'<link href="{_media_url("remote-codex.css")}" rel="stylesheet">\n'
        "  <style>\n"
        "    html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; }\n"
        "    body { background: var(--theme-editor-background, #1f1f1f);\n"
        "            color: var(--theme-editor-foreground, #cccccc); }\n"
        "    " + _THEME_VARS_CSS
        + "    " + _SHARE_PAGE_LIGHT_VARS_CSS + "  </style>"
    )
    auth_modal = (
        '    <div id="auth-modal" style="display:none;" role="dialog" '
        'aria-modal="true"\n'
        '         aria-labelledby="auth-modal-title" '
        'aria-describedby="auth-modal-error">\n'
        '      <div class="auth-modal-content">\n'
        '        <div class="auth-modal-title" id="auth-modal-title">'
        'Remote access password</div>\n'
        '        <label for="auth-modal-input" id="auth-modal-label" '
        'style="display:block; margin-bottom:6px;">'
        'Password for this KISS Sorcar server</label>\n'
        '        <input type="password" id="auth-modal-input" '
        'class="auth-modal-input"\n'
        '               autocomplete="current-password" '
        'placeholder="Enter password">\n'
        '        <div id="auth-modal-error" role="alert" '
        'style="min-height:1.2em; margin-top:6px; '
        'color: var(--theme-errorForeground, #f14c4c);"></div>\n'
        '        <div class="auth-modal-actions">\n'
        '          <button id="auth-modal-cancel" '
        'class="auth-modal-btn auth-modal-cancel"\n'
        '                  type="button">Cancel</button>\n'
        '          <button id="auth-modal-ok" '
        'class="auth-modal-btn auth-modal-ok"\n'
        '                  type="button">Unlock</button>\n'
        '        </div>\n'
        '      </div>\n'
        '    </div>\n'
    )
    subs = {
        "VIEWPORT": "width=device-width,initial-scale=1,maximum-scale=1",
        "HEAD_SCRIPT": (
            "<script>\n"
            + (MEDIA_DIR / "assetLoadGuard.js").read_text(encoding="utf-8")
            + "</script>"
        ),
        "STYLE_HREF": _media_url("main.css"),
        "BRAND_STYLE_HREF": _media_url("brand.css"),
        "WELCOME_LOGO_SRC": _media_url("welcome-logo.png"),
        "WELCOME_LOGO_DARK_SRC": _media_url("welcome-logo-dark.png"),
        # Light Modern is the remote page's default theme: the body is
        # rendered with ``light-theme`` (and the light highlight sheet)
        # so the first paint is already light; main.js drops the class
        # again for a client whose saved choice is dark.
        "HLJS_CSS_HREF": _media_url("highlight-light.css"),
        "HEAD_STYLE": head_style,
        "BODY_CLASS_ATTR": ' class="remote-chat light-theme"',
        "PRODUCT_NAME": html.escape(PRODUCT_NAME),
        # The browser tab names the machine the daemon runs on, so a
        # user with several servers open can tell them apart.
        "PAGE_TITLE": html.escape(_page_title()),
        "TAGLINE": html.escape(BRAND["tagline"]),
        "HOME_DIR": html.escape(HOME_DIR),
        "BRAND_JSON": json.dumps(
            {
                "productName": PRODUCT_NAME,
                "shortName": BRAND["short_name"],
                "homeDir": HOME_DIR,
            },
        ).replace("</", "<\\/"),
        "INPUT_PLACEHOLDER": "Ask anything... (@ for files)",
        "ENTERKEYHINT": ' enterkeyhint="send"',
        "MODEL_NAME": "loading...",
        "VERSION_SUFFIX": f" {version}" if version else "",
        "AUTH_MODAL": auth_modal,
        "HLJS_SRC": _media_url("highlight.min.js"),
        "MARKED_SRC": _media_url("marked.min.js"),
        "API_SRC": _media_url("api.js"),
        "PANEL_COPY_SRC": _media_url("panelCopy.js"),
        "CTX_MENU_SRC": _media_url("contentContextMenu.js"),
        "TREE_MENU_SRC": _media_url("treeContextMenu.js"),
        "BROWSER_TAB_SRC": _media_url("browserTab.js"),
        "TERMINAL_TAB_SRC": _media_url("terminalTab.js"),
        "PDF_VIEW_SRC": _media_url("pdfView.js"),
        "MOBILE_SWIPE_SRC": _media_url("mobileSwipe.js"),
        "MAIN_SRC": _media_url("main.js"),
        "HLJS_THEME_CSS": json.dumps({
            "dark": _media_url("highlight-dark.css"),
            "light": _media_url("highlight-light.css"),
        }),
        "TRANSPORT_SRC": _media_url("transport.js"),
        "TRICKS_JSON": tricks_json,
        "MY_TRICKS_COUNT": str(tricks_data["userCount"]),
        "TIPS_JSON": tips_json,
        "TIPS_SRC": _media_url("tips.js"),
        "VOICE_SRC": _media_url("voice.js"),
        "VOICE_CONFIG": json.dumps({
            "voskSrc": _media_url("vosk.js"),
            "modelUrl": "/voice-model.tar.gz",
            "ackAudioUrl": _media_url("working-on-it.mp3"),
        }),
    }
    tpl = (MEDIA_DIR / "chat.html").read_text(encoding="utf-8")

    def _fill(m: re.Match[str]) -> str:
        space, key = m.group(1), m.group(2)
        if key not in subs:
            return m.group(0)
        # Attribute-string placeholders carry their own leading space;
        # the template writes them after a separating space
        # (``<body {{BODY_CLASS_ATTR}}>``) only so htmlhint can parse the
        # tag.  Drop that space so the page renders exactly
        # ``<body class="remote-chat light-theme">``.
        if key in _ATTR_STRING_KEYS:
            return subs[key]
        return space + subs[key]

    return re.sub(r"( ?)\{\{([A-Z_]+)\}\}", _fill, tpl)


_ATTR_STRING_KEYS = frozenset({"BODY_CLASS_ATTR", "ENTERKEYHINT"})


_MEDIA_URL_RE = re.compile(r"/media/[A-Za-z0-9_.-]+\?v=[0-9a-f]+")


def _app_shell_urls() -> list[str]:
    """Return the URLs the service worker precaches: the app shell.

    The shell is the page itself (``/``) plus every cache-busted
    ``/media/<name>?v=<hash>`` URL the rendered page references
    (stylesheets, scripts, the theme CSS swapped in by ``main.js``,
    the voice engine and its ack sound).  Deriving the list from
    :func:`_build_html` keeps it in lockstep with the page: an asset
    added to the template is precached without a second list to
    maintain.  The hashes come from :func:`_media_url`, so the list
    is stable while the assets on disk are unchanged and changes
    exactly when an asset's bytes change — even under a daemon that
    keeps running across an in-place upgrade.
    """
    urls = sorted(set(_MEDIA_URL_RE.findall(_build_html())))
    return ["/", *urls, *_brand_css_asset_urls()]


_CSS_URL_RE = re.compile(r"""url\(\s*["']?([A-Za-z0-9_.-]+)["']?\s*\)""", re.IGNORECASE)
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _brand_css_asset_names() -> list[str]:
    """Return the media files ``brand.css`` references through relative ``url()``.

    A skin's ``url("kiss-icon.png")`` resolves next to the stylesheet, so
    the browser requests the plain ``/media/<name>`` (no ``?v=``); those
    files are precached under exactly that URL.  Comments are ignored and
    names that do not exist in the media directory are skipped.
    """
    try:
        css = (MEDIA_DIR / "brand.css").read_text(encoding="utf-8")
    except OSError:
        return []
    names = sorted(set(_CSS_URL_RE.findall(_CSS_COMMENT_RE.sub("", css))))
    return [name for name in names if (MEDIA_DIR / name).is_file()]


def _brand_css_asset_urls() -> list[str]:
    """Return the plain ``/media/<name>`` URLs of the assets ``brand.css`` references."""
    return [f"/media/{name}" for name in _brand_css_asset_names()]


def _build_service_worker() -> str:
    """Render ``media/sw.js`` for the ``/sw.js`` endpoint.

    Substitutes the ``__KISS_SW_SHELL__`` placeholder with a JSON
    object ``{"version": <hash of the manifest>, "urls": [...]}``
    (see :func:`_app_shell_urls`).  Because the manifest is part of
    the script, the browser sees a byte-different worker — and
    installs it, with a new cache name — whenever any shell asset
    changes; identical assets yield an identical script and the
    browser keeps the worker it has.

    Returns:
        The complete service-worker script.
    """
    urls = _app_shell_urls()
    # The brand.css assets are listed without ``?v=`` (see
    # _brand_css_asset_urls), so fold their content hashes into the
    # version separately: a swapped logo must still roll the worker.
    version_input = urls + [_media_url(name) for name in _brand_css_asset_names()]
    shell = {
        "version": hashlib.sha256("\n".join(version_input).encode("utf-8")).hexdigest()[:16],
        "urls": urls,
    }
    tpl = (MEDIA_DIR / "sw.js").read_text(encoding="utf-8")
    return tpl.replace("__KISS_SW_SHELL__", json.dumps(shell))


def _read_version() -> str:
    """Return the daemon's version, as reported to ``update_available``.

    Read from the bundled ``_version.py`` on every call (never cached):
    ``install.sh`` replaces the installed package and restarts the
    service, so the file always names the running code.
    """
    vfile = Path(__file__).parent.parent / "core" / "_version.py"
    try:
        text = vfile.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
    return m.group(1) if m else ""


def _version_tuple(v: str) -> tuple[int, ...] | None:
    """Return ``v`` parsed as an int-tuple, or ``None`` on failure.

    ``kiss-agent-framework`` uses CalVer ``YYYY.M.P``; ``None`` is
    returned for anything that cannot be parsed so a malformed PyPI
    payload never triggers a false "update available" notification.
    Each dot-separated component must be ASCII digits only — a bare
    ``int()`` also accepts ``"+1"``, ``"1_0"`` and unicode digits,
    which would misjudge the update direction for malformed version
    strings.
    """
    if not isinstance(v, str):
        return None
    parts = [p for p in v.strip().split(".") if p != ""]
    if not parts:
        return None
    out: list[int] = []
    for p in parts:
        if re.fullmatch(r"\d+", p, re.ASCII) is None:
            return None
        out.append(int(p))
    return tuple(out)


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


_UPDATE_SNOOZE_MS = 24 * 60 * 60 * 1000


def _update_check_cache_path() -> Path:
    """Return the path of the "Remind me later" update-snooze cache."""
    return kiss_home() / ".update-check.json"


def _read_update_check_cache() -> dict[str, Any]:
    """Return the parsed update-check cache, ``{}`` when unreadable."""
    try:
        data = json.loads(
            _update_check_cache_path().read_text(encoding="utf-8"),
        )
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _is_update_snoozed(latest: str) -> bool:
    """Return True while a "Remind me later" snooze covers *latest*.

    The snooze holds until it expires, but a release NEWER than the
    snoozed one breaks through (``_compare_versions`` returns 0 for an
    unparseable ``snoozedLatest``, so a version-less snooze suppresses
    everything until expiry).
    """
    cache = _read_update_check_cache()
    until_ms = cache.get("snoozeUntilMs")
    if not isinstance(until_ms, (int, float)) or until_ms <= 0:
        return False
    if time.time() * 1000 >= until_ms:
        return False
    snoozed = cache.get("snoozedLatest")
    return _compare_versions(latest, snoozed if isinstance(snoozed, str) else "") <= 0


def _record_update_snooze(latest: str) -> None:
    """Record a 24h snooze for release *latest* in the snooze cache.

    Written atomically via a unique temp file + rename so a reader
    never sees a partial file.
    """
    payload = {
        "snoozeUntilMs": int(time.time() * 1000) + _UPDATE_SNOOZE_MS,
        "snoozedLatest": latest,
    }
    try:
        _atomic_write_text(_update_check_cache_path(), json.dumps(payload))
    except Exception:
        logger.debug("Failed to record update snooze", exc_info=True)


def _http_response(
    status: int,
    content_type: str,
    body: bytes,
    extra_headers: list[tuple[str, str]] | None = None,
) -> Response:
    """Build a proper HTTP/1.1 Response for the websockets server.

    Args:
        status: HTTP status code (e.g. 200, 404).
        content_type: MIME type for the Content-Type header.
        body: Response body bytes.
        extra_headers: Additional ``(name, value)`` headers, e.g. a
            ``Content-Disposition`` for downloads.

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
            *(extra_headers or []),
        ]),
        body,
    )


def _error_page(
    status: int, heading: str, advice: str, details: str = "",
) -> Response:
    """Return a small plain-language HTML error page.

    Browsers show HTTP error bodies to people, so they read like a
    note, not a log line: ``heading`` says what happened, ``advice``
    what to do next, and ``details`` (optional, for the person who
    administers the server) is tucked into a collapsed ``<details>``.

    Args:
        status: The HTTP status code (403, 404, 502, ...).
        heading: One short sentence naming the problem.
        advice: One or two sentences telling the reader what to do.
        details: Optional technical hint for administrators.

    Returns:
        A ``text/html`` response carrying the page.
    """
    details_html = (
        "<details><summary>For the server administrator</summary>"
        f"<p>{html.escape(details)}</p></details>"
        if details else ""
    )
    page = (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(heading)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:0;padding:48px 20px;"
        "max-width:36em;line-height:1.5;color:#222;background:#fff}"
        "h1{font-size:1.4em;margin:0 0 .5em}details{margin-top:2em;color:#555}"
        "summary{cursor:pointer}a{color:#0a58ca}</style></head><body>"
        f"<h1>{html.escape(heading)}</h1>"
        f"<p>{html.escape(advice)}</p>"
        "<p><a href=\"/\">Go to the KISS Sorcar start page</a></p>"
        f"{details_html}"
        f"<p style=\"color:#888;font-size:.85em\">HTTP {status}</p>"
        "</body></html>"
    )
    return _http_response(
        status, "text/html; charset=utf-8", page.encode("utf-8"),
    )


def _trajectory_jobs_response() -> Response:
    """Return a JSON HTTP response listing all trajectory jobs.

    Transport wrapper for the ``/api/jobs`` endpoint: the payload is
    produced by the server API
    (:meth:`kiss.server.sorcar.ServerApi.trajectory_jobs`); this
    function only wraps it into an HTTP response.

    Returns:
        A 200 ``application/json`` response with the job list.
    """
    return _http_response(*sorcar_api.ServerApi.trajectory_jobs())


def _trajectory_job_response(path: str) -> Response:
    """Return a JSON HTTP response with the trajectories for one job.

    Transport wrapper for the ``/api/jobs/<job_name>/trajectories``
    endpoint: the payload (including the job-name containment check
    and the no-double-unquote contract) is produced by the server API
    (:meth:`kiss.server.sorcar.ServerApi.job_trajectories`); this
    function only wraps it into an HTTP response.

    Args:
        path: Request path of the form ``/api/jobs/<job_name>/trajectories``.

    Returns:
        A 200 ``application/json`` response with the trajectory list, a 400
        response for an invalid job name, or a 404 response when the job
        directory does not exist.
    """
    return _http_response(*sorcar_api.ServerApi.job_trajectories(path))


def _read_media_file(filepath: Path) -> bytes | None:
    """Return the bytes of *filepath* if it is a real file inside MEDIA_DIR.

    Performs the symlink-safe containment check, the ``is_file`` stat and
    the read together so callers can run the whole lot in one worker
    thread.  Any :class:`OSError` (e.g. ``ELOOP`` from a symlink cycle
    hit by ``resolve()``) is treated as "not found".

    Args:
        filepath: Candidate path beneath :data:`MEDIA_DIR`.

    Returns:
        The file contents, or ``None`` when the path escapes
        :data:`MEDIA_DIR`, is not a regular file, or cannot be resolved
        or read.
    """
    try:
        if (
            filepath.resolve().is_relative_to(MEDIA_DIR.resolve())
            and filepath.is_file()
        ):
            return filepath.read_bytes()
    except OSError:
        return None
    return None


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    """Cancel *task* (if any) and wait for it to unwind.

    Args:
        task: The asyncio task to cancel, or ``None`` for a no-op.
    """
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _read_log_from(path: Path, offset: int) -> bytes:
    """Return the bytes of the append-only log *path* from *offset* on.

    Seeks instead of reading the whole file: the update log grows for
    the lifetime of the installation.  Blocking; run it off the loop.

    Args:
        path: The log file.
        offset: Byte position where the wanted slice begins.

    Returns:
        The slice, or ``b""`` when the log cannot be read.
    """
    try:
        with open(path, "rb") as log:
            log.seek(offset)
            return log.read()
    except OSError:
        return b""


def _close_orphaned_listener(bind: asyncio.Future[WebSocketServer]) -> None:
    """Close a listener whose creator was cancelled while awaiting the bind.

    Done-callback installed by :meth:`RemoteAccessServer._serve_wss` on
    the shielded ``serve()`` future when the outer await is cancelled;
    retrieving the exception also silences "exception never retrieved".

    Args:
        bind: The completed ``websockets.serve`` future.
    """
    if not bind.cancelled() and bind.exception() is None:
        bind.result().close()


class RemoteAccessServer:
    """Web server providing remote browser access to KISS Sorcar.

    Serves the Sorcar chat page over HTTPS and bridges commands/events
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
        ntfy_base_url: Base URL of the ntfy server the active tunnel
            URL is posted to (default the real ``https://ntfy.sh``).
            Tests inject a local emulator here so they never post to
            the production discovery topic.
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
        local_endpoint_file: str | Path | None = None,
        ntfy_base_url: str = _NTFY_BASE_URL,
    ) -> None:
        load_api_keys()
        # ``saveConfig`` was the only caller of apply_config_to_env, so
        # a freshly started daemon kept the DECLARED default budget
        # until the user happened to open and close the settings panel.
        # Applying the persisted config here makes every process start
        # in the state the user last saved.
        apply_config_to_env(load_config())

        self.host = host
        self.port = port
        self.use_tunnel = use_tunnel
        self.tunnel_token = tunnel_token
        self.tunnel_url = tunnel_url
        self._ssl_certfile: str | None = certfile
        self._ssl_keyfile: str | None = keyfile
        self._ssl_context: ssl.SSLContext | None = None
        self._url_file: Path = Path(url_file) if url_file else _url_file_path()

        if not work_dir:
            work_dir = load_config().get("work_dir", "") or None
        if work_dir and is_root_dir(work_dir):
            # A filesystem root (persisted by a pre-guard client, or a
            # degenerate launcher cwd) must never become the instance
            # work dir: it would root every unstamped command — and
            # the @-mention file scan — at the whole disk.  Drop it and
            # let ``AgentServer`` supply its own (also root-guarded)
            # fallback.
            work_dir = None
        self.work_dir: str = work_dir or ""

        self._voice_speaker_identifier: SpeakerIdentifier | None = None
        self._voice_speaker_broken = False
        self._voice_speaker_lock = threading.Lock()

        self._printer = WebPrinter()
        # Task-update reports for the task-info panel (getTaskUpdate).
        self._task_updates = TaskUpdateRunner()
        self._printer.work_dir = self.work_dir
        self._agent_server = AgentServer(printer=self._printer)
        if self.work_dir:
            self._agent_server.work_dir = self.work_dir
        self._server_api = sorcar_api.ServerApi(self)

        self._tunnel_proc: subprocess.Popen[str] | None = None
        self._tunnel_metrics_port: int | None = None
        self._tunnel_unhealthy_ticks = 0
        self._tunnel_started_at: float | None = None
        self._tunnel_failure_count = 0
        self._tunnel_next_retry = 0.0
        self._tunnel_adopted_pid: int | None = None
        self._tunnel_rate_limited = False
        self._tunnel_force_restart_count = 0
        self._tunnel_force_restart_next_allowed = 0.0
        # Serialises publishing a freshly spawned cloudflared into
        # ``_tunnel_proc`` against ``_stop_tunnel``: cancelling the
        # watchdog task does not stop an in-flight executor
        # ``_start_tunnel``, so without this a stop could see "no
        # process" and the start could publish a live one afterwards.
        self._tunnel_lock = threading.Lock()
        self._tunnel_stopped = False
        self._ntfy_base_url = ntfy_base_url
        self._last_posted_url: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_server: Any = None
        # Second WSS listener bound to 127.0.0.1:<port> beside the
        # wildcard one; see :meth:`_bind_loopback_alias`.
        self._ws_loopback_server: Any = None
        # Where same-machine clients learn this daemon's URL and the
        # per-start secret that marks them as local (see
        # :mod:`kiss.agents.sorcar.local_endpoint`).  Written once the WSS
        # listener is bound; removed on shutdown while it is still ours.
        self._local_endpoint_file: Path = (
            Path(local_endpoint_file) if local_endpoint_file
            else local_endpoint.default_endpoint_path()
        )
        self._local_token: str = local_endpoint.new_token()
        # Set by start_private_async: only token-authenticated local
        # clients may use the daemon, never a remote password.
        self._local_only: bool = False
        self._watchdog_task: asyncio.Task[None] | None = None
        self._tls_refresh_task: asyncio.Task[None] | None = None
        self._trusted_clients_task: asyncio.Task[None] | None = None
        # The enrolled machine-link CAs loaded into ``_ssl_context`` (PEM
        # bytes) and the mtime of their directory when last scanned.
        self._trusted_client_cas: set[bytes] = set()
        self._trusted_clients_mtime: float = -1.0
        self._latest_version: str | None = None
        self._version_check_task: asyncio.Task[None] | None = None
        self._machines_watch_task: asyncio.Task[None] | None = None
        self._shutdown_initiated = False
        self._shutdown_future: asyncio.Future[None] | None = None
        self._local_url = f"http://localhost:{self.port}"
        self._active_url: str | None = None
        # Stamp of the latest ``remote_url`` publication; see
        # :meth:`_broadcast_remote_url`.
        self._url_publish_gen = 0
        # The watchdog's in-flight LAN-URL republish (see
        # :meth:`_republish_urls`); tracked so shutdown can cancel it.
        self._republish_task: asyncio.Task[None] | None = None
        self._last_ips: frozenset[str] = frozenset()
        # PEM of the auto-generated server certificate the live SSL
        # context is serving; see :meth:`_refresh_tls_cert`.
        self._tls_loaded_cert: bytes = b""
        self._ips_probed = False
        self._pending_ip_change: frozenset[str] | None = None
        self._pending_ip_change_count: int = 0
        self._auth_failures: dict[str, list[float]] = {}
        self._install_root: Path = _KISS_AI_ROOT
        self._update_log_path: Path = kiss_home() / "update.log"
        self._update_proc: subprocess.Popen[bytes] | None = None
        self._update_starting = False
        self._update_watch_task: asyncio.Task[None] | None = None
        # "Update when idle": ``_update_when_idle_task`` polls the agent
        # registry and runs the installer once no task is in flight.  It
        # stays tracked (for shutdown) until it finishes; ``_armed`` is
        # True only while it is still waiting for idle, i.e. while a
        # Cancel can still call it off.
        self._update_when_idle_task: asyncio.Task[None] | None = None
        self._update_when_idle_armed = False
        self._update_models_log_path: Path = (
            kiss_home() / "update_models.log"
        )
        self._update_models_argv: list[str] = [
            sys.executable,
            "-m",
            "kiss.scripts.update_models",
            "--model-info",
            str(kiss_home() / "MODEL_INFO.json"),
        ]
        self._update_models_proc: subprocess.Popen[bytes] | None = None
        self._update_models_starting = False
        self._update_models_watch_task: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        # Whether ``_on_sea_commands_changed`` is subscribed to the SEA
        # slash-command registry (see ``_start_sea_command_watcher``).
        self._sea_command_subscribed = False

    async def _process_request(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        """Serve HTTP requests for the HTML page and static assets.

        Returns a :class:`Response` for regular HTTP requests, or
        ``None`` to let the WebSocket handshake proceed for ``/ws``.

        This is the choke point of the no-password lockdown for every
        PARSED request: when the configured ``remote_password`` is
        empty, EVERY request from a non-loopback peer — the HTML
        page, static assets, the trajectory data endpoints, and the
        ``/ws`` WebSocket upgrade itself — is refused with ``403``.
        (The one request kind answered before the parser runs, the
        HEAD health check, applies the same rule in
        :func:`_head_health_response`.)  Without this gate the
        default ``0.0.0.0`` bind would let any LAN machine
        authenticate with the empty password.  Legitimate remote
        traffic is unaffected: it arrives via the local cloudflared
        tunnel (a loopback peer), and the tunnel is only started (or
        kept alive) when a password is configured.

        Args:
            connection: The server connection (used for the peer
                address check above).
            request: The incoming HTTP request.

        Returns:
            An HTTP response, or ``None`` for WebSocket upgrade.
        """
        if not self._peer_is_loopback(connection) and not self._peer_client_ca(connection):
            cfg = await asyncio.to_thread(load_config)
            if not str(cfg.get("remote_password", "") or ""):
                addr = getattr(connection, "remote_address", None)
                logger.warning(
                    "Refusing non-localhost request from %s: "
                    "no remote password is set", addr,
                )
                return _error_page(
                    403,
                    "This device is not allowed yet",
                    "Remote access to this KISS Sorcar server is turned "
                    "off. On the computer running KISS Sorcar, run "
                    "install.sh to set a remote password; then come "
                    "back to this page.",
                    "Only localhost may connect while no remote password "
                    f"is set (remote_password in ~/{HOME_DIR}/config.json); "
                    "run install.sh to set one.",
                )
        request_path = urlsplit(request.path).path
        path = unquote(request_path)
        if path in ("", "/"):
            # Rendered per page load (off-thread: it reads the
            # template, TIPS.md and the trick files), never cached
            # for the daemon's lifetime — the page embeds
            # ``window.__TRICKS__``, and a list frozen at startup made
            # the Inject panel disagree with the daemon's own
            # ghost-text completions (which re-read the files) as
            # soon as the user edited ``MY_INJECTION.md``.
            html_page = await asyncio.to_thread(_build_html)
            return _http_response(
                200, "text/html; charset=utf-8", html_page.encode("utf-8"),
            )
        if path == "/ws":
            return None
        if path == "/sw.js":
            # The offline app-shell service worker (media/sw.js with
            # the precache manifest filled in).  Served from the site
            # root so its scope covers "/"; the no-store headers of
            # _http_response make every navigation's update check
            # fetch the current script.
            sw_script = await asyncio.to_thread(_build_service_worker)
            return _http_response(
                200, "text/javascript; charset=utf-8", sw_script.encode("utf-8"),
            )
        if path in ("/trajectories", "/trajectories/"):
            return _http_response(
                200,
                "text/html; charset=utf-8",
                await asyncio.to_thread(TRAJECTORY_TEMPLATE.read_bytes),
            )
        if path == "/api/jobs":
            return await asyncio.to_thread(_trajectory_jobs_response)
        if path.startswith("/api/jobs/") and path.endswith("/trajectories"):
            return await asyncio.to_thread(_trajectory_job_response, path)
        if path == "/ca.crt":
            # The auto-generated local CA certificate (public data, no
            # key) so a phone on the LAN can install and trust it and
            # stop warning about the Local/LAN URLs.  The MIME type
            # makes iOS offer the profile installer; the filename
            # keeps Android's download recognisable.  Absent when the
            # daemon runs with an explicit --certfile pair.
            ca_bytes = await asyncio.to_thread(
                _local_ca_cert_bytes,
            ) if self._serves_local_ca else None
            if ca_bytes is None:
                return _error_page(
                    404,
                    "There is no certificate to download here",
                    "This server runs with its own certificate files, so "
                    "it has no local CA certificate to install. Go back "
                    "to the start page.",
                    "/ca.crt is only served when the daemon generated "
                    "its own local CA (no --certfile/--keyfile).",
                )
            return _http_response(
                200, "application/x-x509-ca-cert", ca_bytes,
                [("Content-Disposition",
                  'attachment; filename="kiss-sorcar-local-ca.crt"')],
            )
        if path == "/voice-model.tar.gz":
            model_file = await asyncio.to_thread(_ensure_voice_model)
            if model_file is None:
                return _error_page(
                    502,
                    "The voice model could not be downloaded",
                    "The server could not fetch the speech-recognition "
                    "model right now. Check the server's internet "
                    "connection and try again in a minute.",
                    "_ensure_voice_model() returned None: the download "
                    "from the model host failed or timed out.",
                )
            # A sibling process (a second daemon, a test beside a live
            # one) may be publishing its own download of the archive at
            # this very moment; on Windows its ``os.replace`` denies a
            # plain open of the name for a few milliseconds.
            body = await asyncio.to_thread(read_bytes_waiting_for_writer, model_file)
            return _http_response(200, "application/gzip", body)
        if path.startswith("/media/"):
            filepath = MEDIA_DIR / path[7:]
            media_body = await asyncio.to_thread(_read_media_file, filepath)
            if media_body is not None:
                ctype = mimetypes.guess_type(str(filepath))[0] or "application/octet-stream"
                return _http_response(200, ctype, media_body)
        return _error_page(
            404,
            "There is nothing at this address",
            "The link may be old or mistyped. Go to the start page to "
            "open KISS Sorcar.",
            f"No route matches {path[:200]!r}.",
        )

    def _peer_client_ca(self, connection: Any) -> str:
        """Return the enrolled CA that issued *connection*'s client certificate, or ''.

        A non-empty name means the peer is another daemon's machine
        link (:mod:`kiss.agents.sorcar.machines`) enrolled in
        :func:`_trusted_clients_dir`: the TLS handshake verified its
        certificate against the loaded CAs and the issuing CA's file is
        still there (:func:`tls_certs.enrolled_client_ca`).  Such a
        peer is admitted like a loopback client with the local token,
        whatever the ``remote_password``.  Empty when the peer sent no
        certificate (every browser) or the transport is not TLS.

        Args:
            connection: A WebSocket server connection (its ``transport``
                is the asyncio TLS transport).
        """
        transport = getattr(connection, "transport", None)
        ssl_object = transport.get_extra_info("ssl_object") if transport is not None else None
        if ssl_object is None:
            return ""
        try:
            der = ssl_object.getpeercert(binary_form=True)
        except ValueError:  # the handshake is not complete
            return ""
        return tls_certs.enrolled_client_ca(der, _trusted_clients_dir()) if der else ""

    def _peer_is_loopback(self, connection: Any) -> bool:
        """Return True when *connection*'s raw TCP peer is loopback.

        Backend primitive for :meth:`ServerApi.authenticate`; thin
        wrapper over :func:`_connection_peer_is_loopback` (see there
        for the fail-closed / no-forwarded-headers rationale).

        Args:
            connection: A WebSocket server connection (or any object
                with a ``remote_address`` tuple).

        Returns:
            True when the TCP peer is an IPv4/IPv6 loopback address.
        """
        return _connection_peer_is_loopback(connection)

    def _client_ip(self, websocket: ServerConnection) -> str:
        """Return the rate-limit bucket key (source IP) of *websocket*.

        The public WSS port is reached through the local ``cloudflared``
        tunnel, which connects over **loopback**.  Using the raw TCP
        peer address as the rate-limit key would therefore collapse
        *every* tunnel visitor onto a single ``127.0.0.1`` bucket, so a
        single bad actor's failed guesses (or one user fat-fingering the
        password) would trip the brute-force lockout for **everyone** —
        after which new visitors are refused with ``auth_locked`` and
        never shown the password prompt at all ("the remote webapp
        doesn't ask for a password").

        To key the lockout on the *real* client instead, when the direct
        TCP peer is loopback we trust the client IP that cloudflared
        forwards in the upgrade request headers (``Cf-Connecting-Ip``,
        falling back to the first hop of ``X-Forwarded-For``).  The
        header is honoured **only** for loopback peers — a non-loopback
        peer connecting directly (bypassing cloudflared) could otherwise
        spoof the header to evade or poison the lockout — so direct
        connections always fall back to their real TCP address.

        Returns:
            A stable per-client string used as the rate-limit key, or
            ``"?"`` when the peer address is unknown.
        """
        addr = getattr(websocket, "remote_address", None)
        peer_ip = str(addr[0]) if addr and len(addr) >= 1 else ""
        if peer_ip and sorcar_api.is_loopback_ip(peer_ip):
            forwarded = _forwarded_client_ip(websocket)
            if forwarded:
                return forwarded
        return peer_ip or "?"

    def _auth_lock_remaining(self, ip: str) -> float:
        """Return the seconds left in *ip*'s rate-limit lock (0.0 if none).

        An IP becomes locked once it has accumulated
        :data:`_AUTH_FAIL_MAX` failures within the most recent
        :data:`_AUTH_FAIL_WINDOW` seconds.  The lock persists until
        :data:`_AUTH_LOCKOUT` seconds have elapsed since the last
        recorded failure; the returned value is the time remaining
        until that expiry, so callers can tell a locked-out client
        exactly when to retry.
        """
        now = time.monotonic()
        fails = self._prune_auth_failures(ip, now)
        if len(fails) < _AUTH_FAIL_MAX:
            return 0.0
        return max(0.0, _AUTH_LOCKOUT - (now - fails[-1]))

    def _prune_auth_failures(self, ip: str, now: float) -> list[float]:
        """Drop *ip*'s failures older than the window; return the rest.

        An IP with no recent failure loses its entry altogether, so
        the dict stays bounded to IPs that failed recently.
        """
        fails = [
            t for t in self._auth_failures.get(ip, ())
            if now - t <= _AUTH_FAIL_WINDOW
        ]
        if fails:
            self._auth_failures[ip] = fails
        else:
            self._auth_failures.pop(ip, None)
        return fails

    def _record_auth_failure(self, ip: str) -> None:
        """Record a failed authentication attempt from *ip*.

        Also sweeps fully-expired entries for EVERY tracked IP:
        :meth:`_auth_lock_remaining` only prunes the entry of the IP that
        reconnects, so on a public tunnel an attacker rotating source
        addresses would otherwise leave one stale entry per distinct
        IP forever.  The sweep bounds the dict to IPs that failed
        within the last :data:`_AUTH_FAIL_WINDOW` seconds.
        """
        now = time.monotonic()
        for other_ip in list(self._auth_failures):
            self._prune_auth_failures(other_ip, now)
        self._auth_failures.setdefault(ip, []).append(now)

    async def _authenticate_ws(
        self, websocket: ServerConnection,
    ) -> sorcar_api.Auth | None:
        """Authenticate a WebSocket client with the ``auth`` handshake.

        Returns an :class:`~kiss.server.sorcar.Auth` of kind
        ``"local"`` for a loopback peer that presented this daemon's
        local token (see :attr:`local_token`), ``"loopback"`` for a
        browser on this machine itself (a loopback peer with no
        forwarded client IP, admitted without a password), ``"remote"``
        for a client that supplied the configured ``remote_password``
        (its ``link`` flag set when the peer declared itself a daemon
        link), and ``None`` (with the socket closed) on failure.

        When the configured ``remote_password`` is empty, only
        loopback peers may connect at all (:meth:`_process_request`
        refuses non-loopback peers with 403 before the upgrade, and
        the handshake re-checks the peer address).  See also
        :meth:`_setup_server` which refuses to advertise the public
        cloudflared tunnel when no password is configured.

        Transport wrapper: the handshake protocol itself (the
        ``auth`` / ``auth_ok`` / ``auth_required`` / ``auth_locked``
        exchange, the rate-limit refusal, and the
        only-non-empty-guesses-count lockout rule) is part of the
        server API and lives in
        :meth:`kiss.server.sorcar.ServerApi.authenticate`, which
        calls back into this server's :meth:`_client_ip` /
        :meth:`_auth_lock_remaining` / :meth:`_record_auth_failure` /
        :attr:`local_token` primitives.
        """
        return await self._server_api.authenticate(websocket)

    @property
    def local_token(self) -> str:
        """The per-start secret a loopback client presents to be local.

        Published to same-machine clients through the endpoint file
        (:mod:`kiss.agents.sorcar.local_endpoint`), whose 0600 mode restricts
        it to the owning user: only that user's processes can act as
        local clients.
        """
        return self._local_token

    @property
    def local_only(self) -> bool:
        """Whether only token-authenticated local clients are admitted.

        True for the private daemon :meth:`start_private_async` serves:
        it replaces an owner-only channel, so a remote-password login
        (an empty password admits anyone who can reach the port) must
        not open it to other users of the machine.
        """
        return self._local_only

    async def _run_cmd(self, cmd: dict[str, Any]) -> None:
        """Run a backend command in the thread-pool executor."""
        assert self._loop is not None
        await self._loop.run_in_executor(
            None, self._agent_server._handle_command, cmd,
        )

    async def _ws_handler(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket client connection.

        Performs the ``auth`` handshake, then relays messages between
        the client and the ``AgentServer`` command dispatcher.  A
        *local* client (a loopback peer that presented the local
        token: ``daemon_client`` runs) joins the printer's local-client
        set — so task-event copies reach it only for the tabs it
        addressed — and its commands run with ``is_local`` set.  A
        remote browser joins the plain client set.  Another daemon's link
        (``MachineLink``) is never local, whatever token it presented:
        it forwards the Explorer, Source Control and file requests of
        that daemon's browser clients, which want the browser replies
        (an SSH machine's link authenticates through the tunnel, from
        loopback, with the local token).

        Args:
            websocket: The WebSocket server connection.
        """
        auth = await self._authenticate_ws(websocket)
        if auth is None:
            return
        is_local = auth.kind == "local" and not auth.link

        self._printer.add_client(websocket, local=is_local, link=auth.link)
        conn_state: dict[str, Any] = {"conn_id": uuid.uuid4().hex}
        self._printer.bind_conn(conn_state["conn_id"], websocket)
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if not isinstance(cmd, dict):
                    continue
                if not await self._frame_password_accepted(cmd, auth.kind, conn_state):
                    # Every frame of a password-authenticated client
                    # must carry the password in force NOW (it is
                    # re-read from config.json each time, so a
                    # rotated password shuts the old session out at
                    # its next command; the one exception is the
                    # client that asked for the rotation, until it
                    # speaks the new password).  The close makes the
                    # browser reconnect and ask for the password again.
                    logger.warning(
                        "Frame %r from %s carries a wrong password; closing",
                        cmd.get("type", ""), websocket.remote_address,
                    )
                    await websocket.send(json.dumps({
                        "type": "error", "code": "auth_failed",
                        "text": "Wrong password.",
                    }))
                    await websocket.close()
                    break
                try:
                    await self._dispatch_client_command(
                        cmd, websocket, conn_state, is_local,
                    )
                except websockets.exceptions.ConnectionClosed:
                    raise
                except Exception:
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
            if is_local:
                self._printer.unregister_local_tabs(conn_state["conn_id"])
            self._agent_server.drop_connection_state(conn_state["conn_id"])
            self._agent_server.browser_tabs.viewer_gone(conn_state["conn_id"])
            self._agent_server.terminals.viewer_gone(conn_state["conn_id"])
            self._printer.unbind_conn(conn_state["conn_id"])
            self._printer.remove_client(websocket)

    @staticmethod
    async def _frame_password_accepted(
        cmd: dict[str, Any], kind: sorcar_api.AuthKind, conn_state: dict[str, Any],
    ) -> bool:
        """Strip the ``password`` of a frame and, for a ``"remote"`` client, check it.

        The ``password`` key is removed from *cmd* whatever *kind* is,
        so no handler ever sees it.  Only a password-authenticated
        (``"remote"``) client is policed: the stored password is
        re-read from ``config.json`` for every frame, so the one in
        force is always the one ``install.sh`` set last.  Local-token
        clients, machine links and loopback browsers (``"local"`` and
        ``"loopback"``) are always accepted.

        A client that rotates the password itself (Settings > Rotate,
        the ``rotateRemotePassword`` frame) learns the new one from the
        ``remotePasswordRotated`` event, but frames it sent before that
        event reached it (a command in flight, the ping that follows a
        reconnect's queued batch) still carry the one it was admitted
        with.  Those are honoured: ``conn_state["rotated_from"]`` keeps
        that password from the rotate frame until the connection's
        first frame with the password in force, which ends the
        allowance.  It is per connection (gone with the socket) and
        never widens who is admitted: that connection was already in.
        Every other password-authenticated client is shut out at its
        next frame, as before.
        """
        client_pw = cmd.pop("password", "")
        if kind != "remote":
            return True
        if not isinstance(client_pw, str):
            client_pw = ""
        stored = (await asyncio.to_thread(load_config)).get("remote_password", "")
        if password_accepted(client_pw, stored):
            conn_state.pop("rotated_from", None)
            if cmd.get("type") == "rotateRemotePassword":
                conn_state["rotated_from"] = stored
            return True
        rotated_from = conn_state.get("rotated_from", "")
        return bool(rotated_from) and password_accepted(client_pw, rotated_from)

    async def _dispatch_client_command(
        self,
        cmd: dict[str, Any],
        endpoint: ServerConnection,
        conn_state: dict[str, Any],
        is_local: bool,
    ) -> None:
        """Hand one parsed client command to the server's code API.

        The single per-message entry point of :meth:`_ws_handler` for
        remote browsers and local clients alike, so the two kinds of
        peer cannot drift in behaviour.  This method owns NO routing: it wraps the connection's
        transport state into a :class:`kiss.server.sorcar.ApiContext`
        and calls :meth:`kiss.server.sorcar.ServerApi.dispatch`, which
        validates the command against the API catalog, applies the
        per-connection stamping (``connId``, tab registration — see
        the invariant documentation on :class:`ServerApi`), and
        invokes the API method the command's catalog entry names, with
        this server as the backend.

        Args:
            cmd: The parsed JSON command dictionary.
            endpoint: The client's :class:`ServerConnection`, used
                for direct replies.
            conn_state: Per-connection mutable state holding the
                connection's unique ``conn_id``.  Each browser tab
                owns exactly one connection, and the API layer's
                stamping of ``connId`` is what guarantees the
                per-tab autocomplete isolation invariant.
            is_local: Whether the connection authenticated with the
                local token (see :meth:`_authenticate_ws`).
        """
        ctx = sorcar_api.ApiContext(
            endpoint=endpoint,
            conn_state=conn_state,
            is_local=is_local,
        )
        await self._server_api.dispatch(cmd, ctx)

    @staticmethod
    def _bootstrap_events() -> list[dict[str, Any]]:
        """The ``tricksData`` and ``tipsData`` events a ``ready`` answers with.

        Reads ``MY_INJECTION.md``, the bundled promptlets and tips, and
        the opt-out marker, so callers run it off the event loop.
        """
        return [
            {"type": "tricksData", **read_tricks_data()},
            {"type": "tipsData", **tips_data(_read_version())},
        ]

    def _broadcast_to_conn(self, event: dict[str, Any], conn_id: str) -> None:
        """Broadcast *event*, stamped with *conn_id* when non-empty.

        A non-empty ``conn_id`` makes :meth:`WebPrinter.broadcast`
        deliver the event ONLY to the requesting connection (the
        browser tab whose user triggered the command), so siblings do
        not pop a banner; ``""`` broadcasts to all.
        """
        broadcast_to_conn(self._printer, event, conn_id)

    async def _handle_server_reset(self, conn_id: str = "") -> None:
        """Restart the ``kiss-web`` daemon at the user's request.

        Server-side handler for the settings-panel "Server reset"
        button.  Broadcasts an acknowledgement ``notification`` to the
        requesting window (stamped with its ``connId`` so siblings do
        not pop a banner), then schedules a ``SIGTERM`` to this very
        process after a short delay so the notification flushes to the client
        before its socket drops.  The ``SIGTERM`` is caught by
        :meth:`_handle_shutdown_signal`, which runs the
        :meth:`_shutdown_on_sigterm` graceful-shutdown path (stopping
        in-flight agent tasks, then unwinding the ``asyncio.run`` loop
        in :meth:`start` so its cleanup runs).  The process
        then exits and the supervising macOS LaunchAgent (``KeepAlive``)
        / Linux systemd unit (``Restart=always``) respawns a fresh
        ``kiss-web`` that re-adopts the same port and ``cloudflared``
        tunnel — so the public URL is preserved across the reset.

        Args:
            conn_id: Requesting connection id (``""`` to broadcast).
        """
        loop = self._loop
        assert loop is not None
        self._broadcast_to_conn({
            "type": "notification",
            "id": "server-reset-restarting",
            "severity": "info",
            "message": f"Restarting the {PRODUCT_NAME} web server…",
        }, conn_id)
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
            # Shared atomic writer (pid/thread-unique temp + replace):
            # a hand-rolled fixed ``.tmp`` sibling let a concurrent
            # writer truncate the temp inode the first writer had
            # already renamed onto the flag, exposing an empty flag.
            _atomic_write_text(
                flag_path,
                json.dumps(
                    {"requested_at": time.time(), "conn_id": conn_id},
                ),
            )
        except OSError:
            logger.debug(
                "Could not write server-reset pending flag at %s",
                flag_path, exc_info=True,
            )

    def _maybe_schedule_server_reset_complete(self) -> None:
        """Schedule the post-restart broadcast iff a pending flag exists.

        Called once from :meth:`_setup_server` after the WSS
        listeners are bound and the watchdog tasks are armed.  The
        flag file written by :meth:`_write_server_reset_flag` in the
        previous daemon instance is CLAIMED first — atomically renamed
        to a pid-unique sibling, so the toast fires at most once per
        user-initiated reset even if the daemon restarts again before
        the timer runs — then its content is checked to be the JSON
        object the writer produces, and only then is the delayed
        "Server restart complete" broadcast queued.  A marker that
        cannot be claimed (unlinkable, or a directory sitting at the
        path) or that holds anything else is never announced: doing
        so used to re-announce a "completed restart" on every start.
        """
        flag_path = self._server_reset_flag_path()
        if not flag_path.is_file():
            return
        claimed = flag_path.with_name(
            f"{flag_path.name}.claimed-{os.getpid()}-{uuid.uuid4().hex[:8]}",
        )
        try:
            os.replace(flag_path, claimed)
        except OSError:
            logger.debug(
                "Could not claim server-reset pending flag at %s",
                flag_path, exc_info=True,
            )
            return
        try:
            marker = json.loads(claimed.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = None
        finally:
            claimed.unlink(missing_ok=True)
        if not isinstance(marker, dict):
            logger.debug(
                "Ignoring malformed server-reset pending flag at %s",
                flag_path,
            )
            return
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
        preserved across the restart, but every page that was
        disconnected by the SIGTERM benefits from the same
        confirmation.  The stable ``id`` lets the existing page's
        dedup (``data-notification-id`` in ``showNotification``)
        replace any stale "restarting" toast in-place instead of
        stacking a duplicate.
        """
        self._printer.broadcast(
            {
                "type": "notification",
                "id": "server-reset-complete",
                "severity": "info",
                "message": f"{PRODUCT_NAME} web server restart complete.",
            },
        )

    def _trigger_server_reset(self) -> None:
        """Send ``SIGTERM`` to this process to trigger a clean restart.

        Runs as a delayed event-loop callback on the main thread (see
        :meth:`_handle_server_reset`).  Delivering ``SIGTERM`` to the
        daemon's own pid routes through :meth:`_handle_shutdown_signal`
        exactly like an external ``pkill``/supervisor stop, so the
        established graceful-shutdown path runs and the supervisor
        respawns a fresh daemon.  On Windows ``os.kill(pid, SIGTERM)``
        is ``TerminateProcess`` (no handler runs, the agents die
        abruptly), so the handler is invoked directly there.
        """
        logger.warning(
            "Server reset requested: pid=%d sending SIGTERM to self",
            os.getpid(),
        )
        if sys.platform == "win32":  # pragma: no cover — Windows only
            self._handle_shutdown_signal(signal.SIGTERM)
            return
        os.kill(os.getpid(), signal.SIGTERM)

    async def _handle_run_update(self, conn_id: str = "") -> None:
        """Run ``~/.kiss/kiss_ai/install.sh`` to update KISS Sorcar.

        The web server locates the installer via
        :func:`_find_install_script` and runs it as a detached
        subprocess (output appended to ``~/.kiss/update.log``) since a
        remote browser has no terminal.  When the clone's
        ``install.sh`` is missing, the curl bootstrap
        (:func:`_bootstrap_install_url`) runs instead, which recreates
        ``~/.kiss/kiss_ai`` and installs from it.

        The acknowledgement ``notice`` / ``error`` events are stamped
        with the requesting connection's ``connId`` (when non-empty)
        so they reach ONLY the browser tab whose user clicked "Update":
        clicking Update in one tab must not pop a banner in every
        sibling tab.

        The installer's exit is watched by :meth:`_watch_update_exit`
        so a failure — above all ``install.sh`` losing its
        cross-process update lock to another installer and exiting 1
        with ``another KISS update is already running (pid N)`` — is
        reported to the same window instead of leaving it believing
        an update is under way.

        Args:
            conn_id: Requesting connection id (``""`` to broadcast).
        """
        loop = self._loop
        assert loop is not None
        if self._update_in_progress():
            # Single-flight guard (F4-13): two windows clicking
            # "Update" concurrently must not launch two installers
            # that fetch/reset/overwrite the same tree in parallel.
            self._broadcast_to_conn({
                "type": "notice",
                "text": (
                    f"A {PRODUCT_NAME} update is already running… "
                    f"(output: {self._update_log_path})"
                ),
            }, conn_id)
            return
        self._update_starting = True
        # From here on the daemon is about to be restarted by the
        # installer: refuse NEW task submits (running ones are the
        # user's explicit choice).  The idle poller has already raised
        # the barrier under the registry lock together with its "no
        # active tasks" verdict; a direct click raises it here.  Every
        # path on which no installer ends up running lowers it again.
        self._set_update_barrier(True)
        # A direct "Update" supersedes an armed "Update when idle": the
        # idle poller must not launch a second installer later.  The
        # poller itself disarms before calling here, so this never
        # cancels the running task.
        spawned = None
        try:
            if await self._cancel_update_when_idle():
                await self._broadcast_update_available()
            # When the clone (or its install.sh) is missing —
            # ~/.kiss/kiss_ai was deleted — fall back to the public
            # curl bootstrap,
            # which recreates the clone and hands over to its
            # install.sh, instead of refusing with "install.sh not
            # found".
            script = await loop.run_in_executor(
                None, _find_install_script, self._install_root,
            )
            self._broadcast_to_conn({
                "type": "notice",
                "text": (
                    f"An update of {PRODUCT_NAME} is getting installed… "
                    f"(output: {self._update_log_path})"
                ),
            }, conn_id)
            spawned = await loop.run_in_executor(
                None, self._spawn_update_script, script, conn_id,
            )
        finally:
            # The spawn is over (a running installer now holds the
            # single-flight guard through ``_update_proc``).
            self._update_starting = False
            if spawned is None:
                # No installer owns the barrier (spawn failure, or a
                # raise/cancellation before the spawn): admit runs
                # again and let a later Update click start over.
                self._set_update_barrier(False)
        if spawned is None:
            return
        self._update_watch_task = asyncio.create_task(
            self._watch_update_exit(*spawned, conn_id),
        )

    def _set_update_barrier(self, up: bool) -> None:
        """Raise or lower the run-admission barrier of the self-update.

        Writes ``AgentServer._update_installing`` under
        :data:`agent_state.STATE_LOCK` — the lock ``_cmd_run`` holds
        while admitting a run — so a submit observes either the barrier
        or its absence, never a torn state.  Cheap enough to call from
        the event loop (the lock is held only for short critical
        sections).

        Args:
            up: ``True`` to refuse new task submits, ``False`` to
                admit them again.
        """
        with agent_state.STATE_LOCK:
            self._agent_server._update_installing = up

    def _arm_update_barrier_if_idle(self) -> bool:
        """Raise the run-admission barrier when no task is in flight.

        The idle verdict (:func:`_snapshot_active_tabs`) and the arming
        happen in ONE :data:`agent_state.STATE_LOCK` critical section
        — the lock ``_cmd_run`` holds while it admits a run — so a
        ``run`` submitted between the poller's "no active tasks"
        observation and the installer spawn is either counted as
        active (and defers the update) or refused by the barrier.
        The armed flag is read in the same section — and
        :meth:`_cancel_update_when_idle` clears it under the same lock
        — so a Cancel that lands while this thread waits for the lock
        is seen here and nothing is armed for an update nobody will
        run (the cancelled poller has already lowered the barrier and
        no exit watcher would ever lower it again).
        Blocks on the lock: call it off the event loop.

        Returns:
            ``True`` when the barrier was raised (idle), ``False`` when
            a task is still live and the poller must wait, or the idle
            update was cancelled meanwhile.
        """
        with agent_state.STATE_LOCK:
            if not self._update_when_idle_armed or _snapshot_active_tabs():
                return False
            self._agent_server._update_installing = True
            return True

    def _spawn_update_script(
        self, script: Path | None, conn_id: str = "",
    ) -> tuple[subprocess.Popen[bytes], int] | None:
        """Start the updater detached, logging to the update log.

        With *script* set, runs the clone's committed
        ``scripts/install.sh`` when present — it synchronizes the
        checkout with origin under the update lock before handing over
        to the root ``install.sh``, which itself never touches git —
        and falls back to running *script* directly for clones that
        predate the bootstrap.  With ``None`` (no
        ``~/.kiss/kiss_ai/install.sh`` on this machine), runs the curl
        bootstrap from :func:`_bootstrap_install_url`, which clones the
        repo into ``~/.kiss/kiss_ai`` and hands over to its
        ``install.sh`` — so Update works even where KISS Sorcar was
        never curl-installed.

        Runs in the executor so file I/O and process spawn never block
        the event loop.  ``start_new_session=True`` keeps the updater
        out of the daemon's process group when ``install.sh`` restarts
        this very daemon; under systemd that is not enough (a stop
        signals the whole control group), so the script's own
        ``kiss-service-cgroup-escape`` block moves it into a transient
        scope first.
        ``--non-interactive`` / ``KISS_NONINTERACTIVE=1`` make the
        script answer its ``[Y/n]`` questions with their
        defaults (it would anyway, having no terminal to ask on), and
        ``stdin=DEVNULL`` detaches it from the daemon's stdin.
        Failures are emitted as ``error`` events instead of raised,
        stamped with the requesting connection's ``connId`` (when
        non-empty) so only the window that clicked "Update" renders the
        error banner.

        Args:
            script: Absolute path of the ``install.sh`` to execute, or
                ``None`` to run the curl bootstrap instead.
            conn_id: Requesting connection id (``""`` to broadcast).

        Returns:
            The started process and the update log's size at that
            moment (the start of this run's output), or ``None`` when
            the spawn failed.
        """
        # Git for Windows keeps bash.exe off PATH; a bare "bash" there
        # fails with WinError 2 (reported through the OSError path below
        # when no bash is installed at all).
        bash = find_bash() or "bash"
        # Pin KISS_HOME to the home THIS daemon resolved (the brand's
        # default unless the environment overrides it): install.sh
        # runs the post-install hooks of,
        # copies MODEL_INFO.json into and writes the reload marker under
        # $KISS_HOME, and defaults to the stock ``~/.kiss`` otherwise —
        # which a white-label brand's daemon never reads.
        env = dict(os.environ)
        env["KISS_HOME"] = str(kiss_home())
        if script is not None:
            bootstrap = script.parent / "scripts" / "install.sh"
            # os.path.isfile, not Path.is_file: an unreadable ``scripts``
            # directory must degrade to the root script below, and on
            # Python 3.13 ``Path.is_file`` re-raises the PermissionError
            # (only ENOENT/ENOTDIR/EBADF/ELOOP are swallowed) whereas
            # ``os.path.isfile`` returns False for every OSError.
            if os.path.isfile(bootstrap):
                # scripts/install.sh (the curl bootstrap committed in the
                # clone) synchronizes the checkout with origin under the
                # cross-process update lock and hands over to the root
                # install.sh — which itself never touches git, so running
                # the root script directly would rebuild the stale
                # checkout as-is.  ``KISS_NONINTERACTIVE=1`` is what
                # both installers read in place of ``--non-interactive``.
                argv = [bash, str(bootstrap)]
                cwd = str(script.parent)
                env["KISS_NONINTERACTIVE"] = "1"
            else:
                argv = [bash, str(script), "--non-interactive"]
                cwd = str(script.parent)
        else:
            argv = [
                bash, "-c",
                'set -o pipefail; '
                'curl -fsSL "$KISS_BOOTSTRAP_URL" | bash',
            ]
            cwd = str(Path.home())
            env["KISS_BOOTSTRAP_URL"] = _bootstrap_install_url()
            env["KISS_NONINTERACTIVE"] = "1"
        spawned = self._spawn_detached(
            argv, cwd, env, self._update_log_path, conn_id,
            f"{PRODUCT_NAME} update",
        )
        if spawned is not None:
            self._update_proc = spawned[0]
        return spawned

    def _spawn_detached(
        self,
        argv: list[str],
        cwd: str,
        env: dict[str, str] | None,
        log_path: Path,
        conn_id: str,
        what: str,
    ) -> tuple[subprocess.Popen[bytes], int] | None:
        """Start *argv* detached from the daemon, appending its output to *log_path*.

        Shared by the self-updater and the model-catalog updater: both
        must survive a restart of this daemon (``start_new_session=True``,
        ``stdin=DEVNULL``) and both report a spawn failure as an
        ``error`` event stamped with the requesting connection instead
        of raising.  Blocking (file I/O, process spawn): run it in the
        executor.

        Args:
            argv: The command line.
            cwd: Working directory of the child.
            env: Environment of the child (``None`` inherits the daemon's).
            log_path: Append-only log the child's stdout/stderr go to.
            conn_id: Requesting connection id (``""`` to broadcast).
            what: What is being started, for the error text.

        Returns:
            The started process and the log's size at that moment (the
            start of this run's output), or ``None`` when the spawn
            failed.
        """
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "ab") as log:
                # Before the spawn: the child shares the open file
                # description, so its first output would move ``tell``.
                log_offset = log.tell()
                proc = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                return proc, log_offset
        except OSError as exc:
            self._broadcast_to_conn({
                "type": "error",
                "text": f"Failed to start {what}: {exc}",
            }, conn_id)
            return None

    async def _watch_update_exit(
        self, proc: subprocess.Popen[bytes], log_offset: int, conn_id: str,
    ) -> None:
        """Report a failed installer to the window that started it.

        Polls the detached installer (no executor thread is tied up for
        the minutes an install takes, and the daemon it may restart
        never waits on it) and, on a non-zero exit, sends the
        requesting connection an ``error``: the installer's own
        refusal line when this run's slice of the update log holds one
        (``install.sh`` lost the cross-process update lock to another
        installer), otherwise a generic failure pointing at the log.
        A clean exit reports nothing more.  Either way the installer
        is gone once this returns, so the run-admission barrier raised
        for it is lowered: a daemon the installer did not restart (a
        failed or no-op update) must accept tasks again.

        Args:
            proc: The installer started by :meth:`_spawn_update_script`.
            log_offset: Size of the update log when *proc* started,
                i.e. where this run's output begins.
            conn_id: Requesting connection id (``""`` to broadcast).
        """
        while proc.poll() is None:
            await asyncio.sleep(0.2)
        self._set_update_barrier(False)
        if proc.returncode == 0:
            return
        output = await asyncio.to_thread(
            _read_log_from, self._update_log_path, log_offset,
        )
        text = f"{PRODUCT_NAME} update failed (exit {proc.returncode}), see {self._update_log_path}"
        for line in output.decode("utf-8", errors="replace").splitlines():
            if "another KISS update is already running" in line:
                text = f"{PRODUCT_NAME} update: {line.strip()}"
                break
        self._broadcast_to_conn({"type": "error", "text": text}, conn_id)

    async def _handle_update_models(self, conn_id: str = "") -> None:
        """Refresh ``~/.kiss/MODEL_INFO.json`` via ``update_models.py``.

        Services the settings panel's "Update Models" button: runs
        :data:`_update_models_argv` —
        ``kiss.scripts.update_models --model-info
        $KISS_HOME/MODEL_INFO.json`` in this daemon's interpreter — as a
        detached subprocess whose output is appended to
        ``~/.kiss/update_models.log``.  The script seeds a missing
        target from the bundled catalog, fetches the latest vendor
        pricing/context tables, capability-tests new models, and
        rewrites the user-local catalog atomically.  Every process
        started AFTER the rewrite reads the refreshed copy through
        ``kiss.core.models.model_info`` (on installed copies; a daemon
        running from a git checkout keeps reading the checkout's bundled
        catalog); this already-running daemon holds the catalog it
        imported at startup until it is restarted.

        Mirrors :meth:`_handle_run_update`: a single-flight guard keeps
        two clicks from racing two updaters over the same file, the
        acknowledgement ``notice`` / ``error`` events are stamped with
        the requesting connection's ``connId`` so they reach only the
        clicking window, and :meth:`_watch_update_models_exit` reports
        the subprocess's exit.

        Args:
            conn_id: Requesting connection id (``""`` to broadcast).
        """
        loop = self._loop
        assert loop is not None
        if self._update_models_starting or (
            self._update_models_proc is not None
            and self._update_models_proc.poll() is None
        ):
            self._broadcast_to_conn({
                "type": "notice",
                "text": (
                    "A model catalog update is already running… "
                    f"(output: {self._update_models_log_path})"
                ),
            }, conn_id)
            return
        self._update_models_starting = True
        self._broadcast_to_conn({
            "type": "notice",
            "text": (
                f"Updating the model catalog {self._update_models_argv[-1]}… "
                f"(output: {self._update_models_log_path})"
            ),
        }, conn_id)
        spawned = await loop.run_in_executor(
            None, self._spawn_update_models, conn_id,
        )
        if spawned is not None:
            self._update_models_watch_task = asyncio.create_task(
                self._watch_update_models_exit(spawned, conn_id),
            )

    def _spawn_update_models(
        self, conn_id: str = "",
    ) -> subprocess.Popen[bytes] | None:
        """Start the model-catalog updater detached, logging its output.

        Runs in the executor so file I/O and the process spawn never
        block the event loop.  ``start_new_session=True`` and
        ``stdin=DEVNULL`` detach the updater from the daemon: the
        catalog refresh (which live-probes new models) may outlive a
        daemon restart.  Spawn failures are emitted as ``error`` events
        instead of raised, stamped with the requesting connection's
        ``connId`` so only the clicking window renders the banner.

        Args:
            conn_id: Requesting connection id (``""`` to broadcast).

        Returns:
            The started process, or ``None`` when the spawn failed.
        """
        try:
            spawned = self._spawn_detached(
                self._update_models_argv, str(Path.home()), None,
                self._update_models_log_path, conn_id,
                "the model catalog update",
            )
            if spawned is None:
                return None
            self._update_models_proc = spawned[0]
            return self._update_models_proc
        finally:
            self._update_models_starting = False

    async def _watch_update_models_exit(
        self, proc: subprocess.Popen[bytes], conn_id: str,
    ) -> None:
        """Report the model-catalog updater's exit to the clicking window.

        Polls the detached updater without tying up an executor thread.
        A clean exit gets a completion ``notice`` (unlike the installer
        watched by :meth:`_watch_update_exit`, the catalog refresh gives
        no other feedback — no window reload, no terminal), a non-zero
        exit an ``error`` pointing at the log.

        Args:
            proc: The updater started by :meth:`_spawn_update_models`.
            conn_id: Requesting connection id (``""`` to broadcast).
        """
        while proc.poll() is None:
            await asyncio.sleep(0.2)
        if proc.returncode == 0:
            self._broadcast_to_conn({
                "type": "notice",
                "text": (
                    "Model catalog update complete "
                    f"(output: {self._update_models_log_path}). New tasks "
                    "pick up the refreshed catalog; use Reset Server to "
                    "refresh this window's model list."
                ),
            }, conn_id)
            return
        self._broadcast_to_conn({
            "type": "error",
            "text": (
                f"Model catalog update failed (exit {proc.returncode}), "
                f"see {self._update_models_log_path}"
            ),
        }, conn_id)

    def _identify_voice_speaker(self, pcm: bytes) -> int | None:
        """Return the stable speaker number for an utterance's PCM.

        Runs on an executor thread.  The Vosk speaker-identification
        model is built lazily on first use (it may need a one-time
        download); any failure — download, model load, recognition —
        latches speaker identification off for the daemon's lifetime
        and degrades to ``None`` so voice dictation keeps working
        without speaker numbers.

        Args:
            pcm: Raw 16kHz mono s16le PCM of the utterance.

        Returns:
            The speaker number (1, 2, ...) or ``None`` on failure.
        """
        with self._voice_speaker_lock:
            if self._voice_speaker_broken:
                return None
            try:
                if self._voice_speaker_identifier is None:
                    self._voice_speaker_identifier = SpeakerIdentifier(
                        default_models_dir(),
                    )
                return self._voice_speaker_identifier.speaker_of(pcm)
            except Exception:
                self._voice_speaker_broken = True
                logger.exception("voice speaker identification failed")
                return None

    async def _handle_voice_transcribe(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Translate a remote-web client's post-wake speech to English.

        voice.js cannot call gpt-audio itself (the API key lives on
        this machine), so after the in-page wake-word detector hears
        "Hey Sorcar" it captures the utterance that follows and ships
        it here as ``{type: 'voiceTranscribe', audio: <base64 16kHz
        mono s16le PCM>}``.  The audio is translated into English by
        the KISS transcription agent (:func:`transcribe_pcm`) — which
        also reports the language that was spoken — the speaker is
        identified locally (best effort), and the client gets back
        ``{type: 'voiceSpeech', text, speaker, language}``, which
        voice.js inserts and submits as the dictated task.  An
        empty/undecodable audio
        payload or a failed translation replies with an empty ``text``
        so the page can clear its transcribing indicator.

        A client that prepended the wake word's own audio to the
        capture marks the message ``wakePrefixed: true``; the
        transcript must then re-confirm the wake word (the dual wake
        check of :func:`transcribe_pcm`) or the utterance is rejected
        with an empty ``text``.  Such a client also sends
        ``wakeSamples`` — the sample count of the prepended audio —
        so speaker identification runs on the capture alone (the
        pre-wake ring may carry another voice).  Messages without the
        flag (older clients, plain dictation) keep the lenient
        behavior.

        Args:
            cmd: The parsed ``voiceTranscribe`` command.
            endpoint: The client connection to reply to.
        """
        raw = cmd.get("audio", "")
        pcm = b""
        if isinstance(raw, str) and 0 < len(raw) <= _MAX_VOICE_AUDIO_B64:
            try:
                pcm = base64.b64decode(raw, validate=True)
            except (binascii.Error, ValueError):
                logger.warning("voiceTranscribe with undecodable audio")
        text = ""
        language: str | None = None
        speaker: int | None = None
        if pcm:
            assert self._loop is not None
            expect_wake_prefix = bool(cmd.get("wakePrefixed"))
            try:
                result = await self._loop.run_in_executor(
                    None,
                    transcribe_pcm,
                    pcm,
                    DEFAULT_AUDIO_MODEL,
                    expect_wake_prefix,
                )
                text = result["text"]
                language = result["language"]
            except Exception:
                # A failed translation must still reply with empty
                # text so the client can clear its transcribing
                # spinner (F4-15).
                logger.warning(
                    "voiceTranscribe transcription failed", exc_info=True,
                )
                text = ""
                language = None
            if text:
                # Speaker identification uses the capture alone: the
                # client's wakeSamples marks where the prepended
                # pre-wake ring (which may carry another voice) ends.
                wake_samples = cmd.get("wakeSamples")
                speaker_pcm = pcm
                if (
                    isinstance(wake_samples, int)
                    and not isinstance(wake_samples, bool)
                    and 0 < wake_samples < len(pcm) // 2
                ):
                    speaker_pcm = pcm[2 * wake_samples:]
                speaker = await self._loop.run_in_executor(
                    None, self._identify_voice_speaker, speaker_pcm,
                )
        await self._endpoint_send(
            endpoint,
            json.dumps(
                {
                    "type": "voiceSpeech",
                    "text": text,
                    "speaker": speaker,
                    "language": language,
                },
            ),
        )

    @staticmethod
    def _cmd_str(cmd: dict[str, Any], key: str) -> str:
        """Return ``cmd[key]`` when it is a string, else ``""``.

        Client commands are untrusted JSON: every file-oriented
        handler (``openFile``, ``shareChat``, ``shareChatTasks``,
        ``checkPaths``, ``ready``) must blank a non-string ``tabId`` /
        ``title`` / ``chatId`` rather than let it flow into path or
        reply construction.  One helper instead of one copy per
        handler.
        """
        value = cmd.get(key, "")
        return value if isinstance(value, str) else ""

    def _cmd_work_dir(self, cmd: dict[str, Any]) -> str:
        """Return the command's ``workDir``, else the daemon work dir.

        An explicit ``workDir`` wins; a missing, empty or non-string
        value falls back to the backend's global working directory and
        then this server's own.  Shared by the file handlers so
        relative paths resolve identically everywhere.
        """
        return (
            self._cmd_str(cmd, "workDir")
            or self._agent_server.work_dir
            or self.work_dir
        )

    def _resolve_tab_file(
        self,
        raw_path: str,
        work_dir: str,
        tab_id: str,
        file_only: bool = False,
    ) -> Path | None:
        """Resolve *raw_path* for a tab, trying its pending worktree too.

        The one resolution ``openFile``, ``checkPaths`` and a path-only
        ``submit`` use: the path is resolved against *work_dir* first;
        when that names no file and the tab
        has a pending worktree (a finished worktree task whose branch is
        not merged yet), the same relative path is tried inside the
        worktree directory — that is where the task's committed
        artifacts live until the merge.

        Args:
            raw_path: The path as printed in the transcript (may be
                relative, absolute, or ``~``-prefixed).
            work_dir: The tab's working directory.
            tab_id: The requesting client's tab id (may be ``""``).
            file_only: Accept regular files only.  A directory
                candidate is skipped rather than returned, so it can
                neither satisfy the lookup nor shadow a pending-worktree
                file at the same relative path.

        Returns:
            The resolved path when it names an existing regular file
            (or, unless *file_only*, a directory), otherwise ``None``.
        """
        try:
            path = Path(os.path.expanduser(raw_path))
            if path.is_absolute() or not work_dir:
                candidates = [path]
            else:
                candidates = [Path(work_dir) / path]
                wt_dir = (
                    self._printer.worktree_dir_for_tab(tab_id)
                    if tab_id
                    else ""
                )
                if wt_dir and wt_dir != work_dir:
                    candidates.append(Path(wt_dir) / path)
            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved.is_file() or (
                    not file_only and resolved.is_dir()
                ):
                    return resolved
        except (OSError, ValueError):
            # ValueError: a path with an embedded NUL byte.
            return None
        return None

    async def _handle_open_file(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Serve the file an ``openFile`` names to its client.

        Handles the ``openFile`` command sent by ``media/main.js`` when
        the user clicks a file link (``span[data-path]``) in a chat
        page.  The path is resolved by :meth:`_resolve_tab_file` (``~``
        expansion, the command's ``workDir``, then the tab's pending
        worktree) and the content is sent directly to the requesting
        *endpoint* via :meth:`_endpoint_send` — never broadcast — as a
        single ``fileContent`` JSON object::

            {"type": "fileContent", "path": <resolved abs path>,
             "name": <basename>, "tabId": <echo of cmd tabId>,
             "line": <echo of cmd line, when a positive int>,
             "content": <utf-8 text>,
             "version": "<st_mtime_ns>:<st_size>"}   # on success
            {"type": "fileContent", "path": ..., "name": ..., "tabId": ...,
             "binary": true, "mime": "application/pdf" | "image/...",
             "size": <bytes>, "base64": <bytes>}   # PDF / image viewer
            {"type": "fileContent", "path": ..., "name": ...,
             "tabId": ..., "error": <message>}  # on failure

        A PDF or image (``_INLINE_BINARY_MIMES``, up to
        ``_OPEN_BINARY_MAX_BYTES``) is served base64-encoded so the
        client can show it in a viewer tab instead of refusing it as a
        binary file.

        A ``path:NN`` link's line number arrives as the command's
        ``line`` field; echoing it lets ``media/main.js`` jump the
        opened content tab to that line.  ``version`` stamps the file
        as read (:func:`_file_version`); the client's ``saveFile``
        sends it back so :meth:`_handle_save_file` can detect a file
        that changed on disk while it was open (directory listings
        carry no ``version``: they are never saved).

        Relative paths are resolved against the command's ``workDir``
        (stamped per-connection by
        :meth:`kiss.server.sorcar.ServerApi.dispatch`) and
        fall back to the daemon work dir.  Missing files, unreadable
        files, files larger than :data:`_OPEN_FILE_MAX_BYTES`, and
        binary files (NUL byte in the first 8 KiB) produce an ``error``
        reply instead of content.  A path naming a directory replies
        with a plain-text listing (:func:`_directory_listing_text`) as
        the ``content``.

        Args:
            cmd: The parsed ``openFile`` command (``path``, optional
                ``workDir``, ``tabId``, ``line``).
            endpoint: The requesting connection.
        """
        raw_path = self._cmd_str(cmd, "path")
        if not raw_path:
            return
        work_dir = self._cmd_work_dir(cmd)
        tab_id = self._cmd_str(cmd, "tabId")
        line = cmd.get("line")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            line = 0

        def _read_file() -> dict[str, Any]:
            reply: dict[str, Any] = {
                "type": "fileContent",
                "path": raw_path,
                "name": Path(raw_path).name,
                "tabId": tab_id,
            }
            if line:
                reply["line"] = line
            if cmd.get("background") is True:
                # The Explorer's "Open to the Side": the client opens
                # the tab without switching to it.
                reply["background"] = True
            try:
                path = self._resolve_tab_file(raw_path, work_dir, tab_id)
                if path is None:
                    reply["error"] = f"File not found: {raw_path}"
                    return reply
                if path.is_dir():
                    # A clicked directory link: reply with a plain-text
                    # listing instead of file content.  isDirectory
                    # tells the client to render the listing as plain
                    # text even when the directory NAME looks like a
                    # markdown/HTML file (foo.md, foo.html, ...).
                    reply["path"] = str(path)
                    reply["name"] = path.name or str(path)
                    reply["isDirectory"] = True
                    reply["content"] = _directory_listing_text(path)
                    return reply
                st = path.stat()
                inline_mime = _inline_binary_mime(path)
                if inline_mime:
                    # A PDF or an image: the client shows the bytes in
                    # a viewer tab (blob: URL) instead of an editor.
                    if st.st_size > _OPEN_BINARY_MAX_BYTES:
                        reply["error"] = (
                            f"File too large to display: {raw_path}"
                        )
                        return reply
                    reply["path"] = str(path)
                    reply["name"] = path.name
                    reply["binary"] = True
                    reply["mime"] = inline_mime
                    reply["size"] = st.st_size
                    reply["base64"] = base64.b64encode(
                        path.read_bytes()
                    ).decode("ascii")
                    return reply
                if st.st_size > _OPEN_FILE_MAX_BYTES:
                    reply["error"] = f"File too large to display: {raw_path}"
                    return reply
                data = path.read_bytes()
                if b"\0" in data[:8192]:
                    reply["error"] = f"Cannot display binary file: {raw_path}"
                    return reply
                reply["path"] = str(path)
                reply["name"] = path.name
                try:
                    reply["content"] = data.decode("utf-8")
                except UnicodeDecodeError:
                    # Not valid UTF-8 (Latin-1, a stray byte, ...): show
                    # it with replacement characters but WITHOUT a
                    # version stamp, so the client keeps its read-only
                    # viewer — saving the U+FFFDs back would corrupt
                    # the bytes the decoder could not represent.
                    reply["content"] = data.decode("utf-8", errors="replace")
                    return reply
                # The client hands this back with its saveFile command,
                # so _handle_save_file can tell that the file changed on
                # disk while it was open in the editor.  Its presence is
                # also what makes the client's editor editable.
                reply["version"] = _file_version(st)
            except OSError as exc:
                reply["error"] = f"Failed to read {raw_path}: {exc}"
            return reply

        reply = await asyncio.to_thread(_read_file)
        await self._reply_direct(endpoint, reply, "openFile")

    async def _handle_save_file(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Write a remote-web client's editor text back to its file.

        Handles the ``saveFile`` command sent by ``media/main.js`` when
        the user saves an editable content tab (Ctrl/Cmd+S or its Save
        button).  The reply is a single ``fileSaved`` JSON object sent
        directly to the requesting *endpoint* — never broadcast —
        with the shape::

            {"type": "fileSaved", "ok": true,
             "path": <resolved abs path>, "name": <basename>,
             "tabId": <echo>, "token": <echo>,
             "version": "<st_mtime_ns>:<st_size>" after the write}  # ok
            {"type": "fileSaved", "ok": false, "path": ..., "name": ...,
             "tabId": ..., "token": ..., "error": <message>,
             "conflict": true}   # conflict=true only for a stale version

        The path is resolved exactly like :meth:`_handle_open_file`
        resolves it and MUST name an existing regular file: the editor
        only ever shows files the daemon served, so a path that names
        nothing (or a directory) is refused rather than created.  The
        content must be a string no larger than
        :data:`_OPEN_FILE_MAX_BYTES` (the display cap — anything bigger
        could not have been opened).  When the command carries the
        ``version`` the ``fileContent`` reply reported, a file whose
        current :func:`_file_version` no longer matches was changed by
        someone else while it was open (the agent, a shell, another
        client):
        the write is refused with ``conflict`` set unless ``force`` is
        true, so the client can offer the user an explicit overwrite.
        Check and write happen under :data:`_SAVE_FILE_LOCK`, so of two
        clients of THIS daemon racing with the same stale stamp exactly
        one wins; a sibling process writing the file concurrently is
        outside that guarantee (the same as any editor's).
        The file is replaced atomically (pid-unique temp +
        ``Path.replace``, see :func:`_atomic_publish`) with its
        permission bits preserved, and written byte-for-byte as UTF-8
        without newline translation so CRLF files stay CRLF.

        Args:
            cmd: The parsed ``saveFile`` command (``path``,
                ``content``, optional ``workDir``, ``tabId``,
                ``token``, ``version``, ``force``).
            endpoint: The requesting WSS connection.
        """
        raw_path = self._cmd_str(cmd, "path")
        if not raw_path:
            return
        content = cmd.get("content")
        work_dir = self._cmd_work_dir(cmd)
        tab_id = self._cmd_str(cmd, "tabId")
        token = self._cmd_str(cmd, "token")
        expected_version = self._cmd_str(cmd, "version")
        force = cmd.get("force") is True

        def _write_file() -> dict[str, Any]:
            reply: dict[str, Any] = {
                "type": "fileSaved",
                "ok": False,
                "path": raw_path,
                "name": Path(raw_path).name,
                "tabId": tab_id,
                "token": token,
            }
            if not isinstance(content, str):
                reply["error"] = "Nothing to save: the content is not text"
                return reply
            try:
                # JSON may legally carry a lone surrogate ("\ud800"),
                # which has no UTF-8 form: refuse it instead of raising
                # past the reply.
                data = content.encode("utf-8")
            except UnicodeEncodeError:
                reply["error"] = "Nothing to save: the content is not valid text"
                return reply
            if len(data) > _OPEN_FILE_MAX_BYTES:
                reply["error"] = f"File too large to save: {raw_path}"
                return reply
            try:
                path = self._resolve_tab_file(raw_path, work_dir, tab_id)
                if path is None:
                    reply["error"] = f"File not found: {raw_path}"
                    return reply
                reply["path"] = str(path)
                reply["name"] = path.name
                if path.is_dir():
                    reply["error"] = f"Cannot save a directory: {raw_path}"
                    return reply
                # The version check and the publish must be one step:
                # two clients saving the same file with the same stale
                # stamp would otherwise both pass the check and the
                # second silently overwrite the first.
                with _SAVE_FILE_LOCK:
                    st = path.stat()
                    if (
                        expected_version
                        and not force
                        and _file_version(st) != expected_version
                    ):
                        reply["error"] = (
                            f"{path.name} changed on disk since it was opened"
                        )
                        reply["conflict"] = True
                        return reply
                    _atomic_publish(
                        path, partial(_write_bytes_with_mode, data=data, st=st),
                    )
                    reply["ok"] = True
                    reply["version"] = _file_version(path.stat())
            except OSError as exc:
                reply["error"] = f"Failed to save {raw_path}: {exc}"
            return reply

        reply = await asyncio.to_thread(_write_file)
        await self._reply_direct(endpoint, reply, "saveFile")

    async def _handle_share_chat(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Save a chat transcript as a standalone HTML page and reply.

        Handles the ``shareChat`` command sent by ``media/main.js``
        when the user clicks the share button next to the mic button:
        the page serialized the highlighted tab's task panel
        and every event panel of its transcript, and this handler
        wraps them into a self-contained page
        (:func:`_build_share_page`) written to
        ``<workDir>/reports/chat-<title-slug>-<chatId>.html`` (see
        :func:`_share_page_filename`).  The reply is a single
        ``share_done`` JSON object sent directly to the requesting
        *endpoint* — never broadcast — with the shape::

            {"type": "share_done", "tabId": <echo>, "ok": true,
             "path": <abs path of the written page>}   # on success
            {"type": "share_done", "tabId": <echo>, "ok": false,
             "error": <message>}                       # on failure

        The chat id is sanitized to a filename-safe token; the target
        directory is created when missing.  ``workDir`` falls back to
        the daemon work dir exactly like :meth:`_handle_open_file`.

        Args:
            cmd: The parsed ``shareChat`` command (``chatId``,
                ``html``, optional ``title``, ``workDir``, ``tabId``).
            endpoint: The requesting connection (local or remote).
        """
        tab_id = self._cmd_str(cmd, "tabId")
        chat_id = cmd.get("chatId", "")
        body_html = cmd.get("html", "")
        title = self._cmd_str(cmd, "title")
        work_dir = self._cmd_work_dir(cmd)

        def _write_page() -> dict[str, Any]:
            reply: dict[str, Any] = {
                "type": "share_done",
                "tabId": tab_id,
                "ok": False,
            }
            if not isinstance(chat_id, str) or not chat_id.strip():
                reply["error"] = "Missing chat id"
                return reply
            if not isinstance(body_html, str) or not body_html.strip():
                reply["error"] = "Nothing to share: the chat is empty"
                return reply
            try:
                page = _build_share_page(title, body_html)
                out_path = (
                    Path(work_dir).expanduser()
                    / "reports"
                    / _share_page_filename(title, chat_id)
                )
                # Atomic: a reader with the previous share of this chat
                # open, or a concurrent share of the same chat from
                # another client, must never observe a torn page.
                _atomic_write_text(out_path, page)
                reply["ok"] = True
                reply["path"] = str(out_path)
            except OSError as exc:
                reply["error"] = f"Failed to write the chat page: {exc}"
            return reply

        reply = await asyncio.to_thread(_write_page)
        await self._reply_direct(endpoint, reply, "shareChat")

    async def _handle_share_chat_tasks(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Send a chat's every persisted task transcript for a share.

        Handles the ``shareChatTasks`` command sent by
        ``media/main.js`` when the user clicks the share button: the
        page needs the transcripts of ALL of the chat's tasks —
        after a reload its DOM holds only the one task the
        session replay repainted — so this returns every non-sub-agent
        ``task_history`` row of the chat, oldest first, straight from
        the history database
        (:func:`~kiss.agents.sorcar.persistence._load_all_chat_events_by_chat_id`).
        The reply is a single ``share_tasks`` JSON object sent
        directly to the requesting *endpoint* — never broadcast —
        with the shape::

            {"type": "share_tasks", "tabId": <echo>, "chatId": <echo>,
             "tasks": [{"task": <str>, "task_id": <str>,
                        "events": [<event>, ...],
                        "subagents": [{"task": <str>, "task_id": <str>,
                                       "parent_task_id": <str>,
                                       "events": [...]}, ...]}, ...],
             "truncated": <bool>}

        Each task's ``subagents`` list carries the transcripts of
        every sub-agent the task fanned out (recursively, parents
        before children — see :func:`_share_subagent_entries`), so the
        shared page can open and close them as tabs exactly like the
        live chat page does.

        An optional ``taskId`` narrows the reply to that ONE task (and
        its sub-agents): a sub-agent tab's share takes this path — its
        row is deliberately absent from the chat's task list, but its
        own fan-outs must still reach its shared page.

        A frame is capped at ``_MAX_LINE_BYTES`` and the connection
        dropped on overflow, so when the tasks do not fit
        ``_SHARE_TASKS_MAX_REPLY_BYTES`` the
        OLDEST ones are left out and ``truncated`` is set — the newest
        transcripts are the ones the export cannot redraw from its own
        DOM.  An unknown or empty chat id yields an empty task list:
        the page then exports just what is on screen.  A history
        database failure yields an ``error`` string in the reply, so
        the share click never silently stalls.

        Args:
            cmd: The parsed ``shareChatTasks`` command (``chatId``,
                optional ``tabId``, optional ``taskId``).
            endpoint: The requesting connection (local or remote).
        """
        tab_id = self._cmd_str(cmd, "tabId")[:_SHARE_TASKS_MAX_ID_CHARS]
        chat_id = self._cmd_str(cmd, "chatId")[:_SHARE_TASKS_MAX_ID_CHARS]
        task_id_filter = self._cmd_str(cmd, "taskId")[
            :_SHARE_TASKS_MAX_ID_CHARS
        ]
        peek = self._printer.peek_recording_for_task

        def _load_tasks() -> dict[str, Any]:
            reply: dict[str, Any] = {
                "type": "share_tasks",
                "tabId": tab_id,
                "chatId": chat_id,
                "tasks": [],
                "truncated": False,
            }
            if not chat_id.strip():
                return reply
            try:
                if task_id_filter.strip():
                    row = _load_chat_events_by_task_id(task_id_filter)
                    rows = [row] if row else []
                    truncated = False
                else:
                    rows, truncated = _load_all_chat_events_by_chat_id(
                        chat_id, _SHARE_TASKS_MAX_REPLY_BYTES,
                    )
            except Exception as exc:
                logger.warning("shareChatTasks: load failed", exc_info=True)
                reply["error"] = f"Failed to load the chat history: {exc}"
                return reply
            # The sub-agent transcripts ride along inside their task's
            # entry, so they share the task's byte budget: the walk
            # below re-charges each entry (task + sub-agents) newest
            # first and drops the OLDEST entries that no longer fit —
            # the same truncation rule the row loader applies.
            entries: list[dict[str, Any]] = []
            budget = _SHARE_TASKS_MAX_REPLY_BYTES
            for row in reversed(rows):
                task_id = str(row.get("task_id") or "")
                entry = {
                    "task": row.get("task", ""),
                    "task_id": task_id,
                    # The settings event lets the export's synthesized
                    # task panels carry the same settings info the live
                    # panel shows (see shareTaskPanel in media/main.js).
                    "events": with_synthesized_events(
                        cast("list[dict[str, Any]]", row.get("events") or []),
                        row,
                    ),
                }
                budget -= len(json.dumps(entry).encode("utf-8"))
                if budget < 0:
                    truncated = True
                    break
                # Every sub-agent the task fanned out (recursively):
                # the export renders them into the shared page's
                # sub-agent tabs (see media/share.js).  The loader is
                # handed the budget LEFT, so an oversized subtree stops
                # loading at the first transcript that cannot ship.
                subs, sub_bytes, overflowed = _share_subagent_entries(
                    task_id, budget, peek,
                )
                if overflowed:
                    truncated = True
                    break
                budget -= sub_bytes
                entry["subagents"] = subs
                entries.append(entry)
            entries.reverse()
            reply["tasks"] = entries
            reply["truncated"] = truncated
            return reply

        reply = await asyncio.to_thread(_load_tasks)
        await self._reply_direct(endpoint, reply, "shareChatTasks")

    async def _handle_check_paths(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Tell a client which candidate file paths exist.

        Handles the ``checkPaths`` command sent by ``media/main.js``
        after it linkifies file-path-looking strings in event panel
        contents: a path is rendered as a
        clickable link ONLY when this check confirms it names an
        existing regular file or directory, i.e. that a subsequent
        ``openFile`` click would actually open something.
        Paths are resolved exactly like :meth:`_handle_open_file`
        resolves them (``~`` expansion, then relative to the command's
        ``workDir``, then the tab's pending worktree).  The reply is
        sent directly to the requesting *endpoint* — never broadcast —
        with the shape::

            {"type": "pathsExist", "results": {<path>: <bool>, ...},
             "workDir": <echo of cmd workDir>,
             "tabId": <echo of cmd tabId>}

        Args:
            cmd: The parsed ``checkPaths`` command (``paths``, optional
                ``workDir``, ``tabId``).
            endpoint: The requesting connection.
        """
        raw_paths = cmd.get("paths")
        if not isinstance(raw_paths, list):
            raw_paths = []
        # The reply's workDir is a correlation key: main.js stamps each
        # candidate with the workDir it sent (data-path-wd) and only
        # applies a reply whose workDir matches.  Echo what the CLIENT
        # sent, or a tab that sent "" would never see its links promoted.
        raw_work_dir = self._cmd_str(cmd, "workDir")
        work_dir = self._cmd_work_dir(cmd)
        tab_id = self._cmd_str(cmd, "tabId")

        def _check_paths() -> dict[str, bool]:
            results: dict[str, bool] = {}
            for raw_path in raw_paths:
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                results[raw_path] = (
                    self._resolve_tab_file(raw_path, work_dir, tab_id)
                    is not None
                )
            return results

        results = await asyncio.to_thread(_check_paths)
        reply = {
            "type": "pathsExist",
            "results": results,
            "workDir": raw_work_dir,
            "tabId": tab_id,
        }
        await self._reply_direct(endpoint, reply, "checkPaths")

    async def _reply_direct(
        self, endpoint: Any, reply: dict[str, Any], what: str,
    ) -> None:
        """Send *reply* to *endpoint* only, logging (not raising) failures.

        Shared tail of the Explorer / Source Control handlers below:
        their replies go to the requesting connection alone, and a
        client that vanished mid-request must not surface an error.
        """
        try:
            await self._endpoint_send(endpoint, json.dumps(reply))
        except Exception:
            logger.debug("%s: failed to write reply", what, exc_info=True)

    async def _handle_list_dir(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """List a directory for the remote webapp's Explorer view.

        Handles the ``listDir`` command sent by ``media/main.js`` when
        the Explorer view opens (the workspace root) or the user expands
        a folder.  ``path`` is resolved like ``openFile`` resolves it
        (``~`` expansion, then relative to the command's ``workDir``,
        falling back to the daemon work dir); an empty ``path`` names
        the work dir itself.  The reply goes directly to the requesting
        *endpoint* — never broadcast — with the shape::

            {"type": "dirListing", "path": <abs dir>, "root": <work dir>,
             "tabId": <echo>, "token": <echo>,
             "entries": [{"name", "path", "isDir"}, ...],
             "truncated": <bool>}                 # on success
            {"type": "dirListing", "path": ..., "root": ..., "tabId": ...,
             "token": ..., "error": <message>}    # on failure

        Args:
            cmd: The parsed ``listDir`` command (optional ``path``,
                ``workDir``, ``tabId``, ``token``).
            endpoint: The requesting WSS connection.
        """
        from kiss.server.explorer import list_directory

        raw_path = self._cmd_str(cmd, "path")
        work_dir = self._cmd_work_dir(cmd)
        tab_id = self._cmd_str(cmd, "tabId")
        token = self._cmd_str(cmd, "token")

        def _list() -> dict[str, Any]:
            reply: dict[str, Any] = {
                "type": "dirListing",
                "path": raw_path or work_dir,
                "root": work_dir,
                "tabId": tab_id,
                "token": token,
            }
            try:
                path = self._resolve_tab_file(
                    raw_path or work_dir, work_dir, tab_id,
                )
                if path is None or not path.is_dir():
                    reply["error"] = (
                        f"Directory not found: {raw_path or work_dir}"
                    )
                    return reply
                reply["path"] = str(path)
                reply.update(list_directory(path))
            except Exception as exc:
                # OSError (unreadable), ValueError (NUL in a name), or
                # anything else: the view must get a reply either way.
                reply["error"] = f"Failed to list {raw_path or work_dir}: {exc}"
            return reply

        reply = await asyncio.to_thread(_list)
        await self._reply_direct(endpoint, reply, "listDir")

    async def _git_reply(
        self,
        cmd: dict[str, Any],
        endpoint: Any,
        reply_type: str,
        provider: Callable[..., dict[str, Any]],
        *args: Any,
        **fields: Any,
    ) -> None:
        """Run a ``kiss.server.explorer`` git provider and reply to *endpoint*.

        The shared body of the Source Control handlers: every reply
        carries ``type``, the command's ``workDir`` (falling back to
        the daemon work dir), the echoed ``tabId`` / ``token`` and the
        handler's extra *fields*, then either the provider's result
        (data or ``{"error": ...}``) or an ``error`` when the work dir
        does not exist or the provider raised (a path with a NUL byte,
        a broken git install, ...), so the client's view never sits at
        "Loading..." for a command that was accepted.

        Args:
            cmd: The parsed client command.
            endpoint: The requesting WSS connection.
            reply_type: The reply's ``type`` field.
            provider: The git provider, called with the work dir
                and *args* on a worker thread.
            *args: Extra positional arguments for *provider*.
            **fields: Extra fields echoed in the reply.
        """
        work_dir = self._cmd_work_dir(cmd)
        reply: dict[str, Any] = {
            "type": reply_type,
            "workDir": work_dir,
            "tabId": self._cmd_str(cmd, "tabId"),
            "token": self._cmd_str(cmd, "token"),
            **fields,
        }
        if not os.path.isdir(work_dir):
            reply["error"] = f"Directory not found: {work_dir}"
        else:
            try:
                reply.update(await asyncio.to_thread(provider, work_dir, *args))
            except Exception as exc:
                reply["error"] = f"{type(exc).__name__}: {exc}"
        await self._reply_direct(endpoint, reply, reply_type)

    async def _handle_git_status(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Report the working-tree changes for the Source Control view.

        Handles the ``gitStatus`` command sent by ``media/main.js`` when
        the Source Control view opens or refreshes.  The repository is
        the one containing the command's ``workDir`` (falling back to
        the daemon work dir).  The reply goes directly to the requesting
        *endpoint* with the shape::

            {"type": "gitStatus", "workDir": <work dir>, "tabId": <echo>,
             "token": <echo>, "repo": <abs repo root>, "branch": <name>,
             "changes": [{"path", "absPath", "status", "group"}, ...]}
            {"type": "gitStatus", "workDir": ..., "tabId": ..., "token": ...,
             "error": <message>}                  # not a repo / git failed

        See :func:`kiss.server.explorer.git_status` for the row fields.

        Args:
            cmd: The parsed ``gitStatus`` command (optional ``workDir``,
                ``tabId``, ``token``).
            endpoint: The requesting WSS connection.
        """
        from kiss.server.explorer import git_status

        await self._git_reply(cmd, endpoint, "gitStatus", git_status)

    async def _handle_git_log(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Report the recent commits for the Source Control graph.

        Handles the ``gitLog`` command sent by ``media/main.js`` when the
        Source Control view opens or refreshes.  The repository is the
        one containing the command's ``workDir`` (falling back to the
        daemon work dir); ``limit`` caps the number of commits
        (default :data:`kiss.server.explorer.GIT_LOG_DEFAULT_LIMIT`).
        The reply goes directly to the requesting *endpoint* with the
        shape::

            {"type": "gitLog", "workDir": <work dir>, "tabId": <echo>,
             "token": <echo>, "repo": <abs repo root>, "head": <sha>,
             "commits": [{"sha", "shortSha", "parents", "author", "date",
                          "refs", "subject", "files"}, ...]}
            {"type": "gitLog", "workDir": ..., "tabId": ..., "token": ...,
             "error": <message>}                  # not a repo / git failed

        See :func:`kiss.server.explorer.git_log` for the row fields.

        Args:
            cmd: The parsed ``gitLog`` command (optional ``workDir``,
                ``tabId``, ``token``, ``limit``).
            endpoint: The requesting WSS connection.
        """
        from kiss.server.explorer import GIT_LOG_DEFAULT_LIMIT, git_log

        limit = cmd.get("limit")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            limit = GIT_LOG_DEFAULT_LIMIT
        await self._git_reply(cmd, endpoint, "gitLog", git_log, limit)

    async def _handle_git_show(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Serve a commit's patch, a file at a commit, or a revision diff.

        Handles the ``gitShow`` command the remote Source Control graph
        sends for its commit context menu: "Open Changes" (the whole
        commit, or one file of it when ``path`` is given), "Open File"
        (``mode: "file"`` — the file's content at that commit),
        "Compare with..." (``base`` given — ``git diff base sha``) and a
        click on a file row (``mode: "diff"`` — both sides of that
        file's change, see :func:`kiss.server.explorer.git_file_diff`).
        The reply goes directly to the requesting *endpoint* with the
        shape::

            {"type": "gitShow", "workDir", "tabId", "token", "sha",
             "path", "base", "mode", "repo", "subject"?, "text",
             "truncated"}                          # on success
            {"type": "gitShow", ..., "mode": "diff", "original",
             "modified", "originalMissing", "modifiedMissing",
             "parent", "originalPath", "truncated"}
            {"type": "gitShow", ..., "error": <message>}

        Args:
            cmd: The parsed ``gitShow`` command (``sha``, optional
                ``path``, ``origPath``, ``base``, ``mode``, ``workDir``,
                ``tabId``, ``token``).
            endpoint: The requesting WSS connection.
        """
        from kiss.server.explorer import (
            git_compare,
            git_file_at,
            git_file_diff,
            git_show,
        )

        sha = self._cmd_str(cmd, "sha")
        path = self._cmd_str(cmd, "path")
        base = self._cmd_str(cmd, "base")
        mode = self._cmd_str(cmd, "mode") or "patch"
        provider: Callable[..., dict[str, Any]]
        args: tuple[str, ...]
        if base:
            provider, args = git_compare, (base, sha)
        elif mode == "diff":
            # Both sides of one file's change (``mode: "diff"``): the
            # commit against its parent, or the working tree against
            # HEAD when ``sha`` is empty; ``origPath`` is the name
            # before a rename.
            provider = git_file_diff
            args = (sha, path, self._cmd_str(cmd, "origPath"))
        elif mode == "file":
            provider, args = git_file_at, (sha, path)
        else:
            provider, args = git_show, (sha, path)
        await self._git_reply(
            cmd, endpoint, "gitShow", provider, *args,
            sha=sha, path=path, base=base, mode=mode,
        )

    async def _handle_git_action(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Run a commit context-menu action (checkout, branch, tag, pick).

        Handles the ``gitAction`` command of the remote Source Control
        graph's context menu; see
        :func:`kiss.server.explorer.git_action` for what each action
        runs.  The reply goes directly to the requesting *endpoint*::

            {"type": "gitActionResult", "workDir", "tabId", "token",
             "action", "sha", "ok": true, "output": <git output>}
            {"type": "gitActionResult", ..., "error": <git's message>}

        Args:
            cmd: The parsed ``gitAction`` command (``action``, ``sha``,
                optional ``name``, ``message``, ``workDir``, ``tabId``,
                ``token``).
            endpoint: The requesting WSS connection.
        """
        from kiss.server.explorer import git_action

        action = self._cmd_str(cmd, "action")
        sha = self._cmd_str(cmd, "sha")
        await self._git_reply(
            cmd, endpoint, "gitActionResult", git_action,
            action, sha, self._cmd_str(cmd, "name"), self._cmd_str(cmd, "message"),
            action=action, sha=sha,
        )

    def _abs_cmd_path(self, raw: str, work_dir: str) -> str:
        """*raw* as an absolute LEXICAL path (``~`` expanded, relative to *work_dir*).

        Lexical: ``.`` / ``..`` segments are folded but no symlink is
        followed, so the path names the Explorer entry itself -- a
        symlink stays the symlink (Delete unlinks it rather than
        deleting its target; a dangling one can still be removed).  It
        also serves an entry that does not exist yet (a rename target),
        which :meth:`_resolve_tab_file` -- existing paths only -- cannot.
        """
        path = Path(os.path.expanduser(raw))
        if not path.is_absolute() and work_dir:
            path = Path(work_dir) / path
        return os.path.normpath(str(path))

    @staticmethod
    def _inside(path: str, root: str) -> bool:
        """Whether the lexical *path* is *root* or below it."""
        root = os.path.normpath(root)
        try:
            return os.path.commonpath([path, root]) == root
        except ValueError:
            return False

    @classmethod
    def _confined(cls, path: str, root: str, follow: bool) -> bool:
        """Whether *path* belongs to the workspace *root* on disk as well.

        Lexical containment (:meth:`_inside`) is not enough: a folder
        symlink inside the workspace that points outside it makes
        ``root/portal/x`` a name for ``/elsewhere/x``.  So the REAL
        location must be under the real root too -- of the entry
        itself when *follow* (a folder to list, search or paste into, a
        file to read), of its parent folder otherwise (an entry that is
        acted on as itself: a symlink is deleted / renamed / copied as
        the link, wherever it points).
        """
        if not cls._inside(path, root):
            return False
        try:
            real_root = os.path.realpath(root)
            probe = path if follow else os.path.dirname(path)
            real = os.path.realpath(probe)
            return os.path.commonpath([real, real_root]) == real_root
        except (OSError, ValueError):
            return False

    async def _handle_fs_action(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Run an Explorer context-menu file action.

        Handles the ``fsAction`` command the remote Explorer view sends
        for New File..., New Folder..., Rename..., Delete, Paste (copy
        / move), Find in Folder... and Compare Selected; see
        :func:`kiss.server.fs_actions.fs_action`.  ``path`` (and
        ``dest`` for ``copy`` / ``move`` / ``compare``) must name
        existing entries; ``rename``'s ``dest`` is the new path.  Paths
        are taken lexically (a symlink is the entry, not its target)
        and must lie inside ``workDir`` -- the Explorer root the menu
        was opened in -- on disk as well as by name (see
        :meth:`_confined`).  The reply goes directly to the requesting
        *endpoint*::

            {"type": "fsResult", "tabId", "token", "action", "path",
             "ok": true, "path": <result path>, "text"?, ...}
            {"type": "fsResult", ..., "error": <message>, "exists"?: true}

        Args:
            cmd: The parsed ``fsAction`` command (``action``, ``path``,
                optional ``dest``, ``name``, ``query``, ``overwrite``,
                ``workDir``, ``tabId``, ``token``).
            endpoint: The requesting WSS connection.
        """
        from kiss.server.fs_actions import FS_ACTIONS, fs_action

        work_dir = self._cmd_work_dir(cmd)
        tab_id = self._cmd_str(cmd, "tabId")
        action = self._cmd_str(cmd, "action")
        raw_path = self._cmd_str(cmd, "path")
        raw_dest = self._cmd_str(cmd, "dest")
        reply: dict[str, Any] = {
            "type": "fsResult",
            "workDir": work_dir,
            "tabId": tab_id,
            "token": self._cmd_str(cmd, "token"),
            "action": action,
            "path": raw_path,
        }

        def _run() -> dict[str, Any]:
            if action not in FS_ACTIONS:
                return {"error": f"Unknown file action: {action}"}
            if not raw_path:
                return {"error": "No path given"}
            try:
                # Lexical paths: the Explorer names entries, and a
                # symlink entry must be acted on as the link, never
                # as its target (see _abs_cmd_path).  Every path an
                # action touches must lie under the Explorer's root:
                # the menu edits the workspace on screen, not the host.
                path = self._abs_cmd_path(raw_path, work_dir)
                if not os.path.lexists(path):
                    return {"error": f"Not found: {raw_path}"}
                # The entry is acted on as itself (delete / rename /
                # copy / move); a folder is entered (new entry, search)
                # and a compared file is read.
                acts_on_entry = action in ("delete", "rename", "copy", "move")
                if not self._confined(path, work_dir, follow=not acts_on_entry):
                    return {"error": f"Not inside the workspace: {raw_path}"}
                dest = raw_dest
                if action in ("copy", "move", "compare", "rename"):
                    if not raw_dest:
                        return {"error": "No destination given"}
                    dest = self._abs_cmd_path(raw_dest, work_dir)
                    if action != "rename" and not os.path.lexists(dest):
                        return {"error": f"Not found: {raw_dest}"}
                    # A paste destination folder is entered, a compared
                    # file read; a rename target is a new sibling name.
                    if not self._confined(dest, work_dir, follow=action != "rename"):
                        return {
                            "error": f"Not inside the workspace: {raw_dest}",
                        }
                result = fs_action(
                    action,
                    path,
                    dest=dest,
                    name=self._cmd_str(cmd, "name"),
                    query=self._cmd_str(cmd, "query"),
                    overwrite=cmd.get("overwrite") is True,
                )
                result.setdefault("path", path)
                return result
            except Exception as exc:
                return {"error": f"{action} failed: {exc}"}

        reply.update(await asyncio.to_thread(_run))
        await self._reply_direct(endpoint, reply, "fsAction")

    def _tab_task_agent(self, tab_id: str) -> Any:
        """Return the agent of the task *tab_id* is running or viewing.

        The tab that launched a task owns its agent state
        (:func:`agent_state.find_by_tab`); a tab that merely views the
        task — the same chat open in another client — is only
        subscribed to its event stream (:meth:`JsonPrinter.tasks_for_tab`),
        so both lookups are combined.  A tab can be attached to more
        than one state at once (its own finished task's state lingers
        while its worktree is pending; a finished task's subscriber set
        lingers a few minutes), so a state whose task is active wins;
        otherwise the tab's own state, then the first subscribed one.

        Args:
            tab_id: The requesting client's tab id.

        Returns:
            The task's live agent, or ``None`` when the tab is attached
            to no task.
        """
        from kiss.server import agent_state

        states: list[agent_state.AgentState] = []
        own = agent_state.find_by_tab(tab_id)
        if own is not None:
            states.append(own)
        for key in self._printer.tasks_for_tab(tab_id):
            state = agent_state.get(key)
            if state is not None and state not in states:
                states.append(state)
        for state in states:
            if state.is_task_active:
                return state.agent
        return states[0].agent if states else None

    async def _handle_get_task_update(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Send a client the task update for the task its tab shows.

        Handles the ``getTaskUpdate`` command polled by ``media/main.js``
        for the info subpanel of the task-info panel: the subpanel shows
        the :mod:`~kiss.agents.seas.ask.ask_sea` agent's short answer
        to what the task RUNNING in the tab (:meth:`_tab_task_agent`)
        has done so far, never a file the task left on disk.
        :class:`TaskUpdateRunner` owns the updates: this poll makes it
        run the agent once the tab's task is
        :data:`~kiss.server.task_update.FIRST_UPDATE_DELAY_S` old and
        has no update yet, when the update is
        :data:`~kiss.server.task_update.UPDATE_INTERVAL_S` old, or when
        the poll carries ``refresh: true`` (the panel's refresh button);
        the reply reflects the state right after that decision, so a
        refresh answers ``running: true`` at once and the answer itself
        arrives with a later poll.

        The reply is sent directly to the requesting *endpoint* — never
        broadcast — with the shape::

            {"type": "taskUpdate", "exists": <bool>, "sig": <str>,
             "content": <html>, "error": <str>, "running": <bool>,
             "cost": <usd>, "updatedAt": <epoch ms>,
             "dueAt": <epoch ms>,                # next unforced run
             "unchanged": true,                  # sig == cmd knownSig
             "tabId": <echo of cmd tabId>, "taskId": <task id>,
             "token": <echo of cmd token>}

        ``token`` is an opaque client request tag the page matches
        replies by.  ``sig`` fingerprints the update state; a poll whose
        ``knownSig`` matches it is answered with ``unchanged: true`` and
        no ``content``.  A tab attached to no running task, or to a task
        whose history row is not allocated yet, replies ``exists: false``
        with empty content — the client renders that as an empty
        subpanel rather than an error.

        Args:
            cmd: The parsed ``getTaskUpdate`` command (``tabId``,
                optional ``knownSig``, ``token``, ``refresh``).
            endpoint: The requesting WSS connection.
        """
        from kiss.agents.sorcar.sorcar_agent import _persisted_task_id
        from kiss.server import agent_state

        tab_id = self._cmd_str(cmd, "tabId")
        known_sig = self._cmd_str(cmd, "knownSig")
        token = self._cmd_str(cmd, "token")
        force = bool(cmd.get("refresh"))
        reply: dict[str, Any] = {
            "type": "taskUpdate",
            "tabId": tab_id,
            "token": token,
            "taskId": "",
            "exists": False,
            "sig": "",
            "content": "",
        }
        agent = self._tab_task_agent(tab_id) if tab_id else None
        state = agent_state.find_by_agent(agent) if agent is not None else None
        task_id = _persisted_task_id(agent) if agent is not None else ""
        # The state is keyed by the persisted task id only once the run
        # has allocated its history row (``agent_task_allocated`` re-keys
        # it); until then a reused tab's agent still reports the PREVIOUS
        # task's id, which must not be shown or re-run as this task's.
        if (
            state is not None
            and state.is_task_active
            and task_id
            and state.task_id == task_id
        ):
            update = await asyncio.to_thread(
                self._task_updates.poll, task_id, agent, force,
            )
            reply["taskId"] = task_id
            reply.update(update.payload())
            if known_sig and known_sig == update.sig:
                reply["unchanged"] = True
                del reply["content"]
        await self._reply_direct(endpoint, reply, "getTaskUpdate")

    async def _handle_active_tasks_query(self, endpoint: Any) -> None:
        """Report in-flight agent tasks back to a single client.

        Used by the installer (``scripts/check-kiss-web-active-tasks.py``)
        before it considers SIGTERMing the daemon: when any task is
        still active the restart is deferred so an in-progress agent
        run is not interrupted.

        The response is a single JSON object sent directly to the
        requesting *endpoint* — a :class:`ServerConnection`, via
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
        await self._reply_direct(endpoint, {
            "type": "activeTasksResponse",
            "count": len(active_tabs),
            "tabs": active_tabs,
        }, "activeTasksQuery")


    @property
    def _loopback_url(self) -> str:
        """The ``http://127.0.0.1:PORT`` URL for local-machine access.

        Plain HTTP: loopback traffic never leaves the machine, so the
        browser needs no certificate trust (LAN and tunnel URLs stay
        ``https://``).
        """
        return f"http://127.0.0.1:{self.port}"

    @property
    def _serves_local_ca(self) -> bool:
        """True when the served certificate is signed by the auto-generated local CA."""
        return not self._ssl_certfile

    def _lan_urls(self) -> list[str]:
        """Return ``https://<lan-ip>:PORT`` URLs for this host's LAN IPs.

        Returns ``[]`` when LAN clients cannot actually reach the
        webapp — the server is bound to a loopback-only host, or no
        ``remote_password`` is configured (``_process_request`` answers
        non-loopback peers 403 in that case) — so the UI never
        advertises a LAN URL that would be refused.

        Uses only the cached :attr:`_last_ips` snapshot (seeded
        off-thread in ``_setup_server`` before the first URL-file
        write, refreshed by the watchdog), so callers on the event
        loop never block on the socket probes in
        :func:`_get_local_ips`.  Before that first probe completes —
        e.g. a client that connects during the listener/tunnel setup
        window and asks for welcome info — no LAN URLs are advertised
        yet rather than probing on the loop.
        """
        if self.host == "localhost" or sorcar_api.is_loopback_ip(self.host):
            return []
        if not str(load_config().get("remote_password", "") or ""):
            return []
        if not self._ips_probed:
            return []
        return [f"https://{ip}:{self.port}" for ip in sorted(self._last_ips)]

    def _write_url_file_sync(self, tunnel_url: str | None) -> None:
        """Write the URL file with local, loopback, LAN + tunnel URLs.

        Blocking (disk + possible LAN-IP probe); callers on the event
        loop must run it via an executor / ``asyncio.to_thread``.

        Args:
            tunnel_url: The Cloudflare tunnel URL, or None.
        """
        _save_url_file(
            self._url_file, self._local_url, tunnel_url,
            self._loopback_url, self._lan_urls(), self._serves_local_ca,
        )

    def _write_url_file_logged(self, tunnel_url: str | None) -> None:
        """Run :meth:`_write_url_file_sync`, logging any failure.

        Executor target for fire-and-forget URL-file re-writes whose
        exceptions would otherwise vanish with the unawaited future.

        Args:
            tunnel_url: The Cloudflare tunnel URL, or None.
        """
        try:
            self._write_url_file_sync(tunnel_url)
        except Exception:
            logger.warning("URL-file re-write failed", exc_info=True)

    def _republish_urls(self) -> None:
        """Re-write the URL file and re-broadcast ``remote_url``.

        Called on the event loop after the watchdog adopts a new
        LAN-IP baseline so the URL file's ``lan`` list and every open
        settings/welcome panel stop showing addresses the machine no
        longer holds.  The disk write runs in the default executor and
        the broadcast in a task (:meth:`_broadcast_remote_url` builds
        its message off-thread) to keep the loop responsive.
        """
        tunnel_url = self._current_tunnel_url()
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, self._write_url_file_logged, tunnel_url)
        self._republish_task = loop.create_task(self._broadcast_remote_url(
            self._active_url or self._local_url,
        ))

    def _current_tunnel_url(self) -> str | None:
        """Return the active Cloudflare tunnel URL, or None when only the local URL is active."""
        if self._active_url and self._active_url != self._local_url:
            return self._active_url
        return None

    def _remote_url_message(self, url: str) -> dict[str, object]:
        """Build the ``remote_url`` event every connected client receives.

        Includes the ``ntfyUrl`` field only when both *url* is
        non-empty and an ntfy topic is configured, matching the
        contract pinned by the welcome-info and tunnel-restart tests.
        Always carries ``loopbackUrl`` (the ``http://127.0.0.1:PORT``
        address for the local machine) and ``lanUrls`` (the
        ``https://<lan-ip>:PORT`` addresses for other devices on the
        LAN) so the settings panel and the welcome page can show how
        to reach the webapp alongside the Cloudflare URL.

        Blocking (reads the stored ntfy topic and, via
        :meth:`_lan_urls`, the config file); callers on the event loop
        go through :meth:`_broadcast_remote_url`.

        Args:
            url: The active URL (``""`` when none is known).

        Returns:
            The event dict, ready for :meth:`WebPrinter.broadcast`.
        """
        ntfy_url = _get_ntfy_url() if url else ""
        msg: dict[str, object] = {
            "type": "remote_url",
            "url": url or "",
            "loopbackUrl": self._loopback_url,
            "lanUrls": self._lan_urls(),
            "localCa": self._serves_local_ca,
        }
        if ntfy_url:
            msg["ntfyUrl"] = ntfy_url
        return msg

    def _broadcast_remote_url(self, url: str) -> Coroutine[Any, Any, None]:
        """Publish a ``remote_url`` event to every connected client.

        Synchronous on purpose: it stamps the publication with a fresh
        :attr:`_url_publish_gen` *now*, while the caller still holds
        the loop and *url* is current, and returns the coroutine that
        delivers it.  A publication queued with ``create_task`` thus
        ranks by the state it captured, not by when it starts running,
        so a stale snapshot can never mint a newer generation than a
        publication made after it.  Callers ``await`` the result or
        hand it to ``create_task``.

        Args:
            url: The active URL (``""`` when none is known).

        Returns:
            The coroutine that builds the message and broadcasts it.
        """
        self._url_publish_gen += 1
        return self._deliver_remote_url(url, self._url_publish_gen)

    async def _deliver_remote_url(self, url: str, gen: int) -> None:
        """Build the ``remote_url`` message off-thread and broadcast it.

        Dropped when a newer publication (a higher
        :attr:`_url_publish_gen`) was stamped while the message was
        being built, so clients never see a superseded URL land after
        its replacement.
        """
        msg = await asyncio.to_thread(self._remote_url_message, url)
        if gen == self._url_publish_gen:
            self._printer.broadcast(msg)

    async def _broadcast_update_available(self) -> None:
        """Broadcast the cached PyPI ``update_available`` state.

        No-op until :meth:`_check_for_update` has cached a latest
        version on :attr:`_latest_version`.  ``_read_version`` scans a
        directory on disk, so it runs off-thread (M10).
        """
        latest = self._latest_version
        if not latest:
            return
        current = await asyncio.to_thread(_read_version)
        available = bool(current) and _compare_versions(latest, current) > 0
        snoozed = available and await asyncio.to_thread(
            _is_update_snoozed, latest,
        )
        self._printer.broadcast({
            "type": "update_available",
            "available": available,
            "latest": latest,
            "current": current,
            "snoozed": snoozed,
            "pendingIdle": self._update_when_idle_armed,
        })

    def _update_in_progress(self) -> bool:
        """Return whether an installer is being spawned or still running."""
        return self._update_starting or (
            self._update_proc is not None and self._update_proc.poll() is None
        )

    async def _cancel_update_when_idle(self) -> bool:
        """Disarm an "Update when idle" still waiting for idle.

        Returns:
            Whether one was armed.  A poller that has already detected
            idle and is launching the installer is left alone: the
            update is under way and the single-flight guard in
            :meth:`_handle_run_update` covers any concurrent request.
            The flag is cleared under :data:`agent_state.STATE_LOCK`
            so an idle check already started in the executor
            (:meth:`_arm_update_barrier_if_idle`) cannot raise the
            barrier after this cancel lowered it.  That acquisition
            waits behind task threads holding the lock, so it runs in
            the executor like the idle check itself and never stalls
            the event loop; only ``Task.cancel`` runs on the loop.
        """
        if not self._update_when_idle_armed:
            return False
        if not await asyncio.to_thread(self._disarm_update_when_idle):
            # The poller armed the barrier and cleared the flag during
            # the wait: it is launching the installer, leave it alone.
            return False
        task = self._update_when_idle_task
        if task is not None:
            task.cancel()
            self._update_when_idle_task = None
        return True

    def _disarm_update_when_idle(self) -> bool:
        """Executor thread: clear the armed flag under the state lock.

        Returns:
            Whether the flag was set (this call disarmed the update).
        """
        with agent_state.STATE_LOCK:  # executor thread: no Stop injection
            was_armed = self._update_when_idle_armed
            self._update_when_idle_armed = False
            return was_armed

    async def _handle_update_when_idle(self, cancel: bool = False) -> None:
        """Arm (or cancel) an update that runs once no task is running.

        Server-side handler for the update toast's "Update when idle"
        and "Cancel" actions.  Arming starts :meth:`_run_update_when_idle`
        (idempotent while one is pending); either way the
        ``update_available`` state is rebroadcast with ``pendingIdle``
        so every chat window's toast shows the armed state.

        Args:
            cancel: ``True`` disarms a pending idle update instead of
                arming one.
        """
        if cancel:
            await self._cancel_update_when_idle()
        elif (
            self._update_when_idle_task is None
            and not self._update_in_progress()
        ):
            # No poller at all — neither armed nor still finishing its
            # handoff after a fast installer exit — so a second one
            # cannot overwrite (and orphan) a live task.
            self._update_when_idle_armed = True
            self._update_when_idle_task = asyncio.create_task(
                self._run_update_when_idle(),
            )
        await self._broadcast_update_available()

    async def _run_update_when_idle(self) -> None:
        """Wait until no task is in flight, then launch the installer.

        Polls :meth:`_arm_update_barrier_if_idle` (off-thread: it takes
        the registry lock) every :data:`_IDLE_UPDATE_POLL_S` seconds.
        The idle verdict and the run-admission barrier are one atomic
        step, so no ``run`` can start between "no active tasks" and the
        installer spawn (the barrier makes ``_cmd_run`` refuse it).
        Once armed it disarms itself and, without yielding to the loop
        in between, runs :meth:`_handle_run_update` for every window
        (``conn_id=""``): the click that armed it may be long gone by
        the time the update starts, so its notices must not be confined
        to one connection.  The task stays tracked on
        :attr:`_update_when_idle_task` until it returns so
        :meth:`stop_async` can cancel it mid-handoff too.
        """
        try:
            while not await asyncio.to_thread(self._arm_update_barrier_if_idle):
                await asyncio.sleep(_IDLE_UPDATE_POLL_S)
            self._update_when_idle_armed = False
            await self._handle_run_update("")
            await self._broadcast_update_available()
        except asyncio.CancelledError:
            # Cancelled (toast "Cancel", a direct Update click, or
            # shutdown) after the barrier went up but before an
            # installer owned it: lower it, or the daemon would refuse
            # every task until restart.  With an installer running,
            # its exit watcher lowers the barrier instead.
            if not self._update_in_progress():
                self._set_update_barrier(False)
            raise
        finally:
            if self._update_when_idle_task is asyncio.current_task():
                self._update_when_idle_task = None

    async def _handle_snooze_update(self, latest: str = "") -> None:
        """Record a 24h "Remind me later" snooze and rebroadcast.

        Writes the snooze into the ``.update-check.json`` cache (file
        I/O off-thread, M10), then rebroadcasts ``update_available``
        with ``snoozed: true`` so every connected client drops its
        sticky update toast at once.

        Args:
            latest: The release version being snoozed ("" falls back
                to the latest version the daemon last fetched).
        """
        await asyncio.to_thread(
            _record_update_snooze, latest or self._latest_version or "",
        )
        await self._broadcast_update_available()

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
                None, _post_url_to_message_board, url, self._ntfy_base_url,
            )
            self._last_posted_url = url

    async def _send_welcome_info(self) -> None:
        """Broadcast the active remote URL to all connected clients.

        Broadcasts the ``remote_url`` event using the in-memory URL,
        the URL file, or — for tunnel-enabled servers only — the
        ``cloudflared`` metrics API as successive fallbacks.

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
        if not url and self.use_tunnel:
            # Only a tunnel-enabled server may adopt a discovered
            # cloudflared URL: the machine-wide scan can find a
            # FOREIGN process's tunnel (another daemon on this host,
            # e.g. the production kiss-web next to a test server)
            # whose URL routes to that other server, not to this one.
            # A tunnel-less server must never advertise — let alone
            # persist to its URL file — a URL it does not own.
            discovered = await loop.run_in_executor(
                None, _discover_tunnel_url_from_metrics,
            )
            # Re-check after the await: the watchdog's
            # ``_restart_tunnel_url`` may have published this daemon's
            # fresh tunnel URL meanwhile, which must not be clobbered
            # by a URL the scan took from a dead or foreign tunnel.
            # The claim happens on the loop with no await between the
            # check and the assignment; the file write follows, and if
            # a publisher overtook the claim during that write (its own
            # write may have landed first) the file is rewritten from
            # the current state.
            if discovered and not self._active_url:
                self._active_url = discovered
                await loop.run_in_executor(
                    None, self._write_url_file_sync, discovered,
                )
                if self._active_url != discovered:
                    await loop.run_in_executor(
                        None, self._write_url_file_logged,
                        self._current_tunnel_url(),
                    )
        # Revalidate after the awaits above: a tunnel (re)start or
        # clear that landed during the file read or the scan has
        # published the current URL, which must not be followed by the
        # obsolete one the file or the scan returned.
        url = self._active_url or url
        await self._broadcast_remote_url(url or "")
        await self._broadcast_update_available()

    async def _endpoint_send(
        self, endpoint: ServerConnection, data: str,
    ) -> None:
        """Send ``data`` to one client connection.

        Delegates to the printer's :meth:`WebPrinter._locked_send`
        (C-R1) rather than duplicating it: that path acquires the
        per-endpoint send lock — so a direct reply cannot overtake a
        broadcast event already in flight to the same client
        (``Connection.send`` waits out write backpressure BEFORE
        queuing the frame) — and routes through
        :meth:`WebPrinter._timed_send`, whose failure handler removes
        a dead peer from the active sets so subsequent broadcasts skip
        it.

        Args:
            endpoint: The connection to send to.
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        await self._printer._locked_send(endpoint, data)

    @staticmethod
    def _sanitized_restored_tabs(cmd: dict[str, Any]) -> list[dict[str, str]]:
        """Sanitize the ``restoredTabs`` field of a ``ready`` command.

        The M7 hardening, applied ONCE by the ``ready`` handler of the
        server API (:meth:`kiss.server.sorcar.ServerApi.ready`), which
        writes the cleaned list back into the command before
        :meth:`_handle_ready` reads it:

        * caps the list at ``_MAX_RESTORED_TABS`` so an
          authenticated-but-malicious or buggy client cannot flood the
          executor with thousands of ``resumeSession`` jobs;
        * skips malformed (non-dict) elements — an ``AttributeError``
          would propagate out of the command dispatch and tear
          down the whole authenticated connection over one bad field;
        * blanks non-str ``tabId`` / ``chatId`` values — a non-str
          ``chatId`` would flow into backend handlers that assume
          strings.

        Every rejection is logged with a ``warning``.

        Args:
            cmd: The ``ready`` command dict.

        Returns:
            Cleaned entries, each ``{"tabId": str, "chatId": str}``
            (fields blanked to ``""`` when missing or malformed).
        """
        restored = cmd.get("restoredTabs") or []
        if not isinstance(restored, list):
            restored = []
        if len(restored) > _MAX_RESTORED_TABS:
            logger.warning(
                "restoredTabs count %d exceeds cap %d; truncating",
                len(restored), _MAX_RESTORED_TABS,
            )
            restored = restored[:_MAX_RESTORED_TABS]
        cleaned: list[dict[str, str]] = []
        for rt in restored:
            if not isinstance(rt, dict):
                logger.warning("ignoring non-dict restoredTabs entry: %r", rt)
                continue
            rt_id = rt.get("tabId", "")
            if not isinstance(rt_id, str):
                logger.warning("ignoring non-str restoredTabs tabId: %r", rt_id)
                rt_id = ""
            chat_id = rt.get("chatId", "")
            if not isinstance(chat_id, str):
                logger.warning(
                    "ignoring non-str restoredTabs chatId: %r", chat_id,
                )
                chat_id = ""
            title = rt.get("title", "")
            if not isinstance(title, str):
                title = ""
            work_dir = rt.get("workDir", "")
            if not isinstance(work_dir, str):
                work_dir = ""
            cleaned.append({
                "tabId": rt_id, "chatId": chat_id,
                "title": title, "workDir": work_dir,
            })
        return cleaned

    async def _handle_ready(
        self, cmd: dict[str, Any], websocket: Any) -> None:
        """Initialize a (re)connecting client from canonical state.

        Fans the ``ready`` out into ``getModels`` / ``getInputHistory``
        / ``getConfig`` (each stamped with the sender's ``connId`` so
        the replies reach ONLY the window that just (re)connected),
        then synchronizes the client with the shared tab registry:
        the client's legacy ``restoredTabs`` are adopted only into an
        EMPTY registry (one-time migration), a canonical ``tabs_state``
        snapshot is broadcast, the tab the client shows (``tabId``) is
        replayed TO THIS CLIENT (``replayConnId``: the other windows
        have that transcript and must not rebuild it because a new
        panel connected), and every other chat-bound registry tab is
        deferred (``deferred: true`` → :meth:`AgentServer._defer_replay`):
        the client gets its running state and a ``replay_deferred``
        offer and fetches the transcript when the user opens the tab.
        A ``ready`` whose legacy ``restoredTabs`` seed an EMPTY
        registry is the exception: those tabs are new to every client,
        so all of them are replayed to everyone.  Tab state is
        server-canonical — clients never keep a tab set of their own.

        Args:
            cmd: The ``ready`` message from the client (already
                stamped with the connection's ``connId`` by
                :meth:`kiss.server.sorcar.ServerApi.dispatch`, its
                ``restoredTabs`` already sanitized by
                :meth:`kiss.server.sorcar.ServerApi.ready`).
            websocket: The client connection (for direct replies).
        """
        tab_id = self._cmd_str(cmd, "tabId")
        conn_id = cmd.get("connId", "")
        # The page's ``ready`` carries no ``workDir`` (``main.js`` sends
        # ``{tabId, restoredTabs}``): the fan-out resolves the daemon's
        # configured folder, which ``setWorkDir`` pins before ``ready``.
        for init_cmd in (
            "getModels", "getInputHistory", "getConfig", "getMyModels",
            "getSeaCommands",
        ):
            await self._run_cmd({"type": init_cmd, "connId": conn_id})
        try:
            await self._endpoint_send(
                websocket, json.dumps({"type": "tasks_updated"}),
            )
        except Exception:
            logger.debug("ready tasks_updated nudge failed", exc_info=True)
        # A browser tab is open on every surface while its page lives
        # and on none once it closes: the snapshot lets this
        # (re)connecting client add the live ones and drop any it kept
        # from before a disconnect or a daemon restart.
        self._broadcast_to_conn(self._agent_server.browser_tabs.snapshot_event(), conn_id)
        await self._send_welcome_info()
        # The Inject promptlets and the tips: the daemon owns both files
        # (and the opt-out marker), so every surface paints them from
        # these events (the page also embeds them at load).
        for bootstrap in await asyncio.to_thread(self._bootstrap_events):
            self._broadcast_to_conn(bootstrap, conn_id)
        try:
            await self._endpoint_send(
                websocket,
                json.dumps({"type": "focusInput", "tabId": tab_id}),
            )
        except Exception:
            pass
        try:
            bound, adopted = await asyncio.to_thread(
                self._agent_server.ready_tab_sync, cmd["restoredTabs"],
            )
        except Exception:
            logger.exception("ready tab-registry sync failed")
            bound, adopted = [], False
        # Tabs this ready just seeded the empty registry with (legacy
        # migration) are new to every other client too: those clients
        # adopt them from the ``tabs_state`` snapshot without asking for
        # their transcripts, so the replays must reach everyone.
        replay_conn_id = "" if adopted else conn_id
        # Only the tab the client is looking at (``tabId``) gets its
        # transcript now.  The other bound tabs are DEFERRED: the
        # client receives their
        # running state and a ``replay_deferred`` offer, and asks for
        # a transcript (a ``scoped`` ``resumeSession``) when the user
        # opens the tab.  Replaying every tab used to send a laptop
        # with twenty tabs hundreds of megabytes per reconnect and
        # stalled the daemon while it loaded and serialized them,
        # which in turn made the client reconnect again.
        for rt_id, rt_chat, rt_task in bound:
            resume: dict[str, Any] = {
                "type": "resumeSession", "chatId": rt_chat,
                "tabId": rt_id, "replayConnId": replay_conn_id,
            }
            if not adopted and rt_id != tab_id:
                resume["deferred"] = True
            # A tab pinned to a specific historical task replays THAT
            # task; without the taskId the replay would silently
            # switch every client's tab to the chat's latest task.
            if rt_task:
                resume["taskId"] = rt_task
            await self._run_cmd(resume)

    async def _open_path_only_prompt(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> bool:
        """Open the file a path-only ``submit`` names; ``True`` when it did.

        A single-line prompt that is nothing but the path of an
        existing regular file, typed into a tab that is not running a
        task, is a request to open that file.  The submitting
        connection gets ``promptOpened`` (the page drops the prompt and
        the task claim it stamped on the tab, see ``main.js``
        ``handleEvent``) and then the file's ``fileContent`` through
        :meth:`_handle_open_file`.  No task starts.

        A tab whose task is running, or that views a task blocked in
        ``ask_user_question``, is skipped: there the prompt is a
        follow-up or an answer that ``_cmd_run`` routes to the worker.
        Directories are skipped too, so a one-word prompt that happens
        to name a folder (``src``, ``tmp``) is still a task.

        Args:
            cmd: The ``submit`` message from the client.
            endpoint: The submitting connection.

        Returns:
            ``True`` when the prompt was answered with the file and the
            caller must not start a run, ``False`` to run as usual.
        """
        prompt = cmd.get("prompt", "")
        if not isinstance(prompt, str):
            return False
        trimmed = prompt.strip()
        if not trimmed or "\n" in trimmed:
            return False
        tab_id = self._cmd_str(cmd, "tabId")
        with agent_state.STATE_LOCK:
            prev = agent_state.find_by_tab(tab_id)
            if prev is not None and prev.task_thread is not None:
                return False
            if self._agent_server._viewer_awaiting_answer(tab_id) is not None:
                return False
        work_dir = self._cmd_work_dir(cmd)
        # Path stats (and possibly a slow filesystem) stay off the loop.
        path = await asyncio.to_thread(
            self._resolve_tab_file, trimmed, work_dir, tab_id, True,
        )
        if path is None:
            return False
        await self._reply_direct(
            endpoint, {"type": "promptOpened", "tabId": tab_id}, "submit",
        )
        await self._handle_open_file(
            {"path": str(path), "workDir": work_dir, "tabId": tab_id},
            endpoint,
        )
        return True

    async def _handle_submit(
        self, cmd: dict[str, Any], endpoint: Any = None,
    ) -> None:
        """Translate a chat page's ``submit`` into a backend ``run``.

        The translation includes the path-only shortcut: a single-line
        prompt that is nothing but the path of an existing regular file
        (relative to the tab's work dir or its pending worktree) is a
        request to open that file, so it is answered through
        :meth:`_open_path_only_prompt` and no task starts.  Only
        regular files qualify: a one-word prompt that happens to name a
        directory (``src``, ``tmp``) is still a task.

        Args:
            cmd: The ``submit`` message from the client.
            endpoint: The submitting connection, which receives a
                path-only prompt's ``promptOpened`` and file reply;
                ``None`` disables the shortcut.
        """
        tab_id = cmd.get("tabId", "")
        if endpoint is not None and await self._open_path_only_prompt(
            cmd, endpoint,
        ):
            return
        if self._shutdown_initiated:
            # Shutdown admission gate (F4-06): a task submitted after
            # the shutdown sweep snapshotted the active workers would
            # be silently killed when the process exits.  A refusal
            # like any other: the composer keeps the draft.
            self._agent_server._refuse_run(
                tab_id, "Server is shutting down; task not started.",
            )
            return
        prompt = cmd.get("prompt", "")
        if isinstance(prompt, str):
            # ``surrogatepass``: json.loads may yield lone surrogates
            # (``"\ud800"``), which strict UTF-8 refuses to encode.
            prompt_size = len(prompt.encode("utf-8", errors="surrogatepass"))
            if prompt_size > _MAX_PROMPT_BYTES:
                # Refuse rather than silently cut the prompt: the tab's
                # composer keeps the draft (only a ``status
                # running:false`` lowers its optimistic running state,
                # which ``_refuse_run`` sends first) and the user is
                # told the limit instead of the agent running on a
                # prompt with its end missing.
                logger.warning(
                    "prompt size %d bytes exceeds cap %d bytes; refusing",
                    prompt_size, _MAX_PROMPT_BYTES,
                )
                self._agent_server._refuse_run(
                    tab_id,
                    f"This prompt is too long to send: it is "
                    f"{prompt_size / 1_000_000:.1f} MB and the limit is "
                    f"{_MAX_PROMPT_BYTES / 1_000_000:.0f} MB. Shorten it, or "
                    f"put the long part in a file and attach that.",
                )
                return
        attachments = cmd.get("attachments")
        notice = ""
        if isinstance(attachments, list) and len(attachments) > _MAX_ATTACHMENTS:
            dropped = len(attachments) - _MAX_ATTACHMENTS
            logger.warning(
                "attachments count %d exceeds cap %d; dropping %d",
                len(attachments), _MAX_ATTACHMENTS, dropped,
            )
            # Carried into the run (``_notice``) and broadcast by
            # ``_cmd_run`` right after the new task's ``clear``: sent
            # from here it would land before that reset and be wiped
            # from the transcript it is meant to explain.
            notice = (
                f"Only the first {_MAX_ATTACHMENTS} attachments were "
                f"sent (the limit per prompt); the last {dropped} "
                f"of your {len(attachments)} were left out."
            )
            attachments = attachments[:_MAX_ATTACHMENTS]
        # NOTE: no setTaskText here — the common run path (_cmd_run)
        # broadcasts it for every origin.
        self._printer.broadcast({"type": "status", "running": True, "tabId": tab_id})
        run_cmd: dict[str, Any] = {
            "type": "run",
            "prompt": prompt,
            "model": cmd.get("model", ""),
            "workDir": cmd.get("workDir") or self._agent_server.work_dir,
            "tabId": tab_id,
            # The file tab the user viewed last (the page's Monaco
            # editor), named in the run's system prompt.
            "activeFile": cmd.get("activeFile"),
            "attachments": attachments,
            "isParallel": cmd.get("isParallel", True),
            # Absent/non-bool means "no per-run override": the task
            # runner then falls back to the persisted "Use web tools"
            # setting (config key ``use_web_browser``).
            "useWebTools": cmd.get("useWebTools"),
            # Same contract for pre-run task classification: the task
            # runner falls back to the persisted "Classify tasks
            # before running" setting (config key ``classify_tasks``)
            # when no boolean override is carried.
            "classifyTasks": cmd.get("classifyTasks"),
            # Same contract for persistent memory: absent/non-bool
            # means "no per-run override" and the agent resolves the
            # persisted ``use_memory`` setting itself.
            "useMemory": cmd.get("useMemory"),
            # Carried over from the ``submit`` this run was built from:
            # ``_run_cmd`` bypasses the dispatcher that stamps it, so
            # without this a browser-launched task would record an
            # empty owning connection while a dispatched ``run``
            # records the real one (F08-7).
            "connId": cmd.get("connId", ""),
            "_notice": notice,
        }
        await self._run_cmd(run_cmd)

    def _spawn_cloudflared(
        self,
        args: list[str],
        retries: int = 3,
        launch_prefix: list[str] | None = None,
    ) -> None:
        """Spawn ``cloudflared`` with *args* and a free ``--metrics`` port.

        Records the subprocess in :attr:`_tunnel_proc`, the metrics
        port in :attr:`_tunnel_metrics_port`, and the start time in
        :attr:`_tunnel_started_at`.  The full argv is
        ``cloudflared tunnel --metrics 127.0.0.1:PORT`` followed by
        *args* (e.g. ``["--url", LOCAL]`` for a
        quick tunnel or ``["run", "--token", TOKEN]`` for a named
        tunnel), optionally preceded by the
        :func:`_cloudflared_launch_prefix` ``systemd-run --scope``
        prefix so cloudflared escapes the ``kiss-web.service`` cgroup
        and survives ``systemctl restart`` (keeping the public tunnel
        URL stable across daemon restarts via pidfile adoption).

        The prefix is best-effort: if ``systemd-run`` is missing, the
        spawn falls back to launching cloudflared directly.  An
        immediate exit under the prefix is retried once more WITH the
        prefix on a fresh port (it may be the metrics-port TOCTOU
        below, and losing the cgroup escape would silently reintroduce
        URL rotation); a second immediate exit means ``systemd-run``
        itself is broken (no session D-Bus, cgroup delegation denied,
        …), so the prefix is dropped — a working tunnel that rotates
        on restart beats no tunnel.  Prefix-related failures do not
        consume *retries* attempts (they are bounded on their own), so
        the fallback works even with ``retries=1``.

        M5: there is a small TOCTOU window between
        :func:`_pick_free_local_port` releasing its probe socket and
        ``cloudflared`` binding the same port — another process could
        grab the port in between, causing ``cloudflared`` to exit
        immediately.  When that happens the spawn is retried up to
        *retries* times with a freshly-picked port.

        Args:
            args: Extra arguments after ``--metrics 127.0.0.1:PORT``.
            retries: Maximum number of bind-failure retries.
            launch_prefix: Argv prefix override for tests; ``None``
                computes it via :func:`_cloudflared_launch_prefix`.
        """
        prefix = (
            _cloudflared_launch_prefix() if launch_prefix is None
            else list(launch_prefix)
        )
        last_proc: subprocess.Popen[str] | None = None
        prefixed_failures = 0
        attempt = 0
        max_attempts = max(1, retries)
        while attempt < max_attempts:
            self._tunnel_metrics_port = _pick_free_local_port()
            base_argv = [
                "cloudflared", "tunnel",
                "--metrics",
                f"127.0.0.1:{self._tunnel_metrics_port}",
                *args,
            ]
            try:
                proc = subprocess.Popen(
                    [*prefix, *base_argv],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    # One non-UTF-8 byte in a log line would otherwise
                    # raise ``UnicodeDecodeError`` inside the sole
                    # stderr drain thread and kill it, leaving the live
                    # cloudflared to block on a full pipe.
                    errors="replace",
                    start_new_session=True,
                )
            except FileNotFoundError:
                if not prefix:
                    # cloudflared itself is missing; caller handles.
                    # Release any failed prefixed proc retained above
                    # so its stderr pipe does not linger until GC.
                    if last_proc is not None:
                        _reap_proc(last_proc)
                    raise
                logger.warning(
                    "%s not found; spawning cloudflared inside the "
                    "service cgroup (tunnel URL will rotate on restart)",
                    prefix[0],
                )
                prefix = []
                continue  # Bounded: the prefix is now empty.
            try:
                proc.wait(timeout=_SPAWN_FAILFAST_WINDOW)
            except subprocess.TimeoutExpired:
                pass
            if proc.poll() is None:
                if last_proc is not None:
                    _reap_proc(last_proc)
                with self._tunnel_lock:
                    if self._tunnel_stopped:
                        # ``_stop_tunnel`` already ran (the watchdog
                        # tick that started us was cancelled by
                        # ``stop_async``); publishing now would leak
                        # a live cloudflared past shutdown.
                        _reap_proc(proc, kill=True)
                        raise RuntimeError(
                            "tunnel stopped while cloudflared was starting",
                        )
                    self._tunnel_proc = proc
                    self._tunnel_started_at = time.monotonic()
                    self._tunnel_adopted_pid = None
                    _save_cloudflared_pidfile(
                        proc.pid, self._tunnel_metrics_port, None,
                    )
                return
            if last_proc is not None:
                _reap_proc(last_proc)
            last_proc = proc
            if prefix:
                # Bounded: at most two prefixed failures before the
                # prefix is dropped, and neither consumes an attempt.
                prefixed_failures += 1
                if prefixed_failures >= 2:
                    logger.warning(
                        "cloudflared under %s exited immediately again "
                        "(rc=%s); dropping the cgroup-escape prefix "
                        "(tunnel URL will rotate on restart)",
                        prefix[0], proc.returncode,
                    )
                    prefix = []
                else:
                    logger.warning(
                        "cloudflared under %s exited immediately "
                        "(rc=%s); retrying once more with the prefix "
                        "on a fresh metrics port",
                        prefix[0], proc.returncode,
                    )
                continue
            logger.info(
                "cloudflared exited immediately on metrics port %d "
                "(attempt %d/%d, rc=%s); retrying with fresh port",
                self._tunnel_metrics_port, attempt + 1, max_attempts,
                proc.returncode,
            )
            attempt += 1
        # Same publish handshake as the success path above: after
        # ``_stop_tunnel`` has reset the tunnel state, this executor
        # thread must not overwrite it with a stale (dead) ``Popen``
        # whose stderr pipe would then outlive ``stop_async``.
        with self._tunnel_lock:
            if self._tunnel_stopped and last_proc is not None:
                _reap_proc(last_proc)
                raise RuntimeError(
                    "tunnel stopped while cloudflared was starting",
                )
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
            ["--url", self._local_url],
        )
        assert self._tunnel_proc is not None
        rate_limit_flag = [False]
        url = _read_url_from_stderr(
            self._tunnel_proc, _parse_quick_tunnel_url, timeout=30,
            rate_limit_flag=rate_limit_flag,
        )
        if not url:
            for _ in range(20):
                if self._tunnel_proc.poll() is not None:
                    break
                if self._tunnel_metrics_port is not None:
                    url = _query_quicktunnel_hostname(
                        self._tunnel_metrics_port,
                    )
                if not url:
                    url = _discover_tunnel_url_from_metrics()
                if url:
                    break
                time.sleep(1)
        if url:
            assert self._tunnel_metrics_port is not None
            _save_cloudflared_pidfile(
                self._tunnel_proc.pid, self._tunnel_metrics_port, url,
            )
            return url
        if rate_limit_flag[0]:
            self._tunnel_rate_limited = True
            logger.warning(
                "cloudflared reported HTTP 429 / Cloudflare error "
                "1015 — Cloudflare is rate-limiting "
                "trycloudflare.com quick-tunnels for this egress IP",
            )
        # URL discovery failed for a still-live process (F4-11): kill
        # it, or the watchdog would forever see a healthy tunnel whose
        # public URL is never advertised (only the local URL is), and
        # never retry discovery or restart it.
        if self._tunnel_proc is not None and self._tunnel_proc.poll() is None:
            logger.warning(
                "cloudflared quick-tunnel started but its URL could "
                "not be discovered; terminating it so the watchdog "
                "can start a fresh tunnel",
            )
            self._terminate_tunnel_proc()
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
        if self._shutdown_initiated:
            # The daemon is going down.  The SIGTERM path has already
            # detached the tunnel (bookkeeping says "no tunnel"), so a
            # watchdog tick landing in the shutdown window would
            # otherwise spawn a fresh cloudflared — orphaning it or
            # rotating the public URL for nothing.
            return
        now = time.monotonic()
        proc = self._tunnel_proc
        adopted_pid = self._tunnel_adopted_pid

        if proc is not None and proc.poll() is not None:
            logger.info(
                "cloudflared tunnel process died (rc=%s), restarting…",
                proc.returncode,
            )
            await asyncio.to_thread(self._terminate_tunnel_proc)
            proc = None

        if adopted_pid is not None and not _is_pid_alive(adopted_pid):
            logger.info(
                "Adopted cloudflared (pid=%d) is gone; restarting…",
                adopted_pid,
            )
            # An adopted tunnel never coexists with an own ``Popen``
            # (adoption happens only at start-up, a spawn clears the
            # adopted pid), so the full reset is exact here.
            self._reset_tunnel_proc_state()
            adopted_pid = None

        cfg = await asyncio.to_thread(load_config)
        if self._shutdown_initiated:
            # Shutdown may have started while the config read was in
            # flight (the guard at the top of this method ran before
            # the flag was set) and detached the tunnel for the next
            # daemon to adopt.  Acting on the pre-read snapshot now
            # would wrongly withdraw the URL of a deliberately
            # surviving tunnel.
            return
        if not cfg.get("remote_password", ""):
            if (
                self._tunnel_proc is not None
                or self._tunnel_adopted_pid is not None
            ):
                # The password was cleared while a tunnel was live.
                # Refusing (re)starts is not enough: the running
                # cloudflared keeps the public URL resolving to this
                # server over loopback, where the empty password
                # authenticates — so the tunnel itself must go.
                logger.warning(
                    "the remote password was cleared; terminating the "
                    "live cloudflared tunnel so the public URL stops "
                    "reaching this server.",
                )
                await asyncio.to_thread(self._terminate_tunnel_proc, True)
            if self._active_url and self._active_url != self._local_url:
                # Also withdraw a stale public URL left behind by a
                # tunnel that died on its own (the dead-proc cleanup
                # above deliberately leaves the URL for a replacement
                # tunnel to overwrite — but with no password there
                # will never be a replacement).
                await self._clear_tunnel_url()
            return

        if proc is None and adopted_pid is None:
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
            return
        if healthy:
            self._tunnel_unhealthy_ticks = 0
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
        await asyncio.to_thread(self._terminate_tunnel_proc, True)
        self._tunnel_force_restart_count += 1
        cooldown = min(
            _TUNNEL_FORCE_RESTART_COOLDOWN_INITIAL
                * (2 ** (self._tunnel_force_restart_count - 1)),
            _TUNNEL_FORCE_RESTART_COOLDOWN_MAX,
        )
        self._tunnel_force_restart_next_allowed = now + cooldown
        if now >= self._tunnel_next_retry:
            await self._restart_tunnel_url()

    async def _clear_tunnel_url(self) -> None:
        """Withdraw the advertised public URL after stopping the tunnel.

        Rewrites ``~/.kiss/remote-url.json`` with only the local URL,
        resets :attr:`_active_url`, and broadcasts the change to
        connected clients — the mirror image of what
        :meth:`_restart_tunnel_url` publishes after a start.  Used by
        the watchdog when the ``remote_password`` is cleared while a
        tunnel is live.
        """
        await asyncio.to_thread(self._write_url_file_sync, None)
        self._active_url = self._local_url
        await self._broadcast_remote_url(self._active_url)
        await self._post_url_if_changed()

    async def _restart_tunnel_url(self) -> None:
        """Start a fresh tunnel and refresh ``~/.kiss/remote-url.json``.

        Always rewrites the URL file (even on failure, so stale data
        does not linger), updates :attr:`_active_url`, and broadcasts
        ``remote_url`` to connected clients.  On failure schedules an
        exponential backoff via :attr:`_tunnel_next_retry`.
        """
        if self._shutdown_initiated:
            # Closes the race where a watchdog tick already past its
            # own shutdown guard reaches here after the SIGTERM path
            # detached the tunnel: spawning a fresh cloudflared now
            # would orphan it or needlessly rotate the public URL.
            return
        assert self._loop is not None
        tunnel_url = await self._loop.run_in_executor(
            None, self._start_tunnel,
        )
        if self._shutdown_initiated:
            # SIGTERM landed while the tunnel start was in flight (the
            # entry guard above ran before the flag was set).  The
            # sigterm thread's early detach has already run, so detach
            # this fresh cloudflared as well — its pid/URL were saved
            # to the pidfile by the start path, so the NEXT daemon
            # adopts it — and publish nothing from a dying process.
            self._detach_tunnel()
            return
        if tunnel_url:
            logger.info("Tunnel restarted: %s", tunnel_url)
            self._tunnel_failure_count = 0
            self._tunnel_next_retry = 0.0
            self._tunnel_rate_limited = False
        else:
            self._tunnel_failure_count += 1
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
        await asyncio.to_thread(self._write_url_file_sync, tunnel_url)
        self._active_url = tunnel_url or self._local_url
        await self._broadcast_remote_url(self._active_url)
        await self._post_url_if_changed()

    def _terminate_tunnel_proc(self, kill_adopted: bool = False) -> None:
        """Terminate ``_tunnel_proc`` and reset per-process state.

        Resets :attr:`_tunnel_proc`, :attr:`_tunnel_metrics_port`,
        :attr:`_tunnel_started_at`, :attr:`_tunnel_unhealthy_ticks`,
        and :attr:`_tunnel_adopted_pid` (via
        :meth:`_reset_tunnel_proc_state`) so the next restart starts
        cleanly.  Leaves
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
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    # Keep shutdown bounded, but retain ownership until
                    # a delayed SIGKILL can take effect and be reaped.
                    threading.Thread(
                        target=proc.wait,
                        name=f"kiss-cloudflared-reaper-{proc.pid}",
                        daemon=True,
                    ).start()
            _unlink_cloudflared_pidfile()
        elif kill_adopted and self._tunnel_adopted_pid is not None:
            # An adopted pid came from the pidfile of a PREVIOUS
            # process, so unlike the self-spawned ``proc`` above it
            # may have been recycled for an unrelated process since
            # adoption.  Verify the identity before every signal
            # (same rule as _terminate_declined_cloudflared).
            adopted_pid = self._tunnel_adopted_pid
            if _looks_like_cloudflared(adopted_pid):
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
                        if _looks_like_cloudflared(adopted_pid):
                            try:
                                os.kill(
                                    adopted_pid,
                                    getattr(signal, "SIGKILL", signal.SIGTERM),
                                )
                            except (
                                ProcessLookupError,
                                PermissionError,
                                OSError,
                            ):
                                pass
            else:
                logger.info(
                    "Adopted pid %d is no longer a cloudflared process "
                    "(pid recycled); not signalling it", adopted_pid,
                )
            _unlink_cloudflared_pidfile()
        self._reset_tunnel_proc_state()

    async def _ping_one_ws(self, ws: Any) -> None:
        """Send a ping to a single WebSocket client, closing if stale.

        A client that answers the protocol-level ping is then sent an
        application-level ``{"type": "heartbeat"}`` frame.  The browser
        answers pings inside its network stack, invisible to page
        JavaScript, so this frame is the only regular proof of life
        the webapp's WebSocket transport (``media/transport.js``) can
        observe: a page that sees no frame at all for 45 s treats its
        socket as half-open and reconnects.

        Every step is bounded.  ``Connection.ping`` waits for the
        transport to drain before it returns the pong waiter, so a
        peer that stopped reading (full socket buffers behind a
        half-open connection) held this coroutine, and with it the
        whole watchdog tick, until some other frame's
        :meth:`WebPrinter._timed_send` happened to abort the peer —
        forever on an idle daemon.  The ping frame gets the tolerance
        every frame gets (:attr:`WebPrinter._send_timeout`), the pong
        and the heartbeat :data:`_WS_PING_TIMEOUT`.  Any failure
        aborts the transport, which ends the connection's handler;
        ``close()`` would wait for the same drain.
        """
        try:
            pong = await asyncio.wait_for(
                ws.ping(), timeout=self._printer._send_timeout,
            )
            await asyncio.wait_for(pong, timeout=_WS_PING_TIMEOUT)
            await asyncio.wait_for(
                self._endpoint_send(ws, _WS_HEARTBEAT_FRAME),
                timeout=_WS_PING_TIMEOUT,
            )
        except Exception:
            transport = getattr(ws, "transport", None)
            if transport is not None:
                transport.abort()

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
            return
        self._latest_version = latest
        await self._broadcast_update_available()

    async def _version_check_loop(self) -> None:
        """Run :meth:`_check_for_update` every hour.

        The very first check runs immediately so clients learn about
        a pending upgrade as soon as the daemon starts, instead of
        waiting an entire hour for the first tick.
        """
        while True:
            try:
                await self._check_for_update()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Version check failed", exc_info=True)
            await asyncio.sleep(_VERSION_CHECK_INTERVAL)

    async def _machines_registry_watch(self) -> None:
        """Push the machine list to every client when the registry file changes.

        ``$KISS_HOME/machines.json`` is written by this daemon's own
        ``machineAction`` and, from other processes, by a ``/machine
        add`` task or the CLI.  Its size and mtime are read every
        :data:`MACHINES_REGISTRY_POLL_SECONDS`; a change after the
        first look sends a ``machines`` reply (pinged anew) to every
        connection, so each surface's MACHINES subpanel refreshes and
        unfolds for a new machine without waiting for its own poll.
        """
        from kiss.agents.sorcar.machines import machines_file

        seen = _file_stamp(machines_file())
        while True:
            await asyncio.sleep(MACHINES_REGISTRY_POLL_SECONDS)
            try:
                stamp = _file_stamp(machines_file())
                if stamp == seen:
                    continue
                seen = stamp
                await self._server_api.broadcast_machines()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("machines registry watch failed", exc_info=True)

    async def _watchdog(self) -> None:
        """Unified periodic watchdog (runs every :data:`TUNNEL_CHECK_INTERVAL`).

        Each tick performs four checks:

        1. **Tunnel health** — if the ``cloudflared`` process died
           (e.g. macOS killed it during sleep), restart it.
        2. **URL-file presence** — re-write ``~/.kiss/remote-url.json``
           if it has been removed (e.g. by a developer's pytest run
           that touches the real file, or by an unrelated cleanup).
           Without this ``kiss-web --url`` cannot discover the active
           URL.
        3. **IP change** — if the host's network addresses changed
           (WiFi switch, DHCP renewal, VPN): in direct-LAN mode
           (``use_tunnel=False``) initiate a graceful shutdown so the
           daemon manager restarts the process on the new address; in
           tunnel mode only log the change — ``cloudflared``
           re-registers with the edge automatically, so no restart is
           needed.
        4. **Loopback alias** — on macOS, retry binding
           ``127.0.0.1:port`` while another process (typically a VS
           Code forwarded port) holds it; see
           :meth:`_bind_loopback_alias`.
        5. **WebSocket ping** — send a ping to every connected client
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
                await asyncio.to_thread(self._watchdog_check_url_file)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog URL-file check error", exc_info=True)
            try:
                ips = await asyncio.to_thread(_get_local_ips)
                if self._watchdog_check_ip_change(ips):
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog IP check error", exc_info=True)
            try:
                await self._watchdog_reclaim_loopback()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog loopback reclaim error", exc_info=True)
            # Single-flight background task: re-issuing a certificate
            # (key generation, signing, lock wait) must not delay the
            # tick and the IP-change restart decision above.
            if self._tls_refresh_task is None or self._tls_refresh_task.done():
                self._tls_refresh_task = asyncio.create_task(
                    self._refresh_tls_cert_logged(),
                )
            try:
                await self._watchdog_ping_clients()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Watchdog WS ping error", exc_info=True)

    def _reload_trusted_clients(self) -> None:
        """Load the newly enrolled machine-link CAs into the live SSL context.

        See :func:`_load_trusted_client_cas`; a no-op without a context
        or when the directory holds nothing new.
        """
        if self._ssl_context is None:
            return
        for name in _load_trusted_client_cas(self._ssl_context, self._trusted_client_cas):
            logger.info("Admitting machine links with certificates issued by %s", name)

    async def _watch_trusted_clients(self) -> None:
        """Reload the enrolled CAs soon after their directory changes.

        A machine enrolling itself (``/machine add user@host`` on the
        other daemon) writes its CA into :func:`_trusted_clients_dir`
        over ssh and then waits for its first handshake to succeed, so
        the directory's mtime is polled every
        :data:`TRUSTED_CLIENTS_POLL` seconds (one ``stat``), with a
        full rescan every :data:`TUNNEL_CHECK_INTERVAL` seconds in
        case a file system's coarse mtime hid a write.  Polling rather
        than a signal from the enrolling command: a signal sent to a
        daemon that predates the handler, or to a pid reused since a
        stale endpoint file was written, would kill it.
        """
        elapsed = 0.0
        while True:
            try:
                mtime = _trusted_clients_dir().stat().st_mtime
            except OSError:
                mtime = -1.0
            if mtime != self._trusted_clients_mtime or elapsed >= TUNNEL_CHECK_INTERVAL:
                self._trusted_clients_mtime, elapsed = mtime, 0.0
                try:
                    self._reload_trusted_clients()
                except Exception:
                    logger.warning("Could not reload the enrolled machine CAs", exc_info=True)
            await asyncio.sleep(TRUSTED_CLIENTS_POLL)
            elapsed += TRUSTED_CLIENTS_POLL

    async def _refresh_tls_cert_logged(self) -> None:
        """Run :meth:`_refresh_tls_cert`, logging any failure (watchdog task body)."""
        try:
            await self._refresh_tls_cert()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Could not re-issue the TLS server certificate", exc_info=True,
            )

    async def _refresh_tls_cert(self) -> None:
        """Keep the live context's auto-generated certificate current.

        Called every watchdog tick.  The certificate must be re-issued
        when the machine's LAN IPs change (in tunnel mode an IP change
        does not restart the daemon, see :meth:`_watchdog_check_ip_change`,
        and the new ``https://<lan-ip>:PORT`` URL would fail hostname
        verification even on a browser that trusts the local CA) and
        when it is expiring (a daemon that stays up for years must not
        end up serving an expired certificate).  A sibling daemon may
        also have re-issued it.  All three cases reduce to: make the
        pair on disk current for :attr:`_last_ips`
        (:func:`_refresh_local_tls_pair`, off-thread, a no-op when it
        already is) and, when the certificate on disk differs from the
        one loaded, hot-load it (:func:`_reload_local_tls_pair_if_unlocked`,
        on the loop thread).  New handshakes then present the new
        certificate; existing connections are untouched.  Explicit
        ``certfile``/``keyfile`` pairs are never rewritten.
        """
        if self._ssl_certfile or self._ssl_context is None:
            return
        cert_pem = await asyncio.to_thread(_refresh_local_tls_pair, self._last_ips)
        if cert_pem == self._tls_loaded_cert or self._ssl_context is None:
            return
        if _reload_local_tls_pair_if_unlocked(self._ssl_context, cert_pem):
            first_load = not self._tls_loaded_cert
            self._tls_loaded_cert = cert_pem
            logger.log(
                logging.DEBUG if first_load else logging.INFO,
                "Reloaded the TLS server certificate (LAN IPs %s)",
                sorted(self._last_ips),
            )

    def _watchdog_check_url_file(self) -> None:
        """Re-write ``~/.kiss/remote-url.json`` if it went missing.

        A developer's pytest run that touches the real file, or an
        unrelated cleanup, can remove it; without a re-write the
        installer's ``wait-for-public-url.sh`` and another machine's
        ``machines.py`` probe cannot discover the active URL.
        """
        if not self._url_file.is_file():
            tunnel_url = (
                self._active_url
                if self._active_url and self._active_url != self._local_url
                else None
            )
            self._write_url_file_sync(tunnel_url)
            logger.info(
                "Re-wrote missing URL file %s (tunnel=%s)",
                self._url_file, tunnel_url,
            )

    def _watchdog_check_ip_change(
        self, current_ips: frozenset[str] | None = None,
    ) -> bool:
        """Detect a debounced local-IP change and initiate a restart.

        Args:
            current_ips: Pre-fetched :func:`_get_local_ips` result
                (the watchdog fetches it off-thread, M10); when
                ``None``, fetched synchronously here.

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
        if current_ips is None:
            current_ips = _get_local_ips()
        if not current_ips:
            self._pending_ip_change = None
            self._pending_ip_change_count = 0
        elif current_ips == self._last_ips:
            self._pending_ip_change = None
            self._pending_ip_change_count = 0
        elif not self._last_ips:
            self._last_ips = current_ips
            self._pending_ip_change = None
            self._pending_ip_change_count = 0
            self._republish_urls()
        else:
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
                    logger.info(
                        "IP address changed: %s → %s; tunnel "
                        "mode — cloudflared will re-register "
                        "automatically",
                        prev_ips,
                        current_ips,
                    )
                    self._republish_urls()
                else:
                    logger.info(
                        "IP address changed: %s → %s, "
                        "restarting server…",
                        prev_ips,
                        current_ips,
                    )
                    self._close_ws_listeners()
                    return True
        return False

    def _close_ws_listeners(self, remove_endpoint_file: bool = True) -> None:
        """Stop accepting on the wildcard and loopback WSS listeners.

        ``close()`` only; callers that must wait for the sockets to be
        released (:meth:`stop_async`) await ``wait_closed()`` themselves.
        The wildcard server object is kept so its pending
        ``serve_forever()`` observes the close.

        Args:
            remove_endpoint_file: Also withdraw the local endpoint file
                (inline; the sidecar flock wait is bounded by
                ``local_endpoint._LOCK_TIMEOUT``).  :meth:`stop_async`
                passes ``False`` and removes it off-loop instead.
        """
        if self._ws_server is not None:
            self._ws_server.close()
        if self._ws_loopback_server is not None:
            self._ws_loopback_server.close()
        if remove_endpoint_file:
            # Nothing can connect any more: the endpoint file must not
            # keep advertising this daemon (a rebind publishes a fresh
            # one).
            local_endpoint.remove_endpoint_if_owned(
                self._local_endpoint_file, self._local_token,
            )

    async def _watchdog_ping_clients(self) -> None:
        """Ping every connected WSS client, closing unresponsive ones.

        Delegates the per-connection timeout/close logic to
        :meth:`_ping_one_ws`; exceptions from individual pings are
        collected via ``return_exceptions`` so one bad client cannot
        skip the rest.
        """
        connections = [
            ws
            for server in (self._ws_server, self._ws_loopback_server)
            if server is not None
            for ws in server.connections
        ]
        if connections:
            await asyncio.gather(
                *[self._ping_one_ws(ws) for ws in connections],
                return_exceptions=True,
            )

    def _reset_tunnel_proc_state(self) -> None:
        """Reset per-process tunnel bookkeeping.

        Shared by :meth:`_terminate_tunnel_proc` and
        :meth:`_detach_tunnel` so the next tunnel (re)start begins
        from a clean slate.
        """
        self._tunnel_proc = None
        self._tunnel_adopted_pid = None
        self._tunnel_metrics_port = None
        self._tunnel_started_at = None
        self._tunnel_unhealthy_ticks = 0

    def _reset_tunnel_backoff_state(self) -> None:
        """Reset tunnel backoff counters and clear the active URL.

        Shared by :meth:`_stop_tunnel` and :meth:`_detach_tunnel`.
        """
        self._tunnel_failure_count = 0
        self._tunnel_next_retry = 0.0
        self._tunnel_rate_limited = False
        self._active_url = None

    def _stop_tunnel(self) -> None:
        """Terminate the tunnel process and reset all tunnel state.

        Calls :meth:`_terminate_tunnel_proc` (which resets per-process
        state), then clears the backoff counters and active URL.  Does
        not delete ``~/.kiss/remote-url.json`` because a replacement
        daemon may have already overwritten it; removing it would
        race with the new instance's ``_save_url_file`` and leave
        ``kiss-web --url`` with no URL.

        Sets :attr:`_tunnel_stopped` under :attr:`_tunnel_lock` so an
        in-flight executor :meth:`_start_tunnel` (whose watchdog task
        was cancelled, not stopped) kills its own cloudflared instead
        of publishing it after this method has already returned.
        """
        with self._tunnel_lock:
            self._tunnel_stopped = True
            self._terminate_tunnel_proc()
        self._reset_tunnel_backoff_state()

    def _detach_tunnel(self) -> None:
        """Reset tunnel bookkeeping without killing ``cloudflared``.

        Used by :meth:`start`'s shutdown ``finally`` so that a
        ``kiss-web`` exit (SIGTERM / KeyboardInterrupt / launchd
        restart / the installer's ``pkill kiss-web``) does **not**
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
        would briefly leave ``kiss-web --url`` with no URL.
        """
        proc = self._tunnel_proc
        if proc is not None and proc.poll() is None:
            self._spawn_stderr_drain_shim(proc)
        self._reset_tunnel_proc_state()
        self._reset_tunnel_backoff_state()

    @staticmethod
    def _spawn_stderr_drain_shim(
        proc: subprocess.Popen[str],
        launch_prefix: list[str] | None = None,
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

        Under systemd the shim must escape the service cgroup exactly
        like cloudflared itself (:func:`_cloudflared_launch_prefix`):
        a ``systemctl restart`` kills every cgroup member, and killing
        the shim closes the pipe's last read end — the very
        ``SIGPIPE`` this shim exists to prevent would then take down
        the escaped cloudflared indirectly.  A prefixed shim that
        exits within 0.2s (broken user manager) falls back to a plain
        ``cat``, which at worst restores the pre-fix behaviour.

        Best-effort: a ``cat`` spawn failure (missing binary, EMFILE,
        permission error) is logged at DEBUG and otherwise ignored.
        The worst case is a return to the pre-fix behaviour for that
        particular shutdown — ``cloudflared`` may die from
        ``SIGPIPE`` and the next ``kiss-web`` will mint a fresh
        public URL — which is still no worse than no detach at all.

        Args:
            proc: The ``cloudflared`` subprocess; must have been
                started with ``stderr=PIPE``.
            launch_prefix: Argv prefix override for tests; ``None``
                computes it via :func:`_cloudflared_launch_prefix`.

        Returns:
            The detached shim's ``Popen`` handle on success, or
            ``None`` if no stderr pipe was available or the shim
            spawn failed.
        """
        stderr = proc.stderr
        if stderr is None:
            return None
        prefix = (
            _cloudflared_launch_prefix() if launch_prefix is None
            else list(launch_prefix)
        )
        candidates: list[list[str]] = [["cat"]]
        if prefix:
            candidates.insert(0, [*prefix, "cat"])
        for argv in candidates:
            try:
                shim = subprocess.Popen(
                    argv,
                    stdin=stderr.fileno(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )
            except (OSError, ValueError):
                logger.debug(
                    "Failed to spawn stderr drain shim via %r",
                    argv,
                    exc_info=True,
                )
                continue
            if len(argv) > 1:
                # Prefixed spawn: confirm systemd-run did not fail
                # outright before trusting the shim with the pipe.
                try:
                    shim.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    return shim
                logger.debug(
                    "Prefixed stderr drain shim exited immediately "
                    "(rc=%s); falling back to a plain cat",
                    shim.returncode,
                )
                continue
            return shim
        return None


    async def _setup_server(self) -> None:
        """Shared setup for both blocking and async server start.

        Binds the WebSocket server, starts the tunnel (if enabled),
        saves the URL file, and starts watchdog tasks.
        """
        self._loop = asyncio.get_running_loop()
        self._printer._loop = self._loop
        # No database write may run here: the legacy side-channel stamp
        # and the orphan sweep live on AgentServer's background thread
        # so a locked history.db never delays binding the listeners.
        try:
            # Sign-in pages that connectors hand to the user open in the
            # streamed Browser tab, focused on every surface, while this
            # daemon is up (kiss.core.browser_handoff.open_for_user).
            # Inside the rollback scope: a setup that fails or is
            # cancelled unregisters it again in _close_partial_setup.
            set_browser_tab_opener(self._agent_server.browser_tabs.open_for_user)
            await self._bind_listeners()
        except BaseException:
            # Rollback (F4-04): a TLS/WSS/tunnel failure or a
            # cancellation must not leave a half-bound WSS listener
            # live in an embedder that catches the exception.
            self._close_partial_setup()
            raise

    def _close_partial_setup(self) -> None:
        """Tear down listeners bound by a failed/cancelled ``_setup_server``."""
        # No surfaces will ever attach to this server: sign-in pages
        # must not be sent to its Browser tab.
        set_browser_tab_opener(None)
        local_endpoint.remove_endpoint_if_owned(
            self._local_endpoint_file, self._local_token,
        )
        if self._trusted_clients_task is not None:
            self._trusted_clients_task.cancel()
            self._trusted_clients_task = None
        if self._ws_server is not None:
            self._ws_server.close()
            self._ws_server = None
        if self._ws_loopback_server is not None:
            self._ws_loopback_server.close()
            self._ws_loopback_server = None

    async def _serve_wss(self, host: str) -> WebSocketServer:
        """Start a WSS listener for this server on ``host:self.port``.

        Every listener (the wildcard one and the loopback alias of
        :meth:`_bind_loopback_alias`) is created here so they share the
        handler, TLS context and connection limits.  Raises ``OSError``
        when the address cannot be bound.

        The bind is shielded: ``loop.create_server`` has a cancellation
        point after the sockets are listening, so a cancel landing
        there (a cancelled startup, or the watchdog's loopback reclaim
        cancelled by :meth:`stop_async`) would otherwise drop the
        ``Server`` object and leave the port bound until process exit.
        On cancellation the listener is closed as soon as it exists.

        The library's keep-alive is off because the daemon runs its
        own: the watchdog pings every connection every
        :data:`TUNNEL_CHECK_INTERVAL` seconds and closes one that does
        not answer within :data:`_WS_PING_TIMEOUT`
        (:meth:`_ping_one_ws`), then sends the ``heartbeat`` frame
        ``media/transport.js`` watches.
        """
        bind = asyncio.ensure_future(serve(
            self._ws_handler,
            host,
            self.port,
            process_request=self._process_request,
            open_timeout=_OPEN_TIMEOUT_SECONDS,
            ping_interval=None,
            ping_timeout=None,
            max_size=_MAX_LINE_BYTES,
            # Each connection sniffs its first byte and speaks TLS or
            # plain HTTP accordingly (``_HeadAwareServerConnection``),
            # so the listener itself is created without ``ssl``.
            create_connection=cast(
                "type[ServerConnection]",
                functools.partial(
                    _HeadAwareServerConnection, ssl_context=self._ssl_context,
                ),
            ),
        ))
        try:
            return await asyncio.shield(bind)
        except asyncio.CancelledError:
            bind.add_done_callback(_close_orphaned_listener)
            raise

    def _wants_loopback_alias(self) -> bool:
        """Whether this server should also bind ``127.0.0.1`` explicitly.

        A wildcard listener on a BSD-derived kernel (macOS) needs it:
        there another process may bind the specific loopback address of
        the same port beside our ``0.0.0.0`` socket and take over every
        ``127.0.0.1`` connection.  Linux refuses that bind while the
        wildcard listener exists.  A listener on a single non-loopback
        address (``--host 192.168.1.5``) needs it on every platform:
        same-machine clients prove they are local by connecting from a
        loopback address, which that listener cannot accept.
        """
        if self.host in ("", "0.0.0.0", "::"):
            return sys.platform == "darwin"
        return self.host != "localhost" and not sorcar_api.is_loopback_ip(self.host)

    async def _bind_loopback_alias(self) -> bool:
        """Bind ``127.0.0.1:port`` beside the wildcard listener (macOS).

        On macOS a ``0.0.0.0:8787`` listener does not stop another
        process from binding ``127.0.0.1:8787`` with ``SO_REUSEADDR``,
        and loopback connections then go to that more specific socket.
        VS Code's Remote-SSH port forwarding does exactly this when the
        remote machine also runs kiss-web on 8787: the forward binds
        ``127.0.0.1:8787`` on the laptop and ``http://127.0.0.1:8787``
        silently reaches the remote daemon, whose certificate the
        laptop does not trust.  Holding ``127.0.0.1`` ourselves makes
        that bind fail with ``EADDRINUSE``, so VS Code maps the forward
        to another local port instead.

        When another process already holds the loopback address, a
        warning names it (``lsof``) and the watchdog retries every
        :data:`TUNNEL_CHECK_INTERVAL` seconds so the address is
        reclaimed as soon as it is released.  Returns True when this
        server holds ``127.0.0.1:port`` (or does not need it).
        """
        if self._ws_loopback_server is not None or not self._wants_loopback_alias():
            return True
        try:
            self._ws_loopback_server = await self._serve_wss("127.0.0.1")
        except OSError as exc:
            holders = await asyncio.to_thread(_describe_port_listeners, self.port)
            logger.warning(
                "127.0.0.1:%d is bound by another process (%s): %s. "
                "http://127.0.0.1:%d reaches that process, not this "
                "server, until it releases the port (a VS Code forwarded "
                "port: Ports view → Stop Forwarding Port); retrying every "
                "%ds. Meanwhile use https://%s.local:%d",
                self.port, holders or "unknown", exc, self.port,
                TUNNEL_CHECK_INTERVAL, platform.node().split(".")[0],
                self.port,
            )
            return False
        return True

    async def _watchdog_reclaim_loopback(self) -> None:
        """Retry :meth:`_bind_loopback_alias` while another process holds ``127.0.0.1``."""
        if self._ws_loopback_server is not None or not self._wants_loopback_alias():
            return
        if await self._bind_loopback_alias():
            logger.info(
                "Reclaimed 127.0.0.1:%d: http://127.0.0.1:%d reaches this "
                "server again",
                self.port, self.port,
            )
            # Local clients may have been pointed at a non-loopback
            # address meanwhile: publish the loopback one.
            self._write_local_endpoint_file()

    def _write_local_endpoint_file(self) -> None:
        """Publish this daemon's URL and local token for same-machine clients.

        Called once the WSS listener is bound.  ``ca`` names the PEM
        file a local client must trust.  With auto-generated
        certificates that is the local CA under the TLS dir; with a
        custom ``--certfile`` it is the ``ca.pem`` beside it when one
        exists (the layout :func:`_generate_self_signed_cert` produces),
        else the certificate itself (a self-signed certificate is its
        own trust anchor).
        """
        if self._ssl_certfile and self._ssl_keyfile:
            sibling_ca = Path(self._ssl_certfile).parent / tls_certs.CA_CERT_FILE
            ca_path = str(sibling_ca) if sibling_ca.is_file() else self._ssl_certfile
        else:
            ca_path = str(_tls_dir() / tls_certs.CA_CERT_FILE)
        host, port = _bound_loopback(self._ws_loopback_server, self._ws_server)
        local_endpoint.write_endpoint(
            self._local_endpoint_file,
            local_endpoint.LocalEndpoint(
                url=f"ws://{host}:{port}/ws",
                token=self._local_token,
                ca=ca_path,
                pid=os.getpid(),
            ),
        )

    async def _bind_listeners(self) -> None:
        """Bind the WSS listener(s), publish the local endpoint, start the tunnel."""
        if self._ssl_context is None:
            lan_ips = await asyncio.to_thread(_get_local_ips)
            self._ssl_context = await asyncio.to_thread(
                _create_ssl_context,
                self._ssl_certfile,
                self._ssl_keyfile,
                lan_ips,
            )
            self._reload_trusted_clients()
            # ``_tls_loaded_cert`` stays empty: the first watchdog tick
            # loads the on-disk certificate under the lock
            # (:meth:`_refresh_tls_cert`), which is race-free — reading
            # the file here, after the lock was released, could record a
            # sibling's newer certificate the context is not serving.

        last_err: OSError | None = None
        for attempt in range(_BIND_RETRY_ATTEMPTS):
            try:
                self._ws_server = await self._serve_wss(self.host)
                if self.port == 0:
                    # An ephemeral port (tests, embedders): record the
                    # one the OS picked so the URL file, the endpoint
                    # file and the loopback alias name the real port.
                    self.port = _bound_loopback(self._ws_server)[1]
                    self._local_url = f"http://localhost:{self.port}"
                break
            except OSError as exc:
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
        await self._bind_loopback_alias()
        # Local clients (``run_agent`` and other ``daemon_client``
        # runs) may connect from here on: the endpoint file is written before
        # the tunnel wait below so they need not wait out the
        # password/tunnel startup.  Written inline (a small file under
        # the endpoint lock) so a cancellation of this startup cannot
        # race a detached writer that publishes after the rollback.
        self._write_local_endpoint_file()
        # A machine enrolling itself finds the endpoint file from here
        # on, so its CA must be picked up from here on too, not after
        # the tunnel wait below (cancelled with the listeners by
        # _close_partial_setup).
        self._trusted_clients_task = asyncio.create_task(self._watch_trusted_clients())

        tunnel_url: str | None = None
        if self.use_tunnel:
            # Close the empty-password startup window FIRST: the WSS
            # listener is already up, so an orphaned cloudflared left
            # by the previous instance is already relaying internet
            # visitors to it as loopback peers — whom an empty
            # password would authenticate.  Waiting up to 30 s for a
            # password before acting (below) would leave that tunnel
            # publicly usable for the whole wait, so when the password
            # is empty RIGHT NOW the orphan is terminated immediately.
            # Cost of the eager kill: a password saved during the wait
            # rotates the public URL instead of re-adopting it.
            initial_cfg = await asyncio.to_thread(load_config)
            if not initial_cfg.get("remote_password", ""):
                await asyncio.to_thread(
                    _terminate_orphan_cloudflared, self.port,
                )
            password = await asyncio.to_thread(
                _wait_for_remote_password, 30.0,
            )
            own_tunnel_pid: int | None = None
            if password:
                adopted = await asyncio.to_thread(
                    _try_adopt_existing_cloudflared, self.port,
                )
                if adopted is not None:
                    adopted_pid, adopted_port, adopted_url = adopted
                    own_tunnel_pid = adopted_pid
                    self._tunnel_adopted_pid = adopted_pid
                    self._tunnel_metrics_port = adopted_port
                    self._tunnel_started_at = time.monotonic()
                    tunnel_url = adopted_url
                    _save_cloudflared_pidfile(
                        adopted_pid, adopted_port, adopted_url,
                    )
            if not password:
                await asyncio.to_thread(
                    _terminate_orphan_cloudflared, self.port,
                )
                logger.warning(
                    "remote_password is not set in ~/%s/config.json; "
                    "refusing to start the cloudflared tunnel and "
                    "refusing non-localhost connections.  Run "
                    "install.sh to set a password and enable remote "
                    "access.",
                    HOME_DIR,
                )
                print(
                    "Warning: no remote password is set; cloudflared "
                    "tunnel disabled and non-localhost connections "
                    "refused.  Run install.sh to set one.",
                    file=sys.stderr,
                )
            elif tunnel_url is None:
                tunnel_url = await asyncio.to_thread(
                    self._start_tunnel,
                )
                spawned = self._tunnel_proc
                if spawned is not None:
                    own_tunnel_pid = spawned.pid
            if own_tunnel_pid is not None:
                # Our tunnel is settled (adopted or spawned); any OTHER
                # cloudflared still forwarding to this port is a
                # leftover from a daemon that lost its pidfile or ran
                # from another home directory — unmonitored, yet its
                # public URL reaches this server.  Only when our own
                # pid is positively known: a SIGTERM landing during
                # startup runs ``_detach_tunnel`` on another thread,
                # which clears ``_tunnel_proc``/``_tunnel_adopted_pid``
                # while leaving the tunnel alive for the next daemon —
                # a cleanup with ``keep_pid=None`` would kill exactly
                # that tunnel.
                await asyncio.to_thread(
                    _terminate_stray_cloudflared, self.port, own_tunnel_pid,
                )

        self._last_ips = await asyncio.to_thread(_get_local_ips)
        self._ips_probed = True
        await asyncio.to_thread(self._write_url_file_sync, tunnel_url)
        self._active_url = tunnel_url or self._local_url
        await self._post_url_if_changed()
        self._watchdog_task = asyncio.create_task(self._watchdog())
        self._version_check_task = asyncio.create_task(
            self._version_check_loop(),
        )
        self._machines_watch_task = asyncio.create_task(
            self._machines_registry_watch(),
        )

        self._start_sea_command_watcher()
        self._maybe_schedule_server_reset_complete()

    def _start_sea_command_watcher(self) -> None:
        """Start the SEA slash-command registry watcher.

        Registers a subscriber that broadcasts a ``seaCommands`` event
        to every connected client on any registry change (a SEA added
        or removed from a watched folder, or ``SEAS.md`` edited), then
        launches the background poller.  Idempotent per server
        instance: the ``_sea_command_subscribed`` guard prevents a
        second call from stacking duplicate subscribers, so a rebound
        listener or a test-time re-setup cannot fan a single rescan
        out twice.
        """
        if self._sea_command_subscribed:
            return
        self._sea_command_subscribed = True
        from kiss.agents.sorcar import sea_commands

        sea_commands.subscribe(self._on_sea_commands_changed)
        sea_commands.start_registry_watcher()

    def _on_sea_commands_changed(self, commands: list[str]) -> None:
        """Broadcast the new SEA slash-command list to every client.

        Args:
            commands: The sorted command names after a registry change.
        """
        self._printer.broadcast({"type": "seaCommands", "commands": commands})

    def _stop_sea_command_watcher(self) -> None:
        """Unsubscribe from the SEA registry and stop its poller.

        Shared by both shutdown paths (the blocking ``start()``
        cleanup and :meth:`stop_async`) so a fresh server rebound in
        the same process never inherits a dead subscriber bound to the
        old printer.  Blocking: the poller join waits up to 5 s.
        """
        from kiss.agents.sorcar import sea_commands

        if self._sea_command_subscribed:
            sea_commands.unsubscribe(self._on_sea_commands_changed)
            self._sea_command_subscribed = False
        sea_commands.stop_registry_watcher()

    async def _serve_async(self) -> None:
        """Internal async entry point for the server.

        Serves until either the WSS listener stops on its own (its
        exception is re-raised) or :meth:`_request_loop_shutdown`
        resolves :attr:`_shutdown_future` — the deterministic
        SIGTERM/"Reset Server" shutdown path, which cannot be swallowed
        by whatever coroutine the loop happens to be executing (unlike
        an injected ``KeyboardInterrupt``).
        """
        await self._setup_server()
        print(f"{PRODUCT_NAME} remote access: {self._local_url}", file=sys.stderr)
        print(f"Local machine:             {self._loopback_url}", file=sys.stderr)
        for lan_url in self._lan_urls():
            print(f"LAN:                       {lan_url}", file=sys.stderr)
        if self.use_tunnel and self._active_url != self._local_url:
            print(f"Cloudflare tunnel:         {self._active_url}", file=sys.stderr)
        elif self.use_tunnel:
            print("Warning: cloudflared tunnel failed to start", file=sys.stderr)
        # Scheduled automations (cron) run in a background daemon
        # thread for the daemon's whole lifetime; prompt jobs are
        # submitted back to this daemon through its own local WSS
        # endpoint.  Only this blocking lifecycle (the real `kiss-web`
        # daemon) owns the scheduler: `start_async()` embedders —
        # in-process helper daemons and tests — must not fire the
        # user's scheduled jobs.  The default jobs every installation
        # gets (the 6-hourly /recommendations run) are seeded first, once
        # per home.
        cron_agent.seed_default_jobs()
        cron_stop = cron_agent.start_scheduler_thread(
            endpoint_file=str(self._local_endpoint_file),
        )
        loop = asyncio.get_running_loop()
        self._shutdown_future = loop.create_future()
        serve_task: asyncio.Task[None] = asyncio.ensure_future(
            self._ws_server.serve_forever(),  # type: ignore[union-attr]
        )
        try:
            await asyncio.wait(
                {serve_task, self._shutdown_future},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            # Joins the scheduler thread (up to 30 s when a tick is
            # stuck on a slow disk): off the loop, which is still
            # delivering the final frames to connected clients.
            await asyncio.to_thread(cron_agent.stop_scheduler_thread, cron_stop)
            self._close_ws_listeners()
            if not serve_task.done():
                serve_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await serve_task
        if serve_task.done() and not serve_task.cancelled():
            exc = serve_task.exception()
            if exc is not None:
                raise exc

    def _handle_shutdown_signal(
        self, signum: int, _frame: Any = None,
    ) -> None:
        """React to a catchable termination signal (SIGTERM / SIGHUP).

        Logs the signal alongside a snapshot of in-flight agent tasks
        (via :func:`_snapshot_active_tabs`, which is signal-safe) and
        current memory.  For ``SIGTERM`` the *first* invocation starts
        the :meth:`_shutdown_on_sigterm` thread, which stops the
        in-flight agent worker threads and then unwinds the event loop
        deterministically (via :meth:`_request_loop_shutdown`) so
        ``asyncio.run`` in :meth:`start` returns and its ``finally``
        cleanup runs.  Only when the loop is not running yet does the
        handler fall back to raising :class:`KeyboardInterrupt`
        (raising it mid-loop is unreliable — a busy loop can swallow
        it inside foreign ``except``/``finally`` frames, leaving the
        daemon and its agents running while every later SIGTERM is
        ignored).

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
        if signum in _SHUTDOWN_SIGNALS:
            if self._shutdown_initiated:
                logger.info(
                    "%s during shutdown ignored: pid=%d "
                    "(cleanup already in progress)",
                    sig_name,
                    os.getpid(),
                )
                return
            self._shutdown_initiated = True
            loop = self._loop
            if loop is not None and loop.is_running():
                try:
                    threading.Thread(
                        target=self._shutdown_on_sigterm,
                        name="kiss-sigterm-shutdown",
                        daemon=True,
                    ).start()
                    return
                except Exception:
                    # Thread exhaustion is exactly when an operator
                    # sends SIGTERM.  The latch above is already set,
                    # so without a fallback every later SIGTERM would
                    # be ignored and nothing would ever shut down.
                    logger.exception(
                        "%s: could not start the shutdown thread; "
                        "unwinding the event loop directly", sig_name,
                    )
                try:
                    loop.call_soon_threadsafe(self._request_loop_shutdown)
                    return
                except RuntimeError:
                    pass  # Loop already closed: fall through.
            if loop is not None:
                # The loop already unwound (the Ctrl-C path): start()'s
                # ``finally`` is running the cleanup, so there is
                # nothing left to interrupt.
                return
            raise KeyboardInterrupt(f"Received {sig_name}")

    def _request_loop_shutdown(self) -> None:
        """Make :meth:`_serve_async` return (runs ON the event loop).

        Scheduled by :meth:`_shutdown_on_sigterm` via
        ``call_soon_threadsafe``.  Resolving :attr:`_shutdown_future`
        completes the ``asyncio.wait`` in :meth:`_serve_async`, so
        ``asyncio.run`` unwinds and :meth:`start` runs its shutdown
        ``finally``.  If the future does not exist yet (SIGTERM landed
        while :meth:`_setup_server` was still binding listeners) raise
        ``KeyboardInterrupt`` instead: from a plain loop callback the
        exception is re-raised by ``Handle._run`` straight out of
        ``run_forever`` — no foreign coroutine frame can swallow it.
        """
        fut = self._shutdown_future
        if fut is not None and not fut.done():
            fut.set_result(None)
            return
        raise KeyboardInterrupt("SIGTERM received before serve loop started")

    def _shutdown_on_sigterm(self) -> None:
        """Drive the SIGTERM graceful shutdown off the main thread.

        Runs in a dedicated daemon thread started by
        :meth:`_handle_shutdown_signal` so the event loop stays live
        (flushing the "Restarting…" notification, answering pings)
        while the in-flight agent worker threads are cooperatively
        stopped and joined.  Ordering matters:

        0. :meth:`_detach_tunnel` FIRST — spawn the stderr drain shim
           that keeps the detached ``cloudflared`` alive, before any
           cleanup that can outlive an impatient supervisor's
           SIGTERM→SIGKILL escalation.  Everything after this point
           may be cut short by SIGKILL without costing the public
           tunnel URL.
        1. :meth:`_stop_active_agent_tasks` — the user-visible
           point of "Reset Server" is that running agents stop, and
           this must not depend on the event loop being able to unwind
           (a wedged loop previously left agents running forever).
        2. Unwind the loop via :meth:`_request_loop_shutdown` so
           ``asyncio.run`` returns and :meth:`start`'s ``finally``
           performs the remaining cleanup (its second
           ``_stop_active_agent_tasks`` call is then a no-op).
        3. Failsafe: if the loop still has not stopped after
           ``_SHUTDOWN_EXIT_FAILSAFE`` seconds — e.g. ``asyncio.run``'s
           cancellation phase is stuck on a task swallowing
           ``CancelledError`` — force-exit so the supervisor respawns a
           fresh daemon instead of leaving a zombie that ignores every
           further SIGTERM.
        """
        # Detach the tunnel FIRST — before any slow cleanup (agent-task
        # joins, merge waits, MCP disconnects).  This spawns the stderr
        # drain shim that keeps the detached cloudflared alive after
        # this process dies.  A supervisor (the installer, launchd)
        # escalates its SIGTERM to SIGKILL after only a few seconds;
        # when that SIGKILL landed mid-cleanup the shim was never spawned (the
        # detach used to run only in ``start()``'s ``finally``), the
        # detached cloudflared died of SIGPIPE on its next stderr
        # write, and the next kiss-web found "pidfile points to dead
        # pid" and minted a fresh public URL.  A healthy tunnel must
        # survive this daemon's death no matter how the daemon dies.
        try:
            self._detach_tunnel()
        except Exception:  # noqa: BLE001 — shutdown must proceed regardless
            logger.exception("SIGTERM shutdown: tunnel detach failed")
        try:
            self._stop_active_agent_tasks()
        except Exception:  # noqa: BLE001 — shutdown must proceed regardless
            logger.exception(
                "SIGTERM shutdown: stopping in-flight agent tasks failed",
            )
        self._await_active_merges()
        self._disconnect_mcp_servers()
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._request_loop_shutdown)
            except RuntimeError:
                pass
        deadline = time.monotonic() + _SHUTDOWN_EXIT_FAILSAFE
        while time.monotonic() < deadline:
            loop = self._loop
            if loop is None or not loop.is_running():
                return
            time.sleep(0.25)
        logger.error(
            "Shutdown failsafe: event loop did not unwind within %.0fs "
            "of SIGTERM (agent tasks already stopped); forcing exit "
            "so the supervisor can respawn: pid=%d",
            _SHUTDOWN_EXIT_FAILSAFE,
            os.getpid(),
        )
        self._detach_tunnel()
        logging.shutdown()
        os._exit(0)

    def _disconnect_mcp_servers(self) -> None:
        """Reap the MCP server children agents left behind.

        :class:`~kiss.agents.sorcar.mcp_servers.MCPManager` keeps one
        long-lived connection per configured MCP server, and a stdio
        server is a **child process** of this daemon.  The manager only
        tears those children down from an ``atexit`` hook, which does
        not run when the daemon is killed, and the daemon itself never
        referenced MCP at all — so every shutdown that was not a clean
        interpreter exit orphaned them.

        Called from every shutdown path (SIGTERM, the blocking
        ``start()`` cleanup, and the embedder/test ``stop_async()``)
        right after the in-flight agent tasks have been joined, so no
        agent can open a fresh connection afterwards.
        ``disconnect_all`` is idempotent, so the repeated calls a
        single shutdown makes are harmless no-ops, and it leaves the
        manager usable for an embedder that starts another server in
        the same process.
        """
        try:
            from kiss.agents.sorcar.mcp_servers import MCPManager

            MCPManager.instance().disconnect_all()
        except Exception:  # noqa: BLE001 — shutdown must proceed regardless
            logger.debug("MCP server disconnect failed", exc_info=True)

    def _await_active_merges(self, timeout: float = 30.0) -> None:
        """Wait for interactive merge/discard work to finish.

        The "Auto-commit and merge" / "Discard" action arrives as a
        forwarded command and therefore runs in the event loop's
        default executor.  By then the task that produced the worktree
        has ended, so ``AgentState.task_thread`` is ``None`` and
        :meth:`_stop_active_agent_tasks` — which requires a thread —
        skips the state even though ``busy()`` is true.  Cancelling the
        asyncio handler does not help either: cancelling a future that
        awaits ``run_in_executor`` never stops the running function.

        Unlike an agent task, a merge must not be *stopped*: it stashes,
        commits, checks out and merges, and interrupting it half way
        leaves the user's repository in a state they did not ask for.
        Shutdown therefore *waits* for it, bounded by *timeout* so a
        wedged git invocation cannot hang the process forever.

        The wait is on the merge CLAIM (``AgentState.merge_in_progress``,
        released in the merge's ``finally``), not on the claiming
        thread: the executor worker that runs a merge is a pooled
        thread that outlives the merge, so joining it waited the whole
        grace period after the repository was already merged.

        Args:
            timeout: Maximum wall-clock seconds to wait, in aggregate,
                for all in-flight merges.
        """
        tabs = _merging_tab_ids()
        if not tabs:
            return
        logger.warning(
            "Shutdown: waiting up to %.0fs for %d interactive merge(s) "
            "to finish rewriting the repository",
            timeout, len(tabs),
        )
        deadline = time.monotonic() + timeout
        while tabs and time.monotonic() < deadline:
            time.sleep(0.05)
            tabs = _merging_tab_ids()
        if tabs:
            logger.error(
                "Shutdown: merge(s) on tab(s) %s did not finish within the "
                "grace period; proceeding without them",
                ", ".join(tabs),
            )

    def _stop_active_agent_tasks(self, timeout: float = 12.0) -> None:
        """Stop in-flight agent worker threads so they unwind cleanly.

        Each task runs in a daemon worker thread spawned by
        :meth:`AgentServer._run_task`.  On process exit those daemon
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
        from kiss.server import agent_state
        from kiss.server.agent_state import AgentState
        from kiss.server.task_runner import (
            _state_owns_thread,
            inject_if_owned,
            wait_for_thread_start,
        )

        active: list[
            tuple[str, AgentState, threading.Event | None, threading.Thread]
        ] = []
        active_task_history_ids: set[str] = set()
        with agent_state.STATE_LOCK:
            # New runs observed after this point pre-cancel instead of
            # starting: ``_cmd_run``'s start/cancel handshake checks
            # this flag — and each swept state's
            # ``interrupted_by_shutdown`` below — under this same lock
            # immediately before ``thread.start()``, so no run can
            # start AFTER this sweep (audit0903 F1).
            self._agent_server._shutdown_stopping = True
            for task_id, state in agent_state.agent_states.items():
                thread = state.task_thread
                # Liveness is AgentState.busy(), not is_task_active
                # alone: the worker raises that flag only after
                # _cmd_run has started it, and a task swept in that
                # window is abandoned outright — no stop event, no
                # join, no cleanup finally — leaving its history row
                # stranded at the abrupt-failure sentinel (F08-2).
                if thread is not None and state.busy():
                    state.interrupted_by_shutdown = True
                    active.append((task_id, state, state.stop_event, thread))
                    active_task_history_ids.add(task_id)

        if not active:
            return

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
            ", ".join(tab_id for tab_id, _, _, _ in active),
        )

        for _tab_id, _state, stop_event, _thread in active:
            if stop_event is not None:
                stop_event.set()

        deadline = time.monotonic() + timeout
        for tab_id, state, _stop_event, thread in active:
            # A worker registered by ``_cmd_run`` but not yet started
            # cannot be joined (``Thread.join`` raises before start).
            # Wait for the start — or for ``_cmd_run``'s pre-start
            # handshake to cancel the run, whose terminal cleanup
            # clears ``state.task_thread`` and drops ownership — via
            # the same primitive the Stop watchdog uses (audit0903 F1).
            if not wait_for_thread_start(
                thread,
                partial(_shutdown_state_owns_thread, state, thread),
                deadline=deadline,
            ):
                continue
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=min(1.0, remaining))
            if thread.is_alive():
                # Ownership is re-checked under STATE_LOCK immediately
                # before injecting, exactly like the Stop watchdog.
                # "Still alive" does not mean "still ignoring the
                # stop": the worker may have honoured the cooperative
                # event already and be inside its legitimate cleanup
                # ``finally`` (persisting the interrupted row can wait
                # out SQLite's busy timeout), which this sweep exists
                # to let finish — injecting there aborted the very
                # persistence/broadcast it wants.  The predicate also
                # refuses while the thread performs the state's own
                # post-task worktree merge (a merge is awaited, never
                # stopped).  Unlike the watchdog, the sweep keeps
                # joining after a refusal: the cleanup must finish
                # before the process exits.
                inject_if_owned(thread, partial(_state_owns_thread, state, thread))
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
        for sig in _SHUTDOWN_SIGNALS:
            try:
                signal.signal(sig, self._handle_shutdown_signal)
            except (OSError, ValueError):
                pass

    def start(self) -> None:
        """Start the server (blocks until interrupted).

        Call this from the main thread.  Press Ctrl-C to stop.
        """
        _raise_open_file_limit()
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
        # A GIL-holding stall (2026-09-12: a quadratic regex over a
        # 5.8 MB tool result) freezes every thread, including logging;
        # the C-level watchdog still dumps all thread stacks to stderr.
        # Disarmed at the end of the ``finally`` below so an in-process
        # restart (or a test) does not leave a heartbeat thread behind.
        stall_watchdog = start_stall_watchdog()

        self._install_signal_handlers()
        # Index the home directory for the ``@``-mention picker now, on
        # the registry's worker thread, so the first ``@`` in any tab
        # below it answers from a warm index instead of waiting on a
        # scan (a restart reloads the persisted listings in well under
        # a second).
        file_index = self._agent_server._file_index
        file_index.ensure(file_index.home)

        try:
            asyncio.run(self._serve_async())
            if self._shutdown_initiated:
                logger.info("Server shutting down: pid=%d (SIGTERM)", pid)
        except KeyboardInterrupt:
            logger.info("Server shutting down: pid=%d (KeyboardInterrupt)", pid)
        finally:
            self._shutdown_initiated = True
            # Detach the tunnel FIRST (mirrors _shutdown_on_sigterm's
            # step 0).  This ``finally`` also serves the
            # KeyboardInterrupt / pre-loop-SIGTERM paths, which never
            # ran the sigterm thread's early detach; running the slow
            # cleanups below first would reopen the window where an
            # impatient supervisor's SIGKILL leaves the spawned
            # cloudflared without its stderr drain shim (SIGPIPE
            # death -> rotated public URL).  A no-op when the sigterm
            # thread already detached.
            self._detach_tunnel()
            self._shutdown_children()
            file_index.stop()
            if stall_watchdog is not None:
                stall_watchdog.stop()
            logger.info("Server stopped: pid=%d", pid)

    def _shutdown_children(self) -> None:
        """Stop everything the daemon owns besides its listeners and tunnel.

        The one shutdown sequence shared by :meth:`start`'s ``finally``
        (called inline, the loop is gone) and :meth:`stop_async` (run
        through ``asyncio.to_thread``, the loop is still serving the
        final frames).  Blocking, with bounded waits: in-flight agent
        worker threads (12 s aggregate), interactive merges/discards
        running in the default executor (30 s aggregate — they rewrite
        the repository, so nothing else is torn down before them),
        MCP servers, the tab registry's deferred re-save (a no-op when
        the last save succeeded), the streamed Browser tab (unregistered
        as the hand-off target first, so a later hand-off in this process
        falls back to the default browser, then closed rather than
        killed by the driver at process exit), terminal-tab shells (hung
        up, killed when they ignore it), the streams of tabs running on
        remote machines, and the SEA-registry watcher.  Every step is
        idempotent: the SIGTERM thread runs the first three early.
        """
        self._stop_active_agent_tasks()
        self._await_active_merges()
        self._disconnect_mcp_servers()
        self._agent_server.tab_registry.flush()
        set_browser_tab_opener(None)
        self._agent_server.browser_tabs.shutdown()
        self._agent_server.terminals.shutdown()
        self._server_api.remote_tabs.close_all()
        self._stop_sea_command_watcher()

    async def start_async(self) -> None:
        """Start the server asynchronously (for use in existing event loops).

        Returns after the server is listening.  The caller must keep
        the event loop running.

        Serialised against :meth:`stop_async` with
        :attr:`_lifecycle_lock` (F4-05): without it a concurrent stop
        could tear down the fields bound so far and return while this
        still-running setup binds the remaining listeners afterwards,
        resurrecting the server after shutdown completed.
        """
        _raise_open_file_limit()
        async with self._lifecycle_lock:
            if self._shutdown_initiated:
                return
            await self._setup_server()

    async def start_private_async(self) -> None:
        """Serve local clients only: loopback WSS on an ephemeral port.

        The channel-agent launcher's in-process daemon
        (``_kiss_web_launcher._ensure_api_server``) needs the command
        API without any of the public daemon's duties: no tunnel, no
        URL file, no watchdogs, no cron scheduler.  Binds
        ``127.0.0.1:0`` (the OS picks the port), records the port in
        :attr:`port` and publishes the endpoint file so
        :func:`kiss.agents.sorcar.local_endpoint.connect` finds it.
        Serialised against :meth:`stop_async` like :meth:`start_async`.
        """
        async with self._lifecycle_lock:
            if self._shutdown_initiated:
                return
            self._loop = asyncio.get_running_loop()
            self._printer._loop = self._loop
            if self._ssl_context is None:
                self._ssl_context = await asyncio.to_thread(
                    _create_ssl_context, self._ssl_certfile, self._ssl_keyfile, [],
                )
            self._local_only = True
            self.port = 0
            try:
                self._ws_server = await self._serve_wss("127.0.0.1")
                self.port = _bound_loopback(self._ws_server)[1]
                self._write_local_endpoint_file()
            except BaseException:
                # A cancelled or failed start must not leave a listener
                # (or an endpoint file pointing at one) behind.
                self._close_partial_setup()
                raise

    async def stop_async(self) -> None:
        """Stop the server gracefully.

        Mirrors the blocking ``start()`` shutdown path for in-flight
        agent tasks: :meth:`_stop_active_agent_tasks` cooperatively
        stops and **joins** each worker thread (off-loop, since the
        join blocks) so its cleanup ``finally`` persists a real
        result instead of abandoning the row at the "Agent Failed
        Abruptly" sentinel when an embedder shuts down.

        Unlike ``start()`` — which *detaches* the spawned cloudflared
        so the next daemon can adopt it and keep the public URL —
        this path deliberately calls :meth:`_stop_tunnel` to kill the
        tunnel: embedders and tests own their server's full lifecycle
        and must not leak a background cloudflared process.

        Ordering: command ingress is quiesced FIRST — the WSS
        listeners are closed, which also closes every established
        client connection (F4-02) — and only then are the in-flight agent
        worker threads stopped, so a surviving peer cannot launch
        fresh work after the worker sweep (F4-06).  The whole method
        is serialised against :meth:`start_async` with
        :attr:`_lifecycle_lock` (F4-05) so a suspended setup cannot
        resurrect the server after this returns.
        """
        self._shutdown_initiated = True
        async with self._lifecycle_lock:
            await _cancel_task(self._watchdog_task)
            self._watchdog_task = None
            await _cancel_task(self._trusted_clients_task)
            self._trusted_clients_task = None
            await _cancel_task(self._tls_refresh_task)
            self._tls_refresh_task = None
            await _cancel_task(self._version_check_task)
            self._version_check_task = None
            await _cancel_task(self._machines_watch_task)
            self._machines_watch_task = None
            await _cancel_task(self._update_watch_task)
            self._update_watch_task = None
            idle_poller = self._update_when_idle_task
            await self._cancel_update_when_idle()
            await _cancel_task(idle_poller)
            self._update_when_idle_task = None
            await _cancel_task(self._update_models_watch_task)
            self._update_models_watch_task = None
            await _cancel_task(self._republish_task)
            self._republish_task = None
            ws_servers = [
                s for s in (self._ws_server, self._ws_loopback_server)
                if s is not None
            ]
            self._close_ws_listeners(remove_endpoint_file=False)
            for ws_server in ws_servers:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(ws_server.wait_closed(), timeout=2)
            self._ws_server = self._ws_loopback_server = None
            # Off-loop: the sidecar flock wait is bounded (5 s) but
            # would otherwise freeze the loop during shutdown.
            await asyncio.to_thread(
                local_endpoint.remove_endpoint_if_owned,
                self._local_endpoint_file, self._local_token,
            )
            # Every child of this daemon (worker threads, merges, MCP,
            # the streamed browser, terminal shells, remote streams):
            # the same sequence as ``start()``'s ``finally``, off the
            # loop so the final frames to connected clients (and any
            # embedder's own tasks) are not frozen by its bounded waits.
            await asyncio.to_thread(self._shutdown_children)
            # ``_stop_tunnel`` blocks in ``Popen.wait(timeout=5)`` while
            # cloudflared shuts down; off the loop for the same reason.
            await asyncio.to_thread(self._stop_tunnel)
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

    parser = argparse.ArgumentParser(description=f"{PRODUCT_NAME} Remote Access Server")
    parser.add_argument(
        "--url", action="store_true",
        help="Print the active remote URL and exit",
    )
    parser.add_argument(
        "--trust-ca", action="store_true",
        help="Install the local CA that signs the webapp's TLS certificate "
        "into this user's browser trust stores (removes the certificate "
        "warning on the Local and LAN URLs) and exit",
    )
    parser.add_argument("--workdir", default=None, help="Working directory")
    args = parser.parse_args()

    if args.url:
        _print_url()
        return
    if args.trust_ca:
        _trust_local_ca()
        return

    tunnel_token, tunnel_url = _resolve_tunnel_settings()

    suppress_error_dialogs()
    server = RemoteAccessServer(
        use_tunnel=True,
        tunnel_token=tunnel_token,
        tunnel_url=tunnel_url,
        work_dir=args.workdir,
    )
    server.start()


if __name__ == "__main__":
    main()
