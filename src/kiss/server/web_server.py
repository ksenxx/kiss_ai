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
import base64
import binascii
import contextlib
import datetime
import errno
import hashlib
import ipaddress
import json
import logging
import math
import mimetypes
import os
import platform
import re
import secrets
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from concurrent.futures import Future as ConcurrentFuture
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from kiss.core.config import get_jobs_root as get_jobs_root
from kiss.core.config import kiss_home
from kiss.core.models.model_info import get_default_model
from kiss.core.vscode_config import (
    apply_config_to_env,
    load_config,
    save_config,
    source_shell_env,
)
from kiss.server import sorcar as sorcar_api
from kiss.server.json_printer import JsonPrinter, stamp_event_ts
from kiss.server.server import VSCodeServer, broadcast_to_conn
from kiss.server.tips import read_tips
from kiss.server.tricks import read_tricks
from kiss.server.voice_wake import (
    MODEL_NAME,
    SpeakerIdentifier,
    default_models_dir,
    transcribe_pcm,
)
from kiss.server.voice_wake_control import VoiceWakeController
from kiss.viz_trajectory.server import find_job_dir as find_job_dir
from kiss.viz_trajectory.server import list_jobs as list_jobs
from kiss.viz_trajectory.server import (
    load_job_trajectories as load_job_trajectories,
)

__all__ = ["RemoteAccessServer", "WebPrinter"]

logger = logging.getLogger(__name__)

MEDIA_DIR = Path(__file__).resolve().parent.parent / "agents" / "vscode" / "media"

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
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_text_to(tmp: Path, text: str) -> None:
    """Write *text* to *tmp* as UTF-8 (writer for :func:`_atomic_publish`)."""
    tmp.write_text(text, encoding="utf-8")


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
    :func:`_ensure_voice_model`.
    """
    urllib.request.urlretrieve(VOICE_MODEL_URL, tmp)


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
_MEDIA_VERSION_CACHE: dict[str, str] = {}

TRAJECTORY_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "viz_trajectory"
    / "templates"
    / "index.html"
)

TUNNEL_CHECK_INTERVAL = 15

_IP_CHANGE_DEBOUNCE_TICKS = 4

_BIND_RETRY_ATTEMPTS = 5
_BIND_RETRY_BACKOFF: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
_BIND_RETRYABLE_ERRNOS: frozenset[int] = frozenset({
    errno.EADDRINUSE, errno.EADDRNOTAVAIL,
})

_VERSION_CHECK_INTERVAL: float = 3600

_PYPI_LATEST_URL = "https://pypi.org/pypi/kiss-agent-framework/json"

_INSTALLED_EXTENSIONS_ROOT: Path | None = None

_EXTENSION_DIR_PREFIX = "ksenxx.kiss-sorcar-"

_PYPI_FETCH_TIMEOUT = 5.0

_WS_PING_TIMEOUT = 10

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


def _truncate_utf8_bytes(text: str, max_bytes: int) -> tuple[str, int]:
    """Return *text* capped to *max_bytes* and its original byte size.

    ``json.loads`` may legitimately produce strings containing lone
    UTF-16 surrogate code points (for example from ``"\\ud800"``).
    Strict UTF-8 encoding raises ``UnicodeEncodeError`` for those
    strings, which aborts that submit command (and can close transports
    whose receive loop does not isolate command errors).  ``surrogatepass``
    gives every Python string a
    deterministic byte representation while preserving such code
    points in an untruncated prompt.  If the cap lands inside any UTF-8
    sequence, the incomplete suffix is removed before decoding.

    Args:
        text: Prompt text to measure and possibly truncate.
        max_bytes: Maximum encoded size.

    Returns:
        ``(possibly_truncated_text, original_encoded_size)``.
    """
    encoded = text.encode("utf-8", errors="surrogatepass")
    original_size = len(encoded)
    if original_size <= max_bytes:
        return text, original_size
    prefix = encoded[:max_bytes]
    while prefix:
        try:
            return prefix.decode("utf-8", errors="surrogatepass"), original_size
        except UnicodeDecodeError as exc:
            prefix = prefix[:exc.start]
    return "", original_size


_MAX_LINE_BYTES = 64 * 1024 * 1024

# ``websockets`` wraps the whole opening handshake - including a plain
# HTTP reply produced by ``process_request`` - in ``open_timeout``.  Its
# 10s default silently guillotines large downloads such as the 40MB
# wake-word model, which reaches the browser as ERR_EMPTY_RESPONSE.
_OPEN_TIMEOUT_SECONDS = 300.0

_MAX_VOICE_AUDIO_B64 = 4 * 1024 * 1024

_KISS_HOME: Path | None = None
_TLS_DIR: Path | None = None


def _kiss_home_dir() -> Path:
    """Return the KISS home dir ($KISS_HOME or ~/.kiss), resolved lazily."""
    return _KISS_HOME if _KISS_HOME is not None else kiss_home()


def _tls_dir() -> Path:
    """Return the directory holding the self-signed TLS cert/key pair."""
    return _TLS_DIR if _TLS_DIR is not None else _kiss_home_dir() / "tls"


def _url_file_path() -> Path:
    """Return the persisted remote-URL path: override or lazy default.

    Assigning ``web_server._URL_FILE`` is a supported test override,
    matching the lazy ``CONFIG_DIR`` / ``CONFIG_PATH`` attributes in
    :mod:`kiss.core.vscode_config`.  The override must be
    consulted here (rather than only exposed through ``__getattr__``)
    because production consumers such as :class:`RemoteAccessServer`
    call this accessor directly.
    """
    override = globals().get("_URL_FILE")
    return override if override is not None else _kiss_home_dir() / "remote-url.json"


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


def _default_uds_path() -> Path:
    """Return the default localhost Unix-domain socket path.

    The socket is exposed by :class:`RemoteAccessServer` in addition
    to the public WSS port.  Local clients (the VS Code extension)
    connect to this socket over the SAME newline-delimited JSON
    protocol that browsers speak over WSS — no password challenge is
    performed because POSIX filesystem permissions (mode 0o600)
    restrict access to the owning user.  A fresh
    ``RemoteAccessServer(uds_path=...)`` argument overrides this
    location for tests so multiple instances do not race on the same
    socket file.
    """
    return _kiss_home_dir() / "sorcar.sock"


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


def _is_loopback_ip(ip: str) -> bool:
    """Return True when *ip* is an IPv4/IPv6 loopback address.

    A loopback TCP peer on the public WSS port is the local
    ``cloudflared`` tunnel relaying a remote visitor (or another
    process on this host).  IPv6-mapped IPv4 loopback
    (``::ffff:127.0.0.1``) counts too.  A malformed / empty string is
    not loopback.

    Args:
        ip: A textual IP address (no port).

    Returns:
        True when *ip* parses to a loopback address.
    """
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return False
    mapped = getattr(parsed, "ipv4_mapped", None)
    if mapped is not None:
        parsed = mapped
    return parsed.is_loopback


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

# Cap on the bytes buffered while waiting for the first CRLF of an
# incoming request line.  Matches the conventional HTTP request-line
# limit; anything longer is fed to the websockets parser (which
# rejects it) instead of being buffered without bound (F4-08).
_MAX_HEAD_LINE_BYTES = 8192


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
                transport.write(_HEAD_200)
                transport.close()
            return
        buffered = self._head_buffer
        self._head_buffer = b""
        super().data_received(buffered)


_OPEN_FILE_MAX_BYTES = 2_000_000

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


_CLOUDFLARED_PIDFILE: Path | None = None


def _cloudflared_pidfile() -> Path:
    """Return the path of the persisted cloudflared PID file."""
    if _CLOUDFLARED_PIDFILE is not None:
        return _CLOUDFLARED_PIDFILE
    return _kiss_home_dir() / "cloudflared.pid"


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
    ``ps -o comm=`` (portable across macOS and Linux) and matches the
    executable basename against ``cloudflared`` — the name of the
    binary both the spawn path and the adoption path launch.

    Returns:
        True when the command basename is exactly ``cloudflared`` (or
        ``cloudflared.exe``); False for any other process, a dead PID,
        or a ``ps`` failure (fail-safe: an unverifiable process is
        never signalled).  The match is exact — a prefix match would
        also kill an unrelated ``cloudflared-helper``-style process
        that inherited a recycled PID.
    """
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
    if not _looks_like_cloudflared(pid):
        logger.info(
            "pidfile pid %d is not a cloudflared process (stale pidfile, "
            "pid recycled); not signalling it",
            pid,
        )
        _unlink_cloudflared_pidfile()
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        _unlink_cloudflared_pidfile()
        return
    for _ in range(20):
        if not _is_pid_alive(pid):
            break
        time.sleep(0.1)
    if _is_pid_alive(pid) and _looks_like_cloudflared(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    _unlink_cloudflared_pidfile()


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
    if ready is not True:
        logger.info(
            "cloudflared pid %d alive but metrics port %d reports "
            "no ready connections; not adopting",
            pid, metrics_port,
        )
        _terminate_declined_cloudflared(pid)
        return None
    url = _query_quicktunnel_hostname(metrics_port)
    if url is None:
        saved = data.get("url")
        if isinstance(saved, str) and saved.startswith("https://"):
            url = saved
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
    url_found_event.wait(timeout=timeout)
    if result[0] is not None:
        return result[0]
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
    kiss_home_path = _kiss_home_dir()
    topic_file = kiss_home_path / "ntfy_topic"
    try:
        stored = topic_file.read_text(encoding="utf-8").strip()
    except OSError:
        stored = ""
    if stored:
        return stored
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
    return frozenset(
        addr for addr in ips
        if not addr.startswith(("127.", "169.254.", "::ffff:"))
    )


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
        print("KISS Sorcar web server is not running.", file=sys.stderr)
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
    the extension's dependency installer SIGTERM the daemon on top of
    a just-launched run (F08-2).
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
    if key_path.exists():
        key_path.unlink()
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key_bytes)
    finally:
        os.close(fd)
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
        tls_dir = _tls_dir()
        cert_path = tls_dir / "cert.pem"
        key_path = tls_dir / "key.pem"
        # Serialise sibling daemons with an exclusive file lock: the
        # check-then-generate sequence and the pair publication are
        # not atomic, so two concurrent processes could otherwise
        # publish (or load) a mismatched cert/key pair (F4-10).
        import fcntl

        tls_dir.mkdir(parents=True, exist_ok=True)
        lock_path = tls_dir / ".tls.lock"
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not cert_path.is_file() or not key_path.is_file():
                logger.info(
                    "Generating self-signed TLS certificate in %s", tls_dir,
                )
                _generate_self_signed_cert(cert_path, key_path)
            elif _self_signed_cert_needs_renewal(cert_path):
                logger.info(
                    "Self-signed TLS certificate %s is expired or "
                    "expiring within 30 days; regenerating",
                    cert_path,
                )
                _generate_self_signed_cert(cert_path, key_path)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            try:
                ctx.load_cert_chain(str(cert_path), str(key_path))
            except ssl.SSLError:
                # Crash-consistent pair publish (F4-10 residual): a
                # daemon that died between writing the key and the
                # cert leaves a mismatched pair on disk that every
                # future load would reject.  Self-heal under the
                # lock: regenerate the pair and load the fresh one.
                logger.warning(
                    "Auto-generated TLS cert/key pair in %s is "
                    "mismatched or corrupt; regenerating", tls_dir,
                )
                _generate_self_signed_cert(cert_path, key_path)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.minimum_version = ssl.TLSVersion.TLSv1_2
                ctx.load_cert_chain(str(cert_path), str(key_path))
            return ctx

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
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
        self._uds_writers: set[asyncio.StreamWriter] = set()
        self._local_uds_tab_counts: dict[str, int] = {}
        self._uds_local_tab_sets: dict[str, set[str]] = {}
        self._conn_endpoints: dict[str, Any] = {}
        self._ws_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.work_dir: str = ""
        self._pending_sends: dict[Any, set[ConcurrentFuture[None]]] = {}
        self._send_locks: dict[Any, asyncio.Lock] = {}

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send *event* to every connected WebSocket client.

        Two code paths:

        * Events that already carry an explicit ``tabId`` (status,
          askUser, commitMessage, etc.) are treated as
          targeted "system" events: sent verbatim to all connected
          clients (which filter by ``tabId``), but **not** recorded
          or persisted — except ``prompt`` echoes that ALSO carry a
          ``taskId``, whose tabId-stripped copy is recorded and
          persisted under that task (see the tabId branch below).
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

        if conn_id:
            self._send_to_conn(conn_id, json.dumps(event))
            return

        if "tabId" in event:
            if event.get("type") in ("prompt", "result") and event.get("taskId"):
                record = {k: v for k, v in event.items() if k != "tabId"}
                with self._lock:
                    self._record_event(record)
                self._persist_event(record)
            if record_only:
                return
            self._send_to_ws_clients(json.dumps(event))
            return

        event = self._inject_task_id(event)

        if not event.get("taskId"):
            if record_only:
                return
            self._send_to_ws_clients(json.dumps(event))
            return

        with self._lock:
            self._record_event(event)
            # Mirror JsonPrinter.broadcast: record the file paths of
            # mutating tool calls so the end-of-task cross-repo
            # auto-commit (_autocommit_changed_repos) also sees tasks
            # run through the web printer.
            self._track_changed_path(event)

        self._persist_event(event)

        if record_only:
            return

        self._fanout_stamped(event)

    def _fanout_stamped(self, event: dict[str, Any]) -> None:
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
        """
        targets = self._fanout_targets(event.get("taskId"))
        if not targets:
            return
        if "tabId" in event:
            event = {k: v for k, v in event.items() if k != "tabId"}
        if event.get("type") == "talk":
            self._fanout_talk(event, targets)
            return
        base = json.dumps(event)[:-1]
        for tab_id in targets:
            self._send_to_ws_clients(
                f'{base}, "tabId": {json.dumps(tab_id)}}}'
            )

    @staticmethod
    def _increment_count(counts: dict[str, int], key: str) -> None:
        """Increment *key* in a small connection-count map."""
        counts[key] = counts.get(key, 0) + 1

    @staticmethod
    def _decrement_count(counts: dict[str, int], key: str) -> None:
        """Decrement *key* in a small connection-count map."""
        count = counts.get(key)
        if count is None:
            return
        if count <= 1:
            del counts[key]
        else:
            counts[key] = count - 1

    def register_local_uds_tab(
        self, conn_id: str, tab_id: str, local_tabs: set[str]
    ) -> None:
        """Mark *tab_id* as shown by the local UDS connection *conn_id*.

        Args:
            conn_id: The UDS connection's id.
            tab_id: The frontend tab id seen on a UDS command.
            local_tabs: The connection's mutable local-tab set (lives
                in its ``conn_state``).  Membership is checked and
                updated under the printer lock, so a concurrent
                canonical-close prune cannot race the registration.
        """
        with self._ws_lock:
            self._uds_local_tab_sets[conn_id] = local_tabs
            if tab_id in local_tabs:
                return
            local_tabs.add(tab_id)
            self._increment_count(self._local_uds_tab_counts, tab_id)

    def sync_local_uds_tabs(
        self, conn_id: str, tab_ids: set[str], local_tabs: set[str]
    ) -> None:
        """Reconcile *conn_id*'s local-tab membership to exactly *tab_ids*.

        The ``ready``-time sync: missing ids are added and stale ones
        dropped (with matching reference-count updates), so a repeated
        ``ready`` self-heals bookkeeping left over from canonical tabs
        that were closed while this connection was attached.

        Args:
            conn_id: The UDS connection's id.
            tab_ids: The tab ids the connection currently shows.
            local_tabs: The connection's mutable local-tab set.
        """
        with self._ws_lock:
            self._uds_local_tab_sets[conn_id] = local_tabs
            for tab_id in tab_ids - local_tabs:
                self._increment_count(self._local_uds_tab_counts, tab_id)
            for tab_id in local_tabs - tab_ids:
                self._decrement_count(self._local_uds_tab_counts, tab_id)
            local_tabs.clear()
            local_tabs.update(tab_ids)

    def prune_local_uds_tab(self, tab_id: str) -> None:
        """Drop *tab_id* from every UDS connection's local-tab bookkeeping.

        Called when a tab is removed from the canonical tab registry
        (explicit close or one-tab-per-chat displacement): the
        ``tabs_state`` broadcast removes the tab from every client UI,
        so no local webview shows it anymore — a still-running task's
        talk for the id must no longer trigger daemon-native playback.

        Args:
            tab_id: The registry-removed frontend tab id.
        """
        with self._ws_lock:
            for local_tabs in self._uds_local_tab_sets.values():
                if tab_id in local_tabs:
                    local_tabs.discard(tab_id)
                    self._decrement_count(self._local_uds_tab_counts, tab_id)

    def unregister_local_uds_tabs(
        self, conn_id: str, tab_ids: set[str]
    ) -> None:
        """Drop a disconnected UDS connection's local-tab registrations.

        Args:
            conn_id: The UDS connection's id.
            tab_ids: The connection's remaining local-tab ids.
        """
        with self._ws_lock:
            self._uds_local_tab_sets.pop(conn_id, None)
            for tab_id in tab_ids:
                self._decrement_count(self._local_uds_tab_counts, tab_id)

    def _fanout_talk(self, event: dict[str, Any], targets: list[str]) -> None:
        """Fan out one ``talk`` event with per-device playback arbitration.

        Local UDS webview tabs (VS Code chat webviews on the daemon's
        machine) CANNOT reliably play the synthesized clip themselves:
        Chromium's autoplay policy rejects ``Audio.play()`` in a
        webview unless the user interacted with it seconds earlier
        (microsoft/vscode#197937 / #178642, closed as not actionable),
        so the talk would stay silent in the webview (whose old
        robotic Web Speech fallback is gone).  When the event carries
        a synthesized clip and a local webview tab is subscribed, the
        DAEMON therefore plays the clip natively on this machine's
        speakers (:mod:`kiss.server.talk_player`, ``afplay`` on
        macOS) and stamps every local UDS webview copy ``muted``.

        Muting is per-ENDPOINT, not per-serialization: the canonical
        tab registry mirrors the same tab ids to every client, so a
        remote WSS browser shows the very tab the local webview does.
        The browser is a different device with its own speakers, so
        when the daemon owns the utterance only the same-machine UDS
        copies are muted while every WSS copy stays playable.

        Otherwise webview subscriber tabs receive the playable copy
        (each webview plays on its own device; ``talkId`` dedupe and
        the talk queue keep intra-webview duplicates silent).

        Args:
            event: The ``talk`` event (no ``tabId`` stamp yet).
            targets: Subscriber tab ids for the event's task.
        """
        with self._ws_lock:
            local_uds_tabs = set(self._local_uds_tab_counts)
        local_web_targets = [t for t in targets if t in local_uds_tabs]
        daemon_plays = bool(local_web_targets) and self._play_talk_clip_locally(
            event
        )
        base = json.dumps(event)[:-1]
        muted_base = json.dumps({**event, "muted": True})[:-1]
        for tab_id in targets:
            tab_suffix = f', "tabId": {json.dumps(tab_id)}}}'
            if daemon_plays and tab_id in local_uds_tabs:
                self._send_to_wss_clients(base + tab_suffix)
                self._send_to_uds_writers(muted_base + tab_suffix)
            else:
                self._send_to_ws_clients(base + tab_suffix)

    @staticmethod
    def _play_talk_clip_locally(event: dict[str, Any]) -> bool:
        """Play a talk event's synthesized clip on this machine's speakers.

        Uses the :class:`~kiss.server.talk_player.TalkPlayer`
        singleton — a real audio-player child process
        (``afplay`` / ``mpg123`` / ``ffplay`` / ``mpv``, overridable
        via ``KISS_SORCAR_PLAY_CMD``) fed from a serialising queue
        with ``talkId`` dedupe, so playback never blocks the event
        loop and never overlaps another playback of the same
        utterance.

        Args:
            event: The unmuted ``talk`` broadcast event.

        Returns:
            ``True`` when the daemon machine has an audio player and
            the event carries a clip so local playback was enqueued;
            ``False`` otherwise (callers then leave the client copies
            playable so devices can fall back on their own).
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

    def _send_to_wss_clients(self, data: str) -> None:
        """Send a pre-serialised JSON payload to WSS clients only.

        WSS peers are remote browsers — separate devices from the
        daemon machine — so talk arbitration sends them the playable
        copy while the same-machine UDS peers get the muted one.

        Args:
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        with self._ws_lock:
            endpoints = list(self._ws_clients)
        for endpoint in endpoints:
            self._schedule_send(endpoint, data)

    def _send_to_uds_writers(self, data: str) -> None:
        """Send a pre-serialised JSON payload to local UDS peers only.

        UDS peers (VS Code extension webviews, Python clients) are
        always on the daemon's machine; talk arbitration sends them
        muted copies when a local player already owns the utterance.

        Args:
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        with self._ws_lock:
            endpoints = list(self._uds_writers)
        for endpoint in endpoints:
            self._schedule_send(endpoint, data)

    def _send_to_ws_clients(self, data: str) -> None:
        """Send a pre-serialised JSON payload to every connected client.

        Factored out of :meth:`broadcast` so fan-out copies for
        subscribed viewer tab ids reuse the same dispatch and pending-
        future tracking as the primary broadcast.  Fans out to BOTH
        WSS clients and local Unix-domain socket writers in lockstep by
        delegating to :meth:`_send_to_wss_clients` and
        :meth:`_send_to_uds_writers` (per-endpoint FIFO order is
        preserved by each endpoint's ``send_lock``).

        Args:
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        self._send_to_wss_clients(data)
        self._send_to_uds_writers(data)

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

    def send_lock(self, endpoint: Any) -> asyncio.Lock:
        """Return the per-endpoint lock serialising outbound sends.

        Every code path that writes to *endpoint* — the broadcast
        fan-out in :meth:`_schedule_send` and the direct replies in
        :meth:`RemoteAccessServer._endpoint_send` — must acquire this
        lock so payloads reach the wire in send-start order even when
        an earlier ``send()`` is suspended on write backpressure.

        Args:
            endpoint: The client connection the payload targets.

        Returns:
            The (lazily created) ``asyncio.Lock`` for *endpoint*.
        """
        with self._ws_lock:
            lock = self._send_locks.get(endpoint)
            if lock is None:
                lock = asyncio.Lock()
                if endpoint in self._pending_sends:
                    self._send_locks[endpoint] = lock
            return lock

    async def _locked_send(self, endpoint: Any, data: str) -> None:
        """Send one payload to one endpoint under its FIFO send lock.

        Args:
            endpoint: The client connection to write to.
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        async with self.send_lock(endpoint):
            if isinstance(endpoint, asyncio.StreamWriter):
                await self._uds_send(endpoint, data)
            else:
                await endpoint.send(data)

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
            fut = asyncio.run_coroutine_threadsafe(
                self._locked_send(endpoint, data), loop,
            )
        except Exception:
            logger.debug("Failed to send to client", exc_info=True)
            return
        with self._ws_lock:
            pending = self._pending_sends.get(endpoint)
            if pending is not None:
                pending.add(fut)
        if pending is None:
            fut.cancel()
            return
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

    def _add_endpoint(self, endpoint: Any, collection: set[Any]) -> None:
        """Register a WSS/UDS *endpoint* in *collection* for broadcasting.

        Shared body of :meth:`add_client` and :meth:`add_uds_writer`.
        """
        with self._ws_lock:
            collection.add(endpoint)
            self._pending_sends.setdefault(endpoint, set())

    def _remove_endpoint(self, endpoint: Any, collection: set[Any]) -> None:
        """Remove a WSS/UDS *endpoint* and cancel its pending sends.

        Shared body of :meth:`remove_client` and
        :meth:`remove_uds_writer`.  Cancelling the pending
        ``run_coroutine_threadsafe`` futures (M8) ensures a
        permanently stuck send queue cannot keep the underlying
        coroutine alive after the peer is gone.
        """
        with self._ws_lock:
            collection.discard(endpoint)
            pending = self._pending_sends.pop(endpoint, set())
            self._send_locks.pop(endpoint, None)
        for fut in pending:
            try:
                fut.cancel()
            except Exception:
                logger.debug("Failed to cancel pending send", exc_info=True)

    def add_client(self, ws: ServerConnection) -> None:
        """Register a WebSocket client for event broadcasting.

        Args:
            ws: The WebSocket server connection to add.
        """
        self._add_endpoint(ws, self._ws_clients)

    def remove_client(self, ws: ServerConnection) -> None:
        """Remove a WebSocket client from event broadcasting.

        Args:
            ws: The WebSocket server connection to remove.
        """
        self._remove_endpoint(ws, self._ws_clients)

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
        self._add_endpoint(writer, self._uds_writers)

    def remove_uds_writer(self, writer: asyncio.StreamWriter) -> None:
        """Remove a Unix-domain socket writer from event broadcasting.

        Args:
            writer: The asyncio stream writer to remove.
        """
        self._remove_endpoint(writer, self._uds_writers)

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
    tricks_json = json.dumps(read_tricks()).replace("</", "<\\/")
    tips_json = json.dumps(
        {"tips": read_tips(), "show": False},
    ).replace("</", "<\\/")
    head_style = (
        f'<link href="{_media_url("remote-codex.css")}" rel="stylesheet">\n'
        "  <style>\n"
        "    html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; }\n"
        "    body { background: var(--vscode-editor-background, #1e1e1e);\n"
        "            color: var(--vscode-editor-foreground, #cccccc); }\n"
        "    :root {\n"
        "      --vscode-font-size: 16px;\n"
        "      --vscode-font-family: -apple-system, BlinkMacSystemFont, "
        "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;\n"
        "      --vscode-editor-font-size: 16px;\n"
        "      --vscode-editor-font-family: Menlo, Monaco, "
        "'Courier New', monospace;\n"
        "      --vscode-editor-background: #1e1e1e;\n"
        "      --vscode-editor-foreground: #cccccc;\n"
        "      --vscode-input-background: #3c3c3c;\n"
        "      --vscode-button-foreground: #ffffff;\n"
        "      --vscode-sideBar-background: #252526;\n"
        "      --vscode-textLink-foreground: #3794ff;\n"
        "      --vscode-descriptionForeground: #8b8b8b;\n"
        "      --vscode-panel-border: #80808059;\n"
        "      --vscode-terminal-ansiRed: #f44747;\n"
        "      --vscode-terminal-ansiGreen: #6a9955;\n"
        "      --vscode-terminal-ansiYellow: #d7ba7d;\n"
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
        "API_SRC": _media_url("api.js"),
        "PANEL_COPY_SRC": _media_url("panelCopy.js"),
        "CTX_MENU_SRC": _media_url("contentContextMenu.js"),
        "MAIN_SRC": _media_url("main.js"),
        "SHIM_SCRIPT": (
            "<script>window.__HLJS_THEME_CSS__ = "
            + json.dumps({
                "dark": _media_url("highlight-github-dark.min.css"),
                "light": _media_url("highlight-github-light.min.css"),
            })
            + f";</script>\n  <script>{_WS_SHIM_JS}</script>\n  "
        ),
        "TRICKS_JSON": tricks_json,
        "TIPS_JSON": tips_json,
        "TIPS_SRC": _media_url("tips.js"),
        "VOICE_SRC": _media_url("voice.js"),
        "VOICE_CONFIG": json.dumps({
            "mode": "browser",
            "voskSrc": _media_url("vosk.js"),
            "modelUrl": "/voice-model.tar.gz",
            "ackAudioUrl": _media_url("working-on-it.mp3"),
        }),
    }
    tpl = (MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
    return re.sub(
        r"\{\{([A-Z_]+)\}\}",
        lambda m: subs.get(m.group(1), m.group(0)),
        tpl,
    )


def _parse_version_py(vfile: Path) -> str:
    """Return the ``__version__`` string from a ``_version.py`` file.

    Returns an empty string when the file is missing, unreadable, or
    does not define ``__version__``.  A best-effort parser that keeps
    the daemon booting even if a foreign ``_version.py`` is malformed.
    Uses the exact same regex as ``readVersionPy`` in the extension's
    ``UpdateChecker.js`` (its documented twin) so the daemon and the
    extension can never disagree about what an installed
    ``_version.py`` says.
    """
    try:
        text = vfile.read_text(encoding="utf-8")
    except Exception:
        return ""
    m = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
    return m.group(1) if m else ""


def _scan_installed_extension_versions(root: Path) -> list[str]:
    """Return every ``__version__`` found under installed KISS extensions.

    Scans direct children of ``root`` for the KISS Sorcar extension
    naming convention (``ksenxx.kiss-sorcar-<VERSION>``) and reads
    each ``kiss_project/src/kiss/core/_version.py`` (the canonical
    location since the version literal moved into ``kiss.core``),
    falling back to the pre-move ``kiss_project/src/kiss/_version.py``
    for extensions installed before the move.  Malformed / missing
    version files are silently skipped so a single broken sibling
    cannot mask an otherwise-valid newer install.
    """
    out: list[str] = []
    try:
        entries = list(root.iterdir())
    except (OSError, ValueError):
        return out
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        if not entry.name.startswith(_EXTENSION_DIR_PREFIX):
            continue
        kiss_dir = entry / "kiss_project" / "src" / "kiss"
        v = _parse_version_py(kiss_dir / "core" / "_version.py")
        if not v:
            v = _parse_version_py(kiss_dir / "_version.py")
        if v:
            out.append(v)
    return out


def _read_version() -> str:
    r"""Return the version reported to ``update_available`` broadcasts.

    Historically this only read the daemon's own bundled
    ``_version.py``.  That was correct as long as the running daemon
    binary and the currently-installed extension matched — but the
    kiss-web launch agent / systemd unit is only ``launchctl
    kickstart -k``\ ed by ``install.sh`` during an upgrade, so the
    supervisor keeps respawning the OLD binary (bundled with the OLD
    ``_version.py``) after the update finishes.  Reporting the OLD
    version as "current" against a PyPI ``latest`` equal to the NEW
    version caused the sticky "update available" toast to re-appear
    with the same NEW-version text the user just clicked.

    Fix: pick the newest ``__version__`` found under
    ``<extensions_root>/ksenxx.kiss-sorcar-*/kiss_project/src/kiss/core/
    _version.py`` so a freshly-installed extension dominates the answer even when
    the running daemon is still the stale one.  Falls back to the
    bundled ``_version.py`` for developer / Docker installs where the
    extension dir does not exist.
    """
    root = _INSTALLED_EXTENSIONS_ROOT
    if root is None:
        root = Path.home() / ".vscode" / "extensions"
    best: tuple[int, ...] | None = None
    best_str = ""
    for v in _scan_installed_extension_versions(root):
        t = _version_tuple(v)
        if t is None:
            continue
        if best is None or t > best:
            best = t
            best_str = v
    if best_str:
        return best_str
    return _parse_version_py(
        Path(__file__).parent.parent / "core" / "_version.py",
    )


def _version_tuple(v: str) -> tuple[int, ...] | None:
    """Return ``v`` parsed as an int-tuple, or ``None`` on failure.

    ``kiss-agent-framework`` uses CalVer ``YYYY.M.P``; ``None`` is
    returned for anything that cannot be parsed so a malformed PyPI
    payload never triggers a false "update available" notification.
    Each dot-separated component must be ASCII digits only (the strict
    ``/^\\d+$/`` check of ``versionTuple`` in the extension's
    ``UpdateChecker.js``, this helper's documented twin) — a bare
    ``int()`` also accepts ``"+1"``, ``"1_0"`` and unicode digits,
    which would let the daemon and the extension disagree on update
    direction for malformed version strings.
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


_WS_SHIM_JS = r"""
// WebSocket shim for the remote webapp: provides acquireVsCodeApi()
// so the extension's media/main.js + media/api.js run unmodified in a
// plain browser.  Every frame sent through it is a command of the
// server API catalog defined in src/kiss/server/sorcar.py (dispatched
// by kiss.server.sorcar.ServerApi.dispatch); the pre-app ``auth``
// handshake frames sent below are serviced by
// kiss.server.sorcar.ServerApi.authenticate before the daemon starts
// dispatching this connection's commands.
(function() {
  var _state = null;
  try { _state = JSON.parse(sessionStorage.getItem('sorcar-state')); } catch(e) {}
  var _ws = null;
  var _pending = [];
  var _authenticated = false;
  // Tracks whether this client has previously completed a full
  // auth handshake.  Once true, the next successful ``auth_ok``
  // after an ``onclose`` (i.e. a server restart or network blip)
  // means the page state is stale relative to the freshly booted
  // backend and we must reload the page so the normal load
  // pipeline replays history, restored tabs, etc.  Without
  // this the page only re-binds the socket and the
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

  // Non-zero while the server has told us (via an ``auth_locked``
  // frame) that this source IP is rate-limited after too many failed
  // logins.  Holds the retry delay in milliseconds; consumed by the
  // ``onclose`` handler so the follow-up reconnect waits out the
  // lockout instead of hammering the server on the fast backoff, and
  // so the overlay keeps showing the lockout explanation rather than
  // the generic "starting ..." label.
  var _lockedRetryMs = 0;

  /**
   * Replace the overlay text with the auth-lockout explanation.
   *
   * Shown when the server answers the handshake with ``auth_locked``:
   * without this the user would stare at a promptless "KISS Sorcar
   * Server is starting ..." spinner with no hint that the remote
   * password rate-limit is what is keeping the password modal away.
   */
  function _showLockedMsg(secs) {
    var msg = document.getElementById('kiss-server-loading-msg');
    if (!msg) return;
    msg.textContent = 'Too many failed login attempts. ' +
      'Asking for the password again in ' + secs + 's ...';
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
        // pipeline (history replay, restored tabs, ...) runs
        // against the new server state.
        // The reload is gated by ``_hadAuthThenClosed`` so the
        // very first authentication on a fresh page load does NOT
        // reload (otherwise we would loop forever).
        if (_hadAuthThenClosed) {
          try { window.location.reload(); } catch (e) {}
          return;
        }
        _authenticated = true;
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
          if (pwd === null || pwd === undefined) {
            // SECURITY — do NOT leave the app usable when the user
            // dismisses the password prompt without authenticating.
            // The ``auth_required`` branch above revealed ``#app`` so
            // the modal (a child of #app) could render; once the modal
            // is cancelled that reveal would otherwise expose the whole
            // unauthenticated webapp, bypassing the remote-password
            // check at the UI layer.  Re-gate by re-showing the loading
            // overlay (``connected:false``).  The still-open, still-
            // unauthenticated socket times out server-side and the
            // ensuing reconnect re-prompts for the password.
            window.dispatchEvent(new MessageEvent('message', {
              data: {type: 'daemonStatus', connected: false}
            }));
            return;
          }
          try { localStorage.setItem('sorcar-remote-pwd', pwd); } catch(e) {}
          if (_ws && _ws.readyState === WebSocket.OPEN) {
            _ws.send(JSON.stringify({type: 'auth', password: pwd}));
          }
        });
        return;
      }
      if (msg.type === 'auth_locked') {
        // The server refused the handshake because this source IP is
        // rate-limited after too many failed logins (behind the
        // cloudflared tunnel EVERY visitor shares one loopback IP, so
        // this can be someone else's guesses).  The server closes the
        // socket right after this frame.  Explain the wait on the
        // overlay and remember the retry delay so ``onclose`` waits
        // out the lockout instead of reconnecting on the fast backoff
        // — the eventual reconnect gets ``auth_required`` again and
        // the password modal finally appears.
        var secs = Math.ceil(Number(msg.retry_after));
        if (!(secs > 0)) secs = 60;
        _lockedRetryMs = secs * 1000;
        _showLockedMsg(secs);
        // Re-gate the app while we wait (idempotent when the loading
        // overlay is already up, e.g. on a fresh page load).
        window.dispatchEvent(new MessageEvent('message', {
          data: {type: 'daemonStatus', connected: false}
        }));
        return;
      }
      // SECURITY — never forward server data frames to the app before
      // the connection is authenticated.  ``auth_ok`` / ``auth_required``
      // are handled above; any other frame that arrives while
      // ``_authenticated`` is false must be dropped so an unauthenticated
      // client can never act on backend data (defense in depth — the
      // server does not send data pre-auth, but a bug or a hostile proxy
      // must not be able to bypass the remote-password gate this way).
      if (!_authenticated) return;
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
      if (_lockedRetryMs > 0) {
        // This close follows an ``auth_locked`` frame: the server is
        // rate-limiting this IP after too many failed logins.  Keep
        // the lockout explanation on the overlay (do NOT overwrite it
        // with the generic label below) and hold off reconnecting
        // until the server-provided lockout expiry — the fast backoff
        // would only harvest more silent refusals.  The wake-up
        // listeners may still reconnect earlier; the server then just
        // re-sends ``auth_locked`` with a fresher ``retry_after``.
        var lockedDelay = _lockedRetryMs;
        _lockedRetryMs = 0;
        window.dispatchEvent(new MessageEvent('message', {
          data: {type: 'daemonStatus', connected: false}
        }));
        try { clearTimeout(_reconnectTimer); } catch (e) {}
        _reconnectTimer = setTimeout(function () {
          _reconnectTimer = null;
          connect();
        }, lockedDelay);
        return;
      }
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


_translate_webview_command = sorcar_api.translate_webview_command


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
        uds_path: str | Path | None = None,
        ntfy_base_url: str = _NTFY_BASE_URL,
    ) -> None:
        source_shell_env()
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
        self.work_dir: str = work_dir or ""

        self._voice_speaker_identifier: SpeakerIdentifier | None = None
        self._voice_speaker_broken = False
        self._voice_speaker_lock = threading.Lock()

        self._printer = WebPrinter()
        self._printer.work_dir = self.work_dir
        self._vscode_server = VSCodeServer(printer=self._printer)
        if self.work_dir:
            self._vscode_server.work_dir = self.work_dir
        self._server_api = sorcar_api.ServerApi(self)
        self._voice_wake = VoiceWakeController()

        self._html_bytes = _build_html().encode("utf-8")
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
        self._ntfy_base_url = ntfy_base_url
        self._last_posted_url: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_server: Any = None
        self._uds_path: Path = (
            Path(uds_path) if uds_path else _default_uds_path()
        )
        self._uds_server: asyncio.Server | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._latest_version: str | None = None
        self._version_check_task: asyncio.Task[None] | None = None
        self._shutdown_initiated = False
        self._shutdown_future: asyncio.Future[None] | None = None
        self._local_url = f"https://localhost:{self.port}"
        self._uds_handler_tasks: set[asyncio.Task[None]] = set()
        self._active_url: str | None = None
        self._last_ips: frozenset[str] = frozenset()
        self._pending_ip_change: frozenset[str] | None = None
        self._pending_ip_change_count: int = 0
        self._auth_failures: dict[str, list[float]] = {}
        self._install_root: Path = _KISS_AI_ROOT
        self._update_log_path: Path = _kiss_home_dir() / "update.log"
        self._update_proc: subprocess.Popen[bytes] | None = None
        self._update_starting = False
        self._lifecycle_lock = asyncio.Lock()
        self._uds_inode: int | None = None

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
        if path in ("", "/"):
            return _http_response(200, "text/html; charset=utf-8", self._html_bytes)
        if path == "/ws":
            return None
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
        if path == "/voice-model.tar.gz":
            model_file = await asyncio.to_thread(_ensure_voice_model)
            if model_file is None:
                return _http_response(
                    502, "text/plain", b"voice model unavailable"
                )
            body = await asyncio.to_thread(model_file.read_bytes)
            return _http_response(200, "application/gzip", body)
        if path.startswith("/media/"):
            filepath = MEDIA_DIR / path[7:]
            media_body = await asyncio.to_thread(_read_media_file, filepath)
            if media_body is not None:
                ctype = mimetypes.guess_type(str(filepath))[0] or "application/octet-stream"
                return _http_response(200, ctype, media_body)
        return _http_response(404, "text/plain", b"Not Found")


    @staticmethod
    def _passwords_equal(a: str, b: str) -> bool:
        """Constant-time string compare to defeat timing attacks.

        Alias of :func:`kiss.server.sorcar.passwords_equal` (the
        server API owns the auth handshake; this staticmethod is kept
        for existing callers and tests).
        """
        return sorcar_api.passwords_equal(a, b)

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
        if peer_ip and _is_loopback_ip(peer_ip):
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
        fails = self._auth_failures.get(ip, [])
        fails = [t for t in fails if now - t <= _AUTH_FAIL_WINDOW]
        if fails:
            self._auth_failures[ip] = fails
        else:
            self._auth_failures.pop(ip, None)
        if len(fails) < _AUTH_FAIL_MAX:
            return 0.0
        return max(0.0, _AUTH_LOCKOUT - (now - fails[-1]))

    def _is_auth_locked(self, ip: str) -> bool:
        """Return True if *ip* is currently rate-limited.

        Thin wrapper over :meth:`_auth_lock_remaining` kept for
        callers/tests that only need the boolean answer.
        """
        return self._auth_lock_remaining(ip) > 0.0

    def _record_auth_failure(self, ip: str) -> None:
        """Record a failed authentication attempt from *ip*.

        Also sweeps fully-expired entries for EVERY tracked IP:
        :meth:`_is_auth_locked` only prunes the entry of the IP that
        reconnects, so on a public tunnel an attacker rotating source
        addresses would otherwise leave one stale entry per distinct
        IP forever.  The sweep bounds the dict to IPs that failed
        within the last :data:`_AUTH_FAIL_WINDOW` seconds.
        """
        now = time.monotonic()
        for other_ip in list(self._auth_failures):
            kept = [
                t for t in self._auth_failures[other_ip]
                if now - t <= _AUTH_FAIL_WINDOW
            ]
            if kept:
                self._auth_failures[other_ip] = kept
            else:
                del self._auth_failures[other_ip]
        self._auth_failures.setdefault(ip, []).append(now)

    async def _authenticate_ws(self, websocket: ServerConnection) -> bool:
        """Authenticate a WebSocket client using the configured password.

        Returns True on success, False (and closes the socket) on failure.

        When the configured ``remote_password`` is empty, all clients
        are still required to send an empty-password ``auth`` message
        (using a constant-time compare).  See also
        :meth:`_setup_server` which refuses to advertise the public
        cloudflared tunnel when no password is configured.

        Transport wrapper: the handshake protocol itself (the
        ``auth`` / ``auth_ok`` / ``auth_required`` / ``auth_locked``
        exchange, the rate-limit refusal, and the
        only-non-empty-guesses-count lockout rule) is part of the
        server API and lives in
        :meth:`kiss.server.sorcar.ServerApi.authenticate`, which
        calls back into this server's :meth:`_client_ip` /
        :meth:`_auth_lock_remaining` / :meth:`_record_auth_failure`
        primitives.
        """
        return await self._server_api.authenticate(websocket)

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
        conn_state: dict[str, Any] = {
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
                        cmd, websocket, conn_state,
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
            self._vscode_server.drop_connection_state(conn_state["conn_id"])
            self._printer.unbind_conn(conn_state["conn_id"])
            self._printer.remove_client(websocket)

    async def _uds_handler(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a local Unix-domain socket client connection.

        Speaks the same JSON command protocol as :meth:`_ws_handler`
        but framed as newline-delimited JSON lines instead of
        WebSocket frames, and skips the password challenge — POSIX
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
        task = asyncio.current_task()
        if task is not None:
            # Tracked so stop_async can DRAIN in-flight handlers:
            # closing the client writer unblocks readline(), but
            # without a join the handler (and its cleanup finally)
            # may still be mid-flight after shutdown returns.
            self._uds_handler_tasks.add(task)
            task.add_done_callback(self._uds_handler_tasks.discard)
        self._printer.add_uds_writer(writer)
        conn_state: dict[str, Any] = {
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
                        cmd, writer, conn_state,
                    )
                except (ConnectionError, asyncio.IncompleteReadError):
                    raise
                except Exception:
                    logger.warning(
                        "Error handling UDS command %r; connection kept",
                        cmd.get("type", ""), exc_info=True,
                    )
        except Exception:
            logger.debug("UDS handler error", exc_info=True)
        finally:
            # A daemon-hosted wake-word listener is owned by exactly
            # this connection: reap it here so a closed VS Code window
            # can never leak a mic-holding child process.
            try:
                await self._voice_wake.stop(conn_state["conn_id"])
            except Exception:
                logger.debug(
                    "voice-wake stop on disconnect failed", exc_info=True,
                )
            local_tabs = conn_state.get("local_tabs")
            self._printer.unregister_local_uds_tabs(
                conn_state["conn_id"],
                local_tabs if isinstance(local_tabs, set) else set(),
            )
            self._vscode_server.drop_connection_state(conn_state["conn_id"])
            self._printer.unbind_conn(conn_state["conn_id"])
            self._printer.remove_uds_writer(writer)
            try:
                writer.close()
            except Exception:
                logger.debug("UDS writer close failed", exc_info=True)


    async def _dispatch_client_command(
        self,
        cmd: dict[str, Any],
        endpoint: Any,
        conn_state: dict[str, Any],
    ) -> None:
        """Hand one parsed client command to the server's code API.

        Single shared per-message entry point for :meth:`_ws_handler`
        (remote browsers) and :meth:`_uds_handler` (the local VS Code
        extension), so the two transports cannot drift in behaviour.
        This method owns NO routing: it wraps the connection's
        transport state into a :class:`kiss.server.sorcar.ApiContext`
        and calls :meth:`kiss.server.sorcar.ServerApi.dispatch`, which
        validates the command against the API catalog, applies the
        per-connection stamping (``connId``, per-window ``workDir``,
        tab registration — see the invariant documentation on
        :class:`ServerApi`), and invokes the API method the command's
        catalog entry names, with this server as the backend.

        Args:
            cmd: The parsed JSON command dictionary.
            endpoint: The client connection — a
                :class:`ServerConnection` (WSS) or an
                :class:`asyncio.StreamWriter` (UDS).  Used for direct
                replies.
            conn_state: Per-connection mutable state holding the
                connection's own ``work_dir`` and unique ``conn_id``.
                Each VS Code window owns exactly one connection, and
                the API layer's stamping of these fields is what
                guarantees the per-window work_dir and autocomplete
                isolation invariants.
        """
        ctx = sorcar_api.ApiContext(
            endpoint=endpoint,
            conn_state=conn_state,
            is_uds=isinstance(endpoint, asyncio.StreamWriter),
        )
        await self._server_api.dispatch(cmd, ctx)

    def _broadcast_to_conn(self, event: dict[str, Any], conn_id: str) -> None:
        """Broadcast *event*, stamped with *conn_id* when non-empty.

        A non-empty ``conn_id`` makes :meth:`WebPrinter.broadcast`
        deliver the event ONLY to the requesting connection (the VS
        Code window / browser tab whose user triggered the command),
        so siblings do not pop a banner; ``""`` broadcasts to all.
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
            "message": "Restarting the KISS Sorcar web server…",
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
        if self._update_starting or (
            self._update_proc is not None
            and self._update_proc.poll() is None
        ):
            # Single-flight guard (F4-13): two windows clicking
            # "Update" concurrently must not launch two installers
            # that fetch/reset/overwrite the same tree in parallel.
            self._broadcast_to_conn({
                "type": "notice",
                "text": (
                    "A KISS Sorcar update is already running… "
                    f"(output: {self._update_log_path})"
                ),
            }, conn_id)
            return
        self._update_starting = True
        script = await loop.run_in_executor(
            None, _find_install_script, self._install_root,
        )
        if script is None:
            self._update_starting = False
            self._broadcast_to_conn({
                "type": "error",
                "text": (
                    "Cannot update KISS Sorcar: install.sh not found "
                    f"in {self._install_root}."
                ),
            }, conn_id)
            return
        self._broadcast_to_conn({
            "type": "notice",
            "text": (
                "An update of KISS Sorcar is getting installed… "
                f"(output: {self._update_log_path})"
            ),
        }, conn_id)
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
                self._update_proc = subprocess.Popen(
                    ["bash", str(script)],
                    cwd=str(script.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            self._broadcast_to_conn({
                "type": "error",
                "text": f"Failed to start KISS Sorcar update: {exc}",
            }, conn_id)
        finally:
            self._update_starting = False

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

        Browser-mode voice.js cannot call gpt-audio itself (the API
        key lives on this machine), so after the in-page wake-word
        detector hears "Sorcar" it captures the utterance that follows
        and ships it here as ``{type: 'voiceTranscribe', audio:
        <base64 16kHz mono s16le PCM>}``.  The audio is translated
        into English by the same KISS transcription agent
        (:func:`transcribe_pcm`) the VS Code extension's local
        listener uses — which also reports the language that was
        spoken — the speaker is identified locally (best effort), and
        the client gets back ``{type: 'voiceSpeech', text, speaker,
        language}`` — the exact message voice.js already consumes in
        webview mode, so the page inserts and submits the dictated
        task with no extra client logic.  An empty/undecodable audio
        payload or a failed translation replies with an empty ``text``
        so the page can clear its transcribing indicator.

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
            try:
                result = await self._loop.run_in_executor(
                    None, transcribe_pcm, pcm,
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
                speaker = await self._loop.run_in_executor(
                    None, self._identify_voice_speaker, pcm,
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

    async def _handle_get_default_model(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Reply with the key-derived default model name.

        Services the ``getDefaultModel`` command (routed by
        :meth:`kiss.server.sorcar.ServerApi.get_default_model`): the
        VS Code extension host historically shelled out to ``uv run
        python -c ...`` for this value; over the socket the daemon
        answers from its own environment instead.  The reply is a
        single direct ``{"type": "defaultModel", "model": <name>}``
        event to the requesting *endpoint* — never broadcast.

        Args:
            cmd: The parsed ``getDefaultModel`` command (unused).
            endpoint: The client connection to reply to.
        """
        model = await asyncio.to_thread(get_default_model)
        await self._endpoint_send(
            endpoint,
            json.dumps({"type": "defaultModel", "model": model}),
        )

    async def _handle_read_kiss_config(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Reply with the raw merged ``~/.kiss/config.json`` contents.

        Services the ``readKissConfig`` command (routed — and gated to
        local UDS clients — by
        :meth:`kiss.server.sorcar.ServerApi.read_kiss_config`).  The
        reply is a single direct ``{"type": "kissConfig", "config":
        {...}}`` event to the requesting *endpoint* — never broadcast
        — carrying :func:`kiss.core.vscode_config.load_config`'s
        sanitized, defaults-merged view of the file, i.e. exactly
        what the daemon itself acts on.

        Args:
            cmd: The parsed ``readKissConfig`` command (unused).
            endpoint: The client connection to reply to.
        """
        cfg = await asyncio.to_thread(load_config)
        await self._endpoint_send(
            endpoint,
            json.dumps({"type": "kissConfig", "config": cfg}),
        )

    async def _handle_write_kiss_config(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Merge the command's ``config`` keys into ``config.json``.

        Services the ``writeKissConfig`` command (routed — and gated
        to local UDS clients — by
        :meth:`kiss.server.sorcar.ServerApi.write_kiss_config`).
        Delegates to :func:`kiss.core.vscode_config.save_config`, so
        the write shares the daemon's atomic, lock-guarded merge path
        (existing keys are preserved, API keys are never written) and
        the freshly saved state is re-applied to the daemon's
        environment exactly like a settings-panel ``saveConfig``.
        The reply is a single direct ``{"type": "kissConfigSaved",
        "ok": <bool>[, "error": <msg>]}`` acknowledgement to the
        requesting *endpoint* — never broadcast.

        Args:
            cmd: The parsed ``writeKissConfig`` command whose
                ``config`` field must be a JSON object.
            endpoint: The client connection to reply to.
        """
        data = cmd.get("config")
        if not isinstance(data, dict):
            await self._endpoint_send(endpoint, json.dumps({
                "type": "kissConfigSaved",
                "ok": False,
                "error": "config must be a JSON object",
            }))
            return
        try:
            await asyncio.to_thread(save_config, data)
            await asyncio.to_thread(
                lambda: apply_config_to_env(load_config())
            )
        except OSError as err:
            await self._endpoint_send(endpoint, json.dumps({
                "type": "kissConfigSaved",
                "ok": False,
                "error": f"failed to save config: {err}",
            }))
            return
        await self._endpoint_send(
            endpoint, json.dumps({"type": "kissConfigSaved", "ok": True}),
        )

    async def _handle_voice_wake_start(
        self, cmd: dict[str, Any], endpoint: Any, conn_id: str,
    ) -> None:
        """Start a daemon-hosted wake-word listener for one connection.

        Services the ``voiceWakeStart`` command (routed — and gated to
        local UDS clients — by
        :meth:`kiss.server.sorcar.ServerApi.voice_wake_start`).  The
        listener child process is owned by *conn_id*: its protocol
        lines stream back to the requesting *endpoint* as
        ``voiceWakeEvent`` / ``voiceWakeState`` events (see
        :mod:`kiss.server.voice_wake_control`), and the
        :meth:`_uds_handler` disconnect cleanup stops it when the
        connection goes away.

        Args:
            cmd: The parsed ``voiceWakeStart`` command; its optional
                ``sensitivity`` field (0..100) tunes wake eagerness.
            endpoint: The client connection to stream events to.
            conn_id: The owning connection's id.
        """
        raw = cmd.get("sensitivity")
        sensitivity = (
            round(raw)
            if isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(raw)
            else None
        )

        async def _send(event: dict[str, Any]) -> None:
            await self._endpoint_send(endpoint, json.dumps(event))

        await self._voice_wake.start(conn_id, sensitivity, _send)

    async def _handle_voice_wake_stop(self, conn_id: str) -> None:
        """Stop *conn_id*'s daemon-hosted wake-word listener, if any.

        Services the ``voiceWakeStop`` command (routed — and gated to
        local UDS clients — by
        :meth:`kiss.server.sorcar.ServerApi.voice_wake_stop`); also
        called by the UDS disconnect cleanup, so it is a no-op when
        the connection owns no listener.

        Args:
            conn_id: The owning connection's id.
        """
        await self._voice_wake.stop(conn_id)

    async def _handle_open_file(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Read a file for a remote-web client and reply with its content.

        Handles the ``openFile`` command sent by ``media/main.js`` when
        the user clicks a file link (``span[data-path]``) in a chat
        webview served by the remote webapp.  The reply is a single
        ``fileContent`` JSON object sent directly to the requesting
        *endpoint* via :meth:`_endpoint_send` — never broadcast — with
        the shape::

            {"type": "fileContent", "path": <resolved abs path>,
             "name": <basename>, "tabId": <echo of cmd tabId>,
             "content": <utf-8 text>}          # on success
            {"type": "fileContent", "path": ..., "name": ...,
             "tabId": ..., "error": <message>}  # on failure

        Relative paths are resolved against the command's ``workDir``
        (stamped per-connection by
        :meth:`kiss.server.sorcar.ServerApi.dispatch`) and
        fall back to the daemon work dir.  Missing files, unreadable
        files, files larger than :data:`_OPEN_FILE_MAX_BYTES`, and
        binary files (NUL byte in the first 8 KiB) produce an ``error``
        reply instead of content.

        Args:
            cmd: The parsed ``openFile`` command (``path``, optional
                ``workDir``, ``tabId``, ``line``).
            endpoint: The requesting WSS connection.
        """
        raw_path = cmd.get("path", "")
        if not isinstance(raw_path, str) or not raw_path:
            return
        work_dir = cmd.get("workDir", "")
        if not isinstance(work_dir, str) or not work_dir:
            work_dir = self._vscode_server.work_dir or self.work_dir
        tab_id = cmd.get("tabId", "")
        if not isinstance(tab_id, str):
            tab_id = ""

        def _read_file() -> dict[str, Any]:
            reply: dict[str, Any] = {
                "type": "fileContent",
                "path": raw_path,
                "name": Path(raw_path).name,
                "tabId": tab_id,
            }
            try:
                path = Path(os.path.expanduser(raw_path))
                if not path.is_absolute() and work_dir:
                    path = Path(work_dir) / path
                path = path.resolve()
                if not path.is_file():
                    reply["error"] = f"File not found: {raw_path}"
                    return reply
                if path.stat().st_size > _OPEN_FILE_MAX_BYTES:
                    reply["error"] = f"File too large to display: {raw_path}"
                    return reply
                data = path.read_bytes()
                if b"\0" in data[:8192]:
                    reply["error"] = f"Cannot display binary file: {raw_path}"
                    return reply
                reply["path"] = str(path)
                reply["name"] = path.name
                reply["content"] = data.decode("utf-8", errors="replace")
            except OSError as exc:
                reply["error"] = f"Failed to read {raw_path}: {exc}"
            return reply

        reply = await asyncio.to_thread(_read_file)
        try:
            await self._endpoint_send(endpoint, json.dumps(reply))
        except Exception:
            logger.debug("openFile: failed to write reply", exc_info=True)

    async def _handle_check_paths(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None:
        """Tell a remote-web client which candidate file paths exist.

        Handles the ``checkPaths`` command sent by ``media/main.js``
        after it linkifies file-path-looking strings in event panel
        contents: a path is rendered as a clickable link ONLY when this
        check confirms it names an existing regular file, i.e. that a
        subsequent ``openFile`` click would actually serve content.
        Paths are resolved exactly like :meth:`_handle_open_file`
        resolves them (``~`` expansion, then relative to the command's
        ``workDir``, falling back to the daemon work dir).  The reply
        is sent directly to the requesting *endpoint* — never
        broadcast — with the shape::

            {"type": "pathsExist", "results": {<path>: <bool>, ...},
             "workDir": <echo of cmd workDir>,
             "tabId": <echo of cmd tabId>}

        Args:
            cmd: The parsed ``checkPaths`` command (``paths``, optional
                ``workDir``, ``tabId``).
            endpoint: The requesting WSS connection.
        """
        raw_paths = cmd.get("paths")
        if not isinstance(raw_paths, list):
            raw_paths = []
        raw_work_dir = cmd.get("workDir", "")
        if not isinstance(raw_work_dir, str):
            raw_work_dir = ""
        work_dir = raw_work_dir
        if not work_dir:
            work_dir = self._vscode_server.work_dir or self.work_dir
        tab_id = cmd.get("tabId", "")
        if not isinstance(tab_id, str):
            tab_id = ""

        def _check_paths() -> dict[str, bool]:
            results: dict[str, bool] = {}
            for raw_path in raw_paths:
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                try:
                    path = Path(os.path.expanduser(raw_path))
                    if not path.is_absolute() and work_dir:
                        path = Path(work_dir) / path
                    results[raw_path] = path.resolve().is_file()
                except OSError:
                    results[raw_path] = False
            return results

        results = await asyncio.to_thread(_check_paths)
        reply = {
            "type": "pathsExist",
            "results": results,
            "workDir": raw_work_dir,
            "tabId": tab_id,
        }
        try:
            await self._endpoint_send(endpoint, json.dumps(reply))
        except Exception:
            logger.debug("checkPaths: failed to write reply", exc_info=True)

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
                None, _post_url_to_message_board, url, self._ntfy_base_url,
            )
            self._last_posted_url = url

    async def _send_welcome_info(self) -> None:
        """Broadcast the active remote URL to all connected clients.

        Broadcasts the ``remote_url`` event using the in-memory URL,
        the URL file, or — for tunnel-enabled servers only — the
        ``cloudflared`` metrics API as successive fallbacks.

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
            if discovered:
                await loop.run_in_executor(
                    None, _save_url_file,
                    self._url_file, self._local_url, discovered,
                )
                self._active_url = discovered
                url = discovered
        tunnel_active = bool(
            self.use_tunnel and url and url != self._local_url
        )
        self._broadcast_remote_url(url or "", tunnel_active)
        await self._broadcast_update_available()

    async def _endpoint_send(self, endpoint: Any, data: str) -> None:
        """Send ``data`` to either a WSS or a UDS endpoint.

        ``endpoint`` is either a :class:`ServerConnection` (WSS) or
        an :class:`asyncio.StreamWriter` (UDS).  This helper hides
        the protocol difference so :meth:`_handle_ready` and
        :meth:`_uds_handler` share a single dispatch path.

        Acquires the printer's per-endpoint send lock so a direct
        reply cannot overtake a broadcast event already in flight to
        the same client: ``Connection.send`` waits out write
        backpressure BEFORE queuing the frame, so without the lock a
        suspended earlier sender (e.g. the ``task_events`` replay
        scheduled by ``resumeSession``) could hit the wire AFTER a
        later direct send, inverting the wire order the client
        depends on.

        Args:
            endpoint: The connection to send to.
            data: The JSON payload (already encoded with ``json.dumps``).
        """
        async with self._printer.send_lock(endpoint):
            if isinstance(endpoint, asyncio.StreamWriter):
                endpoint.write(data.encode("utf-8") + b"\n")
                await endpoint.drain()
            else:
                await endpoint.send(data)

    @staticmethod
    def _sanitized_restored_tabs(cmd: dict[str, Any]) -> list[dict[str, str]]:
        """Sanitize the ``restoredTabs`` field of a ``ready`` command.

        Single source of the M7 hardening shared by the ``ready``
        handler of the server API
        (:meth:`kiss.server.sorcar.ServerApi.ready`) and
        :meth:`_handle_ready`:

        * caps the list at ``_MAX_RESTORED_TABS`` so an
          authenticated-but-malicious or buggy client cannot flood the
          executor with thousands of ``resumeSession`` jobs;
        * skips malformed (non-dict) elements — an ``AttributeError``
          would propagate out of the command dispatch and tear
          down the whole authenticated connection over one bad field;
        * blanks non-str ``tabId`` / ``chatId`` values — a non-str
          ``chatId`` would flow into backend handlers that assume
          strings.

        Every rejection is logged with a ``warning``.  The dispatch
        path writes the cleaned list back into ``cmd`` so the second
        pass inside :meth:`_handle_ready` is a no-op (no duplicate
        warnings); direct callers of :meth:`_handle_ready` (replays,
        tests) still get the full sanitize.

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
        snapshot is broadcast, and every chat-bound registry tab is
        replayed so all connected clients converge on identical
        transcripts.  Tab state is server-canonical — clients never
        keep a tab set of their own — so the same path serves VS Code
        webviews (UDS) and remote web apps (WSS) alike.

        Args:
            cmd: The ``ready`` message from the client (already
                stamped with the connection's ``connId`` by
                :meth:`kiss.server.sorcar.ServerApi.dispatch`).
            websocket: The client connection (for direct replies).
        """
        tab_id = cmd.get("tabId", "")
        if not isinstance(tab_id, str):
            tab_id = ""
        conn_id = cmd.get("connId", "")
        work_dir = cmd.get("workDir", "")
        for init_cmd in ("getModels", "getInputHistory", "getConfig"):
            init: dict[str, Any] = {"type": init_cmd, "connId": conn_id}
            if work_dir:
                init["workDir"] = work_dir
            await self._run_cmd(init)
        try:
            await self._endpoint_send(
                websocket, json.dumps({"type": "tasks_updated"}),
            )
        except Exception:
            logger.debug("ready tasks_updated nudge failed", exc_info=True)
        await self._send_welcome_info()
        try:
            await self._endpoint_send(
                websocket,
                json.dumps({"type": "focusInput", "tabId": tab_id}),
            )
        except Exception:
            pass
        restored = self._sanitized_restored_tabs(cmd)
        try:
            bound = await asyncio.to_thread(
                self._vscode_server.ready_tab_sync, restored,
            )
        except Exception:
            logger.exception("ready tab-registry sync failed")
            bound = []
        for rt_id, rt_chat, rt_task in bound:
            resume: dict[str, Any] = {
                "type": "resumeSession", "chatId": rt_chat,
                "tabId": rt_id,
            }
            # A tab pinned to a specific historical task replays THAT
            # task; without the taskId the replay would silently
            # switch every client's tab to the chat's latest task.
            if rt_task:
                resume["taskId"] = rt_task
            await self._run_cmd(resume)

    async def _handle_submit(self, cmd: dict[str, Any]) -> None:
        """Translate the webview ``submit`` command into a backend ``run``.

        The VS Code TypeScript extension transforms ``submit`` into a
        ``run`` command after resolving paths and tracking running tabs.
        The web server performs the same translation.

        Args:
            cmd: The ``submit`` message from the browser.
        """
        tab_id = cmd.get("tabId", "")
        if self._shutdown_initiated:
            # Shutdown admission gate (F4-06): a task submitted after
            # the shutdown sweep snapshotted the active workers would
            # be silently killed when the process exits.
            self._printer.broadcast(
                {"type": "status", "running": False, "tabId": tab_id},
            )
            self._printer.broadcast({
                "type": "error",
                "text": "Server is shutting down; task not started.",
                "tabId": tab_id,
            })
            return
        prompt = cmd.get("prompt", "")
        if isinstance(prompt, str):
            prompt, prompt_size = _truncate_utf8_bytes(
                prompt, _MAX_PROMPT_BYTES,
            )
            if prompt_size > _MAX_PROMPT_BYTES:
                logger.warning(
                    "prompt size %d bytes exceeds cap %d bytes; truncating",
                    prompt_size, _MAX_PROMPT_BYTES,
                )
        attachments = cmd.get("attachments")
        if isinstance(attachments, list) and len(attachments) > _MAX_ATTACHMENTS:
            logger.warning(
                "attachments count %d exceeds cap %d; truncating",
                len(attachments), _MAX_ATTACHMENTS,
            )
            attachments = attachments[:_MAX_ATTACHMENTS]
        # NOTE: no setTaskText here — the common run path (_cmd_run)
        # broadcasts it for every origin, VS Code and remote alike.
        self._printer.broadcast({"type": "status", "running": True, "tabId": tab_id})
        run_cmd: dict[str, Any] = {
            "type": "run",
            "prompt": prompt,
            "model": cmd.get("model", ""),
            "workDir": cmd.get("workDir") or self._vscode_server.work_dir,
            "tabId": tab_id,
            "attachments": attachments,
            "useWorktree": cmd.get("useWorktree", True),
            "useParallel": cmd.get("useParallel", True),
            "autoCommit": cmd.get("autoCommit", True),
            # Carried over from the ``submit`` this run was built from:
            # ``_run_cmd`` bypasses the dispatcher that stamps it, so
            # without this a browser-launched task would record an
            # empty owning connection while the identical VS Code
            # ``run`` records the real one (F08-7).
            "connId": cmd.get("connId", ""),
        }
        await self._run_cmd(run_cmd)

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
                start_new_session=True,
            )
            try:
                proc.wait(timeout=_SPAWN_FAILFAST_WINDOW)
            except subprocess.TimeoutExpired:
                pass
            if proc.poll() is None:
                self._tunnel_proc = proc
                self._tunnel_started_at = time.monotonic()
                self._tunnel_adopted_pid = None
                _save_cloudflared_pidfile(
                    proc.pid, self._tunnel_metrics_port, None,
                )
                return
            if last_proc is not None:
                if last_proc.stderr is not None:
                    last_proc.stderr.close()
                last_proc.wait()
            last_proc = proc
            logger.info(
                "cloudflared exited immediately on metrics port %d "
                "(attempt %d/%d, rc=%s); retrying with fresh port",
                self._tunnel_metrics_port, attempt + 1, retries,
                proc.returncode,
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
            ["--url", self._local_url, "--no-tls-verify"],
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
            self._tunnel_adopted_pid = None
            self._tunnel_metrics_port = None
            self._tunnel_started_at = None
            self._tunnel_unhealthy_ticks = 0
            adopted_pid = None

        if proc is None and adopted_pid is None:
            cfg = await asyncio.to_thread(load_config)
            if not cfg.get("remote_password", ""):
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
        await asyncio.to_thread(
            _save_url_file, self._url_file, self._local_url, tunnel_url,
        )
        self._active_url = tunnel_url or self._local_url
        self._broadcast_remote_url(self._active_url, bool(tunnel_url))
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
        self._reset_tunnel_proc_state()

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
           (WiFi switch, DHCP renewal, VPN): in direct-LAN mode
           (``use_tunnel=False``) initiate a graceful shutdown so the
           daemon manager restarts the process on the new address; in
           tunnel mode only log the change — ``cloudflared``
           re-registers with the edge automatically, so no restart is
           needed.
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
        race with the new instance's ``_save_url_file`` and cause the
        VS Code sidebar to show no URL.
        """
        self._terminate_tunnel_proc()
        self._reset_tunnel_backoff_state()

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
        self._reset_tunnel_proc_state()
        self._reset_tunnel_backoff_state()

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
        self._loop = asyncio.get_running_loop()
        self._printer._loop = self._loop

        try:
            self._uds_path.parent.mkdir(parents=True, exist_ok=True)
            if self._uds_path.exists() or self._uds_path.is_symlink():
                if await self._uds_socket_is_live():
                    # Another live daemon owns this pathname (F4-03).
                    # Unlinking it would strand that daemon's clients
                    # on an unreachable inode; leave it alone and let
                    # local clients fall back to WSS.
                    raise OSError(
                        f"UDS socket {self._uds_path} is owned by "
                        "another live daemon; refusing to steal it",
                    )
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
            try:
                self._uds_inode = os.stat(self._uds_path).st_ino
            except OSError:
                self._uds_inode = None
        except Exception:
            logger.warning(
                "Failed to bind UDS at %s; local extension clients "
                "will fall back to WSS",
                self._uds_path, exc_info=True,
            )
            self._uds_server = None

        try:
            await self._setup_server_after_uds()
        except BaseException:
            # Rollback (F4-04): a TLS/WSS/tunnel failure or a
            # cancellation must not leave the already-bound UDS
            # listener (or a half-bound WSS listener) live in an
            # embedder that catches the exception.
            self._close_partial_setup()
            raise

    async def _uds_socket_is_live(self) -> bool:
        """Return True when a live peer accepts connections on the UDS path.

        Probes the existing socket pathname before startup unlinks it
        (F4-03) so one daemon cannot silently strand another live
        daemon's listener.
        """
        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._uds_path)),
                timeout=1.0,
            )
        except (OSError, TimeoutError, ValueError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            logger.debug("UDS probe close failed", exc_info=True)
        return True

    def _close_partial_setup(self) -> None:
        """Tear down listeners bound by a failed/cancelled ``_setup_server``."""
        if self._uds_server is not None:
            self._uds_server.close()
            self._uds_server = None
            self._unlink_own_uds_socket()
        if self._ws_server is not None:
            self._ws_server.close()
            self._ws_server = None

    def _unlink_own_uds_socket(self) -> None:
        """Unlink the UDS pathname only when it still names OUR socket.

        A successor daemon may have already rebound the shared
        pathname; blindly unlinking would strand its live listener
        (F4-03).  The inode recorded right after our bind is the
        ownership witness.
        """
        if self._uds_inode is None:
            # No ownership witness — fail CLOSED: never unlink a
            # pathname a successor daemon may have rebound.
            return
        try:
            if os.stat(self._uds_path).st_ino != self._uds_inode:
                return
        except OSError:
            return
        try:
            self._uds_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("UDS unlink failed", exc_info=True)

    async def _setup_server_after_uds(self) -> None:
        """Continue :meth:`_setup_server` after the UDS bind."""
        if self._ssl_context is None:
            self._ssl_context = await asyncio.to_thread(
                _create_ssl_context,
                self._ssl_certfile,
                self._ssl_keyfile,
            )

        last_err: OSError | None = None
        for attempt in range(_BIND_RETRY_ATTEMPTS):
            try:
                self._ws_server = await serve(
                    self._ws_handler,
                    self.host,
                    self.port,
                    process_request=self._process_request,
                    ssl=self._ssl_context,
                    open_timeout=_OPEN_TIMEOUT_SECONDS,
                    ping_interval=None,
                    ping_timeout=None,
                    max_size=_MAX_LINE_BYTES,
                    create_connection=_HeadAwareServerConnection,
                )
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

        tunnel_url: str | None = None
        if self.use_tunnel:
            password = await self._loop.run_in_executor(  # type: ignore[union-attr]
                None, _wait_for_remote_password, 30.0,
            )
            if password:
                adopted = await self._loop.run_in_executor(  # type: ignore[union-attr]
                    None, _try_adopt_existing_cloudflared,
                )
                if adopted is not None:
                    adopted_pid, adopted_port, adopted_url = adopted
                    self._tunnel_adopted_pid = adopted_pid
                    self._tunnel_metrics_port = adopted_port
                    self._tunnel_started_at = time.monotonic()
                    tunnel_url = adopted_url
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
                tunnel_url = await self._loop.run_in_executor(  # type: ignore[union-attr]
                    None, self._start_tunnel,
                )

        await asyncio.to_thread(
            _save_url_file, self._url_file, self._local_url, tunnel_url,
        )
        self._active_url = tunnel_url or self._local_url
        await self._post_url_if_changed()

        self._last_ips = await asyncio.to_thread(_get_local_ips)
        self._watchdog_task = asyncio.create_task(self._watchdog())
        self._version_check_task = asyncio.create_task(
            self._version_check_loop(),
        )

        self._maybe_schedule_server_reset_complete()

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
        print(f"KISS Sorcar remote access: {self._local_url}", file=sys.stderr)
        if self.use_tunnel and self._active_url != self._local_url:
            print(f"Cloudflare tunnel:         {self._active_url}", file=sys.stderr)
        elif self.use_tunnel:
            print("Warning: cloudflared tunnel failed to start", file=sys.stderr)
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
        if signum in (signal.SIGTERM, signal.SIGHUP):
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
                threading.Thread(
                    target=self._shutdown_on_sigterm,
                    name="kiss-sigterm-shutdown",
                    daemon=True,
                ).start()
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

        1. :meth:`_stop_active_agent_tasks` FIRST — the user-visible
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

        Args:
            timeout: Maximum wall-clock seconds to wait, in aggregate,
                for all in-flight merges.
        """
        from kiss.server import agent_state

        with agent_state.STATE_LOCK:
            threads = [
                state.merge_thread
                for state in agent_state.agent_states.values()
                if state.merge_thread is not None
                and state.merge_thread.is_alive()
            ]
        if not threads:
            return
        logger.warning(
            "Shutdown: waiting up to %.0fs for %d interactive merge(s) "
            "to finish rewriting the repository",
            timeout, len(threads),
        )
        deadline = time.monotonic() + timeout
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                logger.error(
                    "Shutdown: merge thread %s did not finish within the "
                    "grace period; proceeding without it",
                    thread.name,
                )

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

        ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
            ctypes.c_ulong,
            ctypes.py_object,
        ]

        from kiss.server import agent_state

        active: list[tuple[str, threading.Event | None, threading.Thread]] = []
        active_task_history_ids: set[str] = set()
        with agent_state.STATE_LOCK:
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
                    active.append((task_id, state.stop_event, thread))
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
            ", ".join(tab_id for tab_id, _, _ in active),
        )

        for _tab_id, stop_event, _thread in active:
            if stop_event is not None:
                stop_event.set()

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

        self._install_signal_handlers()

        try:
            asyncio.run(self._serve_async())
            if self._shutdown_initiated:
                logger.info("Server shutting down: pid=%d (SIGTERM)", pid)
        except KeyboardInterrupt:
            logger.info("Server shutting down: pid=%d (KeyboardInterrupt)", pid)
        finally:
            self._shutdown_initiated = True
            self._stop_active_agent_tasks()
            self._await_active_merges()
            self._disconnect_mcp_servers()
            # Re-persist tab-registry mutations whose save failed
            # (e.g. a briefly unwritable KISS dir); no-op when the
            # last save succeeded.
            self._vscode_server.tab_registry.flush()
            logger.info("Server stopped: pid=%d", pid)
            self._detach_tunnel()

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

    async def _drain_tasks(
        self, tasks: set[asyncio.Task[None]], timeout: float = 2.0,
    ) -> None:
        """Join *tasks*, cancelling any that outlive *timeout*.

        Shutdown helper: waits up to *timeout* seconds for the given
        asyncio tasks (in-flight UDS handlers, deferred tab-close
        tasks) to finish on their own — closed streams already
        unblock them — then cancels and awaits any stragglers so
        none can touch server state after shutdown completes.

        Args:
            tasks: Tasks to join; a snapshot is taken, and the
                current task (if present) is excluded.
            timeout: Seconds to wait before cancelling stragglers.
        """
        current = asyncio.current_task()
        pending = {t for t in tasks if t is not current and not t.done()}
        if not pending:
            return
        _done, pending = await asyncio.wait(pending, timeout=timeout)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

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

        Ordering: command ingress is quiesced FIRST — the WSS/UDS
        listeners are closed and every established UDS client stream
        is closed (F4-02) — and only then are the in-flight agent
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
            await _cancel_task(self._version_check_task)
            self._version_check_task = None
            if self._ws_server is not None:
                self._ws_server.close()
                try:
                    await asyncio.wait_for(
                        self._ws_server.wait_closed(), timeout=2,
                    )
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
                self._unlink_own_uds_socket()
            # asyncio.Server.close() does not close streams that were
            # already accepted: close every established UDS client so
            # its handler unblocks from readline() and exits (F4-02).
            for writer in list(self._printer._uds_writers):
                try:
                    writer.close()
                except Exception:
                    logger.debug(
                        "UDS client close on shutdown failed",
                        exc_info=True,
                    )
            # DRAIN in-flight UDS handlers: closing writers merely
            # unblocks readline(); the handler coroutines (and their
            # cleanup `finally` blocks) may still be running.  Join
            # them so no coroutine touches server state after
            # stop_async returns; cancel stragglers.
            await self._drain_tasks(set(self._uds_handler_tasks))
            # Reap daemon-hosted wake-word listeners: their owning
            # connections are gone (or going), and a leaked child
            # would keep the microphone open past shutdown.
            await self._voice_wake.stop_all()
            # An interactive merge/discard runs in the default executor,
            # not on a task thread: WAIT for it before anything else is
            # torn down, or the repository keeps being rewritten after
            # this method promised the server was down.
            await asyncio.to_thread(self._await_active_merges)
            await asyncio.to_thread(self._stop_active_agent_tasks)
            await asyncio.to_thread(self._disconnect_mcp_servers)
            # Re-persist tab-registry mutations whose save failed
            # (e.g. a briefly unwritable KISS dir) so they survive
            # the restart; a no-op when the last save succeeded.
            await asyncio.to_thread(self._vscode_server.tab_registry.flush)
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
