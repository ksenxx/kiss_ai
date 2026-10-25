# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The kiss-web daemon's local-endpoint file.

Same-machine clients (``run_agent`` / ``daemon_client.run``, cron
prompt jobs, the channel-agent launcher)
talk to the daemon over the same WSS listener browsers use.  What sets
them apart from a remote browser is a per-daemon-start secret: the
daemon writes it, with the listener's URL, to
``$KISS_HOME/sorcar-local.json`` (mode 0600, so only the owning user
can read it — the same access rule the old ``sorcar.sock`` Unix socket
had from its file mode).  A client that presents the token in its
``auth`` frame from a loopback address is treated as *local* and may
use the local-only commands (file saves, directory listings, ...).

The file doubles as the daemon's presence marker: it exists while a
daemon is up and names the daemon's pid, and the client's failed
connection to the URL is the "daemon is down" signal.  ``KISS_SORCAR_LOCAL``
overrides the file's location for clients that must reach a daemon
with a non-default ``KISS_HOME`` (tests, side-by-side installs).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import ssl
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as _ws_connect

from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.utils import atomic_write_text

logger = logging.getLogger(__name__)

LOCAL_ENDPOINT_FILE = "sorcar-local.json"
"""File name of the endpoint file under ``$KISS_HOME``."""

LOCAL_ENDPOINT_ENV = "KISS_SORCAR_LOCAL"
"""Environment variable that overrides the endpoint file's path."""


@dataclass(frozen=True)
class LocalEndpoint:
    """Where a local client connects and how it proves it is local.

    Attributes:
        url: The daemon's WebSocket URL (``ws://127.0.0.1:8787/ws``:
            plain on loopback; ``wss://`` is honoured too).
        token: The daemon's per-start local secret, sent as the
            ``token`` field of the client's ``auth`` frame.
        ca: Path of the PEM certificate the client should trust for a
            ``wss://`` URL (the daemon's local CA, or the custom
            certificate it was started with); ``None`` means the
            system trust store.
        pid: The daemon's process id (a presence hint for probes).
    """

    url: str
    token: str
    ca: str | None
    pid: int


def new_token() -> str:
    """Return a fresh 256-bit local token (hex)."""
    return secrets.token_hex(32)


def default_endpoint_path() -> Path:
    """Return the endpoint file path: ``$KISS_SORCAR_LOCAL`` or the default."""
    env = os.environ.get(LOCAL_ENDPOINT_ENV)
    return Path(env) if env else kiss_home() / LOCAL_ENDPOINT_FILE


_LOCK_TIMEOUT = 5.0
"""Seconds to wait for the endpoint lock before giving up."""


def _lock_path(path: Path) -> Path:
    """The lock file serialising publication and removal of *path*."""
    return path.with_name(f".{path.name}.lock")


@contextmanager
def _endpoint_lock(path: Path) -> Iterator[None]:
    """Hold the endpoint lock for *path*, waiting at most :data:`_LOCK_TIMEOUT`.

    The lock is held for microseconds by its two users, so a long wait
    means a stuck holder; failing then (instead of blocking the daemon's
    event loop for good) lets a startup roll back cleanly.

    Raises:
        TimeoutError: When the lock is not free within the deadline.
    """
    with exclusive_file_lock(_lock_path(path), timeout=_LOCK_TIMEOUT):
        yield


def write_endpoint(path: Path, endpoint: LocalEndpoint) -> None:
    """Atomically write *endpoint* to *path* with mode 0600.

    The temporary file is created with the final mode before it holds
    the token, so no reader ever sees a world-readable copy.  Holds the
    endpoint lock so a predecessor's :func:`remove_endpoint_if_owned`
    cannot interleave its ownership check with this publication and
    unlink the new file.

    Args:
        path: The endpoint file to write.
        endpoint: The daemon's local endpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with _endpoint_lock(path):
        atomic_write_text(path, json.dumps(asdict(endpoint), indent=2) + "\n", mode=0o600)


def read_endpoint(path: Path | None = None) -> LocalEndpoint | None:
    """Read the endpoint file at *path* (default: :func:`default_endpoint_path`).

    Returns:
        The endpoint, or ``None`` when the file is missing, unreadable
        or not a complete endpoint record (a half-written file, or one
        left by a daemon build that wrote another format).
    """
    if path is None:
        path = default_endpoint_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    url = data.get("url")
    token = data.get("token")
    ca = data.get("ca")
    pid = data.get("pid")
    if not isinstance(url, str) or not isinstance(token, str) or not token:
        return None
    if not url.startswith(("ws://", "wss://")):
        return None
    if ca is not None and not isinstance(ca, str):
        return None
    return LocalEndpoint(
        url=url, token=token, ca=ca, pid=pid if isinstance(pid, int) else 0,
    )


def remove_endpoint_if_owned(path: Path, token: str) -> None:
    """Delete *path* only while it still carries *token*.

    A successor daemon may already have written its own endpoint to the
    shared path; deleting that would leave its clients unable to find
    it.  The token written at bind time is the ownership witness, and
    the check-then-unlink runs under the same lock
    :func:`write_endpoint` takes, so a successor cannot publish between
    the two steps.

    Args:
        path: The endpoint file.
        token: The token this daemon wrote.
    """
    try:
        with _endpoint_lock(path):
            current = read_endpoint(path)
            if current is None or current.token != token:
                return
            path.unlink()
    except OSError:
        # Includes the lock's TimeoutError: a stale endpoint file left
        # behind makes every later client fail to connect, so say why.
        logger.warning("could not remove endpoint file %s", path, exc_info=True)


def client_ssl_context(endpoint: LocalEndpoint) -> ssl.SSLContext | None:
    """Return the TLS context a local client uses for *endpoint*.

    ``None`` for a plain ``ws://`` URL (the daemon serves loopback
    without TLS).  For ``wss://`` it trusts exactly the certificate
    file named by ``endpoint.ca`` (the daemon's local CA, whose private
    key is 0600 under ``$KISS_HOME``), falling back to the system
    trust store when no ``ca`` is recorded; hostname verification
    stays on (the daemon's certificate covers ``localhost``,
    ``127.0.0.1`` and ``::1``).

    Args:
        endpoint: The endpoint read from the endpoint file.

    Returns:
        A client-side ``ssl.SSLContext``, or ``None`` for ``ws://``.
    """
    if endpoint.url.startswith("ws://"):
        return None
    if endpoint.ca:
        return ssl.create_default_context(cafile=endpoint.ca)
    return ssl.create_default_context()


def connect(
    endpoint_file: Path | None = None,
    *,
    open_timeout: float = 10.0,
    max_size: int | None = None,
) -> ClientConnection:
    """Connect to the daemon named by *endpoint_file* as a local client.

    Reads the endpoint file, opens the TLS WebSocket (permessage-deflate
    declined: it saves nothing on loopback) and completes the ``auth``
    handshake with the local token.  The returned connection is ready
    for catalog commands.

    Args:
        endpoint_file: The endpoint file to read; ``None`` means
            :func:`default_endpoint_path`.
        open_timeout: Seconds allowed for the TCP/TLS/WebSocket opening
            handshake and for the ``auth_ok`` reply.
        max_size: Largest incoming frame accepted, in bytes (``None``
            for the ``websockets`` default).

    Returns:
        The authenticated synchronous client connection.

    Raises:
        ConnectionError: When no endpoint file exists, the daemon does
            not answer, the TLS/WebSocket handshake fails, or the
            daemon does not accept the token as local.
    """
    path = endpoint_file if endpoint_file is not None else default_endpoint_path()
    endpoint = read_endpoint(path)
    if endpoint is None:
        raise ConnectionError(
            f"Cannot connect to the sorcar daemon: no endpoint file at {path} "
            "— start it with `kiss-web`."
        )
    try:
        ws = _ws_connect(
            endpoint.url,
            ssl=client_ssl_context(endpoint),
            open_timeout=open_timeout,
            close_timeout=2.0,
            compression=None,
            max_size=max_size,
        )
    except (OSError, InvalidHandshake, InvalidURI) as exc:
        raise ConnectionError(
            f"Cannot connect to the sorcar daemon at {endpoint.url}: {exc} "
            "— start it with `kiss-web`."
        ) from exc
    try:
        send(ws, json.dumps({"type": "auth", "token": endpoint.token}), timeout=open_timeout)
        reply = recv_auth_reply(ws, timeout=open_timeout)
    except (ConnectionError, ConnectionClosed, TimeoutError, ValueError) as exc:
        ws.close()
        raise ConnectionError(
            f"The sorcar daemon at {endpoint.url} did not complete the "
            f"local auth handshake: {exc}"
        ) from exc
    if not isinstance(reply, dict) or reply.get("type") != "auth_ok" or not reply.get("local"):
        ws.close()
        raise ConnectionError(
            f"The sorcar daemon at {endpoint.url} rejected the local token "
            f"(reply: {reply!r}); the endpoint file may be stale."
        )
    return ws


def send(ws: ClientConnection, text: str, *, timeout: float = 10.0) -> None:
    """Send *text* on *ws*, giving up after *timeout* seconds.

    The synchronous ``websockets`` client writes with no deadline, so a
    daemon that has stopped reading (frozen, or swapped out) would block
    the caller for good.  The write runs on a helper thread; when it has
    not finished in time the socket is closed underneath it, which
    unblocks the write and makes every later use of *ws* fail fast.

    Args:
        ws: The connection returned by :func:`connect`.
        text: The frame's text payload.
        timeout: Seconds to wait for the write to complete.

    Raises:
        ConnectionError: When the write times out.
        ConnectionClosed: When the daemon closed the connection.
    """
    failure: list[BaseException] = []

    def _write() -> None:
        try:
            ws.send(text)
        except BaseException as exc:  # re-raised on the caller's thread
            failure.append(exc)

    worker = threading.Thread(target=_write, name="sorcar-local-send", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        ws.close_socket()
        raise ConnectionError(
            f"The sorcar daemon did not accept a {len(text)}-byte command "
            f"within {timeout:g} seconds"
        )
    if failure:
        raise failure[0]


def recv_auth_reply(ws: ClientConnection, *, timeout: float) -> object:
    """Return the daemon's answer to the ``auth`` frame just sent on *ws*.

    The daemon's watchdog pushes a ``{"type": "heartbeat"}`` frame at
    every open socket on each tick, and a tick that lands between the
    WebSocket handshake and the daemon's ``auth_ok`` queues a heartbeat
    ahead of it.  Those frames are skipped (as the browser transport
    does); the first other frame, decoded from JSON, is the answer.

    Args:
        ws: The connection the ``auth`` frame was sent on.
        timeout: Seconds allowed for the answer, heartbeats included.

    Raises:
        TimeoutError: When no answer arrives within *timeout* seconds.
        ConnectionClosed: When the daemon closed the connection.
        ValueError: When a frame is not JSON.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:  # a stream of heartbeats must not extend the budget
            raise TimeoutError(f"no answer to the auth frame within {timeout}s")
        reply = json.loads(ws.recv(timeout=remaining))
        if not (isinstance(reply, dict) and reply.get("type") == "heartbeat"):
            return reply
