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

import functools
import json
import logging
import os
import queue
import secrets
import socket
import ssl
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
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
    parts: bool = False,
) -> ClientConnection:
    """Connect to the daemon named by *endpoint_file* as a local client.

    Reads the endpoint file, opens the TLS WebSocket (permessage-deflate
    declined: it saves nothing on loopback) and completes the ``auth``
    handshake with the local token.  The daemon is dialled directly,
    whatever proxy the environment names (``http_proxy``, ...): it is
    on this machine, where no proxy reaches it.  The returned
    connection is ready for catalog commands.  The dial is made on a
    helper thread (:func:`dial_off_thread`): a Stop of the calling
    task ends the wait within :data:`_STOP_WAKE_SECONDS` and never
    lands in the handshake.

    Args:
        endpoint_file: The endpoint file to read; ``None`` means
            :func:`default_endpoint_path`.
        open_timeout: Seconds allowed for the TCP/TLS/WebSocket opening
            handshake and for the ``auth_ok`` reply.
        max_size: Largest incoming frame accepted, in bytes; ``None``
            (the default) accepts frames of any size.
        parts: Whether the caller joins the parts of a big frame
            (:class:`FrameJoiner`): the daemon then sends one above
            :data:`PART_CHARS` as separate ``part`` frames
            (:func:`frame_parts`), else whole.

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
    return dial_off_thread(functools.partial(_dial, endpoint, open_timeout, max_size, parts))


def _dial(
    endpoint: LocalEndpoint, open_timeout: float, max_size: int | None, parts: bool,
) -> ClientConnection:
    """Open the WebSocket to *endpoint* and complete the local ``auth`` handshake.

    The blocking part of :func:`connect`, which documents the arguments
    and the ``ConnectionError`` raised.
    """
    try:
        ws = _ws_connect(
            endpoint.url,
            ssl=client_ssl_context(endpoint),
            open_timeout=open_timeout,
            close_timeout=2.0,
            compression=None,
            max_size=max_size,
            proxy=None,
        )
    except (OSError, InvalidHandshake, InvalidURI) as exc:
        raise ConnectionError(
            f"Cannot connect to the sorcar daemon at {endpoint.url}: {exc} "
            "— start it with `kiss-web`."
        ) from exc
    try:
        send(
            ws, json.dumps({"type": "auth", "token": endpoint.token, "parts": parts}),
            timeout=open_timeout,
        )
        reply = recv_auth_reply(ws, timeout=open_timeout)
    except (ConnectionError, ConnectionClosed, TimeoutError, ValueError) as exc:
        ws.close()
        raise ConnectionError(
            f"The sorcar daemon at {endpoint.url} did not complete the "
            f"local auth handshake: {exc}"
        ) from exc
    except BaseException:
        # Anything else (no thread left for ``send``'s writer): the
        # socket is nobody's until it is returned, so nothing else
        # would close it, and its threads would keep answering the
        # daemon's pings.
        ws.close()
        raise
    if not isinstance(reply, dict) or reply.get("type") != "auth_ok" or not reply.get("local"):
        ws.close()
        raise ConnectionError(
            f"The sorcar daemon at {endpoint.url} rejected the local token "
            f"(reply: {reply!r}); the endpoint file may be stale."
        )
    return ws


_STOP_WAKE_SECONDS = 0.2
"""Seconds between two wakes of a wait for a call on another thread (:func:`_await`)."""


def dial_off_thread(
    dial: Callable[[], ClientConnection], timeout: float | None = None,
) -> ClientConnection:
    """Return the connection *dial* opens, dialled on a thread that no Stop reaches.

    A task is stopped by raising ``KeyboardInterrupt`` asynchronously
    in its thread.  Raised inside ``websockets``' ``connect`` it skips
    that library's cleanup, which catches ``Exception`` only, and the
    socket stays open.  CPython 3.14 can also raise it right after a
    ``with lock:`` acquire returns, before the block that releases the
    lock is entered: landing in the receive of the ``auth`` reply, it
    left the frame assembler's mutex held, and the ``close()`` meant
    to clean up waited for that mutex for good.

    So *dial* runs on a helper thread, and the caller waits on a
    ``queue.SimpleQueue``, whose ``get`` takes no Python-level lock
    (:func:`_await`).  A caller unwound before it holds the connection
    leaves it to the helper thread, which closes it once *dial*
    returns it; one unwound after it told the helper thread it holds
    the connection hangs it up itself (:func:`hang_up`).

    Args:
        dial: Opens and returns a connection, bounded by its own
            timeouts.
        timeout: Seconds to wait for *dial* at most, for one whose own
            timeouts leave something out (the lookup of a host name
            takes none); ``None`` waits until it ends.

    Returns:
        The connection *dial* returned.

    Raises:
        TimeoutError: When *dial* has not ended after *timeout*
            seconds.  The helper thread closes what it opens later.
        BaseException: Whatever *dial* raised.
    """
    outcome: queue.SimpleQueue[ClientConnection | BaseException] = queue.SimpleQueue()
    taken: queue.SimpleQueue[bool] = queue.SimpleQueue()
    result: ClientConnection | BaseException | None = None
    try:
        threading.Thread(
            target=_hand_over, args=(dial, outcome, taken),
            name="sorcar-local-connect", daemon=True,
        ).start()
        result = _await(outcome, timeout)
        if isinstance(result, BaseException):
            raise result
        # The answer and the return stay inside the ``try``: a Stop is
        # raised at the check that follows a call, here the ``put``'s,
        # and none follows a ``return``.
        taken.put(True)
        return result
    except BaseException:
        taken.put(False)
        if isinstance(result, ClientConnection):
            hang_up(result)
        raise


def hang_up(ws: ClientConnection) -> None:
    """Shut *ws*'s connection down: no closing handshake, none of the socket's own locks.

    From any thread, whatever state the socket is in.  A read in
    flight on the socket ends with ``ConnectionClosed``; the socket's
    receive thread reads the end of the stream and closes the socket.
    """
    with suppress(OSError):  # closed already
        ws.socket.shutdown(socket.SHUT_RDWR)


def call_off_thread[T](call: Callable[[], T], timeout: float) -> T:
    """Return ``call()``, run on a thread of its own and given up on after *timeout* seconds.

    For a call that no timeout bounds and no Stop interrupts, a name
    lookup above all: the resolver takes no timeout, comes before any
    socket one could run on, and holds a Stop of the calling task until
    it has given up.  The caller waits as :func:`dial_off_thread`'s
    does; a call that overruns ends on its daemon thread, its outcome
    dropped.

    Raises:
        TimeoutError: When *call* has not ended after *timeout* seconds.
        RuntimeError: When the thread cannot be started (out of threads).
        BaseException: Whatever *call* raised.
    """
    outcome: queue.SimpleQueue[T | BaseException] = queue.SimpleQueue()
    threading.Thread(
        target=_settle, args=(call, outcome), name="sorcar-off-thread", daemon=True,
    ).start()
    result = _await(outcome, timeout)
    if isinstance(result, BaseException):
        raise result
    return result


def _settle[T](call: Callable[[], T], outcome: queue.SimpleQueue[T | BaseException]) -> None:
    """Queue what *call* returns or raises on *outcome*: thread body of :func:`call_off_thread`."""
    try:
        outcome.put(call())
    except BaseException as exc:  # re-raised on the caller's thread
        outcome.put(exc)


def _await[T](outcome: queue.SimpleQueue[T], timeout: float | None = None) -> T:
    """Return the next item of *outcome*, waking every :data:`_STOP_WAKE_SECONDS`.

    A wake is where a Stop injected into the waiting thread is raised,
    without waiting for the daemon's answer.  The loop has a frame of
    its own so that the handler of :func:`dial_off_thread` does not
    depend on how the compiler covers loop jumps: CPython 3.13 leaves
    the back edge of a ``while <condition>:`` loop outside the
    exception table of a ``try`` around it, and a Stop raised there
    skipped that handler.  Raised here, it reaches the handler through
    the call.

    Raises:
        TimeoutError: When *outcome* is still empty after *timeout*
            seconds (``None``: no limit).
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        wait = _STOP_WAKE_SECONDS
        if deadline is not None:
            wait = max(0.0, min(wait, deadline - time.monotonic()))
        try:
            return outcome.get(timeout=wait)
        except queue.Empty:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("timed out") from None


def _hand_over(
    dial: Callable[[], ClientConnection],
    outcome: queue.SimpleQueue[ClientConnection | BaseException],
    taken: queue.SimpleQueue[bool],
) -> None:
    """Queue what *dial* returns or raises on *outcome*; close a connection nobody took.

    Thread body of :func:`dial_off_thread`, whose caller answers a
    queued connection on *taken*: ``True`` when it holds the
    connection, ``False`` when it was unwound first.
    """
    try:
        ws = dial()
    except BaseException as exc:  # re-raised on the caller's thread
        outcome.put(exc)
        return
    outcome.put(ws)
    if not taken.get():
        ws.close()


def send(ws: ClientConnection, text: str, *, timeout: float = 10.0) -> None:
    """Send *text* on *ws*, giving up after *timeout* seconds.

    The synchronous ``websockets`` client writes with no deadline, so a
    daemon that has stopped reading (frozen, or swapped out) would block
    the caller for good.  The write runs on a helper thread; when it has
    not ended in time the socket is closed underneath it, which ends
    the write and makes every later use of *ws* fail fast.  A write
    that ended whole as the deadline passed left the frame with the
    daemon, and the call returns (on a socket closed by then, which the
    caller's next read reports); one that failed, or has still not
    ended a second *timeout* after the close, is the timeout.
    An interrupt that lands while waiting (a Stop) closes the socket
    the same way before it propagates.

    Args:
        ws: The connection returned by :func:`connect`.
        text: The frame's text payload.
        timeout: Seconds to wait for the write to complete.

    Raises:
        ConnectionError: When the write timed out: the daemon does not
            have the frame.
        ConnectionClosed: When the daemon closed the connection.
    """
    ended: list[BaseException | None] = []
    worker = threading.Thread(
        target=_write, args=(ws, text, ended), name="sorcar-local-send", daemon=True,
    )
    try:
        worker.start()
        worker.join(timeout)
        late = not ended
        if late:
            # Judged by the write, not by its thread: one still alive
            # may have written the frame.  The close ends a blocked
            # write at once; the second wait is for a writer that has
            # yet to get a core, and finds the socket closed when it does.
            ws.close_socket()
            worker.join(timeout)
    except BaseException:
        # A Stop's KeyboardInterrupt from the moment the writer exists:
        # it holds the connection's protocol mutex while blocked, so
        # the caller's cleanup ``close()`` would wait for good.
        ws.close_socket()
        raise
    failure = ended[0] if ended else TimeoutError("the write has not ended")
    if failure is None:
        return
    if late:
        raise ConnectionError(
            f"The sorcar daemon did not accept a {len(text)}-byte command "
            f"within {timeout:g} seconds"
        ) from failure
    raise failure


def _write(ws: ClientConnection, text: str, ended: list[BaseException | None]) -> None:
    """Write *text* on *ws* and append to *ended* the error of the write, ``None`` when whole.

    Thread body of :func:`send`.
    """
    try:
        ws.send(text)
    except BaseException as exc:  # re-raised on the caller's thread
        ended.append(exc)
    else:
        ended.append(None)


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


PART_CHARS = 64 * 1024
"""Characters per part of a big frame (:func:`frame_parts`)."""


def frame_parts(text: str) -> list[str]:
    """Return the frames that carry the serialised frame *text* to a client.

    A frame of at most :data:`PART_CHARS` characters is sent as it is.
    A bigger one (a replayed transcript of megabytes) goes out as
    separate ``part`` frames of that many characters of its text each,
    numbered ``seq`` from 0, every one but the last flagged ``more``;
    the receiver joins them (:class:`FrameJoiner`, the webapp's
    ``transport.js``) and sees the frame whole.  Separate frames rather
    than one fragmented WebSocket message because a receiver sees a
    fragmented message only once it is complete: a browser on a slow
    network took minutes of silence for a dead socket and reconnected
    (its 45 s stale check) before a transcript could land, and the
    daemon's own send deadline could not tell a slow reader from a
    stopped one.  Each part is a sign of progress to both.

    Args:
        text: The frame, serialised with ``json.dumps``.
    """
    if len(text) <= PART_CHARS:
        return [text]
    parts = []
    for seq, start in enumerate(range(0, len(text), PART_CHARS)):
        part: dict[str, object] = {
            "type": "part", "seq": seq, "data": text[start:start + PART_CHARS],
        }
        if start + PART_CHARS < len(text):
            part["more"] = True
        parts.append(json.dumps(part))
    return parts


class FrameJoiner:
    """Join the parts of a big frame (:func:`frame_parts`) as a receiver reads them.

    One per connection: the daemon sends one endpoint's frames one
    after the other, so the parts of a frame are never interleaved
    with another frame's.  A part whose ``seq`` is not the next one
    expected (the start of the sequence was read and discarded before
    this joiner looked, as :meth:`MachineLink._drain_buffered` does)
    is dropped with what was gathered, until the next ``seq`` 0.
    """

    def __init__(self) -> None:
        """Start with no part gathered."""
        self._parts: list[str] = []

    def feed(self, msg: object, text: str = "") -> tuple[object, str] | None:
        """Return the frame *msg* (decoded from *text*) completes, or ``None``.

        An ordinary frame is returned as it came; a ``part`` is kept
        and ``None`` returned until the last part of its frame, which
        yields the joined frame (decoded, and its text).

        Args:
            msg: The decoded frame.
            text: Its wire form, for a caller that passes frames on
                as they came (``""`` when the text is not needed).

        Raises:
            ValueError: When the joined text is not JSON.
        """
        if not (isinstance(msg, dict) and msg.get("type") == "part"):
            return msg, text
        if msg.get("seq") != len(self._parts):
            self._parts.clear()
            if msg.get("seq") != 0:
                return None
        self._parts.append(str(msg.get("data") or ""))
        if msg.get("more") is True:
            return None
        whole = "".join(self._parts)
        self._parts.clear()
        return json.loads(whole), whole
