# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Aborting a streamed model response that has gone quiet.

Every provider SDK here streams over httpx, and a thread parked in
``recv()`` on a connection carrying no bytes cannot be rescued by the
agent: the cooperative stop flag is only read when the agent emits
something, and CPython delivers the ``KeyboardInterrupt`` that
``AgentServer._stop_task`` injects only at a bytecode boundary.  Task
``709ebce3`` was unstoppable for 178 seconds for exactly this reason
(``reports/stop_button_delay_2026-08-05.html``).

Closing the response is not enough either — that marks it closed while
the kernel-level read stays blocked.  Shutting the socket down is what
makes the blocked ``recv()`` return, so this module does that first and
then closes.
"""

import contextlib
import logging
import os
import socket
import ssl
import sys
import threading
import time
import weakref
from collections.abc import Generator, Iterable
from typing import Any

import httpcore
import truststore
import truststore._api as truststore_api

from kiss.core import stop_signal
from kiss.core.locked import locked_call

logger = logging.getLogger(__name__)

DEFAULT_STREAM_STALL_TIMEOUT = 180.0

# Connect deadline of the per-request ``httpx.Timeout(stall, connect=...)``
# every transport puts on its streaming requests.  The watchdog below is
# armed only once the response headers have arrived, so before that the
# SDK's own clock is the only one: with a scalar client timeout (1800 s
# on the OpenAI clients) a gateway that accepted the TCP connection and
# never answered parked the agent in ``recv()`` for 30 minutes per
# attempt, deaf to Stop.  Headers that take longer than the stall
# timeout are, by the watchdog's own policy, a stall.
CONNECT_TIMEOUT = 10.0

# How long a read on a stream of an abortable pool waits in the kernel
# before it checks whether its socket was closed under it
# (``_AbortableStream.read``).
_READ_SLICE_SECONDS = 0.5

# How long a connect on an abortable pool waits for its connection
# before it checks whether the pool was aborted
# (``AbortableNetworkBackend.connect_tcp``).
_CONNECT_SLICE_SECONDS = 0.1


class StreamAbortWatchdog:
    """Aborts a streamed response on a user stop or on a stall.

    Two independent reasons to abort, either of which may be disabled:

    * **stop** — *stop_event* fires because the user pressed Stop.  The
      abort takes effect within :attr:`_STOP_POLL_SECONDS` instead of
      whenever the provider's timeout happens to expire (30 minutes on
      the OpenAI clients, 3 minutes on Anthropic).
    * **stall** — no stream event arrives for *stall_timeout* seconds.
      Pass ``None`` to leave a provider's existing timeout policy alone.
      This catches a wedged request that keeps the connection alive with
      keep-alive events the SDK filters out before yielding, which no
      byte-level read timeout can ever notice.

    The caller inspects :attr:`stopped` and :attr:`stalled` to decide
    what to raise, because an aborted socket usually ends the iterator at
    EOF rather than raising.
    """

    _STOP_POLL_SECONDS = 0.1
    _STOP_JOIN_SECONDS = 5.0

    def __init__(
        self,
        stream: Any,
        stall_timeout: float | None = None,
        stop_event: threading.Event | None = None,
        name: str = "model-stream-abort-watchdog",
    ) -> None:
        """Prepare to watch *stream*; :meth:`start` starts the watching thread.

        Args:
            stream: The SDK stream object.  Its ``close()`` and its
                ``response`` (an ``httpx.Response``) are used to abort.
            stall_timeout: Seconds of event-level silence tolerated, or
                ``None`` to not watch for stalls at all.
            stop_event: The requesting thread's stop event, or ``None``
                for callers outside a stoppable task.
            name: Thread name, so a stuck watchdog is identifiable in a
                stack dump.
        """
        self._stream = stream
        self._stall_timeout = stall_timeout
        self._stop_event = stop_event
        self._last_event = time.monotonic()
        # Guards the stop/abort handshake (``stop`` vs ``_claim_abort``).
        self._lock = threading.RLock()
        self._done = threading.Event()
        self.stalled = False
        self.stopped = False
        self._thread = threading.Thread(target=self._watch, name=name, daemon=True)

    def start(self) -> None:
        """Start the watching thread.

        Not part of construction: ``stop_aware_events`` starts the thread
        inside the ``try`` whose ``finally`` calls :meth:`stop`.  A thread
        started by ``__init__`` would be running before that ``try``, and
        a Stop injected in between would leave it with nobody to disarm it.
        """
        self._thread.start()

    def beat(self) -> None:
        """Record that an event arrived (resets the stall clock)."""
        # Unlocked on purpose: this runs once per streamed token and a
        # float store is atomic.  The single read of ``_last_event`` in
        # ``_claim_abort`` is the stall decision's linearization point:
        # a beat stored before that read defers the stall, and a beat
        # stored after it changes nothing, because at the instant of the
        # read the stream had already been silent for the full timeout,
        # so the (irrevocable) stall verdict was correct when taken.
        self._last_event = time.monotonic()

    def stop(self) -> None:
        """Disarm the watchdog and wait for its thread to finish.

        Once this returns, no further abort can happen.  That guarantee
        matters because the caller reaches here only after the stream is
        done, by which point the SDK may already have returned the
        keep-alive connection to httpx's pool: a late
        ``shutdown(SHUT_RDWR)`` would then poison a socket belonging to
        somebody else's next request.  Disarming under the same lock
        :meth:`_claim_abort` uses closes that window, and the join
        closes the "already decided, about to shut down" one.  Both are
        bounded: only the socket shutdown (a syscall) runs under the lock,
        the close fallback runs outside it and the join has a timeout, so
        a transport that wedges during teardown delays the caller instead
        of stalling it forever.
        """
        locked_call(self._lock, self._done.set)
        try:
            self._thread.join(timeout=self._STOP_JOIN_SECONDS)
        except RuntimeError:
            # Joining is impossible before the thread started (a Stop
            # landed before or inside start(); the set ``_done`` ends a
            # thread starting now at once) and during interpreter
            # finalization, when an abandoned stream's generator is
            # collected.
            logger.debug("Exception caught", exc_info=True)

    def _poll_interval(self) -> float:
        """Return how long to sleep between checks."""
        if self._stall_timeout is None:
            return self._STOP_POLL_SECONDS
        interval = max(0.05, min(1.0, self._stall_timeout / 4))
        if self._stop_event is not None:
            interval = min(interval, self._STOP_POLL_SECONDS)
        return interval

    def _watch(self) -> None:
        """Poll until an abort is claimed and carried out, or :meth:`stop` disarms.

        A claimed abort is retried every poll while the stream has no
        response yet: Gemini arms the watchdog before its lazy generator
        issues the request, so a Stop pressed in that window would
        otherwise be acted on once (with nothing to shut down) and then
        forgotten, leaving the task to wait for the read timeout.  A
        claim still pending when :meth:`stop` disarms the watchdog is
        withdrawn, as :meth:`_abort` withdraws one it finds disarmed:
        the stream ended on its own (headers and body all arrived
        within one poll), so the caller has a complete result, not a
        stall; a user stop still surfaces through its event.
        """
        poll = self._poll_interval()
        claimed = False
        while not self._done.wait(poll):
            if not claimed:
                claimed = self._claim_abort()
            if claimed and self._abort():
                return
        if claimed:
            self.stalled = self.stopped = False

    def _claim_abort(self) -> bool:
        """Decide whether to abort, atomically with respect to :meth:`stop`.

        The decision is taken under the lock that :meth:`stop` also holds
        while disarming, so the two cannot interleave: either this
        watchdog claims the abort before the caller disarms it (and
        ``stop`` then waits for the abort to finish), or the disarm wins
        and no abort ever happens (:meth:`_abort` re-checks the disarm
        before touching the socket).  The close fallback runs outside the
        lock, so tearing a stream down can never block the disarming
        caller for as long as the transport takes.

        Returns:
            ``True`` when this watchdog must now abort the stream.
        """
        with self._lock:
            if self._done.is_set():
                return False
            if self._stop_event is not None and self._stop_event.is_set():
                self.stopped = True
                return True
            if self._stall_timeout is not None and (
                time.monotonic() - self._last_event >= self._stall_timeout
            ):
                self.stalled = True
                return True
            return False

    def _abort(self) -> bool:
        """Unblock the iterating thread.

        Returns:
            ``False`` when the stream knows its ``response`` but has none
            yet (the request is still waiting for headers), so the caller
            retries later; closing such a stream would only close a
            generator that is executing on the task thread, which
            raises and achieves nothing.  ``True`` once the abort ran.

        A successful ``shutdown(SHUT_RDWR)`` is the whole abort: it
        wakes a reader already blocked in ``poll()``/``recv()`` AND
        makes every later read on the still-open descriptor return EOF
        immediately, so the reading thread always unwinds and closes
        the stream itself (``stop_aware_events``'s ``finally``, or the
        adapter's ``with`` block).  ``close()`` must NOT run here as
        well: it deallocates the file descriptor from under a reader
        that is just entering its blocking read, and a ``poll()`` on a
        freed — and possibly already reused — descriptor never sees the
        EOF, leaving the thread wedged for the SDK client's full read
        timeout exactly as if the watchdog did not exist.  Closing is
        therefore only the fallback for transports whose socket the
        shutdown could not reach.

        The claim was taken under the lock, but the stream may reach EOF
        and the caller disarm the watchdog before the abort gets here, by
        which time the SDK may have returned the socket to the pool.  So
        the shutdown (a non-blocking syscall) re-checks the disarm under
        the same lock and, when the stream ended on its own, withdraws
        the verdict instead: the caller has a complete result.  Only the
        close fallback, which may block, stays outside the lock.
        """
        if hasattr(self._stream, "response") and self._stream.response is None:
            return False
        with self._lock:
            if self._done.is_set():
                self.stalled = self.stopped = False
                return True
            shut_down = self._shutdown_socket()
        if not shut_down:
            _close_stream(self._stream)
        return True

    def _shutdown_socket(self) -> bool:
        """Half-close the stream's TCP socket, ignoring any failure.

        Reaches the socket through httpcore's documented
        ``network_stream`` response extension.  Best-effort throughout: a
        transport without that extension (or a stream already torn down)
        just leaves the close fallback in :meth:`_abort` to do what it
        can.

        On Windows the socket is closed as well: Winsock's ``shutdown``
        does not wake a ``recv()`` already blocked in another thread
        (measured: the reader stays parked until the SDK's read timeout),
        whereas ``closesocket`` makes it return ``WSAENOTSOCK`` at once.
        The descriptor-reuse hazard that keeps ``close()`` out of the
        POSIX path does not arise there: the woken reader gets an error
        instead of polling a recycled descriptor.

        Returns:
            ``True`` when the socket was found and shut down.
        """
        try:
            network_stream = self._stream.response.extensions.get("network_stream")
        except Exception:
            logger.debug("Exception caught", exc_info=True)
            return False
        return _shutdown_network_stream(network_stream)


def _shutdown_network_stream(network_stream: Any) -> bool:
    """Half-close the socket behind an httpcore network stream, ignoring any failure.

    Args:
        network_stream: An httpcore ``NetworkStream`` (or ``None``); its
            documented ``get_extra_info("socket")`` yields the socket.

    Returns:
        ``True`` when the socket was found and shut down.
    """
    try:
        sock = network_stream.get_extra_info("socket") if network_stream is not None else None
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
            if os.name == "nt":  # pragma: no cover — Windows-only branch
                sock.close()
            return True
    except Exception:
        logger.debug("Exception caught", exc_info=True)
    return False


def shutdown_pool_sockets(http_client: Any) -> bool:
    """Shut down every socket in an httpx client's connection pool.

    The abort for a request that has no response yet: a non-streamed
    call waits for the whole reply before the SDK hands anything back,
    so there is no ``response`` whose socket a :class:`StreamAbortWatchdog`
    could reach, and (as the module docstring explains) closing the
    client does not wake a thread parked in ``recv()``.  The one socket
    carrying the request is only reachable through httpx's transports and
    httpcore's pools (:func:`_client_pools`: the default one and every
    proxy mount's).  A pool given an :class:`AbortableNetworkBackend`
    (:func:`make_pool_abortable`, as the pool of every model adapter's
    client is) is aborted until its backend is re-armed
    (:meth:`AbortableNetworkBackend.rearm`, which only a
    :class:`RequestAbortWatchdog` does, once its request is over): the
    backend kept every stream the pool's connections hold, so its abort
    shuts them all down, and a TCP connection still being set up has no
    stream in the pool yet: its request fails at the abort, and a later
    one (an SDK retry's) is refused.  Any other pool is
    walked best-effort, each of its ``connections`` and each
    connection's ``_connection._network_stream``: the pool's
    ``connections`` and the stream's ``get_extra_info("socket")`` are
    httpcore's documented API, the rest private attributes a future
    httpx may rename, in which case nothing is shut down and the caller
    falls back to abandoning the call until the client timeout.  Idle
    keep-alive sockets go down too, so the caller closes the client
    afterwards rather than reusing the pool.

    Args:
        http_client: The ``httpx.Client`` the SDK sends through.

    Returns:
        ``True`` when at least one socket was shut down.
    """
    shut_down = False
    for pool in _client_pools(http_client):
        backend = getattr(pool, "_network_backend", None)
        if isinstance(backend, AbortableNetworkBackend):
            if backend.abort():
                shut_down = True
            continue
        try:
            connections = list(pool.connections)
        except Exception:
            logger.debug("Exception caught", exc_info=True)
            continue
        streams = [
            getattr(getattr(connection, "_connection", None), "_network_stream", None)
            for connection in connections
        ]
        if _shutdown_network_streams(streams):
            shut_down = True
    return shut_down


class RequestAbortWatchdog:
    """Aborts a non-streamed request on a user stop or at an absolute deadline.

    The counterpart of :class:`StreamAbortWatchdog` for a request whose
    reply the SDK hands back whole (``messages.create`` without
    streaming): until the reply has arrived there is no response whose
    socket could be shut down, and the SDK's ``timeout`` only bounds the
    silence between two reads, so a peer that sends a byte before each
    read timeout holds the calling thread for good.  Used as a context
    manager around the SDK call, this shuts the client's pool sockets
    down (:func:`shutdown_pool_sockets`, which wakes the parked read and
    fails the request at once) when the calling thread's stop event is
    set or *deadline* seconds have passed, and again at every poll until
    the block is left: a request whose connection is still being set up
    has no socket in the pool yet, and is cut the moment it has one (at
    the abort itself, in a pool with an :class:`AbortableNetworkBackend`).
    The idle connections that go down with the request's are discarded
    by httpx's pool as expired, so the client stays usable.  Nothing
    else may be sending through the client meanwhile: the abort reaches
    every connection of its pool.

    Leaving the block disarms the watchdog under the same lock the
    abort decision is taken under, so once it returns no abort can
    happen and the pool may carry the next request: a pool with an
    :class:`AbortableNetworkBackend`, which an abort leaves refusing
    every new connection (the aborted request's retries must not get
    through), is re-armed then (:meth:`AbortableNetworkBackend.rearm`),
    the request being over.

    A Stop injected into ``__exit__`` can skip that disarm.  The
    watchdog is then disarmed by its own thread once the thread that
    armed it is dead, or by the next watchdog that thread arms (it
    sends one watched request at a time): it never outlives the block
    for good, cutting a pool nobody asked it to.
    """

    _POLL_SECONDS = 0.1
    _JOIN_SECONDS = 5.0
    # ``watchdog``: a weak reference to the one this thread armed last.
    _last_armed = threading.local()

    def __init__(
        self, http_client: Any, deadline: float, name: str = "model-request-abort-watchdog"
    ) -> None:
        """Prepare to watch a request on *http_client*; entering the block starts the thread.

        Args:
            http_client: The ``httpx.Client`` the SDK sends through.
            deadline: Seconds from entering the block after which the
                request is aborted whatever arrives.
            name: Thread name, so a stuck watchdog is identifiable in a
                stack dump.
        """
        self._http_client = http_client
        self._deadline = time.monotonic() + deadline
        self._stop_event = stop_signal.get_thread_stop_event()
        self._owner = threading.current_thread()
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._aborted = False
        self._thread = threading.Thread(target=self._watch, name=name, daemon=True)

    def __enter__(self) -> "RequestAbortWatchdog":
        last = getattr(self._last_armed, "watchdog", None)
        stale = last() if last is not None else None
        if stale is not None and not stale._done.is_set():
            locked_call(stale._lock, stale._disarm)
        self._last_armed.watchdog = weakref.ref(self)
        try:
            self._thread.start()
        except BaseException:
            # ``__exit__`` never runs for an ``__enter__`` that raised,
            # and a Stop injected inside ``start()`` may land after the
            # thread was spawned, and after it aborted the pool: disarm
            # it on the way out, as ``__exit__`` does.
            locked_call(self._lock, self._disarm)
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        locked_call(self._lock, self._disarm)
        self._thread.join(self._JOIN_SECONDS)

    def _disarm(self) -> None:
        """End the watch; after an abort, let the pool connect again.  Called under the lock.

        ``_done`` is set last, so a disarm that a Stop cut short is not
        taken for a finished one and is done again.
        """
        if self._aborted:
            for pool in _client_pools(self._http_client):
                backend = getattr(pool, "_network_backend", None)
                if isinstance(backend, AbortableNetworkBackend):
                    backend.rearm()
        self._done.set()

    def _watch(self) -> None:
        """Poll until disarmed; abort at every poll once the stop or the deadline is due."""
        while not self._done.wait(self._POLL_SECONDS):
            with self._lock:
                if self._done.is_set():
                    return
                if not self._owner.is_alive():
                    self._disarm()
                    return
                stopped = self._stop_event is not None and self._stop_event.is_set()
                if stopped or time.monotonic() >= self._deadline:
                    self._aborted = True
                    shutdown_pool_sockets(self._http_client)


def _client_pools(http_client: Any) -> list[Any]:
    """Return the httpcore connection pools an httpx client may send through.

    The default transport's pool, and the pool of every mounted
    transport: a client built with proxies (``HTTP_PROXY`` /
    ``HTTPS_PROXY`` / ``ALL_PROXY`` from the environment, or explicit
    ``mounts``) sends each request whose URL a mount's pattern matches
    through that mount (``httpx.Client._transport_for_url``), never
    through the default pool.  Private attributes read best-effort: a
    transport without a ``_pool`` (a custom or mock transport, or the
    ``None`` a ``NO_PROXY`` pattern maps to) is skipped.

    Args:
        http_client: The ``httpx.Client`` the SDK sends through.

    Returns:
        The distinct pools, the default transport's first (those found
        before an unexpected attribute error, which is logged).
    """
    pools: list[Any] = []
    try:
        transports = [getattr(http_client, "_transport", None)]
        mounts = getattr(http_client, "_mounts", None)
        if isinstance(mounts, dict):
            transports.extend(mounts.values())
        for transport in transports:
            pool = getattr(transport, "_pool", None)
            if pool is not None and all(pool is not seen for seen in pools):
                pools.append(pool)
    except Exception:
        logger.debug("Exception caught", exc_info=True)
    return pools


#: Message of the ``ConnectError`` an aborted backend fails a connect with.
_ABORTED_CONNECT = "the request was aborted before its connection was up"


def _shutdown_network_streams(streams: Iterable[Any]) -> bool:
    """Shut down the socket behind each of *streams*; return whether any was shut down."""
    shut_down = False
    for stream in streams:
        if _shutdown_network_stream(stream):
            shut_down = True
    return shut_down


class _Connecting:
    """A TCP connection the wrapped backend sets up on a thread of its own.

    ``socket.create_connection`` looks the host up before any socket
    exists, and ``getaddrinfo`` takes no timeout: while a resolver did
    not answer, an abort had no socket to shut down and the connect
    timeout no socket to run on, so the request waited for the resolver
    whatever its deadline or its Stop.  The request's thread therefore
    waits for this one, and only until an abort
    (:meth:`AbortableNetworkBackend.connect_tcp`).  A connection that
    comes up after its wait was given up is closed here: nobody else
    knows of it.
    """

    def __init__(self, backend: httpcore.NetworkBackend, args: tuple[Any, ...]) -> None:
        """Prepare ``backend.connect_tcp(*args)``; starting :attr:`thread` runs it."""
        self._backend = backend
        self._args = args
        self._abandoned = False
        # The stream, or the error that kept it from coming up; ``None`` until then.
        self.outcome: httpcore.NetworkStream | BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="model-connect", daemon=True)

    def _run(self) -> None:
        """Connect; close a connection whose wait was given up meanwhile."""
        try:
            self.outcome = self._backend.connect_tcp(*self._args)
        except BaseException as error:  # re-raised in the thread that waits for it
            self.outcome = error
        # Read after the outcome is stored, as abandon() reads the outcome
        # after it has set the flag: whichever runs second closes the stream.
        if self._abandoned:
            self._close()

    def abandon(self) -> None:
        """Give the wait up: close the connection if it is up, else as soon as it is."""
        self._abandoned = True
        self._close()

    def _close(self) -> None:
        """Close the connection, if it came up."""
        outcome = self.outcome
        if outcome is not None and not isinstance(outcome, BaseException):
            outcome.close()


class AbortableNetworkBackend(httpcore.NetworkBackend):
    """An httpcore network backend whose sockets one :meth:`abort` shuts down, until :meth:`rearm`.

    It wraps the backend of an httpx client's connection pool
    (:func:`make_pool_abortable`) and keeps every stream it opens, and
    every stream a TLS upgrade of one of them yields (the upgrade hands
    back a new socket object and detaches the plain one).  A request
    whose connection is still being set up when the user stops it has
    no stream in the pool to shut down: a TCP connection not up at the
    :meth:`abort` fails its request then and is closed when it arrives
    (:meth:`connect_tcp`), and one that is up is shut down before
    httpcore writes the request on it, so the provider never sees (nor
    bills) the stopped request and the thread sending it fails at once.
    And a read whose socket is closed under it ends within a slice
    (:meth:`_AbortableStream.read`) instead of at the client timeout.

    The wrapped backend is httpcore's or httpcore2's (the Anthropic SDK
    sends through httpx2): the two packages have one API and share no
    class, and a pool maps only the exceptions of its own package to its
    client's.  So the backend and its streams raise and catch those of
    the package the wrapped backend is of (:attr:`errors`); the
    annotations name httpcore's types for both.
    """

    def __init__(self, backend: httpcore.NetworkBackend) -> None:
        self._backend = backend
        #: The wrapped backend's package, for its exception classes.
        self.errors: Any = sys.modules[type(backend).__module__.partition(".")[0]]
        # Taken on the task thread (``track`` while it connects, ``abort``
        # when it stops a blocking request), so only through
        # ``locked_call``: a Stop injected there cannot leak it.
        self._lock = threading.RLock()
        self._streams: weakref.WeakSet[httpcore.NetworkStream] = weakref.WeakSet()
        self._aborted = False

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        """Open a TCP connection through the wrapped backend and keep its stream.

        The wrapped backend connects on a thread of its own
        (:class:`_Connecting`) and this call waits for it in slices,
        until :meth:`abort`: the address lookup and the connect block
        where nothing else ends them.  *timeout* stays the wrapped
        backend's, for each address it tries; the deadline of the whole
        connect is the aborter's.  An aborted backend connects nothing:
        the retries of the request it cut are refused here.

        Raises:
            ConnectError: The backend is aborted, or was before the
                connection was up.
        """
        # ``_aborted`` is read unlocked: a stale read costs one slice.
        if self._aborted:
            raise self.errors.ConnectError(_ABORTED_CONNECT)
        connecting = _Connecting(
            self._backend, (host, port, timeout, local_address, socket_options)
        )
        # From here on only this call can give the connection up, so it
        # does on every exit but the return, a Stop injected here included.
        try:
            connecting.thread.start()
            while connecting.thread.is_alive():
                if self._aborted:
                    raise self.errors.ConnectError(_ABORTED_CONNECT)
                connecting.thread.join(_CONNECT_SLICE_SECONDS)
            outcome = connecting.outcome
            if isinstance(outcome, BaseException):
                raise outcome
            assert outcome is not None  # the thread has ended: it stored one
            return self.track(outcome)
        except BaseException:
            connecting.abandon()
            raise

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        """Open a Unix-socket connection through the wrapped backend and keep its stream."""
        return self.track(self._backend.connect_unix_socket(path, timeout, socket_options))

    def sleep(self, seconds: float) -> None:
        """Sleep as the wrapped backend does (httpcore's connect retries)."""
        self._backend.sleep(seconds)

    def track(self, stream: httpcore.NetworkStream) -> httpcore.NetworkStream:
        """Keep *stream* for :meth:`abort`, or shut it down at once after one.

        Args:
            stream: A stream the wrapped backend (or a TLS upgrade) opened.

        Returns:
            *stream* wrapped so that its TLS upgrade is kept too.
        """
        if locked_call(self._lock, self._keep, stream):
            _shutdown_network_stream(stream)
        return _AbortableStream(self, stream)

    def _keep(self, stream: httpcore.NetworkStream) -> bool:
        """Keep *stream*; return whether :meth:`abort` already ran.  Called under the lock."""
        self._streams.add(stream)
        return self._aborted

    def abort(self) -> bool:
        """Shut down every socket opened so far, and from now on every new one.

        Returns:
            ``True`` when at least one socket was shut down.
        """
        return _shutdown_network_streams(locked_call(self._lock, self._abort_kept))

    def _abort_kept(self) -> list[httpcore.NetworkStream]:
        """Mark the backend aborted; return the streams kept so far.  Called under the lock."""
        self._aborted = True
        return list(self._streams)

    def rearm(self) -> None:
        """Keep the connections opened from now on again, the aborted request being over.

        The streams :meth:`abort` shut down stay down (httpx's pool
        discards them as expired), so only a request that starts after
        this call gets through: the aborted one's retries, inside its
        :class:`RequestAbortWatchdog` block, were all cut.
        """
        locked_call(self._lock, setattr, self, "_aborted", False)


def _wrap_before_handshake(
    ssl_context: ssl.SSLContext, sock: socket.socket, server_hostname: str | None
) -> ssl.SSLSocket:
    """Wrap *sock* for TLS without starting the handshake.

    A ``truststore.SSLContext`` (the default context of httpx2, the
    Anthropic SDK's HTTP stack) checks the peer's certificates against
    the OS trust store inside its ``wrap_socket``, before a handshake
    deferred by ``do_handshake_on_connect=False`` has received any:
    macOS and Windows then refuse the empty chain and no connection is
    ever made.  Its inner OpenSSL context is wrapped here the way its own
    ``wrap_socket`` does it, and the caller verifies after the handshake.
    """
    if not isinstance(ssl_context, truststore.SSLContext):
        return ssl_context.wrap_socket(
            sock, server_hostname=server_hostname, do_handshake_on_connect=False
        )
    inner: ssl.SSLContext = ssl_context._ctx
    with contextlib.ExitStack() as configured:
        with ssl_context._ctx_lock:
            configured.enter_context(
                truststore_api._configure_context(inner)  # pyright: ignore[reportPrivateImportUsage]
            )
        return inner.wrap_socket(
            sock, server_hostname=server_hostname, do_handshake_on_connect=False
        )


class _AbortableStream(httpcore.NetworkStream):
    """A stream of an :class:`AbortableNetworkBackend`: the wrapped stream, TLS upgrade kept."""

    def __init__(self, backend: AbortableNetworkBackend, stream: httpcore.NetworkStream) -> None:
        self._backend = backend
        self._stream = stream

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        """Read from the wrapped stream, in slices that notice the socket closed under it.

        Closing a socket wakes no thread already waiting on it, and the
        shutdown of an abort that lands just as the read enters the
        kernel's wait can be missed too (seen on macOS under load): the
        read would then wait out *timeout*, 30 minutes on the OpenAI
        clients.  So it waits at most :data:`_READ_SLICE_SECONDS` at a
        time and fails once its socket is closed (the stopped client is
        closed right after its sockets are shut down).  *timeout* still
        bounds the whole read.

        Raises:
            ReadTimeout: Nothing arrived within *timeout*.
            ReadError: The socket was closed during the read.
        """
        errors = self._backend.errors
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = _READ_SLICE_SECONDS
            if deadline is not None:
                wait = min(wait, deadline - time.monotonic())
                if wait <= 0:
                    raise errors.ReadTimeout("timed out")
            try:
                return self._stream.read(max_bytes, wait)
            except errors.ReadTimeout:
                sock = self._stream.get_extra_info("socket")
                if sock is not None and sock.fileno() == -1:
                    raise errors.ReadError("the socket was closed during the read") from None

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        """Write to the wrapped stream."""
        self._stream.write(buffer, timeout)

    def close(self) -> None:
        """Close the wrapped stream."""
        self._stream.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        """Upgrade the wrapped stream to TLS; the backend keeps the upgraded stream.

        The upgraded stream is kept before its handshake: ``wrap_socket``
        detaches the plain socket the backend kept, so an abort during
        the handshake (a Stop while a peer that accepted the connection
        never answers it) reached nothing, and the connect timeout
        bounded the Stop instead.  What httpcore's ``start_tls`` does,
        the handshake split off; a stream without a socket (a mock
        backend's) or already on TLS (TLS in TLS, through an HTTPS
        proxy) is upgraded by the wrapped stream as before.

        Raises:
            ConnectTimeout: The handshake outlasted *timeout*.
            ConnectError: The handshake failed (an aborted socket
                included).
        """
        errors = self._backend.errors
        sock = self._stream.get_extra_info("socket")
        if sock is None or isinstance(sock, ssl.SSLSocket):
            return self._backend.track(
                self._stream.start_tls(ssl_context, server_hostname, timeout)
            )
        # Until the stream is returned only this call can close the
        # upgraded socket (the plain one is detached), so it does on every
        # failure from the statement that binds it on, a Stop injected
        # there included: one ``try`` spans the upgrade and the return.
        tls: ssl.SSLSocket | None = None
        try:
            try:
                sock.settimeout(timeout)
                tls = _wrap_before_handshake(ssl_context, sock, server_hostname)
            except OSError as exc:
                self.close()
                raise errors.ConnectError(exc) from exc
            # The wrapped stream's class (a ``SyncStream``) takes the
            # socket; ``NetworkStream`` itself declares no constructor.
            stream_cls: Any = type(self._stream)
            stream = self._backend.track(stream_cls(tls))
            try:
                tls.do_handshake()
                if isinstance(ssl_context, truststore.SSLContext):
                    truststore_api._verify_peercerts(tls, server_hostname=server_hostname)
            except TimeoutError as exc:
                raise errors.ConnectTimeout(exc) from exc
            except (OSError, AttributeError) as exc:
                # AttributeError: the abort's ``SSLSocket.shutdown`` dropped
                # the SSL object from under the handshake it was cutting.
                raise errors.ConnectError(exc) from exc
            return stream
        except BaseException:
            if tls is not None:
                tls.close()
            raise

    def get_extra_info(self, info: str) -> Any:
        """Return the wrapped stream's extra info (its ``socket``, ``ssl_object``, ...)."""
        return self._stream.get_extra_info(info)


def make_pool_abortable(http_client: Any) -> None:
    """Give each of an httpx client's connection pools an :class:`AbortableNetworkBackend`.

    Every pool the client may send through (:func:`_client_pools`: the
    default transport's and each proxy mount's) gets one, so a Stop also
    reaches a proxied request whose connection is still being set up.
    Best-effort like :func:`shutdown_pool_sockets`, through the same
    private attributes (``_transport._pool``, ``_mounts`` and httpcore's
    ``_network_backend``): without them a pool keeps its backend and an
    abort reaches only the connections already in it.

    Args:
        http_client: The ``httpx.Client`` the SDK sends through.
    """
    for pool in _client_pools(http_client):
        try:
            if not isinstance(pool._network_backend, AbortableNetworkBackend):
                pool._network_backend = AbortableNetworkBackend(pool._network_backend)
        except Exception:
            logger.debug("Exception caught", exc_info=True)


def _abort_error(
    watchdog: StreamAbortWatchdog,
    stop_event: threading.Event | None,
    stall_timeout: float | None,
) -> BaseException | None:
    """Return the error a stopped or stalled stream must unwind with.

    The watchdog only reports a stop it acted on, and it acts no earlier
    than its first poll: a stream that reaches EOF (or fails) in that
    window would otherwise look like an ordinary result even though the
    user had already pressed Stop.  Reading the event too covers that.
    A stop wins over a stall.  Call only after :meth:`StreamAbortWatchdog.stop`,
    when the flags are final.

    Args:
        watchdog: The disarmed watchdog attached to the stream.
        stop_event: The requesting thread's stop event, or ``None``.
        stall_timeout: The tolerated silence, for the stall message.

    Returns:
        :func:`stop_error` for a user stop, :func:`stall_error` for a
        stall, else ``None`` (the stream ended on its own).
    """
    if watchdog.stopped or (stop_event is not None and stop_event.is_set()):
        return stop_error()
    if watchdog.stalled:
        return stall_error(stall_timeout)
    return None


def stop_error() -> KeyboardInterrupt:
    """Build the error for a request the user stopped.

    A user stop must NOT surface as the retryable :class:`TimeoutError`
    a stall produces, or the agentic loop would re-ask the model and the
    task would keep running.  ``KeyboardInterrupt`` is the same signal
    ``_check_stop`` raises, so the whole stack unwinds into the normal
    "Task stopped by user" path.  Every transport raises this one object
    so the wording lives here once.

    Returns:
        The ``KeyboardInterrupt`` for the caller to raise.
    """
    return KeyboardInterrupt("Agent stop requested")


def stall_error(stall_timeout: float | None, label: str = "") -> TimeoutError:
    """Build the retryable error for a stream the watchdog aborted as stalled.

    Public so a transport whose byte-level clock fired first (an httpx
    read timeout, which the SDK may report with an empty message) raises
    the same error for the same condition; the wording lives here once.

    Args:
        stall_timeout: The tolerated silence, for the message.
        label: What stalled (``"Anthropic model <name>"``), put in front
            of the message for a transport whose stalls are diagnosed
            from log lines that do not say which model was asked; empty
            for none.

    Returns:
        A ``TimeoutError`` the agentic loop retries, rather than the
        silently truncated success an aborted socket would otherwise
        look like (the iterator just ends at EOF).
    """
    seconds = stall_timeout if stall_timeout is not None else 0.0
    prefix = f"{label}: " if label else ""
    return TimeoutError(
        f"{prefix}{STALL_MARKER}: no data received for {seconds:.0f}s "
        f"(model_config 'stream_stall_timeout'). The request was aborted "
        f"instead of hanging; it will be retried."
    )


#: Start of every :func:`stall_error` message after its label; :func:`is_stall_error` keys on it.
STALL_MARKER = "Model stream stalled"


def is_stall_error(error: BaseException) -> bool:
    """Whether *error* is the :func:`stall_error` of an aborted stream.

    The agentic loop counts stalls separately from model errors (a stall
    is the transport's, not the model's), so it needs to tell them apart
    without a dedicated exception class that every transport would have
    to import.  A transport may prefix the message with its model name
    (the *label* of :func:`stall_error`), so the marker is searched, not
    matched at the start.

    Args:
        error: The exception the model call raised.
    """
    return isinstance(error, TimeoutError) and STALL_MARKER in str(error)


def stop_or_stall_error(stall_timeout: float | None, label: str = "") -> BaseException:
    """Classify a timeout raised BEFORE the response headers arrived.

    No watchdog exists yet in that window, so a Stop pressed while the
    request was still silent has to be read from the thread's stop
    signal by hand: reporting it as the retryable stall error would make
    the agentic loop re-ask the model on behalf of a task the user
    already stopped.

    Args:
        stall_timeout: The tolerated silence, for the stall message.
        label: Put in front of the stall message (:func:`stall_error`).

    Returns:
        :func:`stop_error` when the user stopped the task, else
        :func:`stall_error`.
    """
    if stop_signal.stop_requested():
        return stop_error()
    return stall_error(stall_timeout, label)


def _close_stream(stream: Any) -> None:
    """Close *stream*, ignoring transports that do not support it.

    Iterating an SDK stream to its end does not release the underlying
    HTTP response, so without this the connection is never returned to
    httpx's pool — every request would open a fresh socket, and a stream
    abandoned by an exception would strand one indefinitely.

    Args:
        stream: The SDK stream object.
    """
    try:
        stream.close()
    except Exception:
        logger.debug("Exception caught", exc_info=True)


def stop_aware_events(
    stream: Any,
    stall_timeout: float | None = None,
    name: str = "model-stream-abort-watchdog",
) -> Generator[Any]:
    """Yield *stream*'s events, aborting the request on a user stop.

    A drop-in wrapper for ``for event in stream``: the caller's loop body
    is unchanged, but the request is now torn down as soon as the user
    presses Stop instead of holding the agent until the provider's
    timeout expires.  The stop surfaces to the caller as
    ``KeyboardInterrupt`` — the same signal ``JsonPrinter._check_stop``
    raises — because a retryable error would make the agentic loop
    re-ask the model on behalf of a task the user already stopped.

    Args:
        stream: The SDK stream object to iterate and, if needed, abort.
        stall_timeout: Seconds of event-level silence to tolerate, or
            ``None`` to leave the provider's own timeout policy alone.
        name: Watchdog thread name, for readable stack dumps.

    Yields:
        Each event the stream produces.  The result is a generator, not a
        bare iterator: a caller whose loop body can raise must
        ``close()`` it so the watchdog is stopped and the stream released
        deterministically instead of whenever the traceback is dropped.

    Raises:
        KeyboardInterrupt: When the task was stopped mid-stream.
        TimeoutError: When the stream produced no event for
            *stall_timeout* seconds and was aborted.  Retryable, unlike
            the stop above, and raised in preference to returning the
            partial text the abandoned loop body accumulated.
    """
    stop_event = stop_signal.get_thread_stop_event()
    watchdog = StreamAbortWatchdog(
        stream,
        stall_timeout=stall_timeout,
        stop_event=stop_event,
        name=name,
    )
    failure: Exception | None = None
    try:
        watchdog.start()
        for event in stream:
            watchdog.beat()
            yield event
    except Exception as exc:
        # The abort may surface as a transport error rather than as EOF;
        # it is classified below, once the watchdog is disarmed.
        failure = exc
    finally:
        # Nested: a Stop injected into the disarm must not skip the close,
        # which nobody else does once this generator has ended.
        try:
            watchdog.stop()
        finally:
            _close_stream(stream)
    # Classify after disarming: stop() waits out an abort already
    # claimed, so the flags are final here.  Read first, a stall (or
    # stop) claimed between the transport failure and this point would
    # let the raw transport error escape unclassified.  An aborted socket
    # usually just ends the iterator, so the same check covers a normal
    # EOF; without it the caller would keep whatever partial text it
    # accumulated and report it as a completion.
    error = _abort_error(watchdog, stop_event, stall_timeout)
    if error is not None:
        raise error from None
    if failure is not None:
        raise failure
