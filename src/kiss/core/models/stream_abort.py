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

import logging
import os
import socket
import threading
import time
from collections.abc import Generator
from typing import Any

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
        """Start watching *stream*.

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
            # Joining is impossible during interpreter finalization,
            # which is when an abandoned stream's generator is collected.
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
        forgotten, leaving the task to wait for the read timeout.
        """
        poll = self._poll_interval()
        claimed = False
        while not self._done.wait(poll):
            if not claimed:
                claimed = self._claim_abort()
            if claimed and self._abort():
                return

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
    carrying the request is only reachable through httpx's transport and
    httpcore's pool, so this walks ``_transport._pool.connections`` and
    each connection's ``_connection._network_stream`` best-effort: the
    pool's ``connections`` and the stream's ``get_extra_info("socket")``
    are httpcore's documented API, the rest private attributes a future
    httpx may rename, in which case nothing is shut down and the caller
    falls back to abandoning the call until the client timeout.  Idle
    keep-alive sockets go down too, so the caller closes the client
    afterwards rather than reusing the pool.

    Args:
        http_client: The ``httpx.Client`` the SDK sends through.

    Returns:
        ``True`` when at least one socket was shut down.
    """
    try:
        connections = list(http_client._transport._pool.connections)
    except Exception:
        logger.debug("Exception caught", exc_info=True)
        return False
    shut_down = False
    for connection in connections:
        inner = getattr(connection, "_connection", None)
        if _shutdown_network_stream(getattr(inner, "_network_stream", None)):
            shut_down = True
    return shut_down


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


def stall_error(stall_timeout: float | None) -> TimeoutError:
    """Build the retryable error for a stream the watchdog aborted as stalled.

    Public so a transport whose byte-level clock fired first (an httpx
    read timeout, which the SDK may report with an empty message) raises
    the same error for the same condition; the wording lives here once.

    Args:
        stall_timeout: The tolerated silence, for the message.

    Returns:
        A ``TimeoutError`` the agentic loop retries, rather than the
        silently truncated success an aborted socket would otherwise
        look like (the iterator just ends at EOF).
    """
    seconds = stall_timeout if stall_timeout is not None else 0.0
    return TimeoutError(
        f"{STALL_MARKER}: no data received for {seconds:.0f}s "
        f"(model_config 'stream_stall_timeout'). The request was aborted "
        f"instead of hanging; it will be retried."
    )


#: Start of every :func:`stall_error` message; :func:`is_stall_error` keys on it.
STALL_MARKER = "Model stream stalled"


def is_stall_error(error: BaseException) -> bool:
    """Whether *error* is the :func:`stall_error` of an aborted stream.

    The agentic loop counts stalls separately from model errors (a stall
    is the transport's, not the model's), so it needs to tell them apart
    without a dedicated exception class that every transport would have
    to import.  A transport may prefix the message with its model name
    (``AnthropicModel._stall_error``), so the marker is searched, not
    matched at the start.

    Args:
        error: The exception the model call raised.
    """
    return isinstance(error, TimeoutError) and STALL_MARKER in str(error)


def stop_or_stall_error(stall_timeout: float | None) -> BaseException:
    """Classify a timeout raised BEFORE the response headers arrived.

    No watchdog exists yet in that window, so a Stop pressed while the
    request was still silent has to be read from the thread's stop
    signal by hand: reporting it as the retryable stall error would make
    the agentic loop re-ask the model on behalf of a task the user
    already stopped.

    Args:
        stall_timeout: The tolerated silence, for the stall message.

    Returns:
        :func:`stop_error` when the user stopped the task, else
        :func:`stall_error`.
    """
    if stop_signal.stop_requested():
        return stop_error()
    return stall_error(stall_timeout)


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
        for event in stream:
            watchdog.beat()
            yield event
    except Exception as exc:
        # The abort may surface as a transport error rather than as EOF;
        # it is classified below, once the watchdog is disarmed.
        failure = exc
    finally:
        watchdog.stop()
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
