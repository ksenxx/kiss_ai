# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Aborting a streamed model response that has gone quiet.

Every provider SDK here streams over httpx, and a thread parked in
``recv()`` on a connection carrying no bytes cannot be rescued by the
agent: the cooperative stop flag is only read when the agent emits
something, and CPython delivers the ``KeyboardInterrupt`` that
``VSCodeServer._stop_task`` injects only at a bytecode boundary.  Task
``709ebce3`` was unstoppable for 178 seconds for exactly this reason
(``reports/stop_button_delay_2026-08-05.html``).

Closing the response is not enough either — that marks it closed while
the kernel-level read stays blocked.  Shutting the socket down is what
makes the blocked ``recv()`` return, so this module does that first and
then closes.
"""

import logging
import socket
import threading
import time
from collections.abc import Callable, Generator
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STREAM_STALL_TIMEOUT = 180.0


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
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.stalled = False
        self.stopped = False
        self._thread = threading.Thread(target=self._watch, name=name, daemon=True)
        self._thread.start()

    def beat(self) -> None:
        """Record that an event arrived (resets the stall clock)."""
        with self._lock:
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
        bounded: the abort runs outside the lock and the join has a
        timeout, so a transport that wedges during teardown delays the
        caller instead of stalling it forever.
        """
        with self._lock:
            self._done.set()
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
        poll = self._poll_interval()
        while not self._done.wait(poll):
            if not self._claim_abort():
                continue
            self._abort()
            return

    def _claim_abort(self) -> bool:
        """Decide whether to abort, atomically with respect to :meth:`stop`.

        The decision is taken under the lock that :meth:`stop` also holds
        while disarming, so the two cannot interleave: either this
        watchdog claims the abort before the caller disarms it (and
        ``stop`` then waits for the abort to finish), or the disarm wins
        and no abort ever happens.  The abort itself runs outside the
        lock, so tearing a stream down can never block :meth:`beat` or
        the disarming caller for as long as the transport takes.

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

    def _abort(self) -> None:
        """Unblock the iterating thread.

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
        """
        if self._shutdown_socket():
            return
        try:
            self._stream.close()
        except Exception:
            logger.debug("Exception caught", exc_info=True)

    def _shutdown_socket(self) -> bool:
        """Half-close the stream's TCP socket, ignoring any failure.

        Reaches the socket through httpcore's documented
        ``network_stream`` response extension.  Best-effort throughout: a
        transport without that extension (or a stream already torn down)
        just leaves the close fallback in :meth:`_abort` to do what it
        can.

        Returns:
            ``True`` when the socket was found and shut down.
        """
        try:
            extensions = self._stream.response.extensions
            network_stream = extensions.get("network_stream")
            sock = (
                network_stream.get_extra_info("socket")
                if network_stream is not None
                else None
            )
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
                return True
        except Exception:
            logger.debug("Exception caught", exc_info=True)
        return False


def _stop_requested(
    watchdog: StreamAbortWatchdog,
    stop_event: threading.Event | None,
) -> bool:
    """Return whether this request must unwind as a user stop.

    The watchdog only reports a stop it acted on, and it acts no earlier
    than its first poll: a stream that reaches EOF (or fails) in that
    window would otherwise look like an ordinary result even though the
    user had already pressed Stop.  Reading the event too covers that.

    Args:
        watchdog: The watchdog attached to the stream.
        stop_event: The requesting thread's stop event, or ``None``.

    Returns:
        ``True`` when a stop was requested for this request.
    """
    if watchdog.stopped:
        return True
    return stop_event is not None and stop_event.is_set()


def _stall_error(stall_timeout: float | None) -> TimeoutError:
    """Build the retryable error for a stream the watchdog aborted as stalled.

    Args:
        stall_timeout: The tolerated silence, for the message.

    Returns:
        A ``TimeoutError`` the agentic loop retries, rather than the
        silently truncated success an aborted socket would otherwise
        look like (the iterator just ends at EOF).
    """
    seconds = stall_timeout if stall_timeout is not None else 0.0
    return TimeoutError(
        f"Model stream stalled: no data received for {seconds:.0f}s "
        f"(model_config 'stream_stall_timeout'). The request was aborted "
        f"instead of hanging; it will be retried."
    )


stall_error = _stall_error
"""Public name for :func:`_stall_error`.

An adapter that runs its own :class:`StreamAbortWatchdog` loop instead of
:func:`stop_aware_events` (``AnthropicModel._create_message``) still has
to raise the same error for the same condition, so the wording lives
here once rather than being restated per transport.
"""


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
    on_abort: Callable[[], None] | None = None,
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
        on_abort: Called just before the stop is raised, so the caller
            can close whatever the abandoned loop body left open (an
            unterminated thinking block, typically).
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
    from kiss.core import stop_signal

    stop_event = stop_signal.get_thread_stop_event()
    watchdog = StreamAbortWatchdog(
        stream,
        stall_timeout=stall_timeout,
        stop_event=stop_event,
        name=name,
    )
    try:
        for event in stream:
            watchdog.beat()
            yield event
    except Exception:
        # The abort may surface as a transport error rather than as EOF.
        if _stop_requested(watchdog, stop_event):
            if on_abort is not None:
                on_abort()
            raise KeyboardInterrupt("Agent stop requested") from None
        if watchdog.stalled:
            if on_abort is not None:
                on_abort()
            raise _stall_error(stall_timeout) from None
        raise
    finally:
        watchdog.stop()
        _close_stream(stream)
    if _stop_requested(watchdog, stop_event):
        # An aborted socket usually just ends the iterator, so a stop has
        # to be reported after the loop as well.
        if on_abort is not None:
            on_abort()
        raise KeyboardInterrupt("Agent stop requested")
    if watchdog.stalled:
        # Same for a stall: without this the caller would keep whatever
        # partial text it accumulated and report it as a completion.
        if on_abort is not None:
            on_abort()
        raise _stall_error(stall_timeout)
