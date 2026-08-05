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
from collections.abc import Callable, Iterator
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
        """Stop the watchdog thread (stream finished or failed)."""
        self._done.set()

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
            if self._stop_event is not None and self._stop_event.is_set():
                self.stopped = True
                self._abort()
                return
            if self._stall_timeout is None:
                continue
            with self._lock:
                idle = time.monotonic() - self._last_event
            if idle >= self._stall_timeout:
                self.stalled = True
                self._abort()
                return

    def _abort(self) -> None:
        """Unblock the iterating thread, then close the stream."""
        self._shutdown_socket()
        try:
            self._stream.close()
        except Exception:
            logger.debug("Exception caught", exc_info=True)

    def _shutdown_socket(self) -> None:
        """Half-close the stream's TCP socket, ignoring any failure.

        Reaches the socket through httpcore's documented
        ``network_stream`` response extension.  Best-effort throughout: a
        transport without that extension (or a stream already torn down)
        just leaves the close above to do what it can.
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
        except Exception:
            logger.debug("Exception caught", exc_info=True)


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


def stop_aware_events(
    stream: Any,
    stall_timeout: float | None = None,
    on_abort: Callable[[], None] | None = None,
    name: str = "model-stream-abort-watchdog",
) -> Iterator[Any]:
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
        Each event the stream produces.

    Raises:
        KeyboardInterrupt: When the task was stopped mid-stream.
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
        raise
    finally:
        watchdog.stop()
    if _stop_requested(watchdog, stop_event):
        # An aborted socket usually just ends the iterator, so a stop has
        # to be reported after the loop as well.
        if on_abort is not None:
            on_abort()
        raise KeyboardInterrupt("Agent stop requested")
