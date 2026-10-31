# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing anywhere in ``stop_aware_events``'s set-up leaves no watchdog running.

The runner's Stop raises ``KeyboardInterrupt`` in the task thread at the
next check point of the bytecode (``PyThreadState_SetAsyncExc``).  The
watchdog thread used to be started by ``StreamAbortWatchdog.__init__``,
before ``stop_aware_events`` entered the ``try`` whose ``finally``
disarms it, so an interrupt landing in between left a watchdog nobody
would stop: on a Gemini stream (no response yet) it polled forever and
then shut down the socket of the model's next request.  The thread is
now started inside the ``try``.

The interrupt is raised by a line tracer, the same delivery model as
``PyThreadState_SetAsyncExc``, at every line ``stop_aware_events`` runs
before it yields its first event, one run per line.
"""

from __future__ import annotations

import sys
import threading
import time
from types import FrameType
from typing import Any

from kiss.core.models.stream_abort import stop_aware_events

_NAME = "sweep1030-start-window-watchdog"
# Far beyond the test: an orphaned watchdog would still be polling.
_STALL = 600.0


class _Stream:
    """Two events from a stream that records whether it was closed."""

    def __init__(self) -> None:
        self._events = iter(["first", "second"])
        self.closed = False

    def __iter__(self) -> _Stream:
        return self

    def __next__(self) -> str:
        return next(self._events)

    def close(self) -> None:
        self.closed = True


class _StopAtLine:
    """Line tracer raising ``KeyboardInterrupt`` at line number *target* of ``stop_aware_events``.

    Lines are counted in execution order from 0, in the generator's own frame only.
    """

    def __init__(self, target: int) -> None:
        self.target = target
        self.lines = 0
        self.injected = False

    def __call__(self, frame: FrameType, event: str, arg: Any) -> _StopAtLine | None:
        if frame.f_code is not stop_aware_events.__code__:
            return None
        if event == "line" and not self.injected:
            if self.lines == self.target:
                self.injected = True
                raise KeyboardInterrupt("injected stop")
            self.lines += 1
        return self


def _watchdogs() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == _NAME]


def _no_watchdog_within(seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while _watchdogs():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _first_event_with_stop_at(target: int) -> tuple[_StopAtLine, _Stream, str | None]:
    """Ask a fresh wrapped stream for its first event, stopping it at line *target*."""
    stream = _Stream()
    events = stop_aware_events(stream, stall_timeout=_STALL, name=_NAME)
    tracer = _StopAtLine(target)
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        first: str | None = next(events)
    except KeyboardInterrupt:
        first = None
    finally:
        sys.settrace(previous)
    if first is not None:
        events.close()
    return tracer, stream, first


def test_a_stop_at_any_line_before_the_first_event_leaves_no_watchdog() -> None:
    """Wherever the Stop lands, the watchdog is never left running, and a started one is stopped."""
    assert not _watchdogs()
    for target in range(100):
        tracer, stream, first = _first_event_with_stop_at(target)
        assert _no_watchdog_within(5.0), f"a Stop at line {target} orphaned the watchdog"
        if not tracer.injected:
            # Every line up to the first ``yield`` was interrupted once.
            assert target > 3, "the sweep never reached the stream loop"
            assert first == "first"
            assert stream.closed
            return
        assert first is None
    raise AssertionError("stop_aware_events never yielded its first event")
