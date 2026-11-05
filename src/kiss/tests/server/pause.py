# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Hold a daemon thread at the entry of a function, to land a race inside the window."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable
from types import CodeType, FrameType
from typing import Any

# A ``sys.monitoring`` tool id of this helper's own: debuggers hold 0,
# coverage 1, profilers 2, so the hooks they installed stay untouched.
_TOOL = 3
_EVENTS = sys.monitoring.events
_ENTRY = (_EVENTS.PY_START, _EVENTS.PY_RESUME)  # a call, and a generator resumed


def _watch(codes: Iterable[CodeType], on_entry: Callable[[CodeType, int], None]) -> None:
    """Call *on_entry* on the thread that starts or resumes any of *codes*."""
    sys.monitoring.use_tool_id(_TOOL, "kiss.tests.server.pause")
    for event in _ENTRY:
        sys.monitoring.register_callback(_TOOL, event, on_entry)
    for code in codes:
        sys.monitoring.set_local_events(_TOOL, code, _EVENTS.PY_START | _EVENTS.PY_RESUME)


def _unwatch(codes: Iterable[CodeType]) -> None:
    """Undo :func:`_watch`: ``free_tool_id`` alone keeps the events and callbacks."""
    for code in codes:
        sys.monitoring.set_local_events(_TOOL, code, 0)
    for event in _ENTRY:
        sys.monitoring.register_callback(_TOOL, event, None)
    sys.monitoring.free_tool_id(_TOOL)


class Pause:
    """Hold the thread that enters *code* with locals *matching* until :meth:`release`.

    A ``sys.monitoring`` tool of its own, told of every start or resumption
    of *code* on any thread: an instrumentation of the real code path,
    nothing is replaced, and the ``sys.settrace`` hooks of a debugger or
    a coverage run keep tracing (the held thread included).  It widens
    the window between two steps of the daemon (a decision recorded and
    the command sent, a snapshot copied and delivered) so a second
    client's command lands inside it for sure.  ``paused`` is set once
    the thread is held.  A *matching* that returns False never pauses,
    and may act at the entry instead (run a real step of the daemon on
    the entering thread).
    """

    def __init__(self, code: CodeType, matching: Callable[[dict[str, Any]], bool]) -> None:
        self._code = code
        self._matching = matching
        self.paused = threading.Event()
        self._go = threading.Event()

    def __enter__(self) -> Pause:
        _watch([self._code], self._on_entry)
        return self

    def __exit__(self, *exc: object) -> None:
        _unwatch([self._code])
        self.release()

    def release(self) -> None:
        """Let the held thread go on."""
        self._go.set()

    def _on_entry(self, code: CodeType, offset: int) -> None:
        self._hold(sys._getframe(1))

    def _hold(self, frame: FrameType) -> None:
        if frame.f_code is not self._code or self.paused.is_set():
            return
        if self._matching(dict(frame.f_locals)):
            self.paused.set()
            self._go.wait(60)


class Pauses:
    """Install several :class:`Pause` hooks at once (the helper holds one tool id).

    Each hook holds its own thread; leaving the block releases them all.
    """

    def __init__(self, *pauses: Pause) -> None:
        self._pauses = pauses

    def __enter__(self) -> Pauses:
        _watch([pause._code for pause in self._pauses], self._on_entry)
        return self

    def __exit__(self, *exc: object) -> None:
        _unwatch([pause._code for pause in self._pauses])
        for pause in self._pauses:
            pause.release()

    def _on_entry(self, code: CodeType, offset: int) -> None:
        frame = sys._getframe(1)
        for pause in self._pauses:
            pause._hold(frame)
