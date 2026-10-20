# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Hold a daemon thread at the entry of a function, to land a race inside the window."""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import CodeType, FrameType
from typing import Any


class Pause:
    """Hold the thread that enters *code* with locals *matching* until :meth:`release`.

    A ``sys.settrace`` hook on every thread (``threading.settrace_all_threads``):
    an instrumentation of the real code path, nothing is replaced.  It
    widens the window between two steps of the daemon (a decision
    recorded and the command sent, a snapshot copied and delivered) so
    a second client's command lands inside it for sure.  ``paused`` is
    set once the thread is held.  A *matching* that returns False never
    pauses, and may act at the entry instead (run a real step of the
    daemon on the entering thread).
    """

    def __init__(self, code: CodeType, matching: Callable[[dict[str, Any]], bool]) -> None:
        self._code = code
        self._matching = matching
        self.paused = threading.Event()
        self._go = threading.Event()

    def __enter__(self) -> Pause:
        threading.settrace_all_threads(self._trace)
        return self

    def __exit__(self, *exc: object) -> None:
        threading.settrace_all_threads(None)  # type: ignore[arg-type]
        self.release()

    def release(self) -> None:
        """Let the held thread go on."""
        self._go.set()

    def _trace(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "call" or frame.f_code is not self._code or self.paused.is_set():
            return None
        if self._matching(dict(frame.f_locals)):
            self.paused.set()
            self._go.wait(60)
        return None


class Pauses:
    """Install several :class:`Pause` hooks at once (there is one settrace slot).

    Each hook holds its own thread; leaving the block releases them all.
    """

    def __init__(self, *pauses: Pause) -> None:
        self._pauses = pauses

    def __enter__(self) -> Pauses:
        threading.settrace_all_threads(self._trace)
        return self

    def __exit__(self, *exc: object) -> None:
        threading.settrace_all_threads(None)  # type: ignore[arg-type]
        for pause in self._pauses:
            pause.release()

    def _trace(self, frame: FrameType, event: str, arg: Any) -> None:
        for pause in self._pauses:
            pause._trace(frame, event, arg)
        return None
