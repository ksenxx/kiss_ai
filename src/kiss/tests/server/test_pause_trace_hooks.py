# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A :class:`Pause` leaves the trace hooks of a debugger or a coverage run in place.

Real hooks (``sys.settrace`` on this thread, ``threading.settrace`` for
new threads), a real held thread.  The hooks must still be installed
after a normal and after an exceptional exit, and the thread held by
the pause must still have been traced by the pre-existing hook.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

import pytest

from kiss.tests.server.pause import Pause, Pauses

_traced: list[str] = []


def _step(n: int) -> int:
    return n + 1


def _main_hook(frame: Any, event: str, arg: Any) -> None:
    return None


def _thread_hook(frame: Any, event: str, arg: Any) -> None:
    if event == "call":
        _traced.append(f"{threading.current_thread().name}:{frame.f_code.co_name}")
    return None


def _n_is_one(frame_locals: dict[str, Any]) -> bool:
    return bool(frame_locals["n"] == 1)


def _never(_frame_locals: dict[str, Any]) -> bool:
    return False


def test_pause_keeps_pre_existing_trace_hooks() -> None:
    """Both hooks survive a pause that held a thread and one left by an exception."""
    before = sys.gettrace(), threading.gettrace()
    sys.settrace(_main_hook)
    threading.settrace(_thread_hook)
    try:
        with Pause(_step.__code__, _n_is_one) as pause:
            worker = threading.Thread(target=_step, args=(1,), name="held", daemon=True)
            worker.start()
            assert pause.paused.wait(10), "the worker never entered _step"
        worker.join(10)
        assert not worker.is_alive()
        assert sys.gettrace() is _main_hook
        assert threading.gettrace() is _thread_hook
        assert "held:_step" in _traced, "the pause displaced the hook on the held thread"
        with pytest.raises(RuntimeError), Pauses(Pause(_step.__code__, _never)):
            raise RuntimeError("the body failed")
        assert sys.gettrace() is _main_hook
        assert threading.gettrace() is _thread_hook
    finally:
        sys.settrace(before[0])
        threading.settrace(before[1])
