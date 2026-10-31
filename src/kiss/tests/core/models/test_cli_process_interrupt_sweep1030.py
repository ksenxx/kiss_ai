# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a Stop landing right after the CLI child is spawned leaks nothing.

The daemon's Stop watchdog raises ``KeyboardInterrupt`` in the task
thread (``PyThreadState_SetAsyncExc``), and CPython delivers it at the
next check point of the bytecode.  ``_CLIProcess.__init__`` used to build
the stdout queue, the stderr list, the writer's ``Event`` and the stdin
``RLock`` after the child was already running and outside the ``try``
that kills it, so an interrupt delivered at any of those calls escaped
the constructor: the ``with`` block never started, ``close()`` never
ran, and the CLI child stayed alive behind its three pipes.

A profile hook places the interrupt exactly: once ``popen_process_group``
has returned the child, it injects ``KeyboardInterrupt`` at the k-th
Python call made from ``model.py`` code after that, for every k until the
constructor finishes without one, and each time the child must be dead
and its stdin closed.  The interrupt is raised as the callee starts, so it
never lands inside the stdlib's own lock handling (``threading.Condition``
is not safe against it, whoever calls it).
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
from types import FrameType
from typing import Any

from kiss.core import processes
from kiss.core.models import model as model_module
from kiss.core.models.model import _CLIProcess

_set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
_set_async_exc.argtypes = (ctypes.c_ulong, ctypes.py_object)

_CHILD = [sys.executable, "-c", "import time; time.sleep(60)"]
_MODEL_FILE = model_module.__file__


class _InterruptAfterSpawn:
    """Profile hook injecting ``KeyboardInterrupt`` at the k-th model call after the spawn."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.calls = 0
        self.child: subprocess.Popen[str] | None = None

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        """Record the spawned child, then count calls made from model.py and inject at the k-th."""
        if self.child is None:
            if event == "return" and frame.f_code is processes.popen_process_group.__code__:
                self.child = arg
            return
        caller = frame.f_back
        if event == "call" and caller is not None and caller.f_code.co_filename == _MODEL_FILE:
            self.calls += 1
            if self.calls == self.k:
                sys.setprofile(None)
                _set_async_exc(threading.get_ident(), ctypes.py_object(KeyboardInterrupt))


def _construct_with_interrupt(k: int) -> tuple[_CLIProcess | None, subprocess.Popen[str]]:
    """Build a ``_CLIProcess`` with an interrupt at the k-th call after the spawn."""
    hook = _InterruptAfterSpawn(k)
    proc: _CLIProcess | None = None
    sys.setprofile(hook)
    try:
        proc = _CLIProcess(_CHILD, "probe", 60)
    except KeyboardInterrupt:
        pass
    finally:
        sys.setprofile(None)
    assert hook.child is not None
    return proc, hook.child


def test_an_interrupt_at_any_call_after_the_spawn_kills_the_child() -> None:
    leaked: list[int] = []
    k = 0
    while True:
        k += 1
        proc, child = _construct_with_interrupt(k)
        if proc is not None:
            proc.close()
            break
        assert child.stdin is not None and child.stdout is not None and child.stderr is not None
        if child.poll() is None or not child.stdin.closed:
            leaked.append(k)
            child.kill()
            child.wait(timeout=30)
            for pipe in (child.stdin, child.stdout, child.stderr):
                pipe.close()
    assert k > 1, "the constructor made no call after the spawn"
    assert leaked == []
