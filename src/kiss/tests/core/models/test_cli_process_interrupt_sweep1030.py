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
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

from kiss.core import processes
from kiss.core.models import model as model_module
from kiss.core.models.model import _CLIProcess
from kiss.tests.conftest import posix_only

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


# A child whose grandchild, in its own session, inherits stdout and so
# keeps the pipe open after the child's group is killed; the marker file
# names the grandchild once it exists.
_PIPE_HOLDER_CHILD = textwrap.dedent(
    """
    import pathlib, subprocess, sys, time
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    pathlib.Path(sys.argv[1]).write_text(str(grandchild.pid))
    time.sleep(60)
    """
)


@posix_only("the grandchild outlives the child's group in its own POSIX session")
def test_a_stop_inside_the_stdout_reader_start_does_not_hang_cleanup(tmp_path: Path) -> None:
    """A Stop landing in the stdout reader's ``start()`` leaves a live reader on its pipe.

    The reader is live from the spawn inside ``Thread.start()`` on, so a
    Stop delivered before ``start()`` returned used to reach
    ``_abort_start`` with the reader unrecorded: the pipe looked
    ownerless and was closed on the task thread, which blocks on the
    buffer lock the reader holds inside ``read(2)`` for as long as a
    grandchild keeps stdout open.  Cleanup must instead leave that pipe
    to the reader, as ``close()`` does, and let the Stop through.
    """
    marker = tmp_path / "grandchild.pid"
    child = [sys.executable, "-c", _PIPE_HOLDER_CHILD, str(marker)]
    outcome: dict[str, BaseException] = {}

    def stop_as_the_reader_start_returns(frame: FrameType, event: str, arg: Any) -> Any:
        """Raise the Stop as the first ``Thread.start()`` after the spawn is about to return."""
        if event == "return" and frame.f_code is threading.Thread.start.__code__:
            sys.settrace(None)
            deadline = time.monotonic() + 30
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists(), "the child never started its grandchild"
            raise KeyboardInterrupt("Stop inside the stdout reader's start()")
        return stop_as_the_reader_start_returns

    def construct() -> None:
        sys.settrace(stop_as_the_reader_start_returns)
        try:
            _CLIProcess(child, "probe", 60)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=construct, daemon=True, name="interrupted-constructor")
    worker.start()
    worker.join(timeout=30)
    returned_while_the_grandchild_held_stdout = not worker.is_alive()
    # Killing the grandchild sends the reader EOF, which also frees a cleanup
    # stuck on the pipe, so a failing run leaks neither process nor thread.
    os.kill(int(marker.read_text()), signal.SIGKILL)
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert returned_while_the_grandchild_held_stdout, "cleanup blocked closing the held pipe"
    assert isinstance(outcome.get("error"), KeyboardInterrupt)
