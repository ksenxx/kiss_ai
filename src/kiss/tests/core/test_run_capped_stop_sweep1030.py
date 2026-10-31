# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing right after :func:`run_capped` takes its guard does not hang the task.

``run_capped`` runs on task threads (``commit_and_merge`` reaches it
through ``GitWorktreeOps.staged_diff``), which the runner's Stop and the
per-panel tool Stop interrupt by injecting an exception with
``PyThreadState_SetAsyncExc``.  CPython raises it at the periodic check
that ends the CALL of a ``with`` statement's ``__enter__``, outside the
block's cleanup region (gh-74174): ``__exit__`` is skipped and the lock
stays held.  ``CappedOutput._guard`` was a plain ``Lock`` taken with
``with`` on the caller's thread, and ``run_capped``'s ``except
BaseException`` handler takes it again, so the task thread blocked
forever on a lock it held itself: its tab kept the spinner and the
child was neither killed nor reaped.  The guard is now an ``RLock``
taken only through :func:`kiss.core.locked.locked_call`.

Each test parks the worker (a profile hook on that thread only) at the
call that takes the guard, holds the guard on the test thread the way
the timeout timer holds it around its kill, injects a real
``KeyboardInterrupt`` with ``PyThreadState_SetAsyncExc`` once the worker
is blocked in that ``acquire``, then releases it: the interrupt is
delivered exactly where the runner's Stop reaches a thread waiting on a
contended lock.  Before the fix the two ``run_capped`` tests hung (the
worker was still alive 10 s after the Stop) and the ``_capped_timeout``
test found the guard leaked.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from types import CodeType, FrameType
from typing import Any

from kiss.core.processes import (
    CappedOutput,
    _capped_timeout,
    kill_process_tree,
    popen_process_group,
    run_capped,
)
from kiss.tests.lock_injection import inject, lock_is_free


def _py(code: str) -> list[str]:
    """A bare interpreter (no site) running *code*."""
    return [sys.executable, "-I", "-S", "-c", code]


def _stack_local(frame: FrameType | None, name: str) -> Any:
    """The local *name* of the innermost frame on *frame*'s stack that binds it, else ``None``."""
    while frame is not None:
        value = frame.f_locals.get(name)
        if value is not None:
            return value
        frame = frame.f_back
    return None


class _GuardStop:
    """Profile hook parking the worker at its first ``acquire`` of a :class:`CappedOutput` guard.

    On that call it uninstalls itself, records the guard, the child and
    the code of the frame making the call, and waits for the test
    thread to take the guard and let it go on into the ``acquire``.
    """

    def __init__(self) -> None:
        self.parked = threading.Event()
        self.resume = threading.Event()
        self.guard: Any = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.caller: CodeType | None = None

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "c_call" or getattr(arg, "__name__", None) not in ("acquire", "__enter__"):
            return
        out = _stack_local(frame, "out")
        if not isinstance(out, CappedOutput) or getattr(arg, "__self__", None) is not out._guard:
            return
        sys.setprofile(None)
        self.guard = out._guard
        self.proc = _stack_local(frame, "proc")
        self.caller = frame.f_code
        self.parked.set()
        self.resume.wait(30)


class _Worker(threading.Thread):
    """Run *target* with *hook* profiling this thread only, keeping what it raised."""

    def __init__(self, target: Callable[[], object], hook: _GuardStop) -> None:
        super().__init__(daemon=True)
        self.target = target
        self.hook = hook
        self.raised: BaseException | None = None

    def run(self) -> None:
        sys.setprofile(self.hook)
        try:
            self.target()
        except BaseException as exc:  # the injected Stop is the expected outcome
            self.raised = exc
        finally:
            sys.setprofile(None)


def _wait_in_acquire(worker: threading.Thread, caller: CodeType) -> None:
    """Wait until *worker* left the hook: its innermost frame is the one calling ``acquire``.

    No check point lies between the hook's return and the ``acquire``,
    so an interrupt injected from here on is delivered when it returns.
    """
    deadline = time.monotonic() + 30
    while True:
        frame = sys._current_frames().get(worker.ident or 0)
        if frame is not None and frame.f_code is caller:
            return
        assert time.monotonic() < deadline, "the worker never reached the guard's acquire"
        time.sleep(0.001)


def _stop_at_guard(target: Callable[[], object]) -> tuple[_Worker, _GuardStop]:
    """Run *target* on a worker and Stop it the moment its first guard ``acquire`` returns.

    Fails when the worker is still running 10 s after the Stop.
    """
    stop = _GuardStop()
    worker = _Worker(target, stop)
    worker.start()
    assert stop.parked.wait(30), "the worker never took a CappedOutput guard"
    assert stop.caller is not None
    stop.guard.acquire()
    try:
        stop.resume.set()
        _wait_in_acquire(worker, stop.caller)
        assert inject(worker) == 1
    finally:
        stop.guard.release()
    worker.join(10)
    assert not worker.is_alive(), "the task thread hung on its own guard after the Stop"
    return worker, stop


def test_stop_at_the_finish_acquire_still_reaps_the_child() -> None:
    """First take of the guard: marking the result final once the child exited."""
    worker, stop = _stop_at_guard(partial(run_capped, _py("print('hi')"), limit=1000, timeout=60))
    assert isinstance(worker.raised, KeyboardInterrupt)
    assert stop.proc is not None and stop.proc.returncode == 0


def test_stop_at_the_cap_acquire_kills_and_reaps_the_child() -> None:
    """First take of the guard: labelling the output capped while the child still runs."""
    code = "import sys, time; sys.stdout.write('x' * 100); sys.stdout.flush(); time.sleep(60)"
    worker, stop = _stop_at_guard(partial(run_capped, _py(code), limit=10, timeout=60))
    assert isinstance(worker.raised, KeyboardInterrupt)
    # The child sleeps for a minute: an exit status now means the
    # except handler killed and reaped it.
    assert stop.proc is not None and stop.proc.returncode is not None


def test_stop_at_the_timeout_acquire_leaves_the_guard_free() -> None:
    """``_capped_timeout`` (the timer's, or the reader's once the deadline passed) gives it back."""
    proc = popen_process_group(_py("import time; time.sleep(60)"), stdout=subprocess.DEVNULL)
    try:
        out = CappedOutput()
        worker, _ = _stop_at_guard(partial(_capped_timeout, proc, out))
        assert isinstance(worker.raised, KeyboardInterrupt)
        assert lock_is_free(out._guard), "the Stop leaked the guard the reader and timer need"
    finally:
        kill_process_tree(proc)
        proc.wait(10)
