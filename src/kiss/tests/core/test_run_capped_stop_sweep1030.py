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

import contextlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import pytest

from kiss.core.processes import (
    IS_WINDOWS,
    SIGKILL,
    CappedOutput,
    _capped_timeout,
    _wait_exited,
    kill_process_tree,
    pid_alive,
    popen_process_group,
    run_capped,
    run_process_group,
    start_process_group,
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
    """Run *target* with *hook* profiling this thread only, keeping what it raised.

    *log*, when given, traces this thread (``sys.settrace``): CPython
    uninstalls a profile function that raises, and the Stop injected
    while *hook* parks the thread is raised inside *hook*, so a logger
    that must go on observing calls after the Stop is installed apart.
    """

    def __init__(
        self,
        target: Callable[[], object],
        hook: Callable[[FrameType, str, Any], None],
        log: Callable[[FrameType, str, Any], None] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.target = target
        self.hook = hook
        self.log = log
        self.raised: BaseException | None = None

    def run(self) -> None:
        sys.setprofile(self.hook)
        if self.log is not None:
            sys.settrace(self.log)
        try:
            self.target()
        except BaseException as exc:  # the injected Stop is the expected outcome
            self.raised = exc
        finally:
            sys.setprofile(None)
            sys.settrace(None)


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


def _wait_in_pipe_select(worker: threading.Thread) -> subprocess.Popen[bytes]:
    """Wait until *worker* is in ``_read_some``'s readiness wait; the child it is reading."""
    deadline = time.monotonic() + 30
    while True:
        frame = sys._current_frames().get(worker.ident or 0)
        caller = frame.f_back if frame is not None else None
        if frame is not None and frame.f_code.co_name == "select" and caller is not None:
            if caller.f_code.co_name == "_read_some":
                proc = _stack_local(frame, "proc")
                assert isinstance(proc, subprocess.Popen)
                return proc
        assert time.monotonic() < deadline, "the worker never waited on the child's pipe"
        time.sleep(0.001)


@pytest.mark.skipif(sys.platform == "win32", reason="the pipe wait is a readiness wait on POSIX")
def test_stop_while_waiting_on_a_quiet_pipe_kills_and_reaps_the_child() -> None:
    """A Stop landing while the task thread waits on a silent child's pipe is acted on promptly.

    An injected exception is raised between bytecodes, never inside the
    blocking ``epoll``/``kqueue`` wait: a single wait for the whole
    remaining timeout kept the Stop pending — and the child running —
    until the child spoke or the command's timeout (five minutes for a
    staged diff) ran out.
    """
    raised: list[BaseException] = []

    def body() -> None:
        try:
            run_capped(_py("import time; time.sleep(60)"), limit=1000, timeout=60)
        except BaseException as exc:  # the injected Stop is the expected outcome
            raised.append(exc)

    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    proc = _wait_in_pipe_select(worker)
    try:
        assert inject(worker) == 1
        worker.join(5)
        assert not worker.is_alive(), "the Stop sat behind the pipe wait"
        assert isinstance(raised[0], KeyboardInterrupt)
        # The child sleeps for a minute: an exit status now means the
        # except handler killed and reaped it.
        assert proc.returncode is not None
    finally:
        kill_process_tree(proc)  # pipe EOF releases a wait the Stop could not
        worker.join(10)
        proc.wait(10)


class _ThreadStartStop:
    """Profile hook parking the worker as it enters *code* (``Thread.start``: its timer's).

    ``run_capped`` has started its child by then; the Stop is injected
    while the worker is parked and is raised the moment the hook
    returns, i.e. from inside ``timer.start()``.
    """

    def __init__(self, code: CodeType = threading.Thread.start.__code__) -> None:
        self.code = code
        self.parked = threading.Event()
        self.resume = threading.Event()
        self.proc: subprocess.Popen[bytes] | None = None

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "call" or frame.f_code is not self.code:
            return
        proc = _stack_local(frame, "proc")
        if not isinstance(proc, subprocess.Popen):
            return
        sys.setprofile(None)
        self.proc = proc
        self.parked.set()
        self.resume.wait(30)


class _GroupKillLog:
    """Trace or profile hook logging, per ``kill_process_tree`` call, whether the leader was reaped.

    ``proc.returncode`` at the call: ``None`` while the leader is
    unreaped (its pid, the group id, is still its own), an exit status
    once a ``wait`` reaped it (the group id may be a stranger's by now,
    see :func:`kiss.core.processes.kill_process_group`).
    """

    def __init__(self) -> None:
        self.group_kills: list[int | None] = []

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is kill_process_tree.__code__:
            self.group_kills.append(frame.f_locals["proc"].returncode)


class _SpawnReturnStop:
    """Profile hook parking the worker as the function compiled to *code* returns.

    ``Popen.__init__`` by default: the child is running.  The Stop
    injected while the worker is parked is raised the moment the hook
    returns, i.e. as that call returns to its caller, before the caller
    binds its result to any name.
    """

    def __init__(self, code: CodeType = subprocess.Popen.__init__.__code__) -> None:
        self.code = code
        self.parked = threading.Event()
        self.resume = threading.Event()
        self.proc: subprocess.Popen[Any] | None = None

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "return" or frame.f_code is not self.code:
            return
        sys.setprofile(None)
        self.proc = _stack_local(frame, "proc")
        self.parked.set()
        self.resume.wait(30)


def _stop_while_parked(
    hook: _ThreadStartStop | _SpawnReturnStop,
    run: Callable[[], object],
    log: _GroupKillLog | None = None,
) -> tuple[_Worker, subprocess.Popen[Any]]:
    """Run *run* on a worker, Stop it where *hook* parks it, let it go on; the worker and child.

    *log* observes the worker's group kills throughout (see
    :class:`_Worker`).  Asserts that the Stop was raised and that the
    child is gone (:func:`_assert_stopped_child_is_gone`) before the
    cleanup in the ``finally``, which would otherwise reap a child the
    product left running and so hide the leak from that assertion.
    Fails when the worker is still running 10 s after the Stop; a child
    the Stop left running is killed and reaped before that failure.
    """
    worker = _Worker(run, hook, log)
    worker.start()
    assert hook.parked.wait(30), "the worker never reached the parking point"
    proc = hook.proc
    assert proc is not None
    try:
        assert inject(worker) == 1
        hook.resume.set()
        worker.join(10)
        assert not worker.is_alive(), "the task thread hung after the Stop"
        _assert_stopped_child_is_gone(worker, proc)
    finally:
        if proc.returncode is None:
            kill_process_tree(proc)
            worker.join(10)
            proc.wait(10)
    return worker, proc


def _assert_stopped_child_is_gone(worker: _Worker, proc: subprocess.Popen[Any]) -> None:
    """The Stop was raised, the child (a minute's sleeper) killed and reaped, its pipes closed.

    Left open, the pipes live as long as the ``Popen``, which the
    raised Stop's traceback keeps alive for as long as the caller
    keeps the error.  ``returncode`` is read as it stands: a ``poll()``
    would reap the child itself and so pass a product that did not.
    """
    assert isinstance(worker.raised, KeyboardInterrupt)
    assert proc.returncode is not None, "the child was left running after the Stop"
    assert proc.stdout is not None and proc.stdout.closed, "the child's stdout was left open"
    assert proc.stderr is None or proc.stderr.closed, "the child's stderr was left open"


@pytest.mark.parametrize("keep_stderr", [0, 100])
def test_stop_while_starting_the_timer_kills_and_reaps_the_child(keep_stderr: int) -> None:
    """A Stop landing between the spawn and the read loop still takes the child down.

    ``Thread.start`` blocks on the new thread's ``_started`` event, so
    an injected Stop is delivered right there; the setup ran before the
    ``try`` whose handler kills and reaps the child, which was left
    running (its whole 60 s) with its stdout pipe open.  With
    ``keep_stderr`` the first thread started is the stderr drain: a
    Stop raised at its start leaves it never run, and the handler,
    which left stderr to that drain, kept the pipe open.
    """
    _stop_while_parked(
        _ThreadStartStop(),
        partial(
            run_capped, _py("import time; time.sleep(60)"),
            limit=1000, timeout=60, keep_stderr=keep_stderr,
        ),
    )


@pytest.mark.parametrize("keep_stderr", [0, 100])
def test_stop_while_creating_the_timer_kills_and_reaps_the_child(keep_stderr: int) -> None:
    """A Stop landing at a call between the spawn and the ``try`` still takes the child down.

    The deadline, the drain's stop event and the timeout timer were set
    up after the spawn and before the ``try``: three calls, each a
    check point where an injected Stop is raised, with nothing to kill
    the child it then left running (its timer never started).  Those
    calls now run inside the ``try``, or before the spawn.  With
    ``keep_stderr`` the stderr pipe exists but its drain thread does
    not yet: the pipe is the handler's to close.
    """
    _stop_while_parked(
        _ThreadStartStop(threading.Timer.__init__.__code__),
        partial(
            run_capped, _py("import time; time.sleep(60)"),
            limit=1000, timeout=60, keep_stderr=keep_stderr,
        ),
    )


def _runners(cmd: list[str]) -> list[Any]:
    """Both runners of *cmd* (stderr kept, so ``run_capped`` opens both pipes), as test params."""
    return [
        pytest.param(partial(run_process_group, cmd, timeout=60), id="run_process_group"),
        pytest.param(
            partial(run_capped, cmd, limit=1000, timeout=60, keep_stderr=100), id="run_capped"
        ),
    ]


@pytest.mark.parametrize("run", _runners(_py("import time; time.sleep(60)")))
def test_stop_landing_as_the_spawn_returns_kills_and_reaps_the_child(
    run: partial[Any],
) -> None:
    """A Stop raised as the spawn returns, before any caller holds the child, takes it down.

    CPython checks for an injected exception at the end of the CALL
    that spawned the child, before its result is bound to a name: a
    Stop landing there (or inside ``Popen.__init__`` once the child was
    forked) left no ``Popen`` for anyone to kill, so the child ran on,
    a minute here, unreaped.  ``start_process_group`` now initialises a
    ``Popen`` made before the spawn and kills what it holds on any
    exception, closing the pipes with it.

    That handler kills the group and reaps the leader before the Stop
    reaches the runner's own handler: the group is signalled once, by
    the handler that still owns the leader's pid.  ``run_capped``
    signalled it again after the reap, when the pid (the group id)
    could already be a stranger's.  On Windows the reaped leader's pid
    stays the ``Popen`` handle's, and its Job Object may still hold
    descendants (see :func:`kiss.core.processes.pid_handed_on`), so
    ``run_capped``'s handler ends the group once more there, with the
    leader's exit status set; ``run_process_group`` kills only an
    unreaped leader on every platform.
    """
    log = _GroupKillLog()
    _stop_while_parked(_SpawnReturnStop(), run, log)
    if IS_WINDOWS and run.func is run_capped:  # pragma: no cover — Windows-only branch
        assert len(log.group_kills) == 2, "the Job Object of the reaped leader was not ended"
        assert log.group_kills[0] is None and log.group_kills[1] is not None
    else:
        assert log.group_kills == [None], "the group was signalled after its leader was reaped"


@pytest.mark.parametrize("run", _runners(_py("import time; time.sleep(60)")))
def test_stop_landing_as_the_spawning_call_returns_kills_and_reaps_the_child(
    run: Callable[[], object],
) -> None:
    """A Stop raised as the spawning call returns to ``run_*`` still takes the child down.

    The cleanup inside ``popen_process_group`` covers the spawn, not
    its own return: a Stop raised there left its caller without the
    child (``proc = popen_process_group(...)`` never bound it) and
    without a handler (the assignment stood before the caller's
    ``try``), so the child ran on with both pipes open.  The callers
    now make the ``Popen`` first and have :func:`start_process_group`
    initialise it inside their ``try``.
    """
    log = _GroupKillLog()
    _stop_while_parked(_SpawnReturnStop(start_process_group.__code__), run, log)
    assert log.group_kills == [None], "the group was signalled after its leader was reaped"


_LEAVING_LAUNCHER = """\
import pathlib, subprocess, sys
child = subprocess.Popen(
    [sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
"""
"""A launcher leaving a minute's sleeper in its group (its Job Object); writes its pid, exits."""


def _running(pid: int) -> bool:
    """Whether *pid* is alive and, where ``/proc`` tells, not a zombie awaiting init's reap."""
    if not pid_alive(pid):
        return False
    if not Path("/proc/self").exists():
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:  # reaped between the two looks
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_stop_landing_as_the_exited_leader_is_seen_takes_its_descendants_down(
    tmp_path: Path,
) -> None:
    """A Stop raised once the leader exited, a descendant still running, kills the group.

    The launcher left a sleeper in its process group (its Job Object on
    Windows) and exited; ``run_capped`` saw the exit through
    :func:`_wait_exited`, which does not reap the leader, and the Stop
    lands as that returns.  The handler must still kill the group: on
    POSIX the unreaped leader keeps the group id its own
    (``returncode`` is ``None`` at the kill); on Windows
    ``exit_status_unreaped`` read the status with ``poll``, which set
    ``returncode`` while the Job Object still held the sleeper, and the
    handler took that for a reap and skipped the kill, leaving the
    sleeper to its minute (:func:`kiss.core.processes.pid_handed_on`
    tells the two apart).
    """
    marker = tmp_path / "grandchild.pid"
    log = _GroupKillLog()
    _stop_while_parked(
        _SpawnReturnStop(_wait_exited.__code__),
        partial(run_capped, _py(_LEAVING_LAUNCHER) + [str(marker)], limit=1000, timeout=60),
        log,
    )
    grandchild = int(marker.read_text())
    try:
        deadline = time.monotonic() + 10
        while _running(grandchild) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _running(grandchild), "the exited leader's descendant outlived the Stop"
        assert log.group_kills == [0 if IS_WINDOWS else None]
    finally:
        if _running(grandchild):
            with contextlib.suppress(OSError):
                os.kill(grandchild, SIGKILL)


@pytest.mark.parametrize("run", _runners(["/nonexistent/kiss-no-such-command"]))
def test_a_command_that_could_not_start_is_not_signalled(run: Callable[[], object]) -> None:
    """A failed exec leaves a ``Popen`` with a pid and reaped already; nobody signals its group.

    ``Popen.__init__`` forks, learns that the exec failed, reaps the
    child and raises with the ``Popen`` holding its pid and exit
    status.  ``run_capped``'s handler saw the pid and killed the group
    of that reaped pid, which could already be a stranger's.
    """
    log = _GroupKillLog()
    worker = _Worker(run, log)
    worker.start()
    worker.join(10)
    assert isinstance(worker.raised, FileNotFoundError)
    assert log.group_kills == [], "the group of a reaped pid was signalled"
