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

import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.core import processes, stop_signal
from kiss.core.models import model as model_module
from kiss.core.models.model import _CLIProcess
from kiss.tests.conftest import posix_only
from kiss.tests.server.pause import Pause, Pauses

_set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
_set_async_exc.argtypes = (ctypes.c_ulong, ctypes.py_object)

_CHILD = [sys.executable, "-c", "import time; time.sleep(60)"]
_MODEL_FILE = model_module.__file__


class _InterruptAfterSpawn:
    """Profile hook injecting ``KeyboardInterrupt`` at the k-th model call after the spawn."""

    def __init__(self, k: int, task_ident: int) -> None:
        self.k = k
        self.task_ident = task_ident
        self.calls = 0
        self.child: subprocess.Popen[str] | None = None

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        """Record the spawned child, then count calls made from model.py and inject at the k-th."""
        if self.child is None:
            if event == "return" and frame.f_code is processes.popen_process_group.__code__:
                self.child = arg
            return
        if threading.get_ident() != self.task_ident:
            return
        caller = frame.f_back
        if event == "call" and caller is not None and caller.f_code.co_filename == _MODEL_FILE:
            self.calls += 1
            if self.calls == self.k:
                sys.setprofile(None)
                _set_async_exc(threading.get_ident(), ctypes.py_object(KeyboardInterrupt))


def _construct_with_interrupt(k: int) -> tuple[_CLIProcess | None, subprocess.Popen[str]]:
    """Build a ``_CLIProcess`` with an interrupt at the k-th call after the spawn."""
    hook = _InterruptAfterSpawn(k, threading.get_ident())
    proc: _CLIProcess | None = None
    threading.setprofile(hook)  # the child is spawned on a helper thread
    sys.setprofile(hook)
    try:
        proc = _CLIProcess(_CHILD, "probe", 60)
    except KeyboardInterrupt:
        pass
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
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


class _InterruptAtReturn:
    """Profile hook injecting ``KeyboardInterrupt`` at the k-th return into *frames* of model.py.

    The spawned child is recorded on whichever thread spawns it (the
    hook is installed for new threads too); the interrupt is delivered
    at the end of the call instruction in the model.py frame, before
    the callee's result is stored, which is where a Stop lands when the
    daemon forces it while the task thread waits inside that call.
    """

    def __init__(self, k: int, task_ident: int, code_names: tuple[str, ...]) -> None:
        self.k = k
        self.task_ident = task_ident
        self.code_names = code_names
        self.returns = 0
        self.child: subprocess.Popen[str] | None = None

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        """Record the child; count returns into the watched model.py frames; inject at the k-th."""
        if event == "return" and frame.f_code is processes.popen_process_group.__code__:
            self.child = arg
        if threading.get_ident() != self.task_ident:
            return
        if event == "return":
            into = frame.f_back
        elif event == "c_return":
            into = frame
        else:
            return
        if into is None or into.f_code.co_filename != _MODEL_FILE:
            return
        if into.f_code.co_name not in self.code_names:
            return
        self.returns += 1
        if self.returns == self.k:
            sys.setprofile(None)
            _set_async_exc(self.task_ident, ctypes.py_object(KeyboardInterrupt))


def _sweep_returns(run: Callable[[], Any], code_names: tuple[str, ...]) -> list[tuple[int, str]]:
    """Run *run* with an interrupt at every return into *code_names*; list the leaks.

    Each round injects at the k-th return (k = 1, 2, ...) until a round
    completes without an interrupt; a leak is a round after which the
    spawned child is still alive or its stdin still open.
    """
    leaked: list[tuple[int, str]] = []
    k = 0
    while True:
        k += 1
        hook = _InterruptAtReturn(k, threading.get_ident(), code_names)
        threading.setprofile(hook)
        sys.setprofile(hook)
        interrupted = False
        try:
            run()
        except KeyboardInterrupt:
            interrupted = True
        finally:
            sys.setprofile(None)
            threading.setprofile(None)
        child = hook.child
        if child is not None:
            assert child.stdin is not None
            if child.poll() is None or not child.stdin.closed:
                leaked.append((k, "alive" if child.poll() is None else "stdin open"))
                child.kill()
                child.wait(timeout=30)
            for pipe in (child.stdin, child.stdout, child.stderr):
                assert pipe is not None
                pipe.close()
        if not interrupted:
            break
    assert k > 1, "the sweep made no return to interrupt"
    return leaked


def test_an_interrupt_at_any_return_during_construction_kills_the_child() -> None:
    """A Stop landing as the spawn helper itself returns must not lose the child.

    Only the constructor's own frame is watched: the Stop lands at every
    return into ``__init__``, the one as ``popen_process_group`` returns
    (before ``self._proc`` exists) included.
    """

    def construct() -> None:
        _CLIProcess(_CHILD, "probe", 60).close()

    assert _sweep_returns(construct, ("__init__",)) == []


def test_an_interrupt_as_the_turn_enters_the_with_block_kills_the_child() -> None:
    """A Stop landing as the constructor or ``__enter__`` returns to ``_cli_turn`` closes the child.

    ``with _CLIProcess(...) as proc:`` never reaches ``__exit__`` when the
    interrupt lands before the block is entered, so the turn itself must
    own the supervisor from the moment it exists.  The constructor's own
    returns are swept through the turn too: a Stop landing before the
    spawn finds the supervisor registered but half built, and must reach
    the caller as itself, not as the ``AttributeError`` of closing it
    (audit C4, F3; the sweep lets only ``KeyboardInterrupt`` by).
    """
    from kiss.core.models.claude_code_model import ClaudeCodeModel

    cli_model = ClaudeCodeModel("cc/opus")
    cli_model.model_config = {"timeout": 60}

    def turn() -> None:
        with cli_model._cli_turn(_CHILD, "probe"):
            pass

    assert _sweep_returns(turn, ("_cli_turn", "__init__")) == []


@pytest.mark.parametrize("gives_up_on", ["forced Stop", "Stop", "deadline"])
def test_a_child_spawned_after_the_constructor_gave_up_is_ended_by_the_helper(
    gives_up_on: str,
) -> None:
    """A start given up while the spawn is still in progress leaves no orphan.

    The constructor used to wait the grace for the spawn helper and then
    leave with nothing: a child the helper returned afterwards went into
    a list nobody read again, alive behind three open pipes, and the
    supervisor was already marked closed.  The helper must end the child
    it returns once the constructor has given the start up.

    The constructor must give up by itself too (audit C4, F1): it waited
    for the spawn untimed, so a spawn that hung held the task thread past
    the cooperative Stop and past ``model_config["timeout"]``, which must
    end the turn as a Stop and as the retryable stall error.
    """
    from kiss.core.models.claude_code_model import ClaudeCodeModel

    cli_model = ClaudeCodeModel("cc/opus")
    cli_model.model_config = {"timeout": 0.5 if gives_up_on == "deadline" else 60}
    spawn_held = threading.Event()
    release_spawn = threading.Event()
    helpers: list[threading.Thread] = []
    spawned: list[subprocess.Popen[str]] = []

    def hold_the_spawn(frame: FrameType, event: str, arg: Any) -> None:
        """Hold the helper at the spawn until released; record the child it returns."""
        if frame.f_code is not processes.popen_process_group.__code__:
            return
        if event == "call":
            helpers.append(threading.current_thread())
            spawn_held.set()
            release_spawn.wait(timeout=20)  # a constructor that waits this out fails below
        elif event == "return":
            spawned.append(arg)

    def stop_as_the_helper_starts(frame: FrameType, event: str, arg: Any) -> None:
        """Raise the Stop as the helper's ``start()`` returns, with the spawn held."""
        if event == "return" and frame.f_code is threading.Thread.start.__code__:
            sys.setprofile(None)
            assert spawn_held.wait(timeout=30), "the helper never reached the spawn"
            raise KeyboardInterrupt("Stop while the spawn is in progress")

    ended: BaseException | None = None
    # The cooperative Stop is requested the moment the spawn is held.
    stop_signal.set_thread_stop_event(spawn_held if gives_up_on == "Stop" else None)
    threading.setprofile(hold_the_spawn)
    sys.setprofile(stop_as_the_helper_starts if gives_up_on == "forced Stop" else None)
    try:
        with cli_model._cli_turn(_CHILD, "probe"):
            pass
    except (KeyboardInterrupt, TimeoutError) as error:
        ended = error
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        stop_signal.set_thread_stop_event(None)
        release_spawn.set()
    expected = TimeoutError if gives_up_on == "deadline" else KeyboardInterrupt
    assert isinstance(ended, expected), f"the constructor waited the held spawn out: {ended!r}"
    (helper,) = helpers
    helper.join(timeout=30)
    (child,) = spawned
    assert child.stdin is not None and child.stdout is not None and child.stderr is not None
    try:
        assert not helper.is_alive()
        assert child.returncode is not None, "the late child was not reaped"
        assert [pipe.closed for pipe in (child.stdin, child.stdout, child.stderr)] == [True] * 3
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
        for pipe in (child.stdin, child.stdout, child.stderr):
            pipe.close()


# A child that ignores SIGTERM: only the SIGKILL after the grace ends it.
_TERM_IGNORING_CHILD = [
    sys.executable,
    "-c",
    "import signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    'print("ready", flush=True)\n'
    "time.sleep(60)",
]


@posix_only("the child ignores SIGTERM, which Windows has no equivalent of")
@pytest.mark.parametrize("interrupted_after", ["_signal_tree", "_reap"])
def test_a_stop_during_teardown_still_kills_reaps_and_releases_the_child(
    interrupted_after: str,
) -> None:
    """A Stop landing inside ``close()`` leaves the child dead, reaped and its pipes closed.

    ``close()`` used to mark itself closed before doing anything, so a
    Stop delivered as the SIGTERM returned (before the SIGKILL) left a
    child that ignores SIGTERM alive behind its readers and pipes, and
    the retry from ``_cli_turn``'s outer cleanup was a no-op.  The
    teardown must run its hard kill, reap and release however the grace
    wait ends, and a retry must finish a teardown the Stop cut short
    after the reap.
    """
    from kiss.core.models.claude_code_model import ClaudeCodeModel

    cli_model = ClaudeCodeModel("cc/opus")
    cli_model.model_config = {"timeout": 60}
    task_ident = threading.get_ident()
    code = getattr(_CLIProcess, interrupted_after).__code__
    supervisors: list[_CLIProcess] = []

    def stop_as_it_returns(frame: FrameType, event: str, arg: Any) -> None:
        """Deliver the Stop as the first call of the watched method returns."""
        if event == "return" and frame.f_code is code and threading.get_ident() == task_ident:
            sys.setprofile(None)
            _set_async_exc(task_ident, ctypes.py_object(KeyboardInterrupt))

    interrupted = False
    try:
        with cli_model._cli_turn(_TERM_IGNORING_CHILD, "probe") as proc:
            supervisors.append(proc)
            assert next(proc.lines()).strip() == "ready"
            sys.setprofile(stop_as_it_returns)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        sys.setprofile(None)
    (proc,) = supervisors
    child = proc._proc
    assert child.stdin is not None and child.stdout is not None and child.stderr is not None
    try:
        assert interrupted
        assert child.returncode is not None, "the child was not reaped"
        for reader in proc._readers:
            reader.join(timeout=30)
        assert [reader.is_alive() for reader in proc._readers] == [False, False]
        assert [pipe.closed for pipe in (child.stdin, child.stdout, child.stderr)] == [True] * 3
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
        for pipe in (child.stdin, child.stdout, child.stderr):
            pipe.close()


# A launcher that exits on its own, leaving a grandchild of its process
# group (of its Job Object on Windows) behind, as a CLI leaves a
# background command; the marker file names the grandchild once it exists.
_LEAVING_LAUNCHER = textwrap.dedent(
    """
    import pathlib, subprocess, sys
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    pathlib.Path(sys.argv[1]).write_text(str(grandchild.pid))
    print("ready", flush=True)
    """
)


def _running(pid: int) -> bool:
    """Whether *pid* still runs: it exists and, where ``/proc`` tells, is no zombie."""
    if not processes.pid_alive(pid):
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_a_descendant_outliving_a_launcher_that_exited_is_ended_by_the_turn(
    tmp_path: Path,
) -> None:
    """The normal turn ends a grandchild that outlived a launcher exiting on its own.

    ``close()`` signals the child's whole group (its Job Object on
    Windows) whatever the child's own state, but nothing once the child
    has been reaped, since its pid may then belong to a stranger.  Only
    a POSIX reap hands the pid on: Windows keeps it reserved through the
    ``Popen`` handle while the exit status, which ``wait_for_exit`` reads
    there with ``poll()``, is already set.  Guarding the signals on
    ``returncode`` alone therefore left the Job Object alive after a
    clean exit (review R2 C3/F2): the grandchild must be gone on every
    platform.
    """
    from kiss.core.models.claude_code_model import ClaudeCodeModel

    marker = tmp_path / "grandchild.pid"
    launcher = [sys.executable, "-c", _LEAVING_LAUNCHER, str(marker)]
    cli_model = ClaudeCodeModel("cc/opus")
    cli_model.model_config = {"timeout": 60}
    with cli_model._cli_turn(launcher, "probe") as proc:
        assert [line.strip() for line in proc.lines()] == ["ready"]
        proc.raise_for_exit()
        grandchild = int(marker.read_text())
        assert _running(grandchild), "the launcher's grandchild was not left behind"
    try:
        deadline = time.monotonic() + 30
        while _running(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(grandchild), "the grandchild outlived the turn"
    finally:
        with contextlib.suppress(OSError):
            os.kill(grandchild, processes.SIGKILL)


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


@pytest.mark.parametrize("after_the_reap", [False, True], ids=["before-the-kill", "after-the-reap"])
def test_a_second_stop_during_the_start_abort_still_ends_the_child(after_the_reap: bool) -> None:
    """A Stop cutting the constructor's own abort short leaves the child to the turn's close.

    The first Stop lands once the stdout reader has started, the second
    inside the abort that follows: before its kill, or once it has reaped
    the child.  The supervisor used to count as closed before that abort
    ran, so the close ``_cli_turn`` makes was a no-op and the child lived
    on behind its pipes.  After the reap nothing may be signalled again:
    the pid may be a stranger's by then.
    """
    from kiss.core.models.claude_code_model import ClaudeCodeModel

    supervisors: list[_CLIProcess] = []
    late_signals: list[int] = []
    ended: list[BaseException] = []

    def stderr_reader(frame_locals: dict[str, Any]) -> bool:
        """Hold the start of the second reader: the stdout one already runs."""
        supervisors.append(frame_locals["self"])
        return bool(frame_locals["suffix"] == "stderr")

    def reaped(frame_locals: dict[str, Any]) -> bool:
        """Hold the abort's liveness check of the stdout reader: the child is reaped by then."""
        return bool(frame_locals["self"].name == "cli-stdout-drain")

    def signalled(frame_locals: dict[str, Any]) -> bool:
        """Record a signal sent once the child is reaped; holds nothing."""
        if frame_locals["self"]._proc.returncode is not None:
            late_signals.append(frame_locals["self"]._proc.returncode)
        return False

    def turn() -> None:
        try:
            with ClaudeCodeModel("cc/opus")._cli_turn(_CHILD, "probe"):
                pass
        except KeyboardInterrupt as stop:
            ended.append(stop)

    first = Pause(_CLIProcess._start_reader.__code__, stderr_reader)
    second = (
        Pause(threading.Thread.is_alive.__code__, reaped)
        if after_the_reap
        else Pause(_CLIProcess._abort_start.__code__, lambda frame_locals: True)
    )
    task = threading.Thread(target=turn, daemon=True)
    with Pauses(first, second, Pause(_CLIProcess._signal_tree.__code__, signalled)):
        task.start()
        for pause in (first, second):
            assert pause.paused.wait(30), "the start never got there"
            _set_async_exc(task.ident or 0, ctypes.py_object(KeyboardInterrupt))
            pause.release()
        task.join(60)
    assert ended and not task.is_alive(), "the turn did not end with the Stop"
    child = supervisors[0]._proc
    assert child.stdin is not None and child.stdout is not None and child.stderr is not None
    drains = [t for t in threading.enumerate() if t.name.startswith("cli-std")]
    try:
        assert child.returncode is not None, "the child was left running, or a zombie"
        assert [pipe.closed for pipe in (child.stdin, child.stdout, child.stderr)] == [True] * 3
        assert not any(thread.is_alive() for thread in drains), "a drain thread was left"
        if os.name != "nt":
            assert not late_signals, "the group was signalled after the child was reaped"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
        for pipe in (child.stdin, child.stdout, child.stderr):
            pipe.close()
