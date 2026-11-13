# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_bash_streaming`` must kill the shell when its helper threads fail to start.

``UsefulTools._bash_streaming`` (useful_tools.py) spawned the shell,
then started the stop-monitor thread and the stdout reader thread
BEFORE entering the ``try`` whose ``except BaseException`` kills the
process group and whose ``finally`` sets ``done``.  A ``Thread.start()``
that raises there — ``RuntimeError: can't start new thread`` under
thread exhaustion, or a stop injected by the server while
``Thread.start`` waits for the new thread to come up — therefore left
the just-spawned command running unowned, and an already-started stop
monitor polling ``done`` forever.

Reproduced for real: the command is a live ``sleep`` with a unique tag,
and a ``sys.settrace`` tracer raises ``KeyboardInterrupt`` at the exact
``self._started.wait()`` line of ``threading.Thread.start`` for the
helper thread under test — the same arbitrary-boundary delivery the
server's ``PyThreadState_SetAsyncExc`` stop performs, made
deterministic.  Afterwards no process carrying the tag may survive and
every helper thread the call started must have exited.

Thread exhaustion itself (``RLIMIT_NPROC``) is not used: the limit
counts processes and threads alike, so a value low enough to fail
``Thread.start`` also fails the ``fork`` in ``_spawn`` that has to
succeed first, and a value tuned to the live process count is racy on
a shared host.  The injected exception takes the identical code path.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import useful_tools
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core import tool_interrupt
from kiss.core.processes import kill_process_tree, popen_process_group
from kiss.tests.server.pause import Pause

_WAIT_LINE = [
    first + i
    for (lines, first) in [inspect.getsourcelines(threading.Thread.start)]
    for i, text in enumerate(lines)
    if "self._started.wait()" in text
]


def _tagged_processes(tag: str) -> list[str]:
    """Return the command lines of live processes whose arguments carry *tag*."""
    result = subprocess.run(
        ["pgrep", "-af", tag], capture_output=True, text=True, check=False,
    )
    return [line for line in result.stdout.splitlines() if tag in line]


def _run_bash_with_injection(
    tools: UsefulTools, command: str, helper_target_name: str,
) -> dict[str, Any]:
    """Run ``tools.Bash(command)`` interrupting the start of one helper thread.

    Args:
        tools: The tools instance under test.
        command: The shell command to run.
        helper_target_name: ``__name__`` of the helper thread's target
            (``_drain_stdout`` for the reader, ``_own_child`` for the
            owner thread that spawns the shell and runs its stop
            monitor) whose ``Thread.start`` is interrupted.

    Returns:
        ``{"injected": bool, "exc": BaseException | None, "result": str | None}``.
    """
    start_code = threading.Thread.start.__code__
    outcome: dict[str, Any] = {"injected": False, "exc": None, "result": None}

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if (
            event == "line"
            and not outcome["injected"]
            and frame.f_code is start_code
            and frame.f_lineno == _WAIT_LINE[0]
            and getattr(frame.f_locals["self"]._target, "__name__", "")
            == helper_target_name
        ):
            outcome["injected"] = True
            raise KeyboardInterrupt("injected stop inside Thread.start")
        return tracer

    sys.settrace(tracer)
    try:
        outcome["result"] = tools.Bash(command, "tagged sleep", timeout_seconds=30)
    except BaseException as exc:  # noqa: BLE001 — the injected stop is expected
        outcome["exc"] = exc
    finally:
        sys.settrace(None)
    return outcome


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("helper", ["_drain_stdout", "_own_child"])
def test_interrupted_helper_start_kills_the_spawned_shell(helper: str) -> None:
    """A stop landing inside a helper ``Thread.start`` leaves no shell behind."""
    assert len(_WAIT_LINE) == 1, "threading.Thread.start changed shape"
    tag = f"concaudit-f2-{uuid.uuid4().hex}"
    # The tag is the name ``sleep`` runs under (``exec -a``): the probe finds it
    # in the shell's command line and, once the shell has exec'd, in ``sleep``'s
    # (a tag in a shell comment vanished there, hiding a ``sleep`` left running).
    # 30 s is the longest foreground sleep the sleep guard lets run.
    command = f"exec -a {tag} sleep 30"
    stop_event = threading.Event()
    tools = UsefulTools(stop_event=stop_event)
    before = set(threading.enumerate())
    try:
        outcome = _run_bash_with_injection(tools, command, helper)
        assert outcome["injected"], f"{helper}'s Thread.start was never traced"
        assert isinstance(outcome["exc"], KeyboardInterrupt), outcome
        # Give a leaked shell time to exec its ``sleep`` so the probe can
        # see it; a killed group produces nothing in this window.
        time.sleep(0.5)
        assert _tagged_processes(tag) == [], "the shell survived the failed setup"
        # The helper that did start (if any) must exit: ``done`` is set.
        deadline = time.monotonic() + 5
        leftover: list[threading.Thread] = []
        while time.monotonic() < deadline:
            leftover = [
                t for t in threading.enumerate() if t not in before and t.is_alive()
            ]
            if not leftover:
                break
            time.sleep(0.05)
        assert leftover == [], f"helper threads leaked: {leftover}"
    finally:
        stop_event.set()
        subprocess.run(["pkill", "-f", tag], check=False)


def _run_with_injection_after_spawn(
    run: Callable[[], str],
    nth_line: int,
    make_exc: Callable[[], BaseException],
    stop_event: threading.Event | None,
) -> dict[str, Any]:
    """Run *run* raising ``make_exc()`` at the *nth_line*-th line start after the shell exists.

    The tracer watches every thread (``threading.settrace``) for the
    return of :func:`popen_process_group`, the moment the child
    process exists, then raises in the calling thread at its
    *nth_line*-th following line event of the tool's own code
    (``useful_tools``, ``tool_interrupt``): the check-point delivery of
    the server's injected Stop, made deterministic (a trace event comes
    BEFORE its line runs, so the stdlib's lock-restoring ``finally``
    bodies, which a real injection lands after, are left alone).
    *stop_event* is set first when given, as ``_stop_task`` sets the
    task's event before it injects.

    Returns:
        ``{"injected": bool, "exc": BaseException | None, "result": str | None}``.
    """
    spawn_code = popen_process_group.__code__
    tool_files = {useful_tools.__file__, tool_interrupt.__file__}
    task_tid = threading.get_ident()
    outcome: dict[str, Any] = {"injected": False, "exc": None, "result": None}
    state = {"spawned": False, "lines": 0}

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if event == "return" and frame.f_code is spawn_code:
            state["spawned"] = True
        elif (
            event == "line"
            and state["spawned"]
            and not outcome["injected"]
            and threading.get_ident() == task_tid
            and frame.f_code.co_filename in tool_files
        ):
            state["lines"] += 1
            if state["lines"] == nth_line:
                outcome["injected"] = True
                if stop_event is not None:
                    stop_event.set()
                raise make_exc()
        return tracer

    threading.settrace(tracer)
    sys.settrace(tracer)
    try:
        outcome["result"] = run()
    except BaseException as exc:  # noqa: BLE001 — the injected stop is expected
        outcome["exc"] = exc
    finally:
        sys.settrace(None)
        threading.settrace(None)
    return outcome


def _assert_no_survivor(tag: str, tools: UsefulTools) -> None:
    """Assert that no shell carrying *tag* outlives the interrupted launch for long."""
    deadline = time.monotonic() + 5
    survivors = _tagged_processes(tag)
    while survivors and time.monotonic() < deadline:
        time.sleep(0.1)
        survivors = _tagged_processes(tag)
    assert survivors == [], f"the shell survived the interrupted launch: {survivors}"
    for job in tools._jobs.values():
        assert job.returncode() is not None, f"job {job.job_id} was left running"


_INJECTIONS = [
    ("task-stop", lambda: KeyboardInterrupt("injected task Stop")),
    ("setup-failure", lambda: RuntimeError("can't start new thread")),
]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("kind,make_exc", _INJECTIONS, ids=[k for k, _ in _INJECTIONS])
@pytest.mark.parametrize("nth_line", [1, 2, 3])
def test_stop_right_after_the_foreground_spawn_leaves_no_shell(
    nth_line: int, kind: str, make_exc: Callable[[], BaseException],
) -> None:
    """A Stop landing at any check point after the shell was spawned still kills it.

    The shell used to be spawned before its ``done`` event, output queue
    and cleanup region existed, so an exception at the first check
    points after ``_spawn`` returned left it running with no monitor
    and nothing to kill it by.
    """
    tag = f"concaudit-f2-fg-{uuid.uuid4().hex}"
    stop_event = threading.Event()
    tools = UsefulTools(stop_event=stop_event)
    try:
        outcome = _run_with_injection_after_spawn(
            lambda: tools.Bash(f"exec -a {tag} sleep 30", "tagged sleep", timeout_seconds=30),
            nth_line,
            make_exc,
            stop_event if kind == "task-stop" else None,
        )
        assert outcome["injected"], "the shell was never spawned under the tracer"
        # A setup failure is reported as an ``Error:`` result; a Stop propagates.
        assert outcome["exc"] is not None or str(outcome["result"]).startswith("Error:"), outcome
        _assert_no_survivor(tag, tools)
    finally:
        stop_event.set()
        subprocess.run(["pkill", "-f", tag], check=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("kind,make_exc", _INJECTIONS, ids=[k for k, _ in _INJECTIONS])
@pytest.mark.parametrize("nth_line", [1, 2, 3])
def test_stop_right_after_the_background_spawn_leaves_no_job(
    tmp_path: Path, nth_line: int, kind: str, make_exc: Callable[[], BaseException],
) -> None:
    """A Stop landing after a background job's shell was spawned kills it.

    The job's shell used to be spawned, its log context left and its
    watcher started in separate regions, so an exception at the check
    points between them left the shell running unregistered, with no
    watcher and no job id anyone could kill it by.
    """
    tag = f"concaudit-f2-bg-{uuid.uuid4().hex}"
    stop_event = threading.Event()
    tools = UsefulTools(stop_event=stop_event, work_dir=str(tmp_path))
    try:
        outcome = _run_with_injection_after_spawn(
            lambda: tools.Bash(f"exec -a {tag} sleep 30", "tagged sleep", background=True),
            nth_line,
            make_exc,
            stop_event if kind == "task-stop" else None,
        )
        assert outcome["injected"], "the shell was never spawned under the tracer"
        assert outcome["exc"] is not None, outcome
        _assert_no_survivor(tag, tools)
    finally:
        stop_event.set()
        subprocess.run(["pkill", "-f", tag], check=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_tool_stop_at_the_background_launch_cleanup_ends_job_and_watcher(tmp_path: Path) -> None:
    """A tool-panel Stop unwinding a launch before its ``settled`` signal still ends the job.

    The watcher waited for that signal watching the task's Stop alone
    (audit 2026-11-10, S13-F1): a ``ToolCallInterrupted`` landing on
    entry to the ``Event.set`` of the launch's ``finally`` (its second,
    after ``registered``) left the shell running under an id the model
    never got, and the watcher waiting for good.  The Stop is a real
    one, on a registered call; its forced injection, which lands at any
    check point, is raised at that entry.  A launch merely slow to
    settle (held at its first ``Event.set`` until the watcher is back at
    its wait) is no Stop: the job runs on.
    """
    tag = f"concaudit-f2-tool-stop-{uuid.uuid4().hex}"
    stop_event = threading.Event()
    tools = UsefulTools(stop_event=stop_event, work_dir=str(tmp_path))
    launch_code = UsefulTools._start_background_job.__code__
    watch_code = useful_tools._watch_job.__code__
    token = tool_interrupt.new_tool_call("Bash")
    signals: list[threading.Event] = []
    waits: list[threading.Thread] = []  # the watcher's thread, once per wait it starts
    back_at_wait = threading.Event()
    slow_launch: list[bool] = []
    alive_unstopped: list[bool] = []

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if event != "call":
            return None
        if frame.f_code is threading.Event.wait.__code__ and frame.f_back.f_code is watch_code:
            waits.append(threading.current_thread())
            if len(waits) == 2:
                back_at_wait.set()
        elif frame.f_code is threading.Event.set.__code__ and frame.f_back.f_code is launch_code:
            signals.append(frame.f_locals["self"])
            if len(signals) == 1:
                # Held until the watcher starts its second wait: a whole slice and its
                # look at the Stops went by with none of them set.
                slow_launch.append(back_at_wait.wait(20))
                alive_unstopped.extend(j.returncode() is None for j in tools._jobs.values())
            else:
                tool_interrupt.interrupt_tool_call(token.thread_ident, call_id=token.call_id)
                raise tool_interrupt.ToolCallInterrupted("injected tool Stop")
        return None

    tool_interrupt.register_tool_call(token)
    threading.settrace(tracer)  # the watcher's thread is traced too
    sys.settrace(tracer)
    try:
        with pytest.raises(tool_interrupt.ToolCallInterrupted):
            tools.Bash(f"exec -a {tag} sleep 30", "tagged sleep", background=True)
    finally:
        sys.settrace(None)
        threading.settrace(None)
        tool_interrupt.unregister_tool_call(token)
    try:
        assert len(signals) == 2 and not signals[1].is_set(), "the launch settled"
        assert slow_launch == [True], "the watcher never came back to its wait"
        assert alive_unstopped == [True], "the watcher killed a launch no Stop had reached"
        # The watcher ends once it has killed and reaped the job, within a slice of its wait.
        waits[0].join(5)
        assert not waits[0].is_alive(), "the watcher is still waiting"
        _assert_no_survivor(tag, tools)
        assert len(tools._jobs) == 1 and not stop_event.is_set()
    finally:
        stop_event.set()
        subprocess.run(["pkill", "-f", tag], check=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_simultaneous_cleanups_never_signal_a_reaped_shell() -> None:
    """The cleanup of a timeout and that of a Stop reach one shell: it is signalled unreaped.

    Each used to check the shell (``poll``), signal its group and reap
    it, on its own thread: the one held between its check and its
    signal (here at the signal's entry) sent it after the other had
    reaped the shell, to a pid the kernel may have handed to a
    stranger's group by then (audit 2026-11-10, B/S13-F5).  Only the
    thread that owns the shell signals it now, and reaps it afterwards.
    """
    tag = f"concaudit-f2-race-{uuid.uuid4().hex}"
    stop_event = threading.Event()
    tools = UsefulTools(stop_event=stop_event)
    shells: list[subprocess.Popen[Any]] = []
    results: list[str] = []

    def is_the_shell(local: dict[str, Any]) -> bool:
        if tag not in str(local["proc"].args):
            return False
        shells.append(local["proc"])
        return True

    call = threading.Thread(
        target=lambda: results.append(
            tools.Bash(f"exec -a {tag} sleep 30", "tagged sleep", timeout_seconds=1)
        ),
        daemon=True,
    )
    try:
        with Pause(kill_process_tree.__code__, is_the_shell) as pause:
            call.start()
            assert pause.paused.wait(20), "the timeout's cleanup never reached the group signal"
            stop_event.set()
            time.sleep(1)  # five looks of the Stop's cleanup at its event
            assert shells[0].returncode is None, "the shell was reaped under a pending signal"
        call.join(20)
        assert results and "timeout" in results[0], results
        _assert_no_survivor(tag, tools)
    finally:
        stop_event.set()
        subprocess.run(["pkill", "-f", tag], check=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_status_read_does_not_reap_a_job_shell_under_a_pending_stop(tmp_path: Path) -> None:
    """A job's shell that exits while the Stop's signal is pending stays unreaped until it is sent.

    A status read reaped it (``poll``): the signal, due to end the
    child the shell left in its group, then went to a pid that was no
    longer the job's (audit 2026-11-10, B/S13-F5).
    """
    tag = f"concaudit-f2-exit-{uuid.uuid4().hex}"
    stop_event = threading.Event()
    tools = UsefulTools(stop_event=stop_event, work_dir=str(tmp_path))
    try:
        tools.Bash(f"(exec -a {tag} sleep 30) & sleep 1", "leaves a child", background=True)
        (job,) = tools._jobs.values()
        with Pause(kill_process_tree.__code__, lambda local: local["proc"] is job.process) as pause:
            stop_event.set()
            assert pause.paused.wait(20), "the Stop never reached the group signal"
            deadline = time.monotonic() + 20
            while "exited with code 0" not in tools.bash_job(job.job_id):
                assert time.monotonic() < deadline, "the shell never exited"
                time.sleep(0.05)
            assert job.process.returncode is None, "a status read reaped the shell"
        _assert_no_survivor(tag, tools)
    finally:
        stop_event.set()
        subprocess.run(["pkill", "-f", tag], check=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_bash_still_runs_normally_after_the_reorder() -> None:
    """The moved helper-thread starts keep the ordinary path intact."""
    tools = UsefulTools(stop_event=threading.Event())
    assert tools.Bash("echo f2-ok", "echo", timeout_seconds=30).strip() == "f2-ok"
    lines: list[str] = []
    streaming = UsefulTools(stream_callback=lines.append)
    assert "f2-stream" in streaming.Bash("echo f2-stream", "echo", timeout_seconds=30)
    assert lines == ["f2-stream\n"]
