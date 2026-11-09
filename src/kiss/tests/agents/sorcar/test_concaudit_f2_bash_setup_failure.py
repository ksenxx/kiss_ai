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
from kiss.core.processes import popen_process_group

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
    # The tag rides in a shell comment, so it shows in the shell's own
    # command line (what the probe looks for) without reaching ``sleep``.
    # 30 s is the longest foreground sleep the sleep guard lets run.
    command = f"sleep 30 # {tag}"
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
        assert job.process.poll() is not None, f"job {job.job_id} was left running"


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
            lambda: tools.Bash(f"sleep 30 # {tag}", "tagged sleep", timeout_seconds=30),
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
            lambda: tools.Bash(f"sleep 30 # {tag}", "tagged sleep", background=True),
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
def test_bash_still_runs_normally_after_the_reorder() -> None:
    """The moved helper-thread starts keep the ordinary path intact."""
    tools = UsefulTools(stop_event=threading.Event())
    assert tools.Bash("echo f2-ok", "echo", timeout_seconds=30).strip() == "f2-ok"
    lines: list[str] = []
    streaming = UsefulTools(stream_callback=lines.append)
    assert "f2-stream" in streaming.Bash("echo f2-stream", "echo", timeout_seconds=30)
    assert lines == ["f2-stream\n"]
