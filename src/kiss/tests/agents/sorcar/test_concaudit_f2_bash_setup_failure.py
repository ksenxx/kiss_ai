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
from typing import Any

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools

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
            (``_drain_stdout`` for the reader, ``_stop_monitor`` for
            the stop monitor) whose ``Thread.start`` is interrupted.

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
@pytest.mark.parametrize("helper", ["_drain_stdout", "_stop_monitor"])
def test_interrupted_helper_start_kills_the_spawned_shell(helper: str) -> None:
    """A stop landing inside a helper ``Thread.start`` leaves no shell behind."""
    assert len(_WAIT_LINE) == 1, "threading.Thread.start changed shape"
    tag = f"concaudit-f2-{uuid.uuid4().hex}"
    # The tag rides in a shell comment, so it shows in the shell's own
    # command line (what the probe looks for) without reaching ``sleep``.
    command = f"sleep 60 # {tag}"
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_bash_still_runs_normally_after_the_reorder() -> None:
    """The moved helper-thread starts keep the ordinary path intact."""
    tools = UsefulTools(stop_event=threading.Event())
    assert tools.Bash("echo f2-ok", "echo", timeout_seconds=30).strip() == "f2-ok"
    lines: list[str] = []
    streaming = UsefulTools(stream_callback=lines.append)
    assert "f2-stream" in streaming.Bash("echo f2-stream", "echo", timeout_seconds=30)
    assert lines == ["f2-stream\n"]
