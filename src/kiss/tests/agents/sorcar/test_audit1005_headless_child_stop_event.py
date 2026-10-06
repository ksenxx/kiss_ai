# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A fan-out child without a ``JsonPrinter`` still stops its running shell.

``run_tasks_parallel`` binds every child's ``_SubagentStopEvent`` to
the worker thread (``stop_signal.set_thread_stop_event``) so the
per-child ``timeout`` and a parent stop reach the child whatever
printer it has.  ``SorcarAgent.run`` used to read the child's stop
event from ``printer._thread_local.stop_event`` only: a child of a
CLI parent (``ConsolePrinter``, which has no such attribute) ran its
``UsefulTools`` with ``stop_event=None``, so a ``Bash("sleep 120")``
outlived the 8 s timeout by two minutes.

Real ``ThreadPoolExecutor`` fan-out, a real child agent running a real
``Bash`` tool handed to it by a real local stand-in model; the shell is
killed only because the child's stop event reached the tools.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.conftest import posix_only
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-headless-stop-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _SleepingChildModel:
    """Stand-in model: the child's first turn is a 120 s sleep.

    The shell records the sleep's pid in *pid_file* before ``exec``-ing
    it, so the test can prove the process existed and was killed
    rather than never started.
    """

    def __init__(self, pid_file: Path) -> None:
        self.child_turns = 0
        self.first_turn = threading.Event()
        self.command = f"echo $$ > {pid_file} && exec sleep 120"

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Hand the child the sleep, then finish any later turn."""
        if "SLEEPY" in request_text(request):
            self.child_turns += 1
            if self.child_turns == 1:
                self.first_turn.set()
                return tool_call_response(
                    "Bash", {"command": self.command, "description": "wait"},
                )
        return finish_response("done")


def _pid_alive(pid: int) -> bool:
    """True while *pid* exists (a reaped process no longer does)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@posix_only("the probe shell uses $$/exec, SIGKILL and os.kill(pid, 0)")
def test_console_fanout_child_timeout_kills_its_shell(env: IsolatedKissHome) -> None:
    pid_file = env.repo / "sleep.pid"
    model = _SleepingChildModel(pid_file)
    server = StandInModelServer(model)
    parent = SorcarAgent("console-stop-parent")
    parent.printer = ConsolePrinter()
    parent.model_name = STANDIN_MODEL
    parent.model_config = server.model_config
    parent.work_dir = str(env.repo)
    parent._use_web_tools = False
    try:
        started = time.monotonic()
        # Generous timeout: the child must reach its first model turn
        # and start the shell before the timer fires, or the test would
        # pass on the model-request stop path without touching the
        # shell-killing wiring under test.
        (result,) = parent._run_tasks_parallel(
            ["SLEEPY child task"], max_workers=1, timeout=8,
        )
        elapsed = time.monotonic() - started
        assert model.first_turn.is_set(), "the sub-agent never reached the model"
        assert pid_file.is_file(), "the child's shell never started before the timeout"
        sleep_pid = int(pid_file.read_text().strip())
        # The shell is killed when the timeout sets the child's stop
        # event; without that the fan-out waits out the whole sleep.
        assert elapsed < 60, f"the child's Bash outlived its timeout ({elapsed:.0f}s)"
        assert not _pid_alive(sleep_pid), f"sleep {sleep_pid} survived the child's stop"
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert parsed["summary"] == "Sub-agent task did not finish within 8 s and was stopped."
    finally:
        server.stop()
        if pid_file.is_file():
            with contextlib.suppress(ProcessLookupError, ValueError):
                os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
