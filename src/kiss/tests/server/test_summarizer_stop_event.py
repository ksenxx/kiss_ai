# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The summarizer's shell must be killable.

When an executor session dies with a recoverable error,
:meth:`RelentlessAgent.perform_task` falls back to a summarizer agent
armed with ``Read`` and ``Bash``.  It built those tools with
``getattr(self.printer, "stop_event", None)`` — but ``stop_event``
lives on the printer's THREAD-LOCAL, not on the printer, so the value
was always ``None`` and ``UsefulTools`` never started its
process-group killer.  Clicking Stop then took as long as whatever
shell command the summarizer happened to be running.

This test drives the real recovery path: a real agent, a real
executor session that really exceeds its step budget, a real
summarizer, and a real ``Bash`` child process whose PID is checked
with ``os.kill(pid, 0)``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    request_text,
    tool_call_response,
    wait_for,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-f3-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def test_summarizer_bash_is_killed_by_the_thread_stop_event(
    env: IsolatedKissHome,
) -> None:
    """Stop must kill the summarizer's shell process group.

    The executor session is starved of steps so it raises the
    recoverable "exceeded N steps" error, which is exactly the branch
    that spawns the summarizer.  The summarizer's first (and only)
    action is a real long-running shell command that records its own
    process-group leader pid.
    """
    pid_file = env.repo / "summarizer.pid"

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        if "The executor's trajectory is saved at" in request_text(request):
            return tool_call_response(
                "Bash",
                {
                    "command": f"echo $$ > {pid_file}; sleep 120",
                    "description": "long summarizer analysis",
                },
            )
        return tool_call_response(
            "Bash", {"command": "echo working", "description": "executor step"},
        )

    server = StandInModelServer(responder)
    printer = CapturePrinter()
    agent = SorcarAgent("f3-summarizer")
    stop_event = threading.Event()
    outcome: dict[str, Any] = {}

    def run_agent() -> None:
        printer._thread_local.stop_event = stop_event
        try:
            outcome["result"] = agent.run(
                prompt_template="F3 recovery path",
                model_name=STANDIN_MODEL,
                model_config=server.model_config,
                work_dir=str(env.repo),
                printer=printer,
                max_steps=3,
                max_sub_sessions=1,
                web_tools=False,
            )
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            outcome["error"] = exc

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: _pid_recorded(pid_file)), (
            "the summarizer never started its shell command"
        )
        pid = int(pid_file.read_text().strip())
        assert _process_alive(pid), "the shell exited before the stop"

        stop_event.set()

        assert wait_for(lambda: not _process_alive(pid), timeout=10.0), (
            "the summarizer's shell survived the stop: UsefulTools was "
            "built with stop_event=None, so nothing polls the stop"
        )
    finally:
        stop_event.set()
        thread.join(timeout=30)
        server.stop()


def _pid_recorded(pid_file: Path) -> bool:
    """Return whether the shell has written its pid yet."""
    return pid_file.exists() and bool(pid_file.read_text().strip())


def _process_alive(pid: int) -> bool:
    """Return whether *pid* still exists (signal 0 probe)."""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True
