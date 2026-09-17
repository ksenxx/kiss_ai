# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Stopping a ``run_parallel`` tool call stops its sub-agents.

The tool-call panel's Stop button interrupts the parent's
``run_parallel`` with ``ToolCallInterrupted``.  The parent's own stop
event is NOT set (the task goes on), so the children would keep
running — and spending — behind a fan-out the user just stopped.  The
fan-out therefore owns a stop event of its own, set on exactly this
abandonment, that every child's ``_SubagentStopEvent`` chains to.

Real ``ThreadPoolExecutor`` fan-out, a real child agent running a real
``Bash`` tool (``sleep 120``) handed to it by a real local stand-in
model, the real stop-event chain.  Nothing else ends that sleep early:
the child can only finish because it was told to stop.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    begin_tool_call,
    end_tool_call,
    interrupt_tool_call,
    unregister_tool_call,
)
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
    wait_for,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-rp-interrupt-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _SleepingChildModel:
    """Stand-in model: the child's first turn is ``Bash("sleep 120")``.

    A child that is NOT stopped comes back for a second turn once the
    sleep ends (or is killed); a stopped child never does — the Bash
    result print raises its ``KeyboardInterrupt`` first.
    """

    def __init__(self) -> None:
        self.child_turns = 0
        self.first_turn = threading.Event()

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Hand the child a long sleep, then finish any later turn."""
        if "SLEEPY" in request_text(request):
            self.child_turns += 1
            if self.child_turns == 1:
                self.first_turn.set()
                return tool_call_response(
                    "Bash", {"command": "sleep 120", "description": "wait"},
                )
        return finish_response("done")


def test_interrupting_run_parallel_stops_the_children(env: IsolatedKissHome) -> None:
    model = _SleepingChildModel()
    server = StandInModelServer(model)
    printer = CapturePrinter()
    parent = WorktreeSorcarAgent("rp-interrupt-parent")
    parent.printer = printer
    parent.model_name = STANDIN_MODEL
    parent.model_config = server.model_config
    parent.work_dir = str(env.repo)
    fanout: dict[str, Any] = {}

    def run_fanout() -> None:
        printer._thread_local.task_id = "rpinterrupt"
        token = begin_tool_call("run_parallel")
        try:
            try:
                fanout["results"] = parent._run_tasks_parallel(
                    ["SLEEPY child task"], max_workers=1,
                )
                end_tool_call(token)
            except ToolCallInterrupted:
                fanout["interrupted"] = True
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            fanout["error"] = exc
        finally:
            unregister_tool_call(token)

    thread = threading.Thread(target=run_fanout, daemon=True)
    thread.start()
    try:
        assert model.first_turn.wait(60), "the sub-agent never reached the model"
        # The child is now inside Bash("sleep 120") (or about to be).
        assert wait_for(
            lambda: any(
                e.get("name") == "Bash" for e in printer.events_of_type("tool_call")
            ),
            timeout=30,
        ), "the sub-agent never started its Bash call"
        t0 = time.monotonic()
        assert thread.ident is not None
        assert interrupt_tool_call(thread.ident, "run_parallel") is True
        thread.join(timeout=60)
        assert not thread.is_alive(), "the parent never left run_parallel"
        assert fanout.get("interrupted") is True, fanout
        # The child was told to stop through the fan-out's event: its
        # shell is killed and it unwinds long before the sleep ends.
        assert parent.reclaim_abandoned_subagents(timeout=60), (
            "the sub-agent kept running after its run_parallel was stopped"
        )
        assert time.monotonic() - t0 < 60
        assert model.child_turns == 1, "a stopped child must not take another turn"
    finally:
        thread.join(timeout=30)
        server.stop()
