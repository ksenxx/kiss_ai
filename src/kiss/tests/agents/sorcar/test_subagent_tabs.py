# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for subagent tabs created by run_parallel.

When the agent invokes run_parallel, each sub-agent should broadcast
events to a dedicated subagent tab. These tests verify:
- new_tab events are broadcast for each sub-task
- Streaming events are stamped with the sub-agent's own task id
- subagentDone events are broadcast when sub-tasks complete

Uses real LLM calls with claude-haiku-4-5 and tight budgets.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import os
import re
import threading

import pytest

from kiss.server.json_printer import JsonPrinter

FAST_MODEL = "claude-haiku-4-5"


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


skip_no_key = pytest.mark.skipif(
    not _has_anthropic_key(),
    reason="ANTHROPIC_API_KEY not set",
)


class _CapturePrinter(JsonPrinter):
    """Printer that records all broadcast events for inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []
        self._ev_lock = threading.Lock()

    def broadcast(self, event: dict) -> None:
        """Record event and apply tab_id injection."""
        event = self._inject_task_id(event)
        with self._ev_lock:
            self.events.append(event)
        with self._lock:
            self._record_event(event)


# -----------------------------------------------------------------------
# Backend integration tests: verify events from parallel sub-agents
# -----------------------------------------------------------------------


@skip_no_key
class TestSubagentTabEvents:
    """Verify that run_parallel broadcasts subagent tab events."""

    @pytest.mark.slow
    def test_parallel_creates_subagent_tab_events(self) -> None:
        """Running a task with run_parallel creates new_tab events."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        printer = _CapturePrinter()
        agent = SorcarAgent("test-subagent-tabs")
        agent._is_parallel = True
        agent._use_web_tools = False

        parent_tab_id = "parent-tab-123"
        printer._thread_local.task_id = parent_tab_id

        agent.run(
            prompt_template=(
                "Call run_parallel with these two tasks and nothing else: "
                "['Reply with the word ALPHA', 'Reply with the word BETA']. "
                "Then finish with the combined results."
            ),
            model_name=FAST_MODEL,
            work_dir=".",
            printer=printer,
            max_budget=2.0,
            web_tools=False,
            is_parallel=True,
        )

        events = printer.events

        # Task-centric contract: each spawned sub-agent broadcasts one
        # ``new_tab`` event carrying its persisted ``task_id``; the
        # frontend materialises the sub-agent tab from that.
        open_events = [e for e in events if e.get("type") == "new_tab"]
        assert len(open_events) >= 2, (
            f"Expected at least 2 new_tab events, got {len(open_events)}. "
            f"Event types: {[e.get('type') for e in events]}"
        )

        sub_task_ids = set()
        for ev in open_events:
            assert ev.get("task_id"), f"Missing task_id in new_tab: {ev}"
            assert "parent_tab_id" in ev, f"Missing parent_tab_id: {ev}"
            sub_task_ids.add(ev["task_id"])
        assert len(sub_task_ids) == len(open_events), (
            "Sub-agent task IDs must be unique"
        )

        # One ``subagentDone`` per sub-task, carrying the deterministic
        # backend routing id ``task-{parent}__sub_{idx}``.  A failed
        # ``run_parallel`` attempt (LLM tool-arg error) still emits its
        # cleanup ``subagentDone`` without a ``new_tab``, so ``>=``.
        done_events = [e for e in events if e.get("type") == "subagentDone"]
        assert len(done_events) >= len(open_events), (
            f"Expected >= {len(open_events)} subagentDone events, "
            f"got {len(done_events)}"
        )
        for ev in done_events:
            assert re.fullmatch(r"task-.+__sub_\d+", ev.get("tab_id", "")), (
                f"Unexpected subagentDone tab_id: {ev}"
            )

        # Streaming events from the sub-agents are stamped with the
        # sub-agent's own task id.
        sub_events = [
            e for e in events
            if e.get("taskId") in sub_task_ids
            and e.get("type") not in ("new_tab", "subagentDone")
        ]
        assert len(sub_events) > 0, (
            "Expected streaming events stamped with sub-agent task IDs"
        )

    @pytest.mark.slow
    def test_parallel_subagent_events_have_correct_types(self) -> None:
        """Sub-agent events include standard streaming types."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        printer = _CapturePrinter()
        agent = SorcarAgent("test-subagent-types")
        agent._is_parallel = True
        agent._use_web_tools = False

        parent_tab_id = "parent-tab-456"
        printer._thread_local.task_id = parent_tab_id

        agent.run(
            prompt_template=(
                "Use the run_parallel tool to run one task: "
                "'Read this message and reply with DONE'. "
                "Then finish."
            ),
            model_name=FAST_MODEL,
            work_dir=".",
            printer=printer,
            max_budget=2.0,
            web_tools=False,
            is_parallel=True,
        )

        events = printer.events
        open_events = [e for e in events if e.get("type") == "new_tab"]
        assert len(open_events) >= 1

        sub_task_id = open_events[0]["task_id"]
        sub_events = [
            e for e in events
            if e.get("taskId") == sub_task_id
            and e.get("type") not in ("new_tab", "subagentDone")
        ]

        sub_types = {e.get("type") for e in sub_events}
        assert "result" in sub_types or "text_delta" in sub_types, (
            f"Expected result or text_delta in sub-agent events, got: {sub_types}"
        )
