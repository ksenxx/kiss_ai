# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for run_parallel / run_tasks_parallel with real LLM calls.

These tests make actual API calls to verify the parallel execution pipeline
end-to-end. They use claude-haiku-4-5 (fast/cheap) with tight budgets.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import run_tasks_parallel
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.test_run_parallel_integration import (  # noqa: F401
    FAST_MODEL,
    _has_anthropic_key,
    skip_no_key,
)

TINY_BUDGET = 0.50


class _CapturePrinter(JsonPrinter):
    """Real JsonPrinter subclass that captures all broadcast events."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[dict[str, Any]] = []
        self._capture_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Capture event then delegate to parent for recording logic."""
        event = self._inject_task_id(event)
        with self._capture_lock:
            self.captured.append(event)
        super().broadcast(event)


@skip_no_key
class TestNestedParallelReal:
    """Real LLM tests for nested parallel execution.

    Each test exercises a tree of parallel invocations:

    * Tasks per ``run_parallel`` call: at most **2**.
    * Total ``run_parallel`` invocations across the whole tree: at most
      **3** (one outer + two inner = 3, giving four leaf sub-agents).

    These tests verify that the per-level ``budget_used`` /
    ``total_tokens_used`` / ``total_steps`` aggregation done by
    :meth:`SorcarAgent._run_tasks_parallel` and
    :meth:`ChatSorcarAgent._run_tasks_parallel` chains correctly all
    the way up to the top-level parent.
    """

    @pytest.mark.slow
    def test_nested_parallel_subagent_tab_events(
        self, tmp_path: Path,
    ) -> None:
        """Nested parallel emits sub-agent tab events for every level.

        With one outer parent call (2 middles) and one nested call per
        middle (2 leaves each), the printer must observe **6** distinct
        ``new_tab`` events: 2 from the outer invocation plus 2 from
        each of the 2 inner invocations.  Each leaf's ``subagentDone``
        routing id must embed the corresponding middle's persisted task
        id, confirming that the printer thread-local routing chains
        correctly across nesting levels.
        """
        printer = _CapturePrinter()
        printer._thread_local.task_id = "nested-events-root"

        results = run_tasks_parallel(
            [
                (
                    "Call run_parallel with these two tasks and nothing "
                    "else: "
                    "['Reply only with TAB_L1A', 'Reply only with TAB_L1B']."
                    " Then finish."
                ),
                (
                    "Call run_parallel with these two tasks and nothing "
                    "else: "
                    "['Reply only with TAB_L2A', 'Reply only with TAB_L2B']."
                    " Then finish."
                ),
            ],
            max_workers=2,
            model_name=FAST_MODEL,
            work_dir=str(tmp_path),
            printer=printer,
        )
        assert len(results) == 2

        open_events = [
            e for e in printer.captured if e.get("type") == "new_tab"
        ]
        assert len(open_events) >= 6, (
            f"Expected at least 6 new_tab events (2 middles + 4 leaves), "
            f"got {len(open_events)}: "
            f"{[e.get('task_id') for e in open_events]}"
        )
        sub_task_ids = {str(e.get("task_id") or "") for e in open_events}
        assert all(sub_task_ids)
        assert len(sub_task_ids) == len(open_events), (
            "Sub-agent task ids must be unique"
        )

        done_events = [
            e for e in printer.captured if e.get("type") == "subagentDone"
        ]
        done_tab_ids = {e["tab_id"] for e in done_events}
        outer_done = {
            t for t in done_tab_ids
            if t.startswith("task-nested-events-root__sub_")
        }
        assert outer_done == {
            "task-nested-events-root__sub_0",
            "task-nested-events-root__sub_1",
        }, (
            f"Expected exactly 2 outer (middle) subagentDone events as "
            f"direct children of the root, got {sorted(outer_done)}"
        )
        leaf_done = done_tab_ids - outer_done
        assert len(leaf_done) >= 4, (
            f"Expected at least 4 leaf subagentDone events under the "
            f"middles, got {sorted(leaf_done)}"
        )
        leaf_parents = set()
        for tab_id in leaf_done:
            match = re.fullmatch(r"task-(.+)__sub_\d+", tab_id)
            assert match, f"Unexpected leaf subagentDone tab_id: {tab_id!r}"
            leaf_parents.add(match.group(1))
        assert leaf_parents <= sub_task_ids, (
            f"Leaf routing ids must embed a middle's persisted task id: "
            f"parents={sorted(leaf_parents)} "
            f"task_ids={sorted(sub_task_ids)}"
        )
        assert len(leaf_parents) >= 2, (
            f"Leaves must be spread across both middles: {leaf_parents}"
        )


@skip_no_key
class TestSubagentTabEventsE2E:
    """E2E tests verifying subagent tab events with real LLM calls."""

    @pytest.mark.slow
    def test_subagent_tab_events_broadcast(self) -> None:
        """run_tasks_parallel with a printer broadcasts new_tab/subagentDone."""
        printer = _CapturePrinter()
        printer._thread_local.task_id = "parent-e2e"

        results = run_tasks_parallel(
            [
                "Reply with just the word 'ALPHA'.",
                "Reply with just the word 'BETA'.",
            ],
            max_workers=2,
            model_name=FAST_MODEL,
            printer=printer,
        )
        assert len(results) == 2

        open_events = [
            e for e in printer.captured if e.get("type") == "new_tab"
        ]
        assert len(open_events) == 2, (
            f"Expected 2 new_tab events, got {len(open_events)}"
        )
        sub_task_ids = {e.get("task_id") for e in open_events}
        assert len(sub_task_ids) == 2
        assert all(sub_task_ids), f"Empty task_id in new_tab: {open_events}"
        for ev in open_events:
            assert "parent_tab_id" in ev

        done_events = [
            e for e in printer.captured if e.get("type") == "subagentDone"
        ]
        assert len(done_events) == 2, (
            f"Expected 2 subagentDone events, got {len(done_events)}"
        )
        done_tab_ids = {e["tab_id"] for e in done_events}
        assert done_tab_ids == {
            "task-parent-e2e__sub_0", "task-parent-e2e__sub_1",
        }

    @pytest.mark.slow
    def test_subagent_streaming_events_have_tab_ids(self) -> None:
        """Streaming events from sub-agents carry the correct tabId."""
        printer = _CapturePrinter()
        printer._thread_local.task_id = "parent-stream"

        run_tasks_parallel(
            ["Reply with just 'hello'."],
            max_workers=1,
            model_name=FAST_MODEL,
            printer=printer,
        )

        open_events = [
            e for e in printer.captured if e.get("type") == "new_tab"
        ]
        assert len(open_events) == 1
        sub_task_id = open_events[0]["task_id"]
        routed = [
            e for e in printer.captured
            if e.get("taskId") == sub_task_id
            and e.get("type") not in ("new_tab", "subagentDone")
        ]
        assert len(routed) > 0, (
            "Expected streaming events stamped with the sub-agent task id"
        )


    @pytest.mark.slow
    def test_description_reaches_subagent_stream(self) -> None:
        """The sub-agent's task description reaches the frontend stream.

        The ``new_tab`` event only announces the sub-agent's task id;
        the human-readable description is delivered by the sub-agent's
        own ``prompt`` event stamped with that task id (and by the
        persisted ``task_history`` row the frontend loads from).
        """
        printer = _CapturePrinter()
        printer._thread_local.task_id = "parent-desc"

        run_tasks_parallel(
            ["Reply with the word DELTA."],
            max_workers=1,
            model_name=FAST_MODEL,
            printer=printer,
        )

        open_ev = [
            e for e in printer.captured if e.get("type") == "new_tab"
        ]
        assert len(open_ev) == 1
        sub_task_id = open_ev[0]["task_id"]
        prompt_events = [
            e for e in printer.captured
            if e.get("type") == "prompt" and e.get("taskId") == sub_task_id
        ]
        assert prompt_events, "sub-agent must broadcast its prompt"
        assert any("DELTA" in str(e.get("text", "")) for e in prompt_events)
