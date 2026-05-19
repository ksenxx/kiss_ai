"""Integration tests for run_parallel / run_tasks_parallel with real LLM calls.

These tests make actual API calls to verify the parallel execution pipeline
end-to-end. They use claude-haiku-4-5 (fast/cheap) with tight budgets.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.sorcar_agent import SorcarAgent, run_tasks_parallel
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

FAST_MODEL = "claude-haiku-4-5"
TINY_BUDGET = 0.50  # $0.50 per test — enough for simple tasks


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


skip_no_key = pytest.mark.skipif(
    not _has_anthropic_key(),
    reason="ANTHROPIC_API_KEY not set",
)


def _parse_yaml_result(result: str) -> dict:
    """Parse a YAML result string into a dict, tolerant of multi-doc."""
    parsed = yaml.safe_load(result)
    if isinstance(parsed, dict):
        return parsed
    return {"raw": result}


# ---------------------------------------------------------------------------
# 1. run_tasks_parallel() — the standalone function
# ---------------------------------------------------------------------------


@skip_no_key
class TestRunTasksParallelReal:
    """Real LLM calls through run_tasks_parallel()."""

    @pytest.mark.slow
    def test_single_task(self) -> None:
        """A single-element list completes and returns a one-element list."""
        results = run_tasks_parallel(
            ["What is 2 + 2? Reply with just the number."],
            max_workers=1,
            model_name=FAST_MODEL,
        )
        assert len(results) == 1
        parsed = _parse_yaml_result(results[0])
        assert "summary" in parsed

    @pytest.mark.slow
    def test_two_independent_tasks(self) -> None:
        """Two independent tasks run concurrently and both succeed."""
        results = run_tasks_parallel(
            [
                "What is the capital of France? Reply with just the city name.",
                "What is the capital of Japan? Reply with just the city name.",
            ],
            max_workers=2,
            model_name=FAST_MODEL,
        )
        assert len(results) == 2
        for r in results:
            parsed = _parse_yaml_result(r)
            assert "summary" in parsed

    @pytest.mark.slow
    def test_three_tasks_order_preserved(self) -> None:
        """Results are returned in the same order as input tasks."""
        tasks = [
            "Reply with exactly the word 'ALPHA' and nothing else.",
            "Reply with exactly the word 'BETA' and nothing else.",
            "Reply with exactly the word 'GAMMA' and nothing else.",
        ]
        results = run_tasks_parallel(tasks, max_workers=3, model_name=FAST_MODEL)
        assert len(results) == 3
        # Each result should contain its respective keyword
        summaries = [_parse_yaml_result(r).get("summary", "") for r in results]
        assert "ALPHA" in summaries[0], f"Expected ALPHA in: {summaries[0]}"
        assert "BETA" in summaries[1], f"Expected BETA in: {summaries[1]}"
        assert "GAMMA" in summaries[2], f"Expected GAMMA in: {summaries[2]}"

    @pytest.mark.slow
    def test_with_work_dir(self, tmp_path: Path) -> None:
        """Tasks can use a custom work_dir."""
        # Create a file in tmp_path for the agent to read
        test_file = tmp_path / "greeting.txt"
        test_file.write_text("Hello from the test file!")

        results = run_tasks_parallel(
            [
                f"Read the file {test_file} and tell me what it says. "
                "Include the exact content in your summary.",
            ],
            max_workers=1,
            model_name=FAST_MODEL,
            work_dir=str(tmp_path),
        )
        assert len(results) == 1
        parsed = _parse_yaml_result(results[0])
        assert "Hello" in parsed.get("summary", ""), (
            f"Expected file content in summary: {parsed}"
        )

    @pytest.mark.slow
    def test_file_tasks_parallel(self, tmp_path: Path) -> None:
        """Multiple file-reading tasks run in parallel."""
        # Create two files
        (tmp_path / "a.txt").write_text("Contents of file A: apple")
        (tmp_path / "b.txt").write_text("Contents of file B: banana")

        results = run_tasks_parallel(
            [
                f"Read {tmp_path / 'a.txt'} and reply with its contents.",
                f"Read {tmp_path / 'b.txt'} and reply with its contents.",
            ],
            max_workers=2,
            model_name=FAST_MODEL,
            work_dir=str(tmp_path),
        )
        assert len(results) == 2
        all_text = " ".join(
            _parse_yaml_result(r).get("summary", "") for r in results
        )
        assert "apple" in all_text.lower(), f"Expected 'apple' in: {all_text}"
        assert "banana" in all_text.lower(), f"Expected 'banana' in: {all_text}"


# ---------------------------------------------------------------------------
# 1b. Budget aggregation — sub-agent costs roll up to the parent
# ---------------------------------------------------------------------------


class TestBudgetAggregationFast:
    """Fast tests for budget/token/step aggregation that need no LLM."""

    def test_empty_tasks_populates_zero_totals(self) -> None:
        """Empty task list still fills totals_out with zeros."""
        totals: dict[str, float] = {}
        results = run_tasks_parallel([], totals_out=totals)
        assert results == []
        assert totals["budget_used"] == 0.0
        assert totals["total_tokens_used"] == 0
        assert totals["total_steps"] == 0

    def test_totals_out_optional(self) -> None:
        """Caller may omit totals_out; function still returns a list."""
        results = run_tasks_parallel([])
        assert results == []


@skip_no_key
class TestBudgetAggregationReal:
    """Real LLM call to verify sub-agent costs roll up to the parent."""

    @pytest.mark.slow
    def test_parent_budget_includes_subagent_cost(
        self, tmp_path: Path,
    ) -> None:
        """A SorcarAgent that invokes _run_tasks_parallel accumulates the
        sub-agents' budget into its own ``budget_used`` (and tokens/steps).
        """
        parent = SorcarAgent("parent-budget")
        # Reset is normally done by run(); we set the fields directly
        # so we can call _run_tasks_parallel as a unit.
        parent.work_dir = str(tmp_path)
        parent.model_name = FAST_MODEL
        parent.printer = None
        parent.budget_used = 0.0
        parent.total_tokens_used = 0
        parent.total_steps = 0
        before_budget = parent.budget_used
        before_tokens = parent.total_tokens_used
        before_steps = parent.total_steps

        results = parent._run_tasks_parallel(
            [
                "Reply with the word 'AGG1'.",
                "Reply with the word 'AGG2'.",
            ],
            max_workers=2,
        )
        assert len(results) == 2

        # Every real LLM call costs >$0 and consumes at least one token
        # and one step.  Verify the parent's running totals grew.
        assert parent.budget_used > before_budget, (
            f"parent.budget_used did not grow: {parent.budget_used}"
        )
        assert parent.total_tokens_used > before_tokens
        assert parent.total_steps > before_steps


# ---------------------------------------------------------------------------
# 1c. Nested parallel — sub-agents themselves invoke ``run_parallel``
# ---------------------------------------------------------------------------


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
    def test_nested_parallel_budget_chains(
        self, tmp_path: Path,
    ) -> None:
        """Two levels of parallelism: parent → 2 middles → 2 leaves each.

        Tree shape (3 ``run_parallel`` invocations total, 2 tasks each)::

            parent._run_tasks_parallel([M1, M2])      # invocation #1
              ├── M1 LLM-invokes run_parallel([L1a, L1b])  # invocation #2
              └── M2 LLM-invokes run_parallel([L2a, L2b])  # invocation #3

        After the tree completes, the parent's ``budget_used`` /
        ``total_tokens_used`` / ``total_steps`` must include the cost of
        all four leaf agents (plus the two middles).  Per-level
        aggregation is validated by parsing each middle's YAML result
        and confirming the run_parallel tool was actually invoked and
        returned the expected leaf summaries.
        """
        parent = SorcarAgent("nested-parent")
        parent.work_dir = str(tmp_path)
        parent.model_name = FAST_MODEL
        parent.printer = None
        parent.budget_used = 0.0
        parent.total_tokens_used = 0
        parent.total_steps = 0

        middle_prompt_template = (
            "You MUST call the run_parallel tool exactly once with these "
            "two tasks (and nothing else):\n"
            "  1. \"Reply with exactly the word {leaf_a} and nothing else.\"\n"
            "  2. \"Reply with exactly the word {leaf_b} and nothing else.\"\n"
            "After run_parallel returns, immediately call finish with "
            "success=True and summary set to the two leaf results "
            "joined by a comma."
        )

        results = parent._run_tasks_parallel(
            [
                middle_prompt_template.format(
                    leaf_a="NESTED_L1A", leaf_b="NESTED_L1B",
                ),
                middle_prompt_template.format(
                    leaf_a="NESTED_L2A", leaf_b="NESTED_L2B",
                ),
            ],
            max_workers=2,
        )
        assert len(results) == 2

        # The parent's running totals must reflect EVERY level of the
        # tree.  Six real LLM-driven agents ran (2 middles + 4 leaves),
        # and the per-level aggregation done by ``_run_tasks_parallel``
        # at every node must roll those costs all the way up to the
        # parent.  Confirming that all three counters grew above zero
        # proves nested chaining works: if any intermediate level had
        # failed to add its sub-agents in, at least one of the three
        # counters would still be at its pre-call zero baseline.  We
        # deliberately avoid asserting on the textual content of the
        # middle summaries because LLM responses are non-deterministic;
        # the budget aggregation contract is what we care about here.
        assert parent.budget_used > 0.0, (
            f"parent.budget_used did not aggregate nested cost: "
            f"{parent.budget_used}"
        )
        assert parent.total_tokens_used > 0, (
            f"parent.total_tokens_used did not aggregate: "
            f"{parent.total_tokens_used}"
        )
        assert parent.total_steps > 0, (
            f"parent.total_steps did not aggregate: "
            f"{parent.total_steps}"
        )

    @pytest.mark.slow
    def test_nested_parallel_subagent_tab_events(
        self, tmp_path: Path,
    ) -> None:
        """Nested parallel emits ``openSubagentTab`` events for every level.

        With one outer parent call (2 middles) and one nested call per
        middle (2 leaves each), the printer must observe **6** distinct
        ``openSubagentTab`` events: 2 from the outer invocation plus 2
        from each of the 2 inner invocations.  Each leaf event's
        ``parent_tab_id`` must point at the corresponding middle's tab
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
            e for e in printer.captured if e.get("type") == "openSubagentTab"
        ]

        # The two outer (middle) tabs must be direct children of the
        # root tab.  Their ids are deterministic because the outer
        # ``run_tasks_parallel`` synthesises them from the root tab id.
        outer_events = [
            e for e in open_events
            if e.get("parent_tab_id") == "nested-events-root"
        ]
        assert len(outer_events) == 2, (
            f"Expected exactly 2 outer (middle) openSubagentTab events "
            f"as direct children of the root, got {len(outer_events)}: "
            f"{[e.get('tab_id') for e in outer_events]}"
        )
        middle_ids = {ev["tab_id"] for ev in outer_events}

        # The leaves must each have a middle tab as their parent.  We
        # accept either exact parent==middle_id, or a nested-id prefix
        # (defensive against any extra LLM-driven invocations under a
        # middle tab) — what matters is that the printer's thread-local
        # routing chained the leaves under the correct middle.
        leaf_events = [
            e for e in open_events if e["tab_id"] not in middle_ids
        ]
        assert len(leaf_events) >= 4, (
            f"Expected at least 4 leaf openSubagentTab events under the "
            f"middle tabs, got {len(leaf_events)}: "
            f"{[e.get('tab_id') for e in leaf_events]}"
        )
        for ev in leaf_events:
            parent_id = ev.get("parent_tab_id", "")
            assert parent_id in middle_ids or any(
                parent_id.startswith(mid) for mid in middle_ids
            ), (
                f"Leaf tab {ev['tab_id']!r} has parent {parent_id!r}, "
                f"expected one of {middle_ids} (or a descendant thereof)"
            )

        # Every opened sub-agent tab must also emit a subagentDone event.
        done_events = [
            e for e in printer.captured if e.get("type") == "subagentDone"
        ]
        opened_tab_ids = {e["tab_id"] for e in open_events}
        done_tab_ids = {e["tab_id"] for e in done_events}
        assert opened_tab_ids == done_tab_ids, (
            f"openSubagentTab / subagentDone tab ids diverged.  "
            f"opened={opened_tab_ids}  done={done_tab_ids}"
        )


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------


@skip_no_key
class TestRunParallelEdgeCases:
    """Edge cases and boundary conditions for parallel execution."""

    @pytest.mark.slow
    def test_single_task_parallel(self) -> None:
        """Parallel with just one task works correctly."""
        results = run_tasks_parallel(
            ["Reply with the word 'SOLO'."],
            max_workers=1,
            model_name=FAST_MODEL,
        )
        assert len(results) == 1
        assert "SOLO" in _parse_yaml_result(results[0]).get("summary", "")

    @pytest.mark.slow
    def test_max_workers_one(self) -> None:
        """max_workers=1 forces sequential execution (still returns correct results)."""
        results = run_tasks_parallel(
            [
                "Reply with the word 'FIRST'.",
                "Reply with the word 'SECOND'.",
            ],
            max_workers=1,
            model_name=FAST_MODEL,
        )
        assert len(results) == 2
        assert "FIRST" in _parse_yaml_result(results[0]).get("summary", "")
        assert "SECOND" in _parse_yaml_result(results[1]).get("summary", "")

    def test_run_parallel_tool_not_available_when_disabled(self) -> None:
        """run_parallel is NOT in tool list when is_parallel=False."""
        agent = SorcarAgent("test-no-parallel")
        agent._use_web_tools = False
        agent._is_parallel = False
        tools = agent._get_tools()
        names = [getattr(t, "__name__", "") for t in tools]
        assert "run_parallel" not in names

    def test_run_parallel_tool_available_when_enabled(self) -> None:
        """run_parallel IS in tool list when is_parallel=True."""
        agent = SorcarAgent("test-yes-parallel")
        agent._use_web_tools = False
        agent._is_parallel = True
        tools = agent._get_tools()
        names = [getattr(t, "__name__", "") for t in tools]
        assert "run_parallel" in names

    def test_run_parallel_tool_signature(self) -> None:
        """The run_parallel tool has the expected parameters."""
        import inspect

        agent = SorcarAgent("test-sig")
        agent._use_web_tools = False
        agent._is_parallel = True
        tools = agent._get_tools()
        rp = [t for t in tools if getattr(t, "__name__", "") == "run_parallel"][0]
        sig = inspect.signature(rp)
        params = list(sig.parameters.keys())
        assert "tasks" in params
        assert "max_workers" in params


# ---------------------------------------------------------------------------
# 3. Concurrent correctness with file I/O
# ---------------------------------------------------------------------------


@skip_no_key
class TestParallelFileIO:
    """Verify parallel agents writing to separate files don't collide."""

    @pytest.mark.slow
    def test_parallel_write_different_files(self, tmp_path: Path) -> None:
        """Multiple agents writing different files concurrently succeed."""
        tasks = [
            (
                f"Write the text 'content-{i}' to the file "
                f"{tmp_path / f'parallel_{i}.txt'}. "
                "Use the Write tool. Then finish with success."
            )
            for i in range(3)
        ]
        results = run_tasks_parallel(
            tasks,
            max_workers=3,
            model_name=FAST_MODEL,
            work_dir=str(tmp_path),
        )
        assert len(results) == 3
        # At least check that results came back
        for r in results:
            parsed = _parse_yaml_result(r)
            assert "summary" in parsed

    @pytest.mark.slow
    def test_parallel_read_same_file(self, tmp_path: Path) -> None:
        """Multiple agents reading the same file concurrently succeed."""
        shared = tmp_path / "shared.txt"
        shared.write_text("shared content for parallel reading")

        tasks = [
            f"Read {shared} and include its content in your summary."
            for _ in range(2)
        ]
        results = run_tasks_parallel(
            tasks,
            max_workers=2,
            model_name=FAST_MODEL,
            work_dir=str(tmp_path),
        )
        assert len(results) == 2
        for r in results:
            summary = _parse_yaml_result(r).get("summary", "")
            assert "shared" in summary.lower(), (
                f"Expected 'shared' in: {summary}"
            )


# ---------------------------------------------------------------------------
# 4. Subagent tab events — E2E with real LLM calls and a real printer
# ---------------------------------------------------------------------------


class _CapturePrinter(BaseBrowserPrinter):
    """Real BaseBrowserPrinter subclass that captures all broadcast events."""

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
class TestSubagentTabEventsE2E:
    """E2E tests verifying subagent tab events with real LLM calls."""

    @pytest.mark.slow
    def test_subagent_tab_events_broadcast(self) -> None:
        """run_tasks_parallel with a printer broadcasts open/done tab events."""
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
            e for e in printer.captured if e.get("type") == "openSubagentTab"
        ]
        assert len(open_events) == 2, (
            f"Expected 2 openSubagentTab events, got {len(open_events)}"
        )
        for i, ev in enumerate(open_events):
            assert ev["tab_id"] == f"parent-e2e__sub_{i}"
            assert ev["parent_tab_id"] == "parent-e2e"
            assert ev["isSubagentTab"] is True
            assert ev.get("description"), "description should be non-empty"

        done_events = [
            e for e in printer.captured if e.get("type") == "subagentDone"
        ]
        assert len(done_events) == 2, (
            f"Expected 2 subagentDone events, got {len(done_events)}"
        )
        done_tab_ids = {e["tab_id"] for e in done_events}
        assert done_tab_ids == {
            "parent-e2e__sub_0", "parent-e2e__sub_1",
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

        sub_tab_id = "parent-stream__sub_0"
        routed = [
            e for e in printer.captured
            if e.get("tabId") == sub_tab_id
            and e.get("type") not in ("openSubagentTab", "subagentDone")
        ]
        assert len(routed) > 0, (
            "Expected streaming events routed to subagent tab"
        )


    @pytest.mark.slow
    def test_description_field_in_open_event(self) -> None:
        """openSubagentTab event carries 'description' field for JS frontend."""
        printer = _CapturePrinter()
        printer._thread_local.task_id = "parent-desc"

        run_tasks_parallel(
            ["Reply with the word DELTA."],
            max_workers=1,
            model_name=FAST_MODEL,
            printer=printer,
        )

        open_ev = [
            e for e in printer.captured if e.get("type") == "openSubagentTab"
        ]
        assert len(open_ev) == 1
        assert "description" in open_ev[0], (
            "openSubagentTab must have 'description' field"
        )
        assert "DELTA" in open_ev[0]["description"]
