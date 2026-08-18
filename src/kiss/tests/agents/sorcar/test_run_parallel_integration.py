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

import os
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar.sorcar_agent import SorcarAgent, run_tasks_parallel

FAST_MODEL = "claude-haiku-4-5"


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
        summaries = [_parse_yaml_result(r).get("summary", "") for r in results]
        assert "ALPHA" in summaries[0], f"Expected ALPHA in: {summaries[0]}"
        assert "BETA" in summaries[1], f"Expected BETA in: {summaries[1]}"
        assert "GAMMA" in summaries[2], f"Expected GAMMA in: {summaries[2]}"

    @pytest.mark.slow
    def test_with_work_dir(self, tmp_path: Path) -> None:
        """Tasks can use a custom work_dir."""
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

        assert parent.budget_used > before_budget, (
            f"parent.budget_used did not grow: {parent.budget_used}"
        )
        assert parent.total_tokens_used > before_tokens
        assert parent.total_steps > before_steps


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
