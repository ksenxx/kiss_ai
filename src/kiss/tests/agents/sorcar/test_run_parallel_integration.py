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

import threading
from typing import Any

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

    def test_single_task(self) -> None:
        """A single-element list completes and returns a one-element list."""
        results = run_tasks_parallel(
            ["What is 2 + 2? Reply with just the number."],
            max_workers=1,
            model=FAST_MODEL,
        )
        assert len(results) == 1
        parsed = _parse_yaml_result(results[0])
        assert "summary" in parsed

    def test_two_independent_tasks(self) -> None:
        """Two independent tasks run concurrently and both succeed."""
        results = run_tasks_parallel(
            [
                "What is the capital of France? Reply with just the city name.",
                "What is the capital of Japan? Reply with just the city name.",
            ],
            max_workers=2,
            model=FAST_MODEL,
        )
        assert len(results) == 2
        for r in results:
            parsed = _parse_yaml_result(r)
            assert "summary" in parsed

    def test_three_tasks_order_preserved(self) -> None:
        """Results are returned in the same order as input tasks."""
        tasks = [
            "Reply with exactly the word 'ALPHA' and nothing else.",
            "Reply with exactly the word 'BETA' and nothing else.",
            "Reply with exactly the word 'GAMMA' and nothing else.",
        ]
        results = run_tasks_parallel(tasks, max_workers=3, model=FAST_MODEL)
        assert len(results) == 3
        # Each result should contain its respective keyword
        summaries = [_parse_yaml_result(r).get("summary", "") for r in results]
        assert "ALPHA" in summaries[0], f"Expected ALPHA in: {summaries[0]}"
        assert "BETA" in summaries[1], f"Expected BETA in: {summaries[1]}"
        assert "GAMMA" in summaries[2], f"Expected GAMMA in: {summaries[2]}"

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
            model=FAST_MODEL,
            work_dir=str(tmp_path),
        )
        assert len(results) == 1
        parsed = _parse_yaml_result(results[0])
        assert "Hello" in parsed.get("summary", ""), (
            f"Expected file content in summary: {parsed}"
        )

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
            model=FAST_MODEL,
            work_dir=str(tmp_path),
        )
        assert len(results) == 2
        all_text = " ".join(
            _parse_yaml_result(r).get("summary", "") for r in results
        )
        assert "apple" in all_text.lower(), f"Expected 'apple' in: {all_text}"
        assert "banana" in all_text.lower(), f"Expected 'banana' in: {all_text}"


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------


@skip_no_key
class TestRunParallelEdgeCases:
    """Edge cases and boundary conditions for parallel execution."""

    def test_single_task_parallel(self) -> None:
        """Parallel with just one task works correctly."""
        results = run_tasks_parallel(
            ["Reply with the word 'SOLO'."],
            max_workers=1,
            model=FAST_MODEL,
        )
        assert len(results) == 1
        assert "SOLO" in _parse_yaml_result(results[0]).get("summary", "")

    def test_max_workers_one(self) -> None:
        """max_workers=1 forces sequential execution (still returns correct results)."""
        results = run_tasks_parallel(
            [
                "Reply with the word 'FIRST'.",
                "Reply with the word 'SECOND'.",
            ],
            max_workers=1,
            model=FAST_MODEL,
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
            model=FAST_MODEL,
            work_dir=str(tmp_path),
        )
        assert len(results) == 3
        # At least check that results came back
        for r in results:
            parsed = _parse_yaml_result(r)
            assert "summary" in parsed

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
            model=FAST_MODEL,
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
        event = self._inject_tab_id(event)
        with self._capture_lock:
            self.captured.append(event)
        super().broadcast(event)


@skip_no_key
class TestSubagentTabEventsE2E:
    """E2E tests verifying subagent tab events with real LLM calls."""

    def test_subagent_tab_events_broadcast(self) -> None:
        """run_tasks_parallel with a printer broadcasts open/done tab events."""
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-e2e"

        results = run_tasks_parallel(
            [
                "Reply with just the word 'ALPHA'.",
                "Reply with just the word 'BETA'.",
            ],
            max_workers=2,
            model=FAST_MODEL,
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

    def test_subagent_streaming_events_have_tab_ids(self) -> None:
        """Streaming events from sub-agents carry the correct tabId."""
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-stream"

        run_tasks_parallel(
            ["Reply with just 'hello'."],
            max_workers=1,
            model=FAST_MODEL,
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

    def test_subagent_tabs_not_counted_in_max_tabs(self) -> None:
        """Subagent tabs are excluded from MAX_TABS trimming in frontend JS."""
        # This test verifies the JS logic by checking that the
        # trimOldestTabs function filters out isSubagentTab.
        main_js = (
            Path(__file__).resolve().parents[3]
            / "agents" / "vscode" / "media" / "main.js"
        )
        content = main_js.read_text()
        assert "!t.isSubagentTab" in content, (
            "trimOldestTabs must exclude subagent tabs"
        )

    def test_description_field_in_open_event(self) -> None:
        """openSubagentTab event carries 'description' field for JS frontend."""
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-desc"

        run_tasks_parallel(
            ["Reply with the word DELTA."],
            max_workers=1,
            model=FAST_MODEL,
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
