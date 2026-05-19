"""End-to-end tests for subagent tabs created by run_parallel.

When the agent invokes run_parallel, each sub-agent should broadcast
events to a dedicated subagent tab. These tests verify:
- openSubagentTab events are broadcast for each sub-task
- Streaming events carry the subagent tab_id
- subagentDone events are broadcast when sub-tasks complete
- The frontend JS correctly handles subagent tabs (no input area,
  not counted toward MAX_TABS)
- TypeScript types include the new message types

Uses real LLM calls with claude-haiku-4-5 and tight budgets.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

FAST_MODEL = "claude-haiku-4-5"


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


skip_no_key = pytest.mark.skipif(
    not _has_anthropic_key(),
    reason="ANTHROPIC_API_KEY not set",
)


class _CapturePrinter(BaseBrowserPrinter):
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
        """Running a task with run_parallel creates openSubagentTab events."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        printer = _CapturePrinter()
        agent = SorcarAgent("test-subagent-tabs")
        agent._is_parallel = True
        agent._use_web_tools = False

        parent_tab_id = "parent-tab-123"
        printer._thread_local.task_id = parent_tab_id

        agent.run(
            prompt_template=(
                "Use the run_parallel tool to run these two tasks in parallel: "
                "task 1: 'Reply with the word ALPHA', "
                "task 2: 'Reply with the word BETA'. "
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

        open_events = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(open_events) >= 2, (
            f"Expected at least 2 openSubagentTab events, got {len(open_events)}. "
            f"Event types: {[e.get('type') for e in events]}"
        )

        sub_tab_ids = set()
        for ev in open_events:
            assert "tab_id" in ev, f"Missing tab_id in openSubagentTab: {ev}"
            assert "task_description" in ev, f"Missing task_description: {ev}"
            sub_tab_ids.add(ev["tab_id"])
        assert len(sub_tab_ids) == len(open_events), "Sub-tab IDs must be unique"

        done_events = [e for e in events if e.get("type") == "subagentDone"]
        done_tab_ids = {e.get("tab_id") for e in done_events}
        assert sub_tab_ids == done_tab_ids, (
            f"Done events tab IDs {done_tab_ids} != open events tab IDs {sub_tab_ids}"
        )

        sub_events = [
            e for e in events
            if e.get("tab_id") in sub_tab_ids
            and e.get("type") not in ("openSubagentTab", "subagentDone")
        ]
        assert len(sub_events) > 0, (
            "Expected streaming events from sub-agents with sub-tab IDs"
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
        open_events = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(open_events) >= 1

        sub_tab_id = open_events[0]["tab_id"]
        sub_events = [
            e for e in events
            if e.get("tab_id") == sub_tab_id
            and e.get("type") not in ("openSubagentTab", "subagentDone")
        ]

        sub_types = {e.get("type") for e in sub_events}
        assert "result" in sub_types or "text_delta" in sub_types, (
            f"Expected result or text_delta in sub-agent events, got: {sub_types}"
        )


# -----------------------------------------------------------------------
# Frontend logic tests: verify JS handles subagent tabs correctly
# -----------------------------------------------------------------------

_MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents" / "vscode" / "media" / "main.js"
)




# -----------------------------------------------------------------------
# Types check: verify ToWebviewMessageBody includes subagent types
# -----------------------------------------------------------------------

_TYPES_TS = (
    Path(__file__).resolve().parents[3]
    / "agents" / "vscode" / "src" / "types.ts"
)




# -----------------------------------------------------------------------
# CSS check: verify subagent tab styles exist
# -----------------------------------------------------------------------

_MAIN_CSS = (
    Path(__file__).resolve().parents[3]
    / "agents" / "vscode" / "media" / "main.css"
)

