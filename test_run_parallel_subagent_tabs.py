"""Integration test for run_parallel subagent tabs feature.

Tests that:
1. openSubagentTab events are broadcast for each sub-task
2. Streaming events have the correct subagent tab_ids
3. subagentDone events are broadcast when sub-tasks complete
4. Subagent tabs are NOT counted in MAX_TABS trimming logic
"""

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.printer import Printer


class MockPrinter(Printer):
    """Mock printer that captures broadcast events."""

    def __init__(self):
        """Initialize the mock printer."""
        super().__init__()
        self.broadcast_events: list[dict[str, Any]] = []
        self._thread_local = threading.local()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Capture broadcast event."""
        self.broadcast_events.append(event)

    def print(self, text: str, type: str = "text", **kwargs: Any) -> str:
        """Mock print method."""
        return ""

    def token_callback(self, token: str) -> None:
        """Mock token callback method."""
        pass

    def reset(self) -> None:
        """Mock reset method."""
        pass


def test_run_parallel_broadcasts_subagent_tab_events() -> None:
    """Test that run_parallel broadcasts openSubagentTab and subagentDone events."""
    printer = MockPrinter()
    agent = SorcarAgent("test_agent")
    agent.printer = printer

    tasks = [
        "Return success message 1",
        "Return success message 2",
    ]

    # Mock SorcarAgent.run to avoid actual LLM calls but let broadcast code run
    with patch.object(SorcarAgent, "run") as mock_run:
        mock_run.return_value = "success: true\nsummary: task done"

        # Call _run_tasks_parallel (instance method)
        results = agent._run_tasks_parallel(tasks, max_workers=2)

    # Verify results were returned
    assert len(results) == 2
    assert all("success" in r for r in results)

    # Verify broadcast events were sent
    broadcast_events = printer.broadcast_events

    # Should have openSubagentTab events
    open_events = [
        e for e in broadcast_events
        if e.get("type") == "openSubagentTab"
    ]
    assert len(open_events) >= len(tasks), (
        f"Expected at least {len(tasks)} openSubagentTab events, "
        f"got {len(open_events)}"
    )

    # Should have subagentDone events
    done_events = [
        e for e in broadcast_events
        if e.get("type") == "subagentDone"
    ]
    assert len(done_events) >= len(tasks), (
        f"Expected at least {len(tasks)} subagentDone events, "
        f"got {len(done_events)}"
    )

    # Verify tab_ids are unique and match between open and done events
    open_tab_ids = {e.get("tab_id") for e in open_events}
    done_tab_ids = {e.get("tab_id") for e in done_events}
    assert len(open_tab_ids) == len(tasks)
    assert open_tab_ids == done_tab_ids


def test_subagent_tab_ids_match_task_index() -> None:
    """Test that subagent tab_ids follow the pattern {parent_tab_id}__sub_{i}."""
    printer = MockPrinter()
    agent = SorcarAgent("test_agent")
    agent.printer = printer
    # Simulate a parent tab_id
    parent_tab_id = "tab_123"
    printer._thread_local.tab_id = parent_tab_id

    tasks = ["Task 1", "Task 2", "Task 3"]

    with patch.object(SorcarAgent, "run") as mock_run:
        mock_run.return_value = "success: true\nsummary: done"
        agent._run_tasks_parallel(tasks, max_workers=3)

    open_events = [
        e for e in printer.broadcast_events
        if e.get("type") == "openSubagentTab"
    ]

    # Verify tab_ids follow the naming pattern
    for i, event in enumerate(open_events[:len(tasks)]):
        expected_tab_id = f"{parent_tab_id}__sub_{i}"
        assert event.get("tab_id") == expected_tab_id, (
            f"Expected tab_id {expected_tab_id}, got {event.get('tab_id')}"
        )


def test_subagent_tabs_excluded_from_max_tabs_trimming() -> None:
    """Test that subagent tabs don't count toward MAX_TABS limit.

    This is a frontend concern, but we verify the isSubagentTab flag
    is set in the openSubagentTab event.
    """
    printer = MockPrinter()
    agent = SorcarAgent("test_agent")
    agent.printer = printer

    tasks = ["Task A", "Task B"]

    with patch.object(SorcarAgent, "run") as mock_run:
        mock_run.return_value = "success: true\nsummary: done"
        agent._run_tasks_parallel(tasks)

    open_events = [
        e for e in printer.broadcast_events
        if e.get("type") == "openSubagentTab"
    ]

    # All openSubagentTab events should have isSubagentTab: true
    for event in open_events:
        assert event.get("isSubagentTab") is True, (
            f"Expected isSubagentTab=True in event: {event}"
        )


def test_open_subagent_tab_event_contains_required_fields() -> None:
    """Test that openSubagentTab events have all required fields."""
    printer = MockPrinter()
    agent = SorcarAgent("test_agent")
    agent.printer = printer

    tasks = ["Test task description"]

    with patch.object(SorcarAgent, "run") as mock_run:
        mock_run.return_value = "success: true\nsummary: done"
        agent._run_tasks_parallel(tasks)

    open_events = [
        e for e in printer.broadcast_events
        if e.get("type") == "openSubagentTab"
    ]

    assert len(open_events) > 0
    event = open_events[0]

    # Verify required fields
    required_fields = ["type", "tab_id", "parent_tab_id", "description", "isSubagentTab"]
    for field in required_fields:
        assert field in event, (
            f"Missing required field '{field}' in openSubagentTab event: {event}"
        )

    # Verify field types and values
    assert event["type"] == "openSubagentTab"
    assert isinstance(event["tab_id"], str)
    assert isinstance(event["description"], str)
    assert event["isSubagentTab"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
