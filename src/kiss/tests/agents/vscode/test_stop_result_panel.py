"""Integration tests: stopped/killed agent emits a result-panel event.

When a user stops an agent (KeyboardInterrupt) or the agent crashes
(Exception), the backend must broadcast a ``result`` event with
``success=False`` so the frontend displays the error/stop message in
the result panel — the same rich panel used for successful completions.

Previously, stop/error only emitted ``task_stopped`` / ``task_error``
banner events, which rendered as small inline banners rather than the
full result panel with tokens, cost, and step count.
"""

from __future__ import annotations

import os
import queue
import threading
import unittest
from typing import Any
from unittest import TestCase


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.agents.vscode.server import VSCodeServer

    return VSCodeServer()


class TestStoppedTaskEmitsResultEvent(TestCase):
    """When a task is stopped via KeyboardInterrupt, a ``result`` event
    with ``success=False`` must appear in the broadcast stream."""

    def test_keyboard_interrupt_produces_result_event(self) -> None:
        """Simulate a task that raises KeyboardInterrupt and verify a
        ``result`` event with success=False is broadcast."""
        server = _make_server()
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        orig_broadcast = server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with lock:
                events.append(dict(e))
            orig_broadcast(e)

        server.printer.broadcast = capture  # type: ignore[assignment]

        tab_id = "stop-test-1"
        tab = server._get_tab(tab_id)

        # Make the agent's run() raise KeyboardInterrupt immediately
        def fake_run(**kwargs: Any) -> None:
            # Simulate some token usage before stopping
            tab.agent.total_tokens_used = 1234
            tab.agent.budget_used = 0.05
            tab.agent.step_count = 7
            raise KeyboardInterrupt("Stopped by user")

        tab.agent.run = fake_run  # type: ignore[assignment]

        stop_event = threading.Event()
        tab.stop_event = stop_event
        tab.user_answer_queue = queue.Queue()

        task_thread = threading.Thread(
            target=server._run_task,
            args=({"type": "run", "prompt": "test task", "tabId": tab_id},),
            daemon=True,
        )
        tab.task_thread = task_thread
        task_thread.start()
        task_thread.join(timeout=10)

        with lock:
            result_events = [e for e in events if e.get("type") == "result"]
            stopped_events = [e for e in events if e.get("type") == "task_stopped"]

        assert len(result_events) >= 1, (
            f"Expected at least one result event, got {len(result_events)}. "
            f"All events: {[e.get('type') for e in events]}"
        )
        result_ev = result_events[-1]
        assert result_ev.get("success") is False, (
            f"Result event should have success=False, got {result_ev.get('success')}"
        )
        assert "stopped" in (result_ev.get("text") or "").lower(), (
            f"Result text should mention 'stopped', got: {result_ev.get('text')}"
        )
        # Should include token/cost/step info
        assert result_ev.get("total_tokens") == 1234, (
            f"Expected total_tokens=1234, got {result_ev.get('total_tokens')}"
        )
        assert "$0.05" in str(result_ev.get("cost", "")), (
            f"Expected cost containing '$0.05', got {result_ev.get('cost')}"
        )
        assert result_ev.get("step_count") == 7, (
            f"Expected step_count=7, got {result_ev.get('step_count')}"
        )

        # The task_stopped event should still be broadcast (for status bar)
        assert len(stopped_events) >= 1, (
            f"Expected task_stopped event too. Events: {[e.get('type') for e in events]}"
        )

    def test_exception_produces_result_event(self) -> None:
        """Simulate a task that raises an Exception and verify a
        ``result`` event with success=False is broadcast."""
        server = _make_server()
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        orig_broadcast = server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with lock:
                events.append(dict(e))
            orig_broadcast(e)

        server.printer.broadcast = capture  # type: ignore[assignment]

        tab_id = "error-test-1"
        tab = server._get_tab(tab_id)

        def fake_run(**kwargs: Any) -> None:
            tab.agent.total_tokens_used = 500
            tab.agent.budget_used = 0.02
            tab.agent.step_count = 3
            raise RuntimeError("Model API connection failed")

        tab.agent.run = fake_run  # type: ignore[assignment]

        stop_event = threading.Event()
        tab.stop_event = stop_event
        tab.user_answer_queue = queue.Queue()

        task_thread = threading.Thread(
            target=server._run_task,
            args=({"type": "run", "prompt": "test task", "tabId": tab_id},),
            daemon=True,
        )
        tab.task_thread = task_thread
        task_thread.start()
        task_thread.join(timeout=10)

        with lock:
            result_events = [e for e in events if e.get("type") == "result"]
            error_events = [e for e in events if e.get("type") == "task_error"]

        assert len(result_events) >= 1, (
            f"Expected at least one result event, got {len(result_events)}. "
            f"All events: {[e.get('type') for e in events]}"
        )
        result_ev = result_events[-1]
        assert result_ev.get("success") is False
        assert "Model API connection failed" in (result_ev.get("text") or ""), (
            f"Result text should contain error message, got: {result_ev.get('text')}"
        )
        assert result_ev.get("total_tokens") == 500
        assert result_ev.get("step_count") == 3

        # The task_error event should still be broadcast
        assert len(error_events) >= 1

    def test_successful_task_still_works(self) -> None:
        """A successful task should still emit its normal result event
        (no double result events from the error path)."""
        server = _make_server()
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        orig_broadcast = server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with lock:
                events.append(dict(e))
            orig_broadcast(e)

        server.printer.broadcast = capture  # type: ignore[assignment]

        tab_id = "success-test-1"
        tab = server._get_tab(tab_id)

        def fake_run(**kwargs: Any) -> None:
            printer = kwargs.get("printer", server.printer)
            tab.agent.total_tokens_used = 2000
            tab.agent.budget_used = 0.10
            tab.agent.step_count = 15
            # Simulate the agent emitting its own result event
            printer.print(
                "success: true\nsummary: Task completed successfully",
                type="result",
                total_tokens=2000,
                cost="$0.1000",
                step_count=15,
            )

        tab.agent.run = fake_run  # type: ignore[assignment]

        stop_event = threading.Event()
        tab.stop_event = stop_event
        tab.user_answer_queue = queue.Queue()

        task_thread = threading.Thread(
            target=server._run_task,
            args=({"type": "run", "prompt": "test task", "tabId": tab_id},),
            daemon=True,
        )
        tab.task_thread = task_thread
        task_thread.start()
        task_thread.join(timeout=10)

        with lock:
            result_events = [e for e in events if e.get("type") == "result"]

        # Should have exactly one result event (from the agent), not two
        assert len(result_events) == 1, (
            f"Expected exactly 1 result event for success, got {len(result_events)}. "
            f"Events: {result_events}"
        )
        # The success result should NOT have success=False
        assert result_events[0].get("success") is not False


class TestBudgetExceededResultPanel(TestCase):
    """When the agent's cost exceeds the user-configured max budget,
    the task must stop and a ``result`` event with ``success=False``
    containing 'budget exceeded' must appear in the broadcast stream
    so the frontend displays it in the result panel."""

    def test_budget_exceeded_produces_result_event(self) -> None:
        """Simulate an agent that raises KISSError for budget exceeded
        and verify a ``result`` event with success=False and 'budget
        exceeded' in the text is broadcast to the result panel."""
        from kiss.core.kiss_error import KISSError

        server = _make_server()
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        orig_broadcast = server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with lock:
                events.append(dict(e))
            orig_broadcast(e)

        server.printer.broadcast = capture  # type: ignore[assignment]

        tab_id = "budget-test-1"
        tab = server._get_tab(tab_id)

        def fake_run(**kwargs: Any) -> None:
            # Simulate the agent accumulating cost beyond the $1 budget
            tab.agent.total_tokens_used = 50000
            tab.agent.budget_used = 1.05
            tab.agent.step_count = 42
            raise KISSError("Agent budget-test budget exceeded.")

        tab.agent.run = fake_run  # type: ignore[assignment]

        stop_event = threading.Event()
        tab.stop_event = stop_event
        tab.user_answer_queue = queue.Queue()

        task_thread = threading.Thread(
            target=server._run_task,
            args=({"type": "run", "prompt": "test task", "tabId": tab_id},),
            daemon=True,
        )
        tab.task_thread = task_thread
        task_thread.start()
        task_thread.join(timeout=10)

        with lock:
            result_events = [e for e in events if e.get("type") == "result"]
            error_events = [e for e in events if e.get("type") == "task_error"]

        assert len(result_events) >= 1, (
            f"Expected at least one result event, got {len(result_events)}. "
            f"All events: {[e.get('type') for e in events]}"
        )
        result_ev = result_events[-1]
        assert result_ev.get("success") is False, (
            f"Result event should have success=False, got {result_ev.get('success')}"
        )
        assert "budget exceeded" in (result_ev.get("text") or "").lower(), (
            f"Result text should mention 'budget exceeded', got: {result_ev.get('text')}"
        )
        # Should include token/cost/step info
        assert result_ev.get("total_tokens") == 50000, (
            f"Expected total_tokens=50000, got {result_ev.get('total_tokens')}"
        )
        assert "$1.05" in str(result_ev.get("cost", "")), (
            f"Expected cost containing '$1.05', got {result_ev.get('cost')}"
        )
        assert result_ev.get("step_count") == 42, (
            f"Expected step_count=42, got {result_ev.get('step_count')}"
        )

        # The task_error event should also be broadcast
        assert len(error_events) >= 1, (
            f"Expected task_error event. Events: {[e.get('type') for e in events]}"
        )


if __name__ == "__main__":
    unittest.main()
