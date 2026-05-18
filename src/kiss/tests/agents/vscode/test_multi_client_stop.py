"""Integration test: stop button from a subscriber (multi-viewer) tab.

When a second browser/client opens a running task from history,
``_replay_session`` subscribes the new tab to the source tab's event
stream via ``printer.subscribe_tab(source_tab_id, viewer_tab_id)``.

If the viewer tab clicks "Stop", ``_stop_task(viewer_tab_id)`` must
resolve through the subscriber mapping to find the source tab's
``stop_event`` and ``task_thread``, set the event, and force-stop the
thread — otherwise the stop is silently ignored (the viewer's own
``_RunningAgentState`` has ``stop_event=None`` / ``task_thread=None``)
and the result panel is never shown to the viewer.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import unittest
from typing import Any

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.agents.vscode.server import VSCodeServer

    return VSCodeServer()


class TestMultiClientStopResolvesSubscriber(unittest.TestCase):
    """Clicking Stop on a subscriber tab must stop the source tab's task."""

    def test_stop_from_viewer_tab_stops_source_task(self) -> None:
        """A subscriber tab's stop command must reach the source task.

        Setup:
          - source_tab runs a task (has stop_event, task_thread)
          - viewer_tab is subscribed to source_tab via printer.subscribe_tab
          - viewer_tab has its own _RunningAgentState but no stop_event

        Action:
          - _stop_task(viewer_tab_id) is called

        Assert:
          - The source task's stop_event is set
          - The task thread exits
          - A result event with success=False is broadcast
          - A task_stopped event is broadcast
        """
        server = _make_server()
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        orig_broadcast = server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with lock:
                events.append(dict(e))
            orig_broadcast(e)

        server.printer.broadcast = capture  # type: ignore[assignment]

        source_tab_id = "source-tab"
        viewer_tab_id = "viewer-tab"

        # Set up the source tab with a running task
        source_tab = server._get_tab(source_tab_id)
        source_tab.agent = WorktreeSorcarAgent("Sorcar VS Code")

        # The task blocks until stop_event is set
        task_started = threading.Event()

        def blocking_run(**kwargs: Any) -> None:
            source_tab.agent.total_tokens_used = 999
            source_tab.agent.budget_used = 0.03
            source_tab.agent.step_count = 5
            task_started.set()
            # Block until stopped
            while not source_tab.stop_event.is_set():
                time.sleep(0.01)
            raise KeyboardInterrupt("Stopped by user")

        source_tab.agent.run = blocking_run  # type: ignore[assignment]

        source_tab.stop_event = threading.Event()
        source_tab.user_answer_queue = queue.Queue()

        task_thread = threading.Thread(
            target=server._run_task,
            args=({"type": "run", "prompt": "long task", "tabId": source_tab_id},),
            daemon=True,
        )
        source_tab.task_thread = task_thread
        task_thread.start()

        # Wait for the task to actually start running
        assert task_started.wait(timeout=5), "Task did not start in time"

        # Set up the viewer tab (simulates _replay_session subscribing)
        viewer_tab = server._get_tab(viewer_tab_id)
        # The viewer has no stop_event or task_thread — it's just a viewer
        assert viewer_tab.stop_event is None
        assert viewer_tab.task_thread is None

        # Subscribe viewer to source (as _replay_session does)
        server.printer.subscribe_tab(source_tab_id, viewer_tab_id)

        # Now the viewer clicks "Stop" — this should resolve to the
        # source tab and stop it.
        server._stop_task(viewer_tab_id)

        # The source task thread should exit
        task_thread.join(timeout=10)
        assert not task_thread.is_alive(), "Source task thread should have been stopped"

        with lock:
            result_events = [e for e in events if e.get("type") == "result"]
            stopped_events = [e for e in events if e.get("type") == "task_stopped"]
            status_false_events = [
                e for e in events
                if e.get("type") == "status" and e.get("running") is False
            ]

        assert len(result_events) >= 1, (
            f"Expected a result event, got {len(result_events)}. "
            f"All event types: {[e.get('type') for e in events]}"
        )
        result_ev = result_events[-1]
        assert result_ev.get("success") is False, (
            f"Result should have success=False, got {result_ev.get('success')}"
        )
        assert "stopped" in (result_ev.get("text") or "").lower(), (
            f"Result text should mention 'stopped', got: {result_ev.get('text')}"
        )

        assert len(stopped_events) >= 1, (
            f"Expected task_stopped event. Events: {[e.get('type') for e in events]}"
        )

        assert len(status_false_events) >= 1, (
            "Expected status running=False broadcast"
        )

    def test_stop_from_viewer_shows_result_panel_tokens_and_cost(self) -> None:
        """The result event emitted after a viewer-tab stop must include
        token count, cost, and step count so the result panel renders
        correctly for the viewer."""
        server = _make_server()
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        orig_broadcast = server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with lock:
                events.append(dict(e))
            orig_broadcast(e)

        server.printer.broadcast = capture  # type: ignore[assignment]

        source_tab_id = "src-2"
        viewer_tab_id = "view-2"

        source_tab = server._get_tab(source_tab_id)
        source_tab.agent = WorktreeSorcarAgent("Sorcar VS Code")

        task_started = threading.Event()

        def blocking_run(**kwargs: Any) -> None:
            source_tab.agent.total_tokens_used = 4200
            source_tab.agent.budget_used = 0.15
            source_tab.agent.step_count = 12
            task_started.set()
            while not source_tab.stop_event.is_set():
                time.sleep(0.01)
            raise KeyboardInterrupt("Stopped")

        source_tab.agent.run = blocking_run  # type: ignore[assignment]
        source_tab.stop_event = threading.Event()
        source_tab.user_answer_queue = queue.Queue()

        task_thread = threading.Thread(
            target=server._run_task,
            args=({"type": "run", "prompt": "test", "tabId": source_tab_id},),
            daemon=True,
        )
        source_tab.task_thread = task_thread
        task_thread.start()
        assert task_started.wait(timeout=5)

        server._get_tab(viewer_tab_id)
        server.printer.subscribe_tab(source_tab_id, viewer_tab_id)

        server._stop_task(viewer_tab_id)
        task_thread.join(timeout=10)

        with lock:
            result_events = [e for e in events if e.get("type") == "result"]

        assert len(result_events) >= 1
        r = result_events[-1]
        assert r.get("total_tokens") == 4200, f"Got tokens={r.get('total_tokens')}"
        assert "$0.15" in str(r.get("cost", "")), f"Got cost={r.get('cost')}"
        assert r.get("step_count") == 12, f"Got steps={r.get('step_count')}"

    def test_stop_still_works_for_direct_tab(self) -> None:
        """Stopping from the original (non-subscriber) tab still works
        as before — no regression from the subscriber-resolution logic."""
        server = _make_server()

        tab_id = "direct-tab"
        tab = server._get_tab(tab_id)
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")

        task_started = threading.Event()

        def blocking_run(**kwargs: Any) -> None:
            tab.agent.total_tokens_used = 100
            tab.agent.budget_used = 0.01
            tab.agent.step_count = 2
            task_started.set()
            while not tab.stop_event.is_set():
                time.sleep(0.01)
            raise KeyboardInterrupt("Stopped")

        tab.agent.run = blocking_run  # type: ignore[assignment]
        tab.stop_event = threading.Event()
        tab.user_answer_queue = queue.Queue()

        task_thread = threading.Thread(
            target=server._run_task,
            args=({"type": "run", "prompt": "test", "tabId": tab_id},),
            daemon=True,
        )
        tab.task_thread = task_thread
        task_thread.start()
        assert task_started.wait(timeout=5)

        server._stop_task(tab_id)
        task_thread.join(timeout=10)
        assert not task_thread.is_alive(), "Direct tab stop should still work"

    def test_stop_from_viewer_when_no_subscription_is_noop(self) -> None:
        """If a tab with no stop_event and no subscription sends stop,
        nothing crashes — it's a graceful no-op."""
        server = _make_server()
        orphan_tab = server._get_tab("orphan")
        assert orphan_tab.stop_event is None

        # Should not raise
        server._stop_task("orphan")

    def test_stop_from_unknown_tab_is_noop(self) -> None:
        """Stopping an unknown tab_id is a no-op (pre-existing behavior)."""
        server = _make_server()
        server._stop_task("nonexistent")


if __name__ == "__main__":
    unittest.main()
