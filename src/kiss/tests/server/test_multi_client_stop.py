# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: stop button from a subscriber (multi-viewer) tab.

When a second browser/client opens a running task from history,
``_replay_session`` subscribes the new tab to the running task's event
stream via ``printer.subscribe_tab(task_id, viewer_tab_id)``.

If the viewer tab clicks "Stop", ``_stop_task(viewer_tab_id)`` must
resolve through the printer's per-task subscriber mapping to find the
task's ``stop_event`` and ``task_thread``, set the event, and
force-stop the thread — otherwise the stop is silently ignored (the
viewer tab owns no running ``AgentState``) and the result panel is
never shown to the viewer.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import unittest
from typing import Any

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.server import VSCodeServer

    return VSCodeServer()


def _start_source_task(
    server: Any,
    task_id: str,
    tab_id: str,
    *,
    tokens: int,
    cost: float,
    steps: int,
) -> tuple[agent_state.AgentState, threading.Thread, threading.Event]:
    """Register a server-owned state and run a blocking task on it.

    The agent's ``run`` is a deterministic stub that reports the given
    token/cost/step figures, then blocks until the state's
    ``stop_event`` is set and raises ``KeyboardInterrupt`` — the same
    observable behavior as a real agent interrupted by Stop.

    Returns the registered state, the started task thread, and an
    event set once the stubbed run is executing.
    """
    agent = WorktreeSorcarAgent("Sorcar VS Code")
    state = agent_state.AgentState(
        task_id,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
        stop_event=threading.Event(),
    )
    state.user_answer_queue = queue.Queue()
    stop_event = state.stop_event
    assert stop_event is not None
    task_started = threading.Event()

    def blocking_run(**kwargs: Any) -> None:
        agent.total_tokens_used = tokens
        agent.budget_used = cost
        agent.step_count = steps
        task_started.set()
        while not stop_event.is_set():
            time.sleep(0.01)
        raise KeyboardInterrupt("Stopped by user")

    agent.run = blocking_run  # type: ignore[assignment]
    agent_state.register(state)

    task_thread = threading.Thread(
        target=server._run_task,
        args=({"type": "run", "prompt": "long task", "tabId": tab_id},),
        daemon=True,
    )
    state.task_thread = task_thread
    task_thread.start()
    return state, task_thread, task_started


class TestMultiClientStopResolvesSubscriber(unittest.TestCase):
    """Clicking Stop on a subscriber tab must stop the source task."""

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_stop_from_viewer_tab_stops_source_task(self) -> None:
        """A subscriber tab's stop command must reach the source task.

        Setup:
          - a task runs on source-tab (registered AgentState with
            stop_event and task_thread)
          - viewer-tab is subscribed to the task via
            printer.subscribe_tab(task_id, viewer_tab_id) and owns no
            state of its own

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

        task_id = "mcs-task-1"
        source_tab_id = "source-tab"
        viewer_tab_id = "viewer-tab"

        _state, task_thread, task_started = _start_source_task(
            server, task_id, source_tab_id, tokens=999, cost=0.03, steps=5,
        )
        assert task_started.wait(timeout=5), "Task did not start in time"

        assert agent_state.find_by_tab(viewer_tab_id) is None

        server.printer.subscribe_tab(task_id, source_tab_id)
        server.printer.subscribe_tab(task_id, viewer_tab_id)

        server._stop_task(viewer_tab_id)

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

        task_id = "mcs-task-2"
        source_tab_id = "src-2"
        viewer_tab_id = "view-2"

        _state, task_thread, task_started = _start_source_task(
            server, task_id, source_tab_id, tokens=4200, cost=0.15, steps=12,
        )
        assert task_started.wait(timeout=5)

        server.printer.subscribe_tab(task_id, source_tab_id)
        server.printer.subscribe_tab(task_id, viewer_tab_id)

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

        task_id = "mcs-task-3"
        tab_id = "direct-tab"

        _state, task_thread, task_started = _start_source_task(
            server, task_id, tab_id, tokens=100, cost=0.01, steps=2,
        )
        assert task_started.wait(timeout=5)

        server._stop_task(tab_id)
        task_thread.join(timeout=10)
        assert not task_thread.is_alive(), "Direct tab stop should still work"

    def test_stop_from_viewer_when_no_subscription_is_noop(self) -> None:
        """If a tab with no running state and no subscription sends
        stop, nothing crashes — it's a graceful no-op."""
        server = _make_server()
        assert agent_state.find_by_tab("orphan") is None

        server._stop_task("orphan")

    def test_stop_from_unknown_tab_is_noop(self) -> None:
        """Stopping an unknown tab_id is a no-op (pre-existing behavior)."""
        server = _make_server()
        server._stop_task("nonexistent")

    def test_viewer_stop_after_source_tab_fully_closed_is_graceful_noop(
        self,
    ) -> None:
        """Source tab closes while viewer remains; subsequent viewer stop
        gracefully handles the orphaned state without errors.

        End-to-end flow:
          - a task runs on source_tab; viewer_tab subscribes via printer.
          - Source tab's frontend is closed while the task is running →
            ``_close_tab`` marks ``frontend_closed=True`` but keeps the
            ``AgentState`` alive so the running agent can finish (per
            the "Closing a chat tab does NOT stop a running agent"
            invariant).
          - The viewer clicks Stop → the subscriber-resolution path
            finds the still-registered task state and stops the task.
          - The task ends; ``_run_task``'s finally block invokes
            ``_dispose_if_closed`` which unregisters the state because
            ``frontend_closed=True``.
          - The viewer (still open in the frontend) clicks Stop a
            second time — the task's ``AgentState`` is gone and the
            lingering subscription resolves to nothing.  This MUST be
            a graceful no-op: no KeyError, no AttributeError, no
            thread spawn — but it is NOT silent: the tab is told its
            click found nothing to stop.
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

        task_id = "mcs-task-close"
        source_tab_id = "src-close"
        viewer_tab_id = "view-close"

        state, task_thread, task_started = _start_source_task(
            server, task_id, source_tab_id, tokens=50, cost=0.005, steps=1,
        )
        assert task_started.wait(timeout=5), "Task did not start in time"

        server.printer.subscribe_tab(task_id, source_tab_id)
        server.printer.subscribe_tab(task_id, viewer_tab_id)
        assert viewer_tab_id in server.printer._subscribers[task_id]
        assert source_tab_id in server.printer._subscribers[task_id]

        server._close_tab(source_tab_id)
        assert agent_state.get(task_id) is state
        assert state.frontend_closed is True
        assert viewer_tab_id in server.printer._subscribers[task_id]

        server._stop_task(viewer_tab_id)
        task_thread.join(timeout=10)
        assert not task_thread.is_alive(), (
            "Viewer stop must reach the source task even after the "
            "source frontend closed"
        )

        deadline = time.time() + 5.0
        while time.time() < deadline and agent_state.get(task_id) is not None:
            time.sleep(0.01)
        assert agent_state.get(task_id) is None, (
            "Source AgentState should be disposed after task end "
            "because frontend_closed=True"
        )

        events_before = len(events)
        server._stop_task(viewer_tab_id)
        # The orphaned stop is a no-op for the task, but it is NOT
        # silent: it tells the tab its click found nothing to stop.
        # Dropping it without a word is what made a mis-targeted click
        # indistinguishable from a task that had not reacted yet
        # (reports/stop_button_delay_2026-08-05.html).
        new_events = events[events_before:]
        assert new_events == [
            {"type": "stop_ack", "accepted": False, "tabId": viewer_tab_id},
        ], f"Stopping an orphaned viewer broadcast {new_events}"

    def test_stop_with_orphan_subscription_pointing_to_missing_source(
        self,
    ) -> None:
        """Forced-orphan test: a subscription entry points at a task id
        that has no registered ``AgentState``.

        ``_find_viewer_task_states`` sees the subscription, but
        ``agent_state.get(task_id)`` returns ``None``.  ``_stop_task``
        MUST treat this as a no-op rather than dereferencing the
        missing state.
        """
        server = _make_server()

        viewer_tab_id = "lonely-viewer"
        server.printer.subscribe_tab("ghost-task-id", viewer_tab_id)

        assert agent_state.get("ghost-task-id") is None
        assert viewer_tab_id in server.printer._subscribers.get(
            "ghost-task-id", set(),
        )

        server._stop_task(viewer_tab_id)


if __name__ == "__main__":
    unittest.main()
