# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for bugs, redundancies, and inconsistencies in
``kiss.server`` — updated to verify the fixes.

Bugs
----
B1: ``_cmd_run`` queues a duplicate ``run`` while a task is already
    starting and echoes the accepted follow-up without spawning a thread.
B2: ``_close_tab`` refuses to remove the state of a tab whose task
    thread is installed or alive (``AgentState.busy()``).
"""

from __future__ import annotations

import threading
import unittest

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _register_tab_state(
    task_id: str,
    tab_id: str,
    *,
    agent: WorktreeSorcarAgent | None = None,
) -> agent_state.AgentState:
    """Register a server-owned AgentState for *tab_id* and return it."""
    state = agent_state.AgentState(
        task_id,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
    )
    agent_state.register(state)
    return state


class TestCmdRunQueuesFollowup(unittest.TestCase):
    """A second ``run`` during startup is queued on the live task.

    It is echoed as accepted input without spawning a replacement thread.
    """

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_duplicate_run_is_queued_and_echoed_while_task_is_alive(self) -> None:
        state = _register_tab_state(
            "audit-b1", "t1", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        thread.start()
        state.task_thread = thread

        events_before = len(self.events)
        self.server._handle_command({"type": "run", "tabId": "t1", "prompt": "x"})

        new_events = self.events[events_before:]
        assert thread.is_alive()
        assert state.task_thread is thread
        assert state.pending_user_messages == ["x"]
        assert state.unattributed_prompt_echoes == ["x"]
        # ``_cmd_run`` unconditionally mirrors the task-panel text to
        # every client (the ``setTaskText`` submit acknowledgment)
        # before echoing the queued prompt — also for a queued
        # follow-up, so all run origins behave identically.
        assert new_events == [
            {"type": "setTaskText", "text": "x", "tabId": "t1"},
            {"type": "prompt", "text": "x", "tabId": "t1"},
        ]

        blocker.set()
        thread.join(timeout=2)


class TestCloseTabRaceWithTaskStartup(unittest.TestCase):
    """B2 fix: ``_close_tab`` refuses to remove a busy state — a task
    thread that is installed (even before ``Thread.start()``) or alive
    keeps the state registered.
    """

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_close_tab_refuses_when_task_thread_installed_before_start(
        self,
    ) -> None:
        """An installed worker is busy even before ``Thread.start()``."""
        state = _register_tab_state("audit-b2a", "t1")
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        state.task_thread = thread
        state.is_task_active = False

        self.server._close_tab("t1")

        assert agent_state.get("audit-b2a") is state
        assert state.frontend_closed
        blocker.set()
        thread.start()
        thread.join(timeout=2)

    def test_close_tab_refuses_when_task_thread_alive(self) -> None:
        state = _register_tab_state(
            "audit-b2b", "t1", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        thread.start()
        state.task_thread = thread
        state.is_task_active = False

        self.server._close_tab("t1")

        assert agent_state.get("audit-b2b") is state, (
            "B2 fix: state should NOT be removed while task_thread is alive"
        )

        blocker.set()
        thread.join(timeout=2)


class TestCmdRunFollowupNoErrorBroadcast(unittest.TestCase):
    """An accepted follow-up emits a prompt echo, but no error or status."""

    def setUp(self) -> None:
        self.server, self.events = _make_server()

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_no_error_or_status_for_alive_task(self) -> None:
        state = _register_tab_state(
            "audit-b1b", "t1", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        blocker = threading.Event()
        thread = threading.Thread(target=blocker.wait, daemon=True)
        thread.start()
        state.task_thread = thread

        events_before = len(self.events)
        self.server._handle_command({"type": "run", "tabId": "t1", "prompt": "x"})

        new_events = self.events[events_before:]
        assert not any(e.get("type") == "error" for e in new_events)
        assert not any(e.get("type") == "status" for e in new_events)

        blocker.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
