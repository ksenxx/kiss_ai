# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""C-RC1: viewer gates must use ``busy()``, not ``is_task_active``.

``_cmd_run`` installs ``state.task_thread`` under ``_state_lock`` and
the worker raises ``is_task_active`` only after the thread starts
(the documented S3-05 run-startup window).  Two viewer-fan-out sites
gated on ``is_task_active`` alone:

* ``_subscribe_chat_viewers`` — a tab whose OWN task was in the
  startup window got wrongly subscribed to ANOTHER task's stream
  (its content cleared, its status hijacked);
* ``_broadcast_status_end_to_viewers`` — such a tab received
  ``running=False`` for its FRESH task when a previously viewed task
  ended, re-enabling its composer mid-run.

Both must treat the startup window as busy via
``AgentState.busy()`` / ``thread_alive()`` (a created-but-unstarted
thread counts as alive).

The startup window is reproduced with real registry state: a
registered ``AgentState`` carrying a created-but-not-started
``threading.Thread``, exactly what ``_cmd_run`` leaves behind between
installing and starting the worker.  No mocks; the positive controls
prove idle viewers still get the fan-out.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer


def _noop() -> None:
    """Target for never-started placeholder worker threads."""


class _ViewerGateTestBase(unittest.TestCase):
    """Shared server + event-recorder setup."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def recording_broadcast(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = recording_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def _events_for_tab(self, tab_id: str) -> list[dict[str, Any]]:
        with self._events_lock:
            return [e for e in self.events if e.get("tabId") == tab_id]

    def _register_starting_state(
        self, task_key: str, chat_id: str, tab_id: str,
    ) -> AgentState:
        """Register a state exactly as _cmd_run leaves it pre-start."""
        state = AgentState(
            task_key,
            chat_id=chat_id,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
            task_thread=threading.Thread(target=_noop, daemon=True),
        )
        agent_state.register(state)
        return state

    def _register_idle_state(
        self, task_key: str, chat_id: str, tab_id: str,
    ) -> AgentState:
        """Register a finished/idle state (no thread, not active)."""
        state = AgentState(
            task_key,
            chat_id=chat_id,
            tab_id=tab_id,
            server_owned=True,
        )
        agent_state.register(state)
        return state


class TestSubscribeChatViewersGate(_ViewerGateTestBase):
    """_subscribe_chat_viewers must skip tabs in the startup window."""

    def test_starting_viewer_not_hijacked(self) -> None:
        """A viewer whose own task is starting is not subscribed."""
        chat_id = "chat-rc1"
        starting_tab = "starting-tab"
        idle_tab = "idle-tab"
        with self.server._state_lock:
            self.server._tab_chat_views[starting_tab] = chat_id
            self.server._tab_chat_views[idle_tab] = chat_id
        self._register_starting_state("starting-key", chat_id, starting_tab)
        self._register_idle_state("idle-key", chat_id, idle_tab)

        self.server._subscribe_chat_viewers(
            "other-task-id",
            chat_id,
            source_tab_id="launcher-tab",
            start_ms=123,
        )

        hijacked = self._events_for_tab(starting_tab)
        assert not hijacked, (
            "BUG C-RC1: a tab whose own task is in the run-startup "
            f"window was subscribed to another task's stream: {hijacked}"
        )
        with self.server.printer._lock:
            viewers = self.server.printer._subscribers.get(
                "other-task-id", set(),
            )
        assert starting_tab not in viewers

        # Positive control: the genuinely idle viewer still gets the
        # clear + running=True sequence and the subscription.
        idle_events = self._events_for_tab(idle_tab)
        types = [e.get("type") for e in idle_events]
        assert types == ["clear", "status"], types
        assert idle_events[1].get("running") is True
        assert idle_tab in viewers


class TestStatusEndViewerGate(_ViewerGateTestBase):
    """_broadcast_status_end_to_viewers must skip starting viewers."""

    def test_starting_viewer_keeps_running_status(self) -> None:
        """No running=False for a viewer whose fresh task is starting."""
        finished_task = "finished-task-id"
        starting_tab = "starting-tab-2"
        idle_tab = "idle-tab-2"
        # Both tabs were subscribed viewers of the finished task.
        self.server.printer.subscribe_tab(finished_task, starting_tab)
        self.server.printer.subscribe_tab(finished_task, idle_tab)
        # The starting tab meanwhile launched its OWN fresh task
        # (different task id, thread installed but not started).
        self._register_starting_state(
            "fresh-task-key", "some-chat", starting_tab,
        )
        self._register_idle_state("idle-key-2", "some-chat", idle_tab)

        self.server._broadcast_status_end_to_viewers(
            finished_task, "launcher-tab",
        )

        starting_events = self._events_for_tab(starting_tab)
        assert not starting_events, (
            "BUG C-RC1: a viewer whose own fresh task is in the "
            "run-startup window received running=False for it: "
            f"{starting_events}"
        )
        # Positive control: the idle viewer still gets running=False.
        idle_status = [
            e
            for e in self._events_for_tab(idle_tab)
            if e.get("type") == "status"
        ]
        assert len(idle_status) == 1
        assert idle_status[0].get("running") is False

    def test_own_task_end_still_reaches_viewer(self) -> None:
        """running=False for the viewer's OWN task id is not skipped.

        The gate skips a busy viewer only when its task id differs
        from the ending task's — the launcher semantics must be
        preserved for a viewer that owns the ending task.
        """
        ending_task = "ending-task-id"
        viewer_tab = "owning-viewer-tab"
        self.server.printer.subscribe_tab(ending_task, viewer_tab)
        # The viewer's own (still-busy-looking) state IS the ending
        # task: is_task_active raised, same task id.
        state = self._register_idle_state(
            ending_task, "chat-x", viewer_tab,
        )
        state.is_task_active = True

        self.server._broadcast_status_end_to_viewers(
            ending_task, "launcher-tab",
        )
        status = [
            e
            for e in self._events_for_tab(viewer_tab)
            if e.get("type") == "status"
        ]
        assert len(status) == 1 and status[0].get("running") is False


if __name__ == "__main__":
    unittest.main()
