# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group C): viewer fan-out must not corrupt a busy tab.

Two tabs can legitimately have the SAME chat open (e.g. the user opened
one chat from the history sidebar in two windows) and each tab can run
its own task concurrently — ``_cmd_run`` guards per TAB, not per chat.

BUG-TR8-2 — when tab A starts a task on chat C,
``_subscribe_chat_viewers`` broadcasts ``clear`` followed by
``status running=True (startTs=A's start)`` to EVERY other tab viewing
chat C — including a tab B that is actively streaming its OWN task's
events.  The ``clear`` wipes B's live transcript mid-task and the
``status`` re-anchors B's "Running …" timer to A's start time.

BUG-TR8-3 — symmetrically, when A's task ends,
``_broadcast_status_end_to_viewers`` sends ``status running=False`` to
every subscribed viewer — including a tab B that is still running its
own task.  B's frontend (statuses are stamped with B's own ``tabId``)
stops its spinner and flips its input box out of "queue follow-up"
mode while B's agent is still working.

Both tests use the real ``AgentServer``, real ``JsonPrinter``
subscription state and real ``kiss.server.agent_state`` registry
entries; events are captured by replacing ``printer.broadcast`` with a
recorder (the established pattern of the earlier bug-hunt tests).
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer


class _EventRecorder:
    """Thread-safe list of broadcast events."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)

    def for_tab(self, tab_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self.events if e.get("tabId") == tab_id]


class _ParkingPrinter(JsonPrinter):
    """Parks the thread named ``walker`` once it has read a task's viewers."""

    def __init__(self) -> None:
        super().__init__()
        self.walker: threading.Thread | None = None
        self.parked = threading.Event()
        self.release = threading.Event()

    def _fanout_targets(self, task_id: Any) -> list[str]:
        targets = super()._fanout_targets(task_id)
        if threading.current_thread() is self.walker:
            self.parked.set()
            self.release.wait(10)
        return targets


def _register_busy_viewer(tab_id: str, task_id: str) -> AgentState:
    """Register a server-owned state for a tab running its own task."""
    state = AgentState(
        task_id, tab_id=tab_id, server_owned=True, is_task_active=True,
    )
    agent_state.register(state)
    return state


class TestSubscribeChatViewersSkipsBusyViewer(unittest.TestCase):
    """BUG-TR8-2: a viewer running its own task must not be cleared."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.server = AgentServer()
        self.recorder = _EventRecorder()

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_busy_viewer_not_cleared_or_flipped(self) -> None:
        launcher = "bh8c-chat-launcher"
        busy_viewer = "bh8c-chat-busy"
        idle_viewer = "bh8c-chat-idle"
        chat_id = "bh8c-chat-c"

        _register_busy_viewer(busy_viewer, "7999")

        with self.server._state_lock:
            self.server._tab_chat_views[launcher] = chat_id
            self.server._tab_chat_views[busy_viewer] = chat_id
            self.server._tab_chat_views[idle_viewer] = chat_id

        self.server.printer.broadcast = self.recorder  # type: ignore[assignment]
        self.server._subscribe_chat_viewers(
            "7301", chat_id, source_tab_id=launcher, start_ms=123,
        )

        busy_events = self.recorder.for_tab(busy_viewer)
        self.assertEqual(
            busy_events,
            [],
            "BUG-TR8-2: _subscribe_chat_viewers cleared / re-anchored a "
            "viewer tab that is actively streaming its OWN task: "
            f"{busy_events}",
        )

        idle_types = [e.get("type") for e in self.recorder.for_tab(idle_viewer)]
        self.assertEqual(idle_types, ["clear", "status"])


class TestStatusEndSkipsBusyViewer(unittest.TestCase):
    """BUG-TR8-3: running=False must not reach a still-busy viewer."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.server = AgentServer()
        self.recorder = _EventRecorder()

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_busy_viewer_spinner_not_stopped(self) -> None:
        launcher = "bh8c-end-launcher"
        busy_viewer = "bh8c-end-busy"
        idle_viewer = "bh8c-end-idle"

        _register_busy_viewer(busy_viewer, "7999")

        self.server.printer.subscribe_tab("7301", launcher)
        self.server.printer.subscribe_tab("7301", busy_viewer)
        self.server.printer.subscribe_tab("7301", idle_viewer)

        self.server.printer.broadcast = self.recorder  # type: ignore[assignment]
        self.server._broadcast_status_end_to_viewers("7301", launcher, "")

        busy_events = self.recorder.for_tab(busy_viewer)
        self.assertEqual(
            busy_events,
            [],
            "BUG-TR8-3: _broadcast_status_end_to_viewers stopped the "
            "spinner of a viewer tab whose OWN task is still running: "
            f"{busy_events}",
        )

        idle_events = self.recorder.for_tab(idle_viewer)
        self.assertEqual(len(idle_events), 1)
        self.assertEqual(idle_events[0].get("type"), "status")
        self.assertIs(idle_events[0].get("running"), False)

    def test_viewer_is_told_how_the_task_ended(self) -> None:
        """A viewer's tab gets its own copy of the run's end, ahead of its status.

        The launcher's event names the launcher's tab only: without a
        copy of its own a viewer reads ``Done`` for a stopped task and
        keeps the question the task left open, and never opens the
        reports of a finished one.  The copy is marked ``viewer``: a
        page sounds its jingle for the launcher's event alone.
        """
        launcher = "bh8c-fail-launcher"
        busy_viewer = "bh8c-fail-busy"
        idle_viewer = "bh8c-fail-idle"
        _register_busy_viewer(busy_viewer, "7999")
        for tab in (launcher, busy_viewer, idle_viewer):
            self.server.printer.subscribe_tab("7302", tab)
        self.server.printer.broadcast = self.recorder  # type: ignore[assignment]
        stopped = {"type": "task_stopped", "startTs": 1, "endTs": 2, "subagent": False}
        self.server._broadcast_status_end_to_viewers("7302", launcher, "", end_event=stopped)
        self.server._broadcast_status_end_to_viewers(
            "7302", launcher, "", end_event={**stopped, "type": "task_done"},
        )
        self.assertEqual(self.recorder.for_tab(busy_viewer), [])
        self.assertEqual(self.recorder.for_tab(launcher), [])
        self.assertEqual(
            self.recorder.for_tab(idle_viewer),
            [
                {**stopped, "tabId": idle_viewer, "viewer": True},
                {"type": "status", "running": False, "tabId": idle_viewer},
                {**stopped, "type": "task_done", "tabId": idle_viewer, "viewer": True},
                {"type": "status", "running": False, "tabId": idle_viewer},
            ],
        )

    def test_viewer_that_left_the_run_meanwhile_is_told_nothing(self) -> None:
        """The walk reads the run's viewers, then a replay moves one to another chat.

        That tab shows the other chat by the time its turn comes: the
        old run's end would label it Stopped.
        """
        printer = _ParkingPrinter()
        server = AgentServer(printer=printer)
        viewer = "bh8c-moved-viewer"
        self.addCleanup(server._close_tab, viewer)
        _other_task, other_chat = th._add_task("A finished task of another chat")
        printer.subscribe_tab("7303", viewer)
        printer.broadcast = self.recorder  # type: ignore[assignment]
        printer.walker = threading.Thread(
            target=server._broadcast_status_end_to_viewers,
            args=("7303", "bh8c-moved-launcher", ""),
            kwargs={"end_event": {"type": "task_stopped", "startTs": 1, "endTs": 2}},
            daemon=True,
        )
        printer.walker.start()
        self.assertTrue(printer.parked.wait(10))
        server._cmd_resume_session({"chatId": other_chat, "tabId": viewer})
        shown = self.recorder.for_tab(viewer)
        printer.release.set()
        printer.walker.join(10)
        self.assertFalse(printer.walker.is_alive())
        self.assertEqual(self.recorder.for_tab(viewer), shown)


if __name__ == "__main__":
    unittest.main()
