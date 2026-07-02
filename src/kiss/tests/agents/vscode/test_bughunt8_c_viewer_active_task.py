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

Both tests use the real ``VSCodeServer``, real ``JsonPrinter``
subscription state and real ``_RunningAgentState`` registry entries;
events are captured by replacing ``printer.broadcast`` with a recorder
(the established pattern of the earlier bug-hunt tests).
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer


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


class TestSubscribeChatViewersSkipsBusyViewer(unittest.TestCase):
    """BUG-TR8-2: a viewer running its own task must not be cleared."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.server = VSCodeServer()
        self.recorder = _EventRecorder()

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_busy_viewer_not_cleared_or_flipped(self) -> None:
        launcher = "bh8c-chat-launcher"
        busy_viewer = "bh8c-chat-busy"
        idle_viewer = "bh8c-chat-idle"
        chat_id = "bh8c-chat-c"

        self.server._get_tab(launcher)
        busy_state = self.server._get_tab(busy_viewer)
        self.server._get_tab(idle_viewer)

        # All three tabs have chat C open.
        with self.server._state_lock:
            self.server._tab_chat_views[launcher] = chat_id
            self.server._tab_chat_views[busy_viewer] = chat_id
            self.server._tab_chat_views[idle_viewer] = chat_id

        # The busy viewer is actively running its OWN task 7999.
        assert busy_state.agent is not None
        busy_state.agent._last_task_id = "7999"
        busy_state.is_task_active = True

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

        # Regression guard: the idle viewer still gets the standard
        # clear + status running=True sequence.
        idle_types = [e.get("type") for e in self.recorder.for_tab(idle_viewer)]
        self.assertEqual(idle_types, ["clear", "status"])


class TestStatusEndSkipsBusyViewer(unittest.TestCase):
    """BUG-TR8-3: running=False must not reach a still-busy viewer."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.server = VSCodeServer()
        self.recorder = _EventRecorder()

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_busy_viewer_spinner_not_stopped(self) -> None:
        launcher = "bh8c-end-launcher"
        busy_viewer = "bh8c-end-busy"
        idle_viewer = "bh8c-end-idle"

        self.server._get_tab(launcher)
        busy_state = self.server._get_tab(busy_viewer)
        self.server._get_tab(idle_viewer)

        # Both viewers are subscribed to task 7301 (launched from the
        # launcher tab).  The busy viewer has since started its OWN
        # task 7999 which is still running.
        self.server.printer.subscribe_tab("7301", launcher)
        self.server.printer.subscribe_tab("7301", busy_viewer)
        self.server.printer.subscribe_tab("7301", idle_viewer)
        assert busy_state.agent is not None
        busy_state.agent._last_task_id = "7999"
        busy_state.is_task_active = True

        self.server.printer.broadcast = self.recorder  # type: ignore[assignment]
        self.server._broadcast_status_end_to_viewers("7301", launcher)

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


if __name__ == "__main__":
    unittest.main()
