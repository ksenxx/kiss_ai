# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``run_agent`` child on its parent's chat must not hijack the chat's viewers.

A path-mode ``run_agent`` sub-task inherits the calling task's chat id
(``agent_dispatch.inherit_from_parent``), so it starts on a chat that
already has a running task — the parent — and possibly idle viewer tabs
(a history viewer in a sibling window).  ``_on_run_task_id_allocated``
used to subscribe every idle viewer of the chat to the new task,
``clear`` their content and flip their status, which for a sub-agent
interleaves the parent's and the child's streams in those tabs.  The
child's stream belongs in the nested sub-agent tab the ``new_tab``
broadcast opens, so the hook now skips the chat-wide subscription for a
run submitted with ``parentTaskId`` (``is_subagent=True``) while still
registering the launching tab.
"""

from __future__ import annotations

import threading
import unittest
import uuid
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer

CHAT_ID = "chat-shared"
VIEWER_TAB = "idle-viewer-tab"
CHILD_TAB = "api-child-tab"


class TestSubagentSkipsChatViewerSubscription(unittest.TestCase):
    """``_on_run_task_id_allocated(is_subagent=True)`` leaves chat viewers alone."""

    def setUp(self) -> None:
        self.server = AgentServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def recording_broadcast(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = recording_broadcast  # type: ignore[assignment]
        with self.server._state_lock:
            self.server._tab_chat_views[VIEWER_TAB] = CHAT_ID
        agent_state.register(
            AgentState("viewer-key", chat_id=CHAT_ID, tab_id=VIEWER_TAB, server_owned=True),
        )

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def _viewer_events(self) -> list[dict[str, Any]]:
        with self._events_lock:
            return [e for e in self.events if e.get("tabId") == VIEWER_TAB]

    def _subscribers(self, task_id: str) -> set[str]:
        with self.server.printer._lock:
            return set(self.server.printer._subscribers.get(task_id, set()))

    def test_subagent_allocation_registers_only_its_own_tab(self) -> None:
        """The child's tab is wired; the idle viewer of the parent's chat is untouched."""
        self.server._on_run_task_id_allocated(
            "child-task", CHAT_ID, source_tab_id=CHILD_TAB,
            start_ms=123, is_subagent=True,
        )
        assert CHILD_TAB in self._subscribers("child-task")
        assert VIEWER_TAB not in self._subscribers("child-task")
        assert self._viewer_events() == []

    def test_ordinary_allocation_still_subscribes_the_viewer(self) -> None:
        """Positive control: a top-level task on the chat keeps the viewer invariant."""
        self.server._on_run_task_id_allocated(
            "top-task", CHAT_ID, source_tab_id="launcher-tab", start_ms=123,
        )
        assert VIEWER_TAB in self._subscribers("top-task")
        assert "launcher-tab" in self._subscribers("top-task")
        types = [e.get("type") for e in self._viewer_events()]
        assert types == ["clear", "status"], types

    def test_launcher_moved_to_another_chat_is_not_resubscribed(self) -> None:
        """A row allocated after a replay bound the launching tab to another chat stays out of it.

        The replay's commit drops the tab's subscription to its own run;
        the run's next row (its first one when the replay landed in the
        run's setup, a later subtask's otherwise) used to subscribe the
        tab again, and the run streamed into the other chat's transcript.
        """
        launcher = f"launcher-{uuid.uuid4().hex}"
        self.addCleanup(self.server._close_tab, launcher)
        _other_task, other_chat = th._add_task("Task of another chat")
        # What ``_cmd_run`` leaves for a run in its setup: the tab views
        # the run's chat and owns its state, which has no row yet.
        with self.server._state_lock:
            self.server._tab_chat_views[launcher] = CHAT_ID
        agent_state.register(
            AgentState(
                "rowless-key", chat_id=CHAT_ID, tab_id=launcher,
                server_owned=True, is_task_active=True,
            ),
        )
        self.server._cmd_resume_session({"chatId": other_chat, "tabId": launcher})
        self.server._on_run_task_id_allocated(
            "late-task", CHAT_ID, source_tab_id=launcher, start_ms=123,
        )
        assert launcher not in self._subscribers("late-task")
        # The chat's own viewers still follow the run.
        assert VIEWER_TAB in self._subscribers("late-task")
