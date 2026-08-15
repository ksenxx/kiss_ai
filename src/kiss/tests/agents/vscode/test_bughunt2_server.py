# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2: ``_replay_session``'s no-persisted-row branch must keep
its tab-state bookkeeping symmetric with the row-found path.

When ``resumeSession`` arrives before the task's ``task_history`` row
has been written (the writer race the branch itself documents — a tab
that started a task and is immediately closed+reopened, or a VS Code
reload replaying restored tabs), ``_replay_session`` takes an early
``return`` after re-subscribing the tab to a live agent's stream.  That
early branch must still perform the state updates the normal
(row-found) path performs:

1. ``self._tab_chat_views[tab_id] = chat_id`` must be recorded so the
   tab is fanned out to by ``_subscribe_chat_viewers`` when the NEXT
   task starts on the same chat from another tab / window / the CLI.

2. ``state.frontend_closed`` must be cleared so a pending
   deferred-dispose does not tear down the tab the user is actively
   viewing the moment its task ends.

3. The tab must be re-associated with the resumed chat in
   ``_tab_chat_views`` even when it previously displayed another chat.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class _ReplayNoRowBase(unittest.TestCase):
    """Shared fixture: isolated DB + broadcast capture."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt2-noresult-")
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            ev = self.server.printer._inject_task_id(event)
            with self._events_lock:
                self.events.append(ev)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved  # type: ignore[assignment]
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_running_state(
        self, tab_id: str, chat_id: str, task_id: str,
    ) -> agent_state.AgentState:
        """Register a live running task the way ``_cmd_run`` leaves it."""
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._last_task_id = task_id
        state = agent_state.AgentState(
            task_id,
            agent=agent,
            chat_id=chat_id,
            tab_id=tab_id,
            server_owned=True,
            is_task_active=True,
        )
        agent_state.register(state)
        return state

    def _events_for_tab(self, tab_id: str) -> list[dict[str, Any]]:
        with self._events_lock:
            return [e for e in self.events if e.get("tabId") == tab_id]


class TestNoRowReplayRegistersChatViewer(_ReplayNoRowBase):
    """Viewer resumed through the row-write race must still be fanned
    out to by the next task started on the same chat."""

    def test_viewer_receives_next_task_stream(self) -> None:
        chat_id = "chat-norow-viewer"
        self._make_running_state("launcher", chat_id, "task-1")

        self.server._replay_session(chat_id=chat_id, tab_id="viewer")

        viewer_events = self._events_for_tab("viewer")
        assert any(e.get("type") == "status" for e in viewer_events)

        with self._events_lock:
            self.events.clear()
        self.server._subscribe_chat_viewers(
            "task-2", chat_id, source_tab_id="launcher", start_ms=123,
        )

        viewer_events = self._events_for_tab("viewer")
        assert any(e.get("type") == "clear" for e in viewer_events) and any(
            e.get("type") == "status" and e.get("running") is True
            for e in viewer_events
        ), (
            "BUG: _replay_session's no-persisted-row branch never "
            "recorded the viewer tab in _tab_chat_views, so the next "
            "task on the chat does not stream to a tab that has the "
            "chat open"
        )


class TestNoRowReplayClearsFrontendClosed(_ReplayNoRowBase):
    """Re-resuming a close-marked busy tab through the race window must
    cancel the pending deferred disposal (normal path clears
    ``frontend_closed``; the no-row branch must too)."""

    def test_resumed_tab_survives_task_end(self) -> None:
        chat_id = "chat-norow-reopen"
        state = self._make_running_state("t1", chat_id, "task-9")

        self.server._close_tab("t1")
        assert state.frontend_closed is True
        assert agent_state.get("task-9") is state

        self.server._replay_session(chat_id=chat_id, tab_id="t1")

        state.is_task_active = False
        self.server._dispose_if_closed("t1")

        assert agent_state.get("task-9") is state, (
            "BUG: the no-persisted-row resume branch did not clear "
            "frontend_closed, so the deferred disposal tore down the "
            "tab the user is actively viewing"
        )


class TestNoRowReplayAssociatesChatId(_ReplayNoRowBase):
    """The no-row branch must re-associate the tab with the resumed
    chat in ``_tab_chat_views``, mirroring the found-row path, so a
    follow-up run continues the resumed chat."""

    def test_chat_id_reassociated(self) -> None:
        chat_id = "chat-norow-continue"
        self.server._tab_chat_views["t2"] = "old-finished-chat"

        self.server._replay_session(chat_id=chat_id, tab_id="t2")

        assert self.server._tab_chat_views.get("t2") == chat_id, (
            "BUG: no-persisted-row resume left the tab associated "
            "with the previously displayed chat; a follow-up run in "
            "this tab would append to the WRONG chat session"
        )


if __name__ == "__main__":
    unittest.main()
