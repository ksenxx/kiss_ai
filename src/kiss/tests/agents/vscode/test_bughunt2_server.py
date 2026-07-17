# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2: ``_replay_session``'s no-persisted-row branch skips all
tab-state bookkeeping.

When ``resumeSession`` arrives before the task's ``task_history`` row
has been written (the writer race the branch itself documents — a tab
that started a task and is immediately closed+reopened, or a VS Code
reload replaying restored tabs), ``_replay_session`` takes an early
``return`` after re-subscribing the tab to a live agent's stream.  That
early branch omits the three state updates the normal (row-found) path
performs:

1. ``self._tab_chat_views[tab_id] = chat_id`` is never recorded, even
   though ``_cmd_run`` (commands.py) documents that the map "is
   populated unconditionally, even when the per-tab state has not been
   allocated yet" and relies on it both for chat continuation of a
   follow-up run AND for ``_subscribe_chat_viewers`` fan-out.  A tab
   resumed through the race window therefore never receives the live
   event stream of the NEXT task started on the same chat from another
   tab / window / the CLI.

2. ``tab.frontend_closed`` is not cleared.  The normal path clears it
   so "a pending deferred-dispose does not tear down the tab the user
   is actively viewing" — a tab that was close-marked while busy and
   then re-resumed through the race window is silently torn down
   (subscriptions dropped, registry entry popped) the moment its task
   ends, leaving the user staring at a dead webview.

3. ``tab.chat_id`` is not re-associated on an existing registry entry.

The fix mirrors the found-row path's bookkeeping at the end of the
no-row branch (guarded on a non-empty ``chat_id`` and skipping the
``_tab_chat_views`` registration for sub-agent view tabs, exactly like
the normal path's ``subagent_info`` guard).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
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
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_running_tab(
        self, tab_id: str, chat_id: str, task_id: str,
    ) -> _RunningAgentState:
        """Register a live running task the way ``_cmd_run`` leaves it."""
        tab = self.server._get_tab(tab_id)
        tab.chat_id = chat_id
        tab.is_task_active = True
        assert tab.agent is not None
        tab.agent._last_task_id = task_id
        return tab

    def _events_for_tab(self, tab_id: str) -> list[dict[str, Any]]:
        with self._events_lock:
            return [e for e in self.events if e.get("tabId") == tab_id]


class TestNoRowReplayRegistersChatViewer(_ReplayNoRowBase):
    """Viewer resumed through the row-write race must still be fanned
    out to by the next task started on the same chat."""

    def test_viewer_receives_next_task_stream(self) -> None:
        chat_id = "chat-norow-viewer"
        self._make_running_tab("launcher", chat_id, "task-1")

        # The user reopens the chat in a fresh tab before the task's
        # DB row exists: _replay_session finds no persisted row and
        # takes the no-row branch (rebinding to the live task-1).
        self.server._replay_session(chat_id=chat_id, tab_id="viewer")

        # Sanity: the no-row branch did rebind the viewer (status +
        # empty task_events replay).
        viewer_events = self._events_for_tab("viewer")
        assert any(e.get("type") == "status" for e in viewer_events)

        # task-1 ends; a NEW task (task-2) later starts on the same
        # chat from the launcher tab.  _subscribe_chat_viewers (the
        # real fan-out hook invoked via _on_task_id_allocated) must
        # subscribe every tab that has the chat open — including the
        # race-window viewer.
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
        tab = self._make_running_tab("t1", chat_id, "task-9")

        # The frontend closes the busy tab: deferred-dispose is armed.
        self.server._close_tab("t1")
        assert tab.frontend_closed is True
        assert "t1" in _RunningAgentState.running_agent_states

        # VS Code reload restores the tab and replays resumeSession
        # for it — before the task row hits the DB (no-row branch).
        # The user is actively viewing the tab again.
        self.server._replay_session(chat_id=chat_id, tab_id="t1")

        # The task ends; the runner's lifecycle hook fires.
        tab.is_task_active = False
        self.server._dispose_if_closed("t1")

        assert "t1" in _RunningAgentState.running_agent_states, (
            "BUG: the no-persisted-row resume branch did not clear "
            "frontend_closed, so the deferred disposal tore down the "
            "tab the user is actively viewing"
        )


class TestNoRowReplayAssociatesChatId(_ReplayNoRowBase):
    """The no-row branch must re-associate ``tab.chat_id`` on an
    existing registry entry, mirroring the found-row path, so a
    follow-up run continues the resumed chat."""

    def test_chat_id_reassociated(self) -> None:
        chat_id = "chat-norow-continue"
        # A registry entry exists for the tab (it previously ran a
        # task whose chat was since deleted), and the user now resumes
        # a different chat whose row is not yet written.
        tab = self.server._get_tab("t2")
        tab.chat_id = "old-finished-chat"

        self.server._replay_session(chat_id=chat_id, tab_id="t2")

        assert tab.chat_id == chat_id, (
            "BUG: no-persisted-row resume left the tab associated "
            "with the previously displayed chat; a follow-up run in "
            "this tab would append to the WRONG chat session"
        )


if __name__ == "__main__":
    unittest.main()
