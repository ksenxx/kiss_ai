# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 5: ``_replay_session`` must not mint a phantom registry entry.

Pure-viewer tabs (opened from the history sidebar) deliberately have NO
agent-state registry entry — ``_replay_session`` documents this
invariant (the C2/C3 fix) and only *looks up* existing states.  The
same must hold for ``_present_pending_worktree``: when no entry exists
there cannot be a pending worktree to present (the agent holding
worktree state lives on the entry), so neither call may register a new
state for a viewer tab.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class TestReplayPhantomState(unittest.TestCase):
    """History-click viewer tabs must not mint registry entries."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt5-phantom-")
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

    def test_history_click_creates_no_registry_entry(self) -> None:
        chat_id = "chat-bughunt5-phantom"
        task_id, _ = th._add_task("a finished task", chat_id=chat_id)
        th._append_chat_event(
            {"type": "text_delta", "text": "hello"}, task_id=task_id,
        )

        viewer_tab = "viewer-tab-1"
        self.server._replay_session(chat_id=chat_id, tab_id=viewer_tab)

        replays = [e for e in self.events if e.get("type") == "task_events"]
        assert len(replays) == 1 and replays[0]["tabId"] == viewer_tab

        assert agent_state.find_by_tab(viewer_tab) is None, (
            "BUG: _replay_session minted a phantom agent state for a "
            "pure-viewer tab"
        )

    def test_present_pending_worktree_unknown_tab_is_noop(self) -> None:
        self.server._present_pending_worktree("never-seen-tab")
        assert agent_state.find_by_tab("never-seen-tab") is None, (
            "BUG: _present_pending_worktree minted a registry entry for "
            "an unknown tab id"
        )


if __name__ == "__main__":
    unittest.main()
