# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 5: ``_replay_session`` mints a phantom registry entry (BUG-5E-2).

Pure-viewer tabs (opened from the history sidebar) deliberately have NO
``_RunningAgentState`` registry entry — ``_replay_session`` documents
this invariant (the C2/C3 fix) and explicitly avoids creating one.  But
its final ``_emit_pending_worktree(tab_id)`` call delegates to
``_present_pending_worktree``, whose first line is
``tab = self._get_tab(tab_id)`` — which CREATES a registry entry (and
eagerly allocates a ``WorktreeSorcarAgent``) for every viewer tab,
silently regressing the invariant: every history click leaks one
registry entry + agent per viewer tab until the tab is closed, and the
entry participates in every registry scan (busy checks, chat-id
matching in ``_resolve_parent_tab_id_for_sub`` and
``_reattach_running_chat``).

``_present_pending_worktree`` only needs to *look up* the tab: when no
entry exists there cannot be a pending worktree to present (the agent
holding worktree state lives on the entry), so a non-creating ``get``
is behaviourally identical except it does not mint the phantom.
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
from kiss.agents.vscode.server import VSCodeServer


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
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_history_click_creates_no_registry_entry(self) -> None:
        chat_id = "chat-bughunt5-phantom"
        task_id, _ = th._add_task("a finished task", chat_id=chat_id)
        th._append_chat_event(
            {"type": "text_delta", "text": "hello"}, task_id=task_id,
        )

        viewer_tab = "viewer-tab-1"
        self.server._replay_session(chat_id=chat_id, tab_id=viewer_tab)

        # The replay itself must have happened ...
        replays = [e for e in self.events if e.get("type") == "task_events"]
        assert len(replays) == 1 and replays[0]["tabId"] == viewer_tab

        # ... but viewing a chat is a read-only operation: no
        # _RunningAgentState (and no eagerly-created agent) may be
        # registered for the pure-viewer tab.
        assert viewer_tab not in _RunningAgentState.running_agent_states, (
            "BUG: _replay_session minted a phantom _RunningAgentState "
            "(via _emit_pending_worktree -> _present_pending_worktree "
            "-> _get_tab) for a pure-viewer tab"
        )

    def test_present_pending_worktree_unknown_tab_is_noop(self) -> None:
        self.server._present_pending_worktree(
            "never-seen-tab", try_merge_review=True,
        )
        assert (
            "never-seen-tab" not in _RunningAgentState.running_agent_states
        ), (
            "BUG: _present_pending_worktree minted a registry entry for "
            "an unknown tab id"
        )


if __name__ == "__main__":
    unittest.main()
