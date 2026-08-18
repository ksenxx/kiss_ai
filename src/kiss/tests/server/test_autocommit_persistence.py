# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test that autocommit events are persisted to the task history database.

BUG: After a non-worktree task completes and its changes are
auto-committed, the ``autocommit_done`` event was not persisted to the
task history.  When the user later replays the session ("the report"),
the commit never shows up.

Root cause: ``_autocommit_changes`` broadcasts the ``autocommit_done``
event but did not call ``_append_chat_event`` to persist it.  By the
time autocommit happens, the task's recording has already been
stopped, so the automatic persistence path in ``broadcast()`` is also
inactive.  The fix resolves the task id from the tab's registered
agent state and persists the event explicitly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import kiss.agents.sorcar.persistence as th
import kiss.server.merge_flow as _merge_flow_module
from kiss.agents.sorcar.persistence import (
    _add_task,
    _get_db,
    _load_latest_chat_events_by_chat_id,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.json_printer import _DISPLAY_EVENT_TYPES
from kiss.server.server import VSCodeServer


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    """Create a git repo with one committed file so HEAD exists."""
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


class TestAutocommitPersistence(unittest.TestCase):
    """Verify that autocommit_done events are persisted to the database."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)
        self._db_tmpdir = tempfile.mkdtemp()
        self._saved_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self._db_tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff

        def capture(event: dict) -> None:
            self.events.append(event)

        self._real_broadcast = self.server.printer.broadcast
        self.server.printer.broadcast = capture  # type: ignore[assignment]

        def fake_compose(
            diff_text: str,
            user_prompt: str | None = None,
            task_result: str | None = None,
        ) -> str:
            return "feat: autocommit persistence test"

        _merge_flow_module.generate_commit_message_from_diff = fake_compose  # type: ignore[assignment]

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        agent_state.agent_states.clear()
        th._close_db()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_db
        shutil.rmtree(self._db_tmpdir, ignore_errors=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_task_for_tab(self, tab_id: str) -> tuple[str, str]:
        """Create a task history entry and register the tab's agent state."""
        task_id, chat_id = _add_task("test task", "")
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._last_task_id = task_id
        agent._chat_id = chat_id
        state = agent_state.AgentState(
            str(task_id), agent=agent, tab_id=tab_id, server_owned=True,
        )
        agent_state.register(state)
        return task_id, chat_id

    def _load_events_for_task(self, task_id: str) -> list[dict]:
        """Load persisted events for a task_id from the database."""
        db = _get_db()
        rows = db.execute(
            "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ).fetchall()
        return [json.loads(r["event_json"]) for r in rows]

    def test_autocommit_done_persisted_after_commit(self) -> None:
        """BUG REPRODUCTION: After a successful autocommit, the
        autocommit_done event must be persisted to the task history
        so it appears when the session is replayed."""
        task_id, chat_id = self._create_task_for_tab("t1")

        Path(self.tmpdir, "seed.txt").write_text("modified content\n")

        self.server._autocommit_changes("t1")

        status = _run_git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert status == "", f"Working tree should be clean after commit: {status}"

        done_events = [e for e in self.events if e.get("type") == "autocommit_done"]
        assert len(done_events) == 1
        assert done_events[0]["success"] is True
        assert done_events[0]["committed"] is True

        persisted = self._load_events_for_task(task_id)
        ac_events = [e for e in persisted if e.get("type") == "autocommit_done"]
        assert len(ac_events) == 1, (
            f"autocommit_done should be persisted but got {len(ac_events)} events. "
            f"All persisted types: {[e.get('type') for e in persisted]}"
        )
        assert ac_events[0]["committed"] is True
        assert ac_events[0]["success"] is True

    def test_autocommit_without_changes_not_persisted(self) -> None:
        """A clean tree (no commit made) should NOT persist an
        autocommit_done event since no commit was created."""
        task_id, chat_id = self._create_task_for_tab("t2")

        self.server._autocommit_changes("t2")

        done_events = [e for e in self.events if e.get("type") == "autocommit_done"]
        assert len(done_events) == 1
        assert done_events[0]["committed"] is False

        persisted = self._load_events_for_task(task_id)
        ac_events = [e for e in persisted if e.get("type") == "autocommit_done"]
        assert len(ac_events) == 0

    def test_autocommit_done_visible_in_session_replay(self) -> None:
        """When the session is replayed via _load_latest_chat_events_by_chat_id,
        the autocommit_done event must be included in the returned events."""
        task_id, chat_id = self._create_task_for_tab("t3")

        Path(self.tmpdir, "seed.txt").write_text("changed for replay test\n")
        self.server._autocommit_changes("t3")

        result = _load_latest_chat_events_by_chat_id(chat_id)
        assert result is not None, "Should find events for chat_id"
        events = result["events"]
        assert isinstance(events, list)
        ac_done = [e for e in events if e.get("type") == "autocommit_done"]
        assert len(ac_done) == 1, (
            f"autocommit_done should appear in session replay but got "
            f"{[e.get('type') for e in events]}"
        )
        assert ac_done[0]["committed"] is True

    def test_autocommit_without_task_id_does_not_crash(self) -> None:
        """If there's no task_id on the agent (edge case), autocommit
        should still work but just not persist."""
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._last_task_id = None
        state = agent_state.AgentState(
            "", agent=agent, tab_id="t4", server_owned=True,
        )
        agent_state.register(state)

        Path(self.tmpdir, "seed.txt").write_text("no task id\n")
        self.server._autocommit_changes("t4")

        done_events = [e for e in self.events if e.get("type") == "autocommit_done"]
        assert len(done_events) == 1
        assert done_events[0]["success"] is True
        assert done_events[0]["committed"] is True


class TestAutocommitDoneInDisplayEventTypes(unittest.TestCase):
    """autocommit_done must be in _DISPLAY_EVENT_TYPES so the frontend
    can replay it from persisted events."""

    def test_autocommit_done_is_display_event(self) -> None:
        assert "autocommit_done" in _DISPLAY_EVENT_TYPES, (
            "autocommit_done must be in _DISPLAY_EVENT_TYPES for replay"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
