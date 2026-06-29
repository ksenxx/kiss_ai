# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6 (new flat-column schema): starring a row must never flip
its sub-agent classification.

The pre-iteration-5 attack surface was a JSON ``extra`` column whose
corrupt contents could trick the favourite-rewrite into adding /
dropping a top-level ``"subagent"`` key.  In the current flat-column
schema sub-agent identification is driven exclusively by the dedicated
``parent_task_id`` column, which ``_set_task_favorite`` never touches.
These tests therefore lock down the *post-refactor* invariant: toggling
``is_favorite`` is purely a flag write that preserves the row's
sub-agent classification and keeps the row's chat reachable from the
recent-chats list.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _list_recent_chats,
    _load_history,
    _search_history,
    _set_task_favorite,
)


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _parent_task_id(self, task_id: str) -> str:
        db = th._get_db()
        with th._rw_lock.read_lock():
            row = db.execute(
                "SELECT parent_task_id FROM task_history WHERE id = ?",
                (task_id,),
            ).fetchone()
        return str(row["parent_task_id"] or "")

    def _row(self, task_id: str):
        db = th._get_db()
        with th._rw_lock.read_lock():
            return db.execute(
                "SELECT * FROM task_history WHERE id = ?",
                (task_id,),
            ).fetchone()

    def _is_favorite(self, task_id: str) -> bool:
        db = th._get_db()
        with th._rw_lock.read_lock():
            row = db.execute(
                "SELECT is_favorite FROM task_history WHERE id = ?",
                (task_id,),
            ).fetchone()
        return bool(row["is_favorite"])


class TestFavoriteDoesNotFlipClassification(_TempDbTestBase):
    """Starring/unstarring a row must never change its sub-agent classification."""

    def test_star_regular_row_stays_visible(self) -> None:
        task_id, chat_id = _add_task("plain row", extra={"model": "m"})
        # Pre-condition: classified as regular task.
        assert "plain row" in [e["task"] for e in _load_history()]
        assert not self._row(task_id)["parent_task_id"]

        assert _set_task_favorite(task_id, True)

        # The starred row must remain visible and stay classified
        # as a regular (non-sub-agent) task.
        assert "plain row" in [e["task"] for e in _load_history()]
        assert "plain row" in [
            e["task"] for e in _search_history("plain")
        ]
        assert not self._row(task_id)["parent_task_id"]
        assert self._is_favorite(task_id)

    def test_star_row_keeps_chat_in_recent_chats(self) -> None:
        task_id, chat_id = _add_task("plain row")
        assert _set_task_favorite(task_id, True)
        chats = _list_recent_chats(limit=10)
        assert chat_id in [c["chat_id"] for c in chats]

    def test_unstar_row_stays_visible_and_clears_flag(self) -> None:
        task_id, chat_id = _add_task("plain row")
        assert _set_task_favorite(task_id, True)
        assert _set_task_favorite(task_id, False)
        entries = [e for e in _load_history() if e["task"] == "plain row"]
        assert len(entries) == 1
        assert not self._is_favorite(task_id)

    def test_star_preserves_displayed_metadata(self) -> None:
        # Toggling favourite must not corrupt the flat metadata columns
        # used by the history sidebar.
        task_id, chat_id = _add_task(
            "row with meta",
            extra={"model": "m", "work_dir": "/tmp/x", "cost": 1.5},
        )
        assert _set_task_favorite(task_id, True)
        db = th._get_db()
        with th._rw_lock.read_lock():
            row = db.execute(
                "SELECT model, work_dir, cost, is_favorite "
                "FROM task_history WHERE id = ?",
                (task_id,),
            ).fetchone()
        assert row["model"] == "m"
        assert row["work_dir"] == "/tmp/x"
        assert row["cost"] == 1.5
        assert row["is_favorite"] == 1

    def test_star_valid_subagent_row_stays_hidden(self) -> None:
        # A REAL sub-agent row (parent_task_id set) keeps its sub-agent
        # classification through a favourite toggle and stays hidden.
        parent_id, chat = _add_task("parent task")
        sub_id, _ = _add_task(
            "fanned-out subtask",
            chat_id=chat,
            extra={"subagent": {"parent_task_id": parent_id}},
        )
        assert _set_task_favorite(sub_id, True)
        assert "fanned-out subtask" not in [
            e["task"] for e in _load_history()
        ]
        assert self._row(sub_id)["parent_task_id"] == parent_id
