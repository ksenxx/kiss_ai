"""Tests for task deletion: DB removal, event cleanup, and history refresh."""

import shutil
import tempfile
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str):
    """Redirect DB to a temp dir and reset the singleton connection."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved):
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestDeleteTask:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_delete_existing_task(self):
        """Deleting an existing task removes it from task_history."""
        task_id, _ = th._add_task("task to delete")
        assert th._load_history() != []
        result = th._delete_task(task_id)
        assert result is True
        entries = th._load_history()
        assert len(entries) == 0

    def test_delete_nonexistent_task(self):
        """Deleting a non-existent task returns False."""
        result = th._delete_task(99999)
        assert result is False

    def test_delete_task_removes_events(self):
        """Events associated with the deleted task are removed."""
        task_id, _ = th._add_task("task with events")
        th._append_chat_event({"type": "text_delta", "text": "hello"}, task_id=task_id)
        th._append_chat_event({"type": "result", "text": "done"}, task_id=task_id)

        # Verify events exist
        db = th._get_db()
        rows = db.execute("SELECT COUNT(*) FROM events WHERE task_id = ?", (task_id,)).fetchone()
        assert rows[0] == 2

        th._delete_task(task_id)

        # Verify events are gone
        rows = db.execute("SELECT COUNT(*) FROM events WHERE task_id = ?", (task_id,)).fetchone()
        assert rows[0] == 0

    def test_delete_does_not_affect_other_tasks(self):
        """Deleting one task leaves other tasks untouched."""
        id1, _ = th._add_task("keep this")
        time.sleep(0.01)
        id2, _ = th._add_task("delete this")
        th._append_chat_event({"type": "text_delta", "text": "x"}, task_id=id1)
        th._append_chat_event({"type": "text_delta", "text": "y"}, task_id=id2)

        th._delete_task(id2)

        entries = th._load_history()
        assert len(entries) == 1
        assert entries[0]["task"] == "keep this"

        # Events for kept task still exist
        db = th._get_db()
        rows = db.execute("SELECT COUNT(*) FROM events WHERE task_id = ?", (id1,)).fetchone()
        assert rows[0] == 1

    def test_delete_task_in_chat_session(self):
        """Deleting one task from a multi-task chat session keeps others."""
        id1, chat_id = th._add_task("first in session")
        time.sleep(0.01)
        id2, _ = th._add_task("second in session", chat_id=chat_id)
        time.sleep(0.01)
        id3, _ = th._add_task("third in session", chat_id=chat_id)

        th._delete_task(id2)

        entries = th._load_history()
        assert len(entries) == 2
        tasks = [e["task"] for e in entries]
        assert "first in session" in tasks
        assert "third in session" in tasks
        assert "second in session" not in tasks

        # Remaining tasks still share the same chat_id
        for e in entries:
            assert e["chat_id"] == chat_id

    def test_delete_last_task_in_chat(self):
        """Deleting the only task in a chat session removes it entirely."""
        task_id, chat_id = th._add_task("only task")
        th._delete_task(task_id)
        entries = th._load_history()
        assert len(entries) == 0

    def test_delete_task_with_extra_metadata(self):
        """Task with extra JSON metadata is deleted correctly."""
        task_id, _ = th._add_task("task with extra")
        th._save_task_extra({"model": "test", "cost": "0.01"}, task_id=task_id)
        th._save_task_result("done", task_id=task_id)

        result = th._delete_task(task_id)
        assert result is True
        assert th._load_history() == []
