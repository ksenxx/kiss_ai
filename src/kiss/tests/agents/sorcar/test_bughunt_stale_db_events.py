# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Stale asynchronous events must never leak into a swapped database.

``task_history`` ids are AUTOINCREMENT — unique within ONE database
file.  When ``_DB_PATH`` is reassigned (test fixtures, a daemon
restarted against another home dir) the id counter restarts, so an
event that was enqueued against the OLD database carries a numeric
``task_id`` that resolves to a completely unrelated task in the NEW
database.  Before the fix, the background event writer (and the late
``followup_suggestion`` append from the server's fire-and-forget
thread) happily wrote such stale events into whatever database was
active at drain time, attaching them to the wrong task.  This was the
root cause of recurring cross-test event pollution (e.g. a
``followup_suggestion`` row appearing in the middle of an unrelated
test's event stream).

The fix stamps every pending write with the database path it was
produced against (``origin_db_path``) and drops it if the active
database has changed by the time it is persisted.
"""

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str):
    """Redirect the DB to a temp dir and reset the singleton connection."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved):
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestStaleDbEvents:
    def setup_method(self):
        self.tmpdir_a = tempfile.mkdtemp()
        self.tmpdir_b = tempfile.mkdtemp()
        # Make sure no backlog from earlier tests is in flight.
        th._flush_chat_events()
        self.saved = _redirect(self.tmpdir_a)

    def teardown_method(self):
        th._flush_chat_events()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir_a, ignore_errors=True)
        shutil.rmtree(self.tmpdir_b, ignore_errors=True)

    def _events_for(self, task_id: int) -> list:
        loaded = th._load_chat_events_by_task_id(task_id)
        if not loaded:
            return []
        events = loaded["events"]
        assert isinstance(events, list)
        return events

    def test_queued_backlog_does_not_leak_into_swapped_db(self):
        """Events enqueued under DB A never land on DB B's same-id task."""
        task_a, _ = th._add_task("task-a", chat_id="chat-a")
        th._flush_chat_events()
        # Enqueue a backlog far larger than one writer batch so part of
        # it is still queued when the database is swapped underneath.
        for i in range(5000):
            th._queue_chat_event(
                {"type": "text_delta", "content": f"x{i}"}, task_id=task_a,
            )
        # Swap to a fresh database B; its AUTOINCREMENT counter restarts
        # so the first task reuses the same numeric id as DB A's task.
        th._DB_PATH = Path(self.tmpdir_b) / "sorcar.db"
        th._db_conn = None
        task_b, _ = th._add_task("task-b", chat_id="chat-b")
        assert task_b == task_a, "precondition: ids collide across DBs"
        th._flush_chat_events()
        leaked = self._events_for(task_b)
        assert leaked == [], (
            f"{len(leaked)} stale event(s) from the previous database "
            f"leaked into the new database's unrelated task: "
            f"{leaked[:3]}..."
        )

    def test_append_with_stale_origin_path_is_dropped(self):
        """A late append stamped with an old DB path must be a no-op."""
        task_id, _ = th._add_task("current-task", chat_id="chat-cur")
        th._append_chat_event(
            {"type": "followup_suggestion", "text": "late suggestion"},
            task_id=task_id,
            task="some other task that completed in the old database",
            origin_db_path="/nonexistent/old/sorcar.db",
        )
        assert self._events_for(task_id) == []

    def test_append_with_current_origin_path_persists(self):
        """The guard must not drop legitimate same-database appends."""
        task_id, _ = th._add_task("current-task-2", chat_id="chat-cur2")
        th._append_chat_event(
            {"type": "followup_suggestion", "text": "fresh suggestion"},
            task_id=task_id,
            origin_db_path=th._current_db_path(),
        )
        events = self._events_for(task_id)
        assert len(events) == 1
        assert events[0]["type"] == "followup_suggestion"

    def test_append_without_origin_path_keeps_working(self):
        """Default (no origin) appends behave exactly as before."""
        task_id, _ = th._add_task("current-task-3", chat_id="chat-cur3")
        th._append_chat_event({"step": 1}, task_id=task_id)
        events = self._events_for(task_id)
        assert len(events) == 1
        assert events[0]["step"] == 1
