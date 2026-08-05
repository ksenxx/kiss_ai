# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Behavioral parity tests for the two chat-events loaders.

``_load_latest_chat_events_by_chat_id`` and
``_load_chat_events_by_task_id`` historically duplicated the
lock/select/return boilerplate.  These tests pin the shared behavior
(identical session-dict shape and identical values for the same row)
so the duplication can be collapsed into one helper without drift.
"""

import shutil
import tempfile
import uuid
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


def _event_types(session: dict[str, object]) -> list[str]:
    """Return the ``type`` field of each event in a loaded session dict."""
    events = session["events"]
    assert isinstance(events, list)
    return [e["type"] for e in events]


class TestChatEventsLoaderParity:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_loaders_return_identical_session_dict_for_same_row(self):
        task_id, chat_id = th._add_task("parent task", extra={"model": "m1"})
        th._append_chat_event({"type": "text", "text": "hello"}, task_id=task_id)
        th._append_chat_event({"type": "result", "summary": "done"}, task_id=task_id)

        by_chat = th._load_latest_chat_events_by_chat_id(chat_id)
        by_task = th._load_chat_events_by_task_id(task_id)

        assert by_chat is not None
        assert by_task is not None
        assert set(by_chat) == {"task", "task_id", "events", "chat_id", "extra"}
        assert set(by_task) == set(by_chat)
        assert by_chat == by_task
        assert by_chat["task"] == "parent task"
        assert by_chat["task_id"] == task_id
        assert by_chat["chat_id"] == chat_id
        assert _event_types(by_chat) == ["text", "result"]

    def test_chat_id_loader_skips_subagent_rows(self):
        parent_id, chat_id = th._add_task("parent task")
        sub_id, _ = th._add_task(
            "sub task",
            chat_id=chat_id,
            extra={"subagent": {"parent_task_id": parent_id}},
        )
        th._append_chat_event({"type": "text", "text": "sub"}, task_id=sub_id)

        by_chat = th._load_latest_chat_events_by_chat_id(chat_id)
        assert by_chat is not None
        assert by_chat["task_id"] == parent_id

        by_task = th._load_chat_events_by_task_id(sub_id)
        assert by_task is not None
        assert by_task["task_id"] == sub_id
        assert by_task["chat_id"] == chat_id
        assert _event_types(by_task) == ["text"]

    def test_loaders_return_none_for_missing_rows(self):
        assert th._load_latest_chat_events_by_chat_id("") is None
        assert th._load_latest_chat_events_by_chat_id(uuid.uuid4().hex) is None
        assert th._load_chat_events_by_task_id(uuid.uuid4().hex) is None

    def test_task_id_loader_returns_empty_chat_id_for_null_row(self):
        """A task row whose chat_id is NULL must yield chat_id == ""."""
        task_id, _ = th._add_task("orphan task")
        th._append_chat_event({"type": "text", "text": "hi"}, task_id=task_id)
        with th._rw_lock.write_lock():
            conn = th._get_db()
            conn.execute(
                "UPDATE task_history SET chat_id = NULL WHERE id = ?", (task_id,)
            )
            conn.commit()

        by_task = th._load_chat_events_by_task_id(task_id)
        assert by_task is not None
        assert by_task["chat_id"] == ""
        assert by_task["task_id"] == task_id
        assert _event_types(by_task) == ["text"]
