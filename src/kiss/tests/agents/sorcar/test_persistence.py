"""Tests for task history: SQLite storage, events, model/file usage, cleanup."""

import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest

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


class TestTaskHistory:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_history_entry(self):
        th._add_task("first")
        time.sleep(0.01)
        th._add_task("second")
        entry = th._get_history_entry(0)
        assert entry is not None
        assert entry["task"] == "second"
        entry1 = th._get_history_entry(1)
        assert entry1 is not None
        assert entry1["task"] == "first"
        assert th._get_history_entry(99999) is None

    def test_prefix_match_task_empty_query(self):
        th._add_task("anything")
        assert th._prefix_match_task("") == ""


class TestChatEvents:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_set_events_no_task(self):
        task_id, _ = th._add_task("latest", chat_id="1004")
        th._append_chat_event({"a": 1}, task_id=task_id)
        result = th._load_latest_chat_events_by_chat_id("1004")
        assert result is not None
        events = result["events"]
        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["a"] == 1
        assert "_timestamp" in events[0]
        assert isinstance(events[0]["_timestamp"], float)

    def test_load_chat_events_includes_extra(self):
        task_id, _ = th._add_task("extra-task", chat_id="1005")
        th._append_chat_event({"b": 2}, task_id=task_id)
        extra = {"model": "gpt-4o", "is_worktree": True, "is_parallel": False}
        th._save_task_extra(extra, task_id=task_id)
        result = th._load_latest_chat_events_by_chat_id("1005")
        assert result is not None
        loaded = json.loads(str(result["extra"]))
        assert loaded["model"] == "gpt-4o"
        assert loaded["is_worktree"] is True
        assert loaded["is_parallel"] is False

    def test_set_events_stores_timestamps(self):
        task_id, _ = th._add_task("ts-task", chat_id="ts1")
        before = time.time()
        th._append_chat_event({"x": 1}, task_id=task_id)
        th._append_chat_event({"x": 2}, task_id=task_id)
        after = time.time()
        result = th._load_latest_chat_events_by_chat_id("ts1")
        assert result is not None
        events = result["events"]
        assert isinstance(events, list)
        assert len(events) == 2
        for ev in events:
            assert isinstance(ev, dict)
            assert before <= ev["_timestamp"] <= after

    def test_append_event_stores_timestamp(self):
        task_id, _ = th._add_task("append-ts", chat_id="ts2")
        before = time.time()
        th._append_chat_event({"step": 1}, task_id=task_id)
        after_first = time.time()
        time.sleep(0.01)
        before_second = time.time()
        th._append_chat_event({"step": 2}, task_id=task_id)
        after_second = time.time()
        result = th._load_latest_chat_events_by_chat_id("ts2")
        assert result is not None
        events = result["events"]
        assert isinstance(events, list)
        assert len(events) == 2
        assert before <= events[0]["_timestamp"] <= after_first
        assert before_second <= events[1]["_timestamp"] <= after_second
        assert events[1]["_timestamp"] >= events[0]["_timestamp"]

    def test_adjacent_task_events_have_timestamps(self):
        task_id1, _ = th._add_task("adj-first", chat_id="adj1")
        th._append_chat_event({"ev": "a"}, task_id=task_id1)
        time.sleep(0.01)
        task_id2, _ = th._add_task("adj-second", chat_id="adj1")
        th._append_chat_event({"ev": "b"}, task_id=task_id2)
        prev = th._get_adjacent_task_by_chat_id("adj1", "adj-second", "prev")
        assert prev is not None
        prev_events = prev["events"]
        assert isinstance(prev_events, list)
        assert len(prev_events) == 1
        assert isinstance(prev_events[0], dict)
        assert "_timestamp" in prev_events[0]
        assert isinstance(prev_events[0]["_timestamp"], float)

    def test_event_timestamp_in_raw_db(self):
        """Verify the timestamp column exists in the events table at the DB level."""
        task_id, _ = th._add_task("raw-ts", chat_id="raw1")
        th._append_chat_event({"k": "v"}, task_id=task_id)
        db = th._get_db()
        row = db.execute(
            "SELECT timestamp FROM events WHERE task_id = ?", (task_id,)
        ).fetchone()
        assert row is not None
        assert isinstance(row["timestamp"], float)
        assert row["timestamp"] > 0

    def test_save_task_result_no_matching_task(self):
        th._save_task_result(result="result", task="nonexistent")


class TestSaveTaskExtra:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_extra(self):
        task_id, _ = th._add_task("extra test task", chat_id="1002")
        extra = {
            "model": "claude-opus-4-6",
            "work_dir": "/tmp/test",
            "version": "0.2.79",
            "tokens": 1234,
            "cost": 0.0567,
            "is_parallel": False,
            "is_worktree": True,
        }
        th._save_task_extra(extra, task_id=task_id)
        entries = th._load_history(limit=1)
        assert len(entries) == 1
        stored = json.loads(str(entries[0]["extra"]))
        assert stored["model"] == "claude-opus-4-6"
        assert stored["work_dir"] == "/tmp/test"
        assert stored["version"] == "0.2.79"
        assert stored["tokens"] == 1234
        assert stored["cost"] == 0.0567
        assert stored["is_parallel"] is False
        assert stored["is_worktree"] is True

    def test_save_extra_by_task_name(self):
        th._add_task("lookup task")
        th._save_task_extra({"model": "gpt-4o"}, task="lookup task")
        entries = th._load_history(limit=1)
        stored = json.loads(str(entries[0]["extra"]))
        assert stored["model"] == "gpt-4o"

    def test_save_extra_no_matching_task(self):
        th._save_task_extra({"model": "x"}, task="nonexistent")

    def test_extra_default_empty(self):
        th._add_task("no extra")
        entries = th._load_history(limit=1)
        assert entries[0]["extra"] == ""

    def test_extra_in_search_results(self):
        task_id, _ = th._add_task("searchable extra", chat_id="1003")
        th._save_task_extra({"model": "test-model"}, task_id=task_id)
        results = th._search_history("searchable")
        assert len(results) == 1
        stored = json.loads(str(results[0]["extra"]))
        assert stored["model"] == "test-model"

    def test_extra_in_get_history_entry(self):
        task_id, _ = th._add_task("entry extra", chat_id="1001")
        th._save_task_extra({"tokens": 999}, task_id=task_id)
        entry = th._get_history_entry(0)
        assert entry is not None
        stored = json.loads(str(entry["extra"]))
        assert stored["tokens"] == 999


class TestFileUsage:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_eviction(self):
        orig = th._MAX_FILE_USAGE_ENTRIES
        th._MAX_FILE_USAGE_ENTRIES = 3
        try:
            th._record_file_usage("a.py")
            th._record_file_usage("b.py")
            th._record_file_usage("c.py")
            th._record_file_usage("a.py")
            th._record_file_usage("d.py")
            usage = th._load_file_usage()
            assert len(usage) == 3
            assert "b.py" not in usage
        finally:
            th._MAX_FILE_USAGE_ENTRIES = orig


class TestListRecentChats:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestPrintRecentChats:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_print_empty(self, capsys):
        from kiss.agents.sorcar.cli_helpers import _print_recent_chats
        _print_recent_chats()
        assert "No chat sessions found." in capsys.readouterr().out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
