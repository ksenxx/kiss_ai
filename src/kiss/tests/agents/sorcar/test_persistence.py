"""Tests for task history: SQLite storage, events, model/file usage, cleanup."""

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
    th._DB_PATH = kiss_dir / "history.db"
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
        th._add_task("latest")
        th._set_latest_chat_events([{"a": 1}])
        events = th._load_task_chat_events("latest")
        assert events == [{"a": 1}]

    def test_save_task_result_no_matching_task(self):
        th._save_task_result(result="result", task="nonexistent")
        # Should not raise; just returns early


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


class TestGetAdjacentTaskInChat:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_chat_id(self):
        """Task with no chat_id returns None."""
        tid = th._add_task("solo task", chat_id="")
        th._set_latest_chat_events([{"type": "text_delta", "text": "hi"}], task_id=tid)
        result = th._get_adjacent_task_in_chat("solo task", "prev")
        assert result is None

    def test_single_task_in_chat(self):
        """Single task in chat: prev and next both return None."""
        tid = th._add_task("only task", chat_id="chat1")
        th._set_latest_chat_events([{"type": "text_delta", "text": "x"}], task_id=tid)
        assert th._get_adjacent_task_in_chat("only task", "prev") is None
        assert th._get_adjacent_task_in_chat("only task", "next") is None

    def test_prev_and_next(self):
        """Navigate prev/next within a 3-task chat."""
        t1 = th._add_task("task1", chat_id="chat2")
        th._set_latest_chat_events([{"type": "text_delta", "text": "a"}], task_id=t1)
        time.sleep(0.01)
        t2 = th._add_task("task2", chat_id="chat2")
        th._set_latest_chat_events([{"type": "text_delta", "text": "b"}], task_id=t2)
        time.sleep(0.01)
        t3 = th._add_task("task3", chat_id="chat2")
        th._set_latest_chat_events([{"type": "text_delta", "text": "c"}], task_id=t3)

        # From task2, prev is task1
        r = th._get_adjacent_task_in_chat("task2", "prev")
        assert r is not None
        assert r["task"] == "task1"
        evs = list(r["events"])  # type: ignore[arg-type]
        assert len(evs) == 1
        assert evs[0]["text"] == "a"

        # From task2, next is task3
        r = th._get_adjacent_task_in_chat("task2", "next")
        assert r is not None
        assert r["task"] == "task3"
        evs = list(r["events"])  # type: ignore[arg-type]
        assert evs[0]["text"] == "c"

        # From task1, prev is None
        assert th._get_adjacent_task_in_chat("task1", "prev") is None
        # From task3, next is None
        assert th._get_adjacent_task_in_chat("task3", "next") is None

    def test_nonexistent_task(self):
        """Nonexistent task returns None."""
        assert th._get_adjacent_task_in_chat("nope", "prev") is None

    def test_adjacent_no_events(self):
        """Adjacent task with no events returns empty events list."""
        th._add_task("t1", chat_id="chat3")
        time.sleep(0.01)
        th._add_task("t2", chat_id="chat3")
        r = th._get_adjacent_task_in_chat("t2", "prev")
        assert r is not None
        assert r["task"] == "t1"
        assert r["events"] == []


class TestCleanupStaleCsDirs:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_keeps_sorcar_data_when_recent(self):
        kiss_dir = th._KISS_DIR
        sorcar_data = kiss_dir / "sorcar-data"
        sorcar_data.mkdir()
        (sorcar_data / "cs-port").write_text("99999")
        th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert sorcar_data.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
