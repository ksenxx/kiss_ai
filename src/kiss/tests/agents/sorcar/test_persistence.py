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

    def test_duplicate_tasks_all_kept(self):
        th._add_task("dup")
        time.sleep(0.01)
        th._add_task("unique")
        time.sleep(0.01)
        th._add_task("dup")
        entries = th._load_history()
        tasks = [e["task"] for e in entries]
        assert tasks.count("dup") == 2
        assert tasks[0] == "dup"

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

    def test_empty_db(self):
        assert th._list_recent_chats() == []

    def test_multiple_chats_ordered_by_recency(self):
        # Chat A with 2 tasks
        tid1 = th._add_task("task A1", chat_id="aaa")
        th._save_task_result(result="result A1", task_id=tid1)
        time.sleep(0.01)
        tid2 = th._add_task("task A2", chat_id="aaa")
        th._save_task_result(result="result A2", task_id=tid2)
        time.sleep(0.01)
        # Chat B with 1 task (more recent)
        tid3 = th._add_task("task B1", chat_id="bbb")
        th._save_task_result(result="result B1", task_id=tid3)

        chats = th._list_recent_chats(limit=10)
        assert len(chats) == 2
        # Most recent chat first
        assert chats[0]["chat_id"] == "bbb"
        assert chats[1]["chat_id"] == "aaa"
        # Chat A tasks in chronological order
        a_tasks = chats[1]["tasks"]
        assert isinstance(a_tasks, list)
        assert len(a_tasks) == 2
        assert a_tasks[0]["task"] == "task A1"
        assert a_tasks[1]["task"] == "task A2"

    def test_limit(self):
        for i in range(5):
            th._add_task(f"task {i}", chat_id=f"chat{i}")
            time.sleep(0.01)
        chats = th._list_recent_chats(limit=3)
        assert len(chats) == 3

    def test_excludes_empty_chat_id(self):
        th._add_task("no chat id", chat_id="")
        th._add_task("has chat id", chat_id="ccc")
        chats = th._list_recent_chats()
        assert len(chats) == 1
        assert chats[0]["chat_id"] == "ccc"


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

    def test_print_with_data(self, capsys):
        from kiss.agents.sorcar.cli_helpers import _print_recent_chats
        tid = th._add_task("my task", chat_id="abc123")
        th._save_task_result(result="my result", task_id=tid)
        _print_recent_chats()
        out = capsys.readouterr().out
        assert "abc123" in out
        assert "my task" in out
        assert "my result" in out

    def test_print_truncates_long_text(self, capsys):
        from kiss.agents.sorcar.cli_helpers import _print_recent_chats
        long_task = "x" * 300
        tid = th._add_task(long_task, chat_id="trunc")
        th._save_task_result(result="r" * 300, task_id=tid)
        _print_recent_chats()
        out = capsys.readouterr().out
        assert "..." in out


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
