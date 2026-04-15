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
        task_id, _ = th._add_task("latest", chat_id=1004)
        th._set_latest_chat_events([{"a": 1}], task_id=task_id)
        result = th._load_latest_chat_events_by_chat_id(1004)
        assert result is not None
        events = result["events"]
        assert isinstance(events, list)
        assert events == [{"a": 1}]

    def test_load_chat_events_includes_extra(self):
        task_id, _ = th._add_task("extra-task", chat_id=1005)
        th._set_latest_chat_events([{"b": 2}], task_id=task_id)
        extra = {"model": "gpt-4o", "is_worktree": True, "is_parallel": False}
        th._save_task_extra(extra, task_id=task_id)
        result = th._load_latest_chat_events_by_chat_id(1005)
        assert result is not None
        loaded = json.loads(str(result["extra"]))
        assert loaded["model"] == "gpt-4o"
        assert loaded["is_worktree"] is True
        assert loaded["is_parallel"] is False

    def test_save_task_result_no_matching_task(self):
        th._save_task_result(result="result", task="nonexistent")
        # Should not raise; just returns early


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
        task_id, _ = th._add_task("extra test task", chat_id=1002)
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
        # Should not raise; just returns early
        th._save_task_extra({"model": "x"}, task="nonexistent")

    def test_extra_default_empty(self):
        th._add_task("no extra")
        entries = th._load_history(limit=1)
        assert entries[0]["extra"] == ""

    def test_extra_in_search_results(self):
        task_id, _ = th._add_task("searchable extra", chat_id=1003)
        th._save_task_extra({"model": "test-model"}, task_id=task_id)
        results = th._search_history("searchable")
        assert len(results) == 1
        stored = json.loads(str(results[0]["extra"]))
        assert stored["model"] == "test-model"

    def test_extra_in_get_history_entry(self):
        task_id, _ = th._add_task("entry extra", chat_id=1001)
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



class TestMigrateMissingColumns:
    """Verify ALTER TABLE migration adds columns missing from old schemas."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_old_schema(self) -> None:
        """Create a DB with the minimal old schema (missing newer columns)."""
        import sqlite3

        th._ensure_kiss_dir()
        conn = sqlite3.connect(str(th._DB_PATH))
        conn.executescript("""
            CREATE TABLE task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                task TEXT NOT NULL
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL REFERENCES task_history(id),
                seq INTEGER NOT NULL,
                event_json TEXT NOT NULL
            );
            CREATE TABLE model_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL UNIQUE,
                count INTEGER DEFAULT 0
            );
            CREATE TABLE file_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                count INTEGER DEFAULT 0
            );
        """)
        # Insert a row using only the old columns
        conn.execute(
            "INSERT INTO task_history (timestamp, task) VALUES (?, ?)",
            (time.time(), "old task"),
        )
        conn.commit()
        conn.close()

    def test_migration_adds_missing_columns(self):
        self._create_old_schema()
        # Opening the DB triggers _init_tables → _migrate_tables
        task_id, _ = th._add_task("new task", chat_id=1000)
        th._save_task_result("done", task_id=task_id)
        th._save_task_extra({"model": "gpt-4o"}, task_id=task_id)
        entries = th._load_history(limit=10)
        assert len(entries) == 2
        new_entry = entries[0]
        assert new_entry["chat_id"] == 1000
        assert new_entry["result"] == "done"
        stored = json.loads(str(new_entry["extra"]))
        assert stored["model"] == "gpt-4o"
        # Old row should have defaults for the new columns
        old_entry = entries[1]
        assert old_entry["chat_id"] == 0
        assert old_entry["extra"] == ""
        assert old_entry["has_events"] == 0
        assert old_entry["result"] == ""

    def test_migration_model_usage_is_last(self):
        self._create_old_schema()
        th._save_last_model("claude-opus-4-6")
        assert th._load_last_model() == "claude-opus-4-6"

    def test_migration_file_usage_last_used(self):
        self._create_old_schema()
        th._record_file_usage("test.py")
        usage = th._load_file_usage()
        assert "test.py" in usage

    def test_migration_idempotent(self):
        """Running migration twice does not fail or duplicate columns."""
        self._create_old_schema()
        # First open triggers migration
        th._add_task("first")
        # Close and reopen to trigger migration again
        th._close_db()
        th._add_task("second")
        entries = th._load_history(limit=10)
        assert any(e["task"] == "first" for e in entries)
        assert any(e["task"] == "second" for e in entries)


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
