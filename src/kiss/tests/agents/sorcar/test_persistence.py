"""Tests for task history: SQLite storage, dedup, events, model/file usage, cleanup."""

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

    def test_add_and_load(self):
        th._add_task("my task")
        entries = th._load_history(limit=10)
        tasks = [e["task"] for e in entries]
        assert "my task" in tasks

    def test_dedup_keeps_latest(self):
        th._add_task("dup")
        time.sleep(0.01)
        th._add_task("unique")
        time.sleep(0.01)
        th._add_task("dup")
        entries = th._load_history()
        tasks = [e["task"] for e in entries]
        assert tasks.count("dup") == 1
        assert tasks[0] == "dup"

    def test_search_history(self):
        th._add_task("alpha test")
        th._add_task("beta test")
        th._add_task("gamma work")
        results = th._search_history("test", limit=10)
        tasks = [e["task"] for e in results]
        assert "alpha test" in tasks
        assert "beta test" in tasks
        assert "gamma work" not in tasks

    def test_search_empty_query(self):
        th._add_task("x")
        results = th._search_history("", limit=10)
        assert len(results) >= 1

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

    def test_load_history_limit(self):
        for i in range(5):
            th._add_task(f"t{i}")
            time.sleep(0.001)
        entries = th._load_history(limit=3)
        assert len(entries) == 3

    def test_seed_sample_tasks(self):
        entries = th._load_history()
        assert len(entries) == len(th.SAMPLE_TASKS)


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

    def test_set_and_load_events(self):
        th._add_task("my task")
        th._set_latest_chat_events([{"type": "hello"}], task="my task")
        events = th._load_task_chat_events("my task")
        assert events == [{"type": "hello"}]

    def test_set_events_updates_has_events(self):
        th._add_task("t")
        th._set_latest_chat_events([{"x": 1}], task="t", result="done")
        entry = th._load_history(limit=1)[0]
        assert entry["task"] == "t"
        assert entry["has_events"] == 1
        assert entry["result"] == "done"

    def test_set_events_no_task(self):
        th._add_task("latest")
        th._set_latest_chat_events([{"a": 1}])
        events = th._load_task_chat_events("latest")
        assert events == [{"a": 1}]

    def test_load_events_nonexistent(self):
        assert th._load_task_chat_events("nope") == []

    def test_clear_events(self):
        th._add_task("t")
        th._set_latest_chat_events([{"x": 1}], task="t")
        th._set_latest_chat_events([], task="t")
        assert th._load_task_chat_events("t") == []

    def test_save_task_result_updates_result(self):
        th._add_task("my task")
        th._save_task_result("my task", "done!")
        entry = th._load_history(limit=1)[0]
        assert entry["result"] == "done!"

    def test_save_task_result_no_matching_task(self):
        th._save_task_result("nonexistent", "result")
        # Should not raise; just returns early


class TestModelUsage:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_and_load(self):
        th._record_model_usage("gpt-4o")
        th._record_model_usage("gpt-4o")
        usage = th._load_model_usage()
        assert usage["gpt-4o"] == 2

    def test_save_and_load_last_model(self):
        th._save_last_model("claude-opus-4-6")
        assert th._load_last_model() == "claude-opus-4-6"
        th._save_last_model("gemini-2.0-flash")
        assert th._load_last_model() == "gemini-2.0-flash"

    def test_record_sets_last(self):
        th._record_model_usage("a")
        th._record_model_usage("b")
        assert th._load_last_model() == "b"

    def test_load_last_model_empty(self):
        assert th._load_last_model() == ""


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

    def test_record_and_load(self):
        th._record_file_usage("src/main.py")
        th._record_file_usage("src/main.py")
        usage = th._load_file_usage()
        assert usage["src/main.py"] == 2

    def test_recency_ordering(self):
        th._record_file_usage("a.py")
        time.sleep(0.01)
        th._record_file_usage("b.py")
        usage = th._load_file_usage()
        keys = list(usage.keys())
        assert keys.index("a.py") < keys.index("b.py")

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

    def test_removes_old_dirs(self):
        import os

        kiss_dir = th._KISS_DIR
        stale = kiss_dir / "cs-stale123"
        stale.mkdir()
        (stale / "cs-port").write_text("99999")
        old_time = time.time() - 25 * 3600
        os.utime(stale, (old_time, old_time))
        removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert removed == 1
        assert not stale.exists()

    def test_removes_stale_sorcar_data(self):
        import os

        kiss_dir = th._KISS_DIR
        sorcar_data = kiss_dir / "sorcar-data"
        sorcar_data.mkdir()
        (sorcar_data / "cs-port").write_text("99999")
        old_time = time.time() - 25 * 3600
        os.utime(sorcar_data, (old_time, old_time))
        removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert removed >= 1
        assert not sorcar_data.exists()

    def test_removes_legacy_cs_data(self):
        kiss_dir = th._KISS_DIR
        cs_data = kiss_dir / "cs-data"
        cs_data.mkdir()
        removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert removed >= 1
        assert not cs_data.exists()

    def test_removes_legacy_port_files(self):
        kiss_dir = th._KISS_DIR
        pf = kiss_dir / "cs-port-abc123"
        pf.write_text("12345")
        th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert not pf.exists()

    def test_keeps_sorcar_data_when_recent(self):
        kiss_dir = th._KISS_DIR
        sorcar_data = kiss_dir / "sorcar-data"
        sorcar_data.mkdir()
        (sorcar_data / "cs-port").write_text("99999")
        th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert sorcar_data.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
