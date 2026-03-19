"""Tests for task history: JSONL format, migration, stale cleanup, scale, and atomicity."""

import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.task_history as th


def _redirect(tmpdir: str):
    old = (
        th.HISTORY_FILE,
        th._CHAT_EVENTS_DIR,
        th._history_cache,
        th._KISS_DIR,
        th._total_count,
    )
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th.HISTORY_FILE = kiss_dir / "task_history.jsonl"
    th._CHAT_EVENTS_DIR = kiss_dir / "chat_events"
    th._history_cache = None
    th._total_count = 0
    return old


def _restore(saved):
    (
        th.HISTORY_FILE,
        th._CHAT_EVENTS_DIR,
        th._history_cache,
        th._KISS_DIR,
        th._total_count,
    ) = saved


def _write_n_tasks(n: int) -> None:
    th._ensure_kiss_dir()
    with th.HISTORY_FILE.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"task": f"task-{i}", "has_events": False}))
            f.write("\n")


class TestResultAndEventsFile:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_task_chat_events_uses_events_file_from_history(self):
        th._add_task("my task")
        th._set_latest_chat_events([{"type": "hello"}], task="my task")
        events = th._load_task_chat_events("my task")
        assert events == [{"type": "hello"}]
        custom_name = "custom_events.json"
        old_path = th._task_events_path("my task")
        new_path = th._CHAT_EVENTS_DIR / custom_name
        old_path.rename(new_path)
        lines = th.HISTORY_FILE.read_text().strip().split("\n")
        with th.HISTORY_FILE.open("w") as f:
            for line in lines:
                entry = json.loads(line)
                if entry["task"] == "my task":
                    entry["events_file"] = custom_name
                f.write(json.dumps(entry) + "\n")
        th._history_cache = None
        events = th._load_task_chat_events("my task")
        assert events == [{"type": "hello"}]

    def test_update_task_result_empty_cache(self):
        th._update_task_result("nonexistent", "some result")


class TestMigration:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_migrate_old_json_format(self):
        old_file = th.HISTORY_FILE.parent / "task_history.json"
        old_data = [
            {"task": "old task 1", "chat_events": [{"type": "text_delta", "text": "hi"}]},
            {"task": "old task 2", "chat_events": []},
        ]
        old_file.write_text(json.dumps(old_data))
        history = th._load_history()
        assert len(history) == 2
        assert history[0]["task"] == "old task 1"
        assert history[0]["has_events"] is True
        assert history[1]["task"] == "old task 2"
        assert history[1]["has_events"] is False
        assert not old_file.exists()
        assert th.HISTORY_FILE.exists()
        events = th._load_task_chat_events("old task 1")
        assert events == [{"type": "text_delta", "text": "hi"}]
        assert th._load_task_chat_events("old task 2") == []

    def test_migrate_old_json_with_duplicates(self):
        old_file = th.HISTORY_FILE.parent / "task_history.json"
        old_data = [
            {"task": "dup", "chat_events": []},
            {"task": "dup", "chat_events": []},
            {"task": "unique", "chat_events": []},
        ]
        old_file.write_text(json.dumps(old_data))
        history = th._load_history()
        tasks = [e["task"] for e in history]
        assert tasks.count("dup") == 1
        assert not old_file.exists()

    def test_migrate_corrupt_old_file(self):
        old_file = th.HISTORY_FILE.parent / "task_history.json"
        old_file.write_text("not json {{")
        history = th._load_history()
        assert len(history) > 0


class TestCleanupStaleCsDirs:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
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


class TestMaxHistory:
    def test_max_history_value(self):
        assert th.MAX_HISTORY == 1_000_000


class TestLoadHistoryLimit:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestGetHistoryEntry:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_entry_no_file(self):
        entry = th._get_history_entry(100)
        assert entry is None


class TestDeduplication:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestAtomicWrite:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
