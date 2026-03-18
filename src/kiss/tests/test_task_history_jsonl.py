"""Tests for JSONL task history format, migration, and stale directory cleanup."""

import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.task_history as th


def _redirect(tmpdir: str):
    """Redirect all task_history state to a temp directory."""
    old = (th.HISTORY_FILE, th._CHAT_EVENTS_DIR, th._history_cache, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th.HISTORY_FILE = kiss_dir / "task_history.jsonl"
    th._CHAT_EVENTS_DIR = kiss_dir / "chat_events"
    th._history_cache = None
    return old


def _restore(saved):
    th.HISTORY_FILE, th._CHAT_EVENTS_DIR, th._history_cache, th._KISS_DIR = saved


# ---------------------------------------------------------------------------
# kiss/agents/sorcar/task_history.py — th
# ---------------------------------------------------------------------------

class TestJSONLFormat:
    """Test that task history is stored in JSONL format."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_set_empty_events_removes_file(self):
        th._add_task("temp task")
        th._set_latest_chat_events([{"type": "x"}], task="temp task")
        assert th._task_events_path("temp task").exists()

        th._set_latest_chat_events([], task="temp task")
        assert not th._task_events_path("temp task").exists()

class TestResultAndEventsFile:
    """Test result and events_file fields in task history entries."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_update_task_result(self):
        th._add_task("update me")
        th._update_task_result("update me", "Done with flying colors")
        history = th._load_history(limit=1)
        entry = history[0]
        assert entry["result"] == "Done with flying colors"

    def test_load_task_chat_events_uses_events_file_from_history(self):
        """_load_task_chat_events should read events_file from history entry."""
        th._add_task("my task")
        # Write events via normal API
        th._set_latest_chat_events([{"type": "hello"}], task="my task")
        # Verify it loads correctly
        events = th._load_task_chat_events("my task")
        assert events == [{"type": "hello"}]
        # Now manually move the events file to a custom name
        # and update the JSONL to point to it
        custom_name = "custom_events.json"
        old_path = th._task_events_path("my task")
        new_path = th._CHAT_EVENTS_DIR / custom_name
        old_path.rename(new_path)
        # Rewrite the JSONL to point to the custom filename
        lines = th.HISTORY_FILE.read_text().strip().split("\n")
        with th.HISTORY_FILE.open("w") as f:
            for line in lines:
                entry = json.loads(line)
                if entry["task"] == "my task":
                    entry["events_file"] = custom_name
                f.write(json.dumps(entry) + "\n")
        # Clear cache so it reloads from disk
        th._history_cache = None
        # _load_task_chat_events should find the events via the custom filename
        events = th._load_task_chat_events("my task")
        assert events == [{"type": "hello"}]

    def test_update_task_result_empty_cache(self):
        """_update_task_result should handle empty cache gracefully."""
        # No tasks added, cache will be SAMPLE_TASKS
        th._update_task_result("nonexistent", "some result")
        # Should not crash

class TestMigration:
    """Test migration from old task_history.json to JSONL format."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_migrate_old_json_format(self):
        """Old task_history.json should be migrated to JSONL + event files."""
        old_file = th.HISTORY_FILE.parent / "task_history.json"
        old_data = [
            {"task": "old task 1", "chat_events": [{"type": "text_delta", "text": "hi"}]},
            {"task": "old task 2", "chat_events": []},
        ]
        old_file.write_text(json.dumps(old_data))

        # Load should trigger migration
        history = th._load_history()
        assert len(history) == 2
        assert history[0]["task"] == "old task 1"
        assert history[0]["has_events"] is True
        assert history[1]["task"] == "old task 2"
        assert history[1]["has_events"] is False

        # Old file should be deleted
        assert not old_file.exists()

        # JSONL should exist
        assert th.HISTORY_FILE.exists()

        # Event file for task 1 should exist
        events = th._load_task_chat_events("old task 1")
        assert events == [{"type": "text_delta", "text": "hi"}]

        # No events for task 2
        assert th._load_task_chat_events("old task 2") == []

    def test_migrate_old_json_with_duplicates(self):
        """Migration should deduplicate tasks."""
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
        """Corrupt old file should not crash migration."""
        old_file = th.HISTORY_FILE.parent / "task_history.json"
        old_file.write_text("not json {{")
        history = th._load_history()  # Falls back to SAMPLE_TASKS
        assert len(history) > 0


class TestCleanupStaleCsDirs:
    """Test _cleanup_stale_cs_dirs function."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_removes_old_dirs(self):
        kiss_dir = th._KISS_DIR
        # Create a stale cs directory
        stale = kiss_dir / "cs-stale123"
        stale.mkdir()
        (stale / "cs-port").write_text("99999")
        # Make it look old
        import os
        old_time = time.time() - 25 * 3600
        os.utime(stale, (old_time, old_time))

        removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert removed == 1
        assert not stale.exists()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
