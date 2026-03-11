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

    def test_keeps_recent_dirs(self):
        kiss_dir = th._KISS_DIR
        recent = kiss_dir / "cs-recent456"
        recent.mkdir()
        # It's brand new, should be kept
        removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert removed == 0
        assert recent.exists()

    def test_keeps_active_dirs(self):
        """Directories with active processes on their port should be kept."""
        import socket
        kiss_dir = th._KISS_DIR
        active = kiss_dir / "cs-active789"
        active.mkdir()

        # Bind a socket to simulate an active process
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        (active / "cs-port").write_text(str(port))

        # Make it old enough
        import os
        old_time = time.time() - 25 * 3600
        os.utime(active, (old_time, old_time))

        try:
            removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
            assert removed == 0
            assert active.exists()
        finally:
            sock.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
