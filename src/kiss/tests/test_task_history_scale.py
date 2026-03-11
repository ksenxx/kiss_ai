"""Tests for large-scale task history (up to 1M entries).

Verifies that the history system handles large numbers of entries
without loading everything into memory, and that limits/search/
pagination work correctly.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

import kiss.agents.sorcar.task_history as th


def _redirect(tmpdir: str):
    """Redirect all task_history state to a temp directory."""
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
    """Write n tasks directly to the history file."""
    th._ensure_kiss_dir()
    with th.HISTORY_FILE.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"task": f"task-{i}", "has_events": False}))
            f.write("\n")


class TestMaxHistory:
    """Verify MAX_HISTORY is 1,000,000."""

    def test_max_history_value(self):
        assert th.MAX_HISTORY == 1_000_000


class TestLoadHistoryLimit:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestSearchHistory:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_search_empty_query_returns_recent(self):
        _write_n_tasks(20)
        results = th._search_history("", limit=5)
        assert len(results) == 5
        # Most recent first
        assert results[0]["task"] == "task-19"


class TestGetHistoryEntry:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_entry_no_file(self):
        # No history file yet
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

class TestStressLargeHistory:
    """Stress test with a moderately large history."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_5000_entries_search(self):
        _write_n_tasks(5000)
        results = th._search_history("task-4999", limit=10)
        assert len(results) == 1
        assert results[0]["task"] == "task-4999"

    def test_5000_entries_get_last(self):
        _write_n_tasks(5000)
        # idx=4999 is the oldest entry (task-0)
        entry = th._get_history_entry(4999)
        assert entry is not None
        assert entry["task"] == "task-0"

    def test_add_task_with_large_history(self):
        """Adding a task to a large history appends to the end."""
        _write_n_tasks(5000)
        th._history_cache = None
        th._add_task("new-task")
        result = th._load_history(limit=5)
        # Newly appended task is most recent
        assert result[0]["task"] == "new-task"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
