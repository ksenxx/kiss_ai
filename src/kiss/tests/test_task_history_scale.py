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

    def test_limit_zero_returns_all(self):
        _write_n_tasks(50)
        result = th._load_history(limit=0)
        assert len(result) == 50

    def test_limit_returns_at_most_n(self):
        _write_n_tasks(100)
        result = th._load_history(limit=10)
        assert len(result) == 10
        # Most recent first: task-99 is the last written (newest)
        assert result[0]["task"] == "task-99"

    def test_limit_larger_than_entries(self):
        _write_n_tasks(5)
        result = th._load_history(limit=100)
        assert len(result) == 5

    def test_limit_served_from_cache(self):
        """Small limits should use the in-memory cache."""
        _write_n_tasks(20)
        # First call loads cache
        th._load_history(limit=5)
        assert th._history_cache is not None
        cached_len = len(th._history_cache)
        assert cached_len == 20  # < _RECENT_CACHE_SIZE
        # Second call with small limit should return from cache
        result = th._load_history(limit=3)
        assert len(result) == 3
        # Most recent first
        assert result[0]["task"] == "task-19"

    def test_cache_limited_to_recent_cache_size(self):
        """Cache should hold at most _RECENT_CACHE_SIZE entries."""
        n = th._RECENT_CACHE_SIZE + 100
        _write_n_tasks(n)
        th._load_history(limit=5)  # triggers cache load
        assert th._history_cache is not None
        assert len(th._history_cache) <= th._RECENT_CACHE_SIZE


class TestSearchHistory:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_search_finds_matches(self):
        _write_n_tasks(100)
        results = th._search_history("task-5", limit=50)
        # Should find task-5, task-50..task-59
        tasks = [str(e["task"]) for e in results]
        assert "task-5" in tasks
        assert "task-50" in tasks

    def test_search_respects_limit(self):
        _write_n_tasks(100)
        results = th._search_history("task-", limit=10)
        assert len(results) == 10

    def test_search_empty_query_returns_recent(self):
        _write_n_tasks(20)
        results = th._search_history("", limit=5)
        assert len(results) == 5
        # Most recent first
        assert results[0]["task"] == "task-19"

    def test_search_no_match(self):
        _write_n_tasks(10)
        results = th._search_history("nonexistent", limit=10)
        assert len(results) == 0

    def test_search_case_insensitive(self):
        th._ensure_kiss_dir()
        with th.HISTORY_FILE.open("w") as f:
            f.write(json.dumps({"task": "Hello World", "has_events": False}))
            f.write("\n")
        results = th._search_history("hello", limit=10)
        assert len(results) == 1
        assert results[0]["task"] == "Hello World"


class TestCountHistory:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_count_empty_initializes_sample(self):
        count = th._count_history()
        assert count == len(th.SAMPLE_TASKS)

    def test_count_after_writes(self):
        _write_n_tasks(42)
        th._history_cache = None  # reset cache
        th._load_history(limit=1)  # trigger cache refresh
        count = th._count_history()
        assert count == 42

    def test_count_no_file(self):
        # No file, triggers sample task creation
        count = th._count_history()
        assert count > 0


class TestGetHistoryEntry:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_entry_from_cache(self):
        _write_n_tasks(10)
        entry = th._get_history_entry(0)
        assert entry is not None
        # idx=0 is most recent; task-9 is last written (newest)
        assert entry["task"] == "task-9"

    def test_get_entry_beyond_cache(self):
        n = th._RECENT_CACHE_SIZE + 50
        _write_n_tasks(n)
        # idx=0 is most recent (task-(n-1)), idx=k is task-(n-1-k)
        idx = th._RECENT_CACHE_SIZE + 10
        entry = th._get_history_entry(idx)
        assert entry is not None
        assert entry["task"] == f"task-{n - 1 - idx}"

    def test_get_entry_out_of_range(self):
        _write_n_tasks(5)
        entry = th._get_history_entry(100)
        assert entry is None

    def test_get_negative_index(self):
        _write_n_tasks(5)
        entry = th._get_history_entry(-1)
        assert entry is None

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

    def test_add_task_deduplicates(self):
        th._add_task("first")
        th._add_task("second")
        th._add_task("first")  # duplicate appended at end
        history = th._load_history(limit=0)
        tasks = [str(e["task"]) for e in history]
        assert tasks.count("first") == 1
        # "first" was appended last, so it's most recent
        assert tasks[0] == "first"

    def test_file_dedup_on_read(self):
        """Duplicate entries in file are deduped on read."""
        th._ensure_kiss_dir()
        with th.HISTORY_FILE.open("w") as f:
            f.write(json.dumps({"task": "dup", "has_events": False}) + "\n")
            f.write(json.dumps({"task": "dup", "has_events": False}) + "\n")
            f.write(json.dumps({"task": "unique", "has_events": False}) + "\n")
        result = th._load_history(limit=0)
        tasks = [str(e["task"]) for e in result]
        assert tasks.count("dup") == 1


class TestAtomicWrite:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_entries_creates_file(self):
        entries = [{"task": f"t{i}", "has_events": False} for i in range(5)]
        th._write_entries(entries)
        assert th.HISTORY_FILE.exists()
        lines = th.HISTORY_FILE.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_write_entries_atomic(self):
        """No temp files should be left after write."""
        entries = [{"task": "test", "has_events": False}]
        th._write_entries(entries)
        # No .tmp files should remain
        tmp_files = list(th.HISTORY_FILE.parent.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestStressLargeHistory:
    """Stress test with a moderately large history."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_5000_entries_load_recent(self):
        """Loading recent entries from 5000-entry file is fast."""
        _write_n_tasks(5000)
        result = th._load_history(limit=100)
        assert len(result) == 100
        # Most recent first: task-4999 is the last written
        assert result[0]["task"] == "task-4999"

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

    def test_5000_entries_count(self):
        _write_n_tasks(5000)
        th._history_cache = None
        th._load_history(limit=1)
        count = th._count_history()
        assert count == 5000

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
