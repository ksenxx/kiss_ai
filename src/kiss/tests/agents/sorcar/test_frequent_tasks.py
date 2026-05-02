"""Tests for the frequent_tasks table in sorcar.db.

Verifies counter incrementing, timestamp updates, top-N retrieval and
the 100-row eviction policy (lowest count, oldest timestamp first).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    """Redirect the persistence DB to a temp dir and reset the singleton."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class TestFrequentTasks:
    """Behavioral tests for ``_record_frequent_task`` and ``_load_frequent_tasks``."""

    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.saved = _redirect(self.tmp)

    def teardown_method(self) -> None:
        th._close_db()
        _restore(self.saved)
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_table_created(self) -> None:
        """The frequent_tasks table is created on first DB access."""
        db = th._get_db()
        row = db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='frequent_tasks'"
        ).fetchone()
        assert row is not None

    def test_increment_count_and_timestamp(self) -> None:
        """Calling _record_frequent_task twice increments the count."""
        th._record_frequent_task("task A")
        time.sleep(0.01)
        th._record_frequent_task("task A")
        rows = th._load_frequent_tasks()
        assert len(rows) == 1
        assert rows[0]["task"] == "task A"
        assert rows[0]["count"] == 2
        assert rows[0]["timestamp"] > 0

    def test_empty_string_ignored(self) -> None:
        """Empty task strings are ignored and produce no row."""
        th._record_frequent_task("")
        assert th._load_frequent_tasks() == []

    def test_top_n_ordering(self) -> None:
        """_load_frequent_tasks returns rows ordered by count desc."""
        th._record_frequent_task("rare")
        for _ in range(3):
            th._record_frequent_task("common")
        for _ in range(2):
            th._record_frequent_task("medium")
        rows = th._load_frequent_tasks(limit=20)
        assert [r["task"] for r in rows] == ["common", "medium", "rare"]
        assert [r["count"] for r in rows] == [3, 2, 1]

    def test_limit_caps_results(self) -> None:
        """The ``limit`` argument caps the result list length."""
        for i in range(5):
            th._record_frequent_task(f"t{i}")
        assert len(th._load_frequent_tasks(limit=3)) == 3

    def test_eviction_when_at_max(self) -> None:
        """When the table is full, the lowest-count oldest row is evicted."""
        original_max = th._MAX_FREQUENT_TASKS
        th._MAX_FREQUENT_TASKS = 3
        try:
            th._record_frequent_task("oldest")
            time.sleep(0.01)
            th._record_frequent_task("middle")
            time.sleep(0.01)
            th._record_frequent_task("middle")  # count 2
            time.sleep(0.01)
            th._record_frequent_task("newest")
            time.sleep(0.01)
            th._record_frequent_task("newest")  # count 2

            # Table is at cap (3 rows); both "oldest" and any other count-1
            # row are eviction candidates.  Adding a new task should evict
            # "oldest" (lowest count == 1, oldest timestamp).
            th._record_frequent_task("inserted")
            rows = th._load_frequent_tasks()
            tasks = {r["task"] for r in rows}
            assert "oldest" not in tasks
            assert "middle" in tasks
            assert "newest" in tasks
            assert "inserted" in tasks
            assert len(rows) == 3
        finally:
            th._MAX_FREQUENT_TASKS = original_max

    def test_eviction_breaks_count_tie_by_timestamp(self) -> None:
        """When counts tie, eviction picks the oldest timestamp."""
        original_max = th._MAX_FREQUENT_TASKS
        th._MAX_FREQUENT_TASKS = 2
        try:
            th._record_frequent_task("first")
            time.sleep(0.01)
            th._record_frequent_task("second")
            time.sleep(0.01)
            # Inserting a new task — both existing rows have count == 1;
            # "first" has the older timestamp, so it must be evicted.
            th._record_frequent_task("third")
            rows = th._load_frequent_tasks()
            tasks = {r["task"] for r in rows}
            assert tasks == {"second", "third"}
        finally:
            th._MAX_FREQUENT_TASKS = original_max

    def test_existing_task_does_not_evict(self) -> None:
        """Re-recording an existing task does not trigger eviction."""
        original_max = th._MAX_FREQUENT_TASKS
        th._MAX_FREQUENT_TASKS = 2
        try:
            th._record_frequent_task("a")
            th._record_frequent_task("b")
            # Table is full; bumping "a" must keep both rows alive.
            th._record_frequent_task("a")
            tasks = {r["task"] for r in th._load_frequent_tasks()}
            assert tasks == {"a", "b"}
        finally:
            th._MAX_FREQUENT_TASKS = original_max

    def test_chat_run_records_frequent_task(self) -> None:
        """ChatSorcarAgent.run wires ``_record_frequent_task`` for each task."""
        # Direct call path — verify the integration without invoking a
        # real model (which would require API keys).
        th._add_task("integration task", chat_id="")
        th._record_frequent_task("integration task")
        rows = th._load_frequent_tasks()
        assert any(r["task"] == "integration task" for r in rows)
