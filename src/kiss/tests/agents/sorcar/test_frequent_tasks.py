# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the frequent_tasks table in history.db.

Verifies counter incrementing, timestamp updates and the 100-row
eviction policy (lowest count, oldest timestamp first).
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
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _rows() -> list[dict[str, object]]:
    """Every ``frequent_tasks`` row, highest count then newest first."""
    rows = th._get_db().execute(
        "SELECT task, count, timestamp FROM frequent_tasks "
        "ORDER BY count DESC, timestamp DESC"
    ).fetchall()
    return [
        {"task": r["task"], "count": r["count"], "timestamp": r["timestamp"]}
        for r in rows
    ]


class TestFrequentTasks:
    """Behavioral tests for ``_record_frequent_task``."""

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
        rows = _rows()
        assert len(rows) == 1
        assert rows[0]["task"] == "task A"
        assert rows[0]["count"] == 2
        ts = rows[0]["timestamp"]
        assert isinstance(ts, (int, float)) and ts > 0

    def test_empty_string_ignored(self) -> None:
        """Empty task strings are ignored and produce no row."""
        th._record_frequent_task("")
        assert _rows() == []

    def test_counts_per_task(self) -> None:
        """Each distinct task text keeps its own counter."""
        th._record_frequent_task("rare")
        for _ in range(3):
            th._record_frequent_task("common")
        for _ in range(2):
            th._record_frequent_task("medium")
        rows = _rows()
        assert [r["task"] for r in rows] == ["common", "medium", "rare"]
        assert [r["count"] for r in rows] == [3, 2, 1]

    def test_eviction_when_at_max(self) -> None:
        """When the table is full, the lowest-count oldest row is evicted."""
        original_max = th._MAX_FREQUENT_TASKS
        th._MAX_FREQUENT_TASKS = 3
        try:
            th._record_frequent_task("oldest")
            time.sleep(0.01)
            th._record_frequent_task("middle")
            time.sleep(0.01)
            th._record_frequent_task("middle")
            time.sleep(0.01)
            th._record_frequent_task("newest")
            time.sleep(0.01)
            th._record_frequent_task("newest")

            th._record_frequent_task("inserted")
            rows = _rows()
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
            th._record_frequent_task("third")
            rows = _rows()
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
            th._record_frequent_task("a")
            tasks = {r["task"] for r in _rows()}
            assert tasks == {"a", "b"}
        finally:
            th._MAX_FREQUENT_TASKS = original_max

    def test_below_cap_keeps_every_task(self) -> None:
        """Below the cap, every distinct task is kept."""
        for i in range(60):
            th._record_frequent_task(f"task-{i:03d}")
        assert len(_rows()) == 60

    def test_chat_run_records_frequent_task(self) -> None:
        """ChatSorcarAgent.run wires ``_record_frequent_task`` for each task."""
        th._add_task("integration task", chat_id="")
        th._record_frequent_task("integration task")
        assert any(r["task"] == "integration task" for r in _rows())
