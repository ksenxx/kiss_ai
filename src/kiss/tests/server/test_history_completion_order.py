# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The CHATS panel lists tasks by completion time, most recent first.

A task's completion time is its ``end_ts``; a row without one (still
running, or a legacy row) sorts by its launch ``timestamp``.  Running
tasks still come first.  The same key orders the union of several
machines' histories (``machines.merged_history``).
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import machines as mach
from kiss.tests.server.conftest import redirect_persistence, restore_persistence


def _ids(rows: list[th._HistoryEntry]) -> list[str]:
    return [str(r["id"]) for r in rows]


class TestLoadHistoryCompletionOrder:
    """``_load_history`` orders rows by completion time."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _task(self, name: str, launched: float, end_ms: int) -> str:
        task_id, _ = th._add_task(name)
        db = th._get_db()
        db.execute(
            "UPDATE task_history SET timestamp = ?, end_ts = ? WHERE id = ?",
            (launched, end_ms, task_id),
        )
        db.commit()
        return task_id

    def test_latest_completion_first_with_launch_fallback(self) -> None:
        # Launched first but finished last.
        long_run = self._task("long", 1_000.0, 9_000_000)
        short_run = self._task("short", 2_000.0, 3_000_000)
        # No end_ts: sorts by its launch time (5000 s).
        legacy = self._task("legacy", 5_000.0, 0)
        assert _ids(th._load_history()) == [long_run, legacy, short_run]
        assert _ids(th._load_history(limit=1, offset=1)) == [legacy]

    def test_running_tasks_stay_first(self) -> None:
        done = self._task("done", 1_000.0, 9_000_000)
        running = self._task("running", 500.0, 0)
        rows = th._load_history(running_task_ids={running})
        assert _ids(rows) == [running, done]

    def test_date_range_spans_completion_times(self) -> None:
        # The From/To autofill must cover the last completion (9000 s),
        # not just the last launch, or the To filter hides that task.
        self._task("long", 1_000.0, 9_000_000)
        # No end_ts: counts at its launch (500 s), the earliest instant.
        self._task("legacy", 500.0, 0)
        assert th._history_date_range() == (500.0, 9_000.0)


def _row(task_id: str, timestamp: float, end_ms: int, running: bool = False) -> dict[str, Any]:
    return {
        "id": "chat-" + task_id, "task_id": task_id, "title": task_id,
        "timestamp": timestamp, "endTs": end_ms, "is_running": running,
    }


def test_merged_history_orders_by_completion() -> None:
    page = mach.HistoryPage([
        _row("short", 2_000.0, 3_000_000),
        _row("long", 1_000.0, 9_000_000),
        _row("legacy", 5_000.0, 0),
        _row("running", 100.0, 0, running=True),
    ])
    merged = mach.merged_history(page, [], "", "", 0, 50)
    assert [r["task_id"] for r in merged.rows] == ["running", "long", "legacy", "short"]
