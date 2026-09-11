# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Orphan-sweep progress backfill (2026-09-11 incident regression).

A task that finished successfully lost its final result, ``steps``,
``tokens``, ``cost``, and ``end_ts`` when the commits recording them
were destroyed with the server's WAL; the next startup's orphan sweep
then relabelled the row "Task terminated unexpectedly (process
killed)" and the history UI showed "0 steps, 0 tok" for a 4-hour,
$62 task.  The sweep cannot resurrect the lost result text, but the
task's surviving ``events`` rows record its real last-known progress
— these tests pin down that :func:`_recover_orphaned_tasks` now
backfills the still-zero progress columns from that evidence.

Every test uses a REAL temporary SQLite database via a private
``KISS_HOME`` redirect.  No mocks, patches, or doubles.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: Path) -> tuple:
    """Point the persistence module at a private KISS home."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR, th._owner_state)
    kiss_dir = tmpdir / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    th._owner_state = None
    return saved


def _restore(saved: tuple) -> None:
    """Undo :func:`_redirect`."""
    th._close_db()
    (th._DB_PATH, th._db_conn, th._KISS_DIR, th._owner_state) = saved


class _BackfillTestCase(unittest.TestCase):
    """Private database per test, torn down afterwards."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_backfill_"))
        self.saved = _redirect(self.tmp)
        th._close_db()

    def tearDown(self) -> None:
        _restore(self.saved)

    def _insert_orphan_row(self, *, steps: int = 0, tokens: int = 0,
                           cost: float = 0.0, end_ts: int = 0) -> str:
        """Insert a sentinel row owned by nobody (a dead process)."""
        task_id = uuid.uuid4().hex
        db = th._get_db()
        db.execute(
            "INSERT INTO task_history (id, timestamp, task, result, owner,"
            " steps, tokens, cost, end_ts) VALUES (?, ?, ?, ?, '', ?, ?,"
            " ?, ?)",
            (task_id, time.time() - 60, "orphan under test",
             "Agent Failed Abruptly", steps, tokens, cost, end_ts),
        )
        return task_id

    def _insert_event(self, task_id: str, seq: int, event: dict,
                      timestamp: float) -> None:
        """Insert one raw event row exactly as the writer stores it."""
        db = th._get_db()
        db.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (task_id, seq, json.dumps(event), timestamp),
        )

    def _row(self, task_id: str) -> dict:
        row = th._get_db().execute(
            "SELECT result, steps, tokens, cost, end_ts FROM task_history "
            "WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row is not None
        return dict(row)


class OrphanProgressBackfillTest(_BackfillTestCase):
    """The sweep fills zero progress columns from surviving events."""

    def test_progress_recovered_from_usage_info_and_last_event(self) -> None:
        """The incident shape: usage_info then trailing deltas."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 175/10000, Context: 308,919/400,000 tokens, "
                     "Total tokens: 33,641,687, Budget: $56.4682/$1000.00, ",
             "ts": 1789130350505, "taskId": task_id},
            1789130350.505,
        )
        self._insert_event(
            task_id, 2,
            {"type": "thinking_delta", "text": " tim", "taskId": task_id},
            1789130367.993,
        )

        rewritten = th._recover_orphaned_tasks(set(), time.time())

        self.assertEqual(rewritten, 1)
        row = self._row(task_id)
        self.assertEqual(
            row["result"], "Task terminated unexpectedly (process killed)",
        )
        self.assertEqual(row["steps"], 175)
        self.assertEqual(row["tokens"], 33641687)
        self.assertAlmostEqual(row["cost"], 56.4682)
        self.assertEqual(row["end_ts"], 1789130367993)

    def test_newest_usage_info_wins(self) -> None:
        """Progress comes from the LAST usage_info, not an earlier one."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 3/100, Total tokens: 1,000, Budget: "
                     "$0.10/$5.00, "},
            100.0,
        )
        self._insert_event(
            task_id, 2,
            {"type": "usage_info",
             "text": "Steps: 9/100, Total tokens: 9,000, Budget: "
                     "$0.90/$5.00, "},
            200.0,
        )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 9)
        self.assertEqual(row["tokens"], 9000)
        self.assertAlmostEqual(row["cost"], 0.9)
        self.assertEqual(row["end_ts"], 200000)

    def test_task_without_events_keeps_zeros(self) -> None:
        """No evidence means no backfill — and no crash."""
        task_id = self._insert_orphan_row()

        rewritten = th._recover_orphaned_tasks(set(), time.time())

        self.assertEqual(rewritten, 1)
        row = self._row(task_id)
        self.assertEqual(
            row["result"], "Task terminated unexpectedly (process killed)",
        )
        self.assertEqual(row["steps"], 0)
        self.assertEqual(row["tokens"], 0)
        self.assertEqual(row["cost"], 0.0)
        self.assertEqual(row["end_ts"], 0)

    def test_events_without_usage_info_still_date_the_end(self) -> None:
        """Only ``end_ts`` is recoverable from non-usage events."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 1, {"type": "text_delta", "text": "hi"}, 123.456,
        )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["end_ts"], 123456)
        self.assertEqual(row["steps"], 0)
        self.assertEqual(row["tokens"], 0)

    def test_existing_nonzero_columns_are_not_overwritten(self) -> None:
        """A partially-persisted cleanup beats the reconstruction."""
        task_id = self._insert_orphan_row(steps=42, cost=1.5)
        other = self._insert_orphan_row(tokens=777, end_ts=999)
        for tid in (task_id, other):
            self._insert_event(
                tid, 1,
                {"type": "usage_info",
                 "text": "Steps: 7/100, Total tokens: 5,000, Budget: "
                         "$0.50/$5.00, "},
                50.0,
            )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 42)
        self.assertAlmostEqual(row["cost"], 1.5)
        self.assertEqual(row["tokens"], 5000)
        self.assertEqual(row["end_ts"], 50000)
        other_row = self._row(other)
        self.assertEqual(other_row["tokens"], 777)
        self.assertEqual(other_row["end_ts"], 999)
        self.assertEqual(other_row["steps"], 7)
        self.assertAlmostEqual(other_row["cost"], 0.5)

    def test_malformed_usage_info_is_skipped(self) -> None:
        """Unparseable usage_info contributes nothing but end_ts."""
        task_id = self._insert_orphan_row()
        db = th._get_db()
        db.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (?, 2, ?, 20.0)",
            (task_id, '{"type": "usage_info", not valid json'),
        )
        self._insert_event(
            task_id, 1,
            {"type": "usage_info", "text": "no counters in here"},
            10.0,
        )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 0)
        self.assertEqual(row["tokens"], 0)
        self.assertEqual(row["end_ts"], 20000)

    def test_scan_reaches_a_valid_event_below_many_malformed_ones(
        self,
    ) -> None:
        """No row cap: 30 junk rows above one valid event still work."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 4/100, Total tokens: 4,000, Budget: "
                     "$0.40/$5.00, "},
            10.0,
        )
        db = th._get_db()
        for seq in range(2, 32):
            db.execute(
                "INSERT INTO events (task_id, seq, event_json, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (task_id, seq, '{"type": "usage_info", junk', 10.0 + seq),
            )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 4)
        self.assertEqual(row["tokens"], 4000)
        self.assertAlmostEqual(row["cost"], 0.4)

    def test_live_monitor_aggregate_events_are_not_trusted(self) -> None:
        """The cross-task live-usage form is skipped, never written."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 11/100, Total tokens: 8,000, Budget: "
                     "$0.80/$5.00, "},
            10.0,
        )
        self._insert_event(
            task_id, 2,
            {"type": "usage_info",
             "text": "Tokens: 5,215,925, Budget: $10.5066 "
                     "(live, incl. parallel sub-agents),",
             "total_tokens": 5215925, "total_steps": 99, "cost": "$10.5066"},
            20.0,
        )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 11)
        self.assertEqual(row["tokens"], 8000)
        self.assertAlmostEqual(row["cost"], 0.8)
        self.assertEqual(row["end_ts"], 20000)

    def test_valid_but_non_object_json_is_skipped(self) -> None:
        """A JSON string mentioning usage_info must not raise."""
        task_id = self._insert_orphan_row()
        db = th._get_db()
        db.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (?, 2, ?, 20.0)",
            (task_id, json.dumps('a plain "usage_info" string')),
        )
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 2/100, Total tokens: 2,000, Budget: "
                     "$0.20/$5.00, "},
            10.0,
        )

        rewritten = th._recover_orphaned_tasks(set(), time.time())

        self.assertEqual(rewritten, 1)
        row = self._row(task_id)
        self.assertEqual(
            row["result"], "Task terminated unexpectedly (process killed)",
        )
        self.assertEqual(row["steps"], 2)

    def test_comma_only_counter_never_aborts_the_sweep(self) -> None:
        """``Total tokens: ,,,`` is skipped; the rewrite still lands."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 2,
            {"type": "usage_info", "text": "Steps: 5, Total tokens: ,,,"},
            20.0,
        )
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 3/100, Total tokens: 3,000, Budget: "
                     "$0.30/$5.00, "},
            10.0,
        )

        rewritten = th._recover_orphaned_tasks(set(), time.time())

        self.assertEqual(rewritten, 1)
        row = self._row(task_id)
        self.assertEqual(
            row["result"], "Task terminated unexpectedly (process killed)",
        )
        self.assertEqual(row["steps"], 3)
        self.assertEqual(row["tokens"], 3000)

    def test_backfill_failure_cannot_roll_back_the_sentinel_rewrite(
        self,
    ) -> None:
        """The outer per-row barrier: an OverflowError on SQLite
        binding (steps too large for a 64-bit integer) is contained,
        the failing row's rewrite survives, and the NEXT orphan is
        still backfilled."""
        overflow_id = self._insert_orphan_row()
        self._insert_event(
            overflow_id, 1,
            {"type": "usage_info",
             "text": "Steps: 999999999999999999999/9, Total tokens: 1, "
                     "Budget: $0.10/$5.00, "},
            10.0,
        )
        healthy_id = self._insert_orphan_row()
        self._insert_event(
            healthy_id, 1,
            {"type": "usage_info",
             "text": "Steps: 12/100, Total tokens: 1,200, Budget: "
                     "$1.20/$5.00, "},
            12.0,
        )

        rewritten = th._recover_orphaned_tasks(set(), time.time())

        self.assertEqual(rewritten, 2)
        overflow_row = self._row(overflow_id)
        self.assertEqual(
            overflow_row["result"],
            "Task terminated unexpectedly (process killed)",
        )
        self.assertEqual(overflow_row["steps"], 0)
        healthy_row = self._row(healthy_id)
        self.assertEqual(
            healthy_row["result"],
            "Task terminated unexpectedly (process killed)",
        )
        self.assertEqual(healthy_row["steps"], 12)
        self.assertEqual(healthy_row["tokens"], 1200)

    def test_non_usage_dict_mentioning_usage_info_is_skipped(self) -> None:
        """A dict of another type that merely mentions usage_info."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 2,
            {"type": "text_delta",
             "text": 'the "usage_info" literal inside other text'},
            20.0,
        )
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 1/100, Total tokens: 100, Budget: "
                     "$0.01/$5.00, "},
            10.0,
        )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 1)
        self.assertEqual(row["tokens"], 100)

    def test_steps_only_event_backfills_just_steps_and_end_ts(self) -> None:
        """A per-step event lacking token/budget counters still helps."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 1,
            {"type": "usage_info", "text": "Steps: 8/100, Context: full"},
            80.0,
        )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 8)
        self.assertEqual(row["tokens"], 0)
        self.assertEqual(row["cost"], 0.0)
        self.assertEqual(row["end_ts"], 80000)

    def test_null_progress_columns_are_backfilled(self) -> None:
        """A partial legacy INSERT may leave NULLs — treated as zero."""
        task_id = uuid.uuid4().hex
        db = th._get_db()
        db.execute(
            "INSERT INTO task_history (id, timestamp, task, result, owner,"
            " steps, tokens, cost, end_ts) VALUES (?, ?, ?, ?, '', NULL,"
            " NULL, NULL, NULL)",
            (task_id, time.time() - 60, "legacy nulls",
             "Agent Failed Abruptly"),
        )
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 6/100, Total tokens: 6,000, Budget: "
                     "$0.60/$5.00, "},
            60.0,
        )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 6)
        self.assertEqual(row["tokens"], 6000)
        self.assertAlmostEqual(row["cost"], 0.6)
        self.assertEqual(row["end_ts"], 60000)

    def test_live_foreign_rows_are_not_backfilled(self) -> None:
        """The sweep's liveness rules gate the backfill too."""
        task_id, _chat = th._add_task("my own live task")
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 5/100, Total tokens: 2,000, Budget: "
                     "$0.20/$5.00, "},
            30.0,
        )

        rewritten = th._recover_orphaned_tasks(set(), time.time() + 1)

        self.assertEqual(rewritten, 0)
        row = self._row(task_id)
        self.assertEqual(row["result"], "Agent Failed Abruptly")
        self.assertEqual(row["steps"], 0)
        self.assertEqual(row["end_ts"], 0)


if __name__ == "__main__":
    unittest.main()
