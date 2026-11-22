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

import itertools
import json
import shutil
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.server.json_printer import JsonPrinter
from kiss.tests.persistence_helpers import close_db


def _redirect(tmpdir: Path) -> tuple:
    """Point the persistence module at a private KISS home."""
    saved = (th._DB_PATH, th._KISS_DIR, th._owner_state)
    kiss_dir = tmpdir / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._owner_state = None
    return saved


def _restore(saved: tuple) -> None:
    """Undo :func:`_redirect`, closing the liveness marker minted in the private home."""
    th._release_owner_marker()
    close_db()
    (th._DB_PATH, th._KISS_DIR, th._owner_state) = saved


class _BackfillTestCase(unittest.TestCase):
    """Private database per test, torn down afterwards."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_backfill_"))
        self.saved = _redirect(self.tmp)
        close_db()

    def tearDown(self) -> None:
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

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
        """No evidence means no backfill — and no crash.

        The row is finalized all the same (its tags are written), so
        the next sweep, which also picks up unfinalized rows without an
        ``end_ts`` (A11-3 fable), leaves it alone.
        """
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
        self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 0)

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

    def test_an_event_time_past_the_date_range_dates_nothing(self) -> None:
        """No reader accepts such an ``end_ts`` and SQLite refuses the largest: none is written."""
        usage = {"type": "usage_info",
                 "text": "Steps: 3/100, Total tokens: 1,000, Budget: $0.10/$5.00, "}
        task_ids = [self._insert_orphan_row() for _ in range(2)]
        for task_id, seconds in zip(task_ids, (1e300, 8.64e12 + 1)):
            self._insert_event(task_id, 1, usage, seconds)

        self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 2)

        rows = [self._row(task_id) for task_id in task_ids]
        self.assertEqual([(row["end_ts"], row["steps"]) for row in rows], [(0, 3), (0, 3)])
        # Finalized all the same, so the next sweep leaves both alone.
        self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 0)

    def test_recorded_columns_are_raised_never_lowered(self) -> None:
        """What a cleanup recorded before the kill stays where it is more than the events
        show and is raised where it is less, the same whether the events were in at the
        sweep or replayed after it (filling only a zero column made the two disagree)."""
        usage = {"type": "usage_info",
                 "text": "Steps: 7/100, Total tokens: 5,000, Budget: $0.50/$5.00, "}
        task_id = self._insert_orphan_row(steps=42, cost=1.5)
        other = self._insert_orphan_row(tokens=777, end_ts=999)
        for tid in (task_id, other):
            self._insert_event(tid, 1, usage, 50.0)
        swept_first = self._insert_orphan_row(tokens=777, end_ts=999)

        th._recover_orphaned_tasks(set(), time.time())
        th._journal_failed_events(
            [(swept_first, json.dumps(usage), 50.0, str(th._DB_PATH))], 1)
        th._replay_failed_events()

        row = self._row(task_id)
        self.assertEqual(row["steps"], 42)
        self.assertAlmostEqual(row["cost"], 1.5)
        self.assertEqual(row["tokens"], 5000)
        self.assertEqual(row["end_ts"], 50000)
        for tid in (other, swept_first):
            other_row = self._row(tid)
            self.assertEqual(other_row["tokens"], 5000)
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
        """A JSON array naming usage_info must not raise (a string's escaped quotes never
        pass the SQL filter, so it would not reach the check)."""
        task_id = self._insert_orphan_row()
        db = th._get_db()
        db.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (?, 2, ?, 20.0)",
            (task_id, json.dumps(["usage_info"])),
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
        """A count SQLite cannot store (steps too large for a 64-bit
        integer) reads as zero, the row's rewrite survives, and the
        NEXT orphan is still backfilled.  The per-row ``except``
        barrier behind it is reached only by a database error now,
        which no test can cause without a double."""
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
        """A dict of another type that names usage_info and carries totals of its own."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 2,
            {"type": "text_delta", "text": "", "about": "usage_info",
             "total_steps": 99, "total_tokens": 9, "cost_usd": 9.0},
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


class _CapturePrinter(JsonPrinter):
    """A real ``JsonPrinter`` whose broadcasts are collected, not sent."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def broadcast(self, event: dict) -> None:  # type: ignore[override]
        self.events.append(dict(event))


class ContinuationTotalsBackfillTest(_BackfillTestCase):
    """A killed continuation session recovers the task total, not its own."""

    def test_structured_task_totals_beat_the_session_text(self) -> None:
        """Session 0 banked $10 / 1M tokens / 50 steps; the killed session 1
        printed its own "Budget: $2.0000" text.  The printer rewrote the
        event's structured fields to the task totals, which must win."""
        task_id = self._insert_orphan_row()
        printer = _CapturePrinter()
        printer._thread_local.task_id = task_id
        printer.set_usage_offsets(task_id, 10.0, 1_000_000, 50)
        printer.print(
            "Steps: 7/10000, Context: 50,000/1,000,000 tokens, "
            "Total tokens: 200,000, Budget: $2.0000/$990.00, ",
            type="usage_info", total_tokens=200_000, cost="$2.0000",
            total_steps=7,
        )
        self._insert_event(task_id, 1, printer.events[-1], 300.0)

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 57)
        self.assertEqual(row["tokens"], 1_200_000)
        self.assertAlmostEqual(row["cost"], 12.0)

    def test_malformed_structured_fields_fall_back_to_the_text(self) -> None:
        """Non-numeric or boolean structured fields are ignored."""
        task_id = self._insert_orphan_row()
        self._insert_event(
            task_id, 1,
            {"type": "usage_info",
             "text": "Steps: 4/100, Total tokens: 4,000, Budget: $0.40/$5.00, ",
             "total_steps": True, "total_tokens": "many", "cost": "N/A"},
            100.0,
        )

        th._recover_orphaned_tasks(set(), time.time())

        row = self._row(task_id)
        self.assertEqual(row["steps"], 4)
        self.assertEqual(row["tokens"], 4000)
        self.assertAlmostEqual(row["cost"], 0.4)


class OrphanCostFloorTest(_BackfillTestCase):
    """A recovered row's cost is at least its own records plus its sub-agents' spend."""

    def _insert_child(self, parent: str, cost: float, *, dead: bool = False) -> str:
        """Insert a sub-agent row of *parent*: finished, or dead when *dead*."""
        task_id = uuid.uuid4().hex
        th._get_db().execute(
            "INSERT INTO task_history (id, timestamp, task, result, owner, cost,"
            " end_ts, tags, parent_task_id) VALUES (?, ?, ?, ?, '', ?, ?, ?, ?)",
            (task_id, time.time() - 30, "sub-agent",
             "Agent Failed Abruptly" if dead else "done", cost,
             0 if dead else 1789130360000, "" if dead else "work", parent),
        )
        return task_id

    def _llm_call(self, task_id: str, seq: int, model: str, cost: float) -> None:
        self._insert_event(
            task_id, seq,
            {"type": "llm_call", "model": model, "cost": cost,
             "input_tokens": 100, "output_tokens": 10},
            1789130300.0 + seq,
        )

    def _usage(self, task_id: str, seq: int, cost: float) -> None:
        self._insert_event(
            task_id, seq,
            {"type": "usage_info",
             "text": f"Steps: 3/100, Total tokens: 1,000, Budget: ${cost:.4f}/$500.00, ",
             "total_steps": 3, "total_tokens": 1000, "cost": f"{cost:.4f}",
             "cost_usd": cost},
            1789130300.0 + seq,
        )

    def test_terminal_run_without_usage_events_keeps_its_records(self) -> None:
        """The 8f05ebc0 shape: llm_calls, a charge and a finished sub-agent; no usage_info."""
        task_id = self._insert_orphan_row()
        self._llm_call(task_id, 1, "gpt-6-astra", 0.4262975)
        self._insert_event(
            task_id, 2,
            {"type": "gate_stage", "stage": "charge", "model": "claude-sonnet-5-5",
             "cost_usd": 0.0398705, "tokens": 14479},
            1789130310.0,
        )
        self._insert_child(task_id, 1.046220858)

        th._recover_orphaned_tasks(set(), time.time())

        self.assertAlmostEqual(
            self._row(task_id)["cost"], 0.4262975 + 0.0398705 + 1.046220858, places=9,
        )

    def test_sub_agents_that_died_with_the_parent_are_counted(self) -> None:
        """The 3bb386ba shape: the parent's last event predates its sub-agents' spend."""
        parent = self._insert_orphan_row()
        self._llm_call(parent, 1, "claude-opus-5-5", 3.5)
        self._usage(parent, 2, 4.0)  # held a gate charge or an early fold
        dead = self._insert_child(parent, 0.0, dead=True)
        self._llm_call(dead, 1, "claude-opus-5-5", 150.0)
        self._usage(dead, 2, 200.0)  # its own sub-agents' folds included
        self._insert_child(parent, 20.0)  # finished, but folded after the last event

        th._recover_orphaned_tasks(set(), time.time())

        self.assertAlmostEqual(self._row(dead)["cost"], 200.0)
        self.assertAlmostEqual(self._row(parent)["cost"], 3.5 + 200.0 + 20.0)

    def test_the_floor_never_lowers_a_recovered_cost(self) -> None:
        """A last event above the records (spend that left no record here) wins."""
        task_id = self._insert_orphan_row()
        self._llm_call(task_id, 1, "claude-opus-5-5", 1.0)
        self._usage(task_id, 2, 5.0)
        self._insert_child(task_id, 2.0)

        th._recover_orphaned_tasks(set(), time.time())

        self.assertAlmostEqual(self._row(task_id)["cost"], 5.0)

    def test_a_remote_sub_agent_counts_its_folded_record(self) -> None:
        """A remote sub-agent's mirror row may be ahead of what was folded: the record counts."""
        parent = self._insert_orphan_row()
        self._llm_call(parent, 1, "claude-opus-5-5", 1.0)
        mirror = self._insert_child(parent, 3.0)
        th._get_db().execute(
            "INSERT INTO remote_subtasks (task_id, parent_task_id, machine, cost, tokens,"
            " steps, timestamp) VALUES (?, ?, 'gpu-box', ?, 0, 0, ?)",
            (mirror, parent, 2.0, time.time()),
        )

        th._recover_orphaned_tasks(set(), time.time())

        self.assertAlmostEqual(self._row(parent)["cost"], 3.0)

    def test_records_replayed_from_the_journal_after_the_sweep_reach_the_floor(self) -> None:
        """A swept row's record journalled before the kill lands after the sweep: a killed
        sub-agent's, that of a task whose result was saved before the kill (floored at the
        sweep, but again only under the killed text: S10 F1), and that of a row a build
        without the sweep's mark left killed."""
        parent = self._insert_orphan_row()
        self._llm_call(parent, 1, "claude-opus-5-5", 1.0)
        child = self._insert_child(parent, 0.0, dead=True)
        self._llm_call(child, 1, "claude-opus-5-5", 0.5)
        saved = self._insert_orphan_row()
        th._save_task_result(result="done before the kill", task_id=saved, success=True)
        unmarked = self._insert_child("a-parent-since-deleted", 0.0)
        th._get_db().execute(
            "UPDATE task_history SET result = ? WHERE id = ?", (th._KILLED_RESULT, unmarked))
        th._recover_orphaned_tasks(set(), time.time())
        self.assertAlmostEqual(self._row(child)["cost"], 0.5)
        self.assertAlmostEqual(self._row(parent)["cost"], 1.5)
        self._journal(
            (child, saved, unmarked),
            {"type": "llm_call", "model": "claude-opus-5-5", "cost": 2.0,
             "input_tokens": 100, "output_tokens": 10},
        )

        th._replay_failed_events()

        self.assertAlmostEqual(self._row(child)["cost"], 2.5)
        self.assertAlmostEqual(self._row(parent)["cost"], 3.5)
        self.assertAlmostEqual(self._row(saved)["cost"], 2.0)
        self.assertAlmostEqual(self._row(unmarked)["cost"], 2.0)
        self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 0)
        self.assertAlmostEqual(self._row(parent)["cost"], 3.5)  # nothing counted twice

    def _journal(self, task_ids: tuple[str, ...], *events: dict, at: float = 1789130400.0) -> None:
        """Journal *events* for each of *task_ids*, as writes the database refused."""
        th._journal_failed_events(
            [(task_id, json.dumps(event), at + seq, str(th._DB_PATH))
             for task_id in task_ids for seq, event in enumerate(events)],
            1,
        )

    def test_a_row_swept_before_any_of_its_events_takes_its_progress_from_the_replay(
        self,
    ) -> None:
        """A killed sub-agent whose every event was still journalled at the sweep (S10 F1)."""
        parent = self._insert_orphan_row()
        self._llm_call(parent, 1, "claude-opus-5-5", 1.0)
        child = self._insert_child(parent, 0.0, dead=True)
        th._recover_orphaned_tasks(set(), time.time())
        self.assertEqual(self._row(child)["end_ts"], 0)  # finalized, with no event to date it
        self._journal(
            (child,),
            {"type": "llm_call", "model": "claude-opus-5-5", "cost": 2.5,
             "input_tokens": 100, "output_tokens": 10},
            {"type": "usage_info", "text": "", "total_steps": 3, "total_tokens": 110,
             "cost_usd": 2.5},
        )

        th._replay_failed_events()

        row = self._row(child)
        self.assertEqual((row["steps"], row["tokens"], row["end_ts"]), (3, 110, 1789130401000))
        self.assertAlmostEqual(row["cost"], 2.5)
        (model_costs,) = th._get_db().execute(
            "SELECT model_costs FROM task_history WHERE id = ?", (child,)).fetchone()
        self.assertEqual(json.loads(model_costs), {"claude-opus-5-5": [2.5, 110]})
        self.assertAlmostEqual(self._row(parent)["cost"], 3.5)  # the killed parent's floor
        # The row has its end now: a later replay and the next sweep leave it as it is.
        self._journal((child,), {"type": "text_delta", "text": "late"})
        th._replay_failed_events()
        self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 0)
        self.assertEqual(self._row(child), row)
        self.assertAlmostEqual(self._row(parent)["cost"], 3.5)

    def _progress(self, task_id: str) -> tuple[int, int, float, int]:
        row = self._row(task_id)
        return row["steps"], row["tokens"], row["cost"], row["end_ts"]

    def test_a_swept_row_takes_the_usage_each_later_snapshot_brings(self) -> None:
        """The first snapshot dates the row; the usage of the next ones still reaches it.

        A text-only snapshot a refused replay left claimed with counters journalled behind
        it, then larger counters in a later replay: each raises the row (S10 F1 again).
        """
        task_id = self._insert_orphan_row()
        th._recover_orphaned_tasks(set(), time.time())
        path = th._failed_events_path(str(th._DB_PATH))
        self._journal((task_id,), {"type": "text_delta", "text": "first chunk"})
        with th._journal_file_lock(path) as locked:
            self.assertTrue(locked)
            self.assertEqual(len(th._claim_journal_snapshots(path)), 1)
        self._journal((task_id,), th.usage_totals_event(task_id, 40, 1.0, 1))

        th._replay_failed_events()

        self.assertEqual(self._progress(task_id), (1, 40, 1.0, 1789130400000))
        self._journal((task_id,), th.usage_totals_event(task_id, 110, 2.5, 3))
        th._replay_failed_events()
        self.assertEqual(self._progress(task_id), (3, 110, 2.5, 1789130400000))
        self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 0)

    def test_usage_replayed_after_the_sweep_counts_each_remote_charge_once(self) -> None:
        """The replayed totals hold the charge banked before them: only the other is added,
        to a row its charges alone had put above those totals.

        Nothing is added to a finished row whose newest usage event is its own totals after
        a late charge (they hold every charge of the row, and show its cost rounded up to
        four decimals), and a row whose process lives is that process's to write.
        """
        db = th._get_db()
        swept = self._insert_orphan_row()
        db.execute(
            "INSERT INTO remote_charges (task_id, parent_task_id, cost, tokens, steps, banked,"
            " banked_cost, banked_tokens, banked_steps) VALUES"
            " ('banked-child', ?, 0.5, 20, 1, 1, 1.0, 40, 1),"
            " ('unbanked-child', ?, 3.0, 500, 5, 0, 0, 0, 0)",
            (swept, swept),
        )
        th._recover_orphaned_tasks(set(), time.time())
        self.assertEqual(self._progress(swept), (6, 520, 3.5, 0))  # the charges, on no event
        finished = self._insert_child("", 2.99996)
        updated, _running = th._add_late_task_usage(finished, 25, 0.25, 1)
        shown = th.usage_totals_event(*updated[0])
        del shown["cost_usd"]  # as an older version published it: "$3.2500"
        th._append_chat_event(shown, task_id=finished)
        live, _chat = th._add_task("finished by this process")
        db.execute(
            "UPDATE task_history SET result = 'done', tags = 'work', steps = 1, tokens = 10,"
            " cost = 0.5, end_ts = 1789130360000 WHERE id = ?",
            (live,),
        )
        th._flush_chat_events()
        self._journal(
            (swept, live),
            {"type": "usage_info", "text": "", "total_steps": 3, "total_tokens": 110,
             "cost_usd": 2.5, "banked": [2.5, 110, 3]},
        )
        self._journal((finished,), {"type": "text_delta", "text": "late"})

        th._replay_failed_events()

        self.assertEqual(self._progress(swept), (8, 610, 5.5, 1789130400000))
        self.assertEqual(self._progress(finished), (1, 25, 2.99996 + 0.25, 1789130360000))
        self.assertEqual(self._progress(live), (1, 10, 0.5, 1789130360000))
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM remote_charges WHERE settled = 0").fetchone()[0], 0)

    def test_closing_totals_replayed_after_the_sweep_settled_larger_charges(self) -> None:
        """The task's last publication arrives after the sweep settled charges larger than it.

        The task banked a remote charge and published its closing totals, which hold it; a
        larger charge was reported and never banked; the sweep found no event and settled
        both.  The totals replayed afterwards are below the row in steps and tokens, yet the
        row is those totals plus the charge they do not hold (it kept the charges alone: S10
        F1), for the final save's totals event and for the run's closing event alike.  The
        latter is sent with the ledger's own totals, which is no reason for a warning.
        """
        for run_closing in (False, True):
            parent = self._insert_orphan_row()
            child = f"remote-of-{parent}"
            th._record_remote_subtask(child, parent, "box", "claude-opus-5-5", 0.0, 0, 0)
            first = th._charge_remote_subtask(child, 0.5, 20, 1, {})
            assert first is not None
            th._mark_remote_charge_banked(first[-1], 1.0, 40, 1)
            closing = th.usage_totals_event(parent, 110, 2.5, 3)  # as ChatSorcarAgent saves it
            if run_closing:  # as RelentlessAgent._publish_usage_totals sends it
                closing = {"type": "usage_info", "total_tokens": 110, "cost": "$2.5000",
                           "text": "Steps: 3, Total tokens: 110, Budget: $2.5000, ",
                           "cost_usd": 2.5, "total_steps": 3}
            late = th._charge_remote_subtask(child, 3.5, 520, 6, {})
            assert late is not None and late[1:4] == (3.0, 500, 5)
            self._journal((parent,), {"type": "text_delta", "text": "last words"}, closing)
            self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 1)
            self.assertEqual(self._progress(parent), (6, 520, 3.5, 0))

            if run_closing:
                with self.assertNoLogs(th.logger, level="WARNING"):
                    th._replay_failed_events()
            else:
                th._replay_failed_events()

            self.assertEqual(self._progress(parent), (8, 610, 5.5, 1789130401000))

    def test_a_swept_row_holds_what_its_events_and_charges_show_in_every_arrival_order(
        self,
    ) -> None:
        """The sweep, two journal snapshots and a remote report, in each of their 24 orders.

        The task banked two remote charges, published a per-step event that holds the first
        (its ``banked`` stamp says so) and then its closing totals, which hold both and carry
        no stamp; a third charge is reported and never banked.  From the sweep on the row is
        what the evidence so far shows: the latest publication in, plus the charges it does
        not hold, each once.  Two of the orders lost usage (S10 F1): the closing totals
        replayed after the sweep had settled charges larger than them, and the events
        arriving one snapshot at a time.  The totals a finished row republishes after a late
        charge are not evidence, and a swept row no event dated takes the charge at once.

        The task's parent died with it, having published its own ledger.  It holds the charge
        no ledger held, once, whichever of the two rows the sweep takes first: the walk a late
        add makes up the chain stopped at the task while it was unfinished and goes on at the
        sweep (the charge reached the parent only when the sweep came before the report).  The
        finished task above them takes it too, and its transcript ends with its new totals,
        queued once the sweep has committed (a rolled-back sweep leaves no event behind; only
        an interrupt can roll one back, which no test raises without a double).  The walk stops
        at the running task above that: it folds its sub-task's totals, the charge in them, so
        nothing is left owed there (the sweep left a record its final save counted again).
        """
        db = th._get_db()
        step = {"type": "usage_info", "total_steps": 2, "total_tokens": 60, "cost_usd": 1.5,
                "text": "Steps: 1/100, Total tokens: 40, Budget: $1.00/$5.00, ",
                "banked": [0.5, 20, 1]}
        roots = []
        live, _chat = th._add_task("still running")  # owned by this process
        orders = itertools.permutations(("sweep", "step", "closing", "report"))
        for index, order in enumerate(orders):
            rows = [self._insert_orphan_row(), self._insert_orphan_row()]
            parent, top = rows if index % 2 else rows[::-1]  # either is the older row
            done = self._insert_child(live, 20.0)
            roots.append(done)
            for row_id, above_id in ((parent, top), (top, done)):
                db.execute(
                    "UPDATE task_history SET parent_task_id = ? WHERE id = ?", (above_id, row_id))
            self._insert_event(top, 1, th.usage_totals_event(top, 1000, 10.0, 10), 1789130300.0)
            child = f"remote-of-{parent}"
            th._record_remote_subtask(child, parent, "box", "claude-opus-5-5", 0.0, 0, 0)
            for report, ledger in (((0.5, 20, 1), (0.0, 0, 0)), ((0.75, 30, 2), (0.5, 20, 1))):
                charged = th._charge_remote_subtask(child, *report, {})
                assert charged is not None
                th._mark_remote_charge_banked(charged[-1], *ledger)
            arrived: set[str] = set()
            for arrival in order:
                arrived.add(arrival)
                if arrival == "sweep":
                    self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 2)
                elif arrival == "report":
                    charged = th._charge_remote_subtask(child, 3.75, 530, 7, {})
                    assert charged is not None
                    self.assertEqual(charged[5], live if "sweep" in arrived else parent)
                    for row in charged[4]:  # as task_update._publish_rows shows a finished row
                        th._append_chat_event(th.usage_totals_event(*row), task_id=row[0])
                    th._flush_chat_events()
                elif arrival == "step":
                    self._journal((parent,), step)
                    th._replay_failed_events()
                else:
                    closing = th.usage_totals_event(parent, 110, 2.75, 4)
                    self._journal((parent,), closing, at=1789130460.0)
                    th._replay_failed_events()
                if "sweep" not in arrived:
                    continue
                steps, tokens, cost, end_ts = (
                    (4, 110, 2.75, 1789130460000) if "closing" in arrived
                    else (3, 70, 1.75, 1789130400000) if "step" in arrived else (2, 30, 0.75, 0)
                )
                above = (10, 1000, 10.0, 1789130300000)
                if "report" in arrived:
                    steps, tokens, cost = steps + 5, tokens + 500, cost + 3.0
                    above = (15, 1500, 13.0, 1789130300000)
                self.assertEqual(
                    self._progress(parent), (steps, tokens, cost, end_ts), (order, arrival))
                self.assertEqual(self._progress(top), above, (order, arrival))
            th._flush_chat_events()
            last = db.execute(
                "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq DESC LIMIT 1",
                (done,),
            ).fetchone()
            shown = th._structured_usage_totals(json.loads(last[0]))
            self.assertEqual(shown, {"steps": 5, "tokens": 500, "cost": 23.0}, order)
            self.assertEqual(self._progress(done)[:3], (5, 500, 23.0), order)
        owed = db.execute(
            "SELECT parent_task_id FROM remote_charges WHERE settled = 0 ORDER BY id").fetchall()
        # Only what a late add left the running task: one report per order after its sweep.
        self.assertEqual([row[0] for row in owed], [live] * 12)
        self.assertEqual(self._progress(live)[:3], (0, 0, 0.0))
        self.assertEqual(db.execute("SELECT COUNT(*) FROM swept_tasks").fetchone()[0], 48)
        for done in roots:
            th._delete_task(done)  # the marks go with the rows below it
        self.assertEqual(db.execute("SELECT COUNT(*) FROM swept_tasks").fetchone()[0], 0)

    def test_a_replay_leaves_an_undated_row_the_sweep_did_not_finalize(self) -> None:
        """Without an end only a swept row is refreshed: a running task's row and the dead
        rows the next sweep takes (one at the sentinel, one without tags) keep their zeros."""
        db = th._get_db()
        live, _chat = th._add_task("still running")  # owned by this process
        db.execute("UPDATE task_history SET result = 'done', tags = 'work' WHERE id = ?", (live,))
        at_sentinel = self._insert_orphan_row()
        db.execute("UPDATE task_history SET tags = 'work' WHERE id = ?", (at_sentinel,))
        untagged = self._insert_orphan_row()
        db.execute("UPDATE task_history SET result = 'done' WHERE id = ?", (untagged,))
        rows = (live, at_sentinel, untagged)
        self._journal(rows, {"type": "usage_info", "text": "", "total_steps": 3,
                             "total_tokens": 110, "cost_usd": 2.5})

        th._replay_failed_events()

        for task_id in rows:
            row = self._row(task_id)
            self.assertEqual(
                (row["steps"], row["tokens"], row["cost"], row["end_ts"]), (0, 0, 0.0, 0))
        self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 2)
        self.assertEqual(
            [self._row(task_id)["end_ts"] for task_id in rows], [0, 1789130400000, 1789130400000])

    def test_the_reopened_chat_shows_the_floored_cost(self) -> None:
        """Replay appends the row's totals after a last usage event the floor passed."""
        from kiss.agents.sorcar.persistence import _load_chat_events_by_task_id
        from kiss.server.json_printer import with_synthesized_events

        parent = self._insert_orphan_row()
        self._llm_call(parent, 1, "claude-opus-5-5", 3.5)
        self._usage(parent, 2, 4.0)
        self._insert_child(parent, 220.0)
        th._recover_orphaned_tasks(set(), time.time())
        session = _load_chat_events_by_task_id(parent)
        assert session is not None
        stored = session["events"]
        assert isinstance(stored, list)

        events = with_synthesized_events(list(stored), session)

        usage = [ev for ev in events if ev.get("type") == "usage_info"]
        self.assertEqual(len(usage), 2)
        self.assertAlmostEqual(usage[-1]["cost_usd"], 223.5)
        self.assertEqual(usage[-1]["cost"], "$223.5000")
        # A replay whose last usage event shows the row's cost gets nothing more.
        synced = [*stored, usage[-1]]
        replayed = with_synthesized_events(synced, session)
        self.assertEqual(replayed[-1], usage[-1])
        self.assertEqual(sum(ev.get("type") == "usage_info" for ev in replayed), 2)

    def test_the_reopened_chat_shows_the_floored_cost_after_a_text_only_usage(self) -> None:
        """An older usage event with its cost only in its text is compared by that text."""
        from kiss.agents.sorcar.persistence import _load_chat_events_by_task_id
        from kiss.server.json_printer import with_synthesized_events

        parent = self._insert_orphan_row()
        self._llm_call(parent, 1, "claude-opus-5-5", 3.5)
        self._insert_event(
            parent, 2,
            {"type": "usage_info",
             "text": "Steps: 3/100, Total tokens: 1,000, Budget: $4.0000/$500.00, "},
            1789130302.0,
        )
        self._insert_child(parent, 220.0)
        th._recover_orphaned_tasks(set(), time.time())
        session = _load_chat_events_by_task_id(parent)
        assert session is not None
        stored = session["events"]
        assert isinstance(stored, list)

        events = with_synthesized_events(list(stored), session)

        self.assertEqual(events[-1]["type"], "usage_info")
        self.assertAlmostEqual(events[-1]["cost_usd"], 223.5)


if __name__ == "__main__":
    unittest.main()
