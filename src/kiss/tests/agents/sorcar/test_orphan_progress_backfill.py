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
        """A killed sub-agent's record journalled before the kill lands after the sweep."""
        parent = self._insert_orphan_row()
        self._llm_call(parent, 1, "claude-opus-5-5", 1.0)
        child = self._insert_child(parent, 0.0, dead=True)
        self._llm_call(child, 1, "claude-opus-5-5", 0.5)
        th._recover_orphaned_tasks(set(), time.time())
        self.assertAlmostEqual(self._row(child)["cost"], 0.5)
        self.assertAlmostEqual(self._row(parent)["cost"], 1.5)
        th._journal_failed_events(
            [(child,
              json.dumps({"type": "llm_call", "model": "claude-opus-5-5", "cost": 2.0,
                          "input_tokens": 100, "output_tokens": 10}),
              1789130400.0, str(th._DB_PATH))],
            1,
        )

        th._replay_failed_events()

        self.assertAlmostEqual(self._row(child)["cost"], 2.5)
        self.assertAlmostEqual(self._row(parent)["cost"], 3.5)
        self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 0)
        self.assertAlmostEqual(self._row(parent)["cost"], 3.5)  # nothing counted twice

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
