# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.scripts.backfill_subtask_costs`` on a real ``sorcar.db``."""

from __future__ import annotations

import contextlib
import io
import shutil
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from kiss.agents.sorcar import persistence as th
from kiss.scripts import backfill_subtask_costs as script
from kiss.tests.agents.sorcar.test_history_date_range import (
    _redirect,
    _restore,
    _set_timestamp,
)


def _task(
    task: str, cost: float, tokens: int, parent: str = "", finished: bool = True,
    model: str = "claude-fable-5-1",
) -> str:
    """Persist a task row through the daemon's own writer and return its id."""
    extra: dict[str, object] = {
        "model": model, "cost": cost, "tokens": tokens, "endTs": 1_790_000_000_000,
    }
    if parent:
        extra["parent_task_id"] = parent
    if not finished:
        del extra["endTs"]
    task_id, _ = th._add_task(task, "", extra)
    return task_id


def _llm_call(task_id: str, cost: float, tokens: int) -> None:
    """Record one ``llm_call`` event of *cost* and *tokens* under *task_id*."""
    th._append_chat_event(
        {
            "type": "llm_call", "model": "claude-fable-5-1", "cost": cost,
            "input_tokens": tokens - 30, "output_tokens": 10, "cache_read": 20,
            "cache_write": 0,
        },
        task_id,
    )


def _row(task_id: str) -> tuple[float, int]:
    """Return the persisted ``(cost, tokens)`` of *task_id*."""
    row = th._get_db().execute(
        "SELECT cost, tokens FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    return (round(float(row["cost"]), 6), int(row["tokens"]))


class TestBackfillSubtaskCosts(unittest.TestCase):
    """The script restores unfolded sub-task spend and nothing else."""

    def setUp(self) -> None:
        self.db_dir = tempfile.mkdtemp()
        self.saved = _redirect(self.db_dir)
        self.db_path = str(th._DB_PATH)

    def tearDown(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.db_dir, ignore_errors=True)

    def _run(self, *args: str) -> tuple[int, list[str]]:
        """Run the script's ``main`` on the test database, capturing stdout."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = script.main(["--db", self.db_path, *args])
        return code, out.getvalue().splitlines()

    def test_repairs_bottom_up_and_is_idempotent(self) -> None:
        # bc92cb57's shape: the parent has its own LLM call, a side
        # channel that was charged, a classifier's spend with no event,
        # and a $56 run_agent sub-task that was never folded in.
        parent = _task("/revise_and_review_paper Writing", 0.519, 55_971)
        _llm_call(parent, 0.2217, 16_648)
        _task("What has the task done so far", 0.2787, 21_945, parent=parent)
        big = _task("Writing: write the paper", 56.4846, 67_276_728, parent=parent)
        # The sub-task's own tree: two reviewer rows, one persisted with
        # cost 0 although its leaves spent $3.
        _llm_call(big, 2.3208, 3_000_000)
        rev = _task("Review the paper", 0.0, 0, parent=big)
        _task("Check section 3", 1.0, 100_000, parent=rev)
        _task("Check section 4", 2.0, 200_000, parent=rev)
        _task("Fix the citations", 51.1637, 63_976_728, parent=big)
        # A healthy tree (children already folded, classifier on top).
        ok = _task("Fix the bug", 10.02, 150_000)
        _llm_call(ok, 4.0, 60_000)
        _task("Review the fix", 6.0, 89_000, parent=ok)
        # A task without sub-tasks is never touched, whatever its events say.
        lone = _task("Say hello", 0.0, 0)
        _llm_call(lone, 0.08, 6_000)

        code, lines = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(lines[-1], "dry run: re-run with --apply to write these totals")
        self.assertEqual(lines[-2], "2 tasks short by $59.4660 in total")
        self.assertTrue(lines[0].startswith(rev[:8]), lines)
        self.assertTrue(lines[1].startswith(parent[:8]), lines)
        self.assertIn("$0.5190 -> $56.9850  (+$56.4660, 2 sub-tasks)", lines[1])
        # A dry run writes nothing.
        self.assertEqual(_row(parent), (0.519, 55_971))
        self.assertEqual(_row(rev), (0.0, 0))

        code, lines = self._run("--apply")
        self.assertEqual(code, 0)
        self.assertEqual(lines[-1], "updated 2 rows (+$59.4660)")
        # rev: 0 -> its leaves.  big: rev's repaired $3 still fits in the
        # $56.4846 big already holds (own $2.3208 + $3 + $51.1637), so
        # big is unchanged.  parent: own + side channel + big.
        self.assertEqual(_row(rev), (3.0, 300_000))
        self.assertEqual(_row(big), (56.4846, 67_276_728))
        self.assertEqual(_row(parent), (56.985, 67_315_321))
        self.assertEqual(_row(ok), (10.02, 150_000))
        self.assertEqual(_row(lone), (0.0, 0))
        # The Spend panel now reports the top-level tasks' full spend.
        total = sum(script._num(r["cost"]) for r in th._spend_by_day_and_model())
        self.assertAlmostEqual(total, 56.985 + 10.02, places=6)

        code, lines = self._run("--apply")
        self.assertEqual(code, 0)
        self.assertEqual(lines, ["0 tasks short by $0.0000 in total", "updated 0 rows (+$0.0000)"])

    def test_skips_unsettled_tasks_small_shortfalls_and_cycles(self) -> None:
        running = _task("Still going", 1.0, 10, finished=False)
        _task("Sub-task of a live task", 5.0, 50, parent=running)
        # An unended row that started two days ago is dead, not running.
        dead = _task("Died without saving", 0.0, 0, finished=False)
        _set_timestamp(dead, time.time() - 2 * 86_400)
        _task("Sub-task of the dead task", 4.0, 40, parent=dead)
        recent = _task("Just finished", 1.0, 10)
        fresh_child = _task("Sub-task charged in a moment", 5.0, 50, parent=recent)
        th._save_task_extra({"endTs": int(time.time() * 1000) - 60_000}, fresh_child)
        tiny = _task("Nearly right", 1.0, 10)
        _task("Sub-task", 1.00005, 10, parent=tiny)
        # Two tasks that list each other as parent: each has its own $1
        # call and a $1 row, so folding would add $1 to both on every run.
        a = _task("Cycle a", 1.0, 10)
        b = _task("Cycle b", 1.0, 10, parent=a)
        th._save_task_extra({"parent_task_id": b}, a)
        _llm_call(a, 1.0, 40)
        _llm_call(b, 1.0, 40)
        # Malformed, non-object and non-numeric events contribute nothing.
        th._append_chat_event({"type": "llm_call", "cost": "n/a"}, tiny)
        th._get_db().executemany(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, 0)",
            [(tiny, 98, '{"type": "llm_call", "cost": 1'), (tiny, 99, '["llm_call", 1]')],
        )
        th._get_db().commit()

        code, lines = self._run("--apply")
        self.assertEqual(code, 0)
        self.assertTrue(lines[0].startswith(dead[:8]), lines)
        self.assertEqual(
            lines[1:],
            [
                "1 task short by $4.0000 in total; 2 skipped as running or ended in the "
                "last 10 minutes; 2 skipped as sitting on a parent_task_id cycle",
                "updated 1 row (+$4.0000)",
            ],
        )
        self.assertEqual(_row(dead), (4.0, 40))
        for task_id in (running, recent, tiny, a, b):
            self.assertEqual(_row(task_id), (1.0, 10))

        code, lines = self._run("--min-shortfall", "0.00001")
        self.assertEqual(code, 0)
        self.assertTrue(lines[0].startswith(tiny[:8]), lines)
        self.assertTrue(lines[1].startswith("1 task short by $0.0001 in total; 2 skipped"), lines)
        # Once the fresh sub-task has settled, its parent is repaired.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        repairs, unsettled, cyclic = script.plan_repairs(
            conn, 0.0001, now_ms=time.time() * 1000 + script.SETTLE_MS,
        )
        conn.close()
        self.assertEqual((unsettled, cyclic), (1, 2))
        # No llm_call events for ``recent``: its own spend is unknown,
        # so it is raised to its sub-task's total only.
        self.assertEqual([(r.task_id, r.new_cost) for r in repairs], [(recent, 5.0)])

        for bad in ("0", "-1", "nan"):
            with self.assertRaises(SystemExit):
                self._run("--min-shortfall", bad)

    def test_rows_changed_after_planning_are_skipped(self) -> None:
        parent = _task("Parent", 0.0, 0)
        _task("Child", 2.0, 20, parent=parent)
        other = _task("Other parent", 0.0, 0)
        _task("Other child", 3.0, 30, parent=other)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        repairs, unsettled, cyclic = script.plan_repairs(conn, 0.0001)
        self.assertEqual((unsettled, cyclic), (0, 0))
        self.assertEqual({r.task_id for r in repairs}, {parent, other})
        # The daemon finishes ``other`` meanwhile: its row is left alone.
        th._save_task_extra({"cost": 7.5, "tokens": 70}, other)
        skipped = script.apply_repairs(conn, repairs)
        conn.close()
        self.assertEqual([r.task_id for r in skipped], [other])
        self.assertEqual(_row(parent), (2.0, 20))
        self.assertEqual(_row(other), (7.5, 70))

    def test_reports_rows_it_could_not_update(self) -> None:
        parent = _task("Parent", 0.0, 0)
        _task("Child", 2.0, 20, parent=parent)
        # Stand in for a concurrent writer: the row refuses this update.
        th._get_db().execute(
            "CREATE TRIGGER refuse BEFORE UPDATE ON task_history "
            f"WHEN NEW.id = '{parent}' BEGIN SELECT RAISE(IGNORE); END"
        )
        th._get_db().commit()
        code, lines = self._run("--apply")
        self.assertEqual(code, 0)
        self.assertEqual(lines[-2], "updated 0 rows (+$0.0000)")
        self.assertEqual(
            lines[-1],
            f"skipped 1 row that changed meanwhile; run again to repair them: {parent[:8]}",
        )
        self.assertEqual(_row(parent), (0.0, 0))

    def test_failed_write_rolls_back_everything(self) -> None:
        first = _task("First parent", 0.0, 0)
        _task("Child", 2.0, 20, parent=first)
        second = _task("Second parent", 0.0, 0)
        _task("Child", 3.0, 30, parent=second)
        th._get_db().execute(
            "CREATE TRIGGER refuse BEFORE UPDATE ON task_history "
            f"WHEN NEW.id = '{second}' BEGIN SELECT RAISE(ABORT, 'disk full'); END"
        )
        th._get_db().commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self._run("--apply")
        self.assertEqual(_row(first), (0.0, 0))
        self.assertEqual(_row(second), (0.0, 0))

    def test_dry_run_with_nothing_to_repair(self) -> None:
        parent = _task("Parent", 5.0, 50)
        _task("Child", 2.0, 20, parent=parent)
        code, lines = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(lines, ["0 tasks short by $0.0000 in total"])

    def test_missing_database_fails(self) -> None:
        code = script.main(["--db", str(Path(self.db_dir) / "missing.db")])
        self.assertEqual(code, 1)
        code = script.main(["--db", str(Path(self.db_dir) / "missing.db"), "--apply"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
