# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The spend of a sub-agent run on another machine, as its local parent accounts for it.

Two real history databases stand for the two daemons: the machine's,
which holds the sub-agent's row (its parent is not there), and this
daemon's, which holds the parent and the ``remote_subtasks`` record the
fold at dispatch return writes.  The machine's report is what its
``getTaskUsage`` answers (``_task_usage``); this daemon charges the
growth (``charge_remote_subtask_usage``):

* the Spend panel files the sub-agent's dollars under the models they
  ran on, not the parent's (audit 2026-11-03, A3-2);
* a late charge to the sub-agent's row on the machine reaches the
  parent's row, its persisted ``usage_info`` and the Spend panel, once
  per report however often it is seen (A3-1); a running parent banks
  it on its live agent;
* the machine's report reads its two tables in one transaction and
  takes no history lock: a stop anywhere in it ends the transaction;
  the Spend panel reads them the same way.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from functools import partial
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import _agent_usage
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.task_update import charge_remote_subtask_usage
from kiss.tests.agents.sorcar.test_history_date_range import _redirect, _restore

OPUS = "claude-opus-5-5"
ASTRA = "gpt-6-astra"
FABLE = "claude-fable-5-1"


def _row(task_id: str) -> tuple[float, int, int]:
    """The persisted ``(cost, tokens, steps)`` of *task_id*."""
    row = th._get_db().execute(
        "SELECT cost, tokens, steps FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    return float(row[0]), int(row[1]), int(row[2])


def _task(model: str, cost: float, tokens: int, steps: int, parent: str = "",
          finished: bool = True) -> str:
    """Add a task row with these totals; return its id."""
    extra: dict[str, Any] = {"model": model, "cost": cost, "tokens": tokens, "steps": steps}
    if finished:
        extra["endTs"] = int(time.time() * 1000)
    if parent:
        extra["parent_task_id"] = parent
    task_id, _ = th._add_task(f"task on {model}", "", extra)
    return task_id


def _costs_by_model() -> dict[str, float]:
    """The Spend panel's cost per model over the whole history."""
    costs: dict[str, float] = {}
    for entry in th._spend_by_day_and_model():
        model = str(entry["model"])
        costs[model] = costs.get(model, 0.0) + float(entry["cost"])  # type: ignore[arg-type]
    return {model: round(cost, 9) for model, cost in costs.items()}


def _charge_as_the_records_are_read(
    db: sqlite3.Connection, peer: sqlite3.Connection, parent: str, child: str, statement: str,
) -> None:
    """Trace callback of *db*: as it reads the records, *peer* commits $0.25 more for *child*."""
    if "FROM remote_subtasks" in statement:
        db.set_trace_callback(None)
        peer.execute("BEGIN IMMEDIATE")
        peer.execute("UPDATE task_history SET cost = cost + 0.25 WHERE id = ?", (parent,))
        peer.execute("UPDATE remote_subtasks SET cost = cost + 0.25 WHERE task_id = ?", (child,))
        peer.execute("COMMIT")


class _StopAt:
    """Trace function that raises ``KeyboardInterrupt`` before the *at*-th bytecode it sees."""

    def __init__(self, at: int) -> None:
        self.at = at
        self.seen = 0
        self.where = ""

    def __call__(self, frame: FrameType, event: str, arg: object) -> _StopAt:
        frame.f_trace_opcodes = True
        if event == "opcode":
            if self.seen == self.at:
                sys.settrace(None)
                self.where = f"{frame.f_code.co_qualname}+{frame.f_lasti}"
                raise KeyboardInterrupt
            self.seen += 1
        return self


class TestRemoteSubagentCosts(unittest.TestCase):
    """P ($1 on claude-opus-5-5) dispatched C ($2) to the machine ``box``."""

    def setUp(self) -> None:
        self.local_dir = tempfile.mkdtemp()
        self.remote_dir = tempfile.mkdtemp()
        self.saved = _redirect(self.local_dir)

    def tearDown(self) -> None:
        self._close()
        _restore(self.saved)
        shutil.rmtree(self.local_dir, ignore_errors=True)
        shutil.rmtree(self.remote_dir, ignore_errors=True)

    def _close(self) -> None:
        th._close_thread_db()

    def _on(self, tmpdir: str) -> None:
        """Make *tmpdir*'s database the one the persistence functions use."""
        self._close()
        _redirect(tmpdir)

    def _dispatch(self, parent_finished: bool = True) -> tuple[str, str]:
        """P folds C's $2 (C: $1.5 own on astra, a $0.5 grandchild on fable)."""
        parent = _task(OPUS, 3.0, 300, 5, finished=parent_finished)
        self._on(self.remote_dir)
        child = _task(ASTRA, 2.0, 200, 3, parent=parent)
        _task(FABLE, 0.5, 50, 1, parent=child)
        self._on(self.local_dir)
        th._record_remote_subtask(child, parent, "box", ASTRA, 2.0, 200, 3)
        return parent, child

    def _report(self, child: str) -> dict[str, Any]:
        """What the machine's ``getTaskUsage`` answers for *child* now."""
        self._on(self.remote_dir)
        report = th._task_usage([child, "no-such-task"])
        self._on(self.local_dir)
        assert list(report) == [child]
        return report[child]

    def _late_charge_on_the_machine(self, child: str, cost: float) -> None:
        """An ``/ask`` answer on the machine ends after C: its row gets *cost* more."""
        self._on(self.remote_dir)
        updated, running = th._add_late_task_usage(child, 25, cost, 1)
        # The walk stops at P, which has no row there.
        assert ([u[0] for u in updated], running) == ([child], "")
        self._on(self.local_dir)

    def test_spend_by_model_files_the_remote_child_under_its_models(self) -> None:
        parent, child = self._dispatch()
        # Folded at dispatch return: the child's model, not the parent's.
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 2.0})
        # The machine's split of C's tree replaces it on the next sync,
        # which charges nothing: the report equals the folded amount.
        report = self._report(child)
        self.assertEqual((report["cost"], report["tokens"], report["steps"]), (2.0, 200, 3))
        self.assertEqual(report["models"], {ASTRA: [1.5, 150.0], FABLE: [0.5, 50.0]})
        charge_remote_subtask_usage(None, child, report)
        self.assertEqual(_row(parent), (3.0, 300, 5))
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.5, FABLE: 0.5})
        # Not mirrored: the local history has no row for the child.
        self.assertIsNone(th._load_chat_events_by_task_id(child))

    def test_the_spend_panel_reads_its_two_tables_as_one_snapshot(self) -> None:
        """A charge another process commits between the two reads shows whole or not at all.

        Read apart, the parent's row predated the charge and the record
        held it: the panel took the $0.25 out of the parent's own model.
        """
        parent, child = self._dispatch()
        db = th._get_db()
        peer = sqlite3.connect(th._DB_PATH, isolation_level=None)
        self.addCleanup(peer.close)
        db.set_trace_callback(partial(_charge_as_the_records_are_read, db, peer, parent, child))
        try:
            during = _costs_by_model()
        finally:
            db.set_trace_callback(None)
        self.assertEqual(during, {OPUS: 1.0, ASTRA: 2.0})
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 2.25})

    def test_a_late_charge_on_the_machine_reaches_the_finished_parent_once(self) -> None:
        parent, child = self._dispatch()
        self._late_charge_on_the_machine(child, 0.25)
        report = self._report(child)
        self.assertEqual(report["cost"], 2.25)
        charge_remote_subtask_usage(None, child, report)
        self.assertEqual(_row(parent), (3.25, 325, 6))
        # The same report again (a replayed or repeated sync) charges nothing.
        charge_remote_subtask_usage(None, child, report)
        self.assertEqual(_row(parent), (3.25, 325, 6))
        # The parent's transcript ends with the new cost, as the chat shows it.
        session = th._load_chat_events_by_task_id(parent)
        assert session is not None
        usage = [e for e in session["events"] if e.get("type") == "usage_info"]  # type: ignore[attr-defined]
        self.assertEqual(usage[-1]["cost_usd"], 3.25)
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.75, FABLE: 0.5})
        # A report below the record (an older answer) neither charges nor lowers it.
        charge_remote_subtask_usage(None, child, {"cost": 2.0, "tokens": 200, "steps": 3})
        self.assertEqual(_row(parent), (3.25, 325, 6))
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.75, FABLE: 0.5})

    def test_a_running_parent_banks_the_late_charge_on_its_live_agent(self) -> None:
        parent, child = self._dispatch(parent_finished=False)
        agent = ChatSorcarAgent("remote-late-parent")
        state = AgentState(parent, agent=agent, tab_id="tab-remote-late", server_owned=True)
        agent_state.register(state)
        try:
            self._late_charge_on_the_machine(child, 0.25)
            report = self._report(child)
            charge_remote_subtask_usage(None, child, report)
            charge_remote_subtask_usage(None, child, report)
        finally:
            agent_state.unregister(parent, state)
        # Its final save writes the banked spend; the row is untouched until then.
        self.assertEqual(_agent_usage(agent), (pytest.approx(0.25), 25, 1))
        self.assertEqual(_row(parent), (3.0, 300, 5))
        # The charge the row owes is recorded and marked banked with the ledger
        # totals it was banked over (the agent's $0 before it); the final save's
        # totals from that ledger hold it, so the save adds nothing to them,
        # and the charge stays with the row, settled into it.
        charges = th._get_db().execute(
            "SELECT parent_task_id, cost, banked, banked_cost, settled FROM remote_charges"
        ).fetchall()
        self.assertEqual([tuple(c) for c in charges], [(parent, 0.25, 1, 0.0, 0)])
        th._save_task_extra(
            {"cost": 3.25, "tokens": 325, "steps": 6, "endTs": time.time()}, parent
        )
        self.assertEqual(_row(parent), (3.25, 325, 6))
        charges = th._get_db().execute(
            "SELECT parent_task_id, cost, banked, banked_cost, settled FROM remote_charges"
        ).fetchall()
        self.assertEqual([tuple(c) for c in charges], [(parent, 0.25, 1, 0.0, 1)])

    def test_a_task_never_recorded_is_not_charged(self) -> None:
        parent = _task(OPUS, 1.0, 100, 1)
        charge_remote_subtask_usage(None, "not-a-remote-child", {"cost": 5.0, "tokens": 9})
        self.assertEqual(_row(parent), (1.0, 100, 1))
        self.assertEqual(th._remote_subtask_ids("box", 0.0), [])

    def test_a_stop_anywhere_in_the_report_leaves_no_transaction_open(self) -> None:
        """One stop at each bytecode boundary of the report, in turn.

        The server stops a thread at any boundary
        (``PyThreadState_SetAsyncExc``).  While the stop is being
        handled, and the frames its traceback holds are alive, the
        thread's connection is in no transaction; the next report is
        whole.
        """
        _, child = self._dispatch()
        self._on(self.remote_dir)
        th._record_remote_subtask("far", child, "box2", OPUS, 0.25, 25, 1)
        expected = th._task_usage([child])
        db = th._get_db()
        tracer = sys.gettrace()
        at = 0
        while True:
            stop = _StopAt(at)
            sys.settrace(stop)
            try:
                th._task_usage([child])
            except KeyboardInterrupt:
                self.assertFalse(db.in_transaction, f"stop {at} at {stop.where}")
            else:
                break
            finally:
                sys.settrace(tracer)
            self.assertEqual(th._task_usage([child]), expected, f"stop {at} at {stop.where}")
            at += 1
        self.assertGreater(at, 100)  # the sweep did visit the report's bytecodes
