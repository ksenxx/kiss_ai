"""Cost audit 2026-11-04: the gaps found in the remote sub-agent accounting.

* A late remote charge whose parent update fails is not acknowledged:
  the record's advance and the charge are one transaction, so the
  record stays where it was (a daemon killed in between included) and
  the next sync charges it.
* A sync asks for every recorded sub-agent, in batches of 500, instead
  of the newest 500 only.
* A remote child mirrored into the local history counts once, by its
  record once the record grew past the copy.
* The sync starts on every page-one history refresh (the "This machine"
  filter included) and on a Spend refresh.
* A late remote charge owed to a RUNNING parent is recorded in the
  acknowledgement transaction (``remote_charges``), marked banked with
  the parent's ledger totals BEFORE it is banked on the live agent, and
  kept as long as the parent's row exists.  Its ``settled`` flag says
  whether the row's totals hold it: every writer of absolute totals (the
  final saves, the orphan recovery from a persisted usage event) resets
  the flags to what its totals hold and the finalization adds the other
  charges, so a daemon that dies anywhere around the bank, a second final
  save, and a charge recorded by a process without the parent's agent all
  count the charge once.
* A charge another process commits while the parent's final save runs
  (after its read of the pending charges) is banked by that save too:
  the ledger its caller folds holds what the row holds.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar import sorcar_agent
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.machines import LOCAL_MACHINE
from kiss.agents.sorcar.relentless_agent import _OFFSET_PUBLISH_LOCK
from kiss.server import agent_state, task_update
from kiss.server import sorcar as server_api
from kiss.server.agent_state import AgentState
from kiss.server.task_update import (
    charge_remote_subtask_usage,
    charge_side_channel_usage,
    sync_remote_subtask_usage,
)
from kiss.tests.agents.sorcar import test_audit1103_remote_costs as base
from kiss.tests.agents.sorcar.test_audit1103_remote_costs import (
    ASTRA,
    FABLE,
    OPUS,
    _costs_by_model,
    _row,
    _task,
)
from kiss.tests.agents.sorcar.test_audit1109_remote_final_bank import (
    _save_on_thread,
    _wait_parked_in_bank,
)
from kiss.tests.agents.sorcar.test_history_date_range import _redirect, _restore
from kiss.tests.server.test_history_date_range import _make_server


def _charges(parent: str) -> list[tuple[float, int, int]]:
    """The ``remote_charges`` deltas *parent* owes, oldest first."""
    rows = th._get_db().execute(
        "SELECT cost, tokens, steps FROM remote_charges WHERE parent_task_id = ? ORDER BY id",
        (parent,),
    ).fetchall()
    return [(float(r[0]), int(r[1]), int(r[2])) for r in rows]


def _settled(parent: str) -> list[int]:
    """The ``settled`` flags of *parent*'s charges, oldest first: 1 once the row holds one."""
    rows = th._get_db().execute(
        "SELECT settled FROM remote_charges WHERE parent_task_id = ? ORDER BY id", (parent,),
    ).fetchall()
    return [int(r[0]) for r in rows]


class TestRemoteFixes(base.TestRemoteSubagentCosts):
    """Reuses the P ($3, opus) -> C ($2 on box: astra $1.5 + fable $0.5) set-up."""

    def test_a_failed_parent_charge_is_charged_by_the_next_sync(self) -> None:
        parent, child = self._dispatch()
        self._late_charge_on_the_machine(child, 0.25)
        report = self._report(child)
        with mock.patch.object(th, "_add_usage_locked", side_effect=RuntimeError("db")):
            charge_remote_subtask_usage(None, child, report)  # logged, not raised
        self.assertEqual(_row(parent), (3.0, 300, 5))
        # the uncharged report's split is rolled back with its totals
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 2.0})
        charge_remote_subtask_usage(None, child, report)
        self.assertEqual(_row(parent), (3.25, 325, 6))
        charge_remote_subtask_usage(None, child, report)
        self.assertEqual(_row(parent), (3.25, 325, 6))

    def test_a_daemon_killed_between_the_advance_and_the_charge_loses_no_spend(self) -> None:
        """The record's advance and the parent's charge are one transaction: a daemon that
        dies between them (here at the parent's row update) leaves the record where it
        was, so the next sync charges the report instead of owing nothing."""
        parent, child = self._dispatch()
        self._late_charge_on_the_machine(child, 0.25)
        report = self._report(child)
        th._get_db()  # keeps the WAL sidecars alive across the other process's death
        script = textwrap.dedent(f"""
            import os
            import sqlite3
            from kiss.agents.sorcar import persistence as th
            from kiss.server.task_update import charge_remote_subtask_usage

            def die_at_the_parent_update(action, table, *_):
                if action == sqlite3.SQLITE_UPDATE and table == "task_history":
                    os._exit(0)  # the daemon dies with the parent's charge unwritten
                return sqlite3.SQLITE_OK

            th._get_db().set_authorizer(die_at_the_parent_update)
            charge_remote_subtask_usage(None, {child!r}, {report!r})
            raise SystemExit("the parent's row was never updated")
            """)
        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
            env={**os.environ, "KISS_HOME": str(th._KISS_DIR),
                 "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(_row(parent), (3.0, 300, 5))  # died before the charge...
        charge_remote_subtask_usage(None, child, report)  # ...so the next sync owes it
        self.assertEqual(_row(parent), (3.25, 325, 6))
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.75, FABLE: 0.5})

    def _die_after(self, child: str, report: dict[str, Any], function: str) -> str:
        """In another process: a running parent (its row owned there) is charged *report*
        for *child*, and the process dies as *function* returns.  Returns the parent's id."""
        script = textwrap.dedent(f"""
            import os
            import sys
            from kiss.agents.sorcar import persistence as th
            from kiss.agents.sorcar import sorcar_agent
            from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
            from kiss.server import agent_state
            from kiss.server.agent_state import AgentState
            from kiss.server.task_update import charge_remote_subtask_usage

            parent, _chat = th._add_task(
                "task on {OPUS}", "", {{"model": {OPUS!r}, "cost": 3.0, "tokens": 300, "steps": 5}},
            )
            th._record_remote_subtask({child!r}, parent, "box", {ASTRA!r}, 2.0, 200, 3)
            agent = ChatSorcarAgent("remote-late-parent")
            sorcar_agent._attribute_sub_usage(agent, 3.0, 300, 5)  # its own spend, as the row
            agent_state.register(AgentState(parent, agent=agent, tab_id="tab", server_owned=True))
            print(parent, flush=True)
            code = ({function!r}).__code__

            def die(frame, event, arg):
                if event == "return" and frame.f_code is code:
                    os._exit(77)

            sys.setprofile(die)
            charge_remote_subtask_usage(None, {child!r}, {report!r})
            raise SystemExit("the charge did not reach " + {function!r})
            """).replace(f"({function!r}).__code__", f"{function}.__code__")
        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
            env={**os.environ, "KISS_HOME": str(th._KISS_DIR),
                 "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        self.assertEqual(done.returncode, 77, done.stderr)
        return done.stdout.split()[-1]

    def test_a_daemon_dying_around_a_running_parents_bank_loses_no_spend(self) -> None:
        """The delta owed to a running parent is durable from the acknowledgement on: the
        recovery of the parent's row (its process died before the durable mark, after it
        but before the live bank, or after the bank with no usage event persisted since)
        folds it in, once."""
        self._on(self.remote_dir)
        report = self._report(_task(ASTRA, 2.25, 225, 4))  # C's row, a late $0.25 included
        self._on(self.local_dir)
        th._get_db()  # keeps the WAL sidecars alive across the other process's death
        for n, function in enumerate((
            "th._charge_remote_subtask",  # acknowledged, never marked nor banked
            "th._mark_remote_charge_banked",  # marked banked, never banked
            "sorcar_agent._attribute_sub_usage",  # banked, no usage event persisted after
        )):
            child = f"remote-child-{n}"  # a record of its own: the dead parents keep theirs
            with self.subTest(dies_after=function):
                parent = self._die_after(child, report, function)
                record = th._get_db().execute(
                    "SELECT cost, tokens, steps FROM remote_subtasks WHERE task_id = ?", (child,),
                ).fetchone()
                self.assertEqual(tuple(record), (2.25, 225, 4))  # the growth is acknowledged
                self.assertEqual(_row(parent), (3.0, 300, 5))  # ...and its parent not charged
                self.assertEqual(th._recover_orphaned_tasks(set()), 1)
                self.assertEqual(_row(parent), (3.25, 325, 6))
                charge_remote_subtask_usage(None, child, report)  # the next sync owes nothing
                self.assertEqual(_row(parent), (3.25, 325, 6))
                self.assertEqual(th._recover_orphaned_tasks(set()), 0)  # finalized once
                self.assertEqual(_row(parent), (3.25, 325, 6))
                self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 2.25})
                # The charge stays with the row, settled into it, and goes with it.
                self.assertEqual(_charges(parent), [(0.25, 25, 1)])
                self.assertEqual(_settled(parent), [1])
                self.assertEqual(th._delete_task(parent), 1)
                self.assertEqual(_charges(parent), [])

    def _die_after_a_usage_event(
        self, child: str, report: dict[str, Any], function: str, legacy: bool = False,
        step: bool = True,
    ) -> str:
        """In another process: the running parent's real printer and agent take the charge
        for *child*; the charging thread stops as *function* returns, the parent's own
        thread then prints a per-step ``usage_info`` (persisted, with the printer's task
        offsets) and the process dies.  With *legacy* the event is persisted as a printer
        from before the ``banked`` stamp wrote it: the same event without the stamp, its
        only field added since.  Returns the parent's id."""
        script = textwrap.dedent(f"""
            import json
            import os
            import sys
            import threading
            from kiss.agents.sorcar import persistence as th
            from kiss.agents.sorcar import sorcar_agent
            from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
            from kiss.server import agent_state
            from kiss.server.agent_state import AgentState
            from kiss.server.json_printer import JsonPrinter
            from kiss.server.task_update import charge_remote_subtask_usage

            parent, _chat = th._add_task("task on {OPUS}")  # a running row: zero totals
            th._record_remote_subtask({child!r}, parent, "box", {ASTRA!r}, 2.0, 200, 3)
            agent = ChatSorcarAgent("remote-late-parent")
            with agent._task_id_lock:
                agent._last_task_id = parent
            printer = JsonPrinter()
            printer.adopt_thread_binding(parent)
            agent.printer = printer
            agent_state.register(AgentState(parent, agent=agent, tab_id="tab", server_owned=True))
            sorcar_agent._attribute_sub_usage(agent, 3.0, 300, 5)  # its own spend, published
            print(parent, flush=True)
            code = {function}.__code__
            stopped = threading.Event()

            def stop(frame, event, arg):
                if event == "return" and frame.f_code is code:
                    stopped.set()
                    threading.Event().wait(60)

            def charge():
                sys.setprofile(stop)
                # Without a step after it, the bank's own publication is the last event.
                charge_remote_subtask_usage(
                    None if {step!r} else printer, {child!r}, {report!r},
                )

            threading.Thread(target=charge, daemon=True).start()
            assert stopped.wait(30), "the charge did not reach " + {function!r}
            # The parent's executor prints its next step: session-relative text and
            # counters, which the printer offsets to the task's totals.
            if {step!r}: printer.print(
                "Steps: 2/100, Context: 1,000/200,000 tokens, Total tokens: 40, "
                "Budget: $3.5100/$10.00, ",
                type="usage_info", total_tokens=40, cost="$0.0100", cost_usd=0.01,
                total_steps=2, model={OPUS!r}, cache_read=0,
            )
            th._flush_chat_events(parent)
            if {legacy!r}:
                db = th._get_db()
                row = db.execute(
                    "SELECT id, event_json FROM events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                    (parent,),
                ).fetchone()
                event = json.loads(row["event_json"])
                del event["banked"]
                with th._rw_lock.write_lock(), th._immediate_txn(db):
                    db.execute(
                        "UPDATE events SET event_json = ? WHERE id = ?",
                        (json.dumps(event), row["id"]),
                    )
            os._exit(77)
            """)
        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
            env={**os.environ, "KISS_HOME": str(th._KISS_DIR),
                 "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        self.assertEqual(done.returncode, 77, done.stderr)
        return done.stdout.split()[-1]

    def test_a_usage_event_persisted_around_the_bank_decides_the_recovered_charge(self) -> None:
        """The parent's last persisted usage event may or may not hold the banked delta: its
        offsets (totals minus the session's own figures) against the ledger totals the
        charge was marked with tell, so recovery adds the delta exactly when the event
        does not hold it."""
        self._on(self.remote_dir)
        report = self._report(_task(ASTRA, 2.5, 250, 4))  # C's row, a late $0.5 included
        self._on(self.local_dir)
        th._get_db()
        for n, (function, event_totals) in enumerate((
            # marked, not banked: the event carries the ledger before the charge
            ("th._mark_remote_charge_banked", (3.01, 340, 7)),
            # banked, not yet followed by... anything: the event carries the charge
            ("sorcar_agent._attribute_sub_usage", (3.51, 390, 8)),
        )):
            child = f"remote-event-child-{n}"
            with self.subTest(dies_after=function):
                parent = self._die_after_a_usage_event(child, report, function)
                last = th._get_db().execute(
                    "SELECT event_json FROM events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                    (parent,),
                ).fetchone()
                saved = json.loads(last[0])
                self.assertEqual(
                    (saved["cost_usd"], saved["total_tokens"], saved["total_steps"]),
                    event_totals,
                )
                charges = th._get_db().execute(
                    "SELECT banked, banked_cost, banked_tokens, banked_steps FROM remote_charges "
                    "WHERE parent_task_id = ?", (parent,),
                ).fetchall()
                self.assertEqual([tuple(c) for c in charges], [(1, 3.0, 300, 5)])
                self.assertEqual(_row(parent), (0.0, 0, 0))
                self.assertEqual(th._recover_orphaned_tasks(set()), 1)
                self.assertEqual(_row(parent), (3.51, 390, 8))  # the event's spend, delta once
                charge_remote_subtask_usage(None, child, report)  # the same report: nothing
                self.assertEqual(th._recover_orphaned_tasks(set()), 0)
                self.assertEqual(_row(parent), (3.51, 390, 8))
                self.assertEqual(_charges(parent), [(0.5, 50, 1)])
                self.assertEqual(_settled(parent), [1])

    def test_spend_banked_after_the_last_step_is_recovered_once(self) -> None:
        """The process dies after the charge's whole publication and before the parent's
        next step: its last usage event is the bank's own (no text, the task's totals, the
        ledger stamp), and recovery adopts it, the delta once, not the step before it."""
        self._on(self.remote_dir)
        report = self._report(_task(ASTRA, 2.5, 250, 4))  # C's row, a late $0.5 included
        self._on(self.local_dir)
        th._get_db()
        parent = self._die_after_a_usage_event(
            "remote-bank-last-child", report,
            "sys.modules['kiss.server.task_update']._publish_charge", step=False,
        )
        last = th._get_db().execute(
            "SELECT event_json FROM events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (parent,),
        ).fetchone()
        saved = json.loads(last[0])
        self.assertEqual(saved["text"], "")
        self.assertEqual(saved["banked"], [3.5, 350, 6])
        self.assertEqual((saved["cost_usd"], saved["total_tokens"], saved["total_steps"]),
                         (3.5, 350, 6))
        self.assertEqual(_row(parent), (0.0, 0, 0))
        self.assertEqual(th._recover_orphaned_tasks(set()), 1)
        self.assertEqual(_row(parent), (3.5, 350, 6))
        charge_remote_subtask_usage(None, "remote-bank-last-child", report)  # nothing new
        self.assertEqual(th._recover_orphaned_tasks(set()), 0)
        self.assertEqual(_row(parent), (3.5, 350, 6))
        self.assertEqual(_charges(parent), [(0.5, 50, 1)])
        self.assertEqual(_settled(parent), [1])

    def test_a_charge_of_cost_alone_is_recovered_once_around_the_bank(self) -> None:
        """A report grown in cost alone (an ``/ask`` answer billed without tokens or steps)
        moves no token or step offset: the exact offsets every per-step usage event
        carries (the printer's ``banked`` stamp) tell whether the event holds the charge,
        so the recovery of the parent's row adds it exactly when its last event does not
        (its process died before the mark, between the mark and the bank, or after the
        bank)."""
        self._on(self.remote_dir)
        remote_row = _task(ASTRA, 2.0, 200, 3)
        charge_side_channel_usage(None, None, remote_row, 0.5, 0, 0)  # C's row: $0.5 alone
        self.assertEqual(_row(remote_row), (2.5, 200, 3))
        report = self._report(remote_row)
        self._on(self.local_dir)
        th._get_db()
        for n, function in enumerate((
            "th._charge_remote_subtask",
            "th._mark_remote_charge_banked",
            "sorcar_agent._attribute_sub_usage",
        )):
            child = f"remote-cost-child-{n}"
            with self.subTest(dies_after=function):
                parent = self._die_after_a_usage_event(child, report, function)
                last = th._get_db().execute(
                    "SELECT event_json FROM events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                    (parent,),
                ).fetchone()
                saved = json.loads(last[0])
                banked = function == "sorcar_agent._attribute_sub_usage"
                self.assertEqual(saved["banked"], [3.5 if banked else 3.0, 300, 5])
                self.assertEqual(saved["cost_usd"], 3.51 if banked else 3.01)
                self.assertEqual(_row(parent), (0.0, 0, 0))
                self.assertEqual(th._recover_orphaned_tasks(set()), 1)
                self.assertEqual(_row(parent), (3.51, 340, 7))
                charge_remote_subtask_usage(None, child, report)  # the same report: nothing
                self.assertEqual(th._recover_orphaned_tasks(set()), 0)
                self.assertEqual(_row(parent), (3.51, 340, 7))
                self.assertEqual(_charges(parent), [(0.5, 0, 0)])
                self.assertEqual(_settled(parent), [1])

    def test_a_usage_event_without_the_stamp_decides_the_charge_by_its_totals(self) -> None:
        """An event persisted before the stamp rests at most on its totals (its session's
        own cost is in its total, not told apart in its text): a marked charge of cost
        alone is held only when the event's total reaches the marked totals plus the
        charge, so the recovery adds it when its process died before the mark or between
        the mark and the bank (the event's total is below), and not after the bank, when
        it logs the charges the event holds.  Left as the limitation of such events: a
        session whose own spend already reached the charge makes an event printed
        between the mark and the bank hold it too (lost once, bounded by the charge)."""
        self._on(self.remote_dir)
        remote_row = _task(ASTRA, 2.0, 200, 3)
        charge_side_channel_usage(None, None, remote_row, 0.5, 0, 0)  # C's row: $0.5 alone
        report = self._report(remote_row)
        self._on(self.local_dir)
        th._get_db()
        for n, function in enumerate((
            "th._charge_remote_subtask",
            "th._mark_remote_charge_banked",
            "sorcar_agent._attribute_sub_usage",
        )):
            child = f"remote-legacy-child-{n}"
            with self.subTest(dies_after=function):
                parent = self._die_after_a_usage_event(child, report, function, legacy=True)
                last = th._get_db().execute(
                    "SELECT event_json FROM events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                    (parent,),
                ).fetchone()
                saved = json.loads(last[0])
                self.assertNotIn("banked", saved)
                banked = function == "sorcar_agent._attribute_sub_usage"
                self.assertEqual(saved["cost_usd"], 3.51 if banked else 3.01)
                with self.assertLogs("kiss.agents.sorcar.persistence", "WARNING") as logs:
                    self.assertEqual(th._recover_orphaned_tasks(set()), 1)
                self.assertEqual(
                    any("from before the stamp" in line for line in logs.output), banked,
                )
                self.assertEqual(_row(parent), (3.51, 340, 7))
                charge_remote_subtask_usage(None, child, report)  # the same report: nothing
                self.assertEqual(th._recover_orphaned_tasks(set()), 0)
                self.assertEqual(_row(parent), (3.51, 340, 7))
                self.assertEqual(_charges(parent), [(0.5, 0, 0)])
                self.assertEqual(_settled(parent), [1])

    def _charge_from_another_process(self, child: str, report: dict[str, Any]) -> None:
        """A process without the parent's live agent charges *report* for *child*."""
        done = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(f"""
                from kiss.server.task_update import charge_remote_subtask_usage
                charge_remote_subtask_usage(None, {child!r}, {report!r})
                """)],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "KISS_HOME": str(th._KISS_DIR),
                 "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_every_final_save_after_another_processs_charge_keeps_it(self) -> None:
        """A charge recorded by a process without the parent's agent is never on its ledger:
        the chat agent's final save and the server's final save of the same row (two
        absolute snapshots of that ledger) each add it, and a refresh keeps it.  A report
        arriving between the two saves grows the finished row directly, and that growth
        is retained with the row too: the server's save (and a repeat of it) replays it
        over the unchanged ledger snapshot instead of erasing it."""
        from kiss.server.task_runner import _save_subtask_extra

        parent, _chat = th._add_task(f"task on {OPUS}")
        agent = ChatSorcarAgent("remote-late-parent")
        with agent._task_id_lock:
            agent._last_task_id = parent
        state = AgentState(parent, agent=agent, tab_id="tab", server_owned=True)
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, parent, state)
        sorcar_agent._attribute_sub_usage(agent, 3.0, 300, 5)
        self._on(self.remote_dir)
        remote_row = _task(ASTRA, 2.5, 250, 4)
        report = self._report(remote_row)
        self._on(self.local_dir)
        th._record_remote_subtask("remote-final-child", parent, "box", ASTRA, 2.0, 200, 3)
        self._charge_from_another_process("remote-final-child", report)
        self.assertEqual(_charges(parent), [(0.5, 50, 1)])
        self.assertEqual(_row(parent), (0.0, 0, 0))

        th._save_task_result("Done", parent)
        with th.TASK_USAGE_LOCK:
            agent._save_final_usage({}, parent, result_summary="Done")
        self.assertEqual(_row(parent), (3.5, 350, 6))
        # C's row grows again on the machine, and the report lands between the saves:
        # the parent's row is finished, so the growth goes straight into it.
        self._late_charge_on_the_machine(remote_row, 0.25)
        later = self._report(remote_row)
        self._charge_from_another_process("remote-final-child", later)
        self.assertEqual(_row(parent), (3.75, 375, 7))
        self.assertEqual(_charges(parent), [(0.5, 50, 1), (0.25, 25, 1)])
        self.assertEqual(_settled(parent), [1, 1])
        with th.TASK_USAGE_LOCK:
            _save_subtask_extra(state, parent, OPUS, self.local_dir, 1_000, 2_000)
        self.assertEqual(_row(parent), (3.75, 375, 7))
        th._refresh_finished_rows({parent})
        self.assertEqual(_row(parent), (3.75, 375, 7))
        for again in (report, later):  # every report seen again: nothing more
            charge_remote_subtask_usage(None, "remote-final-child", again)
            self.assertEqual(_row(parent), (3.75, 375, 7))
        with th.TASK_USAGE_LOCK:  # a repeated server finalization of the same row
            _save_subtask_extra(state, parent, OPUS, self.local_dir, 1_000, 2_000)
        self.assertEqual(_row(parent), (3.75, 375, 7))
        self.assertEqual(_charges(parent), [(0.5, 50, 1), (0.25, 25, 1)])
        self.assertEqual(_settled(parent), [1, 1])
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 2.75})

    def test_a_charge_another_process_commits_during_the_final_save_reaches_the_ledger(
        self,
    ) -> None:
        """Another process charges the row after the final save read its pending charges
        and before its row transaction: the save banks that charge too, so the ledger the
        run's caller folds, the row and an ancestor's row agree; a retried save adds nothing."""
        parent, _chat = th._add_task(f"task on {OPUS}")
        agent = ChatSorcarAgent("remote-raced-parent")
        sorcar_agent._attribute_sub_usage(agent, 3.0, 300, 5)
        self._on(self.remote_dir)
        report = self._report(_task(ASTRA, 2.5, 250, 4))
        self._on(self.local_dir)
        for child in ("raced-child-a", "raced-child-b"):
            th._record_remote_subtask(child, parent, "box", ASTRA, 2.0, 200, 3)
        charge_remote_subtask_usage(None, "raced-child-a", report)
        th._save_task_result("Done", parent)
        outcome: list[str] = []
        saver = threading.Thread(
            target=_save_on_thread, args=(agent, parent, outcome), daemon=True)
        with _OFFSET_PUBLISH_LOCK:  # parks the save right after its read of the pending charges
            saver.start()
            _wait_parked_in_bank(saver)
            self._charge_from_another_process("raced-child-b", report)
        saver.join(60)
        self.assertEqual(outcome, ["saved"])
        self.assertEqual(agent.usage_snapshot(), (4.0, 400, 7))
        self.assertEqual(_row(parent), (4.0, 400, 7))
        top, _chat = th._add_task("ancestor of the raced parent")
        ancestor = ChatSorcarAgent("remote-raced-ancestor")
        sorcar_agent._attribute_sub_usage(ancestor, *agent.usage_snapshot())
        with th.TASK_USAGE_LOCK:
            ancestor._save_final_usage({}, top, result_summary="Done")
            agent._save_final_usage({}, parent, result_summary="Done", unless_finished=True)
        self.assertEqual((_row(top), _row(parent)), ((4.0, 400, 7), (4.0, 400, 7)))
        self.assertEqual(_settled(parent), [1, 1])

    def test_spend_already_on_the_ledger_is_mirrored_to_the_finished_row_once(self) -> None:
        """Spend made after a row's final save that its agent's ledger already holds (the
        retirement's commit message, ``_charge_last_task``: no live agent is passed) is
        mirrored to the finished row without a replay record, so the next absolute
        snapshot of that ledger, which carries it, adds nothing.  A finished ancestor's
        ledger never had it: its share is retained and replayed over its own snapshot."""
        from kiss.server.json_printer import JsonPrinter
        from kiss.server.task_runner import _save_subtask_extra

        ancestor = _task(OPUS, 3.0, 300, 5)
        task = _task(OPUS, 3.0, 300, 5, parent=ancestor)
        agents = {}
        for row_id in (ancestor, task):
            agent = ChatSorcarAgent("late-own-spend")
            with agent._task_id_lock:
                agent._last_task_id = row_id
            sorcar_agent._attribute_sub_usage(agent, 3.0, 300, 5)  # the ledger the row holds
            agents[row_id] = agent
        states = {
            row_id: AgentState(row_id, agent=agent, tab_id="tab", server_owned=True)
            for row_id, agent in agents.items()
        }
        own = agents[task]
        # Through the daemon's bridge (charge_task_usage), then without one.
        for printer, delta in ((JsonPrinter(), 0.25), (None, 0.5)):
            own.printer = printer
            sorcar_agent._attribute_sub_usage(own, delta, int(delta * 100), 0)  # on the ledger
            own._charge_last_task(delta, int(delta * 100))
            total = sorcar_agent._live_agent_usage(own)
            self.assertEqual(_row(task), total)
            self.assertEqual(_row(ancestor), total)
            self.assertEqual(_charges(task), [])  # its ledger holds the spend: nothing to replay
            self.assertEqual(_charges(ancestor)[-1], (delta, int(delta * 100), 0))
            self.assertEqual(_settled(ancestor), [1] * len(_charges(ancestor)))
            for row_id in (task, ancestor):  # each row's own ledger snapshot, twice
                with th.TASK_USAGE_LOCK:
                    _save_subtask_extra(states[row_id], row_id, OPUS, self.local_dir, 1, 2)
                self.assertEqual(_row(row_id), total)
                with th.TASK_USAGE_LOCK:
                    agents[row_id]._save_final_usage({}, row_id, result_summary="Done")
                self.assertEqual(_row(row_id), total)
            th._refresh_finished_rows({task, ancestor})
            self.assertEqual((_row(task), _row(ancestor)), (total, total))
        self.assertEqual(_row(task), (3.75, 375, 5))
        self.assertEqual(_charges(ancestor), [(0.25, 25, 0), (0.5, 50, 0)])
        # Without a bridge the mirror waits for the final writers' lock: a writer that
        # snapshotted the ledger before the mirror would otherwise overwrite it, and a
        # banked mirror leaves no record to replay.
        import threading

        sorcar_agent._attribute_sub_usage(own, 0.25, 25, 0)
        mirror = threading.Thread(target=own._charge_last_task, args=(0.25, 25))
        with th.TASK_USAGE_LOCK:  # a final writer between its snapshot and its row write
            mirror.start()
            deadline = time.monotonic() + 0.5
            while mirror.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(mirror.is_alive())
            self.assertEqual(_row(task), (3.75, 375, 5))  # not yet
        mirror.join(5)
        self.assertFalse(mirror.is_alive())
        self.assertEqual(_row(task), (4.0, 400, 5))
        with th.TASK_USAGE_LOCK:  # the writer's snapshot, taken after the mirror, agrees
            own._save_final_usage({}, task, result_summary="Done")
        self.assertEqual(_row(task), (4.0, 400, 5))

    def test_several_pending_charges_are_each_counted_once(self) -> None:
        """Charges another process recorded stay off the ledger while the owning daemon banks
        its own; the ledger, the marks, both final saves and the next syncs agree on each
        charge counted once."""
        from kiss.server.task_runner import _save_subtask_extra

        parent, _chat = th._add_task(f"task on {OPUS}")
        agent = ChatSorcarAgent("remote-late-parent")
        with agent._task_id_lock:
            agent._last_task_id = parent
        state = AgentState(parent, agent=agent, tab_id="tab", server_owned=True)
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, parent, state)
        sorcar_agent._attribute_sub_usage(agent, 9.0, 900, 14)  # $3 own, three $2 folds
        self._on(self.remote_dir)
        children = {
            name: _task(ASTRA, cost, int(cost * 100), 4)
            for name, cost in (("a", 2.25), ("b", 2.25), ("c", 2.5))
        }
        usage = th._task_usage(list(children.values()))
        self._on(self.local_dir)
        reports = {name: usage[child] for name, child in children.items()}
        for name in reports:
            th._record_remote_subtask(f"remote-{name}", parent, "box", ASTRA, 2.0, 200, 3)
        self._charge_from_another_process("remote-a", reports["a"])
        self._charge_from_another_process("remote-b", reports["b"])
        self.assertEqual(_charges(parent), [(0.25, 25, 1), (0.25, 25, 1)])
        self.assertEqual(sorcar_agent._live_agent_usage(agent), (9.0, 900, 14))

        charge_remote_subtask_usage(None, "remote-c", reports["c"])

        self.assertEqual(sorcar_agent._live_agent_usage(agent), (9.5, 950, 15))
        marks = th._get_db().execute(
            "SELECT banked, banked_cost, banked_tokens, banked_steps FROM remote_charges "
            "WHERE parent_task_id = ? ORDER BY id", (parent,),
        ).fetchall()
        self.assertEqual(
            [tuple(m) for m in marks], [(0, 0.0, 0, 0), (0, 0.0, 0, 0), (1, 9.0, 900, 14)]
        )
        for name, report in reports.items():  # every report seen again: nothing more
            charge_remote_subtask_usage(None, f"remote-{name}", report)
        self.assertEqual(sorcar_agent._live_agent_usage(agent), (9.5, 950, 15))
        th._save_task_result("Done", parent)
        with th.TASK_USAGE_LOCK:
            agent._save_final_usage({}, parent, result_summary="Done")
        self.assertEqual(_row(parent), (10.0, 1000, 17))
        self.assertEqual(_settled(parent), [1, 1, 1])
        with th.TASK_USAGE_LOCK:
            _save_subtask_extra(state, parent, OPUS, self.local_dir, 1_000, 2_000)
        self.assertEqual(_row(parent), (10.0, 1000, 17))
        self.assertEqual(_settled(parent), [1, 1, 1])
        self.assertEqual(_costs_by_model(), {OPUS: 3.0, ASTRA: 7.0})

    def test_every_record_is_asked_for_in_batches(self) -> None:
        parent = _task(OPUS, 1.0, 100, 1)
        ids = [f"remote-child-{n:04d}" for n in range(501)]
        for task_id in ids:
            th._record_remote_subtask(task_id, parent, "box", ASTRA, 0.0, 0, 0)
        asked: list[list[str]] = []

        class Link:
            def request(self, cmd: dict[str, Any], reply_type: str, **_: Any) -> dict[str, Any]:
                asked.append(list(cmd["taskIds"]))
                return {"type": reply_type, "tasks": {
                    i: {"cost": 0.01, "tokens": 1, "steps": 0} for i in cmd["taskIds"]}}

        task_update._last_full_sweep.clear()
        self.addCleanup(task_update._last_full_sweep.clear)
        with mock.patch("kiss.agents.sorcar.machines.link_for", return_value=Link()):
            sync_remote_subtask_usage(None, SimpleNamespace(name="box"))
        self.assertEqual([len(batch) for batch in asked], [500, 1])
        # an old record (dispatched 30 days ago) is asked for by the hourly full sweep
        th._get_db().execute("UPDATE remote_subtasks SET timestamp = ? WHERE task_id = ?",
                             (time.time() - 30 * 86400, ids[0]))
        th._get_db().commit()
        asked.clear()
        with mock.patch("kiss.agents.sorcar.machines.link_for", return_value=Link()):
            sync_remote_subtask_usage(None, SimpleNamespace(name="box"))  # within the hour
        self.assertNotIn(ids[0], [i for batch in asked for i in batch])
        asked.clear()
        with mock.patch("kiss.agents.sorcar.machines.link_for", return_value=Link()), \
                mock.patch.dict(task_update._last_full_sweep, {"box": 0.0}):
            sync_remote_subtask_usage(None, SimpleNamespace(name="box"))  # sweep due
        self.assertIn(ids[0], [i for batch in asked for i in batch])
        self.assertEqual(len(asked), 2)
        self.assertEqual(sorted(i for batch in asked for i in batch), ids)
        cost, tokens, _ = _row(parent)
        self.assertAlmostEqual(cost, 1.0 + 5.01)
        self.assertEqual(tokens, 100 + 501)

    def _mirror(self, parent: str, child: str, with_grandchild: bool) -> None:
        """Copy C's row (and its $0.5 fable grandchild) into the local history."""
        copy = _task(ASTRA, 2.0, 200, 3, parent=parent)
        db = th._get_db()
        db.execute("UPDATE task_history SET id = ? WHERE id = ?", (child, copy))
        db.commit()
        if with_grandchild:
            _task(FABLE, 0.5, 50, 1, parent=child)

    def test_a_mirrored_child_counts_once_by_its_grown_record(self) -> None:
        parent, child = self._dispatch()
        self._mirror(parent, child, with_grandchild=False)
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 2.0})
        self._late_charge_on_the_machine(child, 0.25)
        charge_remote_subtask_usage(None, child, self._report(child))
        self.assertEqual(_row(parent), (3.25, 325, 6))
        # The grown record (with the machine's split) replaces the stale copy.
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.75, FABLE: 0.5})

    def test_a_mirrored_child_counts_by_a_record_that_grew_in_tokens_alone(self) -> None:
        """Late spend on a free model moves the record's tokens and not its cost."""
        parent, child = self._dispatch()
        self._mirror(parent, child, with_grandchild=False)
        charge_remote_subtask_usage(None, child, {"cost": 2.0, "tokens": 230, "steps": 4})
        self.assertEqual(_row(parent), (3.0, 330, 6))
        tokens = {entry["model"]: entry["tokens"] for entry in th._spend_by_day_and_model()}
        self.assertEqual(tokens, {OPUS: 100, ASTRA: 230})
        models = th._task_usage([parent])[parent]["models"]
        self.assertEqual(models, {OPUS: [1.0, 100.0], ASTRA: [2.0, 230.0]})

    def test_a_mirrored_tree_is_not_counted_twice_by_the_grown_record(self) -> None:
        parent, child = self._dispatch()
        self._mirror(parent, child, with_grandchild=True)
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.5, FABLE: 0.5})
        self._late_charge_on_the_machine(child, 0.25)
        charge_remote_subtask_usage(None, child, self._report(child))
        # The record's split covers C's whole tree: the copied grandchild is not
        # walked again.
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.75, FABLE: 0.5})

    def test_an_equal_mirrored_copy_takes_the_machine_split(self) -> None:
        parent, child = self._dispatch()
        self._mirror(parent, child, with_grandchild=False)
        report = self._report(child)
        self.assertEqual(report["cost"], 2.0)
        charge_remote_subtask_usage(None, child, report)  # owes nothing, stores the split
        self.assertEqual(_row(parent), (3.0, 300, 5))
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.5, FABLE: 0.5})

    def _split(self, fable: float) -> dict[str, list[float]]:
        return {ASTRA: [1.5, 150.0], FABLE: [fable, 50.0]}

    def test_a_sub_agent_dispatched_back_here_is_inside_the_remote_split(self) -> None:
        """P -> Q on box -> R back on this machine: Q's split already covers R."""
        parent = _task(OPUS, 3.0, 300, 5)
        remote = uuid.uuid4().hex  # Q's id: its row is on box only
        th._record_remote_subtask(remote, parent, "box", ASTRA, 2.0, 200, 3)
        _task(FABLE, 0.5, 50, 1, parent=remote)
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.5, FABLE: 0.5})
        charge_remote_subtask_usage(None, remote, {"cost": 2.0, "tokens": 200, "steps": 3,
                                                   "models": self._split(0.5)})
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.5, FABLE: 0.5})

    def test_an_older_report_keeps_the_newer_split(self) -> None:
        parent = _task(OPUS, 3.0, 300, 5)
        th._record_remote_subtask("q", parent, "box", ASTRA, 2.0, 200, 3)
        charge_remote_subtask_usage(None, "q", {"cost": 2.25, "tokens": 225, "steps": 4,
                                                "models": self._split(0.75)})
        charge_remote_subtask_usage(None, "q", {"cost": 2.0, "tokens": 200, "steps": 3,
                                                "models": self._split(0.5)})
        self.assertEqual(_row(parent), (3.25, 325, 6))
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 1.5, FABLE: 0.75})


class TestSyncTriggers:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        th._close_thread_db()
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_a_this_machine_history_refresh_starts_the_sync(self) -> None:
        server, _events = _make_server()
        with mock.patch("kiss.server.server.start_remote_usage_sync") as start:
            server._get_history(query=None, offset=0, generation=0, machine=LOCAL_MACHINE)
            server._get_history(query=None, offset=50, generation=0, machine=LOCAL_MACHINE)
        assert start.call_count == 1

    def test_a_spend_refresh_starts_the_sync(self) -> None:
        replies: list[dict[str, Any]] = []

        async def reply(endpoint: Any, event: dict[str, Any], kind: str) -> None:
            replies.append(event)

        backend = mock.Mock(_printer="printer", _reply_direct=reply)
        api = mock.Mock(_backend=backend)
        machines = [mock.Mock(name="box")]
        with mock.patch.object(task_update, "start_remote_usage_sync") as start, \
                mock.patch("kiss.agents.sorcar.machines.load_machines", return_value=machines), \
                mock.patch.object(server_api.sidebar_panels, "spend_report",
                                  return_value={"total": 1.0}):
            asyncio.run(server_api.ServerApi.get_spend_report(api, {}, mock.Mock()))
        start.assert_called_once_with("printer", machines)
        assert replies == [{"type": "spendReport", "total": 1.0}]

    def test_a_usage_request_answers_then_starts_a_rate_limited_sync(self) -> None:
        """A nested dispatch: the machine asked for usage syncs its own remote sub-agents."""
        from kiss.server.commands import _CommandsMixin

        sent: list[dict[str, Any]] = []
        handler = SimpleNamespace(printer="printer",
                                  _broadcast_to_conn=lambda event, conn: sent.append(event))
        machines = [SimpleNamespace(name="box")]
        with mock.patch.object(task_update, "start_remote_usage_sync") as start, \
                mock.patch("kiss.agents.sorcar.machines.load_machines", return_value=machines):
            _CommandsMixin._cmd_get_task_usage(
                cast(Any, handler), {"taskIds": ["nope"], "connId": "c"}
            )
        assert sent == [{"type": "taskUsage", "tasks": {}}]
        start.assert_called_once_with(
            "printer", machines, min_interval=task_update.REMOTE_USAGE_NESTED_INTERVAL_S)

    def test_the_min_interval_keeps_daemons_from_asking_each_other_without_end(self) -> None:
        runs: list[str] = []

        def fake_sync(printer: Any, machines: list[Any]) -> None:
            runs.append(printer)
            task_update._remote_sync_lock.release()

        machines = [SimpleNamespace(name="box")]
        with mock.patch.object(task_update, "_sync_machines", fake_sync), \
                mock.patch.object(task_update, "_last_sync_started", 0.0), \
                mock.patch.object(task_update.threading, "Thread") as thread:
            thread.side_effect = lambda target, args, **_: SimpleNamespace(
                start=lambda: target(*args))
            task_update.start_remote_usage_sync("first", machines, min_interval=60.0)
            task_update.start_remote_usage_sync("nested", machines, min_interval=60.0)
            task_update.start_remote_usage_sync("refresh", machines)
        assert runs == ["first", "refresh"]
        assert not task_update._remote_sync_lock.locked()
