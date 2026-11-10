"""Cost audit 2026-11-09: remote charges a row got with no live agent, and orphan model splits.

* A late remote charge reported while its parent row runs with no live
  agent in the daemon's registry (a bridge-owned run, unregistered
  before its final save; a run outside the daemon) used to reach the
  row only, at its finalization: the run's ledger, whose totals its
  caller folds, never held it.  The run's final save now banks it on
  the ledger first, so the ledger, the row and the server's later
  snapshot of the same ledger all hold it exactly once.
* Orphan recovery now writes the row's ``model_costs`` from its
  persisted ``llm_call`` events, as a final save does, so the Spend
  panel splits a recovered task's own spend by the models it ran on.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from typing import cast
from unittest import mock

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar import sorcar_agent
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.task_update import charge_remote_subtask_usage
from kiss.tests.agents.sorcar import test_audit1103_remote_costs as base
from kiss.tests.agents.sorcar.test_audit1103_remote_costs import (
    ASTRA,
    FABLE,
    OPUS,
    _costs_by_model,
    _row,
    _task,
)


def _flags(parent: str) -> list[tuple[int, int]]:
    """``(banked, settled)`` of each remote charge *parent* owes, oldest first."""
    rows = th._get_db().execute(
        "SELECT banked, settled FROM remote_charges WHERE parent_task_id = ? ORDER BY id",
        (parent,),
    ).fetchall()
    return [(int(r[0]), int(r[1])) for r in rows]


class TestRemoteFinalBank(base.TestRemoteSubagentCosts):
    """Reuses the P ($3, opus) -> C ($2 on box) set-up of the 11-03 audit."""

    def _charge_from_another_process(self, child: str, report: dict[str, object]) -> None:
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

    def _parent_agent(self, register: bool) -> tuple[str, ChatSorcarAgent, AgentState]:
        parent, _chat = th._add_task(f"task on {OPUS}")
        agent = ChatSorcarAgent("remote-final-bank-parent")
        with agent._task_id_lock:
            agent._last_task_id = parent
        state = AgentState(parent, agent=agent, tab_id="tab", server_owned=True)
        if register:
            agent_state.register(state)
            self.addCleanup(agent_state.unregister, parent, state)
        sorcar_agent._attribute_sub_usage(agent, 3.0, 300, 5)
        return parent, agent, state

    def _late_report(self) -> dict[str, object]:
        self._on(self.remote_dir)
        remote_row = _task(ASTRA, 2.5, 250, 4)
        report = self._report(remote_row)
        self._on(self.local_dir)
        return report

    def test_the_final_save_banks_a_charge_no_live_agent_took(self) -> None:
        """Reported while the row runs and no agent is registered for it (the bridge's
        unregister-before-final-save window): the final save banks it on the ledger,
        so the ledger the caller folds holds it, and the row holds it once."""
        from kiss.server.task_runner import _save_subtask_extra

        parent, agent, state = self._parent_agent(register=False)
        self._closing_usage(parent)
        report = self._late_report()
        th._record_remote_subtask("remote-unreg-child", parent, "box", ASTRA, 2.0, 200, 3)
        charge_remote_subtask_usage(None, "remote-unreg-child", report)
        self.assertEqual(_flags(parent), [(0, 0)])
        self.assertEqual(agent.usage_snapshot(), (3.0, 300, 5))

        th._save_task_result("Done", parent)
        with th.TASK_USAGE_LOCK:
            agent._save_final_usage({}, parent, result_summary="Done")
        self.assertEqual(agent.usage_snapshot(), (3.5, 350, 6))
        self.assertEqual(_row(parent), (3.5, 350, 6))
        self.assertEqual(_flags(parent), [(1, 1)])
        # The server's own final save of the same row: one more snapshot of the ledger.
        with th.TASK_USAGE_LOCK:
            _save_subtask_extra(state, parent, OPUS, self.local_dir, 1_000, 2_000)
        self.assertEqual(_row(parent), (3.5, 350, 6))
        th._refresh_finished_rows({parent})
        self.assertEqual(_row(parent), (3.5, 350, 6))
        charge_remote_subtask_usage(None, "remote-unreg-child", report)  # seen again
        self.assertEqual(_row(parent), (3.5, 350, 6))
        self.assertEqual(_costs_by_model(), {OPUS: 1.0, ASTRA: 2.5})
        # A retried save of the finished row writes nothing and banks nothing more.
        with th.TASK_USAGE_LOCK:
            agent._save_final_usage({}, parent, result_summary="Done", unless_finished=True)
        self.assertEqual(agent.usage_snapshot(), (3.5, 350, 6))
        self.assertEqual(_row(parent), (3.5, 350, 6))
        # The task's latest persisted usage event (its replayed header) holds it too.
        self.assertEqual(self._last_usage(parent), (3.5, 350, 6))

    def _closing_usage(self, task_id: str) -> None:
        """The run's own closing usage event, at its ledger's $3.00 / 300 / 5."""
        th._append_chat_event(
            {"type": "usage_info", "text": "Budget: $3.00", "taskId": task_id,
             "total_tokens": 300, "cost": "$3.0000", "cost_usd": 3.0, "total_steps": 5},
            task_id=task_id,
        )
        self.assertEqual(self._last_usage(task_id), (3.0, 300, 5))

    def _last_usage(self, task_id: str) -> tuple[float, int, int]:
        th._flush_chat_events(task_id)
        rows = th._get_db().execute(
            "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq", (task_id,)
        ).fetchall()
        usage = [e for e in (json.loads(r[0]) for r in rows) if e.get("type") == "usage_info"]
        last = usage[-1]
        return (float(last["cost_usd"]), int(last["total_tokens"]), int(last["total_steps"]))

    def test_a_stop_between_the_mark_and_the_bank_is_banked_by_the_retried_save(self) -> None:
        """A Stop lands after the charge's mark committed and before its record: the
        retried final save (``unless_finished``) still banks it, once."""
        parent, agent, _state = self._parent_agent(register=False)
        self._closing_usage(parent)
        report = self._late_report()
        th._record_remote_subtask("remote-stop-child", parent, "box", ASTRA, 2.0, 200, 3)
        charge_remote_subtask_usage(None, "remote-stop-child", report)
        th._save_task_result("Done", parent)
        real_mark = th._mark_remote_charge_banked

        def mark_then_stop(*args: object) -> None:
            real_mark(*args)  # type: ignore[arg-type]
            raise KeyboardInterrupt

        with mock.patch.object(th, "_mark_remote_charge_banked", mark_then_stop):
            with self.assertRaises(KeyboardInterrupt), th.TASK_USAGE_LOCK:
                agent._save_final_usage({}, parent, result_summary="Done")
        self.assertEqual(_flags(parent), [(1, 0)])
        self.assertEqual(agent.usage_snapshot(), (3.0, 300, 5))
        with th.TASK_USAGE_LOCK:
            agent._save_final_usage({}, parent, result_summary="Done", unless_finished=True)
        self.assertEqual(agent.usage_snapshot(), (3.5, 350, 6))
        self.assertEqual(_row(parent), (3.5, 350, 6))
        self.assertEqual(_flags(parent), [(1, 1)])
        self.assertEqual(self._last_usage(parent), (3.5, 350, 6))

    def test_a_stop_after_the_bank_still_persists_the_final_usage_on_retry(self) -> None:
        """A Stop lands after the charge is on the ledger, before the row is finished:
        the retried save finds nothing more to bank and still persists the final usage."""
        parent, agent, _state = self._parent_agent(register=False)
        self._closing_usage(parent)
        report = self._late_report()
        th._record_remote_subtask("remote-stop2-child", parent, "box", ASTRA, 2.0, 200, 3)
        charge_remote_subtask_usage(None, "remote-stop2-child", report)
        th._save_task_result("Done", parent)
        real_attr = sorcar_agent._attribute_sub_usage

        def bank_then_stop(*args: object, **kwargs: object) -> None:
            real_attr(*args, **kwargs)  # type: ignore[arg-type]
            raise KeyboardInterrupt

        with mock.patch.object(sorcar_agent, "_attribute_sub_usage", bank_then_stop):
            with self.assertRaises(KeyboardInterrupt), th.TASK_USAGE_LOCK:
                agent._save_final_usage({}, parent, result_summary="Done")
        self.assertEqual(agent.usage_snapshot(), (3.5, 350, 6))
        self.assertEqual(self._last_usage(parent), (3.0, 300, 5))
        with th.TASK_USAGE_LOCK:
            agent._save_final_usage({}, parent, result_summary="Done", unless_finished=True)
        self.assertEqual(agent.usage_snapshot(), (3.5, 350, 6))
        self.assertEqual(_row(parent), (3.5, 350, 6))
        self.assertEqual(_flags(parent), [(1, 1)])
        self.assertEqual(self._last_usage(parent), (3.5, 350, 6))

    def test_a_charge_from_another_process_is_banked_by_the_final_save(self) -> None:
        parent, agent, _state = self._parent_agent(register=True)
        report = self._late_report()
        th._record_remote_subtask("remote-proc-child", parent, "box", ASTRA, 2.0, 200, 3)
        self._charge_from_another_process("remote-proc-child", report)
        self.assertEqual(_flags(parent), [(0, 0)])
        th._save_task_result("Done", parent)
        with th.TASK_USAGE_LOCK:
            agent._save_final_usage({}, parent, result_summary="Done")
        self.assertEqual(agent.usage_snapshot(), (3.5, 350, 6))
        self.assertEqual(_row(parent), (3.5, 350, 6))
        self.assertEqual(_flags(parent), [(1, 1)])

    def test_a_charge_banked_live_is_not_banked_again(self) -> None:
        parent, agent, _state = self._parent_agent(register=True)
        report = self._late_report()
        th._record_remote_subtask("remote-live-child", parent, "box", ASTRA, 2.0, 200, 3)
        charge_remote_subtask_usage(None, "remote-live-child", report)
        self.assertEqual(agent.usage_snapshot(), (3.5, 350, 6))
        self.assertEqual(_flags(parent), [(1, 0)])
        th._save_task_result("Done", parent)
        with th.TASK_USAGE_LOCK:
            agent._save_final_usage({}, parent, result_summary="Done")
        self.assertEqual(agent.usage_snapshot(), (3.5, 350, 6))
        self.assertEqual(_row(parent), (3.5, 350, 6))
        self.assertEqual(_flags(parent), [(1, 1)])

    def test_orphan_recovery_writes_the_split_by_model(self) -> None:
        """A killed task that switched models: the recovered row's Spend split follows
        its own calls, not its launch model."""
        script = textwrap.dedent(f"""
            import os
            from kiss.agents.sorcar import persistence as th
            p, _ = th._add_task('orphan with a model switch', '', {{'model': {OPUS!r}}})
            for model, cost in (({OPUS!r}, 0.25), ({FABLE!r}, 0.75)):
                th._append_chat_event({{'type': 'llm_call', 'model': model, 'cost': cost,
                                        'input_tokens': int(cost * 100)}}, task_id=p)
            th._append_chat_event({{'type': 'usage_info', 'text': 'Steps: 5/100',
                                    'cost_usd': 1.0, 'total_tokens': 100,
                                    'total_steps': 5}}, task_id=p)
            th._flush_chat_events(p)
            print(p, flush=True)
            os._exit(0)
            """)
        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
            env={**os.environ, "KISS_HOME": str(th._KISS_DIR),
                 "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        orphan = done.stdout.strip().splitlines()[-1]
        self.assertEqual(th._recover_orphaned_tasks(set()), 1)
        row = th._get_db().execute(
            "SELECT cost, model_costs FROM task_history WHERE id = ?", (orphan,)
        ).fetchone()
        self.assertEqual(float(row[0]), 1.0)
        self.assertEqual(json.loads(row[1]), {OPUS: [0.25, 25], FABLE: [0.75, 75]})
        models = cast(dict[str, list[float]], th._task_usage([orphan])[orphan]["models"])
        self.assertEqual({m: v[0] for m, v in models.items()}, {OPUS: 0.25, FABLE: 0.75})
