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
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.machines import LOCAL_MACHINE
from kiss.server import sorcar as server_api
from kiss.server import task_update
from kiss.server.task_update import charge_remote_subtask_usage, sync_remote_subtask_usage
from kiss.tests.agents.sorcar import test_audit1103_remote_costs as base
from kiss.tests.agents.sorcar.test_audit1103_remote_costs import (
    ASTRA,
    FABLE,
    OPUS,
    _costs_by_model,
    _row,
    _task,
)
from kiss.tests.agents.sorcar.test_history_date_range import _redirect, _restore
from kiss.tests.server.test_history_date_range import _make_server


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
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
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
