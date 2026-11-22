# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cost audit 2026-11-22: the orphan sweep's usage events against a concurrent late charge.

The sweep adds what a dead row owed to its finished ancestors and then
persists a ``usage_info`` with each one's totals.  It used to publish
the totals its transaction had read, after the commit and without
``TASK_USAGE_LOCK``: a late charge landing in between published the
newer totals first and the sweep appended its older ones, so the row
held $13 and its transcript ended with a ``usage_info`` of $12, the
cost every reader of the latest totals event took for the task's.

A real temporary history database, a real second thread and no sleeps:
the charge runs from a profile hook of the sweeping thread, at the
release of the sweep's write lock.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import FrameType
from typing import Any, cast

from kiss.agents.sorcar import persistence as th
from kiss.server.task_update import charge_remote_subtask_usage
from kiss.tests.agents.sorcar.test_audit1103_remote_costs import _row
from kiss.tests.agents.sorcar.test_orphan_progress_backfill import _redirect, _restore

PARENT, DEAD, OWED, LATE = "1" * 32, "2" * 32, "3" * 32, "4" * 32


def _charge_late(seen: list[object]) -> None:
    """Thread body: ``LATE`` reports $1 more; *seen* gets PARENT's row as the charge starts."""
    try:
        seen.append(_row(PARENT))
        charge_remote_subtask_usage(None, LATE, {
            "cost": 1.0, "tokens": 100, "steps": 1, "models": {"remote-model": [1.0, 100]},
        })
    finally:
        th._close_thread_db()


class _SweepGap:
    """Profile function of the sweeping thread.

    As the sweep's write lock is released (its transaction committed,
    its publication ahead), *charge* runs to its end on its own thread.
    Each event the sweep queues afterwards records whether its thread
    holds ``TASK_USAGE_LOCK``.
    """

    def __init__(self, charge: threading.Thread) -> None:
        self.charge = charge
        self.locked: list[bool] = []

    def __call__(self, frame: FrameType, what: str, arg: object) -> None:
        code = frame.f_code
        if what == "return" and code is th._RWLock._teardown.__code__:
            if frame.f_locals["writer"] and self.charge.ident is None:
                self.charge.start()
                self.charge.join(60)
        elif what == "call" and code is th._queue_chat_event.__code__:
            self.locked.append(cast(Any, th.TASK_USAGE_LOCK)._is_owned())


class TestOrphanSweepUsageOrder(unittest.TestCase):
    """PARENT ($10, finished) has a dead child that owes a $2 remote charge."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_sweep_order_"))
        self.saved = _redirect(self.tmp)
        db = th._get_db()
        now = time.time()
        for task_id, parent, cost, end_ts, result in (
            (PARENT, "", 10.0, int((now - 20) * 1000), "finished"),
            (DEAD, PARENT, 0.0, 0, "Agent Failed Abruptly"),
        ):
            db.execute(
                "INSERT INTO task_history (id, timestamp, task, result, owner, parent_task_id, "
                "model, cost, tokens, steps, end_ts) VALUES (?, ?, 'sweep', ?, '', ?, 'model', "
                "?, ?, ?, ?)",
                (task_id, now - 100, result, parent, cost, int(cost * 100), int(cost), end_ts),
            )
        result_event = {"type": "result", "taskId": PARENT, "success": True, "cost": "$10.0000",
                        "cost_usd": 10.0, "total_tokens": 1000, "step_count": 10}
        db.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, 1, ?, ?)",
            (PARENT, json.dumps(result_event), now - 20),
        )
        th._record_remote_subtask(OWED, DEAD, "remote", "remote-model", 0.0, 0, 0)
        th._charge_remote_subtask(OWED, 2.0, 200, 2, {})  # DEAD, unfinished, owes it
        th._record_remote_subtask(LATE, PARENT, "remote", "remote-model", 0.0, 0, 0)

    def tearDown(self) -> None:
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_last_usage_event_holds_a_charge_made_as_the_sweep_committed(self) -> None:
        seen: list[object] = []
        gap = _SweepGap(threading.Thread(target=_charge_late, args=(seen,), daemon=True))
        sys.setprofile(gap)
        try:
            self.assertEqual(th._recover_orphaned_tasks(set(), time.time()), 1)
        finally:
            sys.setprofile(None)
        # The charge found the sweep's $2 committed, and ended before the sweep published.
        self.assertEqual(seen, [(12.0, 1200, 12)])
        self.assertFalse(gap.charge.is_alive())
        self.assertEqual(_row(PARENT), (13.0, 1300, 13))
        th._flush_chat_events()
        session = th._load_chat_events_by_task_id(PARENT)
        assert session is not None
        events = cast("list[dict[str, Any]]", session["events"])
        last = [event for event in events if event["type"] == "usage_info"][-1]
        self.assertEqual(
            (last["cost_usd"], last["total_tokens"], last["total_steps"]), (13.0, 1300, 13),
        )
        self.assertEqual(gap.locked, [True])  # the sweep's one event, serialized with the charge's
