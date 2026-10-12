# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test that the history event marks failed tasks with ``failed: True``
and successful tasks with ``failed: False``.

This is the backend half of the "red marker for failed tasks in the
task history panel" feature.  The frontend reads ``s.failed`` from each
session entry in ``main.js`` ``renderHistory`` and renders a red dot
(``.sidebar-item-failed``).
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.server.server import VSCodeServer


class TestHistoryFailedFlag(unittest.TestCase):
    """``getHistory`` reports per-task failed status for UI marking."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
        th._close_db()
        th._DB_PATH = Path(self._tmp) / "history.db"  # type: ignore[attr-defined]

        self.server = VSCodeServer()
        self.server.work_dir = self._tmp
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        orig = self.server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with self._lock:
                self.events.append(dict(e))
            orig(e)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        th._close_db()
        th._DB_PATH = self._orig_db_path  # type: ignore[attr-defined]
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            for e in reversed(self.events):
                if e["type"] == "history":
                    return list(e["sessions"])
        return []

    def test_failed_flag_for_each_result_kind(self) -> None:
        ok_id, _ = th._add_task("ok task")
        th._save_task_result(result="all good", task_id=ok_id)

        fail_id, _ = th._add_task("fail task")
        th._save_task_result(result="Task failed", task_id=fail_id)

        fail2_id, _ = th._add_task("fail2 task")
        th._save_task_result(result="Task failed: boom", task_id=fail2_id)

        crash_id, _ = th._add_task("crash task")
        del crash_id

        stop_id, _ = th._add_task("stop task")
        th._save_task_result(result="Task stopped by user", task_id=stop_id)

        restart_id, _ = th._add_task("restart task")
        th._save_task_result(
            result="Task interrupted by server restart/shutdown",
            task_id=restart_id,
        )

        self.server._handle_command({"type": "getHistory"})
        sessions = self._sessions()
        by_task = {s["preview"]: s for s in sessions}

        self.assertFalse(by_task["ok task"]["failed"])
        self.assertTrue(by_task["fail task"]["failed"])
        self.assertTrue(by_task["fail2 task"]["failed"])
        self.assertTrue(by_task["crash task"]["failed"])
        self.assertTrue(by_task["stop task"]["failed"])
        self.assertTrue(by_task["restart task"]["failed"])

    def test_failed_flag_for_process_killed_recovery(self) -> None:
        """A row left as ``"Agent Failed Abruptly"`` by a SIGKILL'd
        previous process gets rewritten by ``_recover_orphaned_tasks``
        on fresh-server startup to::

            "Task terminated unexpectedly (process killed)"

        which represents a real failure and must therefore surface a
        red dot in the history sidebar (``failed=True``).
        """
        killed_id, _ = th._add_task("killed task")
        th._save_task_result(
            result="Task terminated unexpectedly (process killed)",
            task_id=killed_id,
        )

        recovered_id, _ = th._add_task("recovered task")
        th._recover_orphaned_tasks(set())

        self.server._handle_command({"type": "getHistory"})
        sessions = self._sessions()
        by_task = {s["preview"]: s for s in sessions}

        self.assertTrue(by_task["killed task"]["failed"])
        self.assertTrue(by_task["recovered task"]["failed"])
        del recovered_id

    def test_failed_flag_for_agent_verdict(self) -> None:
        """A ``finish(success=False)`` run keeps its HTML summary as the
        result text (no ``Task failed`` marker), so only the stored
        verdict can tell the history it failed: ``success=False`` must
        give ``failed=True`` (and the derived ``failed`` tag), while
        ``success=True`` and a verdict-less row stay successful.
        """
        summary = "<p>Expected <code>add &lt;name&gt; &lt;address&gt;</code>.</p>"
        verdict_fail_id, _ = th._add_task("/machine add ksen@host")
        th._save_task_result(result=summary, task_id=verdict_fail_id, success=False)
        th._save_task_extra({"endTs": 1.0}, task_id=verdict_fail_id)

        verdict_ok_id, _ = th._add_task("verdict ok task")
        th._save_task_result(result=summary, task_id=verdict_ok_id, success=True)
        th._save_task_extra({"endTs": 1.0}, task_id=verdict_ok_id)

        no_verdict_id, _ = th._add_task("no verdict task")
        th._save_task_result(result=summary, task_id=no_verdict_id)
        th._save_task_extra({"endTs": 1.0}, task_id=no_verdict_id)

        self.server._handle_command({"type": "getHistory"})
        by_task = {s["preview"]: s for s in self._sessions()}

        self.assertTrue(by_task["/machine add ksen@host"]["failed"])
        self.assertIn("failed", by_task["/machine add ksen@host"]["tags"])
        self.assertFalse(by_task["verdict ok task"]["failed"])
        self.assertNotIn("failed", by_task["verdict ok task"]["tags"])
        self.assertFalse(by_task["no verdict task"]["failed"])
        self.assertNotIn("failed", by_task["no verdict task"]["tags"])

    def test_success_column_added_to_existing_database(self) -> None:
        """A database created before the ``success`` column existed gets
        it on open, and its legacy rows (NULL verdict) are not failed.
        """
        th._close_db()
        db_path = Path(self._tmp) / "legacy.db"
        th._DB_PATH = db_path  # type: ignore[attr-defined]
        with sqlite3.connect(str(db_path)) as legacy:
            legacy.execute(
                "CREATE TABLE task_history ("
                "id TEXT PRIMARY KEY, timestamp REAL NOT NULL, task TEXT NOT NULL,"
                " has_events INTEGER DEFAULT 0, result TEXT DEFAULT '',"
                " chat_id CHAR(32) DEFAULT '', model TEXT DEFAULT '',"
                " work_dir TEXT DEFAULT '', version TEXT DEFAULT '',"
                " tokens INTEGER DEFAULT 0, cost REAL DEFAULT 0.0,"
                " steps INTEGER DEFAULT 0, is_parallel INTEGER DEFAULT 1,"
                " start_ts INTEGER DEFAULT 0, end_ts INTEGER DEFAULT 0,"
                " is_favorite INTEGER DEFAULT 0, parent_task_id TEXT DEFAULT '')"
            )
        columns = {
            r[1]
            for r in th._get_db().execute("PRAGMA table_info(task_history)").fetchall()
        }
        self.assertIn("success", columns)

        legacy_id, _ = th._add_task("legacy task")
        th._save_task_result(result="all good", task_id=legacy_id)
        verdict_id, _ = th._add_task("verdict task")
        th._save_task_result(result="<p>no</p>", task_id=verdict_id, success=False)

        self.server._handle_command({"type": "getHistory"})
        by_task = {s["preview"]: s for s in self._sessions()}
        self.assertFalse(by_task["legacy task"]["failed"])
        self.assertTrue(by_task["verdict task"]["failed"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
