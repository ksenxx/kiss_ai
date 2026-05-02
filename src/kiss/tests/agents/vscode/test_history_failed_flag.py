"""Test that the history event marks failed tasks with ``failed: True``
and successful tasks with ``failed: False``.

This is the backend half of the "red marker for failed tasks in the
task history panel" feature.  The frontend reads ``s.failed`` from each
session entry in ``main.js`` ``renderHistory`` and renders a red dot
(``.sidebar-item-failed``).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.agents.vscode.server import VSCodeServer


class TestHistoryFailedFlag(unittest.TestCase):
    """``getHistory`` reports per-task failed status for UI marking."""

    def setUp(self) -> None:
        # Use a fresh in-memory-style db file under a temp dir.
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
        # Reset connection so the next _get_db() opens against the
        # new path and re-creates schema.
        th._close_db()
        th._DB_PATH = Path(self._tmp) / "sorcar.db"  # type: ignore[attr-defined]

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
        # Successful task: result is a normal summary string.
        ok_id, _ = th._add_task("ok task")
        th._save_task_result(result="all good", task_id=ok_id)

        # Failed task: result is "Task failed".
        fail_id, _ = th._add_task("fail task")
        th._save_task_result(result="Task failed", task_id=fail_id)

        # Failed-with-message task: result starts with "Task failed: ".
        fail2_id, _ = th._add_task("fail2 task")
        th._save_task_result(result="Task failed: boom", task_id=fail2_id)

        # Crashed task: result still has the placeholder
        # "Agent Failed Abruptly" because the agent never wrote a result.
        crash_id, _ = th._add_task("crash task")
        # No _save_task_result call: row keeps default placeholder.
        del crash_id  # row exists; we only need it to appear

        # User-stopped task: should NOT be marked failed.
        stop_id, _ = th._add_task("stop task")
        th._save_task_result(result="Task stopped by user", task_id=stop_id)

        self.server._handle_command({"type": "getHistory"})
        sessions = self._sessions()
        by_task = {s["preview"]: s for s in sessions}

        self.assertFalse(by_task["ok task"]["failed"])
        self.assertTrue(by_task["fail task"]["failed"])
        self.assertTrue(by_task["fail2 task"]["failed"])
        self.assertTrue(by_task["crash task"]["failed"])
        self.assertFalse(by_task["stop task"]["failed"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
