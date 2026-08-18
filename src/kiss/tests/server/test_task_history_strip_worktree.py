# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test that the working directory saved to the task history table
has the ``.kiss-worktrees/kiss_wt-<slug>`` worktree suffix stripped.

BUG: When a task runs inside a git worktree, the agent's ``work_dir``
points at ``<repo>/.kiss-worktrees/kiss_wt-<slug>``.  This worktree
directory is ephemeral — it is removed once the worktree is merged or
discarded.  Persisting that path verbatim in ``task_history.extra``
means later history loads see a workspace path that no longer exists
on disk, breaking the history sidebar's "Workspace" filter (the row
appears to belong to a workspace the user never opened).

Fix: strip the ``.kiss-worktrees/kiss_wt-<slug>[/...]`` suffix before
persisting ``work_dir`` to ``task_history.extra``, leaving the parent
repository path (the user-visible workspace folder) in the database.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import _add_task, _save_task_extra
from kiss.tests.agents.sorcar.test_task_history_strip_worktree import (  # noqa: F401
    _redirect,
    _restore,
)


class TestSaveTaskExtraEndToEnd(unittest.TestCase):
    """End-to-end: a payload whose ``work_dir`` is a worktree path,
    persisted via ``_save_task_extra``, must round-trip with the
    parent repo path stored in ``task_history.extra``."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def tearDown(self) -> None:
        if th._db_conn is not None:
            try:
                th._db_conn.close()
            except Exception:
                pass
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read_extra(self, task_id: str) -> dict:
        db = th._get_db()
        row = db.execute(
            "SELECT * FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        assert row is not None
        raw = th._row_to_extra_json(row)
        result: dict = json.loads(raw) if raw else {}
        return result

    def test_task_runner_payload_persists_stripped_work_dir(self) -> None:
        """Mirror the literal payload built in
        ``kiss.server.task_runner._run_task_inner`` and assert
        that it persists the parent repo path (the fix must apply at
        the task_runner call site too)."""
        from kiss.server.task_runner import build_task_extra_payload

        task_id, _chat_id = _add_task("runner task", "")
        wt = "/repo/.kiss-worktrees/kiss_wt-XYZ-87654321"
        payload = build_task_extra_payload(
            model="claude-opus-4-7",
            work_dir=wt,
            version="test",
            tokens=10,
            cost=0.01,
            steps=3,
            is_parallel=False,
            is_worktree=True,
            auto_commit_mode=False,
            start_ms=1,
            end_ms=2,
        )
        _save_task_extra(payload, task_id=task_id)

        stored = self._read_extra(task_id)
        assert stored["work_dir"] == "/repo", (
            f"task_runner persisted raw worktree path: {stored['work_dir']!r}"
        )
        assert stored["is_worktree"] is True
        assert stored["model"] == "claude-opus-4-7"


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
