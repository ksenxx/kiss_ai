# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Non-worktree auto-commit must also commit repos other than work_dir.

The user-observed defect:

    "in the auto-commit and non worktree mode, if a task changes
    files, it does not auto-commit the files."

Root cause: at task end the auto-commit pass ran ``git add -A`` in
the repository containing the tab's *work_dir* — and nowhere else.  A
task is free to change files anywhere (the file tools take absolute
paths), so files it wrote in a DIFFERENT repository were silently left
uncommitted (observed in production: tasks with ``work_dir`` in one
project editing a sibling project's checkout).

The fix tracks the paths of ``Write`` / ``Edit`` tool calls per task
(in the printer, since event persistence is asynchronous), groups them
by containing repository at task end, and commits each extra
repository — staging ONLY the recorded paths so unrelated dirty state
in a repository the user never designated as work_dir is not swept in.

Each test drives the real :meth:`VSCodeServer._run_task_inner` against
fresh git repos, replacing only the stateful agent's parent ``run``
with a deterministic stub that changes files and reports the same
``tool_call`` events the real tool loop broadcasts (no mocks).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import kiss.agents.sorcar.persistence as _persistence


class TestSubtaskChangedPathsHelpers(unittest.TestCase):
    """The DB walk over parent_task_id descendants."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-subtaskpaths-test-")
        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

    def tearDown(self) -> None:
        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_collects_descendants_but_not_root_or_reads(self) -> None:
        root_id, _ = _persistence._add_task("root")
        child_id, _ = _persistence._add_task(
            "child", extra={"parent_task_id": root_id},
        )
        grandchild_id, _ = _persistence._add_task(
            "grandchild", extra={"parent_task_id": child_id},
        )
        unrelated_id, _ = _persistence._add_task("unrelated")
        _persistence._append_chat_event(
            {"type": "tool_call", "name": "Write", "path": "/r/root.txt"},
            task_id=root_id,
        )
        _persistence._append_chat_event(
            {"type": "tool_call", "name": "Edit", "path": "/r/child.txt"},
            task_id=child_id,
        )
        _persistence._append_chat_event(
            {"type": "tool_call", "name": "Write", "path": "/r/grand.txt"},
            task_id=grandchild_id,
        )
        _persistence._append_chat_event(
            {"type": "tool_call", "name": "Read", "path": "/r/read.txt"},
            task_id=child_id,
        )
        _persistence._append_chat_event(
            {"type": "tool_call", "name": "Write", "path": "/r/other.txt"},
            task_id=unrelated_id,
        )

        ids = _persistence._descendant_task_ids(root_id)
        assert set(ids) == {child_id, grandchild_id}
        paths = _persistence._changed_paths_of_tasks(ids)
        assert paths == {"/r/child.txt", "/r/grand.txt"}

    def test_no_descendants_returns_empty(self) -> None:
        root_id, _ = _persistence._add_task("lonely root")
        assert _persistence._descendant_task_ids(root_id) == []
        assert _persistence._changed_paths_of_tasks([]) == set()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
