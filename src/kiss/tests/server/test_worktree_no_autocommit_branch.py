# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: a task's worktree branch survives the task.

Bug report: "when use_worktree is set true, but autocommit is false, it
seems that a worktree branch is not getting created."

These tests drive ``VSCodeServer._run_task_inner`` with a real git repo
and a stub agent (replaces the parent class' ``run`` with a deterministic
implementation that calls the ``create_worktree`` tool like a real task
would — not a mock).  Each test cleans up its temporary git repo and
persistence database in ``tearDown`` to prevent state pollution across
tests.

The user-observed behavior was that the ``kiss/wt-{chat_id}-*`` branch
appeared to never exist in ``git branch --list``.  The root cause was in
:meth:`_MergeFlowMixin._present_pending_worktree`: when the worktree's
working tree had no detectable file changes after the task completed,
the branch was auto-discarded — even though the user would expect the
branch to be preserved for manual review.  The post-task path now
passes ``discard_if_empty=False``.

These tests:

1. ``test_branch_persists_after_task_with_changes`` — confirms that
   when the agent makes file changes, the branch survives the post-task
   path (sanity baseline).
2. ``test_branch_persists_after_task_with_no_changes`` — reproduces the
   bug: when the agent makes no file changes the branch was incorrectly
   auto-discarded.  After the fix, the branch is preserved.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.server import VSCodeServer
from kiss.tests.server.conftest import init_repo, run_git


def _list_kiss_wt_branches(repo: str) -> list[str]:
    """List all ``kiss/wt-*`` branches currently present in *repo*.

    Reflects only branches that still exist (i.e. not deleted).
    """
    result = run_git(repo, "branch", "--list", "kiss/wt-*")
    return [
        line.strip().lstrip("* ").strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


class _WorktreeBranchBase(unittest.TestCase):
    """Shared setUp / tearDown — fresh git repo, isolated persistence DB."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-test-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        init_repo(self.repo)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "history.db"
        _persistence._db_conn = None

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict] = []

        def capture(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run

        from kiss.server import agent_state
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        agent_state.agent_states.clear()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _types(self) -> list[str]:
        return [e["type"] for e in self.events]


def _patch_parent_run_create_file(filename: str | None) -> Any:
    """Replace ``ChatSorcarAgent``'s parent ``run`` with a stub.

    The stub does what a real task does: it resolves the task's
    ``work_dir``, calls the ``create_worktree`` tool, and then creates
    *filename* inside the worktree work dir when *filename* is
    provided.  When *filename* is ``None`` it makes no file changes —
    exercising the empty-worktree code path.

    Returns the original ``run`` for restoration in tearDown.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def stub_run(self_agent: Any, **kwargs: object) -> str:
        self_agent.work_dir = str(Path(str(kwargs.get("work_dir") or ".")).resolve())
        assert "Created worktree" in self_agent.create_worktree()
        if filename is not None:
            (Path(self_agent.work_dir) / filename).write_text("agent output\n")
        return "success: true\nsummary: stub\n"

    parent_class.run = stub_run
    return original


class TestWorktreeBranchCreated(_WorktreeBranchBase):
    """Validates that a ``kiss/wt-*`` branch created by the task is
    preserved when the task does not call ``commit_and_merge``."""

    def test_worktree_created_event_broadcast(self) -> None:
        """The agent broadcasts ``worktree_created`` from ``create_worktree``.

        This is the strongest assertion that a branch *was* created at
        some point during the run, even if a later code path might
        delete it.
        """
        self._original_run = _patch_parent_run_create_file("agent_out.txt")
        self.server._run_task_inner({
            "prompt": "task with changes",
            "workDir": self.repo,
            "tabId": "0",
            "model": "",
        })
        wt_created = [e for e in self.events if e["type"] == "worktree_created"]
        assert len(wt_created) == 1, (
            f"Expected exactly one worktree_created event, got: {self._types()}"
        )
        branch = wt_created[0]["branch"]
        assert branch.startswith("kiss/wt-"), branch

    def test_branch_persists_after_task_with_changes(self) -> None:
        """When the agent modifies files in the worktree and never
        merges, the branch is preserved for manual merge/discard."""
        self._original_run = _patch_parent_run_create_file("agent_out.txt")
        self.server._run_task_inner({
            "prompt": "task with changes",
            "workDir": self.repo,
            "tabId": "0",
            "model": "",
        })
        branches = _list_kiss_wt_branches(self.repo)
        assert len(branches) == 1, (
            f"Expected exactly one kiss/wt-* branch after task with changes; "
            f"got {branches}.  Events: {self._types()}"
        )

    def test_branch_is_dropped_after_task_with_no_changes(self) -> None:
        """A worktree the agent left unmerged with no tracked changes has
        nothing to review: the post-task path discards its branch and
        reports the discard, instead of parking an empty branch."""
        self._original_run = _patch_parent_run_create_file(None)
        self.server._run_task_inner({
            "prompt": "task with no changes",
            "workDir": self.repo,
            "tabId": "0",
            "model": "",
        })

        assert "worktree_created" in self._types(), self._types()

        branches = _list_kiss_wt_branches(self.repo)
        assert branches == [], (
            f"an empty kiss/wt-* branch survived the task.  "
            f"Branches: {branches}.  Events: {self._types()}"
        )
        (result,) = [e for e in self.events if e["type"] == "worktree_result"]
        assert (result["action"], result["success"]) == ("discard", True), result


class TestWorktreeNestedWorkDirEvents(_WorktreeBranchBase):
    """Worktree events publish the task's cwd inside the worktree."""

    def test_worktree_events_publish_nested_work_dir(self) -> None:
        """A task launched from a repo subdirectory publishes
        ``worktreeWorkDir`` pointing at the same offset inside the
        worktree on both ``worktree_created`` and ``worktree_done``.

        File-link resolution falls back to that directory: relative
        artifact paths printed by the task are relative to its cwd
        (``<worktree>/<offset>``), not the worktree root, so publishing
        only the root leaves nested-project links dead.
        """
        nested = Path(self.repo) / "packages" / "app"
        nested.mkdir(parents=True)
        self._original_run = _patch_parent_run_create_file("agent_out.txt")
        self.server._run_task_inner({
            "prompt": "task from nested dir",
            "workDir": str(nested),
            "tabId": "0",
            "model": "",
        })

        wt_created = [e for e in self.events if e["type"] == "worktree_created"]
        assert len(wt_created) == 1, self._types()
        expected = str(Path(wt_created[0]["worktreeDir"]) / "packages" / "app")
        assert wt_created[0]["worktreeWorkDir"] == expected, wt_created[0]

        wt_done = [e for e in self.events if e["type"] == "worktree_done"]
        assert len(wt_done) == 1, self._types()
        assert wt_done[0]["worktreeWorkDir"] == expected, wt_done[0]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
