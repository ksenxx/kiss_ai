# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for worktree branch creation when autoCommit=False.

Bug report: "when use_worktree is set true, but autocommit is false, it
seems that a worktree branch is not getting created."

These tests drive ``VSCodeServer._run_task_inner`` with a real git repo
and a stub agent (replaces the parent class' ``run`` with a deterministic
implementation — not a mock).  Each test cleans up its temporary git
repo and persistence database in ``tearDown`` to prevent state pollution
across tests.

The user-observed behavior is that the ``kiss/wt-{chat_id}-*`` branch
appears to never exist in ``git branch --list``.  The root cause is in
:meth:`_MergeFlowMixin._present_pending_worktree`: when the worktree's
working tree has no detectable file changes after the task completes
and no non-worktree task is running on the main tree, the branch is
auto-discarded — even though the user explicitly disabled auto-commit
and would expect the branch to be preserved for manual review.

These tests:

1. ``test_branch_created_with_changes_no_autocommit`` — confirms that
   when the agent makes file changes, the branch survives the post-task
   path (sanity baseline).
2. ``test_branch_created_no_changes_no_autocommit`` — reproduces the
   bug: when the agent makes no file changes and auto-commit is off,
   the branch is incorrectly auto-discarded.  After the fix, the branch
   is preserved.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.server import VSCodeServer


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    """Create a fresh git repo with one seed commit so HEAD exists."""
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


def _list_kiss_wt_branches(repo: str) -> list[str]:
    """List all ``kiss/wt-*`` branches currently present in *repo*.

    Reflects only branches that still exist (i.e. not deleted).
    """
    result = _run_git(repo, "branch", "--list", "kiss/wt-*")
    return [
        line.strip().lstrip("* ").strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


class _WorktreeNoAutocommitBase(unittest.TestCase):
    """Shared setUp / tearDown — fresh git repo, isolated persistence DB."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-test-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        # Redirect persistence to a temp DB so this test does not
        # contaminate the user's real ``~/.kiss/sorcar.db``.
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

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict] = []

        def capture(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        # Save the stateful-agent parent's ``run`` for restoration; the
        # individual tests patch it with a deterministic stub.
        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run

        # Clean up any pending worktree from the agent before
        # destroying the repo, so leftover ``git worktree`` metadata
        # does not contaminate later tests.
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState
        for tab in list(_RunningAgentState.running_agent_states.values()):
            if tab.agent is not None and tab.agent._wt_pending:
                try:
                    tab.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        _RunningAgentState.running_agent_states.clear()

        # Restore persistence.
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

    The stub creates *filename* inside the per-task work_dir when
    *filename* is provided.  When *filename* is ``None`` it makes no
    file changes — exercising the empty-worktree code path.

    Returns the original ``run`` for restoration in tearDown.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def stub_run(self_agent: object, **kwargs: object) -> str:
        if filename is not None:
            work_dir = kwargs.get("work_dir")
            if isinstance(work_dir, str) and work_dir:
                (Path(work_dir) / filename).write_text("agent output\n")
        return "success: true\nsummary: stub\n"

    parent_class.run = stub_run
    return original


class TestWorktreeBranchCreatedNoAutocommit(_WorktreeNoAutocommitBase):
    """Validates that a ``kiss/wt-*`` branch is created and preserved
    when ``use_worktree=True`` and ``autoCommit=False``."""

    def test_worktree_created_event_broadcast(self) -> None:
        """The agent always broadcasts ``worktree_created`` regardless
        of the auto-commit setting.

        This is the strongest assertion that a branch *was* created at
        some point during the run, even if a later code path might
        delete it.
        """
        self._original_run = _patch_parent_run_create_file("agent_out.txt")
        self.server._run_task_inner({
            "prompt": "task with changes",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })
        wt_created = [e for e in self.events if e["type"] == "worktree_created"]
        assert len(wt_created) == 1, (
            f"Expected exactly one worktree_created event, got: {self._types()}"
        )
        branch = wt_created[0]["branch"]
        assert branch.startswith("kiss/wt-"), branch

    def test_branch_persists_after_task_with_changes(self) -> None:
        """When the agent modifies files in the worktree and
        ``autoCommit=False``, the branch is preserved for manual
        merge/discard."""
        self._original_run = _patch_parent_run_create_file("agent_out.txt")
        self.server._run_task_inner({
            "prompt": "task with changes",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })
        branches = _list_kiss_wt_branches(self.repo)
        assert len(branches) == 1, (
            f"Expected exactly one kiss/wt-* branch after task with changes "
            f"and autoCommit=False; got {branches}.  Events: {self._types()}"
        )

    def test_branch_persists_after_task_with_no_changes(self) -> None:
        """REPRODUCES THE USER-REPORTED BUG.

        When ``use_worktree=True`` and ``autoCommit=False``, and the
        agent makes no file changes in the worktree, the branch should
        still be preserved — the user explicitly opted out of
        auto-commit and may want to inspect / continue work on the
        branch manually.  Previously the branch was auto-discarded
        when the worktree appeared empty, leading to the user-observed
        symptom of "branch is not getting created."
        """
        self._original_run = _patch_parent_run_create_file(None)
        self.server._run_task_inner({
            "prompt": "task with no changes",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })

        # Sanity: the agent did broadcast a worktree_created event
        # during the run, so the branch was created.
        assert "worktree_created" in self._types(), self._types()

        # Bug repro: the branch should still exist after the run when
        # auto-commit is off, even if no file changes were made.
        branches = _list_kiss_wt_branches(self.repo)
        assert len(branches) == 1, (
            f"BUG: when use_worktree=True and autoCommit=False, the "
            f"kiss/wt-* branch is missing after the task.  "
            f"Branches: {branches}.  Events: {self._types()}"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
