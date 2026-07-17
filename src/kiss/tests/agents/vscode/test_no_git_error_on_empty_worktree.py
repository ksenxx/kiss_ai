# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: no "Not a git repository" error on empty worktree.

Bug: when a task runs in worktree mode in a git repo and does not modify
any files, a subsequent ``worktreeAction`` (merge or discard) was
returning ``{"success": False, "message": "Not a git repository"}``.

Root cause: ``_run_task``'s finally block set ``tab.agent = None``,
destroying the in-memory worktree state.  A later ``worktreeAction``
created a fresh ``WorktreeSorcarAgent`` via ``_get_tab``'s lazy
instantiation — that agent had ``_wt = None`` and ``_repo_root = None``,
leading to the misleading error.

Fix:
- Preserve the agent when a worktree branch is pending (so merge/discard
  continues to work after the task ends).
- Replace "Not a git repository" with "No pending worktree changes to
  act on" for the lost-state scenario.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.server import VSCodeServer


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


class TestNoGitRepoErrorOnEmptyWorktree(unittest.TestCase):
    """Verifies no "Not a git repository" error after worktree task with no changes."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-err-test-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

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
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

        # Stub agent run to make no file changes.
        def stub_run(self_agent: object, **kwargs: object) -> str:
            return "success: true\nsummary: no changes\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        for tab in list(_RunningAgentState.running_agent_states.values()):
            if tab.agent is not None and tab.agent._wt_pending:
                try:
                    tab.agent.discard()
                except Exception:
                    pass
        _RunningAgentState.running_agent_states.clear()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_not_a_git_repo_error_after_empty_worktree_task(self) -> None:
        """After a worktree task with no changes, worktreeAction must not
        return "Not a git repository"."""
        tab_id = "t-empty-wt"

        # Run the full task lifecycle (including _run_task's finally
        # block that disposes the agent).
        self.server._run_task({
            "prompt": "task with no changes",
            "workDir": self.repo,
            "tabId": tab_id,
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })

        # Now trigger a worktreeAction (as the frontend might).
        self.events.clear()
        result = self.server._handle_worktree_action("merge", tab_id)

        # The error must NOT be "Not a git repository".
        assert "Not a git repository" not in result.get("message", ""), (
            f"BUG: got misleading 'Not a git repository' error: {result}"
        )

    def test_worktree_discard_after_empty_task_no_git_error(self) -> None:
        """Discard action also must not report "Not a git repository"."""
        tab_id = "t-discard-wt"

        self.server._run_task({
            "prompt": "task with no changes",
            "workDir": self.repo,
            "tabId": tab_id,
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })

        self.events.clear()
        result = self.server._handle_worktree_action("discard", tab_id)

        assert "Not a git repository" not in result.get("message", ""), (
            f"BUG: got misleading 'Not a git repository' error: {result}"
        )

    def test_worktree_action_after_changes_works(self) -> None:
        """When the agent modifies files and the task ends, merge/discard
        should succeed (agent preserved)."""
        tab_id = "t-with-changes"

        # Stub agent run to create a file.
        def stub_with_file(self_agent: object, **kwargs: object) -> str:
            work_dir = kwargs.get("work_dir")
            if isinstance(work_dir, str) and work_dir:
                (Path(work_dir) / "output.txt").write_text("agent output\n")
            return "success: true\nsummary: created file\n"

        self._parent_class.run = stub_with_file

        self.server._run_task({
            "prompt": "task with changes",
            "workDir": self.repo,
            "tabId": tab_id,
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })

        # The agent should be preserved (worktree pending with changes).
        tab = _RunningAgentState.running_agent_states.get(tab_id)
        assert tab is not None
        assert tab.agent is not None, "Agent should be preserved when worktree has changes"
        assert tab.agent._wt_pending, "Worktree should still be pending"

        # The post-task flow opened a hunk merge review for the
        # worktree changes (``is_merging`` is held until the review
        # finishes).  The frontend only offers Merge/Discard after the
        # review's ``all-done`` mergeAction, which routes to
        # ``_finish_merge`` — mirror that here.
        self.server._finish_merge(tab_id)

        # Discard should work (not error).
        result = self.server._handle_worktree_action("discard", tab_id)
        assert result["success"], f"Discard should succeed: {result}"


if __name__ == "__main__":
    unittest.main()
