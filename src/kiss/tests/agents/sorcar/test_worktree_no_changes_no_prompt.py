# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: no merge/discard prompt when the worktree is empty.

Bug report: "in worktree mode, why are you asking for Auto-commit and
merge when nothing has been changed in the worktree?"

When ``use_worktree=True`` and ``autoCommit=False`` and the agent makes
no file changes in the worktree, the previous code path broadcast a
``worktree_done`` event whose frontend handler unconditionally renders
the "Auto-commit and merge or Discard?" bar.  There is nothing to merge
in that scenario, so the prompt is meaningless noise.

These tests reuse the shared base class from
``test_worktree_no_autocommit_branch`` (fresh git repo + isolated
persistence DB per test) and verify that:

1. ``worktree_done`` is NOT broadcast when the worktree has no changes
   (auto-commit OFF, branch preserved).
2. ``worktree_done`` IS broadcast (with the merge/discard prompt) when
   the worktree has real changes — the existing behavior we don't want
   to regress.
"""

from __future__ import annotations

from kiss.tests.agents.sorcar.test_worktree_no_autocommit_branch import (
    _list_kiss_wt_branches,
    _patch_parent_run_create_file,
    _WorktreeNoAutocommitBase,
)


class TestNoWorktreePromptWhenEmpty(_WorktreeNoAutocommitBase):
    """Worktree post-task UX in ``autoCommit=False`` mode."""

    def test_no_worktree_done_when_no_changes(self) -> None:
        """Bug repro: the merge/discard prompt must not appear when the
        worktree has no changes."""
        self._original_run = _patch_parent_run_create_file(None)
        self.server._run_task_inner({
            "prompt": "task with no changes",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })
        # The branch is still preserved (see test_worktree_no_autocommit_branch).
        branches = _list_kiss_wt_branches(self.repo)
        assert len(branches) == 1, (
            f"Expected the worktree branch to remain preserved; got {branches}"
        )
        # But the frontend prompt must be silent — no worktree_done event.
        types = self._types()
        assert "worktree_done" not in types, (
            f"BUG: worktree_done was broadcast even though the worktree "
            f"is empty. Events: {types}"
        )

    def test_merge_review_starts_when_changes_exist(self) -> None:
        """Sanity / non-regression: when the agent actually modified
        files in the worktree, the interactive merge review starts
        (which itself leads to the merge/discard prompt once the user
        finishes reviewing each hunk)."""
        self._original_run = _patch_parent_run_create_file("agent_out.txt")
        self.server._run_task_inner({
            "prompt": "task with changes",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })
        types = self._types()
        assert "merge_started" in types, (
            f"Expected merge_started event when changes exist; got {types}"
        )


if __name__ == "__main__":  # pragma: no cover
    import unittest
    unittest.main()
