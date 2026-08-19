# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 9: Tests verifying fixes for bugs, inconsistencies, and
redundancies in both non-worktree and worktree workflows.

BUG-39: `is_running_non_wt` flag is now cleared at the very start of
    the finally block's try (before any risky calls) AND in the outer
    except handler, so it can never get permanently stuck.

BUG-40 / INC-4: `_do_merge` now returns `(MergeResult.CHECKOUT_FAILED, "")`
    instead of `(None, checkout_error_str)`.  `_release_worktree` checks
    `result == MergeResult.CHECKOUT_FAILED` instead of `result is None`,
    so the checkout error is never misattributed to `_stash_pop_warning`.

BUG-41 / RED-6 (obsolete): covered `_start_merge_session`, removed
    together with the interactive diff/merge review workflow.

BUG-42 / INC-5: Auto-discard of an empty pending worktree is safe and
    runs even while a non-worktree task is active (it touches neither
    the main tree's files nor its HEAD).

BUG-43: Manual merge instructions now use `git cherry-pick --no-commit
    baseline..branch` when a baseline commit exists, matching what the
    auto-merge actually does.

BUG-44: `_new_chat` guard now checks `tab.agent._wt_pending` regardless
    of `tab.use_worktree`, so a tab that switched modes still gets the
    non-wt-running guard.

INC-6: `_check_merge_conflict` now checks both unstaged AND staged
    files for dirty-file overlap (historically via the since-removed
    `unstaged_files()`/`staged_files()` helpers).

RED-5: The two consecutive `if not tab.use_worktree:` blocks in
    `_run_task_inner`'s finally are now a single block.
"""

from __future__ import annotations

import subprocess

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
)
from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
)
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.agents.sorcar.test_worktree_audit9 import (  # noqa: F401
    _make_repo,
)


class TestInc6Fix:
    """INC-6 FIX: _check_merge_conflict checks both unstaged and staged files."""


    def test_staged_overlap_detected(self, tmp_path):
        """A staged file overlapping with worktree changes IS detected."""
        repo = _make_repo(tmp_path)

        branch = "kiss/wt-inc6-test"
        wt_dir = repo / ".kiss-worktrees" / "wt-inc6"
        GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        (wt_dir / "init.txt").write_text("agent changes")
        subprocess.run(["git", "add", "-A"], cwd=wt_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "agent work"],
            cwd=wt_dir, capture_output=True,
        )

        (repo / "init.txt").write_text("user staged change")
        subprocess.run(["git", "add", "init.txt"], cwd=repo, capture_output=True)

        server = VSCodeServer()
        server.work_dir = str(repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )
        st = agent_state.AgentState(
            "task-inc6",
            agent=agent,
            tab_id="inc6-tab",
            server_owned=True,
        )
        st.use_worktree = True
        agent_state.register(st)
        try:
            has_conflict = server._check_merge_conflict("inc6-tab")
            assert has_conflict, (
                "INC-6 fix: staged file overlap must be detected"
            )
        finally:
            agent_state.unregister("task-inc6", st)

        subprocess.run(["git", "reset", "HEAD", "init.txt"], cwd=repo, capture_output=True)
        subprocess.run(["git", "checkout", "--", "init.txt"], cwd=repo, capture_output=True)
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)
