# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 9: Tests verifying fixes for bugs, inconsistencies, and
redundancies in both non-worktree and worktree workflows.

BUG-39 / BUG-40 / BUG-44 / RED-5 (obsolete): covered `is_running_non_wt`,
    `_release_worktree` and `tab.use_worktree`, all removed with the
    `create_worktree` / `commit_and_merge` tools.
BUG-41 / RED-6 (obsolete): covered `_start_merge_session`, removed
    together with the interactive diff/merge review workflow.
BUG-42 / INC-5: Auto-discard of an empty pending worktree is safe and
    runs even while a task is active (it touches neither the main
    tree's files nor its HEAD).
BUG-43: Manual merge instructions now use `git cherry-pick --no-commit
    baseline..branch` when a baseline commit exists, matching what the
    merge actually does.
INC-6: `_conflict_of` now checks both unstaged AND staged
    files for dirty-file overlap (historically via the since-removed
    `unstaged_files()`/`staged_files()` helpers).
"""

from __future__ import annotations

import subprocess

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
)
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.agents.sorcar.test_worktree_audit9 import (  # noqa: F401
    _make_repo,
)
from kiss.tests.conftest import seed_worktree
from kiss.tests.server.worktree_helpers import merge_conflict


class TestInc6Fix:
    """INC-6 FIX: _conflict_of checks both unstaged and staged files."""


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

        server = AgentServer()
        server.work_dir = str(repo)
        agent = ChatSorcarAgent("Sorcar")
        seed_worktree(agent, GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        ))
        st = agent_state.AgentState(
            "task-inc6",
            agent=agent,
            tab_id="inc6-tab",
            server_owned=True,
        )
        agent_state.register(st)
        try:
            has_conflict = merge_conflict(server, "inc6-tab")
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
