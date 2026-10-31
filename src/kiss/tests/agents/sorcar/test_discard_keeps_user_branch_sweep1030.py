# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Discarding a task worktree leaves the user's checkout on the user's branch.

A worktree task never moves the main checkout, so the user may switch it
to another branch while the task runs.  ``discard`` must then not check
out the branch the task started from (which would also carry the user's
uncommitted edits along); it moves the checkout back only when the user
checked out the task branch itself, which would block its deletion.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.tests.agents.sorcar.test_autocommit_notifications import _setup_worktree_agent


def _git(repo: Path, *args: str) -> str:
    """Run ``git -C repo args`` and return its stdout; fail the test on error."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout


def test_discard_keeps_the_branch_the_user_switched_to(tmp_path: Path) -> None:
    agent, repo, wt_dir = _setup_worktree_agent(tmp_path, "keep-user-branch")
    (wt_dir / "task.txt").write_text("task work\n")
    _git(repo, "checkout", "-b", "feature")
    (repo / "README.md").write_text("# user edit in progress\n")

    message = agent.discard()

    assert message == "Discarded branch 'kiss/wt-test-keep-user-branch'."
    assert GitWorktreeOps.current_branch(repo) == "feature"
    assert (repo / "README.md").read_text() == "# user edit in progress\n"
    assert not wt_dir.exists()
    assert not _git(repo, "branch", "--list", "kiss/wt-test-keep-user-branch").strip()
    assert not agent._wts


def test_discard_moves_off_the_task_branch_to_delete_it(tmp_path: Path) -> None:
    """The user checked out the task branch itself after its worktree was removed."""
    agent, repo, wt_dir = _setup_worktree_agent(tmp_path, "on-task-branch")
    _git(repo, "worktree", "remove", "--force", str(wt_dir))
    _git(repo, "checkout", "kiss/wt-test-on-task-branch")

    message = agent.discard()

    assert message == "Discarded branch 'kiss/wt-test-on-task-branch'."
    assert GitWorktreeOps.current_branch(repo) == "main"
    assert not _git(repo, "branch", "--list", "kiss/wt-test-on-task-branch").strip()
