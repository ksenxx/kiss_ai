# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the manual merge-fix command block must actually work.

Bug 1 (guidance): when an auto-merge or explicit ``merge()`` fails
with a conflict (or a pre-commit rejection), the agent prints a shell
command block the user is supposed to run verbatim.  That block ended
with ``git branch -d <branch>`` — but a squash-merge / cherry-pick
resolution never makes the task branch an ancestor of the original
branch, so ``git branch -d`` ALWAYS refuses with "the branch ... is
not fully merged" and the user is left with a dangling branch after
faithfully following the instructions.  (The agent's own automatic
path already knew this: :meth:`GitWorktreeOps.delete_branch` falls
back to ``-D``.)  These tests execute the suggested command block in
a real shell against a real repo and assert it completes, for both
the legacy squash-merge path and the baseline cherry-pick path.

Bug 2 (inconsistency): the ``_pending_review`` attribute contract
says it is cleared whenever the user explicitly merges or discards,
but only the VS Code mixin cleared it — calling the public
:meth:`WorktreeSorcarAgent.merge` / :meth:`~WorktreeSorcarAgent.discard`
API directly left the stopped-task flag stale on the agent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path) -> Path:
    """Create a real git repo on branch ``main`` with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    (path / "f.txt").write_text("original\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _extract_fix_block(message: str) -> list[str]:
    """Extract the suggested shell command block from a merge message.

    Returns the four-space-indented lines from the ``cd `` line
    through the ``git branch`` line, stripped of indentation.
    """
    lines: list[str] = []
    collecting = False
    for raw in message.splitlines():
        if not raw.startswith("    "):
            continue
        line = raw[4:]
        if line.startswith("cd "):
            collecting = True
        if collecting:
            lines.append(line)
            if line.startswith("git branch"):
                break
    assert lines, f"no command block found in message:\n{message}"
    assert lines[-1].startswith("git branch"), message
    return lines


def _run_fix_block(lines: list[str], resolution: str) -> None:
    """Execute the suggested command block in a real shell.

    Comment lines like ``# resolve conflicts ...`` are replaced by
    *resolution* (a shell command that writes the resolved file), the
    way a user would resolve the conflict in an editor.  ``git
    commit`` runs with ``GIT_EDITOR=true`` so the message prepared by
    git (SQUASH_MSG / cherry-pick template) is accepted as-is.
    """
    script_lines: list[str] = []
    resolved = False
    for line in lines:
        if line.lstrip().startswith("#"):
            if not resolved:
                script_lines.append(resolution)
                resolved = True
            continue
        script_lines.append(line)
    env = {**os.environ, "GIT_EDITOR": "true"}
    subprocess.run(
        ["bash", "-c", "\n".join(script_lines)],
        capture_output=True,
        env=env,
        check=False,
    )


class TestManualFixStepsExecutable:
    """The printed manual-resolution steps must succeed when followed."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_squash_conflict_fix_steps_delete_the_branch(self) -> None:
        """Legacy (no-baseline) conflict: following the printed steps
        verbatim must merge the work AND delete the task branch."""
        agent = WorktreeSorcarAgent("test")
        wt_work = agent._try_setup_worktree(self.repo, None)
        assert wt_work is not None
        wt = agent._wt
        assert wt is not None
        assert wt.baseline_commit is None
        branch = wt.branch

        # Agent commits a change in the worktree; main independently
        # commits a conflicting change to the same file.
        (wt.wt_dir / "f.txt").write_text("agent edit\n")
        assert GitWorktreeOps.commit_all(wt.wt_dir, "agent work")
        (self.repo / "f.txt").write_text("main edit\n")
        _git("commit", "-am", "main work", cwd=self.repo)

        msg = agent.merge()
        assert "Merge conflict detected" in msg
        block = _extract_fix_block(msg)

        _run_fix_block(block, "echo resolved > f.txt")

        # Following the instructions must leave the merge committed
        # and the task branch deleted.
        assert (self.repo / "f.txt").read_text() == "resolved\n"
        assert GitWorktreeOps.head_sha(self.repo) is not None
        assert not GitWorktreeOps.has_uncommitted_changes(self.repo)
        assert not GitWorktreeOps.branch_exists(self.repo, branch), (
            "the suggested command block failed to delete the branch "
            f"(git branch -d refuses after a squash merge):\n{msg}"
        )

    def test_baseline_cherry_pick_conflict_fix_steps_delete_the_branch(
        self,
    ) -> None:
        """Baseline (cherry-pick) conflict: following the printed steps
        verbatim must apply the work AND delete the task branch."""
        # Dirty main state at setup so a baseline commit is created.
        (self.repo / "f.txt").write_text("user dirty edit\n")
        agent = WorktreeSorcarAgent("test")
        wt_work = agent._try_setup_worktree(self.repo, None)
        assert wt_work is not None
        wt = agent._wt
        assert wt is not None
        assert wt.baseline_commit is not None
        branch = wt.branch

        # Agent commits a change in the worktree; main COMMITS an
        # independent conflicting change (HEAD advances past
        # baseline^, so no -X theirs and the conflict is real).
        (wt.wt_dir / "f.txt").write_text("agent edit\n")
        assert GitWorktreeOps.commit_all(wt.wt_dir, "agent work")
        (self.repo / "f.txt").write_text("main committed edit\n")
        _git("commit", "-am", "main work", cwd=self.repo)

        msg = agent.merge()
        assert "Merge conflict detected" in msg
        assert f"git cherry-pick --no-commit {wt.baseline_commit}" in msg
        block = _extract_fix_block(msg)

        _run_fix_block(block, "echo resolved > f.txt")

        assert (self.repo / "f.txt").read_text() == "resolved\n"
        assert not GitWorktreeOps.has_uncommitted_changes(self.repo)
        assert not GitWorktreeOps.branch_exists(self.repo, branch), (
            "the suggested command block failed to delete the branch "
            f"(git branch -d refuses after a cherry-pick):\n{msg}"
        )


class TestPendingReviewClearedByExplicitAction:
    """merge()/discard() must clear the stopped-task review flag."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_discard_clears_pending_review(self) -> None:
        agent = WorktreeSorcarAgent("test")
        assert agent._try_setup_worktree(self.repo, None) is not None
        agent._pending_review = True  # as task_runner sets on user-Stop

        result = agent.discard()

        assert "Discarded branch" in result
        assert agent._pending_review is False
        # Teardown's preserve-for-review path must now be a no-op.
        assert agent._preserve_pending_worktree_for_review() is False

    def test_merge_clears_pending_review(self) -> None:
        agent = WorktreeSorcarAgent("test")
        assert agent._try_setup_worktree(self.repo, None) is not None
        wt = agent._wt
        assert wt is not None
        (wt.wt_dir / "g.txt").write_text("new file\n")
        assert GitWorktreeOps.commit_all(wt.wt_dir, "agent work")
        agent._pending_review = True  # as task_runner sets on user-Stop

        result = agent.merge()

        assert "Successfully merged" in result
        assert agent._pending_review is False
        assert agent._preserve_pending_worktree_for_review() is False
