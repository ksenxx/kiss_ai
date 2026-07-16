# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUGHUNT-4: a failed ``git stash push`` must abort the merge.

``GitWorktreeOps.stash_if_dirty`` returns ``False`` both when the main
tree is clean AND when ``git stash push`` itself fails (e.g. an
untracked file the process cannot read makes git error out with
``Cannot save the untracked files``).  ``_do_merge`` treated the two
cases identically and proceeded to ``git merge --squash`` on a still
dirty main tree, which:

1. silently committed the USER's staged changes into the agent's
   squash-merge commit (``git commit`` commits the whole index), and
2. on merge failure ran ``git reset --hard HEAD``, permanently
   DESTROYING the user's staged and unstaged changes.

These integration tests use real on-disk git repos (no mocks).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path) -> Path:
    """Create a git repo on branch ``main`` with two committed files."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    for key, val in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(
            ["git", "-C", str(path), "config", key, val],
            capture_output=True,
            check=True,
        )
    (path / "f.txt").write_text("f0\n")
    (path / "g.txt").write_text("g0\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _setup_pending_worktree(
    repo: Path, agent: WorktreeSorcarAgent, agent_file: str
) -> None:
    """Create a worktree for *agent* and commit one agent change in it."""
    wt_work = agent._try_setup_worktree(repo, None)
    assert wt_work is not None
    assert agent._wt is not None
    (agent._wt.wt_dir / agent_file).write_text("agent change\n")
    _git("add", "-A", cwd=agent._wt.wt_dir)
    result = _git("commit", "-m", "agent work", cwd=agent._wt.wt_dir)
    assert result.returncode == 0, result.stderr


def _dirty_main_with_unstashable_state(repo: Path) -> Path:
    """Stage a user edit to f.txt and add an unreadable untracked file.

    The mode-000 untracked file makes ``git stash push
    --include-untracked`` fail, while the staged edit to ``f.txt`` is
    the user state that must survive the merge attempt.
    """
    (repo / "f.txt").write_text("f0\nuser staged work\n")
    _git("add", "f.txt", cwd=repo)
    unreadable = repo / "unreadable.txt"
    unreadable.write_text("secret\n")
    unreadable.chmod(0o000)
    return unreadable


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
class TestStashFailureAbortsMerge:
    """A dirty main tree that cannot be stashed must abort the merge."""

    def test_merge_aborts_and_preserves_user_staged_changes(self) -> None:
        """merge() must not commit or destroy the user's staged work."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            agent = WorktreeSorcarAgent("bh4-stash-fail")
            _setup_pending_worktree(repo, agent, "g.txt")
            branch = agent._wt.branch  # type: ignore[union-attr]
            head_before = GitWorktreeOps.head_sha(repo)

            unreadable = _dirty_main_with_unstashable_state(repo)
            try:
                msg = agent.merge()

                # The merge must have been aborted: main HEAD unchanged.
                assert GitWorktreeOps.head_sha(repo) == head_before, (
                    "merge committed on a dirty main tree after the "
                    f"stash failed; merge() said: {msg}"
                )
                # The user's staged edit must still be staged.
                cached = _git("diff", "--name-only", "--cached", cwd=repo)
                assert "f.txt" in cached.stdout.splitlines(), (
                    "user's staged change to f.txt was lost; "
                    f"merge() said: {msg}"
                )
                show = _git("show", ":f.txt", cwd=repo)
                assert "user staged work" in show.stdout
                # The task branch must be kept for a later retry.
                assert GitWorktreeOps.branch_exists(repo, branch)
                assert agent._wt is not None, (
                    "worktree state cleared even though the merge "
                    "was not performed"
                )
                # The message must explain the stash failure.
                assert "stash" in msg.lower()
            finally:
                unreadable.chmod(0o644)

    def test_merge_conflict_path_does_not_destroy_user_changes(self) -> None:
        """Agent and user edited the same file: reset --hard after the
        failed squash merge must never wipe the user's unstashed work."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            agent = WorktreeSorcarAgent("bh4-stash-fail-overlap")
            # Agent edits f.txt — the SAME file the user has staged.
            _setup_pending_worktree(repo, agent, "f.txt")

            unreadable = _dirty_main_with_unstashable_state(repo)
            try:
                msg = agent.merge()

                # The user's staged edit must still exist (not nuked by
                # the conflict-path ``git reset --hard HEAD``).
                show = _git("show", ":f.txt", cwd=repo)
                assert "user staged work" in show.stdout, (
                    "user's staged change to f.txt was DESTROYED by "
                    f"the aborted merge; merge() said: {msg}"
                )
            finally:
                unreadable.chmod(0o644)

    def test_release_worktree_aborts_and_warns(self) -> None:
        """_release_worktree must keep the branch and set a warning
        mentioning the stash failure instead of merging a dirty tree."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            agent = WorktreeSorcarAgent("bh4-stash-fail-release")
            _setup_pending_worktree(repo, agent, "g.txt")
            branch = agent._wt.branch  # type: ignore[union-attr]
            head_before = GitWorktreeOps.head_sha(repo)

            unreadable = _dirty_main_with_unstashable_state(repo)
            try:
                released = agent._release_worktree()

                assert released is None
                assert GitWorktreeOps.head_sha(repo) == head_before, (
                    "auto-merge committed on a dirty main tree after "
                    "the stash failed"
                )
                cached = _git("diff", "--name-only", "--cached", cwd=repo)
                assert "f.txt" in cached.stdout.splitlines()
                assert GitWorktreeOps.branch_exists(repo, branch), (
                    "task branch deleted even though the auto-merge "
                    "was aborted"
                )
                warning = agent._merge_conflict_warning or ""
                assert "stash" in warning.lower(), (
                    f"expected a stash-failure warning, got: {warning!r}"
                )
            finally:
                unreadable.chmod(0o644)
