# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUGHUNT-3: discard() must fully remove corrupted or locked worktrees.

``GitWorktreeOps.remove`` ran ``git worktree remove --force`` once and
gave up on failure.  Two real-world states defeat that:

1. Corrupted worktree — the ``.git`` link file inside the worktree
   directory was deleted (crash, overzealous cleaner, user ``rm``).
   ``git worktree remove`` fails validation and the directory stays
   on disk forever, while ``discard()`` still reports
   ``"Discarded branch ..."``.

2. Locked worktree — ``git worktree lock`` requires ``--force`` twice
   for removal.  A single ``--force`` fails, ``git worktree prune``
   refuses to prune a locked registration, so the branch deletion
   fails too and the worktree directory survives the discard.

These integration tests use real on-disk git repos (no mocks).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps, _git
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _setup_pending_agent(repo: Path, branch: str) -> tuple[WorktreeSorcarAgent, Path]:
    """Create a worktree and an agent with that worktree pending."""
    GitWorktreeOps.ensure_excluded(repo)
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    agent = WorktreeSorcarAgent("bh3-discard")
    agent._wt = GitWorktree(
        repo_root=repo,
        branch=branch,
        original_branch="main",
        wt_dir=wt_dir,
        baseline_commit=None,
    )
    return agent, wt_dir


class TestDiscardCorruptedWorktree:
    """discard() on a worktree whose .git link file was deleted."""

    def test_discard_removes_corrupt_dir_and_branch(self) -> None:
        """The directory and the branch are both gone after discard()."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-bh3-corrupt"
            agent, wt_dir = _setup_pending_agent(repo, branch)

            # Corrupt the worktree: its .git link file disappears.
            (wt_dir / ".git").unlink()

            msg = agent.discard()

            assert not wt_dir.exists(), (
                "discard() left the corrupted worktree directory on disk"
            )
            assert not GitWorktreeOps.branch_exists(repo, branch)
            assert "⚠️" not in msg, msg
            assert agent._wt is None


class TestDiscardLockedWorktree:
    """discard() on a worktree locked via ``git worktree lock``."""

    def test_discard_removes_locked_dir_and_branch(self) -> None:
        """The directory and the branch are both gone after discard()."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-bh3-locked"
            agent, wt_dir = _setup_pending_agent(repo, branch)

            lock = _git("worktree", "lock", str(wt_dir), cwd=repo)
            assert lock.returncode == 0, lock.stderr

            msg = agent.discard()

            assert not wt_dir.exists(), (
                "discard() left the locked worktree directory on disk"
            )
            assert not GitWorktreeOps.branch_exists(repo, branch), (
                "discard() left the locked worktree's branch behind"
            )
            assert "⚠️" not in msg, msg
            assert agent._wt is None
            # Registration must be gone too.
            listing = _git("worktree", "list", "--porcelain", cwd=repo).stdout
            assert str(wt_dir) not in listing
