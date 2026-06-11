# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUGHUNT-3: merge() with a deleted worktree directory must still
delete the task branch.

When the worktree directory has been removed from disk (crashed
cleanup, manual ``rm -rf``, host reboot wiping a tmpfs) but the
worktree is still REGISTERED in git's bookkeeping, ``git branch -d/-D``
refuses to delete the branch ("used by worktree at ...").

``WorktreeSorcarAgent.merge()`` skipped ``_finalize_worktree()`` (the
only step that runs ``git worktree prune``) whenever ``wt_dir`` did not
exist, so a successful squash-merge silently left the ``kiss/wt-*``
branch behind forever: ``_do_merge`` ignores the ``delete_branch``
return value on SUCCESS.

These integration tests use real on-disk git repos (no mocks).
"""

from __future__ import annotations

import shutil
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


def _create_worktree(repo: Path, branch: str) -> Path:
    """Create a registered worktree at repo/.kiss-worktrees/<slug>."""
    GitWorktreeOps.ensure_excluded(repo)
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    return wt_dir


def _is_registered_worktree(repo: Path, wt_dir: Path) -> bool:
    """True if *wt_dir* is still in ``git worktree list`` bookkeeping."""
    result = _git("worktree", "list", "--porcelain", cwd=repo)
    return str(wt_dir) in result.stdout


class TestMergeWithStaleWorktreeRegistration:
    """merge() must prune stale registrations so the branch is deletable."""

    def test_merge_deletes_branch_after_wt_dir_removed(self) -> None:
        """Squash-merge path (no baseline): branch must be gone after merge."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-bh3-stale-1"
            wt_dir = _create_worktree(repo, branch)

            (wt_dir / "work.txt").write_text("agent work\n")
            assert GitWorktreeOps.commit_all(wt_dir, "agent: add work.txt")

            # Simulate crashed cleanup: directory gone, registration kept.
            shutil.rmtree(wt_dir)
            assert _is_registered_worktree(repo, wt_dir)

            agent = WorktreeSorcarAgent("bh3-stale")
            agent._wt = GitWorktree(
                repo_root=repo,
                branch=branch,
                original_branch="main",
                wt_dir=wt_dir,
                baseline_commit=None,
            )

            msg = agent.merge()

            assert msg.startswith("Successfully merged"), msg
            assert (repo / "work.txt").read_text() == "agent work\n"
            assert agent._wt is None
            # The whole point: the task branch must not be left behind.
            assert not GitWorktreeOps.branch_exists(repo, branch), (
                "merge() succeeded but the kiss/wt-* branch was left "
                "behind because the stale worktree registration was "
                "never pruned"
            )
            assert not _is_registered_worktree(repo, wt_dir)

    def test_merge_deletes_branch_after_wt_dir_removed_with_baseline(
        self,
    ) -> None:
        """Cherry-pick-from-baseline path: branch must be gone after merge."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-bh3-stale-2"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "feature.txt").write_text("feature\n")
            assert GitWorktreeOps.commit_all(wt_dir, "agent: add feature")

            shutil.rmtree(wt_dir)
            assert _is_registered_worktree(repo, wt_dir)

            agent = WorktreeSorcarAgent("bh3-stale-baseline")
            agent._wt = GitWorktree(
                repo_root=repo,
                branch=branch,
                original_branch="main",
                wt_dir=wt_dir,
                baseline_commit=baseline,
            )

            msg = agent.merge()

            assert msg.startswith("Successfully merged"), msg
            assert (repo / "feature.txt").read_text() == "feature\n"
            assert not GitWorktreeOps.branch_exists(repo, branch), (
                "merge() (baseline path) left the kiss/wt-* branch behind"
            )
            assert not _is_registered_worktree(repo, wt_dir)
