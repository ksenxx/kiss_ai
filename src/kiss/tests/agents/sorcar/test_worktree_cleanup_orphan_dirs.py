"""Integration tests: ``cleanup_orphans`` must remove orphan directories
under ``.kiss-worktrees/``.

BUG: ``GitWorktreeOps.cleanup_orphans`` only inspects ``kiss/wt-*``
branches.  Directories under ``.kiss-worktrees/`` that are not registered
as git worktrees (and have no matching branch) are never cleaned up.

Real-world reproduction (observed in the user's repo before this fix):
``.kiss-worktrees/kiss_wt-bb76a8b0-...`` existed as a plain directory
containing leftover user files, no ``.git`` link, no ``git worktree``
record, no matching branch.  Running ``--cleanup`` did not remove it.

FIX: after the existing orphan-branch scan, ``cleanup_orphans`` also
iterates ``.kiss-worktrees/`` and removes any subdirectory that is not
registered as a git worktree.  A matching ``kiss/wt-*`` branch without
``kiss-original`` (true orphan, per BUG-58 semantics) is also deleted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps


def _make_repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=repo, capture_output=True, check=True,
    )
    (repo / "init.txt").write_text("init\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo, capture_output=True, check=True,
    )
    return repo


def _registered_worktrees(repo: Path) -> set[str]:
    res = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    out: set[str] = set()
    for line in res.stdout.splitlines():
        if line.startswith("worktree "):
            out.add(line.split(" ", 1)[1])
    return out


class TestOrphanDirectoryRemoval:
    """Directories under ``.kiss-worktrees/`` with no git worktree
    registration and no matching branch must be removed."""

    def test_plain_orphan_dir_removed(self, tmp_path: Path) -> None:
        """A plain directory under ``.kiss-worktrees/`` (never a real
        worktree, just leftover files) is removed by ``cleanup_orphans``."""
        repo = _make_repo(tmp_path)
        orphan = repo / ".kiss-worktrees" / "kiss_wt-orphan-123-456"
        (orphan / "src" / "kiss").mkdir(parents=True)
        (orphan / "src" / "kiss" / "stale.py").write_text("# stale\n")
        assert orphan.exists()

        result = GitWorktreeOps.cleanup_orphans(repo)

        assert not orphan.exists(), (
            "Orphan directory under .kiss-worktrees/ must be removed"
        )
        assert "kiss_wt-orphan-123-456" in result

    def test_active_worktree_dir_kept(self, tmp_path: Path) -> None:
        """A directory registered as an active git worktree is NOT removed."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-active-1"
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-active-1"
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        assert wt_dir.exists()
        GitWorktreeOps.cleanup_orphans(repo)
        assert wt_dir.exists(), "Active worktree dir must not be removed"
        assert str(wt_dir.resolve()) in {
            str(Path(p).resolve()) for p in _registered_worktrees(repo)
        }

        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.delete_branch(repo, branch)

    def test_orphan_dir_with_matching_orphan_branch_both_removed(
        self, tmp_path: Path,
    ) -> None:
        """A directory whose matching ``kiss/wt-*`` branch exists but has
        no git worktree registration and no ``kiss-original`` config is
        fully cleaned: branch deleted AND directory removed."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-stalepair-7"
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-stalepair-7"

        assert GitWorktreeOps.create(repo, branch, wt_dir)
        (wt_dir / ".git").unlink()
        GitWorktreeOps.prune(repo)

        assert wt_dir.exists()
        assert GitWorktreeOps.branch_exists(repo, branch)
        assert str(wt_dir.resolve()) not in {
            str(Path(p).resolve()) for p in _registered_worktrees(repo)
        }

        GitWorktreeOps.cleanup_orphans(repo)

        assert not wt_dir.exists(), "Stale worktree directory must be removed"
        assert not GitWorktreeOps.branch_exists(repo, branch), (
            "True orphan branch must be deleted"
        )

    def test_pending_merge_dir_absent_branch_kept(self, tmp_path: Path) -> None:
        """BUG-58 invariant: a branch with ``kiss-original`` whose
        worktree directory is already gone is kept (pending merge).
        Cleanup must not resurrect or delete anything."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-pending-9"
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-pending-9"
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        assert not wt_dir.exists()
        assert GitWorktreeOps.branch_exists(repo, branch)

        GitWorktreeOps.cleanup_orphans(repo)

        assert GitWorktreeOps.branch_exists(repo, branch), (
            "BUG-58: pending-merge branch must be kept"
        )

    def test_no_worktrees_dir_is_noop(self, tmp_path: Path) -> None:
        """If ``.kiss-worktrees/`` does not exist, cleanup is a no-op."""
        repo = _make_repo(tmp_path)
        assert not (repo / ".kiss-worktrees").exists()
        result = GitWorktreeOps.cleanup_orphans(repo)
        assert "orphan" in result.lower() or "No orphans" in result
