# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: orphan reclaim must never touch the main tree.

Bug reproduced here (observed live on a machine provisioned by
``./sorcar-cloud``): the deployed checkout was on the branch
``kiss/wt-1785810724-29b4d878`` — a perfectly ordinary user branch that
happens to carry the agent-worktree prefix.  ``git worktree list``
reports the main working tree too, so
:meth:`GitWorktreeOps.reclaim_orphaned_worktrees` classified the *main*
checkout as an orphan agent worktree, squash-merged the branch into
itself and then called :meth:`GitWorktreeOps.remove`, whose fallback is
``shutil.rmtree`` when ``git worktree remove`` refuses.  The entire
project directory was deleted:

    worktree remove failed: fatal: '/home/ksen/kiss' is a main working
    tree; deleting directory directly

Both tests below fail before the fix (the directory is gone) and pass
after it.  Real git repositories are used throughout — no mocks.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps


def _make_repo(path: Path, branch: str) -> Path:
    """Create a real git repo whose checked-out branch is *branch*.

    Args:
        path: Directory to initialize (created if missing).
        branch: Branch name to check out.

    Returns:
        The repository path.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", branch, str(path)], capture_output=True, check=True
    )
    for key, value in (("user.email", "test@test.com"), ("user.name", "Test")):
        subprocess.run(
            ["git", "-C", str(path), "config", key, value],
            capture_output=True, check=True,
        )
    (path / "README.md").write_text("# Deployed project\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def test_reclaim_leaves_main_tree_on_worktree_named_branch() -> None:
    """A main checkout on a ``kiss/wt-*`` branch survives reclaim."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "project", "kiss/wt-1785810724-29b4d878")

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(repo)

        assert reclaimed == 0
        assert repo.exists()
        assert (repo / "README.md").read_text() == "# Deployed project\n"
        assert (repo / ".git").exists()
        assert GitWorktreeOps.current_branch(repo) == (
            "kiss/wt-1785810724-29b4d878"
        )


def test_remove_refuses_to_delete_the_repo_root() -> None:
    """``remove`` never deletes the main working tree itself."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "project", "kiss/wt-abc123")

        GitWorktreeOps.remove(repo, repo)

        assert repo.exists()
        assert (repo / "README.md").read_text() == "# Deployed project\n"


def test_reclaim_still_reclaims_a_real_orphan_worktree() -> None:
    """The guard does not stop genuine orphan worktrees from merging."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "project", "main")
        branch = "kiss/wt-orphan"
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-orphan"
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        assert GitWorktreeOps.save_original_branch(repo, branch, "main")
        (wt_dir / "work.txt").write_text("agent work\n")

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(repo)

        assert reclaimed == 1
        assert not wt_dir.exists()
        assert repo.exists()
        assert (repo / "work.txt").read_text() == "agent work\n"
