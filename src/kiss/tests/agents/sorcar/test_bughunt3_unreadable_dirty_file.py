# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUGHUNT-3: an unreadable dirty file must not crash worktree setup.

``run()``'s contract promises graceful fallback to direct execution
when worktree setup fails ("Falls back to direct execution (no
worktree) when: ... Any git command fails during setup").  But
``_try_setup_worktree`` called ``GitWorktreeOps.copy_dirty_state``
unguarded: a dirty file the process cannot read (mode 000 — e.g. a
credentials placeholder or a file owned by a container user) made
``shutil.copy2`` raise ``PermissionError``, which propagated out of
``run()`` and killed the whole task — and left a half-created
worktree directory and ``kiss/wt-*`` branch behind.

These integration tests use real on-disk git repos (no mocks).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import _git
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


def _kiss_branches(repo: Path) -> list[str]:
    """All kiss/wt-* branch names in *repo*."""
    result = _git(
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads/kiss/wt-*",
        cwd=repo,
    )
    return [b for b in result.stdout.strip().splitlines() if b]


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root can read mode-000 files"
)
class TestUnreadableDirtyFile:
    """Worktree setup must degrade gracefully on unreadable dirty files."""

    def test_setup_falls_back_and_leaves_no_partial_state(self) -> None:
        """_try_setup_worktree returns None (fallback) instead of raising,
        and cleans up the partially-created worktree and branch.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            secret = repo / "unreadable.txt"
            secret.write_text("secret\n")
            secret.chmod(0o000)
            try:
                agent = WorktreeSorcarAgent("bh3-unreadable")
                result = agent._try_setup_worktree(repo, None)

                assert result is None, (
                    "expected graceful fallback (None), got a worktree "
                    f"work dir: {result}"
                )
                assert agent._wt is None
                # No half-created state may be left behind.
                assert _kiss_branches(repo) == [], (
                    "partial worktree branch left behind after the "
                    "dirty-state copy failed"
                )
                wt_root = repo / ".kiss-worktrees"
                leftovers = (
                    [p.name for p in wt_root.iterdir()]
                    if wt_root.is_dir()
                    else []
                )
                assert leftovers == [], (
                    f"partial worktree dirs left behind: {leftovers}"
                )
            finally:
                secret.chmod(0o644)
