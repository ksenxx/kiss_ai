# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUGHUNT-3: cleanup_orphans must serialize under the per-repo lock.

``WorktreeSorcarAgent`` serializes every multi-step mutation of the
main repository (worktree create + dirty-state copy, stash → checkout
→ merge → pop, discard) under :func:`git_worktree.repo_lock` — the
RACE-2 fix in ``_try_setup_worktree`` documents this invariant.

``GitWorktreeOps.cleanup_orphans`` violated it: it snapshots
``git worktree list``, then later rmtree's every directory under
``.kiss-worktrees/`` missing from that *stale* snapshot, and force-
deletes branches — all without taking ``repo_lock``.  Run concurrently
with a task start (e.g. the user triggers cleanup while another tab
creates its worktree), it can delete a worktree directory that was
registered *after* the snapshot, destroying an active task.

The test below proves the mutual-exclusion contract with a real repo
and real threads (no mocks): while another thread holds ``repo_lock``,
``cleanup_orphans`` must not mutate the repo; once the lock is
released it must proceed and remove the orphan directory.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, repo_lock


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


class TestCleanupOrphansHonorsRepoLock:
    """cleanup_orphans must not mutate the repo while repo_lock is held."""

    def test_cleanup_blocks_while_lock_held_then_proceeds(self) -> None:
        """While another thread holds repo_lock, the orphan dir must
        survive; after release, cleanup removes it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            orphan = repo / ".kiss-worktrees" / "kiss_wt-orphan"
            orphan.mkdir(parents=True)
            (orphan / "junk.txt").write_text("leftover\n")

            done = threading.Event()
            output: dict[str, str] = {}

            def worker() -> None:
                output["summary"] = GitWorktreeOps.cleanup_orphans(repo)
                done.set()

            lock = repo_lock(repo)
            with lock:
                t = threading.Thread(target=worker, daemon=True)
                t.start()
                # Give cleanup ample time to (incorrectly) proceed.
                time.sleep(0.5)
                assert orphan.exists(), (
                    "cleanup_orphans mutated the repository while "
                    "another thread held repo_lock — worktree "
                    "operations must serialize under the per-repo lock"
                )
                assert not done.is_set(), (
                    "cleanup_orphans completed while repo_lock was held"
                )

            assert done.wait(timeout=10), "cleanup_orphans never finished"
            t.join(timeout=10)
            assert not orphan.exists(), (
                "cleanup_orphans failed to remove the orphan dir after "
                "the lock was released"
            )
            assert "kiss_wt-orphan" in output["summary"]
