# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fix #1: ``GitWorktreeOps.create`` must fail closed on owner stamping.

``create`` registers the worktree with ``git worktree add`` and stamps
this process's pid on the branch (``branch.<b>.kiss-owner-pid``) under
the cross-process reclaim lock.  The stamp is the ONLY thing that stops
a second Sorcar process's ``reclaim_orphaned_worktrees`` from treating
the brand-new worktree as a legacy orphan and deleting it under the
live owner.  Before the fix ``create`` ignored ``save_owner_pid``'s
result and returned ``True`` anyway, so a transient ``git config``
failure (``.git/config.lock`` held by another git process) handed the
caller an unprotected worktree.

Real git repositories, a real ``.git/config.lock``, and a real second
Python process running the reclaim; nothing is mocked.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import _WORKTREE_SUBDIR, GitWorktreeOps


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "-C", str(path), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def _peer_reclaim(repo: Path) -> str:
    """Run ``reclaim_orphaned_worktrees`` in a second process; return its count."""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            from pathlib import Path
            from kiss.agents.sorcar.git_worktree import GitWorktreeOps
            print(GitWorktreeOps.reclaim_orphaned_worktrees(Path({str(repo)!r})))
        """)],
        capture_output=True, text=True, check=True, cwd=str(repo), timeout=120,
    )
    return proc.stdout.strip().splitlines()[-1]


@pytest.fixture
def repo() -> Iterator[Path]:
    tmp = tempfile.mkdtemp(prefix="audit0902-fix-create-")
    try:
        yield _make_repo(Path(tmp) / "repo")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_create_fails_closed_when_owner_stamp_cannot_be_written(repo: Path) -> None:
    """A held ``.git/config.lock`` makes ``create`` return False and leave nothing."""
    branch = worktree_pool.new_task_branch(repo)
    wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
    config_lock = repo / ".git" / "config.lock"
    config_lock.write_text("")  # what a concurrent `git config` holds
    try:
        created = GitWorktreeOps.create(repo, branch, wt_dir)
    finally:
        config_lock.unlink()

    assert created is False, "create() reported success without an owner stamp"
    assert GitWorktreeOps.load_owner_pid(repo, branch) is None
    assert not wt_dir.exists(), "the unprotected worktree was left on disk"
    assert not GitWorktreeOps.branch_exists(repo, branch)
    assert all(b != branch for _, b in GitWorktreeOps.registered_worktrees(repo))


def test_peer_reclaim_finds_nothing_after_failed_owner_stamp(repo: Path) -> None:
    """The peer's reclaim must have no owner-less live worktree to destroy."""
    branch = worktree_pool.new_task_branch(repo)
    wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
    config_lock = repo / ".git" / "config.lock"
    config_lock.write_text("")
    try:
        created = GitWorktreeOps.create(repo, branch, wt_dir)
    finally:
        config_lock.unlink()
    assert created is False

    # Before the fix this printed "1": the peer squash-merged (a no-op)
    # and deleted the worktree while its creator was still alive.
    assert _peer_reclaim(repo) == "0"
    assert not wt_dir.exists()
    assert not GitWorktreeOps.branch_exists(repo, branch)


def test_create_succeeds_and_stamps_owner_when_config_is_writable(repo: Path) -> None:
    """The happy path is unchanged: one stamp, owned by this process."""
    branch = worktree_pool.new_task_branch(repo)
    wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir) is True
    assert wt_dir.is_dir()
    assert GitWorktreeOps.load_owner_pid(repo, branch) == os.getpid()
    assert _peer_reclaim(repo) == "0"
    assert wt_dir.is_dir()
