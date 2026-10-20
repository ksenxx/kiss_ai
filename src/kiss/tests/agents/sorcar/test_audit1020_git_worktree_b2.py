# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-20, slice B2: ``git_worktree.py`` fixes.

F1  ``reattach`` whose owner stamp fails must tear the worktree down
    through ``remove`` (which escalates to ``--force --force`` and
    ``rmtree``), not one raw ``git worktree remove --force`` whose
    failure left the directory and its registration behind.  A
    ``post-checkout`` hook that locks the new worktree is the real way
    a single ``--force`` fails.
F2  ``begin_conflicted_merge`` takes the caller's ``before`` snapshot
    instead of walking the tree a second time; the failure path must
    restore the tree against that snapshot.
F4  A nested ``_reclaim_process_lock`` take re-enters without spawning
    ``git rev-parse --git-common-dir``; aliases of the repo path and a
    second checkout of the same ``--git-common-dir`` still re-enter.
F5  ``inside_kiss_worktree`` and ``kiss_worktree_root`` share one path
    scanner, so they agree on backslash-separated paths.
F8  The cross-process lock timeout names a busy peer, not a hung one.

Real git repositories, a real hook, a real ``.git/config.lock``, a real
PATH shim and a real second process; nothing is mocked.
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

from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SUBDIR,
    GitWorktreeOps,
    _git_common_dir,
    _reclaim_process_lock,
    inside_kiss_worktree,
    kiss_worktree_root,
)
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import posix_only


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "T")
    (path / "README.md").write_text("# repo\n")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "initial")
    return path


@pytest.fixture
def repo() -> Iterator[Path]:
    tmp = tempfile.mkdtemp(prefix="audit1020-b2-")
    try:
        yield _make_repo(Path(tmp) / "repo")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- F1 -----------------------------------------------------------------


def test_reattach_failed_stamp_removes_a_locked_worktree(repo: Path) -> None:
    """A failed owner stamp leaves nothing on disk even when git needs ``-f -f``."""
    branch = GitWorktreeOps.new_task_branch(repo)
    _git(repo, "branch", branch)
    wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
    hook = repo / ".git" / "hooks" / "post-checkout"
    hook.write_text('#!/bin/sh\ngit worktree lock "$PWD"\n')
    hook.chmod(0o755)
    config_lock = repo / ".git" / "config.lock"
    config_lock.write_text("")  # what a concurrent `git config` holds
    try:
        reattached = GitWorktreeOps.reattach(repo, branch, wt_dir)
    finally:
        config_lock.unlink()

    assert reattached is False
    assert not wt_dir.exists(), "the unstamped worktree was left on disk"
    assert all(b != branch for _, b in GitWorktreeOps.registered_worktrees(repo))
    assert GitWorktreeOps.branch_exists(repo, branch), "reattach must keep the branch"


# --- F2 -----------------------------------------------------------------


def test_begin_conflicted_merge_restores_against_the_callers_snapshot(repo: Path) -> None:
    """The failure path keeps files that were untracked before the call."""
    (repo / "notes.txt").write_text("mine\n")
    before = GitWorktreeOps.status_porcelain(repo)
    assert before == "?? notes.txt"
    head = _git(repo, "rev-parse", "HEAD").strip()

    assert GitWorktreeOps.begin_conflicted_merge(repo, "no-such", head, before) is None

    assert (repo / "notes.txt").read_text() == "mine\n"
    assert GitWorktreeOps.status_porcelain(repo) == before


# --- F4 -----------------------------------------------------------------


def _git_shim(log: Path) -> Path:
    """Install a ``git`` on PATH that records every argv line in *log*."""
    real_git = shutil.which("git")
    assert real_git
    shim_dir = log.parent / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(f'#!/bin/sh\necho "$@" >> {log}\nexec {real_git} "$@"\n')
    shim.chmod(0o755)
    return shim_dir


def _common_dir_spawns(log: Path) -> int:
    return log.read_text().count("rev-parse --git-common-dir") if log.exists() else 0


@posix_only("the git shim is a sh script")
def test_nested_reclaim_lock_does_not_spawn_git(repo: Path) -> None:
    """Re-entry is decided from the repo path before ``git rev-parse`` runs."""
    log = repo.parent / "git.log"
    alias = repo.parent / "alias"
    alias.symlink_to(repo)
    other = repo.parent / "other"
    _git(repo, "worktree", "add", "-q", str(other))
    path_before = os.environ["PATH"]
    os.environ["PATH"] = f"{_git_shim(log)}{os.pathsep}{path_before}"
    try:
        with _reclaim_process_lock(repo):
            spawns = _common_dir_spawns(log)
            assert spawns == 1
            with _reclaim_process_lock(repo):
                pass
            with _reclaim_process_lock(alias):
                pass
            assert _common_dir_spawns(log) == spawns, "nested take spawned git"
            # A second checkout of the same --git-common-dir cannot be
            # recognised by path; it still re-enters through the lock
            # path instead of self-deadlocking on a fresh descriptor.
            with _reclaim_process_lock(other, timeout=5.0):
                pass
            assert _common_dir_spawns(log) == spawns + 1
    finally:
        os.environ["PATH"] = path_before
    # The hold is withdrawn: a fresh take spawns again and succeeds.
    with _reclaim_process_lock(repo, timeout=5.0):
        pass


# --- F5 -----------------------------------------------------------------


def test_worktree_path_helpers_agree_on_backslash_paths() -> None:
    """One scanner: both helpers see the worktree in a Windows-style path."""
    path = "C:\\r\\.kiss-worktrees\\kiss_wt-x\\src"
    assert inside_kiss_worktree(path)
    assert kiss_worktree_root(path) == Path("C:/r/.kiss-worktrees/kiss_wt-x")
    posix = "/u/r/.kiss-worktrees/kiss_wt-x/src"
    assert kiss_worktree_root(posix) == Path("/u/r/.kiss-worktrees/kiss_wt-x")
    assert kiss_worktree_root(Path(posix)) == Path("/u/r/.kiss-worktrees/kiss_wt-x")
    assert kiss_worktree_root("/.kiss-worktrees/kiss_wt-x") == Path("/.kiss-worktrees/kiss_wt-x")
    assert kiss_worktree_root("/u/r/.kiss-worktrees/other") is None
    assert not inside_kiss_worktree("/u/r/.kiss-worktrees/other")
    assert not inside_kiss_worktree("")


# --- F8 -----------------------------------------------------------------


@posix_only("the peer holds a fcntl.flock")
def test_lock_timeout_names_a_busy_peer(repo: Path) -> None:
    """A peer holding the flock past the bound reads as busy, not hung."""
    import fcntl

    git_common = _git_common_dir(repo)
    assert git_common is not None
    lock_path = git_common / "kiss-reclaim.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
            import fcntl, sys
            f = open({str(lock_path)!r}, "a+")
            fcntl.flock(f, fcntl.LOCK_EX)
            print("held", flush=True)
            sys.stdin.readline()
        """),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        with pytest.raises(KISSError, match="still using this repository.*retry later"):
            with _reclaim_process_lock(repo, timeout=0.5):
                pass
    finally:
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.close()
        holder.wait(timeout=10)
    with open(lock_path, "a+") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)
