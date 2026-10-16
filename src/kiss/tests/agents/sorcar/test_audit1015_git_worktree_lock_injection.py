# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``git_worktree``'s per-repo locks survive the runner's Stop injection on a task thread.

``repo_lock(repo)`` serialises every multi-step git operation of one
repository, and ``_repo_locks_guard`` protects the registry that hands
those locks out.  Both are taken on task threads: ``SorcarAgent`` calls
:meth:`GitWorktreeOps.prune`, :meth:`GitWorktreeOps.remove`,
:meth:`GitWorktreeOps.ensure_excluded`, :meth:`GitWorktreeOps.create`,
:meth:`GitWorktreeOps.sweep_orphaned_state` and
:meth:`GitWorktreeOps.reclaim_orphaned_worktrees` while setting up,
merging or discarding a task's worktree.  The Stop watchdog delivers
``KeyboardInterrupt`` to that thread with ``PyThreadState_SetAsyncExc``;
a thread blocked in a contended ``acquire`` receives it at the check
point right after ``__enter__`` returns, outside the ``with`` block
(CPython gh-74174), so a plain ``with repo_lock(repo):`` there leaks
the lock for the daemon's life: every later worktree setup, merge and
discard of that repository hangs on it, and a leaked
``_repo_locks_guard`` hangs ``repo_lock()`` for EVERY repository.
Each site therefore goes through
:func:`~kiss.core.locked.locked_call`.

Each hammer runs in a child process (a leaked lock would wedge the
pytest process): a contender thread holds the lock half the time so
the worker's ``acquire`` really blocks, and the worker loops one real
path on a real git repository while hundreds of interrupts are
injected into it.  The test fails on the pre-fix ``with`` sites and
passes on the fixed ones.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar import git_worktree
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, repo_lock
from kiss.tests.lock_injection import hammer


def _make_repo() -> Path:
    """Create a throw-away git repository with one commit and return its root."""
    repo = Path(tempfile.mkdtemp(prefix="audit1015-wt-lock-"))
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=a",
         "commit", "-q", "--allow-empty", "-m", "init"],
        check=True,
    )
    return repo


class _Prune:
    """``GitWorktreeOps.prune`` on the worker: the shortest ``with repo_lock(repo)`` body."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def __call__(self) -> None:
        GitWorktreeOps.prune(self.repo)


class _EnsureExcluded:
    """``GitWorktreeOps.ensure_excluded``: ``repo_lock`` around the ``info/exclude`` flock."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def __call__(self) -> None:
        GitWorktreeOps.ensure_excluded(self.repo)


class _Sweep:
    """``GitWorktreeOps.sweep_orphaned_state``: ``repo_lock`` then the reclaim flock."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def __call__(self) -> None:
        GitWorktreeOps.sweep_orphaned_state(self.repo)


class _Lookup:
    """``repo_lock(repo)`` itself: the registry guard is the contended lock."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def __call__(self) -> None:
        repo_lock(self.repo)


_DRIVERS: dict[str, type[_Prune | _EnsureExcluded | _Sweep | _Lookup]] = {
    "prune": _Prune, "ensure_excluded": _EnsureExcluded, "sweep": _Sweep, "lookup": _Lookup,
}


def run_hammer(name: str) -> None:
    """Child-process body: hammer the path *name* and check the repo locks stay free."""
    repo = _make_repo()
    driver = _DRIVERS[name](repo)
    lock = repo_lock(repo)
    locks: dict[str, object] = {
        "repo_lock": lock, "_repo_locks_guard": git_worktree._repo_locks_guard,
    }
    contended = git_worktree._repo_locks_guard if name == "lookup" else lock
    # The locks are checked once the contender is gone: it holds them most of the time.
    hammer(driver, locks, contend=contended)
    # The repository still works afterwards: the next task's worktree setup gets the lock.
    GitWorktreeOps.prune(repo)
    assert repo_lock(repo) is lock


@pytest.mark.parametrize("name", sorted(_DRIVERS))
def test_git_worktree_lock_survives_injection(name: str) -> None:
    """No ``repo_lock`` site a task thread reaches in ``git_worktree`` leaks under injection."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer({name!r})"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, f"{name}: {proc.stderr[-3000:]}"


if __name__ == "__main__":
    pytest.main([__file__])
