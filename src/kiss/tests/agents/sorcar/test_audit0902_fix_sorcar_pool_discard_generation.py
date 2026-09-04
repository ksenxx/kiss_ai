# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fix #7: ``worktree_pool.discard_all`` must not be followed by a
late spare publication.

``discard_all`` snapshots and clears ``_refill_threads``, joins that
snapshot, then sweeps ``_spares``.  A refill scheduled AFTER the
snapshot (``prewarm_async`` on another repo while the sweep is still
removing the first repo's spare) was not joined and published its spare
after ``discard_all`` had returned — leaving a ``kiss/wt-*`` worktree
the caller believed gone.  A snapshotted thread that outlived the
bounded join had the same effect.

Fix under test: a pool generation counter.  ``discard_all`` advances it
when it starts and again when it finishes; a refill captures the
generation when it starts and, immediately before publishing, drops
and removes its spare unless the generation is unchanged.

The interleaving is made deterministic with the real per-repo locks:
the test holds ``repo_lock(A)`` so ``discard_all`` blocks inside its
sweep of A's spare (after its thread snapshot), schedules a refill for
B that blocks on ``repo_lock(B)``, then releases A (discard returns)
and finally B (the refill reaches its publication point).
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import repo_lock
from kiss.tests.agents.sorcar.test_worktree_pool import (
    _enable_pool,
    _make_repo,
    _restore_pool_env,
)


def _wt_branches(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", "kiss/wt-*"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _wait_until(predicate, timeout: float = 30.0) -> None:  # type: ignore[no-untyped-def]
    """Poll *predicate* (a pool STATE, not a timing guess) until it holds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "pool never reached the expected state"
        time.sleep(0.005)


@pytest.fixture
def repos() -> Iterator[tuple[Path, Path]]:
    prev = _enable_pool()
    tmp = tempfile.TemporaryDirectory()
    try:
        yield (
            _make_repo(Path(tmp.name) / "repo_a"),
            _make_repo(Path(tmp.name) / "repo_b"),
        )
    finally:
        worktree_pool.discard_all()
        tmp.cleanup()
        _restore_pool_env(prev)


def test_refill_scheduled_during_discard_never_publishes(
    repos: tuple[Path, Path],
) -> None:
    repo_a, repo_b = repos
    assert worktree_pool.prewarm(repo_a)
    assert worktree_pool.spare_branches() != set()

    lock_a = repo_lock(repo_a)
    lock_b = repo_lock(repo_b)
    lock_a.acquire()
    lock_b.acquire()
    discard = threading.Thread(target=worktree_pool.discard_all, daemon=True)
    refill: threading.Thread | None = None
    try:
        discard.start()
        # discard_all has taken its thread snapshot and cleared the
        # spare map once _spares is empty; it is now blocked on
        # repo_lock(A) while removing A's spare.
        _wait_until(lambda: worktree_pool.spare_branches() == set())
        assert discard.is_alive()

        refill = worktree_pool.prewarm_async(repo_b)
        assert refill is not None
        # The refill runs until it needs repo_lock(B), which we hold.
        _wait_until(lambda: worktree_pool._repo_key(repo_b) in worktree_pool._prewarming)
    finally:
        lock_a.release()
    discard.join(timeout=120)
    assert not discard.is_alive()
    assert worktree_pool.spare_branches() == set()

    # discard_all has RETURNED; only now does the late refill proceed.
    lock_b.release()
    assert refill is not None
    refill.join(timeout=120)
    assert not refill.is_alive()

    assert worktree_pool.spare_branches() == set(), (
        "a refill scheduled during discard_all published a spare after it returned"
    )
    assert _wt_branches(repo_b) == "", "the late spare's worktree was left behind"
    assert _wt_branches(repo_a) == ""


def test_refill_publishing_while_discard_sweeps_is_dropped(
    repos: tuple[Path, Path],
) -> None:
    """Review-2 #2: a refill that reaches its publication point WHILE
    ``discard_all`` is still sweeping (after the spare snapshot, before
    the final generation bump) must not publish.

    Interleaving (B released before A): ``discard_all`` blocks on
    ``repo_lock(A)`` inside its sweep; a refill for B, scheduled after
    the discard started, captures the already-advanced generation and is
    released first, so its generation check alone still matches; only
    then is A released and ``discard_all`` returns.
    """
    repo_a, repo_b = repos
    assert worktree_pool.prewarm(repo_a)
    assert worktree_pool.spare_branches() != set()

    lock_a = repo_lock(repo_a)
    lock_b = repo_lock(repo_b)
    lock_a.acquire()
    lock_b.acquire()
    discard = threading.Thread(target=worktree_pool.discard_all, daemon=True)
    refill: threading.Thread | None = None
    try:
        discard.start()
        # discard_all has snapshotted and cleared _spares and is now
        # blocked on repo_lock(A) while removing A's spare.
        _wait_until(lambda: worktree_pool.spare_branches() == set())
        assert discard.is_alive()

        refill = worktree_pool.prewarm_async(repo_b)
        assert refill is not None
        _wait_until(lambda: worktree_pool._repo_key(repo_b) in worktree_pool._prewarming)

        # Release B FIRST: the refill runs to its publication point while
        # discard_all is still sweeping A.
        lock_b.release()
        refill.join(timeout=120)
        assert not refill.is_alive()
        assert discard.is_alive(), "discard_all must still be blocked on repo A"
        published_during_sweep = worktree_pool.spare_branches()
    finally:
        lock_a.release()
    discard.join(timeout=120)
    assert not discard.is_alive()

    assert published_during_sweep == set(), (
        "a refill published a spare while discard_all was sweeping: "
        f"{published_during_sweep}"
    )
    assert worktree_pool.spare_branches() == set(), (
        "discard_all returned with a spare the sweep never saw"
    )
    assert _wt_branches(repo_b) == "", "the mid-sweep spare's worktree was left behind"
    assert _wt_branches(repo_a) == ""
    # The pool is fully usable afterwards.
    assert worktree_pool.prewarm(repo_b)
    assert len(worktree_pool.spare_branches()) == 1


def test_refill_started_before_discard_does_not_publish_after_it(
    repos: tuple[Path, Path],
) -> None:
    """A refill that outlives discard_all (here: parked on the repo lock
    instead of timing out the bounded join) is dropped, not published."""
    repo_a, _repo_b = repos
    lock_a = repo_lock(repo_a)
    lock_a.acquire()
    try:
        refill = worktree_pool.prewarm_async(repo_a)
        assert refill is not None
        _wait_until(lambda: worktree_pool._repo_key(repo_a) in worktree_pool._prewarming)
        # Bump the generation the way discard_all does, without joining
        # the parked refill (the equivalent of the join timing out).
        with worktree_pool._pool_lock:
            worktree_pool._generation += 1
    finally:
        lock_a.release()
    refill.join(timeout=120)
    assert not refill.is_alive()
    assert worktree_pool.spare_branches() == set()
    assert _wt_branches(repo_a) == ""
    # The pool is fully usable afterwards.
    assert worktree_pool.prewarm(repo_a)
    assert len(worktree_pool.spare_branches()) == 1


def test_discard_all_then_prewarm_publishes_normally(
    repos: tuple[Path, Path],
) -> None:
    """Sequential use is unchanged: a refill after discard_all publishes."""
    repo_a, repo_b = repos
    assert worktree_pool.prewarm(repo_a)
    worktree_pool.discard_all()
    assert worktree_pool.spare_branches() == set()
    thread = worktree_pool.prewarm_async(repo_b)
    assert thread is not None
    thread.join(timeout=120)
    assert len(worktree_pool.spare_branches()) == 1
    assert _wt_branches(repo_b) != ""
