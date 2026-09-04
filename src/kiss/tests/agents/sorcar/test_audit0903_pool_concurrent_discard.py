# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03: two overlapping ``discard_all`` calls must not let a
refill publish a spare that the still-running sweep never sees.

``worktree_pool.discard_all`` documents that every refill started
before the call returns drops its spare instead of publishing it.  The
gate was a single boolean, ``_discarding``, raised by every starting
``discard_all`` and cleared by every finishing one.  With TWO
overlapping calls the first finisher cleared the flag (and, by bumping
the generation a second time, re-stabilised it) while the second call
was still sweeping — so a refill that started after the first call
returned found ``generation == _generation and not _discarding``, and
published a spare while the second ``discard_all`` was still inside
its sweep.  The second call then returned with the pool non-empty: its
caller (a test teardown, an embedder shutting down) believed every
``kiss/wt-*`` spare was gone while one live spare worktree remained on
disk and in the pool.

Fix under test: the boolean is replaced by an active-discard COUNTER;
a refill publishes only while no ``discard_all`` is in flight.

The interleaving is deterministic, pinned with the real per-repo locks
exactly like ``test_audit0902_fix_sorcar_pool_discard_generation``:

1. a spare exists for repo A; the test holds ``repo_lock(A)``, so a
   ``discard_all`` (D2) blocks inside its sweep while removing it;
2. a second ``discard_all`` (D1) starts and finishes immediately (its
   own snapshots are empty) — on the broken code this clears the gate;
3. a refill for repo B runs to completion while D2 is still sweeping;
4. only then is A released and D2 returns.

No mocks: real git repositories, real threads, the real pool.
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


def test_refill_during_overlapping_discards_never_publishes(
    repos: tuple[Path, Path],
) -> None:
    repo_a, repo_b = repos
    assert worktree_pool.prewarm(repo_a)
    assert worktree_pool.spare_branches() != set()

    lock_a = repo_lock(repo_a)
    lock_a.acquire()
    d2 = threading.Thread(target=worktree_pool.discard_all, daemon=True)
    refill: threading.Thread | None = None
    try:
        d2.start()
        # D2 has snapshotted and cleared _spares and is now blocked on
        # repo_lock(A) while removing A's spare.
        _wait_until(lambda: worktree_pool.spare_branches() == set())
        assert d2.is_alive()

        # D1 overlaps D2 and finishes at once: it snapshots no refill
        # threads and no spares.  On the broken code this cleared the
        # _discarding boolean while D2 was still sweeping.
        d1 = threading.Thread(target=worktree_pool.discard_all, daemon=True)
        d1.start()
        d1.join(timeout=120)
        assert not d1.is_alive()
        assert d2.is_alive(), "D2 must still be blocked inside its sweep"

        # A refill for repo B, started AFTER D1 returned, runs to
        # completion while D2 is still sweeping.  It must observe the
        # in-flight D2 and drop its spare instead of publishing it.
        refill = worktree_pool.prewarm_async(repo_b)
        assert refill is not None
        refill.join(timeout=120)
        assert not refill.is_alive()
        assert d2.is_alive(), "D2 must still be blocked inside its sweep"
        published_during_sweep = worktree_pool.spare_branches()
    finally:
        lock_a.release()
    d2.join(timeout=120)
    assert not d2.is_alive()

    assert published_during_sweep == set(), (
        "a refill published a spare while an overlapping discard_all "
        f"was still sweeping: {published_during_sweep}"
    )
    assert worktree_pool.spare_branches() == set(), (
        "discard_all returned with a spare the sweep never saw"
    )
    assert _wt_branches(repo_b) == "", "the mid-sweep spare's worktree was left behind"
    assert _wt_branches(repo_a) == ""

    # The pool is fully usable afterwards: with no discard in flight a
    # refill publishes normally.
    assert worktree_pool.prewarm(repo_b)
    assert len(worktree_pool.spare_branches()) == 1
