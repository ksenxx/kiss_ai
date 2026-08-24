# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``prewarm_async`` scheduling race (F-RC1).

Before the fix, ``worktree_pool.prewarm_async`` performed its dedup
check (``key in _spares or key in _prewarming``) in one ``_pool_lock``
acquisition and the ``_refill_threads[key] = thread`` registration in a
second one, and ``_prewarming`` is only populated INSIDE ``prewarm()``
after the thread has started.  Concurrent callers could therefore all
pass the check, spawn one refill thread each, and overwrite each
other's registrations — so ``discard_all()`` joined only the
last-registered thread and an unjoined in-flight refill could publish
a spare AFTER the sweep, exactly what ``discard_all``'s docstring
promises cannot happen.

These tests run the real pool against real temporary git repositories
— no mocks — following the patterns of ``test_worktree_pool.py``.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path

from kiss.agents.sorcar import worktree_pool
from kiss.tests.agents.sorcar.test_worktree_pool import (
    _enable_pool,
    _make_repo,
    _restore_pool_env,
)


class TestPrewarmAsyncScheduling:
    """Concurrent ``prewarm_async`` dedup and ``discard_all`` joining."""

    def setup_method(self) -> None:
        self._prev_env = _enable_pool()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self._tmpdir.name) / "repo")

    def teardown_method(self) -> None:
        worktree_pool.discard_all()
        self._tmpdir.cleanup()
        _restore_pool_env(self._prev_env)

    def _wt_branches(self) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repo), "branch", "--list", "kiss/wt-*"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()

    def test_concurrent_prewarm_async_spawns_single_refill(self) -> None:
        # All callers release together on the barrier, hitting the
        # dedup check before any refill thread has had a chance to add
        # itself to ``_prewarming`` — the exact interleaving that made
        # every caller spawn (and overwrite) a thread before the fix.
        n = 8
        results: list[threading.Thread | None] = [None] * n
        barrier = threading.Barrier(n)

        def call(i: int) -> None:
            barrier.wait(timeout=30)
            results[i] = worktree_pool.prewarm_async(self.repo)

        callers = [
            threading.Thread(target=call, args=(i,), daemon=True)
            for i in range(n)
        ]
        for t in callers:
            t.start()
        for t in callers:
            t.join(timeout=60)
            assert not t.is_alive()
        spawned = [t for t in results if t is not None]
        assert len(spawned) == 1, (
            f"{len(spawned)} concurrent prewarm_async calls spawned a "
            f"refill thread; the dedup guard must admit exactly one"
        )
        spawned[0].join(timeout=60)
        assert not spawned[0].is_alive()
        assert len(worktree_pool.spare_branches()) == 1

    def test_discard_all_after_concurrent_scheduling_leaves_nothing(self) -> None:
        # discard_all() must join EVERY scheduled refill: a spare
        # published after the sweep (by a thread whose registration was
        # overwritten) would leak a kiss/wt-* branch forever.
        n = 8
        barrier = threading.Barrier(n)

        def call() -> None:
            barrier.wait(timeout=30)
            worktree_pool.prewarm_async(self.repo)

        callers = [
            threading.Thread(target=call, daemon=True) for _ in range(n)
        ]
        for t in callers:
            t.start()
        for t in callers:
            t.join(timeout=60)
        worktree_pool.discard_all()
        assert worktree_pool.spare_branches() == set()
        # Wait out any refill thread that discard_all failed to join;
        # its spare would appear now.
        with worktree_pool._pool_lock:
            leftovers = list(worktree_pool._refill_threads.values())
        for t in leftovers:
            t.join(timeout=60)
        assert worktree_pool.spare_branches() == set()
        assert self._wt_branches() == "", (
            "a refill missed by discard_all published a spare after "
            "the sweep"
        )

    def test_finished_refill_thread_does_not_block_next_refill(self) -> None:
        # A DEAD registered thread is not "refill in flight": after the
        # spare is consumed a new refill must be schedulable.
        first = worktree_pool.prewarm_async(self.repo)
        assert first is not None
        first.join(timeout=60)
        assert worktree_pool.take_spare(self.repo) is not None
        second = worktree_pool.prewarm_async(self.repo)
        assert second is not None, (
            "a finished refill thread left in the registry must not "
            "suppress the next refill"
        )
        second.join(timeout=60)
        assert len(worktree_pool.spare_branches()) == 1
