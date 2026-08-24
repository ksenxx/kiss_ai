# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regressions for area-E race fixes in
``sorcar/sorcar_agent.py``.

Covers audit finding **E-RC1**: ``reclaim_abandoned_subagents``
snapshotted the pending list under ``_abandoned_lock`` but ran
``item.unbanked_usage()`` (a read-modify-write of ``item.counted``)
and ``_attribute_sub_usage`` (a read-modify-write of the parent's
``budget_used`` / ``total_tokens_used`` / ``total_steps``) OUTSIDE
the lock.  The agent thread and server-side worktree cleanup call the
method concurrently, so an abandoned child's spend could be counted
twice and parent totals could lose updates.  The whole bank-and-forget
sequence now runs under the lock; ``KISS_RACE_DELAY`` widens the
read-modify-write window inside ``unbanked_usage`` so the pre-fix
double-count is deterministic.

The test uses REAL ``SorcarAgent`` objects, REAL completed
``concurrent.futures.Future`` instances, and REAL threads.  No mocks,
patches or doubles, and no LLM calls.

E-R3 (the redundant middle ``_coerce_tasks`` call removed from
``_run_tasks_parallel``) is a pure deletion with no new branches; its
behaviour is covered by the existing ``run_parallel`` coercion suites
(e.g. ``test_bughunt3_coerce_tasks_json.py``).
"""

from __future__ import annotations

import os
import threading
import unittest
from concurrent.futures import Future

from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _AbandonedSubagent

_RACE_DELAY_ENV = "KISS_RACE_DELAY"


def _finished_child(budget: float, tokens: int, steps: int) -> tuple:
    """Return ``(future, agent)`` for a child that finished after being
    abandoned, having spent the given totals."""
    child = SorcarAgent("abandoned-child")
    child.budget_used = budget
    child.total_tokens_used = tokens
    child.total_steps = steps
    future: Future[str] = Future()
    future.set_result("success: true\nsummary: done\n")
    return future, child


class TestReclaimAbandonedSubagentsRace(unittest.TestCase):
    """Concurrent reclaimers must bank each child's spend exactly once."""

    def setUp(self) -> None:
        os.environ[_RACE_DELAY_ENV] = "0.05"

    def tearDown(self) -> None:
        os.environ.pop(_RACE_DELAY_ENV, None)

    def test_two_threads_never_double_count(self) -> None:
        for _ in range(5):
            parent = SorcarAgent("parent")
            future, child = _finished_child(5.0, 100, 7)
            with parent._abandoned_lock:
                parent._abandoned_subagents.append(
                    _AbandonedSubagent(future, child, (0.0, 0, 0)),
                )

            barrier = threading.Barrier(2)
            outcomes: list[bool] = []

            def reclaim(barrier=barrier, parent=parent,
                        outcomes=outcomes) -> None:
                barrier.wait(timeout=30)
                outcomes.append(parent.reclaim_abandoned_subagents())

            threads = [
                threading.Thread(target=reclaim) for _ in range(2)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            self.assertEqual(outcomes, [True, True])
            # The child's spend was banked exactly once — the pre-fix
            # code counted it once per racing thread.
            self.assertEqual(parent.budget_used, 5.0)
            self.assertEqual(parent.total_tokens_used, 100)
            self.assertEqual(parent.total_steps, 7)
            self.assertEqual(parent._abandoned_subagents, [])

    def test_still_running_child_banked_incrementally(self) -> None:
        parent = SorcarAgent("parent")
        child = SorcarAgent("slow-child")
        child.budget_used = 1.0
        child.total_tokens_used = 10
        child.total_steps = 1
        future: Future[str] = Future()  # never finishes until we say so
        with parent._abandoned_lock:
            parent._abandoned_subagents.append(
                _AbandonedSubagent(future, child, (0.0, 0, 0)),
            )

        # A short-timeout reclaim banks the spend so far and keeps the
        # still-running child registered.
        self.assertFalse(parent.reclaim_abandoned_subagents(timeout=0.01))
        self.assertEqual(parent.budget_used, 1.0)
        self.assertEqual(len(parent._abandoned_subagents), 1)

        # A second reclaim with no new spend must be a no-op (delta 0).
        self.assertFalse(parent.reclaim_abandoned_subagents())
        self.assertEqual(parent.budget_used, 1.0)
        self.assertEqual(parent.total_tokens_used, 10)

        # The child finishes after spending more; only the DELTA is
        # banked, and the entry is forgotten.
        child.budget_used = 3.5
        child.total_tokens_used = 40
        child.total_steps = 4
        future.set_result("success: true\nsummary: done\n")
        self.assertTrue(parent.reclaim_abandoned_subagents())
        self.assertEqual(parent.budget_used, 3.5)
        self.assertEqual(parent.total_tokens_used, 40)
        self.assertEqual(parent.total_steps, 4)
        self.assertEqual(parent._abandoned_subagents, [])

    def test_reclaim_with_nothing_pending_is_true(self) -> None:
        parent = SorcarAgent("parent")
        self.assertTrue(parent.reclaim_abandoned_subagents())
        self.assertTrue(parent.reclaim_abandoned_subagents(timeout=0.01))


if __name__ == "__main__":
    unittest.main()
