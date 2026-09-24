# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a deferred worktree merge retry that loses the
worktree to a competing claimant stays silent.

``_merge_deferred_worktrees`` snapshots the tabs whose merge was
deferred, releases ``_state_lock``, and only then asks
``_handle_worktree_action`` to merge each one.  Between the snapshot
and that claim another claimant — a second trigger of the retry from
another thread (a task finishing while the user presses Git Commit),
or the user's own Merge click — can take the worktree.  The retry's
own attempt is then refused by the busy guard (``merge_in_progress()``)
or finds nothing pending any more, but the tab's deferral marker was
already cleared by the winner, so the retry used to read "the marker
is gone, therefore I merged" and broadcast the refusal to the tab as a
failed ``worktree_result`` — a spurious "A merge or discard is already
in progress" panel next to the winner's real result.

The schedule is forced, not hoped for: the server INSTANCE's
``_handle_worktree_action`` is replaced by a plain function that makes
both trigger threads meet at a ``threading.Barrier`` before delegating
to the real method.  A thread reaches that barrier only after it has
snapshotted the deferred tab as a candidate, so both threads snapshot
BEFORE either claims; the loser must then find the marker cleared by
the winner and get ``_DEFERRAL_SUPERSEDED``.  (The wall-clock
``KISS_RACE_DELAY`` hook is not needed: a thread parked at the barrier
cannot claim, so the other thread's snapshot always still sees the
marker.)  Real temp git repo, real ``VSCodeServer`` merge flow, real
threads; only the LLM agent body and the LLM commit-message generator
are the harness's deterministic stand-ins.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from kiss.server.merge_flow import _DEFERRAL_SUPERSEDED
from kiss.tests.server.test_worktree_deferred_auto_merge import (
    _DeferredMergeBase,
)
from kiss.tests.server.test_worktree_repo_aware_busy_guard import _WT_TAB


class TestConcurrentDeferredRetries(_DeferredMergeBase):
    """Two simultaneous triggers: one merge, one report, no spurious failure."""

    def _gate_worktree_action(
        self, parties: int,
    ) -> list[dict[str, Any]]:
        """Make the server's ``_handle_worktree_action`` a rendezvous.

        Replaces the bound method on the server instance (a plain
        attribute assignment, restored in cleanup) with a function that
        records the call, waits until *parties* threads have arrived,
        then delegates to the real method and records the result.

        Args:
            parties: Number of callers that must arrive before any of
                them is allowed to claim.

        Returns:
            The call log; each entry holds ``kwargs`` and ``result``.
        """
        calls: list[dict[str, Any]] = []
        gate = threading.Barrier(parties, timeout=30)
        real = self.server._handle_worktree_action

        def rendezvous_then_claim(
            action: str, tab_id: str = "", **kwargs: Any,
        ) -> dict[str, Any]:
            entry: dict[str, Any] = {
                "action": action, "tab_id": tab_id, "kwargs": kwargs,
                "thread": threading.current_thread().name,
            }
            calls.append(entry)
            gate.wait()
            result = real(action, tab_id, **kwargs)
            entry["result"] = result
            return result

        self.server._handle_worktree_action = (  # type: ignore[method-assign]
            rendezvous_then_claim
        )
        self.addCleanup(
            setattr, self.server, "_handle_worktree_action", real,
        )
        return calls

    def test_losing_trigger_stays_silent(self) -> None:
        self._strand_worktree()
        state = self._wt_state()
        branch = state.wt_merge_deferred_branch
        assert branch is not None
        calls = self._gate_worktree_action(parties=2)

        errors: list[BaseException] = []

        def _trigger() -> None:
            try:
                self.server._merge_deferred_worktrees(Path(self.repo))
            except BaseException as exc:  # pragma: no cover — surfaced below
                errors.append(exc)

        triggers = [
            threading.Thread(target=_trigger, name=f"deferred-trigger-{i}")
            for i in range(2)
        ]
        for t in triggers:
            t.start()
        for t in triggers:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in triggers), "triggers must finish"
        assert not errors, errors

        # Both triggers snapshotted the same deferral and carried it into
        # their claim: two calls, one per thread, both owning ``branch``.
        assert len(calls) == 2, calls
        assert {c["thread"] for c in calls} == {t.name for t in triggers}
        for call in calls:
            assert call["action"] == "merge" and call["tab_id"] == _WT_TAB
            assert call["kwargs"]["deferred_branch"] == branch, call
        outcomes = [c["result"] for c in calls]
        superseded = [r for r in outcomes if r is _DEFERRAL_SUPERSEDED]
        won = [r for r in outcomes if r is not _DEFERRAL_SUPERSEDED]
        assert len(superseded) == 1 and len(won) == 1, outcomes
        assert won[0]["success"], won

        self._assert_merged()
        results = [
            e for e in self._worktree_results() if e.get("tabId") == _WT_TAB
        ]
        assert len(results) == 1 and results[0]["success"], (
            "exactly the winning trigger reports; the loser lost the "
            f"worktree before claiming it and must stay silent: {results}"
        )
