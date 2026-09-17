# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E regressions for the round-7 usage-ledger hardening.

The round-6 review (``tmp/review6-usage.md``) demonstrated five
defects with deterministic interleavings
(``tmp/review-scratch6/repro_round6_interleavings.py``,
``repro_classifier_source_publish.py``,
``repro_abandoned_registration_liveness.py``); each is pinned here as
a permanent regression test against the SIMPLIFIED ledger design (the
per-epoch records list is append-only and never swapped, cut or
reordered — the append IS the commit — and compaction publishes one
immutable view with a single store):

* Finding 1 — the old compactor published an incomplete tail before
  draining the old list: a commit that returned during the gap
  vanished from snapshots, an injected stop before the drain lost it
  permanently, and a one-shot writer stopped after a stale-list append
  had no retry identity.  Now there is no swap, no drain and no stale
  list: a committed record is visible the instant its append returns,
  under every compaction/interrupt interleaving.
* Finding 2 — the reclaim validated ``item.epoch`` and then committed
  through the parent's CURRENT ledger, so a reset between check and
  append charged old-child spend to the new epoch.  Every reclaim
  commit is now BOUND to the epoch object captured at registration:
  late spend settles into the (discarded) prior epoch, never the new
  one.
* Finding 3 — the classifier stored its fold key and the three spend
  fields separately, so a stop between the stores made the mandatory
  fold consume and clear a zero triple.  The whole outcome is now ONE
  immutable ``_ClassifierSpend`` published with a single store —
  all-or-nothing, never a torn subset.
* Finding 4 — ``_register_abandoned`` captured the tracking list
  before locking, so a concurrent reclaim (which replaced the list)
  stranded a live child in a detached list and worktree cleanup
  reported "safe" under a live writer.  Registration now fetches the
  list INSIDE the lock and reclaim updates it in place.
* Finding 5 — every 200-record compaction copied the whole historical
  key set (``base.keys | folded``), making maintenance quadratic
  (640k events took 25.5s).  The folded dedup state is now a leveled
  max-seq-per-source map with geometric merging — O(n log n) total.

All tests drive the real classes with real threads, a real loopback
HTTP model server where a model call is involved, and real injected
``KeyboardInterrupt``s at the exact seams the review exercised.  No
mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import inspect
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from typing import Any

import pytest

from kiss.agents.sorcar.relentless_agent import (
    _COMPACTION_THRESHOLD,
    RelentlessAgent,
    _ledger_totals,
)
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _AbandonedSubagent,
    _register_abandoned,
)
from kiss.agents.sorcar.task_classifier import (
    classify_task,
    clear_classification_cache,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
)

_WAIT = 30.0


def _line_of(func: Callable[..., Any], marker: str) -> int:
    """Return the absolute line number of *marker* inside *func*.

    Resolving the seam by source text instead of a hard-coded number
    keeps the interleaving tests pinned to the STATEMENT they must
    interrupt, not to a file offset that drifts with every edit.
    """
    source, start = inspect.getsourcelines(func)
    for offset, line in enumerate(source):
        if marker in line:
            return start + offset
    raise AssertionError(f"marker {marker!r} not found in {func.__qualname__}")


def _executor_with_spend(
    name: str, budget: float, tokens: int, steps: int,
) -> KISSAgent:
    """Return a real KISSAgent carrying the given accrued counters."""
    executor = KISSAgent(name)
    executor.budget_used = budget
    executor.total_tokens_used = tokens
    executor.step_count = steps
    return executor


def _prefilled_agent(name: str) -> RelentlessAgent:
    """Return an agent one commit short of the compaction threshold."""
    agent = RelentlessAgent(name)
    for i in range(_COMPACTION_THRESHOLD - 1):
        agent._attribute_usage(0.5, 100, 1, key=f"prefill:{i}")
    return agent


class _PauseAt:
    """Pause (or interrupt) a thread when it reaches a source line.

    A ``sys.settrace`` line tracer installed INSIDE the target thread:
    when execution reaches ``(code, line)`` for the first time it
    either parks on an event pair (pause seam) or raises
    ``KeyboardInterrupt`` (the exact delivery model of the server stop
    watchdog's ``PyThreadState_SetAsyncExc`` at that boundary).
    """

    def __init__(
        self,
        func: Callable[..., Any],
        marker: str,
        interrupt: bool = False,
    ) -> None:
        self._code = func.__code__
        self._line = _line_of(func, marker)
        self._interrupt = interrupt
        self.reached = threading.Event()
        self.resume = threading.Event()

    def run(self, operation: Callable[[], object]) -> BaseException | None:
        """Run *operation* under the tracer; return the escaped exception."""
        fired = False

        def tracer(frame: Any, event: str, arg: Any) -> Any:
            nonlocal fired
            if (
                event == "line"
                and frame.f_code is self._code
                and frame.f_lineno == self._line
                and not fired
            ):
                fired = True
                if self._interrupt:
                    raise KeyboardInterrupt("injected stop")
                self.reached.set()
                assert self.resume.wait(_WAIT), "resume never signalled"
            return tracer

        error: BaseException | None = None
        sys.settrace(tracer)
        try:
            operation()
        except BaseException as exc:  # noqa: BLE001 — inspected by tests
            error = exc
        finally:
            sys.settrace(None)
        assert fired, "the traced seam was never reached"
        return error


# ---------------------------------------------------------------------------
# Finding 1 — no visibility gap, no interrupt loss, no stale-list append.
# ---------------------------------------------------------------------------


def test_commit_returning_mid_compaction_is_immediately_visible() -> None:
    """A commit that returns while compaction is paused is never hidden.

    Round-6 broken state: between the tail swap and the drain, a
    snapshot taken AFTER a racing commit returned read 200 steps
    instead of 201.  The compactor now publishes one view and never
    touches the records list, so the racer is visible from the instant
    its append returns — including while the compactor is parked one
    statement before the view store.
    """
    agent = _prefilled_agent("visible-during-fold")
    pause = _PauseAt(RelentlessAgent._maybe_compact, "ledger.view = _LedgerView(")
    errors: list[BaseException] = []

    def compacting_commit() -> None:
        error = pause.run(
            lambda: agent._attribute_usage(0.5, 100, 1, key="compaction-trigger")
        )
        if error is not None:
            errors.append(error)

    compactor = threading.Thread(target=compacting_commit)
    compactor.start()
    assert pause.reached.wait(_WAIT)
    # This commit appends and RETURNS while the fold is mid-flight.
    agent._attribute_usage(1.0, 10, 1, key="returned-racer")
    expected = (
        _COMPACTION_THRESHOLD * 0.5 + 1.0,
        _COMPACTION_THRESHOLD * 100 + 10,
        _COMPACTION_THRESHOLD + 1,
    )
    # The round-6 gap read: the racer has committed, the fold has not
    # published — the snapshot MUST already include the racer.
    assert agent.usage_snapshot() == expected
    pause.resume.set()
    compactor.join(_WAIT)
    assert not compactor.is_alive() and errors == []
    assert agent.usage_snapshot() == expected


def test_interrupted_compaction_loses_nothing_and_recovers() -> None:
    """An injected stop at the view publication cannot lose any commit.

    Round-6 broken state: a ``KeyboardInterrupt`` between the tail
    swap and the drain permanently stranded a returned commit on an
    unreachable list.  Compaction now performs exactly one mutation
    (the view store); interrupting it at that seam leaves the old view
    and every record in place, and the next commit simply re-folds.
    """
    agent = _prefilled_agent("interrupted-fold")
    pause = _PauseAt(RelentlessAgent._maybe_compact, "ledger.view = _LedgerView(")
    outcome: list[BaseException | None] = []

    def compacting_commit() -> None:
        def commit_then_wait() -> None:
            agent._attribute_usage(0.5, 100, 1, key="compaction-trigger")

        outcome.append(pause.run(commit_then_wait))

    compactor = threading.Thread(target=compacting_commit)
    compactor.start()
    assert pause.reached.wait(_WAIT)
    agent._attribute_usage(1.0, 10, 1, key="returned-racer")
    pause.resume.set()
    compactor.join(_WAIT)
    assert not compactor.is_alive() and outcome == [None]
    expected = (
        _COMPACTION_THRESHOLD * 0.5 + 1.0,
        _COMPACTION_THRESHOLD * 100 + 10,
        _COMPACTION_THRESHOLD + 1,
    )
    assert agent.usage_snapshot() == expected

    # Now the interrupt variant on a fresh agent: the compactor dies
    # exactly at the single view store (the stop watchdog's delivery
    # model, raised between two bytecodes at that seam).
    interruptor = _PauseAt(
        RelentlessAgent._maybe_compact,
        "ledger.view = _LedgerView(",
        interrupt=True,
    )
    agent2 = _prefilled_agent("interrupted-fold-2")
    error = interruptor.run(
        lambda: agent2._attribute_usage(0.5, 100, 1, key="compaction-trigger")
    )
    assert isinstance(error, KeyboardInterrupt)
    ledger = agent2._usage_ledger_object()
    # The aborted fold published nothing and lost nothing.
    assert ledger.view.fold_index == 0
    base_expected = (
        _COMPACTION_THRESHOLD * 0.5,
        _COMPACTION_THRESHOLD * 100,
        _COMPACTION_THRESHOLD,
    )
    assert agent2.usage_snapshot() == base_expected
    # The next commit re-folds successfully; totals stay exact.
    agent2._attribute_usage(1.0, 10, 1, key="after-interrupt")
    assert ledger.view.fold_index > 0
    assert agent2.usage_snapshot() == (
        base_expected[0] + 1.0,
        base_expected[1] + 10,
        base_expected[2] + 1,
    )


def test_one_shot_commit_survives_compaction_and_immediate_stop() -> None:
    """A one-shot append racing a full compaction is counted, stop or not.

    Round-6 broken state: a one-shot writer that loaded the records
    list, lost a full compaction (swap plus drain), appended to the
    stale list and was stopped before its confirmation loop lost its
    spend with no retry identity.  The records list is never replaced
    now, so the writer's append lands in the ONLY list; a stop
    delivered on the very next line (before the compaction check)
    leaves the spend committed.
    """
    agent = _prefilled_agent("one-shot")
    stop_after_append = _line_of(
        RelentlessAgent._commit_usage_event, "if len(ledger.records)"
    )
    append_line = _line_of(
        RelentlessAgent._commit_usage_event, "ledger.records.append("
    )
    code = RelentlessAgent._commit_usage_event.__code__
    errors: list[BaseException] = []
    paused = threading.Event()
    resume = threading.Event()

    def one_shot_writer() -> None:
        state = {"paused": False}

        def tracer(frame: Any, event: str, arg: Any) -> Any:
            if event == "line" and frame.f_code is code:
                if frame.f_lineno == append_line and not state["paused"]:
                    state["paused"] = True
                    paused.set()
                    assert resume.wait(_WAIT)
                elif frame.f_lineno == stop_after_append and state["paused"]:
                    # The append committed on the previous line; the
                    # stop lands before anything else runs.
                    raise KeyboardInterrupt("injected after append")
            return tracer

        sys.settrace(tracer)
        try:
            agent._attribute_usage(1.0, 10, 1)  # one-shot: no key
        except BaseException as exc:  # noqa: BLE001 — asserted below
            errors.append(exc)
        finally:
            sys.settrace(None)

    writer = threading.Thread(target=one_shot_writer)
    writer.start()
    assert paused.wait(_WAIT)
    # A complete production compaction runs while the writer is parked
    # before its append.
    agent._attribute_usage(0.5, 100, 1, key="compaction-trigger")
    assert agent._usage_ledger_object().view.fold_index == _COMPACTION_THRESHOLD
    resume.set()
    writer.join(_WAIT)
    assert not writer.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt)
    # Round-6 asserted the spend was LOST; it must now be counted.
    assert agent.usage_snapshot() == (
        _COMPACTION_THRESHOLD * 0.5 + 1.0,
        _COMPACTION_THRESHOLD * 100 + 10,
        _COMPACTION_THRESHOLD + 1,
    )


# ---------------------------------------------------------------------------
# Finding 2 — reclaim commits are bound to the captured epoch object.
# ---------------------------------------------------------------------------


def test_reclaim_crossing_reset_settles_into_old_epoch_only() -> None:
    """A reset between epoch check and append never charges the new epoch.

    Round-6 broken state: the reclaimer validated ``item.epoch`` under
    the lock, was paused before the attribution, a ``reset_usage()``
    swapped the ledger, and the resumed append charged the OLD child's
    spend to the NEW epoch.  The commit is now bound to the epoch
    object captured at registration, so the late spend settles in the
    discarded old ledger and the new epoch stays clean.
    """
    parent = SorcarAgent("epoch-parent")
    old_epoch = parent._usage_epoch()
    child = RelentlessAgent("epoch-child")
    child._accumulate_usage(_executor_with_spend("epoch-session", 2.0, 200, 2))
    future: Future[str] = Future()
    future.set_result("done")
    item = _AbandonedSubagent(future, child, (0.0, 0, 0), epoch=old_epoch)
    parent._abandoned_subagents.append(item)

    pause = _PauseAt(_AbandonedSubagent.bank_unbanked, "_race_delay()")
    errors: list[BaseException] = []

    def reclaim() -> None:
        error = pause.run(lambda: parent.reclaim_abandoned_subagents())
        if error is not None:
            errors.append(error)

    thread = threading.Thread(target=reclaim)
    thread.start()
    assert pause.reached.wait(_WAIT)
    # The round-6 race: reset lands after the epoch check, before the
    # attribution append.
    parent.reset_usage()
    assert parent._usage_epoch() is not old_epoch
    pause.resume.set()
    thread.join(_WAIT)
    assert not thread.is_alive() and errors == []
    # Round-6 asserted the NEW epoch read (2.0, 200, 2); it must be
    # clean now, with the spend settled into the discarded old epoch.
    assert parent.usage_snapshot() == (0.0, 0, 0)
    assert _ledger_totals(old_epoch) == (2.0, 200, 2)
    assert parent._abandoned_subagents == []


def test_stale_epoch_spend_advances_checkpoint_without_new_epoch_charge() -> None:
    """Repeated reclaims of a stale-epoch child never touch the new epoch."""
    parent = SorcarAgent("stale-parent")
    old_epoch = parent._usage_epoch()
    child = RelentlessAgent("stale-child")
    child._accumulate_usage(_executor_with_spend("stale-session", 1.0, 100, 1))
    future: Future[str] = Future()
    item = _AbandonedSubagent(future, child, (0.0, 0, 0), epoch=old_epoch)
    parent._abandoned_subagents.append(item)
    parent.reset_usage()
    # Live child: tracked (blocks worktree deletion), spend settled old.
    assert not parent.reclaim_abandoned_subagents()
    assert parent.usage_snapshot() == (0.0, 0, 0)
    assert _ledger_totals(old_epoch) == (1.0, 100, 1)
    # The checkpoint advanced: a second reclaim banks nothing more.
    child._accumulate_usage(_executor_with_spend("stale-late", 2.0, 200, 2))
    future.set_result("done")
    assert parent.reclaim_abandoned_subagents()
    assert parent.usage_snapshot() == (0.0, 0, 0)
    assert _ledger_totals(old_epoch) == (3.0, 300, 3)
    assert parent._abandoned_subagents == []


# ---------------------------------------------------------------------------
# Finding 3 — the classifier outcome publishes atomically.
# ---------------------------------------------------------------------------

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"


@pytest.fixture
def classifier_env() -> Iterator[IsolatedKissHome]:
    """Isolated KISS_HOME with the classifier kill switch lifted."""
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "0"
    isolated = IsolatedKissHome("kiss-usage-round7-")
    clear_classification_cache()
    try:
        yield isolated
    finally:
        clear_classification_cache()
        if saved is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved
        isolated.cleanup()


def _classifier_responder(request: dict[str, Any]) -> dict[str, Any]:
    """A fixed OpenAI-compatible classification with nonzero usage."""
    return {
        "id": "chatcmpl-round7-classifier",
        "object": "chat.completion",
        "created": 0,
        "model": STANDIN_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"is_simple": true, "is_development": false}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 12,
            "total_tokens": 52,
        },
    }


def test_classifier_outcome_publication_is_all_or_nothing(
    classifier_env: IsolatedKissHome,
) -> None:
    """A stop around the publication leaves a full outcome or none.

    Round-6 broken state (real loopback HTTP): a stop after the key
    store but before the spend stores left ``(key, 0, 0, 0)`` visible,
    and the mandatory fold consumed and cleared the zero triple —
    real provider spend vanished.  The outcome is now ONE immutable
    record: a stop BEFORE the store publishes nothing (the fold banks
    nothing) and a stop AFTER it publishes everything (the fold banks
    the complete triple).  A torn subset cannot exist.
    """
    server = StandInModelServer(_classifier_responder)
    try:
        baseline = classify_task(
            task="round7 classifier baseline unique",
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
        )
        assert baseline.budget_used > 0
        assert baseline.tokens_used == 52
        assert baseline.steps == 1
        complete = (baseline.budget_used, baseline.tokens_used, baseline.steps)

        # Stop BEFORE the single publication store: nothing published.
        agent = SorcarAgent("classifier-atomic-before")
        before = _PauseAt(
            SorcarAgent._classify_task_once,
            "self._classifier_spend = _ClassifierSpend(",
            interrupt=True,
        )
        error = before.run(
            lambda: agent._classify_task_once(
                STANDIN_MODEL,
                "round7 classifier interrupted-before unique",
                server.model_config,
                enabled_override=True,
            )
        )
        assert isinstance(error, KeyboardInterrupt)
        assert agent._classifier_spend is None
        agent._fold_classifier_usage()  # the mandatory finally action
        assert agent.usage_snapshot() == (0.0, 0, 0)

        # Stop AFTER the store (round-6's exact seam, one line later):
        # the complete outcome is published and the fold banks it all.
        agent2 = SorcarAgent("classifier-atomic-after")
        after = _PauseAt(
            SorcarAgent._classify_task_once,
            "self._task_classification = outcome.classification",
            interrupt=True,
        )
        error2 = after.run(
            lambda: agent2._classify_task_once(
                STANDIN_MODEL,
                "round7 classifier interrupted-after unique",
                server.model_config,
                enabled_override=True,
            )
        )
        assert isinstance(error2, KeyboardInterrupt)
        spend = agent2._classifier_spend
        assert spend is not None
        assert (spend.budget, spend.tokens, spend.steps) == complete
        agent2._fold_classifier_usage()
        assert agent2.usage_snapshot() == complete
        # The fold is exactly-once: repeating it changes nothing.
        agent2._fold_classifier_usage()
        assert agent2.usage_snapshot() == complete
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Finding 4 — registration and reclaim agree on one tracking list.
# ---------------------------------------------------------------------------


def test_registration_racing_reclaim_keeps_live_child_tracked() -> None:
    """A reclaim completing mid-registration cannot strand a live child.

    Round-6 broken state: registration captured the tracking list,
    a reclaim replaced the attribute with a new list, and the live
    child was appended only to the detached one — the parent then
    reported "no abandoned children" while the child's thread kept
    writing into the (deletable) worktree.  Registration now fetches
    the list inside the lock and reclaim mutates it in place.
    """
    parent = SorcarAgent("registration-parent")
    tracked_list = parent._abandoned_subagents
    # A finished child that the racing reclaim will bank and forget.
    old_child = RelentlessAgent("already-done-child")
    old_child._accumulate_usage(_executor_with_spend("old-spend", 1.0, 100, 1))
    old_future: Future[str] = Future()
    old_future.set_result("done")
    parent._abandoned_subagents.append(
        _AbandonedSubagent(
            old_future, old_child, (0.0, 0, 0), epoch=parent._usage_epoch(),
        )
    )

    live_child = RelentlessAgent("new-live-child")
    live_future: Future[str] = Future()
    pause = _PauseAt(_register_abandoned, "with lock:")
    errors: list[BaseException] = []

    def register() -> None:
        error = pause.run(
            lambda: _register_abandoned(
                parent, [live_future], [live_child], [(0.0, 0, 0)],
            )
        )
        if error is not None:
            errors.append(error)

    thread = threading.Thread(target=register)
    thread.start()
    assert pause.reached.wait(_WAIT)
    # The racing reclaim runs to completion while registration is
    # parked before the lock.
    assert parent.reclaim_abandoned_subagents()
    assert parent.usage_snapshot() == (1.0, 100, 1)
    assert parent._abandoned_subagents == []
    pause.resume.set()
    thread.join(_WAIT)
    assert not thread.is_alive() and errors == []

    # Round-6 asserted zero tracked items and a True cleanup verdict
    # while the child lived; the live child MUST be tracked now, in
    # the very list object every reader consults.
    assert not live_future.done()
    assert parent._abandoned_subagents is tracked_list
    assert len(parent._abandoned_subagents) == 1
    assert parent._abandoned_subagents[0].future is live_future
    assert parent.reclaim_abandoned_subagents() is False
    # Once the child finishes, its spend is banked and cleanup is safe.
    live_child._accumulate_usage(_executor_with_spend("late", 2.0, 200, 2))
    live_future.set_result("done")
    assert parent.reclaim_abandoned_subagents() is True
    assert parent.usage_snapshot() == (3.0, 300, 3)
    assert parent._abandoned_subagents == []


# ---------------------------------------------------------------------------
# Finding 5 — compaction maintenance is near-linear, dedup state bounded.
# ---------------------------------------------------------------------------


def test_compaction_maintenance_is_near_linear() -> None:
    """The reviewer's unique-key flood scales near-linearly now.

    Round-6 measured the old design at 2.76 us/event for 80k events
    and 39.87 us/event (25.5 s total) for 640k — the per-fold copy of
    the whole historical key set is quadratic.  The leveled max-seq
    map merges geometrically, so the same 640k-unique-key flood must
    finish in a small constant multiple of the 80k per-event cost and
    far under the old wall-clock time.
    """

    def flood(events: int) -> float:
        agent = RelentlessAgent(f"scaling-{events}")
        started = time.monotonic()
        for i in range(events):
            agent._attribute_usage(0.5, 100, 1, key=f"prefill:{i}")
        elapsed = time.monotonic() - started
        assert agent.usage_snapshot() == (
            events * 0.5, events * 100, events,
        )
        view = agent._usage_ledger_object().view
        # One dedup entry per keyed source, organized in O(log n)
        # geometric levels — never a flat per-fold copy.
        assert view.seen.size == view.fold_index
        assert len(view.seen.levels) <= 24
        return elapsed

    small_events = 80_000
    big_events = 640_000
    small = flood(small_events)
    big = flood(big_events)
    assert big < 15.0, (
        f"640k-event flood took {big:.1f}s — quadratic maintenance is back"
    )
    small_us = small / small_events * 1e6
    big_us = big / big_events * 1e6
    assert big_us <= 4 * small_us + 0.5, (
        f"per-event cost grew {small_us:.2f} -> {big_us:.2f} us/event "
        "across an 8x flood — maintenance is not near-linear"
    )


def test_one_shot_flood_leaves_no_dedup_residue() -> None:
    """Keyless one-shot adjustments retain no folded identity at all."""
    agent = RelentlessAgent("one-shot-flood")
    events = _COMPACTION_THRESHOLD * 50
    for _ in range(events):
        agent._attribute_usage(0.5, 100, 1)
    assert agent.usage_snapshot() == (events * 0.5, events * 100, events)
    ledger = agent._usage_ledger_object()
    view = ledger.view
    assert view.seen.size == 0
    assert view.seen.levels == ()
    assert len(ledger.records) - view.fold_index < _COMPACTION_THRESHOLD


def test_per_source_sequences_dedup_across_folds() -> None:
    """Retries of folded generations dedup; new generations count.

    The bounded dedup design stores one MAX seq per source; this pins
    its contract end to end: generations committed in order, folded,
    then retried (same source, same seq, same values) count once, and
    a later generation still counts.
    """
    agent = RelentlessAgent("seq-dedup")
    agent._attribute_usage(1.0, 10, 1, key="reclaim:x", seq=0)
    agent._attribute_usage(2.0, 20, 2, key="reclaim:x", seq=1)
    # Force a fold over both generations.
    for _ in range(_COMPACTION_THRESHOLD):
        agent._attribute_usage(0.0, 1, 0)
    ledger = agent._usage_ledger_object()
    assert ledger.view.seen.get("reclaim:x") == 1
    expected = (3.0, 30 + _COMPACTION_THRESHOLD, 3)
    assert agent.usage_snapshot() == expected
    # Retries of the folded generations are no-ops.
    agent._attribute_usage(1.0, 10, 1, key="reclaim:x", seq=0)
    agent._attribute_usage(2.0, 20, 2, key="reclaim:x", seq=1)
    assert agent.usage_snapshot() == expected
    # The next generation counts exactly once, retry included.
    agent._attribute_usage(4.0, 40, 4, key="reclaim:x", seq=2)
    agent._attribute_usage(4.0, 40, 4, key="reclaim:x", seq=2)
    assert agent.usage_snapshot() == (
        expected[0] + 4.0, expected[1] + 40, expected[2] + 4,
    )
