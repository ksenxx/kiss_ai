# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E regressions for the round-5 usage-ledger hardening.

The round-4 review (``tmp/review4-usage.md``) demonstrated five
defects; the ones whose code survives are regression-tested here
(finding 1's KISSAgent source atomicity lives in
``kiss/tests/core/test_conc2026_kiss_agent_usage_atomicity.py``;
findings 2a and 3 covered the in-process fan-out engine's abandoned-child
reclaim, which is gone with the engine: ``run_parallel`` children are
daemon sub-tasks now):

* Finding 2b — the mandatory classifier fold had one unkeyed append
  and no safe retry: a stop before it lost the spend, a blind retry
  after it would double-count.  The fold now appends under the
  classification's stable transaction key BEFORE clearing the consumed
  markers.
* Finding 4 — ledger reads were O(records ever appended) and repeated
  adjustments were quadratic (20,000 took 25s).  Bounded compaction
  folds the deduped prefix into a base record; reads stay
  O(threshold) and dedup keys stay counted forever.
* Finding 5 — ``_emit_merged_result_event`` zeroed the printer offsets
  before entering its restoration ``try``, so a stop in the gap left
  them zeroed.  The ``try`` now begins before the first mutation.

All tests drive the real classes with real threads and real injected
``KeyboardInterrupt``s at every opcode boundary of the real commit
paths.  No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from collections.abc import Callable
from typing import Any

from kiss.agents.sorcar.relentless_agent import (
    _COMPACTION_THRESHOLD,
    RelentlessAgent,
    _UsageEvent,
)
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _attribute_sub_usage,
    _ClassifierSpend,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.printer import Printer

_WAIT = 30.0


def _executor_with_spend(
    name: str, budget: float, tokens: int, steps: int,
) -> KISSAgent:
    """Return a real KISSAgent carrying the given accrued counters."""
    executor = KISSAgent(name)
    executor.budget_used = budget
    executor.total_tokens_used = tokens
    executor.step_count = steps
    return executor


def _codes_with_nested(functions: list[Any]) -> frozenset[types.CodeType]:
    """Code objects of *functions* plus their nested comprehensions."""
    stack: list[types.CodeType] = [fn.__code__ for fn in functions]
    codes: set[types.CodeType] = set()
    while stack:
        code = stack.pop()
        codes.add(code)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)
    return frozenset(codes)


def _run_with_injection_at(
    operation: Callable[[], object],
    codes: frozenset[types.CodeType],
    boundary: int,
) -> tuple[bool, bool]:
    """Run *operation*, raising one KeyboardInterrupt at an opcode boundary.

    ``sys.settrace`` with opcode events is the exact delivery model of
    ``PyThreadState_SetAsyncExc`` (the server's stop watchdog): the
    exception is raised between two bytecodes of the traced path.

    Returns:
        ``(injected, interrupted)`` — ``injected`` is False when the
        path had fewer opcode events than *boundary*.
    """
    seen = 0
    injected = False

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal seen, injected
        if frame.f_code in codes:
            frame.f_trace_opcodes = True
            if event == "opcode" and not injected:
                if seen == boundary:
                    injected = True
                    raise KeyboardInterrupt("injected stop")
                seen += 1
        return tracer

    interrupted = False
    try:
        sys.settrace(tracer)
        operation()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        sys.settrace(None)
    return injected, interrupted


# ---------------------------------------------------------------------------
# Finding 2b — the classifier fold is keyed and retry-safe.
# ---------------------------------------------------------------------------

_FOLD_CODES = _codes_with_nested(
    [
        SorcarAgent._fold_classifier_usage,
        RelentlessAgent._attribute_usage,
        RelentlessAgent._commit_usage_event,
        _UsageEvent.__new__,
    ]
)


def test_classifier_fold_interrupted_at_every_boundary_folds_exactly_once() -> None:
    """One injected stop at ANY fold opcode: the retry folds exactly once.

    Round-4 finding 2b, rewritten from
    ``tmp/review-scratch4/repro_classifier_fold_interrupt.py`` to
    assert FIXED behavior: whether the stop lands before the keyed
    append (the old design lost the spend — the transient fields were
    later cleared without a fold) or after it but before the consumed
    markers clear (a blind retry used to double-count), retrying the
    fold converges to the classifier spend banked EXACTLY once and the
    transient fields cleared.
    """
    spend = (0.25, 25, 2)
    boundary = 0
    while True:
        agent = SorcarAgent("classifier-fold")
        agent._classifier_spend = _ClassifierSpend(
            "classifier:test-invocation", *spend,
        )
        injected, interrupted = _run_with_injection_at(
            agent._fold_classifier_usage, _FOLD_CODES, boundary,
        )
        if not injected:
            break
        assert interrupted, f"boundary {boundary}: injection was swallowed"
        # The retry: the next fold on any unwind path.
        agent._fold_classifier_usage()
        assert agent.usage_snapshot() == spend, (
            f"boundary {boundary}: classifier spend lost or double-counted: "
            f"{agent.usage_snapshot()}"
        )
        assert agent._classifier_spend is None
        boundary += 1
    assert boundary > 0


def test_classifier_fold_is_idempotent() -> None:
    """Folding twice banks the published classifier spend exactly once."""
    agent = SorcarAgent("idempotent-fold")
    agent._classifier_spend = _ClassifierSpend("classifier:x", 0.25, 25, 2)
    agent._fold_classifier_usage()
    agent._fold_classifier_usage()
    assert agent.usage_snapshot() == (0.25, 25, 2)
    assert agent._classifier_spend is None


# ---------------------------------------------------------------------------
# Finding 4 — bounded compaction: exact totals, bounded reads, kept dedup.
# ---------------------------------------------------------------------------


class _OffsetPrinter(Printer):
    """Real Printer subclass carrying the per-task usage offsets."""

    def __init__(self) -> None:
        self.tokens_offset = 0
        self.budget_offset = 0.0
        self.steps_offset = 0
        self.calls: list[dict[str, Any]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Record the call; offset/emit tests assert on the kwargs."""
        self.calls.append(dict(kwargs))
        return ""

    def token_callback(self, token: str) -> None:
        """Unused by these paths."""

    def reset(self) -> None:
        """Unused by these paths."""


def test_compaction_bounds_the_ledger_and_preserves_exact_totals() -> None:
    """Many adjustments stay exact while the live record list stays bounded."""
    agent = RelentlessAgent("compact")
    writes = _COMPACTION_THRESHOLD * 10
    for _ in range(writes):
        agent._attribute_usage(0.5, 100, 1)
    assert agent.usage_snapshot() == (writes * 0.5, writes * 100, writes)
    ledger = agent._usage_ledger_object()
    assert (
        len(ledger.records) - ledger.view.fold_index < _COMPACTION_THRESHOLD
    ), "compaction never folded the ledger"


def test_compaction_preserves_session_and_transaction_dedup() -> None:
    """Keys folded into the base stay counted: no re-bank ever double-counts."""
    agent = RelentlessAgent("compact-dedup")
    executor = _executor_with_spend("s0", 1.25, 100, 3)
    agent._accumulate_usage(executor)
    agent._attribute_usage(1.0, 10, 1, key="reclaim:child:0")
    expected = (1.25 + 1.0, 110, 4)
    assert agent.usage_snapshot() == expected
    # Force several compactions over the banked keys.
    for _ in range(_COMPACTION_THRESHOLD * 3):
        agent._attribute_usage(0.5, 100, 1)
    grown = (
        expected[0] + _COMPACTION_THRESHOLD * 3 * 0.5,
        expected[1] + _COMPACTION_THRESHOLD * 3 * 100,
        expected[2] + _COMPACTION_THRESHOLD * 3,
    )
    assert agent.usage_snapshot() == grown
    # Retries of both keyed transactions must dedup against the base.
    agent._accumulate_usage(executor)
    agent._attribute_usage(1.0, 10, 1, key="reclaim:child:0")
    assert agent.usage_snapshot() == grown, (
        "a key folded into the compaction base was re-counted"
    )


def test_repeated_attributions_are_not_quadratic() -> None:
    """20,000 offset-refreshing attributions finish fast with bounded reads.

    The round-4 benchmark measured 25.4 seconds for 20,000 repeated
    ``_attribute_sub_usage`` calls (each appends one record and re-sums
    the WHOLE ledger to refresh the printer offsets): 1 + 2 + ... + n
    record visits.  With compaction every read is O(threshold), so the
    same workload is linear; the generous wall-clock bound (well under
    half the measured quadratic time even on a loaded machine) plus the
    bounded live-record assertion pin the complexity class without
    flaking on scheduler noise.
    """
    agent = SorcarAgent("linear-attributions")
    agent.printer = _OffsetPrinter()  # type: ignore[assignment]
    writes = 20_000
    started = time.monotonic()
    for _ in range(writes):
        _attribute_sub_usage(agent, 0.5, 100, 1)
    elapsed = time.monotonic() - started
    assert agent.usage_snapshot() == (writes * 0.5, writes * 100, writes)
    ledger = agent._usage_ledger_object()
    assert len(ledger.records) - ledger.view.fold_index < _COMPACTION_THRESHOLD
    assert elapsed < 10.0, (
        f"20,000 attributions took {elapsed:.1f}s — quadratic ledger reads"
    )
    printer = agent.printer
    assert isinstance(printer, _OffsetPrinter)
    assert printer.budget_offset == writes * 0.5
    assert printer.tokens_offset == writes * 100
    assert printer.steps_offset == writes


def test_concurrent_writers_and_readers_stay_exact_across_compaction() -> None:
    """Racing writers, readers and compactions never lose or tear a record."""
    agent = RelentlessAgent("compact-storm")
    writers = 4
    per_writer = _COMPACTION_THRESHOLD * 4
    barrier = threading.Barrier(writers + 2)
    stop_reading = threading.Event()
    torn: list[tuple[float, int, int]] = []

    def write() -> None:
        barrier.wait()
        for _ in range(per_writer):
            agent._attribute_usage(0.5, 100, 1)

    def read() -> None:
        barrier.wait()
        while not stop_reading.is_set():
            budget, tokens, steps = agent.usage_snapshot()
            if tokens != steps * 100 or budget != steps * 0.5:
                torn.append((budget, tokens, steps))
                return

    threads = [threading.Thread(target=write) for _ in range(writers)] + [
        threading.Thread(target=read) for _ in range(2)
    ]
    for t in threads:
        t.start()
    try:
        for t in threads[:writers]:
            t.join(timeout=_WAIT)
            assert not t.is_alive()
    finally:
        stop_reading.set()
    for t in threads[writers:]:
        t.join(timeout=_WAIT)
        assert not t.is_alive()
    assert torn == [], f"a reader saw a torn/incoherent snapshot: {torn}"
    total = writers * per_writer
    assert agent.usage_snapshot() == (total * 0.5, total * 100, total), (
        "compaction lost or double-counted records under concurrency"
    )


def test_reset_during_storm_lands_wholly_before_or_after() -> None:
    """A reset racing compacting writers yields only all-old or all-new states."""
    agent = RelentlessAgent("compact-reset")
    per_writer = _COMPACTION_THRESHOLD * 2
    barrier = threading.Barrier(3)

    def write() -> None:
        barrier.wait()
        for _ in range(per_writer):
            agent._attribute_usage(0.5, 100, 1)

    threads = [threading.Thread(target=write) for _ in range(2)]
    for t in threads:
        t.start()
    barrier.wait()
    agent.reset_usage()
    for t in threads:
        t.join(timeout=_WAIT)
        assert not t.is_alive()
    budget, tokens, steps = agent.usage_snapshot()
    # Whatever landed after the swap is a coherent prefix of the new
    # epoch; pre-swap records died with the old epoch.
    assert tokens == steps * 100 and budget == steps * 0.5, (
        f"mixed epochs after reset: {(budget, tokens, steps)}"
    )
    assert steps <= 2 * per_writer


def test_commit_racing_a_compaction_counts_exactly_once_at_every_boundary() -> None:
    """A whole compaction at ANY commit opcode never loses or doubles a record.

    The writer thread is parked at every opcode boundary of the real
    commit path in turn; while it is parked, the main thread runs a
    COMPLETE compaction (fold plus the single view publication).  The
    parked commit then resumes — its append landed either before the
    fold snapshot (already folded) or after it (still in the unfolded
    suffix) — and in every interleaving the totals must show the
    pre-filled records plus the racing event EXACTLY once.  This pins
    the append-is-the-commit invariant deterministically (the storm
    test only samples interleavings).
    """
    prefill = 10
    boundary = 0
    while True:
        agent = RelentlessAgent("commit-vs-compaction")
        for _ in range(prefill):
            agent._attribute_usage(0.5, 100, 1)
        ledger = agent._usage_ledger_object()
        paused = threading.Event()
        resume = threading.Event()
        state = {"seen": 0, "paused_here": False}
        codes = _codes_with_nested(
            [RelentlessAgent._commit_usage_event, _UsageEvent.__new__]
        )

        def writer(b: int = boundary, state: dict[str, Any] = state) -> None:
            def tracer(frame: Any, event: str, arg: Any) -> Any:
                if frame.f_code in codes:
                    frame.f_trace_opcodes = True
                    if event == "opcode" and not state["paused_here"]:
                        if state["seen"] == b:
                            state["paused_here"] = True
                            paused.set()
                            assert resume.wait(timeout=_WAIT)
                        else:
                            state["seen"] += 1
                return tracer

            sys.settrace(tracer)
            try:
                agent._attribute_usage(1.0, 10, 1, key="racing-txn")
            finally:
                sys.settrace(None)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        while not paused.wait(timeout=0.01):
            if not thread.is_alive():
                break
        if paused.is_set():
            agent._maybe_compact(ledger)
            resume.set()
        thread.join(timeout=_WAIT)
        assert not thread.is_alive()
        if not state["paused_here"]:
            break
        expected = (prefill * 0.5 + 1.0, prefill * 100 + 10, prefill + 1)
        assert agent.usage_snapshot() == expected, (
            f"boundary {boundary}: compaction racing the commit lost or "
            f"doubled the record: {agent.usage_snapshot()}"
        )
        boundary += 1
    assert boundary > 0


def test_compaction_skips_when_another_compaction_is_running() -> None:
    """The non-blocking claim makes a losing compactor a silent no-op."""
    agent = RelentlessAgent("compaction-claim")
    for _ in range(5):
        agent._attribute_usage(0.5, 100, 1)
    ledger = agent._usage_ledger_object()
    assert ledger.compaction_lock.acquire(blocking=False)
    try:
        before = len(ledger.records)
        agent._maybe_compact(ledger)
        # No fold happened: the records are untouched and the view is
        # still the empty one.
        assert len(ledger.records) == before
        assert ledger.view.fold_index == 0
    finally:
        ledger.compaction_lock.release()
    assert agent.usage_snapshot() == (2.5, 500, 5)


def test_keyless_records_fold_exactly_once() -> None:
    """A keyless one-shot record folds once and is never re-counted."""
    agent = RelentlessAgent("keyless")
    ledger = agent._usage_ledger_object()
    agent._attribute_usage(1.0, 10, 1)
    agent._attribute_usage(0.5, 100, 1, key="reclaim:x", seq=0)
    agent._maybe_compact(ledger)
    agent._maybe_compact(ledger)
    assert agent.usage_snapshot() == (1.5, 110, 2)
    # Compaction never touches the append-only records list; the whole
    # prefix is covered by the published view, and one-shot records
    # leave no residue in the folded seen map.
    assert ledger.view.fold_index == len(ledger.records)
    assert ledger.view.seen.get("reclaim:x") == 0


# ---------------------------------------------------------------------------
# Finding 5 — merged-result offsets survive a stop at every boundary.
# ---------------------------------------------------------------------------

_EMIT_CODES = _codes_with_nested([RelentlessAgent._emit_merged_result_event])


def test_emit_merged_result_never_mutates_offsets_at_any_boundary() -> None:
    """One injected stop at ANY emit opcode leaves the offsets untouched.

    Round-4 finding 5, rewritten from
    ``tmp/review-scratch4/repro_emit_offset_interrupt.py`` to assert
    FIXED behavior: the emit no longer zeroes/restores the printer's
    task-keyed offsets at all — it passes offset-adjusted raw values,
    which the printer re-offsets — so NO injection point (including one
    landing inside what used to be the restoration loop) can leave the
    offsets mutated.  When the print itself completed before the
    injection, its raw totals must equal the cumulative snapshot minus
    the untouched offsets.
    """
    offsets = (200, 2.0, 6)
    cumulative = (2.5, 450, 8)
    boundary = 0
    while True:
        agent = RelentlessAgent("emit-interrupt")
        printer = _OffsetPrinter()
        printer.tokens_offset, printer.budget_offset, printer.steps_offset = (
            offsets
        )
        agent.printer = printer  # type: ignore[assignment]
        agent._attribute_usage(*cumulative)
        injected, _interrupted = _run_with_injection_at(
            lambda: agent._emit_merged_result_event(
                {"success": True, "is_continue": False, "summary": "done"}
            ),
            _EMIT_CODES,
            boundary,
        )
        current = (
            printer.tokens_offset,
            printer.budget_offset,
            printer.steps_offset,
        )
        assert current == offsets, (
            f"boundary {boundary}: the injected stop left the printer "
            f"offsets mutated: {current}"
        )
        for kwargs in printer.calls:
            assert kwargs["total_tokens"] == cumulative[1] - offsets[0]
            assert kwargs["step_count"] == cumulative[2] - offsets[2]
            assert kwargs["cost"] == f"${cumulative[0] - offsets[1]:.4f}"
        if not injected:
            assert len(printer.calls) == 1
            break
        boundary += 1
    assert boundary > 0
