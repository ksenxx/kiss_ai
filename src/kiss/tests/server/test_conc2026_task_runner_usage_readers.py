# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The server's usage readers and reset must be coherent, never torn.

Round-3 review (``tmp/review3-core-sorcar.md``) findings 2 and 3:

* ``task_runner._subtask_metrics`` read a ``RelentlessAgent``'s
  counters through three independent property reads.  A server-thread
  abandoned-subagent reclaim attributing concurrently could land
  between two reads, so the helper returned an impossible mix (old
  tokens with new budget/steps) — and its output is PERSISTED
  (task-history extras) and BROADCAST (failure result banners), so a
  torn read is never repaired.
* ``task_runner._zero_usage_counters`` reset a reused agent through
  three property setters — three separate transactions — so a racing
  attribution produced mixed states like ``(0.0, 800, 0)``.

Both now go through the agent's append-only usage ledger:
``usage_snapshot()`` sums ONE ledger reference and ``reset_usage()``
swaps in a fresh ledger with one atomic store, so every observable
state is a coherent prefix of the accounting history.  Plain
agent-shaped objects keep the per-attribute fallbacks.

All tests drive the real helpers on real agents with real threads.  No
mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Callable

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.kiss_agent import KISSAgent
from kiss.server.task_runner import _subtask_metrics, _zero_usage_counters

#: Generous bound for operations that must terminate promptly.
_WAIT = 30.0


def _executor_with_spend(name: str, budget: float, tokens: int, steps: int) -> KISSAgent:
    """Return a real KISSAgent carrying the given accrued counters."""
    executor = KISSAgent(name)
    executor.budget_used = budget
    executor.total_tokens_used = tokens
    executor.step_count = steps
    return executor


def _run_paused_at(
    operation: Callable[[], None],
    code: types.CodeType,
    event_kind: str,
    paused: threading.Event,
    resume: threading.Event,
) -> None:
    """Run *operation*, pausing once at *event_kind* of *code*.

    Models any admitted scheduling delay at an exact execution point;
    ``sys.settrace`` is thread-local, so only this thread is delayed.
    """
    stopped = False

    def trace(frame, event, arg):  # type: ignore[no-untyped-def]
        nonlocal stopped
        if frame.f_code is code and event == event_kind and not stopped:
            stopped = True
            paused.set()
            assert resume.wait(timeout=_WAIT)
        return trace

    sys.settrace(trace)
    try:
        operation()
    finally:
        sys.settrace(None)


def test_subtask_metrics_never_torn_under_banking_storm() -> None:
    """Every ``_subtask_metrics`` read mid-storm is a coherent triple.

    Four threads bank distinct sessions of exactly (1.0, 100, 3) each
    while the main thread reads continuously.  Any coherent read
    satisfies ``tokens == 100*k``, ``cost == k`` and ``steps == 3*k``
    for one k (sums of 1.0 are exact in binary floating point); the
    pre-fix independent property reads produced mixed k's.
    """
    agent = RelentlessAgent("storm-metrics")
    banks_per_thread = 100
    barrier = threading.Barrier(5)

    def storm(worker: int) -> None:
        barrier.wait()
        for i in range(banks_per_thread):
            agent._accumulate_usage(
                _executor_with_spend(f"w{worker}-s{i}", 1.0, 100, 3)
            )

    bankers = [threading.Thread(target=storm, args=(w,)) for w in range(4)]
    for banker in bankers:
        banker.start()
    barrier.wait()
    observed: list[tuple[int, float, int]] = []
    while any(banker.is_alive() for banker in bankers):
        observed.append(_subtask_metrics(agent))
    for banker in bankers:
        banker.join(timeout=_WAIT)
        assert not banker.is_alive()
    observed.append(_subtask_metrics(agent))

    for tokens, cost, steps in observed:
        assert tokens % 100 == 0, (tokens, cost, steps)
        k = tokens // 100
        assert steps == 3 * k, (tokens, cost, steps)
        assert abs(cost - float(k)) < 1e-9, (tokens, cost, steps)
    total = 4 * banks_per_thread
    assert observed[-1] == (100 * total, float(total), 3 * total)


def test_subtask_metrics_reads_whole_snapshot_after_attribution() -> None:
    """The round-3 torn-reader interleaving now yields a coherent triple.

    The reviewer paused ``_subtask_metrics`` between its token and cost
    reads, attributed one coherent (2.0, 200, 2), and observed the
    impossible ``(100, 3.0, 3)``.  The reader now takes its ONE
    snapshot after the resume, so the same interleaving — reader
    paused at the snapshot boundary, one concurrent attribution —
    returns the full post-attribution triple.
    """
    agent = RelentlessAgent("terminal-reader")
    agent._attribute_usage(1.0, 100, 1)
    paused = threading.Event()
    resume = threading.Event()
    outcome: dict[str, tuple[int, float, int]] = {}

    def read_metrics() -> None:
        outcome["metrics"] = _subtask_metrics(agent)

    reader = threading.Thread(
        target=_run_paused_at,
        args=(
            read_metrics,
            RelentlessAgent.usage_snapshot.__code__,
            "call",
            paused,
            resume,
        ),
    )
    reader.start()
    assert paused.wait(timeout=_WAIT), "reader never reached the snapshot"
    agent._attribute_usage(2.0, 200, 2)
    assert agent.usage_snapshot() == (3.0, 300, 3)
    resume.set()
    reader.join(timeout=_WAIT)
    assert not reader.is_alive()
    assert outcome["metrics"] == (300, 3.0, 3), (
        f"torn terminal metrics: {outcome['metrics']}"
    )


def test_subtask_metrics_step_fallback_and_plain_agents() -> None:
    """Step fallback and non-snapshot agents keep their exact semantics."""
    # RelentlessAgent whose sessions reported zero steps: fall back to
    # the live step_count, exactly as for plain agents.
    agent = RelentlessAgent("no-steps")
    agent._attribute_usage(1.0, 100, 0)
    agent.step_count = 7
    assert _subtask_metrics(agent) == (100, 1.0, 7)
    # Plain KISSAgent: per-attribute reads.
    plain = _executor_with_spend("plain", 2.0, 20, 5)
    assert _subtask_metrics(plain) == (20, 2.0, 5)
    # A nulled agent yields zeros.
    assert _subtask_metrics(None) == (0, 0.0, 0)


def test_zero_usage_counters_reset_is_never_torn_by_attribution() -> None:
    """The round-3 torn-reset interleaving now yields only coherent states.

    The reviewer paused the reset between its token and budget setters,
    attributed one coherent (8.0, 800, 24), and the final state was the
    impossible mix ``(0.0, 800, 0)``.  The reset is now ONE atomic
    ledger swap: pausing the resetter thread just before the swap and
    attributing concurrently must end in a coherent state — here the
    attribution lands in the pre-reset epoch and the swap discards it
    wholesale (attribution-then-reset), never a mix.
    """
    agent = RelentlessAgent("server-reset")
    agent._attribute_usage(5.0, 500, 15)
    paused = threading.Event()
    resume = threading.Event()

    resetter = threading.Thread(
        target=_run_paused_at,
        args=(
            lambda: _zero_usage_counters(agent),
            RelentlessAgent.reset_usage.__code__,
            "call",
            paused,
            resume,
        ),
    )
    resetter.start()
    assert paused.wait(timeout=_WAIT), "reset never reached the ledger swap"
    # Nothing is zeroed yet — the pre-fix partial state (5.0, 0, 15)
    # can no longer exist.
    assert agent.usage_snapshot() == (5.0, 500, 15)
    agent._attribute_usage(8.0, 800, 24)
    assert agent.usage_snapshot() == (13.0, 1300, 39)
    resume.set()
    resetter.join(timeout=_WAIT)
    assert not resetter.is_alive()
    assert agent.usage_snapshot() == (0.0, 0, 0), (
        f"reset left a mixed state: {agent.usage_snapshot()}"
    )
    assert agent.step_count == 0


def test_zero_usage_counters_then_attribution_lands_whole() -> None:
    """An attribution after the reset carries its whole triple."""
    agent = RelentlessAgent("reset-then-attribute")
    agent._attribute_usage(5.0, 500, 15)
    _zero_usage_counters(agent)
    assert agent.usage_snapshot() == (0.0, 0, 0)
    agent._attribute_usage(8.0, 800, 24)
    assert agent.usage_snapshot() == (8.0, 800, 24)


def test_zero_usage_counters_plain_agent_fallback() -> None:
    """Plain agent-shaped objects are zeroed attribute by attribute."""
    plain = _executor_with_spend("plain", 2.0, 20, 5)
    plain.total_steps = 4  # type: ignore[attr-defined]
    _zero_usage_counters(plain)
    assert plain.budget_used == 0.0
    assert plain.total_tokens_used == 0
    assert plain.total_steps == 0  # type: ignore[attr-defined]
    assert plain.step_count == 0
