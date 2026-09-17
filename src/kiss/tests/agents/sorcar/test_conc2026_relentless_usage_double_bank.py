# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Usage accounting must be exact-once, coherent, and deadlock-free.

``RelentlessAgent.perform_task``'s session try block calls
``_accumulate_usage(executor)`` on its success path AND in its
``except BaseException`` handler before re-raising.  The server's stop
watchdog injects ``KeyboardInterrupt`` asynchronously
(``PyThreadState_SetAsyncExc``), which can land between ANY two
bytecodes of the first call.  The accounting state is an APPEND-ONLY
ledger of immutable ``_UsageEvent`` records: committing is one atomic
``list.append``, duplicates of one session share a retry-stable
``(source, seq)`` identity and count once on read, and a reset swaps in a fresh
ledger with one ``STORE_ATTR``.  No lock, no compare-and-swap — so no
injection can leak a lock, no delayed writer can overwrite another
writer's commit, and no reader can observe a torn triple.

The round-3 review (``tmp/review3-core-sorcar.md``) demonstrated three
defects of the previous lock+CAS design that the ledger removes by
construction, each regression-tested here:

1. A healthy publisher delayed between its identity check and its
   store overwrote every publication completed during its pause
   (``test_banking_storm_with_delayed_publisher_loses_nothing``), and
   after one injected ``KeyboardInterrupt`` a later session bank that
   RETURNED SUCCESSFULLY was permanently erased by the interrupted
   publisher's stale store
   (``test_interrupted_bank_then_concurrent_banks_lose_nothing``).
2. Terminal readers tore the triple across separate property reads
   (``test_final_reclaim_reads_one_coherent_snapshot``; the server-side
   readers are covered in
   ``kiss/tests/server/test_conc2026_task_runner_usage_readers.py``).
3. A reset racing an attribution produced mixed states
   (``test_reset_lands_wholly_before_or_after_attribution`` and
   ``test_attribution_holding_old_epoch_is_discarded_by_reset``).

All tests drive the real classes — real threads, real injected
``KeyboardInterrupt``s at every opcode boundary of the real banking
path, and a real ``SorcarAgent.run`` over a local HTTP model server.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
import sys
import threading
import types
from collections.abc import Callable
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar import relentless_agent, sorcar_agent
from kiss.agents.sorcar.relentless_agent import RelentlessAgent, _UsageEvent
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _AbandonedSubagent,
    _attribute_sub_usage,
    _ClassifierSpend,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.printer import Printer
from kiss.tests.core.test_budget_enforcement_e2e import (
    _CHEAP,
    _read_body,
    _send_json,
    _start_server,
    _tool_call_response,
)

#: Generous bound for operations that must terminate promptly; the
#: pre-fix defects blocked forever or lost spend, so any completion
#: within this bound distinguishes fixed from broken.
_WAIT = 30.0


def _executor_with_spend(name: str, budget: float, tokens: int, steps: int) -> KISSAgent:
    """Return a real KISSAgent carrying the given accrued counters."""
    executor = KISSAgent(name)
    executor.budget_used = budget
    executor.total_tokens_used = tokens
    executor.step_count = steps
    return executor


def _session_records(agent: RelentlessAgent, executor: KISSAgent) -> list[_UsageEvent]:
    """Return the ledger records banked for *executor*'s session."""
    key = executor.__dict__.get("_usage_session_key")
    return [
        event
        for event in agent._usage_events()
        if key is not None and event.source == key
    ]


def _banking_path_codes() -> frozenset[types.CodeType]:
    """Code objects of the whole banking path, nested comprehensions included.

    The commit spans ``_accumulate_usage`` (key, dedup scan, source
    read), ``_session_key`` (retry-stable key assignment),
    ``_usage_ledger_object`` (lazy epoch access), the executor's
    ``usage_snapshot`` (the atomic source read),
    ``_commit_usage_event`` / ``_maybe_compact`` (the append and its
    compaction-safe confirmation) and the ``_UsageEvent`` record
    constructor: a stop can land between any two bytecodes of ANY of
    them, so the injection sweep must cover them all.
    """
    stack: list[types.CodeType] = [
        RelentlessAgent._accumulate_usage.__code__,
        RelentlessAgent._usage_ledger_object.__code__,
        RelentlessAgent._commit_usage_event.__code__,
        RelentlessAgent._maybe_compact.__code__,
        KISSAgent.usage_snapshot.__code__,
        relentless_agent._session_key.__code__,
        _UsageEvent.__new__.__code__,
    ]
    codes: set[types.CodeType] = set()
    while stack:
        code = stack.pop()
        codes.add(code)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)
    return frozenset(codes)


_BANKING_CODES = _banking_path_codes()


def _bank_with_injection_at(
    boundary: int, make_agent: Callable[[], RelentlessAgent],
) -> tuple[bool, bool, RelentlessAgent]:
    """Bank one executor, injecting one KeyboardInterrupt at an opcode boundary.

    Runs ``_accumulate_usage`` under ``sys.settrace`` with opcode
    tracing enabled for every banking-path code object and raises one
    real ``KeyboardInterrupt`` at the ``boundary``-th opcode event —
    the exact delivery model of ``PyThreadState_SetAsyncExc``.  On
    interruption it performs exactly the recovery call
    ``perform_task``'s ``except BaseException`` handler makes, and
    asserts the totals are exact-once (one ledger record for the
    session) afterwards.

    Args:
        boundary: Ordinal of the opcode event at which to inject.
        make_agent: Factory for the agent under test (RelentlessAgent
            or SorcarAgent).

    Returns:
        ``(injected, committed_before_recovery, agent)``.
    """
    agent = make_agent()
    executor = _executor_with_spend("s0", 1.25, 100, 3)
    seen = 0
    injected = False

    def tracer(frame, event, arg):  # type: ignore[no-untyped-def]
        nonlocal seen, injected
        if frame.f_code in _BANKING_CODES:
            frame.f_trace_opcodes = True
            if event == "opcode" and not injected:
                if seen == boundary:
                    injected = True
                    raise KeyboardInterrupt("injected stop")
                seen += 1
        return tracer

    committed_before_recovery = False
    try:
        sys.settrace(tracer)
        agent._accumulate_usage(executor)
    except KeyboardInterrupt:
        sys.settrace(None)
        committed_before_recovery = len(_session_records(agent, executor)) == 1
        # perform_task's except-BaseException recovery call.  It must
        # neither double-count nor lose the session's spend, whether
        # the injection landed before or after the atomic append.
        agent._accumulate_usage(executor)
    finally:
        sys.settrace(None)
    assert agent.usage_snapshot() == (1.25, 100, 3), (
        boundary, agent.usage_snapshot(),
    )
    assert len(_session_records(agent, executor)) == 1, (
        boundary, _session_records(agent, executor),
    )
    return injected, committed_before_recovery, agent


def _agent_after_one_interrupted_bank(
    make_agent: Callable[[], RelentlessAgent],
) -> RelentlessAgent:
    """Return an agent that survived one injected stop inside a real bank."""
    injected, _committed, agent = _bank_with_injection_at(0, make_agent)
    assert injected, "boundary 0 must inject before the commit"
    return agent


def _run_paused_at(
    operation: Callable[[], None],
    code: types.CodeType,
    event_kind: str,
    paused: threading.Event,
    resume: threading.Event,
) -> None:
    """Run *operation*, pausing once at *event_kind* of *code*.

    The pause models any admitted scheduling delay (GIL preemption, a
    debugger, an OS stall) at an exact execution point; ``sys.settrace``
    is thread-local, so only the calling thread is delayed.
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


def _attribute_and_signal(agent: RelentlessAgent, done: threading.Event) -> None:
    """Attribute one multi-dimensional delta, then signal completion."""
    agent._attribute_usage(0.25, 10, 1)
    done.set()


def test_one_injection_at_every_opcode_boundary_banks_exactly_once() -> None:
    """One injected stop at ANY opcode boundary: exact-once, no deadlock.

    Sweeps every opcode boundary of the real banking path.  At each
    one, a real ``KeyboardInterrupt`` unwinds the first bank and the
    recovery call re-banks; the totals must be exact and the ledger
    must hold exactly ONE record for the session at EVERY boundary
    (asserted inside the harness).  Both interruption classes must
    occur (before the atomic append: recovery banks the full amount
    once; after: recovery finds the session record and no-ops), and
    after every injection a multi-dimensional attribution — what
    ``SorcarAgent.run``'s mandatory ``finally`` performs via
    ``_fold_classifier_usage`` — must terminate and be exact.
    """
    boundary = 0
    saw_pre_commit = False
    saw_post_commit = False
    while True:
        injected, committed, agent = _bank_with_injection_at(
            boundary, lambda: RelentlessAgent("t"),
        )
        if not injected:
            break
        saw_pre_commit = saw_pre_commit or not committed
        saw_post_commit = saw_post_commit or committed
        finished = threading.Event()
        waiter = threading.Thread(
            target=_attribute_and_signal, args=(agent, finished), daemon=True,
        )
        waiter.start()
        waiter.join(timeout=_WAIT)
        assert finished.is_set(), (
            f"attribution after the injection at boundary {boundary} hung"
        )
        assert agent.usage_snapshot() == (1.5, 110, 4), (
            boundary, agent.usage_snapshot(),
        )
        boundary += 1
    assert boundary > 0
    assert saw_pre_commit, "no injection landed before the commit"
    assert saw_post_commit, "no injection landed after the commit"


def test_banking_storm_with_delayed_publisher_loses_nothing() -> None:
    """A delayed publisher can never erase completed publications.

    Round-3 reproduction A, rewritten to assert fixed behavior: under
    the lock+CAS design, one HEALTHY publisher paused between its
    identity check and its store overwrote every publication completed
    during its pause (four banks' $8 collapsed back to $1).  With the
    append-only ledger the delayed banker holds only a list reference:
    eight concurrent banks append eight records it cannot disturb, and
    its own resume appends a ninth — nothing is lost, in any order.
    """
    agent = RelentlessAgent("delayed-holder")
    delayed_executor = _executor_with_spend("delayed", 1.0, 100, 3)
    paused = threading.Event()
    resume = threading.Event()
    delayed = threading.Thread(
        target=_run_paused_at,
        args=(
            lambda: agent._accumulate_usage(delayed_executor),
            RelentlessAgent._usage_ledger_object.__code__,
            "return",
            paused,
            resume,
        ),
    )
    delayed.start()
    assert paused.wait(timeout=_WAIT), "delayed banker never reached the ledger"

    executors = [_executor_with_spend(f"s{i}", 1.0, 100, 3) for i in range(8)]
    barrier = threading.Barrier(8)

    def bank(executor: KISSAgent) -> None:
        barrier.wait()
        agent._accumulate_usage(executor)

    bankers = [
        threading.Thread(target=bank, args=(executor,)) for executor in executors
    ]
    for banker in bankers:
        banker.start()
    for banker in bankers:
        banker.join(timeout=_WAIT)
        assert not banker.is_alive()

    before_delayed_resumes = agent.usage_snapshot()
    assert before_delayed_resumes == (8.0, 800, 24), before_delayed_resumes
    resume.set()
    delayed.join(timeout=_WAIT)
    assert not delayed.is_alive()
    assert agent.usage_snapshot() == (9.0, 900, 27), (
        f"delayed publisher erased completed publications: "
        f"{agent.usage_snapshot()}"
    )
    for executor in [delayed_executor, *executors]:
        assert len(_session_records(agent, executor)) == 1


def test_interrupted_bank_then_concurrent_banks_lose_nothing() -> None:
    """After one injected stop, later concurrent session banks all stick.

    Round-3 reproduction B, rewritten to assert fixed behavior: under
    the lock+CAS design one injected ``KeyboardInterrupt`` leaked the
    usage lock, and a later session bank that RETURNED SUCCESSFULLY
    (totals and banked mark visible) was then permanently erased by the
    other publisher's stale store.  With the ledger, the delayed first
    bank appends its own record and cannot overwrite the second bank's.
    """
    agent = _agent_after_one_interrupted_bank(
        lambda: RelentlessAgent("one-interrupt"),
    )
    assert agent.usage_snapshot() == (1.25, 100, 3)

    first_executor = _executor_with_spend("first-later-session", 1.0, 10, 1)
    second_executor = _executor_with_spend("second-later-session", 2.0, 20, 2)

    paused = threading.Event()
    resume = threading.Event()
    first = threading.Thread(
        target=_run_paused_at,
        args=(
            lambda: agent._accumulate_usage(first_executor),
            RelentlessAgent._usage_ledger_object.__code__,
            "return",
            paused,
            resume,
        ),
    )
    first.start()
    assert paused.wait(timeout=_WAIT)

    second = threading.Thread(target=lambda: agent._accumulate_usage(second_executor))
    second.start()
    second.join(timeout=_WAIT)
    assert not second.is_alive()
    after_second = agent.usage_snapshot()
    assert after_second == (3.25, 120, 5), after_second
    assert len(_session_records(agent, second_executor)) == 1

    resume.set()
    first.join(timeout=_WAIT)
    assert not first.is_alive()
    assert agent.usage_snapshot() == (4.25, 130, 6), (
        f"a completed session bank was erased: {agent.usage_snapshot()}"
    )
    assert len(_session_records(agent, first_executor)) == 1
    assert len(_session_records(agent, second_executor)) == 1


class _FinishHandler(BaseHTTPRequestHandler):
    """Local chat-completions server whose model finishes immediately."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        _send_json(
            self,
            _tool_call_response(
                "finish",
                json.dumps(
                    {
                        "success": True,
                        "is_continue": False,
                        "summary_in_html": "<p>done</p>",
                    }
                ),
                *_CHEAP,
            ),
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def test_sorcar_run_finally_folds_classifier_after_interrupted_bank(
    tmp_path: Path,
) -> None:
    """``SorcarAgent.run``'s mandatory ``finally`` always terminates.

    The round-2 deadlock: one injected stop leaked the usage lock
    inside a bank, and the next run's unwind — ``SorcarAgent.run``'s
    unconditional ``finally`` -> ``_fold_classifier_usage()`` — blocked
    forever.  The ledger fold is one lock-free append, so nothing can
    block.  This drives a REAL ``SorcarAgent.run`` (local HTTP model
    server, one session, one ``finish`` tool call) on an agent that
    survived a real injected stop inside a real bank, and asserts the
    whole run — reset, session bank, classifier fold — terminates with
    the classifier spend folded exactly once.
    """
    agent = _agent_after_one_interrupted_bank(lambda: SorcarAgent("stopped-parent"))
    assert isinstance(agent, SorcarAgent)
    # Preseed the classification state exactly as a prior classifier
    # run leaves it: verdict cached (no classifier model call inside
    # run) with unbanked classifier spend awaiting the finally's fold.
    agent._classification_attempted = True
    agent._classifier_spend = _ClassifierSpend(
        "classifier:preseeded-run", 0.25, 25, 2,
    )
    srv, url = _start_server(_FinishHandler)
    outcome: dict[str, object] = {}

    def drive() -> None:
        outcome["result"] = agent.run(
            model_name="gpt-4o-mini",
            prompt_template="Finish immediately.",
            max_steps=3,
            max_budget=1.0,
            max_sub_sessions=2,
            work_dir=str(tmp_path / "wd"),
            web_tools=False,
            is_parallel=False,
            append_basic_tools=False,
            verbose=False,
            model_config={"base_url": url, "api_key": "test-key"},
        )
        outcome["done"] = True

    runner = threading.Thread(target=drive, daemon=True)
    try:
        runner.start()
        runner.join(timeout=120)
        assert outcome.get("done"), (
            "SorcarAgent.run hung: its finally never finished folding "
            "the classifier spend"
        )
    finally:
        srv.shutdown()
    payload = yaml.safe_load(str(outcome["result"]))
    assert payload["success"] is True
    budget, tokens, steps = agent.usage_snapshot()
    # The session banked the executor's spend (10 prompt + 5 completion
    # tokens, 1 step) and the finally folded the classifier's exactly
    # once on top.
    assert tokens == 15 + 25
    assert steps == 1 + 2
    assert budget >= 0.25
    assert agent._classifier_spend is None


def test_final_reclaim_reads_one_coherent_snapshot() -> None:
    """A final abandoned-child reclaim must never lose spend to a torn read.

    Round-2 demonstration: the reclaim's ``_agent_usage`` read the
    three properties separately; paused after materializing the old
    budget, eight real banks then published eight coherent records,
    and the resumed reader combined the OLD budget with the NEW
    tokens/steps — the completed child was discarded with $8 of its
    spend permanently missing from the parent.  The reader now sums
    ONE ledger reference (``usage_snapshot``), so the same
    interleaving — reader paused at the read boundary, eight
    concurrent banks, child future completing before the resume —
    must attribute the full triple.
    """
    child = RelentlessAgent("child")
    parent = SorcarAgent("parent")
    future: Future[str] = Future()
    item = _AbandonedSubagent(future, child, (0.0, 0, 0))
    parent._abandoned_subagents.append(item)

    paused = threading.Event()
    resume = threading.Event()

    reader = threading.Thread(
        target=_run_paused_at,
        args=(
            lambda: parent.reclaim_abandoned_subagents(),
            sorcar_agent._agent_usage.__code__,
            "call",
            paused,
            resume,
        ),
    )
    reader.start()
    assert paused.wait(timeout=_WAIT), "reclaimer never reached the usage read"

    executors = [
        _executor_with_spend(f"bank-{i}", 1.0, 100, 3) for i in range(8)
    ]
    barrier = threading.Barrier(8)

    def bank(executor: KISSAgent) -> None:
        barrier.wait()
        child._accumulate_usage(executor)

    bankers = [
        threading.Thread(target=bank, args=(executor,)) for executor in executors
    ]
    for banker in bankers:
        banker.start()
    for banker in bankers:
        banker.join(timeout=_WAIT)
        assert not banker.is_alive()
    assert child.usage_snapshot() == (8.0, 800, 24)

    # The child completes while the reclaimer is paused at the read:
    # this reclaim is the LAST look at the child, so a torn read here
    # is never repaired.
    future.set_result("done")
    resume.set()
    reader.join(timeout=_WAIT)
    assert not reader.is_alive(), "reclaimer hung"

    assert parent.usage_snapshot() == (8.0, 800, 24), (
        f"final reclaim lost spend: {parent.usage_snapshot()}"
    )
    assert parent._abandoned_subagents == []


def test_reset_lands_wholly_before_or_after_attribution(tmp_path: Path) -> None:
    """A reset crossing a live attribution never yields a mixed state.

    The attribution thread is paused BEFORE it loads the ledger; the
    real ``_reset`` then swaps in the new epoch.  The reset must be
    coherently all-zero, and the resumed attribution must land its
    whole triple in the NEW epoch — never a mix.
    """
    agent = SorcarAgent("reused-parent")
    agent._attribute_usage(5.0, 500, 15)

    paused = threading.Event()
    resume = threading.Event()

    worker = threading.Thread(
        target=_run_paused_at,
        args=(
            lambda: _attribute_sub_usage(agent, 8.0, 800, 24),
            RelentlessAgent._attribute_usage.__code__,
            "call",
            paused,
            resume,
        ),
    )
    worker.start()
    assert paused.wait(timeout=_WAIT)

    # The next run's real reset, racing the paused attribution.
    agent._reset(
        model_name="m",
        max_sub_sessions=1,
        max_steps=1,
        max_budget=100.0,
        work_dir=str(tmp_path / "work"),
        docker_image=None,
    )
    assert agent.usage_snapshot() == (0.0, 0, 0), (
        f"reset published a mixed state: {agent.usage_snapshot()}"
    )

    resume.set()
    worker.join(timeout=_WAIT)
    assert not worker.is_alive(), "attribution hung after the reset"
    assert agent.usage_snapshot() == (8.0, 800, 24), (
        f"attribution crossing the reset tore: {agent.usage_snapshot()}"
    )


def test_attribution_holding_old_epoch_is_discarded_by_reset() -> None:
    """An attribution that loaded the pre-reset ledger linearizes before it.

    The other legal serialization of a reset/attribution race: the
    attribution already holds the OLD list reference when the reset
    swaps epochs, so its record lands in the discarded epoch — the
    observable history is attribution-then-reset, coherently zero.  A
    mixed state (some dimensions surviving the reset) must never occur.
    """
    agent = RelentlessAgent("reset-epoch")
    agent._attribute_usage(5.0, 500, 15)
    old_ledger = agent._usage_events()

    paused = threading.Event()
    resume = threading.Event()
    worker = threading.Thread(
        target=_run_paused_at,
        args=(
            lambda: agent._attribute_usage(8.0, 800, 24),
            RelentlessAgent._usage_ledger_object.__code__,
            "return",
            paused,
            resume,
        ),
    )
    worker.start()
    assert paused.wait(timeout=_WAIT)

    agent.reset_usage()
    assert agent.usage_snapshot() == (0.0, 0, 0)

    resume.set()
    worker.join(timeout=_WAIT)
    assert not worker.is_alive()
    assert agent.usage_snapshot() == (0.0, 0, 0), (
        f"a pre-reset attribution leaked into the new epoch: "
        f"{agent.usage_snapshot()}"
    )
    # The whole triple landed in the discarded epoch — not torn.
    assert any(
        (event.budget, event.tokens, event.steps) == (8.0, 800, 24)
        for event in old_ledger
    )


def test_eight_threads_bank_eight_distinct_executors_exactly() -> None:
    """Eight concurrent banks of eight distinct sessions all count once."""
    agent = RelentlessAgent("t")
    executors = [
        _executor_with_spend(f"s{i}", 1.0, 100, 3) for i in range(8)
    ]
    barrier = threading.Barrier(8)

    def bank(executor: KISSAgent) -> None:
        barrier.wait()
        agent._accumulate_usage(executor)

    threads = [
        threading.Thread(target=bank, args=(executor,)) for executor in executors
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_WAIT)
        assert not t.is_alive()
    assert agent.usage_snapshot() == (8.0, 800, 24)


def test_concurrent_banks_and_attributions_thrash_exactly() -> None:
    """Interleaved banks and attributions on one agent sum exactly.

    Concurrent appenders must never lose either side's increment; with
    an append-only ledger every commit is preserved by construction.
    """
    agent = RelentlessAgent("t")
    executors = [
        _executor_with_spend(f"s{i}", 1.0, 100, 3) for i in range(4)
    ]
    barrier = threading.Barrier(8)

    def bank(executor: KISSAgent) -> None:
        barrier.wait()
        agent._accumulate_usage(executor)

    def attribute() -> None:
        barrier.wait()
        _attribute_sub_usage(agent, 0.5, 50, 1)

    threads = [
        threading.Thread(target=bank, args=(executor,)) for executor in executors
    ] + [threading.Thread(target=attribute) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_WAIT)
        assert not t.is_alive()
    assert agent.usage_snapshot() == (6.0, 600, 16)


def test_double_bank_of_same_executor_counts_once() -> None:
    """The except-BaseException re-bank of a session must be a no-op."""
    agent = RelentlessAgent("t")
    executor = _executor_with_spend("s0", 1.25, 100, 3)
    # First bank: the success path of the session try block.
    agent._accumulate_usage(executor)
    # Second bank: what the ``except BaseException`` handler does after
    # a stop-injected KeyboardInterrupt unwound the first call.
    agent._accumulate_usage(executor)
    assert agent.usage_snapshot() == (1.25, 100, 3)
    assert len(_session_records(agent, executor)) == 1


def test_distinct_executors_still_accumulate() -> None:
    """Idempotency is per session; separate sessions all bank."""
    agent = RelentlessAgent("t")
    agent._accumulate_usage(_executor_with_spend("s0", 1.0, 10, 1))
    agent._accumulate_usage(_executor_with_spend("s1", 2.0, 20, 2))
    assert agent.usage_snapshot() == (3.0, 30, 3)


def test_concurrent_double_bank_races_count_once() -> None:
    """Racing bank attempts for ONE executor never double-count.

    The reclaim path (``reclaim_abandoned_subagents``) runs on server
    threads while the agent thread ends a session; racing bankers can
    each append a record, but they share one retry-stable session key,
    so readers count the session once.
    """
    agent = RelentlessAgent("t")
    executor = _executor_with_spend("s0", 0.5, 50, 5)
    barrier = threading.Barrier(8)

    def bank() -> None:
        barrier.wait()
        agent._accumulate_usage(executor)

    threads = [threading.Thread(target=bank) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_WAIT)
        assert not t.is_alive()
    assert agent.usage_snapshot() == (0.5, 50, 5)


def test_zero_delta_attribution_appends_nothing() -> None:
    """A zero delta (an empty classifier fold) grows no ledger."""
    agent = RelentlessAgent("t")
    agent._attribute_usage(1.0, 10, 1)
    before = len(agent._usage_events())
    agent._attribute_usage(0.0, 0, 0)
    assert len(agent._usage_events()) == before
    assert agent.usage_snapshot() == (1.0, 10, 1)


def test_duplicate_ledger_records_count_once() -> None:
    """Read-side dedup makes a duplicate session record a no-op.

    Exactly-once banking does NOT depend on the pre-append scan in
    ``_accumulate_usage`` (two racers can both pass it): duplicates of
    one session share the retry-stable key and the FIRST record wins on
    read.  This pins the read-side guarantee deterministically.
    """
    agent = RelentlessAgent("t")
    executor = _executor_with_spend("s0", 1.25, 100, 3)
    agent._accumulate_usage(executor)
    key = executor.__dict__["_usage_session_key"]
    # Exactly the record a fast-path-bypassing racer would append.
    agent._usage_events().append(_UsageEvent(key, 0, 1.25, 100, 3))
    assert agent.usage_snapshot() == (1.25, 100, 3)
    assert len(_session_records(agent, executor)) == 2


def test_property_setters_overwrite_coherently() -> None:
    """Absolute counter overwrites derive exact deltas from the ledger."""
    agent = RelentlessAgent("t")
    agent._attribute_usage(1.5, 110, 4)
    agent.budget_used = 5.0
    agent.total_tokens_used = 7
    agent.total_steps = 2
    assert agent.usage_snapshot() == (5.0, 7, 2)
    before = len(agent._usage_events())
    # Overwriting with the current value appends nothing.
    agent.budget_used = 5.0
    agent.total_tokens_used = 7
    agent.total_steps = 2
    assert len(agent._usage_events()) == before
    assert agent.usage_snapshot() == (5.0, 7, 2)


def test_reset_clears_banked_marks(tmp_path: Path) -> None:
    """A fresh run must bank fresh executors even after many banks."""
    agent = RelentlessAgent("t")
    executor = _executor_with_spend("s0", 1.0, 10, 1)
    agent._accumulate_usage(executor)
    agent._reset(
        model_name="m", max_sub_sessions=1, max_steps=1,
        max_budget=1.0, work_dir=str(tmp_path / "w"), docker_image=None,
    )
    assert agent.usage_snapshot() == (0.0, 0, 0)
    # After _reset the new epoch has no record for the SAME executor,
    # so a (hypothetically reused) session banks again into zero.
    agent._accumulate_usage(executor)
    assert agent.usage_snapshot() == (1.0, 10, 1)


class _RecordingPrinter(Printer):
    """Real Printer subclass that records every ``print`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Record the call; the merged-result kwargs are asserted on."""
        self.calls.append((type, dict(kwargs)))
        return ""

    def token_callback(self, token: str) -> None:
        """Unused by the merged-result emit path."""

    def reset(self) -> None:
        """Unused by the merged-result emit path."""


def test_emit_merged_result_event_coherent_under_banking_storm() -> None:
    """The merged-result emit reads one coherent triple mid-storm.

    Round-3 finding 2: ``_emit_merged_result_event`` read
    ``total_steps``, ``total_tokens_used`` and ``budget_used``
    separately, so a bank landing between two of the reads produced an
    impossible mix in the emitted Result event.  Every banked unit here
    is (1.0, 100, 3), so any coherent read satisfies
    ``tokens == 100*k``, ``cost == k`` and ``steps == 3*k`` for one k;
    a torn read breaks the proportion.
    """
    agent = RelentlessAgent("emitter")
    printer = _RecordingPrinter()
    agent.printer = printer  # type: ignore[assignment]
    stop = threading.Event()

    def storm() -> None:
        i = 0
        while not stop.is_set():
            agent._accumulate_usage(
                _executor_with_spend(f"storm-{i}", 1.0, 100, 3)
            )
            i += 1

    bankers = [threading.Thread(target=storm) for _ in range(4)]
    for banker in bankers:
        banker.start()
    try:
        for _ in range(50):
            agent._emit_merged_result_event(
                {"success": True, "is_continue": False, "summary": "s"}
            )
    finally:
        stop.set()
        for banker in bankers:
            banker.join(timeout=_WAIT)
            assert not banker.is_alive()

    assert len(printer.calls) == 50
    for type_, kwargs in printer.calls:
        assert type_ == "result"
        tokens = kwargs["total_tokens"]
        steps = kwargs["step_count"]
        cost = float(str(kwargs["cost"]).lstrip("$"))
        assert tokens % 100 == 0, (tokens, cost, steps)
        k = tokens // 100
        assert steps == 3 * k, (tokens, cost, steps)
        assert abs(cost - float(k)) < 1e-9, (tokens, cost, steps)
