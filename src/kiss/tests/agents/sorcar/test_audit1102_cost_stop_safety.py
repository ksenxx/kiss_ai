# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cost audit of 2026-11-02: gate-review charges bound to their run; cleanup steps retried.

Three findings of the review of the 2026-11-02 cost audit:

1. The finish gate's review runs on a thread of its own, and a Stop (an
   injected ``KeyboardInterrupt`` in ``CodingGate.run_stages``) lets the
   run end without waiting for it.  The review's charge
   (``SorcarAgent._charge_gate_usage``) then landed on the agent's
   *current* ledger epoch — the next task's after a ``reset_usage`` —
   or in a ledger nothing reads again once the row's final save ran.
   The gate's charge is now bound to the run's epoch and persisted row
   when the gate is built, and a run with a row goes through the
   printer's late-charge bridge (``charge_task_usage``) like a
   dispatched sub-task's spend: a finished row gets the spend added,
   with a persisted ``usage_info`` carrying its new totals.
2. and 3. ``ChatSorcarAgent.run``'s cleanup caught a second Stop
   injection landing in the keyed classifier fold or in the row's final
   save and went on without them: the row kept cost 0 (a run that
   persists its own row) or the classifier's spend stayed unbanked.
   Both idempotent steps are retried once (``_retried_cleanup_step``).

The review of the fixes (round 2) added two cases: the gate charges a
stopped attempt twice with one key (its ``finally`` and the ``except
BaseException`` that lets the stop propagate), which the bridge must
count once; and the retry of a final save that did commit must not
replace the row's totals, since a late charge may have been added to the
finished row in between.

The audit of 2026-11-09 (S11 F1) found the first case's guard losing a
charge: the key was claimed before the bridge ran, so a Stop landing in
between made the gate's second charge a no-op.  The key now travels
through the bridge and counts once where the spend is committed (the
ledger record, the finished row's retained charge), wherever a Stop
cuts the first charge.  The review of that fix (R4) found the bridge
adding a key to a row whose final save had already written it from the
ledger; only the gate's own look at the ledger hid it.  The bridge asks
the ledger itself now.
"""

from __future__ import annotations

import functools
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar import chat_sorcar_agent as chat_module
from kiss.agents.sorcar import persistence, sorcar_agent
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent, _retried_cleanup_step
from kiss.agents.sorcar.coding_gate import CodingGate, run_on_host
from kiss.agents.sorcar.relentless_agent import _ledger_totals
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _ClassifierSpend
from kiss.agents.sorcar.task_classifier import TaskClassification
from kiss.core.locked import locked_call
from kiss.server import agent_state, task_update
from kiss.server.agent_state import AgentState
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    text_body,
    tool_call_body,
)
from kiss.tests.lock_injection import inject
from kiss.tests.server.parallel_agent_harness import CapturePrinter, IsolatedKissHome

SEEDED = _ClassifierSpend("classifier:seeded", 0.25, 123, 2)


def _charge_of(gate: Any) -> Callable[[float, int, int, str], None]:
    """The gate's charge callback (built with one by ``SorcarAgent._new_coding_gate``)."""
    assert gate.charge is not None
    return gate.charge  # type: ignore[no-any-return]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB; no live classifier."""
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
    isolated = IsolatedKissHome("kiss-audit1102-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _row(task_id: str) -> dict[str, Any]:
    db = persistence._get_db()
    with persistence._rw_lock.read_lock():
        row = db.execute(
            "SELECT cost, tokens, steps, end_ts FROM task_history WHERE id = ?",
            (task_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _persisted_usage(task_id: str) -> list[tuple[str, float]]:
    """``(type, cost_usd)`` of the persisted ``usage_info`` / ``result`` events, in order."""
    persistence._flush_chat_events(task_id)
    loaded = persistence._load_chat_events_by_task_id(task_id)
    assert loaded is not None
    events = loaded["events"]
    assert isinstance(events, list)
    return [
        (e["type"], e.get("cost_usd")) for e in events if e.get("type") in ("usage_info", "result")
    ]


def _zero_session_run(
    env: IsolatedKissHome,
    printer: CapturePrinter | None,
    *,
    seed_classifier: bool = False,
    agent: ChatSorcarAgent | None = None,
) -> ChatSorcarAgent:
    """Run a persisted ``ChatSorcarAgent`` with no session: setup, cleanup and the row only.

    With *seed_classifier* the pre-run classifier already spent
    :data:`SEEDED`, as a verdict the task runner pre-seeded
    (``classify_task_for_run``), so the fold at cleanup has something to bank.
    """
    if agent is None:
        agent = ChatSorcarAgent("audit1102-chat")
    if seed_classifier:
        agent._classification_attempted = True
        agent._task_classification = TaskClassification(True)
        agent._classifier_spend = SEEDED
        agent._classification_preseeded = True
    with pytest.raises(BaseException):  # noqa: B017, PT011 — the outcome is under test
        agent.run(
            model_name="gpt-4o-mini",
            prompt_template="noop",
            max_sub_sessions=0,
            max_budget=5.0,
            append_basic_tools=False,
            use_memory=False,
            web_tools=False,
            work_dir=str(env.repo),
            printer=printer,
            verbose=False,
        )
    return agent


# --- finding 1: the gate's charge is the run's -------------------------------------------


def test_gate_charge_lands_in_the_epoch_of_the_run_that_built_the_gate() -> None:
    """A review charged after the agent moved on to its next task charges the run it reviewed."""
    run = SorcarAgent("audit1102-gate-epoch")
    epoch = run._usage_epoch()
    gate = run._new_coding_gate(with_reviewer=False)
    run.reset_usage()  # the reusable agent began its next task
    _charge_of(gate)(0.5, 500, 1, "finish-review:late")
    _charge_of(gate)(0.5, 500, 1, "finish-review:late")  # a retried charge counts once
    assert _ledger_totals(epoch) == (0.5, 500, 1)
    assert run.usage_snapshot() == (0.0, 0, 0)


def test_gate_charge_of_a_run_with_a_row_goes_through_the_printers_bridge() -> None:
    """With a persisted row and a server printer the charge takes the late-charge bridge."""
    charges: list[tuple[Any, ...]] = []

    class _Printer:
        def charge_task_usage(
            self,
            task_agent: Any,
            task_id: str,
            budget: float,
            tokens: int,
            steps: int,
            epoch: Any = None,
            key: str | None = None,
        ) -> None:
            charges.append((task_agent, task_id, budget, tokens, steps, epoch, key))

    run = SorcarAgent("audit1102-gate-bridge")
    run.printer = _Printer()  # type: ignore[assignment]
    epoch = run._usage_epoch()
    without_row = run._new_coding_gate(with_reviewer=False)
    run.last_task_id = "row-1"  # type: ignore[attr-defined]
    with_row = run._new_coding_gate(with_reviewer=False)
    _charge_of(with_row)(0.25, 250, 1, "finish-review:a")
    assert charges == [(run, "row-1", 0.25, 250, 1, epoch, "finish-review:a")]
    assert run.usage_snapshot() == (0.0, 0, 0)  # the bridge owns the bank
    _charge_of(without_row)(0.75, 750, 1, "finish-review:b")  # no row: the ledger, keyed
    assert len(charges) == 1
    assert run.usage_snapshot() == (0.75, 750, 1)


def test_a_review_charged_after_the_rows_final_save_reaches_the_row(
    env: IsolatedKissHome,
) -> None:
    """End to end: a late gate charge adds to the finished row and persists a ``usage_info``.

    The run's gate is charged after the row was saved and the agent's
    ledger reset for its next task (the Stop-abandoned review thread's
    timing): the row and its replayed transcript carry the review's
    spend, the next task's ledger does not.
    """
    printer = CapturePrinter()
    agent = _zero_session_run(env, printer)
    task_id = agent.last_task_id
    before = _row(task_id)
    assert before["end_ts"] > 0
    gate = agent._coding_gate
    agent.reset_usage()
    _charge_of(gate)(0.5, 500, 1, "finish-review:late")
    after = _row(task_id)
    assert after["cost"] == pytest.approx(before["cost"] + 0.5)
    assert after["tokens"] == before["tokens"] + 500
    assert after["steps"] == before["steps"] + 1
    assert _persisted_usage(task_id)[-1] == ("usage_info", pytest.approx(after["cost"]))
    assert agent.usage_snapshot() == (0.0, 0, 0)


def test_a_review_charged_after_the_rows_final_save_reaches_the_row_headless(
    env: IsolatedKissHome,
) -> None:
    """A CLI run (a persisted row, no server printer) gets the daemon's late-charge policy.

    The ledger-only fallback left a stopped headless task's late review
    spend out of its finished row (audit 2026-11-04, S2 F2): the row and
    its transcript now get it, the stop's second charge of the attempt
    counts once, and the next task's ledger stays clean.
    """
    agent = _zero_session_run(env, None)
    task_id = agent.last_task_id
    before = _row(task_id)
    assert before["end_ts"] > 0
    gate = agent._coding_gate
    agent.reset_usage()
    _charge_of(gate)(0.5, 500, 1, "finish-review:late")
    _charge_of(gate)(0.5, 500, 1, "finish-review:late")
    after = _row(task_id)
    assert after["cost"] == pytest.approx(before["cost"] + 0.5)
    assert after["tokens"] == before["tokens"] + 500
    assert after["steps"] == before["steps"] + 1
    assert _persisted_usage(task_id)[-1] == ("usage_info", pytest.approx(after["cost"]))
    assert agent.usage_snapshot() == (0.0, 0, 0)


def test_a_headless_review_charged_while_the_row_is_unfinished_is_in_its_final_save(
    env: IsolatedKissHome,
) -> None:
    """End to end, no printer: a real review charged mid-run lands in the row's final totals.

    The row is unfinished when the review's charge arrives, so the spend
    is banked on the run's ledger, which the final save reads: the row
    carries every call the scripted endpoint answered, the review's
    among them (the gate refuses the unchecked finish; every later call,
    the review's included, gets the repeated 5000-token answer).  The
    charge's failure branch (a database error under the lock) is not
    reachable with a real database and is left untested.
    """
    env.write_config(coding_gate_reviewer=MODEL, classify_tasks=False, use_memory=False)
    script = [
        tool_call_body(
            "Write", {"file_path": str(env.repo / "hello.py"), "content": "print(42)\n"}, 500
        ),
        finish_body("<p>Implemented hello.py.</p>"),
        text_body("No findings.\nVERDICT: OK", 4000, 1000),
    ]
    agent = ChatSorcarAgent("audit1102-headless-review")
    with serve(script) as (url, requests):
        agent.run(
            model_name=MODEL,
            model_config={"base_url": url, "api_key": "local"},
            prompt_template="Create hello.py then finish.",
            max_sub_sessions=1,
            max_budget=1,
            max_steps=5,
            tool_profile="shell+edit",
            web_tools=False,
            use_memory=False,
            work_dir=str(env.repo),
            verbose=False,
        )
    reviews = [b for b in requests if "You are reviewing" in str(b["messages"][-1]["content"])]
    assert len(reviews) == 1 and len(requests) > 2
    row = _row(agent.last_task_id)
    assert row["end_ts"] > 0
    assert (row["tokens"], row["steps"]) == (5700 + 5000 * (len(requests) - 2), len(requests))
    assert row["cost"] == pytest.approx(agent.usage_snapshot()[0]) and row["cost"] > 0


# --- findings 2 and 3: the cleanup's fold and final save complete under a second Stop ----


def test_retried_cleanup_step_retries_once_and_keeps_the_first_error() -> None:
    calls: list[int] = []

    def flaky(failures: int) -> Any:
        def action() -> None:
            calls.append(1)
            if len(calls) <= failures:
                raise KeyboardInterrupt(f"stop {len(calls)}")

        return action

    assert _retried_cleanup_step(None, "ok", flaky(0)) is None and len(calls) == 1
    calls.clear()
    first = _retried_cleanup_step(None, "once", flaky(1))
    assert isinstance(first, KeyboardInterrupt) and str(first) == "stop 1" and len(calls) == 2
    calls.clear()
    earlier = RuntimeError("earlier step")
    assert _retried_cleanup_step(earlier, "twice", flaky(2)) is earlier and len(calls) == 2


def _interrupt_first_call_from(
    monkeypatch: pytest.MonkeyPatch, cls: type, name: str, caller: str
) -> list[str]:
    """Make ``cls.name`` raise ``KeyboardInterrupt`` the first time *caller* calls it.

    Models the runner's second Stop injection landing inside that
    cleanup step; every other call runs the real method.  Returns the
    list of callers seen, for the test's checks.
    """
    real = getattr(cls, name)
    seen: list[str] = []

    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(sys._getframe(1).f_code.co_name)
        if seen[-1] == caller and seen.count(caller) == 1:
            raise KeyboardInterrupt("second stop injection")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, method)
    return seen


def test_a_stop_in_the_final_save_is_retried_so_the_row_is_finished(
    env: IsolatedKissHome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row's final usage save completes on its retry: end time and cost land."""
    seen = _interrupt_first_call_from(
        monkeypatch,
        ChatSorcarAgent,
        "_save_final_usage",
        "locked_call",
    )
    agent = _zero_session_run(env, CapturePrinter(), seed_classifier=True)
    assert seen.count("locked_call") == 2, seen
    row = _row(agent.last_task_id)
    assert row["end_ts"] > 0
    assert (row["cost"], row["tokens"], row["steps"]) == (0.25, 123, 2)


def test_a_stop_in_the_last_classifier_fold_is_retried_so_its_spend_is_banked(
    env: IsolatedKissHome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cleanup's keyed classifier fold completes on its retry and reaches the row.

    ``SorcarAgent``'s own cleanup fold is stopped too (the first
    injection), so the run's cleanup is the last fold the spend has.
    """
    real = SorcarAgent._fold_classifier_usage
    seen: list[str] = []

    def fold(self: Any, task_id: str | None = None) -> None:
        seen.append(sys._getframe(1).f_code.co_name)
        if seen[-1] != "_retried_cleanup_step" or seen.count("_retried_cleanup_step") == 1:
            raise KeyboardInterrupt("stop injection")
        real(self, task_id)

    monkeypatch.setattr(SorcarAgent, "_fold_classifier_usage", fold)
    agent = _zero_session_run(env, CapturePrinter(), seed_classifier=True)
    assert seen.count("_retried_cleanup_step") == 2, seen
    assert agent._classifier_spend is None  # consumed by the fold that completed
    row = _row(agent.last_task_id)
    assert row["end_ts"] > 0
    assert (row["cost"], row["tokens"], row["steps"]) == (0.25, 123, 2)


# --- round 2: one key counts once through the bridge; a committed save is not replaced ---


def _stop_at(code: Any, event: str, frame: Any, what: str, _arg: Any) -> Any:
    """Trace function (*code* and *event* bound): a Stop as *code* is entered or left."""
    if frame.f_code is not code:
        return None
    if what == event:
        raise KeyboardInterrupt("injected stop")
    return functools.partial(_stop_at, code, event)


def _finish(agent: ChatSorcarAgent, task_id: str) -> None:
    """The run's final save of its row."""
    with persistence.TASK_USAGE_LOCK:
        agent._save_final_usage({}, task_id, result_summary="Done")


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("with_printer", [True, False])
@pytest.mark.parametrize("finished", [False, True])
def test_a_stop_in_a_gate_charge_and_its_retry_charge_the_run_once(
    env: IsolatedKissHome, finished: bool, with_printer: bool, after_commit: bool
) -> None:
    """A Stop cuts a gate charge before its accounting or right after that committed, and
    the gate charges the attempt again with its key: the run is charged exactly once, on
    the ledger of a running row (which its final save writes) or in a finished row, through
    the printer's bridge and without one (audit 2026-11-09, S11 F1)."""
    if not after_commit:
        code, event = locked_call.__code__, "call"  # at the usage lock: nothing charged yet
    elif not finished:
        code, event = sorcar_agent._attribute_sub_usage.__code__, "return"  # on the ledger
    elif with_printer:
        code, event = task_update._publish_charge.__code__, "call"  # in the row
    else:
        code, event = sorcar_agent._append_late_usage_events.__code__, "call"  # in the row
    task_id, _chat = persistence._add_task("audit1102 stopped gate charge")
    agent = ChatSorcarAgent("audit1102-stopped-charge")
    agent.printer = CapturePrinter() if with_printer else None
    with agent._task_id_lock:
        agent._last_task_id = task_id
    sorcar_agent._attribute_sub_usage(agent, 1.0, 100, 2)
    charge = _charge_of(agent._new_coding_gate(with_reviewer=False))
    if finished:
        _finish(agent, task_id)
    tracer = sys.gettrace()
    sys.settrace(functools.partial(_stop_at, code, event))
    try:
        with pytest.raises(KeyboardInterrupt):
            charge(0.5, 500, 1, "finish-review:stopped")
    finally:
        sys.settrace(tracer)
    charge(0.5, 500, 1, "finish-review:stopped")
    if not finished:
        _finish(agent, task_id)
    row = _row(task_id)
    assert (row["cost"], row["tokens"], row["steps"]) == (1.5, 600, 3)
    assert agent.usage_snapshot() == ((1.0, 100, 2) if finished else (1.5, 600, 3))
    # The retry also publishes what the stopped charge committed but never
    # published: a replay (the UI) shows the row's cost (audit 2026-11-11).
    assert _persisted_usage(task_id)[-1][1] == 1.5


def test_a_retried_keyed_charge_publishes_nothing_when_its_first_charge_was_published(
    env: IsolatedKissHome,
) -> None:
    """A keyed charge made twice with nothing cut short persists its new totals once."""
    task_id, _chat = persistence._add_task("audit1111 keyed charge published once")
    agent = ChatSorcarAgent("audit1111-published-once")
    agent.printer = None
    with agent._task_id_lock:
        agent._last_task_id = task_id
    sorcar_agent._attribute_sub_usage(agent, 1.0, 100, 2)
    charge = _charge_of(agent._new_coding_gate(with_reviewer=False))
    _finish(agent, task_id)
    charge(0.5, 500, 1, "finish-review:once")
    before = _persisted_usage(task_id)
    charge(0.5, 500, 1, "finish-review:once")
    assert _persisted_usage(task_id) == before
    assert before[-1][1] == 1.5
    assert persistence._unpublished_keyed_charge(task_id, "finish-review:once") == ([], "", False)


def _park_at(
    code: Any, event: str, reached: threading.Event, gate: threading.Lock,
    frame: Any, what: str, _arg: Any,
) -> Any:
    """Trace function (all but the last three bound): the thread says it *reached* the
    entry or the return of *code* and waits there for the lock *gate*, where a Stop
    injected meanwhile is raised as the wait returns."""
    if frame.f_code is not code:
        return None
    if what == event:
        reached.set()
        gate.acquire(timeout=30)
    return functools.partial(_park_at, code, event, reached, gate)


def _charge_traced(charge: Callable[[], None], trace: Any, stops: list[BaseException]) -> None:
    """Thread target: *charge* under the trace function *trace*; a Stop that ends it is kept."""
    sys.settrace(trace)
    try:
        charge()
    except KeyboardInterrupt as stop:
        stops.append(stop)


@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.parametrize("route", ["bridge", "gate", "gate-cli"])
def test_a_keyed_charge_stopped_before_the_final_save_and_made_again_after_it_counts_once(
    env: IsolatedKissHome, route: str, banked: bool
) -> None:
    """A real Stop reaches a keyed charge of a running row as its bank on the ledger begins
    or has just returned; the row's final save follows, and only then does the caller make
    the charge again.  The finished row holds it once: in the ledger snapshot the save
    wrote, or as the second charge's late add, never both.  The bridge itself sees to that
    (review of the S11 F1 fix, R4), for its own callers and for the gate, which keeps its
    look at the ledger only for a run without the bridge (``gate-cli``)."""
    task_id, _chat = persistence._add_task("audit1102 keyed charge across the final save")
    agent = ChatSorcarAgent("audit1102-transition")
    agent.printer = CapturePrinter() if route == "gate" else None
    sorcar_agent._attribute_sub_usage(agent, 1.0, 100, 2)
    key = "finish-review:transition"
    if route == "bridge":
        charge = functools.partial(
            task_update.charge_side_channel_usage, None, agent, task_id, 0.5, 500, 1, key=key)
    else:
        charge = functools.partial(agent._charge_gate_usage, 0.5, 500, 1, key, task_id=task_id)
    bound = functools.partial(charge, epoch=agent._usage_epoch())  # as the gate charges
    reached, gate = threading.Event(), threading.Lock()
    stops: list[BaseException] = []
    park = functools.partial(
        _park_at, sorcar_agent._attribute_sub_usage.__code__, "return" if banked else "call",
        reached, gate,
    )
    worker = threading.Thread(target=_charge_traced, args=(bound, park, stops), daemon=True)
    with gate:
        worker.start()
        assert reached.wait(30)
        assert inject(worker) == 1  # the runner's Stop, pending when the thread goes on
    worker.join(30)
    assert len(stops) == 1 and not worker.is_alive()
    ledger = (1.5, 600, 3) if banked else (1.0, 100, 2)
    assert agent.usage_snapshot() == ledger
    _finish(agent, task_id)
    bound()
    charge()  # and once more bound to no epoch: the agent's current one, the same ledger
    row = _row(task_id)
    assert (row["cost"], row["tokens"], row["steps"]) == (1.5, 600, 3)
    assert agent.usage_snapshot() == ledger


@pytest.mark.parametrize("stopped", [False, True])
def test_a_gate_charge_of_a_finished_child_reaches_its_running_parent_once(
    env: IsolatedKissHome, stopped: bool
) -> None:
    """The finished child's row takes the charge and its running parent owes it: banked on
    the parent's live agent at once, or, when a Stop cut the charge before that bank, by
    the parent's final save; the gate's second charge of the key adds nothing."""
    parent_id, _chat = persistence._add_task("audit1102 running parent")
    parent = ChatSorcarAgent("audit1102-running-parent")
    with parent._task_id_lock:
        parent._last_task_id = parent_id
    state = AgentState(parent_id, agent=parent, tab_id="tab", server_owned=True)
    agent_state.register(state)
    try:
        sorcar_agent._attribute_sub_usage(parent, 2.0, 200, 4)
        child_id, _chat = persistence._add_task("audit1102 finished child")
        persistence._save_task_extra({"parent_task_id": parent_id}, task_id=child_id)
        child = ChatSorcarAgent("audit1102-finished-child")
        child.printer = CapturePrinter()
        with child._task_id_lock:
            child._last_task_id = child_id
        sorcar_agent._attribute_sub_usage(child, 1.0, 100, 2)
        charge = _charge_of(child._new_coding_gate(with_reviewer=False))
        _finish(child, child_id)
        if stopped:
            tracer = sys.gettrace()
            sys.settrace(functools.partial(_stop_at, task_update._publish_charge.__code__, "call"))
            try:
                with pytest.raises(KeyboardInterrupt):
                    charge(0.5, 500, 1, "finish-review:parent")
            finally:
                sys.settrace(tracer)
        else:
            charge(0.5, 500, 1, "finish-review:parent")
        charge(0.5, 500, 1, "finish-review:parent")
        row = _row(child_id)
        assert (row["cost"], row["tokens"], row["steps"]) == (1.5, 600, 3)
        assert parent.usage_snapshot() == ((2.0, 200, 4) if stopped else (2.5, 700, 5))
        # The child's replay shows its row's cost, published once with the
        # share its waiting parent must not fold again (audit 2026-11-11).
        persistence._flush_chat_events(child_id)
        loaded = persistence._load_chat_events_by_task_id(child_id)
        assert loaded is not None
        late = [e for e in cast("list[dict[str, Any]]", loaded["events"])
                if e.get("type") == "usage_info" and "ancestor_charged" in e]
        assert [(e["cost_usd"], e["ancestor_charged"]) for e in late] == [
            (1.5, {"cost": 0.5, "tokens": 500, "steps": 1}),
        ]
        assert _persisted_usage(child_id)[-1][1] == 1.5
        _finish(parent, parent_id)
    finally:
        agent_state.unregister(parent_id, state)
    row = _row(parent_id)
    assert (row["cost"], row["tokens"], row["steps"]) == (2.5, 700, 5)
    assert parent.usage_snapshot() == (2.5, 700, 5)


@pytest.mark.parametrize("cut", ["after_live_event", "before_publication_then_other_charge"])
def test_a_retried_keyed_charge_of_a_child_is_netted_once_by_its_waiting_parent(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch, cut: str
) -> None:
    """The retry of a keyed charge cut short in its publication publishes the share its
    running parent owes once per key: the ``run_agent`` caller waiting on the child nets
    the child's own spend, from the live events and from a replay (audit 2026-11-11)."""
    from kiss.agents.sorcar.daemon_client import _net_totals

    parent_id, _chat = persistence._add_task("audit1111 running parent")
    parent = ChatSorcarAgent("audit1111-running-parent")
    with parent._task_id_lock:
        parent._last_task_id = parent_id
    state = AgentState(parent_id, agent=parent, tab_id="tab", server_owned=True)
    agent_state.register(state)
    live: list[dict[str, Any]] = []
    try:
        sorcar_agent._attribute_sub_usage(parent, 2.0, 200, 4)
        child_id, _chat = persistence._add_task("audit1111 finished child")
        persistence._save_task_extra({"parent_task_id": parent_id}, task_id=child_id)
        child = ChatSorcarAgent("audit1111-finished-child")
        printer = CapturePrinter()
        child.printer = printer
        with child._task_id_lock:
            child._last_task_id = child_id
        sorcar_agent._attribute_sub_usage(child, 1.0, 100, 2)
        charge = _charge_of(child._new_coding_gate(with_reviewer=False))
        _finish(child, child_id)
        original = printer.broadcast_transient
        stop_once = [cut == "after_live_event"]

        def transient(event: dict[str, Any], **kwargs: Any) -> None:
            original(event, **kwargs)
            if event.get("taskId") == child_id:
                live.append(dict(event))
            if stop_once[0]:
                stop_once[0] = False
                raise KeyboardInterrupt("injected stop after the live event")

        monkeypatch.setattr(printer, "broadcast_transient", transient)
        tracer = sys.gettrace()
        if cut != "after_live_event":
            sys.settrace(functools.partial(_stop_at, task_update._publish_charge.__code__, "call"))
        try:
            with pytest.raises(KeyboardInterrupt):
                charge(0.5, 500, 1, "finish-review:a")
        finally:
            sys.settrace(tracer)
        if cut != "after_live_event":
            charge(0.25, 250, 1, "finish-review:b")  # publishes the row's totals first
        charge(0.5, 500, 1, "finish-review:a")
        charge(0.5, 500, 1, "finish-review:a")  # a third charge publishes nothing more
        _finish(parent, parent_id)
    finally:
        agent_state.unregister(parent_id, state)
    extra = 0.25 if cut != "after_live_event" else 0.0
    assert _row(parent_id)["cost"] == pytest.approx(2.5 + extra)
    persistence._flush_chat_events(child_id)
    loaded = persistence._load_chat_events_by_task_id(child_id)
    assert loaded is not None
    persisted = [e for e in cast("list[dict[str, Any]]", loaded["events"])
                 if e.get("type") in ("usage_info", "result")]
    assert sorted(e["charge_key"] for e in persisted if "charge_key" in e) == (
        ["finish-review:a"] if cut == "after_live_event"
        else ["finish-review:a", "finish-review:b"]
    )
    for events in (live, persisted):
        charged: dict[str, Any] = {"cost": 0.0, "tokens": 0, "steps": 0}
        totals: dict[str, Any] = {}
        for event in events:
            totals = _net_totals(event, charged)
        assert totals["cost"] == pytest.approx(1.0)  # the child's own spend
        assert totals["total_tokens"] == 100


def test_a_stopped_review_charged_twice_with_one_key_charges_the_run_once(
    env: IsolatedKissHome, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``review_verdict`` charges a stopped attempt in its ``finally`` and again before the
    stop propagates; through the printer's bridge the key still counts once."""
    from kiss.core.kiss_agent import KISSAgent

    real_run = KISSAgent.run

    def run(self: KISSAgent, *args: Any, **kwargs: Any) -> str:
        result: str = real_run(self, *args, **kwargs)
        if self.name == "Finish reviewer":
            raise KeyboardInterrupt("stop")  # the stop event, seen as the call ended
        return result

    monkeypatch.setattr(KISSAgent, "run", run)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "hello.py").write_text("print('hello')\n")
    task_id, _chat = persistence._add_task("audit1102 stopped review")
    agent = ChatSorcarAgent("audit1102-bridge-once")
    agent.printer = CapturePrinter()
    with agent._task_id_lock:
        agent._last_task_id = task_id
    charge = functools.partial(  # what _new_coding_gate binds
        agent._charge_gate_usage, epoch=agent._usage_epoch(), task_id=task_id
    )
    keys: list[str] = []

    def counted(budget: float, tokens: int, steps: int, key: str) -> None:
        keys.append(key)
        charge(budget, tokens, steps, key)

    def shell(command: str, timeout: int) -> tuple[int | None, str]:
        return run_on_host(command, str(tmp_path), timeout)

    with serve([text_body("Looking.\nVERDICT: OK")]) as (url, requests):
        gate = CodingGate(
            shell,
            str(tmp_path),
            "./tmp/",
            MODEL,
            counted,
            model_config={"base_url": url, "api_key": "local"},
        )
        with pytest.raises(KeyboardInterrupt):
            gate.review_verdict({"success": True, "summary_in_html": "<p>done</p>"}, "hello")
        assert len(requests) == 1
    assert len(keys) == 2 and keys[0] == keys[1], keys  # the gate's two charges, one key
    cost, tokens, steps = agent.usage_snapshot()  # the running row's ledger took it once
    assert cost > 0 and tokens > 0 and steps == 1


def test_the_retry_of_a_committed_final_save_keeps_a_late_charge_on_the_row(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Stop right after the first save's write: the retry leaves the finished row alone.

    Between the two attempts a late gate charge finds the finished row
    and adds to it; a retry that replaced the totals from the ledger
    would erase it (review of the fixes, finding 2).
    """
    agent = ChatSorcarAgent("audit1102-committed-save")
    real_save = ChatSorcarAgent._save_final_usage
    saves: list[bool] = []

    def save(self: Any, *args: Any, **kwargs: Any) -> None:
        saves.append(bool(kwargs.get("unless_finished")))
        real_save(self, *args, **kwargs)
        if len(saves) == 1:
            raise KeyboardInterrupt("stop at the save's return")

    real_first = chat_module._first_cleanup_error

    def late_charge_then_log(first: Any, step: str, exc: BaseException) -> BaseException:
        if step == "final save":  # between the attempts, the lock released
            _charge_of(agent._coding_gate)(0.5, 500, 1, "finish-review:late")
        return real_first(first, step, exc)

    monkeypatch.setattr(ChatSorcarAgent, "_save_final_usage", save)
    monkeypatch.setattr(chat_module, "_first_cleanup_error", late_charge_then_log)
    _zero_session_run(env, CapturePrinter(), seed_classifier=True, agent=agent)
    assert saves == [False, True]
    row = _row(agent.last_task_id)
    assert row["end_ts"] > 0
    assert (row["cost"], row["tokens"], row["steps"]) == (0.75, 623, 3)
    assert _persisted_usage(agent.last_task_id)[-1] == ("usage_info", pytest.approx(0.75))
