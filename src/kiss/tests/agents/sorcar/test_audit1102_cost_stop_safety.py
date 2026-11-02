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
BaseException`` that lets the stop propagate), which the key-less bridge
must count once; and the retry of a final save that did commit must not
replace the row's totals, since a late charge may have been added to the
finished row in between.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import chat_sorcar_agent as chat_module
from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent, _retried_cleanup_step
from kiss.agents.sorcar.coding_gate import CodingGate, run_on_host
from kiss.agents.sorcar.relentless_agent import _ledger_totals
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _ClassifierSpend
from kiss.agents.sorcar.task_classifier import TaskClassification
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve, text_body
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
    printer: CapturePrinter,
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
        ) -> None:
            charges.append((task_agent, task_id, budget, tokens, steps, epoch))

    run = SorcarAgent("audit1102-gate-bridge")
    run.printer = _Printer()  # type: ignore[assignment]
    epoch = run._usage_epoch()
    without_row = run._new_coding_gate(with_reviewer=False)
    run.last_task_id = "row-1"  # type: ignore[attr-defined]
    with_row = run._new_coding_gate(with_reviewer=False)
    _charge_of(with_row)(0.25, 250, 1, "finish-review:a")
    assert charges == [(run, "row-1", 0.25, 250, 1, epoch)]
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

    def fold(self: Any) -> None:
        seen.append(sys._getframe(1).f_code.co_name)
        if seen[-1] != "_retried_cleanup_step" or seen.count("_retried_cleanup_step") == 1:
            raise KeyboardInterrupt("stop injection")
        real(self)

    monkeypatch.setattr(SorcarAgent, "_fold_classifier_usage", fold)
    agent = _zero_session_run(env, CapturePrinter(), seed_classifier=True)
    assert seen.count("_retried_cleanup_step") == 2, seen
    assert agent._classifier_spend is None  # consumed by the fold that completed
    row = _row(agent.last_task_id)
    assert row["end_ts"] > 0
    assert (row["cost"], row["tokens"], row["steps"]) == (0.25, 123, 2)


# --- round 2: one key counts once through the bridge; a committed save is not replaced ---


def test_a_stopped_review_charged_twice_with_one_key_takes_the_bridge_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``review_verdict`` charges a stopped attempt in its ``finally`` and again before the
    stop propagates; through the key-less bridge the key must still count once."""
    from kiss.core.kiss_agent import KISSAgent

    charges: list[tuple[float, int, int]] = []

    class _Printer:
        def charge_task_usage(
            self,
            task_agent: Any,
            task_id: str,
            budget: float,
            tokens: int,
            steps: int,
            epoch: Any = None,
        ) -> None:
            charges.append((budget, tokens, steps))

    real_run = KISSAgent.run

    def run(self: KISSAgent, *args: Any, **kwargs: Any) -> str:
        result: str = real_run(self, *args, **kwargs)
        if self.name == "Finish reviewer":
            raise KeyboardInterrupt("stop")  # the stop event, seen as the call ended
        return result

    monkeypatch.setattr(KISSAgent, "run", run)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "hello.py").write_text("print('hello')\n")
    agent = SorcarAgent("audit1102-bridge-once")
    agent.printer = _Printer()  # type: ignore[assignment]
    agent.last_task_id = "row-1"  # type: ignore[attr-defined]
    charge = functools.partial(  # what _new_coding_gate binds
        agent._charge_gate_usage, epoch=agent._usage_epoch(), task_id="row-1"
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
    assert len(charges) == 1  # the bridge saw the key once
    assert charges[0][0] > 0 and charges[0][1] > 0 and charges[0][2] == 1


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
