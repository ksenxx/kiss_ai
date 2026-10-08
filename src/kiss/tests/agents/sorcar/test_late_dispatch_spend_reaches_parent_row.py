# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``run_agent`` job settling after its caller's row was saved still charges the caller.

Found by the 2026-10-07 cost audit.  ``SorcarAgent.run``'s ``finally``
stops every leftover ``run_agent`` job (``kill_jobs_of``) but joins the
job threads for only ``_JOB_END_GRACE_SECONDS``; a sub-task whose stop
takes longer to confirm settles its spend on its own thread AFTER
``ChatSorcarAgent.run`` persisted the caller's ``task_history`` row.
The fold then landed only in the caller's in-memory ledger, which
nothing reads again: the caller's row (and so the History and Spend
panels, whose totals are the top-level rows') omitted the sub-task's
spend, and no event ever carried it.

``_attribute_dispatch_usage`` now charges a caller with a persisted row
like a side channel's task: a finished row gets the spend added (with
a persisted ``usage_info`` carrying its new totals); an unfinished one
banks it on the live ledger and broadcasts its live totals, so the
chat header follows every fold.

End to end: a real :class:`ChatSorcarAgent` (real SQLite history in an
isolated ``KISS_HOME``) runs against a local chat-completions server
whose model calls ``run_agent(wait="false")`` against a local-WSS
daemon stand-in that confirms the stop only after the end-of-run
grace, then finishes.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent, persistence
from kiss.agents.sorcar.agent_dispatch import make_agent_job_tool, make_run_agent_tool
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.agents.sorcar.test_dispatch_timeout import (
    _SlowFinishDaemon,
    _StopConfirmingDaemon,
)
from kiss.tests.agents.sorcar.test_leftover_job_spend_usage_event import (
    _CHEAP,
    _HELPER_SEA,
    _RunAgentThenFinishHandler,
    _send_response,
)
from kiss.tests.core.test_budget_enforcement_e2e import (
    _read_body,
    _start_server,
    _tool_call_response,
)
from kiss.tests.server.parallel_agent_harness import CapturePrinter, IsolatedKissHome

_STOPPED_COST = 1.0842
_STOPPED_TOKENS = 4321
_STOPPED_STEPS = 7


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB."""
    isolated = IsolatedKissHome("kiss-late-dispatch-spend-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _WaitingRunAgentHandler(_RunAgentThenFinishHandler):
    """Model script: a blocking ``run_agent`` once, then ``finish``."""

    def do_POST(self) -> None:  # noqa: N802
        if type(self).requests == 0:
            type(self).requests += 1
            arguments = json.dumps({"task": "finishes soon", "agent": type(self).script})
            stream = bool(json.loads(_read_body(self) or "{}").get("stream", False))
            _send_response(self, _tool_call_response("run_agent", arguments, *_CHEAP), stream)
            return
        super().do_POST()


def _row(task_id: str) -> dict[str, Any]:
    db = persistence._get_db()
    with persistence._rw_lock.read_lock():
        row = db.execute(
            "SELECT cost, tokens, steps, end_ts FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _usage_events(task_id: str) -> list[dict[str, Any]]:
    """The persisted ``usage_info`` / ``result`` events of *task_id*, in order."""
    persistence._flush_chat_events(task_id)
    loaded = persistence._load_chat_events_by_task_id(task_id)
    assert loaded is not None
    events = loaded["events"]
    assert isinstance(events, list)
    return [e for e in events if e.get("type") in ("usage_info", "result")]


def _run(env: IsolatedKissHome, printer: CapturePrinter, model_url: str) -> ChatSorcarAgent:
    """Run one persisted ``ChatSorcarAgent`` session against the scripted model."""
    agent = ChatSorcarAgent("late-dispatch-parent")
    agent.run(
        tools=[make_run_agent_tool(str(env.repo), agent), make_agent_job_tool(agent)],
        model_name="gpt-4o-mini",
        prompt_template="dispatch and finish",
        max_steps=4,
        max_budget=5.0,
        append_basic_tools=False,
        use_memory=False,
        web_tools=False,
        work_dir=str(env.repo),
        printer=printer,
        verbose=False,
        model_config={"base_url": model_url, "api_key": "test-key"},
    )
    return agent


def _wait_for_row_cost(task_id: str, at_least: float, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        row = _row(task_id)
        if row["cost"] is not None and row["cost"] >= at_least or time.monotonic() > deadline:
            return row
        time.sleep(0.05)


def test_job_settling_after_the_row_was_saved_is_added_to_the_row(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stop confirms after the grace: the saved row gains the spend, with an event.

    The end-of-run grace is cut to 0.2 s and the stand-in holds the
    stop's confirmation until the test has seen the saved row, so
    ``kill_jobs_of`` returns, the row is saved without the child's
    spend, and only then does the job thread settle.
    """
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
    monkeypatch.setattr(agent_dispatch, "_JOB_END_GRACE_SECONDS", 0.2)
    confirm_gate = threading.Event()
    daemon = _StopConfirmingDaemon(confirm_delay=0.5, confirm_gate=confirm_gate, stopped_result={
        "type": "result", "taskId": "task-stopped-1", "success": False,
        "text": "Task stopped", "cost": f"${_STOPPED_COST:.4f}",
        "total_tokens": _STOPPED_TOKENS, "step_count": _STOPPED_STEPS,
    })
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = env.repo / "helper.py"
    script.write_text(_HELPER_SEA)
    _RunAgentThenFinishHandler.requests = 0
    _RunAgentThenFinishHandler.script = str(script)
    srv, url = _start_server(_RunAgentThenFinishHandler)
    printer = CapturePrinter()
    try:
        agent = _run(env, printer, url)
        task_id = agent.last_task_id
        saved = _row(task_id)
        # The run ended before the stop confirmed: the row holds the
        # caller's own spend only, and is finished.
        assert saved["end_ts"], saved
        assert 0 < saved["cost"] < _STOPPED_COST, saved
        own_cost, own_tokens, own_steps = saved["cost"], saved["tokens"], saved["steps"]
        assert daemon.wait_for_command("stop"), "the leftover job was not stopped"
        confirm_gate.set()
        row = _wait_for_row_cost(task_id, own_cost + _STOPPED_COST - 1e-6)
    finally:
        confirm_gate.set()
        agent_dispatch.kill_jobs_of(None)
        srv.shutdown()
        daemon.close()
    assert row["cost"] == pytest.approx(own_cost + _STOPPED_COST, abs=1e-6), row
    assert row["tokens"] == own_tokens + _STOPPED_TOKENS
    assert row["steps"] == own_steps + _STOPPED_STEPS
    # The replayed transcript ends with the row's totals.
    last = _usage_events(task_id)[-1]
    assert last["type"] == "usage_info", last
    assert last["cost"] == f"${row['cost']:.4f}"
    assert last["total_tokens"] == row["tokens"]
    assert last["total_steps"] == row["steps"]
    # The old in-memory fold path was not taken: the finished run's
    # ledger still holds the caller's own spend only.
    assert agent.usage_snapshot()[0] == pytest.approx(own_cost, abs=1e-6)


def test_child_finishing_during_the_run_publishes_the_folded_totals(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocking ``run_agent`` whose child finishes banks the spend live, with an event.

    The fold lands on the caller's live ledger (the row is unfinished),
    a ``usage_info`` with the folded totals follows at once, and the
    final row equals the caller's own spend plus the child's.
    """
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
    daemon = _SlowFinishDaemon(delay=0.2)  # result: $0.0100 / 5 tokens / 1 step
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = env.repo / "helper.py"
    script.write_text(_HELPER_SEA)
    _WaitingRunAgentHandler.requests = 0
    _WaitingRunAgentHandler.script = str(script)
    srv, url = _start_server(_WaitingRunAgentHandler)
    printer = CapturePrinter()
    try:
        agent = _run(env, printer, url)
    finally:
        agent_dispatch.kill_jobs_of(None)
        srv.shutdown()
        daemon.close()
    task_id = agent.last_task_id
    row = _row(task_id)
    budget, tokens, steps = agent.usage_snapshot()
    assert row["cost"] == pytest.approx(budget, abs=1e-6)
    assert budget > 0.01 and tokens >= 5 + 20 and steps >= 1 + 2, (budget, tokens, steps)
    events = _usage_events(task_id)
    # The fold's own usage_info: the first event carrying the child's
    # $0.01 comes right after the dispatch step's usage_info, before
    # the finish step's, and the result's cost equals the row's.
    costs = [float(str(e["cost"]).lstrip("$")) for e in events]
    folded = next(i for i, c in enumerate(costs) if c >= 0.01)
    assert events[folded]["type"] == "usage_info", events[folded]
    assert events[folded].get("text", "") == "", events[folded]
    assert folded >= 1 and events[folded - 1]["type"] == "usage_info"
    # The result and the run's trailing totals both carry the row's cost.
    assert [e["type"] for e in events[-2:]] == ["result", "usage_info"], events[-2:]
    assert events[-2]["cost"] == events[-1]["cost"] == f"${row['cost']:.4f}"


def test_charge_racing_the_final_save_is_never_lost(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A charge landing while the row is being finalized reaches the row or the ledger it reads.

    ``KISS_RACE_DELAY`` widens the final save's snapshot-to-write window
    and the charge's row-check-to-bank window to 100 ms each; a
    charging thread fires through the printer bridge the moment the
    run's trailing ``usage_info`` goes out (the final save follows at
    once).  ``TASK_USAGE_LOCK`` serializes the two, so the row holds
    the caller's spend plus the charge whichever side wins: a charge
    that lost the lock finds the finished row and adds to it (with a
    persisted ``usage_info`` as the task's last word); one that won it
    lands on the ledger before the save reads it (its live
    ``usage_info`` is persisted through the server's agent registry,
    which this standalone run has no entry in).
    """
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
    monkeypatch.setenv("KISS_RACE_DELAY", "0.1")

    class _FinishOnly(_RunAgentThenFinishHandler):
        requests = 1  # skip the dispatch reply

    srv, url = _start_server(_FinishOnly)
    printer = CapturePrinter()
    agent = ChatSorcarAgent("late-dispatch-race")
    charged = threading.Event()

    def charge_when_the_run_ends() -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            with printer._capture_lock:
                types = [e.get("type") for e in printer.captured]
            if "result" in types and types[-1] == "usage_info":
                break
            time.sleep(0.001)
        printer.charge_task_usage(agent, agent.last_task_id, 2.0, 20, 2)
        charged.set()

    charger = threading.Thread(target=charge_when_the_run_ends, daemon=True)
    charger.start()
    try:
        agent.run(
            tools=[],
            model_name="gpt-4o-mini",
            prompt_template="just finish",
            max_steps=4,
            max_budget=5.0,
            append_basic_tools=False,
            use_memory=False,
            web_tools=False,
            work_dir=str(env.repo),
            printer=printer,
            verbose=False,
            model_config={"base_url": url, "api_key": "test-key"},
        )
        assert charged.wait(30.0), "the charge never fired"
        charger.join(5.0)
    finally:
        srv.shutdown()
    task_id = agent.last_task_id
    row = _row(task_id)
    budget, tokens, steps = agent.usage_snapshot()
    # The charge either won the lock (banked on the ledger before the
    # save read it) or lost it (added to the finished row); the row
    # holds the caller's own spend plus the charge either way.
    own = budget - 2.0 if budget >= 2.0 else budget
    assert 0 < own < 0.01, (budget, tokens, steps)
    assert row["cost"] == pytest.approx(own + 2.0, abs=1e-6), (row, budget)
    assert row["tokens"] >= 20 and row["steps"] >= 2, row
    if budget < 2.0:
        # Finished-row path: its persisted usage_info is the last word.
        last = _usage_events(task_id)[-1]
        assert last["type"] == "usage_info" and last["text"] == "", last
        assert last["cost"] == f"${row['cost']:.4f}", (last, row)
