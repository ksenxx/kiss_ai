"""Cost audit 2026-11-04: a side call's spend is filed under the model it ran on.

A model call made for a task outside its agent loop (the commit message,
a ``talk`` synthesis, a ``decide`` call) emits no ``llm_call`` event; its
spend is charged to the task.  Each charge leaves a ``gate_stage``
record with ``stage`` ``charge`` (``model``, ``cost_usd``, ``tokens``)
under the task's row (``SorcarAgent._charge_side_call_usage`` ->
``persistence._record_model_charge``), which ``_own_model_costs``
counts, so the Spend-by-model split credits the call's model, also for
a call charged twice with one key and for one charged after the row's
final save.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _attribute_side_call_usage,
    _persisted_task_id,
)
from kiss.core.file_lock import lock_exclusive
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve
from kiss.tests.agents.sorcar.test_cost_audit_20261023 import (
    _by_model,
    _llm_call,
    _model_costs,
    _set_timestamp,
    history_db,  # noqa: F401  (the fixture)
)

FABLE = "claude-fable-5-1"
ASTRA = "gpt-6-astra"


def _charge_record(model: str, cost: float, tokens: int) -> dict[str, object]:
    return {"type": "gate_stage", "stage": "charge", "seconds": 0.0, "outcome": "charged",
            "model": model, "cost_usd": cost, "tokens": tokens, "key": f"k{cost}"}


def _review_record(model: str, cost: float) -> dict[str, object]:
    """The stage report the coding SEA's gate publishes: never counted (the charges are)."""
    return {"type": "gate_stage", "stage": "review", "seconds": 1.0, "outcome": "pass",
            "model": model, "cost_usd": cost, "tokens": 10, "round": 1, "findings": 0}


def _task(cost: float, tokens: int, events: list[dict[str, object]],
          finished: bool = True) -> str:
    extra: dict[str, object] = {"model": FABLE}
    task_id, _ = th._add_task(f"task {cost}", "", extra)
    for event in events:
        th._queue_chat_event(event, task_id)
    final: dict[str, object] = {"cost": cost, "tokens": tokens, "steps": 1}
    if finished:
        final["endTs"] = int(time.time() * 1000)
    th._flush_chat_events()
    th._save_task_extra(final, task_id=task_id)
    _set_timestamp(task_id, 1_790_000_000.0)
    return task_id


@pytest.mark.usefixtures("history_db")
class TestChargeRecords:
    def test_charge_records_are_filed_under_their_model(self) -> None:
        task = _task(5.0, 5_000, [_llm_call(FABLE, 4.0, 4_000),
                                  _charge_record(ASTRA, 0.75, 900),
                                  _charge_record(ASTRA, 0.25, 100),
                                  _review_record(ASTRA, 1.0)])
        assert json.loads(_model_costs(task)) == {FABLE: [4.0, 4_000], ASTRA: [1.0, 1_000]}
        assert _by_model() == {FABLE: (4.0, 4_000), ASTRA: (1.0, 1_000)}

    def test_a_charge_on_the_task_model_merges_with_its_calls(self) -> None:
        task = _task(3.0, 3_000, [_llm_call(ASTRA, 2.0, 2_000), _charge_record(ASTRA, 1.0, 1_000)])
        assert json.loads(_model_costs(task)) == {ASTRA: [3.0, 3_000]}

    def test_review_reports_and_free_records_move_nothing(self) -> None:
        task = _task(4.0, 4_000, [_llm_call(FABLE, 4.0, 4_000), _review_record(ASTRA, 2.0),
                                  _charge_record(ASTRA, 0.0, 0)])
        assert json.loads(_model_costs(task)) == {FABLE: [4.0, 4_000]}
        assert _by_model() == {FABLE: (4.0, 4_000)}

    def test_a_charge_after_the_final_save_refreshes_the_split(self) -> None:
        task = _task(4.5, 4_600, [_llm_call(FABLE, 4.0, 4_000)])
        assert json.loads(_model_costs(task)) == {FABLE: [4.0, 4_000]}
        th._record_model_charge(task, ASTRA, 0.5, 600, "late-key")
        assert json.loads(_model_costs(task)) == {FABLE: [4.0, 4_000], ASTRA: [0.5, 600]}
        assert _by_model() == {FABLE: (4.0, 4_000), ASTRA: (0.5, 600)}

    def test_a_retry_while_the_journalled_charge_awaits_replay_charges_once(self) -> None:
        # A11-1 (astra): a charge the database refused is journalled; a
        # retry of the same key while a peer holds the journal lock (the
        # replay is deferred) found nothing in ``events`` and queued the
        # charge again, and the replay then inserted the second copy.
        task = _task(4.5, 4_600, [_llm_call(FABLE, 4.0, 4_000)])
        charge = {**_charge_record(ASTRA, 0.5, 600), "key": "retry-key", "taskId": task}
        th._journal_failed_events(
            [(task, json.dumps(charge), time.time(), str(th._DB_PATH))], 4)
        peer = open(th._failed_events_path(str(th._DB_PATH)) + ".lock", "a+")  # noqa: SIM115
        assert lock_exclusive(peer, blocking=False)

        def peer_lets_go_once_the_retry_looked(statement: str) -> None:
            if "json_extract(event_json, '$.key')" in statement:
                peer.close()

        db = th._get_db()
        db.set_trace_callback(peer_lets_go_once_the_retry_looked)
        try:
            th._record_model_charge(task, ASTRA, 0.5, 600, "retry-key")
        finally:
            db.set_trace_callback(None)
            peer.close()
        for _attempt in range(5):
            th._replay_failed_events()
        copies = th._get_db().execute(
            "SELECT COUNT(*) FROM events WHERE task_id = ? "
            "AND json_extract(event_json, '$.key') = 'retry-key'", (task,),
        ).fetchone()[0]
        assert copies == 1
        assert json.loads(_model_costs(task)) == {FABLE: [4.0, 4_000], ASTRA: [0.5, 600]}

    def test_a_charge_before_the_final_save_is_in_its_split(self) -> None:
        task_id, _ = th._add_task("running", "", {"model": FABLE})
        th._record_model_charge(task_id, ASTRA, 0.5, 600, "early-key")
        assert _model_costs(task_id) == ""  # not finished: the final save computes it
        th._queue_chat_event(_llm_call(FABLE, 4.0, 4_000), task_id)
        th._flush_chat_events()
        th._save_task_extra({"cost": 4.5, "tokens": 4_600, "steps": 2,
                             "endTs": int(time.time() * 1000)}, task_id=task_id)
        assert json.loads(_model_costs(task_id)) == {FABLE: [4.0, 4_000], ASTRA: [0.5, 600]}


@pytest.mark.usefixtures("history_db")
def test_a_side_call_charged_twice_with_one_key_is_filed_under_its_model_once() -> None:
    """A call that a Stop ended is charged twice with one key: the finished row takes the
    spend once, through the daemon's bridge, and it is recorded once under the call's model."""
    row = _task(4.0, 4_000, [_llm_call(FABLE, 4.0, 4_000)])
    agent = SorcarAgent("audit1104-charged-twice")
    agent.printer = JsonPrinter()  # the daemon's late-charge bridge
    for _attempt in range(2):
        agent._charge_side_call_usage(0.5, 600, 1, "side:twice", epoch=agent._usage_epoch(),
                                      task_id=row, model=ASTRA)
    assert json.loads(_model_costs(row)) == {FABLE: [4.0, 4_000], ASTRA: [0.5, 600]}
    assert _row_cost(row) == pytest.approx(4.5)  # the row took it once


@pytest.mark.usefixtures("history_db")
def test_a_persisted_run_without_a_server_printer_files_a_side_call_too() -> None:
    """A standalone ChatSorcarAgent (no late-charge bridge) still records the call's model."""
    parent = ChatSorcarAgent("audit1104-standalone")
    task_id, _ = th._add_task("side call without a server printer", "", {"model": FABLE})
    parent._last_task_id = task_id
    parent.model_name = FABLE
    parent.printer = None
    parent._attribute_usage(4.0, 4_000, 1)
    _attribute_side_call_usage(parent, ASTRA, 0.5, 600)
    assert parent.usage_snapshot() == (4.5, 4_600, 1)  # banked on the running row's ledger
    parent._save_final_usage({}, task_id, "Say ok.", result_summary="Done.")
    assert json.loads(_model_costs(task_id))[ASTRA] == [0.5, 600]
    # the charge record is metadata: the standalone run still gets its synthesized replay
    session = th._load_chat_events_by_task_id(task_id)
    assert session is not None
    assert any(e.get("type") != "gate_stage" for e in session["events"])  # type: ignore[attr-defined]


def _run_ok(agent: Any, work_dir: Path) -> str:
    """One run of *agent* against a one-answer local model; its persisted row id."""
    with serve([finish_body("<p>ok</p>")]) as (url, _requests):
        agent.run(prompt_template="Say ok.", model_name=MODEL, max_steps=2, max_budget=1.0,
                  model_config={"base_url": url, "api_key": "local"}, use_memory=False,
                  web_tools=False, verbose=False, work_dir=str(work_dir))
    return str(_persisted_task_id(agent))


def _row_cost(task_id: str) -> float:
    """The cost the row *task_id* holds now, late charges included."""
    return float(th._task_usage([task_id])[task_id]["cost"])  # type: ignore[arg-type]


@pytest.mark.usefixtures("history_db")
def test_a_side_call_charge_key_counts_once_however_long_the_agent_is_reused(
    tmp_path: Path,
) -> None:
    """A finished row keeps the key of each charge it took, the agent none.

    A stop charges a call twice, so a key counts once.  The agent
    used to hold the keys it charged (every one, then those of its last
    two runs: audit 2026-11-07, A13_claude-8) and claimed a key before
    charging it, which lost a charge a stop cut (audit 2026-11-09, S11
    F1); the row's retained charge is the receipt now, however many
    runs the reused agent made since.
    """
    agent = ChatSorcarAgent("audit1104-charge-keys")
    row1 = _run_ok(agent, tmp_path)
    assert row1
    cost = _row_cost(row1)
    agent._charge_side_call_usage(0.5, 10, 1, "side:one", task_id=row1)
    agent._charge_side_call_usage(0.5, 10, 1, "side:one", task_id=row1)  # the stop's retry
    assert _row_cost(row1) == pytest.approx(cost + 0.5)
    row2 = _run_ok(agent, tmp_path)
    assert row2 and row2 != row1
    # One run later the key is still held: a call the stop left charging.
    agent._charge_side_call_usage(0.5, 10, 1, "side:one", task_id=row1)
    assert _row_cost(row1) == pytest.approx(cost + 0.5)
    _run_ok(agent, tmp_path)
    # Two runs later too: the row holds the key.
    agent._charge_side_call_usage(0.5, 10, 1, "side:one", task_id=row1)
    assert _row_cost(row1) == pytest.approx(cost + 0.5)


@pytest.mark.usefixtures("history_db")
def test_charge_records_are_not_transcript() -> None:
    task_id, _ = th._add_task("metadata only", "", {"model": FABLE})
    th._record_model_charge(task_id, ASTRA, 0.5, 600, "k1")
    th._record_model_charge(task_id, ASTRA, 0.5, 600, "k2")
    assert th._task_has_transcript_events(task_id) is False
    # A11-3 (astra): the settings record and the charges of an API run's
    # decide/talk calls stay metadata however many there are (a window
    # of 20 metadata rows used to count as a transcript, so the run got
    # no synthesized prompt/result and replayed blank).
    th._queue_chat_event({"type": "task_settings", "settings": {}}, task_id)
    for i in range(3, 25):
        th._record_model_charge(task_id, ASTRA, 0.01, 1, f"k{i}")
    assert th._task_has_transcript_events(task_id) is False
    th._queue_chat_event({"type": "text", "text": "hello"}, task_id)
    assert th._task_has_transcript_events(task_id) is True
