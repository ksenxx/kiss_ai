"""Cost audit 2026-11-04: a finish-gate review is filed under the reviewer's model.

The finish gate's reviewer runs on another model family with no printer,
so it emits no ``llm_call`` event; its spend is charged to the task.
Each charged attempt now leaves a ``gate_stage`` record with ``stage``
``charge`` (``model``, ``cost_usd``, ``tokens``) under the task's row
(``SorcarAgent._charge_gate_usage`` -> ``persistence._record_model_charge``),
which ``_own_model_costs`` counts, so the Spend-by-model split credits
the reviewer's model, also for a review a Stop ended and for one charged
after the row's final save.
"""

from __future__ import annotations

import functools
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.coding_gate import CodingGate, run_on_host
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.file_lock import lock_exclusive
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.test_coding_gate import MODEL, serve, text_body
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
    """The stage report ``review_verdict`` publishes: never counted (the charges are)."""
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
class TestGateChargeRecords:
    def test_charge_records_are_filed_under_the_reviewer(self) -> None:
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
def test_a_stopped_review_is_filed_under_the_reviewer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Stop ends the review: no review report is published, the attempt is charged
    twice with one key, and its spend is still recorded once under the reviewer."""
    from kiss.core.kiss_agent import KISSAgent

    row = _task(4.0, 4_000, [_llm_call(FABLE, 4.0, 4_000)])
    real_run = KISSAgent.run

    def run(self: KISSAgent, *args: Any, **kwargs: Any) -> str:
        result: str = real_run(self, *args, **kwargs)
        if self.name == "Finish reviewer":
            raise KeyboardInterrupt("stop")
        return result

    monkeypatch.setattr(KISSAgent, "run", run)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "hello.py").write_text("print('hello')\n")
    agent = SorcarAgent("audit1104-stopped-review")
    agent.printer = JsonPrinter()  # the daemon's late-charge bridge
    charge = functools.partial(  # what _new_coding_gate binds
        agent._charge_gate_usage, epoch=agent._usage_epoch(), task_id=row, model=MODEL,
    )

    def shell(command: str, timeout: int) -> tuple[int | None, str]:
        return run_on_host(command, str(tmp_path), timeout)

    with serve([text_body("Looking.\nVERDICT: OK")]) as (url, _requests):
        gate = CodingGate(shell, str(tmp_path), "./tmp/", MODEL, charge,
                          model_config={"base_url": url, "api_key": "local"})
        with pytest.raises(KeyboardInterrupt):
            gate.review_verdict({"success": True, "summary_in_html": "<p>done</p>"}, "hello")
    split = json.loads(_model_costs(row))
    assert set(split) == {FABLE, MODEL}
    assert split[MODEL][0] > 0 and split[MODEL][1] > 0
    assert _row_cost(row) == pytest.approx(4.0 + split[MODEL][0])  # the row took it once


@pytest.mark.usefixtures("history_db")
def test_a_persisted_run_without_a_server_printer_files_the_review_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A standalone ChatSorcarAgent (no late-charge bridge) still records the review's model."""
    from kiss.agents.sorcar import sorcar_agent as agent_module
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    parent = ChatSorcarAgent("audit1104-standalone-review")
    task_id, _ = th._add_task("review without a server printer", "", {"model": FABLE})
    parent._last_task_id = task_id
    parent.model_name = FABLE
    parent.printer = None
    parent.work_dir = str(tmp_path)
    parent._attribute_usage(4.0, 4_000, 1)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "hello.py").write_text("print('hello')\n")
    monkeypatch.setattr(agent_module, "gate_reviewer", lambda _: MODEL)
    with serve([text_body("VERDICT: OK")]) as (url, requests):
        parent.model_config = {"base_url": url, "api_key": "local"}
        gate = parent._new_coding_gate(with_reviewer=True)
        assert gate.review_verdict({"success": True, "summary_in_html": "<p>Done.</p>"},
                                   "Review") is None
        assert len(requests) == 1
    cost, tokens, _ = parent.usage_snapshot()
    parent._save_final_usage({}, task_id, "Review", result_summary="Done.")
    assert json.loads(_model_costs(task_id))[MODEL] == [pytest.approx(cost - 4.0), tokens - 4_000]
    # the charge record is metadata: the standalone run still gets its synthesized replay
    session = th._load_chat_events_by_task_id(task_id)
    assert session is not None
    assert any(e.get("type") != "gate_stage" for e in session["events"])  # type: ignore[attr-defined]


def _run_ok(agent: Any, work_dir: Path) -> str:
    """One run of *agent* against a one-answer local model; its persisted row id."""
    from kiss.agents.sorcar.sorcar_agent import _persisted_task_id
    from kiss.tests.agents.sorcar.local_model_server import finish_body

    with serve([finish_body("<p>ok</p>")]) as (url, _requests):
        agent.run(prompt_template="Say ok.", model_name=MODEL, max_steps=2, max_budget=1.0,
                  model_config={"base_url": url, "api_key": "local"}, use_memory=False,
                  web_tools=False, verbose=False, work_dir=str(work_dir))
    return str(_persisted_task_id(agent))


def _row_cost(task_id: str) -> float:
    """The cost the row *task_id* holds now, late charges included."""
    return float(th._task_usage([task_id])[task_id]["cost"])  # type: ignore[arg-type]


@pytest.mark.usefixtures("history_db")
def test_a_gate_charge_key_counts_once_however_long_the_agent_is_reused(tmp_path: Path) -> None:
    """A finished row keeps the key of each charge it took, the agent none.

    A stop charges an attempt twice, so a key counts once.  The agent
    used to hold the keys it charged (every one, then those of its last
    two runs: audit 2026-11-07, A13_claude-8) and claimed a key before
    charging it, which lost a charge a stop cut (audit 2026-11-09, S11
    F1); the row's retained charge is the receipt now, however many
    runs the reused agent made since.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    agent = ChatSorcarAgent("audit1104-charge-keys")
    row1 = _run_ok(agent, tmp_path)
    assert row1
    cost = _row_cost(row1)
    agent._charge_gate_usage(0.5, 10, 1, "finish-review:one", task_id=row1)
    agent._charge_gate_usage(0.5, 10, 1, "finish-review:one", task_id=row1)  # the stop's retry
    assert _row_cost(row1) == pytest.approx(cost + 0.5)
    row2 = _run_ok(agent, tmp_path)
    assert row2 and row2 != row1
    # One run later the key is still held: a review the stop left charging.
    agent._charge_gate_usage(0.5, 10, 1, "finish-review:one", task_id=row1)
    assert _row_cost(row1) == pytest.approx(cost + 0.5)
    _run_ok(agent, tmp_path)
    # Two runs later too: the row holds the key.
    agent._charge_gate_usage(0.5, 10, 1, "finish-review:one", task_id=row1)
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
