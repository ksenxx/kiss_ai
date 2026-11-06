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
    charges: list[float] = []

    class _Printer:
        def charge_task_usage(self, task_agent: Any, task_id: str, budget: float,
                              tokens: int, steps: int, epoch: Any = None) -> None:
            charges.append(budget)

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
    agent.printer = _Printer()  # type: ignore[assignment]
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
    assert len(charges) == 1 and charges[0] > 0
    split = json.loads(_model_costs(row))
    assert set(split) == {FABLE, MODEL}
    assert split[MODEL][0] == pytest.approx(charges[0]) and split[MODEL][1] > 0


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


@pytest.mark.usefixtures("history_db")
def test_charge_records_are_not_transcript() -> None:
    task_id, _ = th._add_task("metadata only", "", {"model": FABLE})
    th._record_model_charge(task_id, ASTRA, 0.5, 600, "k1")
    th._record_model_charge(task_id, ASTRA, 0.5, 600, "k2")
    assert th._task_has_transcript_events(task_id) is False
    th._queue_chat_event({"type": "text", "text": "hello"}, task_id)
    assert th._task_has_transcript_events(task_id) is True
