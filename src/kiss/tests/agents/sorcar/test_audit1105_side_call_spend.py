"""Cost audit 2026-11-05: model calls made outside the agent loop are filed under their model.

The commit message (the fast model), a ``talk`` synthesis (the audio
model) and a ``decide`` call (the decisions model) emit no ``llm_call``:
their spend joined the task's total but the Spend-by-model split filed it
under the task's own model.  Each now leaves a ``gate_stage`` ``charge``
record (``sorcar_agent._attribute_side_call_usage`` ->
``persistence._record_model_charge``) that ``_own_model_costs`` counts.
Such a call is bound to its task as it starts (``side_call_binding``) and
charged by the finish gate's row-or-ledger policy, so one returning after
the task ended still charges that task.  The finish gate's own ``decide``
call (``coding_gate.answered_findings``) was not charged to the task at
all; it now is, under the decisions model, and the coding SEA's trial
cost counts its findings check too.
"""

from __future__ import annotations

import functools
import json
import time
from typing import Any

import pytest

from kiss.agents.sorcar import commit_message, decide_tool
from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.coding_gate import CodingGate, answered_findings, run_on_host
from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL, make_decide_tool
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _attribute_side_call_usage,
    _attribute_tts_usage,
    side_call_binding,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.tests.agents.sorcar.test_cost_audit_20261023 import (
    _llm_call,
    _model_costs,
    history_db,  # noqa: F401  (the fixture)
)
from kiss.tests.agents.sorcar.test_decide_tool import (
    QUESTIONS_JSON,
    STATE,
    _decisions_server,
)

FABLE = "claude-fable-5-1"
AUDIO = "gpt-audio-1.5"


def _row(finished: bool) -> str:
    """A task row on FABLE with one own call of $1 / 1,000 tokens."""
    task_id, _ = th._add_task("side-call task", "", {"model": FABLE})
    th._queue_chat_event(_llm_call(FABLE, 1.0, 1_000), task_id)
    th._flush_chat_events()
    final: dict[str, object] = {"cost": 1.0, "tokens": 1_000, "steps": 1}
    if finished:
        final["endTs"] = int(time.time() * 1000)
    th._save_task_extra(final, task_id=task_id)
    return task_id


def _agent(task_id: str) -> SorcarAgent:
    agent = SorcarAgent("side-call test")
    agent._use_web_tools = False
    agent.last_task_id = task_id  # type: ignore[attr-defined]  # as ChatSorcarAgent exposes it
    return agent


def _split(task_id: str) -> dict[str, list[float]]:
    split: dict[str, list[float]] = json.loads(_model_costs(task_id))
    return split


def _row_cost(task_id: str) -> float:
    row = th._get_db().execute("SELECT cost FROM task_history WHERE id = ?",
                               (task_id,)).fetchone()
    return float(row[0])


@pytest.mark.usefixtures("history_db")
class TestSideCallRecords:
    def test_a_finished_row_gets_the_spend_filed_under_its_model(self) -> None:
        task = _row(finished=True)
        agent = _agent(task)
        _attribute_side_call_usage(agent, AUDIO, 0.25, 300)
        assert _row_cost(task) == pytest.approx(1.25)
        assert agent.budget_used == 0.0  # on the row, not on a ledger nothing reads again
        assert _split(task) == {FABLE: [1.0, 1_000], AUDIO: [0.25, 300]}

    def test_an_unfinished_row_banks_the_spend_and_gets_the_split_at_its_final_save(
        self,
    ) -> None:
        task = _row(finished=False)
        agent = _agent(task)
        _attribute_side_call_usage(agent, AUDIO, 0.5, 10)
        assert agent.budget_used == pytest.approx(0.5)
        assert agent.total_tokens_used == 10
        assert agent.total_steps == 0  # not an agent step
        th._save_task_extra({"cost": 1.5, "tokens": 1_010, "steps": 1,
                             "endTs": int(time.time() * 1000)}, task_id=task)
        assert _split(task) == {FABLE: [1.0, 1_000], AUDIO: [0.5, 10]}

    def test_a_call_returning_after_its_task_ended_charges_that_task(self) -> None:
        first = _row(finished=False)
        agent = _agent(first)
        binding = side_call_binding(agent)
        # The task ends and the agent moves on to its next task before the reply.
        th._save_task_extra({"cost": 1.0, "tokens": 1_000, "steps": 1,
                             "endTs": int(time.time() * 1000)}, task_id=first)
        second = _row(finished=False)
        agent.reset_usage()  # the next task's ledger epoch
        agent.last_task_id = second  # type: ignore[attr-defined]
        _attribute_side_call_usage(agent, AUDIO, 0.5, 10, binding)
        assert _row_cost(first) == pytest.approx(1.5)
        assert _split(first) == {FABLE: [1.0, 1_000], AUDIO: [0.5, 10]}
        assert agent.budget_used == 0.0
        assert _split(second) == {FABLE: [1.0, 1_000]}

    def test_free_calls_and_rowless_agents_record_nothing(self) -> None:
        task = _row(finished=True)
        _attribute_side_call_usage(_agent(task), AUDIO, 0.0, 0)
        _attribute_side_call_usage(_agent(task), "", 0.1, 5)  # no model: total only
        rowless = SorcarAgent("no row")
        _attribute_side_call_usage(rowless, AUDIO, 0.1, 5)
        assert rowless.budget_used == pytest.approx(0.1)
        assert _split(task) == {FABLE: [1.0, 1_000]}
        assert _row_cost(task) == pytest.approx(1.1)

    def test_talk_synthesis_is_filed_under_the_audio_model(self) -> None:
        task = _row(finished=True)
        _attribute_tts_usage(_agent(task), {"model": AUDIO, "budget_used": 0.064,
                                            "total_tokens_used": 1_000})
        assert _split(task) == {FABLE: [1.0, 1_000], AUDIO: [0.064, 1_000]}

    def test_commit_message_is_filed_under_the_fast_model(self, monkeypatch: Any) -> None:
        task = _row(finished=True)
        agent = _agent(task)
        monkeypatch.setattr(commit_message, "get_fast_model", lambda: "claude-sonnet-5-5")

        def fake_run(self: KISSAgent, **kwargs: Any) -> str:
            assert kwargs["model_name"] == "claude-sonnet-5-5"
            self.budget_used = 0.18
            self.total_tokens_used = 40_000
            return "fix: a thing"

        monkeypatch.setattr(KISSAgent, "run", fake_run)
        message = commit_message.generate_commit_message_from_diff("diff --git a b", bill_to=agent)
        assert message.startswith("fix: a thing")
        assert _row_cost(task) == pytest.approx(1.18)
        assert _split(task) == {FABLE: [1.0, 1_000], "claude-sonnet-5-5": [0.18, 40_000]}

    def test_decide_tool_is_filed_under_the_decisions_model(self) -> None:
        task = _row(finished=True)
        agent = _agent(task)
        with _decisions_server() as base_url:
            decide = make_decide_tool(agent, model_config={"base_url": base_url,
                                                           "api_key": "test-key"})
            decide(STATE, QUESTIONS_JSON)
        assert _row_cost(task) == pytest.approx(1.0 + 1.722e-05)
        assert _split(task) == {FABLE: [1.0, 1_000],
                                DEFAULT_DECISIONS_MODEL: [pytest.approx(1.722e-05), 481]}


@pytest.mark.usefixtures("history_db")
class TestGateDecideCharge:
    def _serve_decide(self, monkeypatch: Any, base_url: str) -> None:
        real = decide_tool.make_decide_tool
        monkeypatch.setattr(decide_tool, "decisions_tool_available", lambda: True)
        monkeypatch.setattr(decide_tool, "make_decide_tool", functools.partial(
            real, model_config={"base_url": base_url, "api_key": "test-key"}))

    def test_answered_findings_reports_the_spend(self, monkeypatch: Any) -> None:
        spent: list[tuple[float, int]] = []
        with _decisions_server() as base_url:
            self._serve_decide(monkeypatch, base_url)
            answered = answered_findings("summary", ["1. a finding"],
                                         lambda c, t: spent.append((c, t)))
        assert answered is not None
        assert spent == [(pytest.approx(1.722e-05), 481)]

    def test_the_gate_charges_its_decide_call_to_the_run_under_the_decisions_model(
        self, monkeypatch: Any, tmp_path: Any,
    ) -> None:
        task = _row(finished=True)
        agent = _agent(task)
        charge = functools.partial(  # what _new_coding_gate binds
            agent._charge_gate_usage, epoch=None, task_id=task, model="gpt-6-astra")
        gate = CodingGate(lambda c, t: run_on_host(c, str(tmp_path), t), str(tmp_path),
                          str(tmp_path), "gpt-6-astra", charge)
        gate.open_findings = ["1. a finding"]
        with _decisions_server() as base_url:
            self._serve_decide(monkeypatch, base_url)
            gate.unanswered_verdict({"summary_in_html": "done"})
        row = th._get_db().execute("SELECT cost FROM task_history WHERE id = ?",
                                   (task,)).fetchone()
        assert float(row[0]) == pytest.approx(1.0 + 1.722e-05)
        assert _split(task) == {FABLE: [1.0, 1_000],
                                DEFAULT_DECISIONS_MODEL: [pytest.approx(1.722e-05), 481]}


class TestCodingSeaFindingsCheck:
    def test_the_findings_check_spend_is_logged_and_counted_in_the_trial_cost(
        self, monkeypatch: Any, tmp_path: Any,
    ) -> None:
        from types import SimpleNamespace

        from kiss.agents.seas.coding import coding_sea

        def fake_answered(summary: str, findings: list[str],
                          on_usage: Any = None) -> list[bool]:
            on_usage(0.0123, 123)
            return [False] * len(findings)

        monkeypatch.setattr(coding_sea, "answered_findings", fake_answered)
        trajectory = tmp_path / "trajectory.jsonl"
        harness = SimpleNamespace(
            open_findings=["1. a finding"], turns=3,
            _log=lambda record: trajectory.open("a").write(json.dumps(record) + "\n"))
        verdict = coding_sea.ContainerHarness.unanswered_verdict(
            harness, {"summary_in_html": "done"})  # type: ignore[arg-type]
        assert verdict is not None and "a finding" in verdict
        assert coding_sea.review_usage(trajectory) == (pytest.approx(0.0123), 123)
