"""Cost audit 2026-11-06: the pre-run classifier's spend is filed under the model that made it.

The task classifier (one ``decide`` call on the decisions model, or the
LLM fallback on the run's model) emits no ``llm_call``.  Its spend was
folded into the row's totals (``SorcarAgent._fold_classifier_usage``) but
left no per-model record, so the Spend-by-model split
(``persistence._split_own_share``) filed the decisions model's spend
under the task's own model.  ``ClassifierRun.charges`` now carries the
split by model and the fold records each part
(``SorcarAgent._file_classifier_charges`` ->
``persistence._record_model_charge``), as a ``decide`` call gets.

The gpt-6-astra review of the change found two more gaps, fixed here:
``_record_model_charge`` is idempotent by key, so a fold a Stop cut
short files its records again on the retry (and the record stays until
filed); ``ChatSorcarAgent.run``'s cleanup passes its own row id, since a
Stop before the id is published leaves ``last_task_id`` on the previous
row.  And the vision bridge's description calls, billed into the calling
agent's totals, were filed under the agent's text-only model: the model
now lists each call with its model (``Model.helper_calls``) and the
agent hands it to ``helper_charge_hook``, which ``SorcarAgent`` points
at ``_record_model_charge``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar import sorcar_agent as sorcar_agent_module
from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _ClassifierSpend
from kiss.agents.sorcar.task_classifier import TaskClassification, classify_task
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import calculate_cost
from kiss.tests.agents.sorcar.local_model_server import serve, tool_call_body
from kiss.tests.agents.sorcar.test_cost_audit_20261023 import _llm_call
from kiss.tests.agents.sorcar.test_task_classifier_decisions import (  # noqa: F401
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    _expected_cost,
    _llm_responder,
    env,  # the fixture
)
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer
from kiss.tests.core.test_vision_bridge_2026_10_31 import (
    LOCAL_VISION_MODEL,
    TEXT_ONLY_MODEL,
    VISION_COST,
    VISION_INPUT_TOKENS,
    VISION_OUTPUT_TOKENS,
    _route_the_bridge_to,
    _vision_turn,
    _write_png,
)
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
)

FABLE = "claude-fable-5-1"
JEV_TOKENS = INPUT_TOKENS + OUTPUT_TOKENS


def _row(task: str) -> str:
    """An unfinished task row on FABLE with one own call of $1 / 1,000 tokens."""
    task_id, _ = th._add_task(task, "", {"model": FABLE})
    th._queue_chat_event(_llm_call(FABLE, 1.0, 1_000), task_id)
    th._flush_chat_events()
    return task_id


def _finish(task_id: str, cost: float, tokens: int) -> None:
    th._save_task_extra({"cost": cost, "tokens": tokens, "steps": 1,
                         "endTs": int(time.time() * 1000)}, task_id=task_id)


def _split(task_id: str) -> dict[str, list[float]]:
    row = th._get_db().execute("SELECT model_costs FROM task_history WHERE id = ?",
                               (task_id,)).fetchone()
    split: dict[str, list[float]] = json.loads(row[0] or "{}")
    return split


def _charge_records(task_id: str) -> int:
    th._flush_chat_events()
    count: int = th._get_db().execute(
        "SELECT COUNT(*) FROM events WHERE task_id = ? "
        "AND json_extract(event_json, '$.type') = 'gate_stage' "
        "AND json_extract(event_json, '$.stage') = 'charge'",
        (task_id,),
    ).fetchone()[0]
    return count


def _agent(task_id: str) -> SorcarAgent:
    agent = SorcarAgent("classifier-split test")
    agent._use_web_tools = False
    agent.last_task_id = task_id  # type: ignore[attr-defined]  # as ChatSorcarAgent exposes it
    return agent


def test_a_decisions_verdict_reports_its_spend_under_the_decisions_model(
    env: IsolatedKissHome,  # noqa: F811
) -> None:
    outcome = classify_task("kind:internet what is the weather", "claude-haiku-4-5")
    assert outcome.classification == TaskClassification(is_simple=False)
    assert outcome.charges == ((DEFAULT_DECISIONS_MODEL, outcome.budget_used, JEV_TOKENS),)
    assert outcome.budget_used == pytest.approx(_expected_cost())
    # A memoised verdict costs nothing and reports no charge.
    again = classify_task("kind:internet what is the weather", "claude-haiku-4-5")
    assert (again.budget_used, again.tokens_used, again.charges) == (0.0, 0, ())


def test_a_fallback_reports_both_models_and_the_parts_add_up(
    env: IsolatedKissHome,  # noqa: F811
) -> None:
    llm = StandInModelServer(_llm_responder('{"is_simple": true}'))
    try:
        outcome = classify_task("unknown-kind task", STANDIN_MODEL, llm.model_config)
    finally:
        llm.stop()
    assert outcome.classification == TaskClassification(is_simple=True)
    models = [model for model, _, _ in outcome.charges]
    assert models == [DEFAULT_DECISIONS_MODEL, STANDIN_MODEL]
    assert sum(cost for _, cost, _ in outcome.charges) == pytest.approx(outcome.budget_used)
    assert sum(tokens for _, _, tokens in outcome.charges) == outcome.tokens_used
    assert outcome.charges[1][2] == 52  # the stand-in LLM's usage


def test_the_fold_files_the_classifier_spend_under_the_decisions_model(
    env: IsolatedKissHome,  # noqa: F811
) -> None:
    task_id = _row("classified task")
    agent = _agent(task_id)
    agent._classify_task_once(
        "claude-haiku-4-5", "kind:internet what is the weather", None, enabled_override=True,
    )
    spend = agent._classifier_spend
    assert spend is not None
    assert spend.charges == ((DEFAULT_DECISIONS_MODEL, spend.budget, JEV_TOKENS),)
    agent._fold_classifier_usage()
    cost = agent.usage_snapshot()[0]
    assert cost == pytest.approx(_expected_cost())  # the row's totals still get it once
    _finish(task_id, 1.0 + cost, 1_000 + JEV_TOKENS)
    assert _split(task_id) == {
        FABLE: [1.0, 1_000],
        DEFAULT_DECISIONS_MODEL: [pytest.approx(cost), JEV_TOKENS],
    }
    # A second fold (the chat agent's retry) and a repeated filing record nothing more.
    agent._fold_classifier_usage()
    agent._file_classifier_charges(spend, task_id)
    assert _charge_records(task_id) == 1


def test_a_rowless_run_and_a_free_classification_record_nothing(
    env: IsolatedKissHome,  # noqa: F811
) -> None:
    rowless = SorcarAgent("rowless classifier")
    rowless._use_web_tools = False
    rowless._classify_task_once(
        "claude-haiku-4-5", "kind:internet what is the weather", None, enabled_override=True,
    )
    rowless._fold_classifier_usage()
    assert rowless.usage_snapshot()[0] == pytest.approx(_expected_cost())
    # The same task again is a memo hit: no spend, so no record on the row.
    task_id = _row("memoised task")
    agent = _agent(task_id)
    agent._classify_task_once(
        "claude-haiku-4-5", "kind:internet what is the weather", None, enabled_override=True,
    )
    agent._fold_classifier_usage()
    assert agent.usage_snapshot()[0] == 0.0
    _finish(task_id, 1.0, 1_000)
    assert _split(task_id) == {FABLE: [1.0, 1_000]}


def _jev_spend(key: str) -> _ClassifierSpend:
    return _ClassifierSpend(key, 0.25, 123, 0, ((DEFAULT_DECISIONS_MODEL, 0.25, 123),))


def test_a_charge_recorded_twice_under_one_key_counts_once(
    env: IsolatedKissHome,  # noqa: F811
) -> None:
    task_id = _row("idempotent charge")
    th._record_model_charge(task_id, LOCAL_VISION_MODEL, 0.5, 10, "k1")
    th._record_model_charge(task_id, LOCAL_VISION_MODEL, 0.5, 10, "k1")
    th._record_model_charge(task_id, LOCAL_VISION_MODEL, 0.25, 5, "k2")
    _finish(task_id, 1.75, 1_015)
    expected = {FABLE: [1.0, 1_000], LOCAL_VISION_MODEL: [0.75, 15]}
    assert _split(task_id) == expected
    # On the finished row a retried record refreshes the split and adds nothing.
    th._record_model_charge(task_id, LOCAL_VISION_MODEL, 0.5, 10, "k1")
    assert _split(task_id) == expected
    assert _charge_records(task_id) == 2


@pytest.mark.parametrize("stop", ["before the record", "after the record"])
def test_a_fold_a_stop_cut_short_files_the_record_on_the_retry(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch, stop: str,  # noqa: F811
) -> None:
    task_id = _row("stopped fold")
    agent = _agent(task_id)
    agent._classifier_spend = _jev_spend("classifier:stopped")
    real = th._record_model_charge
    stops: list[str] = []

    def stopped_once(*args: object) -> None:
        if stops or stop == "after the record":
            real(*args)  # type: ignore[arg-type]
        if not stops:
            stops.append(stop)
            raise KeyboardInterrupt(stop)

    monkeypatch.setattr(sorcar_agent_module, "_record_model_charge", stopped_once)
    with pytest.raises(KeyboardInterrupt):
        agent._fold_classifier_usage()
    assert agent._classifier_spend is not None  # kept for the retry
    agent._fold_classifier_usage()
    assert agent._classifier_spend is None
    assert agent.usage_snapshot() == (0.25, 123, 0)  # the ledger fold counted once
    _finish(task_id, 1.25, 1_123)
    assert _split(task_id) == {FABLE: [1.0, 1_000], DEFAULT_DECISIONS_MODEL: [0.25, 123]}
    assert _charge_records(task_id) == 1


def test_the_fold_files_on_the_row_it_is_given(
    env: IsolatedKissHome,  # noqa: F811
) -> None:
    previous = _row("previous task")
    current = _row("current task")
    agent = _agent(previous)  # a Stop before the new id was published
    agent._classifier_spend = _jev_spend("classifier:given-row")
    agent._fold_classifier_usage(current)
    _finish(previous, 1.0, 1_000)
    _finish(current, 1.25, 1_123)
    assert _split(previous) == {FABLE: [1.0, 1_000]}
    assert _split(current) == {FABLE: [1.0, 1_000], DEFAULT_DECISIONS_MODEL: [0.25, 123]}


def test_sorcar_files_a_helper_call_under_its_model(
    env: IsolatedKissHome,  # noqa: F811
) -> None:
    task_id = _row("described picture")
    agent = _agent(task_id)
    agent._file_helper_charge(LOCAL_VISION_MODEL, 0.01, 100, "vision-bridge:k")
    agent._file_helper_charge(LOCAL_VISION_MODEL, 0.01, 100, "vision-bridge:k")  # handed twice
    _finish(task_id, 1.01, 1_100)
    assert _split(task_id) == {FABLE: [1.0, 1_000], LOCAL_VISION_MODEL: [0.01, 100]}
    SorcarAgent("rowless")._file_helper_charge(LOCAL_VISION_MODEL, 0.01, 100, "k")  # no row: no-op


def test_a_vision_description_is_handed_to_the_helper_charge_hook_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    png = tmp_path / "shot.png"
    _write_png(png)
    script = [
        tool_call_body("Read", {"file_path": str(png)}, prompt_tokens=100),
        tool_call_body("finish", {"result": "done"}, prompt_tokens=100),
    ]
    filed: list[tuple[str, float, int, str]] = []
    agent = KISSAgent("vision-split")
    agent.helper_charge_hook = lambda model, cost, tokens, key: filed.append(
        (model, cost, tokens, key),
    )
    with ScriptedAnthropicServer(_vision_turn("A white PNG.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        with serve(script) as (url, _requests):
            agent.run(
                model_name=TEXT_ONLY_MODEL,
                prompt_template="Read {path} and report what it shows.",
                arguments={"path": str(png)},
                tools=[UsefulTools().Read],
                model_config={"base_url": url, "api_key": "local"},
                max_budget=1.0,
                max_steps=5,
                verbose=False,
                print_prompts=False,
            )
    assert [(model, tokens) for model, _, tokens, _ in filed] == [
        (LOCAL_VISION_MODEL, VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS),
    ]
    assert filed[0][1] == pytest.approx(VISION_COST)
    # Billed once into the totals, as before: the hook only files it.
    chat_call = calculate_cost(TEXT_ONLY_MODEL, 100, 100)
    assert agent.budget_used == pytest.approx(2 * chat_call + VISION_COST)
