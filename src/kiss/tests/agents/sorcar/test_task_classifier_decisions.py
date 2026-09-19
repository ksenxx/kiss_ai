# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the decisions (Jev) route of the task classifier.

``kiss.agents.sorcar.task_classifier.classify_task`` asks OpenRouter's
``~typesafe/jev-latest`` decisions model one ``choice`` question through
the ``decide`` tool and maps the chosen kind of task to the
``{is_simple, is_development}`` verdict.  When that route is switched off
(``classify_with_decisions`` config key), unavailable (no
``OPENROUTER_API_KEY``) or fails (HTTP error, unreachable endpoint,
unusable answer), the LLM classifier on the run's own model takes over.

The decisions endpoint is played by a local HTTP server replaying the
live response shapes, reached through the ``KISS_DECISIONS_BASE_URL``
override; the LLM fallback is the real OpenAI-compatible stand-in
endpoint from ``parallel_agent_harness``.  One live test (skipped
without ``OPENROUTER_API_KEY``) confirms the wire format against
OpenRouter itself.

Unreachable without test doubles, documented per the testing policy
rather than mocked: the ``except Exception`` in
``task_classifier._attempt_decisions_classification`` — the ``decide``
tool it wraps converts every request, parsing and validation failure
into an ``"Error: ..."`` string, and its JSON output always parses, so
the guard only fires on a bug in the tool itself; it exists so that
such a bug can never keep a task from launching.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.task_classifier import (
    _DECISIONS_CRITERIA,
    _DECISIONS_QUESTIONS,
    _KIND_VERDICTS,
    CLASSIFIER_DECISIONS_MIN_PROBABILITY,
    CLASSIFIER_DECISIONS_TIMEOUT_SECONDS,
    CLASSIFIER_TASK_MAX_CHARS,
    ClassifierRun,
    TaskClassification,
    _decisions_model_config,
    cached_classification,
    classification_will_call_model,
    classify_task,
    clear_classification_cache,
    decisions_classification_enabled,
)
from kiss.core import config as config_module
from kiss.core.models.model_info import calculate_cost
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
)

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"
_BASE_URL_ENV = "KISS_DECISIONS_BASE_URL"
SERVED_MODEL = "typesafe/jev-1.13-20260917"
INPUT_TOKENS = 612
OUTPUT_TOKENS = 5


def _choice_answer(kind: str) -> dict[str, Any]:
    """A live-shaped ``choice`` answer picking *kind*."""
    options = list(_KIND_VERDICTS)
    probabilities = {option: 0.02 for option in options}
    probabilities[kind] = 0.92
    return {
        "kind": {
            "type": "choice",
            "choice": kind,
            "probabilities": probabilities,
            "confidence": 0.9,
        }
    }


class _JevHandler(BaseHTTPRequestHandler):
    """A ``/alpha/decisions`` replay; the ``state`` text selects the scenario.

    A state starting with ``kind:<option>`` answers that option;
    ``answers:<json>`` returns that JSON verbatim as the ``answers`` object;
    other prefixes drive the failure scenarios; anything else is ``simple``.
    """

    requests: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, D102
        return

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = json.loads(self.rfile.read(length))
        _JevHandler.requests.append({"path": self.path, "body": body})
        state = body.get("state", "")
        usage = {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS, "cost": 2.57e-05}
        if state.startswith("http-400"):
            self._send(400, {"error": {"message": "Model x does not exist", "code": 400}})
        elif state.startswith("unknown-kind"):
            self._send(
                200, {"model": SERVED_MODEL, "answers": _choice_answer("banana"), "usage": usage}
            )
        elif state.startswith("no-kind"):
            self._send(200, {"model": SERVED_MODEL, "answers": {}, "usage": usage})
        elif state.startswith("no-usage"):
            self._send(200, {"model": SERVED_MODEL, "answers": _choice_answer("development")})
        elif state.startswith("answers:"):
            answers = json.loads(state[len("answers:"):])
            self._send(200, {"model": SERVED_MODEL, "answers": answers, "usage": usage})
        else:
            kind = state.split(":", 1)[1].split()[0] if state.startswith("kind:") else "simple"
            self._send(
                200, {"model": SERVED_MODEL, "answers": _choice_answer(kind), "usage": usage}
            )

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _llm_responder(verdict: str) -> Any:
    """An OpenAI-compatible stand-in responder returning *verdict* JSON."""

    def responder(_request: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-decisions-fallback",
            "object": "chat.completion",
            "created": 0,
            "model": STANDIN_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": verdict},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
        }

    return responder


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[IsolatedKissHome]:
    """Isolated KISS_HOME with the classifier on, a fake key, and a Jev replay.

    Yields the isolated home; ``KISS_DECISIONS_BASE_URL`` points at the
    replay server and ``OPENROUTER_API_KEY`` is a placeholder the replay
    ignores, so the decisions route is enabled whatever the developer's
    real keys are.
    """
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "0"
    isolated = IsolatedKissHome("kiss-task-classifier-decisions-")
    clear_classification_cache()
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "test-key")
    server = HTTPServer(("127.0.0.1", 0), _JevHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _JevHandler.requests = []
    monkeypatch.setenv(_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/api")
    try:
        yield isolated
    finally:
        server.shutdown()
        clear_classification_cache()
        if saved is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved
        isolated.cleanup()


def _expected_cost() -> float:
    return calculate_cost(DEFAULT_DECISIONS_MODEL, INPUT_TOKENS, OUTPUT_TOKENS)


# ---------------------------------------------------------------------------
# The decisions route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_KIND_VERDICTS))
def test_each_kind_maps_to_its_verdict(env: IsolatedKissHome, kind: str) -> None:
    """Every option of the ``kind`` question yields its documented verdict, catalog-priced."""
    task = f"kind:{kind} classify me"
    outcome = classify_task(task=task, model_name="claude-haiku-4-5")
    expected_simple, expected_development = _KIND_VERDICTS[kind]
    assert outcome.classification == TaskClassification(
        is_simple=expected_simple, is_development=expected_development
    )
    assert outcome.budget_used == pytest.approx(_expected_cost())
    assert outcome.tokens_used == INPUT_TOKENS + OUTPUT_TOKENS
    assert outcome.steps == 0  # not an LLM step
    assert len(_JevHandler.requests) == 1
    body = _JevHandler.requests[0]["body"]
    assert _JevHandler.requests[0]["path"] == "/api/alpha/decisions"
    assert body["model"] == "~typesafe/jev-latest"
    assert body["state"] == task
    assert body["questions"] == _DECISIONS_QUESTIONS


def test_verdict_is_memoised_separately_from_the_llm_memo(env: IsolatedKissHome) -> None:
    """A repeat is served from the memo at zero cost; the LLM memo stays empty."""
    task = "kind:development add a flag"
    assert classification_will_call_model(task, "claude-haiku-4-5") is True
    first = classify_task(task=task, model_name="claude-haiku-4-5")
    assert first.classification == TaskClassification(is_simple=False, is_development=True)
    assert classification_will_call_model(task, "claude-haiku-4-5") is False
    second = classify_task(task=task, model_name="gpt-5.6-sol")  # any run model shares it
    assert second.classification == first.classification
    assert (second.budget_used, second.tokens_used, second.steps) == (0.0, 0, 0)
    assert len(_JevHandler.requests) == 1
    # Keyed by the decisions criteria, not the LLM prompt.
    assert cached_classification(task, "claude-haiku-4-5") is None
    assert cached_classification(
        task, DEFAULT_DECISIONS_MODEL, _decisions_model_config(), _DECISIONS_CRITERIA
    ) == first.classification


def test_memo_is_bound_to_the_decisions_endpoint(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verdict memoised from one endpoint is not served for another."""
    task = "kind:simple what time is it"
    classify_task(task=task, model_name="claude-haiku-4-5")
    assert classification_will_call_model(task, "claude-haiku-4-5") is False
    monkeypatch.setenv(_BASE_URL_ENV, os.environ[_BASE_URL_ENV] + "/other")
    assert classification_will_call_model(task, "claude-haiku-4-5") is True


def test_run_to_completion_models_are_classified_by_decisions(env: IsolatedKissHome) -> None:
    """``cc/*`` and ``codex/*`` runs, skipped by the LLM classifier, get a Jev verdict."""
    task = "kind:git_only push the branch"
    assert classification_will_call_model(task, "codex/default") is True
    outcome = classify_task(task=task, model_name="cc/claude-fable-5")
    assert outcome.classification == TaskClassification(is_simple=True, is_development=False)
    assert len(_JevHandler.requests) == 1


def test_task_is_truncated_before_it_is_sent(env: IsolatedKissHome) -> None:
    """The state sent to Jev is the truncated task, like the LLM prompt."""
    task = "kind:simple " + "x" * (CLASSIFIER_TASK_MAX_CHARS * 2)
    outcome = classify_task(task=task, model_name="claude-haiku-4-5")
    assert outcome.classification is not None
    state = _JevHandler.requests[0]["body"]["state"]
    assert state.endswith("[task truncated for classification]")
    assert len(state) < len(task)


def test_response_without_usage_still_yields_a_verdict(env: IsolatedKissHome) -> None:
    """A usage-less response is a verdict at zero recorded spend."""
    outcome = classify_task(task="no-usage build it", model_name="claude-haiku-4-5")
    assert outcome.classification == TaskClassification(is_simple=False, is_development=True)
    assert (outcome.budget_used, outcome.tokens_used, outcome.steps) == (0.0, 0, 0)


def test_decisions_model_config_honours_the_base_url_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override env var adds ``base_url``; without it only the timeout is set."""
    monkeypatch.setenv(_BASE_URL_ENV, "http://127.0.0.1:1/api")
    assert _decisions_model_config() == {
        "timeout": CLASSIFIER_DECISIONS_TIMEOUT_SECONDS,
        "base_url": "http://127.0.0.1:1/api",
    }
    monkeypatch.delenv(_BASE_URL_ENV)
    assert _decisions_model_config() == {"timeout": CLASSIFIER_DECISIONS_TIMEOUT_SECONDS}


def _answers_state(answers: dict[str, Any]) -> str:
    """A state the replay server echoes back verbatim as ``answers``."""
    return "answers:" + json.dumps(answers)


def test_low_confidence_choice_gets_the_conservative_verdict(env: IsolatedKissHome) -> None:
    """A ``simple`` pick below the probability floor is treated as ambiguous (conservative)."""
    answers = _choice_answer("simple")
    answers["kind"]["probabilities"]["simple"] = CLASSIFIER_DECISIONS_MIN_PROBABILITY - 0.01
    outcome = classify_task(task=_answers_state(answers), model_name="claude-haiku-4-5")
    assert outcome.classification == TaskClassification(is_simple=False, is_development=True)
    assert outcome.budget_used == pytest.approx(_expected_cost())


def test_probability_floor_applies_only_below_it(env: IsolatedKissHome) -> None:
    """At the floor the chosen kind stands; missing or non-numeric probabilities are ignored."""
    at_floor = _choice_answer("git_only")
    at_floor["kind"]["probabilities"]["git_only"] = CLASSIFIER_DECISIONS_MIN_PROBABILITY
    without = _choice_answer("git_only")
    del without["kind"]["probabilities"]
    odd = _choice_answer("git_only")
    odd["kind"]["probabilities"] = {"git_only": "high"}
    for answers in (at_floor, without, odd):
        outcome = classify_task(task=_answers_state(answers), model_name="claude-haiku-4-5")
        assert outcome.classification == TaskClassification(is_simple=True, is_development=False)
    assert len(_JevHandler.requests) == 3


# ---------------------------------------------------------------------------
# Falling back to the LLM classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        "http-400 task",
        "unknown-kind task",
        "no-kind task",
        _answers_state({"kind": "development"}),  # the answer is not an object
        _answers_state({"other": {"choice": "simple"}}),  # no ``kind`` answer
        _answers_state({"kind": {"choice": ["simple", "git_only"]}}),  # unhashable choice
        _answers_state({"kind": {"choice": {"simple": 1}}}),  # object as choice
        _answers_state({"kind": {"choice": None}}),
    ],
)
def test_decisions_failure_falls_back_to_the_llm(env: IsolatedKissHome, state: str) -> None:
    """An HTTP error or an unusable answer hands the task to the LLM classifier."""
    llm = StandInModelServer(_llm_responder('{"is_simple": true, "is_development": false}'))
    try:
        outcome = classify_task(task=state, model_name=STANDIN_MODEL, model_config=llm.model_config)
    finally:
        llm.stop()
    assert outcome.classification == TaskClassification(is_simple=True, is_development=False)
    assert outcome.steps == 1  # the LLM generation
    assert len(_JevHandler.requests) == 1
    # A failed decisions attempt costs nothing on HTTP 400; a 200 with an
    # unusable answer still spent tokens, which the outcome includes.
    if state.startswith("http-400"):
        assert outcome.tokens_used == 52
    else:
        assert outcome.tokens_used == 52 + INPUT_TOKENS + OUTPUT_TOKENS
        assert outcome.budget_used > _expected_cost()
    # The LLM verdict is memoised under the LLM key, not the decisions key.
    assert cached_classification(state, STANDIN_MODEL, llm.model_config) == outcome.classification
    assert (
        cached_classification(
            state, DEFAULT_DECISIONS_MODEL, _decisions_model_config(), _DECISIONS_CRITERIA
        )
        is None
    )


def test_unreachable_decisions_endpoint_falls_back(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection failure to the decisions API is a soft failure: the LLM decides."""
    closed = HTTPServer(("127.0.0.1", 0), _JevHandler)
    port = closed.server_port
    closed.server_close()
    monkeypatch.setenv(_BASE_URL_ENV, f"http://127.0.0.1:{port}/api")
    llm = StandInModelServer(_llm_responder('{"is_simple": false, "is_development": true}'))
    try:
        outcome = classify_task(
            task="anything", model_name=STANDIN_MODEL, model_config=llm.model_config
        )
    finally:
        llm.stop()
    assert outcome.classification == TaskClassification(is_simple=False, is_development=True)
    assert outcome.steps == 1
    assert _JevHandler.requests == []


def test_both_classifiers_failing_yields_no_verdict(env: IsolatedKissHome) -> None:
    """When Jev errors and the LLM's answer is unparseable the run proceeds unclassified."""
    llm = StandInModelServer(_llm_responder("I cannot say."))
    try:
        outcome = classify_task(
            task="http-400 x", model_name=STANDIN_MODEL, model_config=llm.model_config
        )
    finally:
        llm.stop()
    assert outcome.classification is None
    assert outcome.steps == 2  # structured attempt + plain retry
    assert outcome.tokens_used == 104


def test_run_to_completion_model_falls_back_to_skip(env: IsolatedKissHome) -> None:
    """When Jev fails for a ``cc/*`` run, the LLM route's skip applies: no verdict, no call."""
    outcome = classify_task(task="http-400 x", model_name="cc/claude-fable-5")
    assert outcome == ClassifierRun(classification=None, budget_used=0.0, tokens_used=0, steps=0)
    assert len(_JevHandler.requests) == 1


# ---------------------------------------------------------------------------
# When the decisions route is off
# ---------------------------------------------------------------------------


def test_without_openrouter_key_the_llm_classifies(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``OPENROUTER_API_KEY`` means the decide tool is never called."""
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "")
    assert decisions_classification_enabled() is False
    llm = StandInModelServer(_llm_responder('{"is_simple": true, "is_development": false}'))
    try:
        assert classification_will_call_model("kind:development t", STANDIN_MODEL, llm.model_config)
        assert classification_will_call_model("kind:development t", "codex/default") is False
        outcome = classify_task(
            task="kind:development t", model_name=STANDIN_MODEL, model_config=llm.model_config
        )
    finally:
        llm.stop()
    assert outcome.classification == TaskClassification(is_simple=True, is_development=False)
    assert outcome.steps == 1
    assert _JevHandler.requests == []


def test_config_key_off_uses_the_llm(env: IsolatedKissHome) -> None:
    """``classify_with_decisions: false`` pins the LLM classifier even with a key."""
    assert decisions_classification_enabled() is True
    env.write_config(classify_with_decisions=False)
    assert decisions_classification_enabled() is False
    llm = StandInModelServer(_llm_responder('{"is_simple": false, "is_development": false}'))
    try:
        outcome = classify_task(
            task="kind:simple t", model_name=STANDIN_MODEL, model_config=llm.model_config
        )
    finally:
        llm.stop()
    assert outcome.classification == TaskClassification(is_simple=False, is_development=False)
    assert _JevHandler.requests == []


# ---------------------------------------------------------------------------
# SorcarAgent wiring
# ---------------------------------------------------------------------------


def test_sorcar_agent_banks_the_decisions_spend(env: IsolatedKissHome) -> None:
    """``_classify_task_once`` records Jev's cost and tokens at zero steps; the fold banks them."""
    agent = SorcarAgent("decisions-classifier-wiring")
    verdict = agent._classify_task_once(
        "claude-haiku-4-5", "kind:internet what is the weather", None, enabled_override=True
    )
    assert verdict == TaskClassification(is_simple=False, is_development=False)
    spend = agent._classifier_spend
    assert spend is not None
    assert spend.budget == pytest.approx(_expected_cost())
    assert spend.tokens == INPUT_TOKENS + OUTPUT_TOKENS
    assert spend.steps == 0
    agent._fold_classifier_usage()
    banked = agent.usage_snapshot()
    assert banked[0] == pytest.approx(_expected_cost())
    assert banked[1:] == (INPUT_TOKENS + OUTPUT_TOKENS, 0)


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


@pytest.mark.live_api
@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_live_jev_classifies_arithmetic_as_simple(monkeypatch: pytest.MonkeyPatch) -> None:
    """Against OpenRouter itself, a pure-arithmetic question is simple and not development."""
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "0"
    monkeypatch.delenv(_BASE_URL_ENV, raising=False)
    isolated = IsolatedKissHome("kiss-task-classifier-decisions-live-")
    clear_classification_cache()
    try:
        outcome = classify_task(
            task="What is 2 + 2? Answer in text only.", model_name="claude-haiku-4-5"
        )
    finally:
        clear_classification_cache()
        if saved is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved
        isolated.cleanup()
    assert outcome.classification == TaskClassification(is_simple=True, is_development=False)
    assert outcome.steps == 0
    assert 0 < outcome.budget_used < 0.001
    assert outcome.tokens_used > 0
