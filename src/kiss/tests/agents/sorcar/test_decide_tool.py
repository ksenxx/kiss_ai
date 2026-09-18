# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Sorcar ``decide`` tool.

The tool lets a running agent classify, route or score text through
:meth:`kiss.core.models.decisions_model.DecisionsModel.decide`.  A local
HTTP server plays OpenRouter's ``/alpha/decisions`` endpoint with the
response shapes recorded from the live service, so the tool, its question
parsing, its output rendering and its task-cost attribution are all driven
over a real socket against a real :class:`SorcarAgent`.  One live test
(skipped without ``OPENROUTER_API_KEY``) confirms the wire format against
OpenRouter itself.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from kiss.agents.sorcar.decide_tool import (
    DEFAULT_DECISIONS_MODEL,
    decisions_tool_available,
    format_decision,
    make_decide_tool,
    parse_questions,
)
from kiss.agents.sorcar.mcp_servers import _RESERVED_TOOL_NAMES
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import calculate_cost, model

LIVE_ANSWERS: dict[str, Any] = {
    "is_bug": {"type": "noul", "noul": 0.97},
    "route": {
        "type": "choice",
        "choice": "support",
        "probabilities": {"billing": 0.02, "support": 0.96, "sales": 0.02},
        "confidence": 0.94,
    },
    "urgency": {
        "type": "score",
        "score": 2.3,
        "legend": {"0": "can wait a week", "1": "this week", "2": "today", "3": "right now"},
        "probabilities": {"0": 0, "1": 0.1, "2": 0.5, "3": 0.4},
        "confidence": 0.7,
    },
}

QUESTIONS_JSON = json.dumps(
    {
        "is_bug": {"type": "noul", "instructions": "Does the message report a software bug?"},
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "payments and invoices",
                "support": "product help",
                "sales": "pricing and upgrades",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the message?",
            "criteria": ["can wait a week", "this week", "today", "right now"],
        },
    }
)

STATE = "The export button crashes the app every time I click it; I need the report today."


class _DecisionsHandler(BaseHTTPRequestHandler):
    """A ``/alpha/decisions`` endpoint; the ``state`` string picks the scenario."""

    requests: list[dict[str, Any]] = []

    def log_message(self, *_args: Any) -> None:  # noqa: D102 — silence the test log
        return

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = json.loads(self.rfile.read(length))
        _DecisionsHandler.requests.append({"path": self.path, "body": body})
        if body.get("state") == "bad-request":
            payload = {"error": {"message": "Model x does not exist", "code": 400}}
            self._send(400, payload)
        elif body.get("state") == "no-usage":
            self._send(200, {"model": "typesafe/jev-1.13-20260917", "answers": LIVE_ANSWERS})
        else:
            self._send(
                200,
                {
                    "model": "typesafe/jev-1.13-20260917",
                    "answers": LIVE_ANSWERS,
                    "usage": {"input_tokens": 410, "output_tokens": 71, "cost": 1.722e-05},
                    "id": "gen-dec-test",
                    "provider": "TypeSafe",
                },
            )

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@contextmanager
def _decisions_server() -> Iterator[str]:
    """Serve :class:`_DecisionsHandler` on a free port; yield the API root."""
    server = HTTPServer(("127.0.0.1", 0), _DecisionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _DecisionsHandler.requests = []
    try:
        yield f"http://127.0.0.1:{server.server_port}/api"
    finally:
        server.shutdown()


def _tool_names(tools: list[Any]) -> list[str]:
    return [t.__name__ for t in tools if callable(t)]


def _find_tool(tools: list[Any], name: str) -> Any:
    for tool in tools:
        if callable(tool) and tool.__name__ == name:
            return tool
    raise AssertionError(f"Tool {name!r} not found in {_tool_names(tools)}")


def _make_agent() -> SorcarAgent:
    agent = SorcarAgent("test-decide-tool")
    agent._use_web_tools = False
    return agent


# --------------------------------------------------------------------------
# parse_questions
# --------------------------------------------------------------------------


def test_parse_questions_normalises_every_type() -> None:
    parsed = parse_questions(QUESTIONS_JSON)
    assert parsed["is_bug"] == {
        "type": "noul",
        "instructions": "Does the message report a software bug?",
    }
    assert parsed["route"]["type"] == "choice"
    assert parsed["route"]["criteria"] == {
        "billing": "payments and invoices",
        "support": "product help",
        "sales": "pricing and upgrades",
    }
    assert parsed["urgency"] == {
        "type": "score",
        "instructions": "How urgent is the message?",
        "criteria": ["can wait a week", "this week", "today", "right now"],
    }


def test_parse_questions_choice_list_becomes_id_mapping() -> None:
    parsed = parse_questions(
        '{"kind": {"type": "choice", "instructions": "Kind?", "criteria": ["a", "b"]}}'
    )
    assert parsed["kind"]["criteria"] == {"a": "a", "b": "b"}


@pytest.mark.parametrize(
    ("questions", "fragment"),
    [
        ("not json", "not valid JSON"),
        ("[]", "non-empty JSON object"),
        ("{}", "non-empty JSON object"),
        ('{"q": "noul"}', "must be a JSON object"),
        ('{"q": {"type": "yesno", "instructions": "x"}}', "expected one of noul, choice, score"),
        ('{"q": {"instructions": "x"}}', "expected one of noul, choice, score"),
        ('{"q": {"type": "noul"}}', "non-empty 'instructions'"),
        ('{"q": {"type": "noul", "instructions": "  "}}', "non-empty 'instructions'"),
        ('{"q": {"type": "choice", "instructions": "x"}}', "choice question 'q' needs"),
        ('{"q": {"type": "choice", "instructions": "x", "criteria": []}}', "choice question"),
        ('{"q": {"type": "choice", "instructions": "x", "criteria": "a"}}', "choice question"),
        ('{"q": {"type": "score", "instructions": "x", "criteria": {"0": "a"}}}', "score question"),
        ('{"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}', "at least two"),
        # Shapes the live endpoint rejects with HTTP 400 must be caught locally.
        ('{"": {"type": "noul", "instructions": "x"}}', "names must be non-empty"),
        ('{"q": {"type": "choice", "instructions": "x", "criteria": [1, 2]}}', "choice question"),
        ('{"q": {"type": "choice", "instructions": "x", "criteria": [["a"], "b"]}}', "choice"),
        ('{"q": {"type": "choice", "instructions": "x", "criteria": ["a", ""]}}', "choice"),
        (
            '{"q": {"type": "choice", "instructions": "x", "criteria": {"a": 3, "b": false}}}',
            "choice question",
        ),
        ('{"q": {"type": "choice", "instructions": "x", "criteria": {"": "blank id"}}}', "choice"),
        ('{"q": {"type": "score", "instructions": "x", "criteria": [0, 1]}}', "non-empty rubric"),
        ('{"q": {"type": "score", "instructions": "x", "criteria": ["low", " "]}}', "rubric"),
    ],
)
def test_parse_questions_rejects_malformed_input(questions: str, fragment: str) -> None:
    with pytest.raises(KISSError) as excinfo:
        parse_questions(questions)
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    "questions",
    [
        '{"q": {"type": "choice", "instructions": "x", "criteria": [1, 2]}}',
        '{"q": {"type": "choice", "instructions": "x", "criteria": [["a"], "b"]}}',
        '{"": {"type": "noul", "instructions": "x"}}',
    ],
)
def test_decide_tool_returns_error_text_for_bad_criteria_without_a_request(
    questions: str,
) -> None:
    with _decisions_server() as base_url:
        decide = make_decide_tool(
            _make_agent(), model_config={"base_url": base_url, "api_key": "test-key"}
        )
        result = decide(STATE, questions)
    assert result.startswith("Error: ")
    assert _DecisionsHandler.requests == []


# --------------------------------------------------------------------------
# format_decision
# --------------------------------------------------------------------------


def test_format_decision_prices_from_catalog_and_reports_served_model() -> None:
    response = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": LIVE_ANSWERS,
        "usage": {"input_tokens": 410, "output_tokens": 71, "cost": 1.722e-05},
    }
    text, cost, tokens = format_decision(DEFAULT_DECISIONS_MODEL, response)
    rendered = json.loads(text)
    assert rendered["answers"] == LIVE_ANSWERS
    assert rendered["model"] == "typesafe/jev-1.13-20260917"
    assert rendered["usage"] == {"input_tokens": 410, "output_tokens": 71, "cost_usd": cost}
    assert cost == calculate_cost(DEFAULT_DECISIONS_MODEL, 410, 71) > 0
    assert tokens == 481


def test_format_decision_without_usage_is_free_and_falls_back_to_catalog_name() -> None:
    text, cost, tokens = format_decision(DEFAULT_DECISIONS_MODEL, {"answers": LIVE_ANSWERS})
    rendered = json.loads(text)
    assert rendered["model"] == DEFAULT_DECISIONS_MODEL
    assert rendered["usage"] == {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    assert (cost, tokens) == (0.0, 0)


# --------------------------------------------------------------------------
# make_decide_tool over a real socket
# --------------------------------------------------------------------------


def test_decide_tool_answers_and_attributes_spend_to_agent() -> None:
    agent = _make_agent()
    budget_before = agent.budget_used
    tokens_before = agent.total_tokens_used
    steps_before = agent.total_steps
    with _decisions_server() as base_url:
        decide = make_decide_tool(
            agent, model_config={"base_url": base_url, "api_key": "test-key"}
        )
        result = decide(STATE, QUESTIONS_JSON)
    rendered = json.loads(result)
    assert rendered["answers"]["route"]["choice"] == "support"
    assert rendered["answers"]["is_bug"]["noul"] == 0.97
    assert rendered["answers"]["urgency"]["score"] == 2.3
    assert rendered["model"] == "typesafe/jev-1.13-20260917"
    expected_cost = calculate_cost(DEFAULT_DECISIONS_MODEL, 410, 71)
    assert rendered["usage"]["cost_usd"] == expected_cost
    # The wire request carries the un-prefixed model id and the normalised questions.
    [request] = _DecisionsHandler.requests
    assert request["path"] == "/api/alpha/decisions"
    assert request["body"]["model"] == "~typesafe/jev-latest"
    assert request["body"]["state"] == STATE
    assert request["body"]["questions"] == parse_questions(QUESTIONS_JSON)
    # Spend folds into the task's accounting; no agent step is charged.
    assert agent.budget_used == pytest.approx(budget_before + expected_cost)
    assert agent.total_tokens_used == tokens_before + 481
    assert agent.total_steps == steps_before


def test_decide_tool_without_agent_skips_attribution() -> None:
    with _decisions_server() as base_url:
        decide = make_decide_tool(
            None, model_config={"base_url": base_url, "api_key": "test-key"}
        )
        rendered = json.loads(decide(STATE, QUESTIONS_JSON))
    assert rendered["answers"] == LIVE_ANSWERS


def test_decide_tool_free_response_leaves_accounting_untouched() -> None:
    agent = _make_agent()
    budget_before, tokens_before = agent.budget_used, agent.total_tokens_used
    with _decisions_server() as base_url:
        decide = make_decide_tool(
            agent, model_config={"base_url": base_url, "api_key": "test-key"}
        )
        rendered = json.loads(decide("no-usage", QUESTIONS_JSON))
    assert rendered["usage"]["cost_usd"] == 0.0
    assert (agent.budget_used, agent.total_tokens_used) == (budget_before, tokens_before)


def test_decide_tool_reports_http_errors_as_text() -> None:
    agent = _make_agent()
    with _decisions_server() as base_url:
        decide = make_decide_tool(
            agent, model_config={"base_url": base_url, "api_key": "test-key"}
        )
        result = decide("bad-request", QUESTIONS_JSON)
    assert result.startswith("Error: Decisions request failed (HTTP 400 Bad Request)")
    assert "Model x does not exist" in result


def test_decide_tool_rejects_malformed_questions_before_any_request() -> None:
    with _decisions_server() as base_url:
        decide = make_decide_tool(
            _make_agent(), model_config={"base_url": base_url, "api_key": "test-key"}
        )
        result = decide(STATE, '{"q": {"type": "maybe", "instructions": "x"}}')
    assert result.startswith("Error: Question 'q' has type 'maybe'")
    assert _DecisionsHandler.requests == []


def test_make_decide_tool_refuses_non_decisions_model() -> None:
    with pytest.raises(KISSError, match="is not a decisions model"):
        make_decide_tool(None, model_name="gpt-5.5")


# --------------------------------------------------------------------------
# Agent wiring
# --------------------------------------------------------------------------


def test_decide_tool_schema_exposes_both_arguments() -> None:
    decide = make_decide_tool(None, model_config={"api_key": "test-key"})
    schema = model(DEFAULT_DECISIONS_MODEL)._function_to_openai_tool(decide)["function"]
    assert schema["name"] == "decide"
    assert "Classify, route or score text" in schema["description"]
    assert schema["parameters"]["required"] == ["state", "questions"]
    questions_doc = schema["parameters"]["properties"]["questions"]["description"]
    assert '"type": "noul"|"choice"|"score"' in questions_doc
    assert "Example:" in questions_doc


def test_agent_offers_decide_when_openrouter_key_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "test-key")
    assert decisions_tool_available()
    tools = _make_agent()._get_tools()
    decide = _find_tool(tools, "decide")
    assert _tool_names(tools).count("decide") == 1
    # The registered tool validates locally, without a key-consuming request.
    assert decide(STATE, "not json").startswith("Error: questions is not valid JSON")


def test_agent_hides_decide_without_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "")
    assert not decisions_tool_available()
    assert "decide" not in _tool_names(_make_agent()._get_tools())


def test_decide_is_reserved_against_mcp_name_collisions() -> None:
    assert "decide" in _RESERVED_TOOL_NAMES


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_decide_tool_live_against_openrouter() -> None:
    agent = _make_agent()
    decide = make_decide_tool(agent)
    rendered = json.loads(decide(STATE, QUESTIONS_JSON))
    assert set(rendered["answers"]) == {"is_bug", "route", "urgency"}
    assert rendered["answers"]["is_bug"]["noul"] > 0.5
    assert rendered["answers"]["route"]["choice"] in {"billing", "support", "sales"}
    assert 0 <= rendered["answers"]["urgency"]["score"] <= 3
    assert rendered["usage"]["input_tokens"] > 0
    assert agent.total_tokens_used >= rendered["usage"]["input_tokens"]
