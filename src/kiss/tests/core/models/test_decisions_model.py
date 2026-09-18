# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the OpenRouter decisions backend.

A local HTTP server plays the ``/alpha/decisions`` endpoint with the exact
request/response shapes recorded from the live service on 2026-09-17, so
every branch of :mod:`kiss.core.models.decisions_model` is exercised over
a real socket.  One live test (skipped without ``OPENROUTER_API_KEY``)
confirms the wire format against OpenRouter itself.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.decisions_model import (
    OPENROUTER_DECISIONS_BASE_URL,
    DecisionsModel,
    api_model_id,
    choice,
    noul,
    score,
)
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import MODEL_INFO, model

JEV = "openrouter/~typesafe/jev-latest"

LIVE_ANSWERS: dict[str, Any] = {
    "damaged": {"type": "noul", "noul": 0.98},
    "issue": {
        "type": "choice",
        "choice": "damage",
        "probabilities": {"damage": 1, "other": 0, "delay": 0},
        "confidence": 1,
    },
    "severity": {
        "type": "score",
        "score": 1.88,
        "legend": {"0": "minor", "1": "moderate", "2": "severe"},
        "probabilities": {"0": 0, "1": 0.12, "2": 0.88},
        "confidence": 0.82,
    },
}

QUESTIONS = {
    "damaged": noul("Was the delivered item damaged?"),
    "issue": choice("What kind of issue is this?", ["damage", "delay", "other"]),
    "severity": score("How severe is the problem?", ["minor", "moderate", "severe"]),
}


class _DecisionsHandler(BaseHTTPRequestHandler):
    """A ``/alpha/decisions`` endpoint whose behaviour is chosen per request.

    The ``state`` string selects the scenario so one server covers every
    response branch: ``"bad-request"`` → 400 with OpenRouter's error
    envelope, ``"unauthorized"`` → 401, ``"html-error"`` → 502 with a
    non-JSON body, ``"weird-status"`` → 599 (no HTTP reason phrase),
    ``"not-json"`` → 200 with a non-JSON body, ``"no-answers"`` → 200
    without an ``answers`` object, anything else → the recorded live
    answers.  Every request body is appended to ``requests`` for
    assertions.
    """

    requests: list[dict[str, Any]] = []
    auth_headers: list[str] = []
    headers_seen: list[dict[str, str]] = []

    def log_message(self, *_args: Any) -> None:  # noqa: D102 — silence the test log
        return

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = json.loads(self.rfile.read(length))
        _DecisionsHandler.requests.append({"path": self.path, "body": body})
        _DecisionsHandler.auth_headers.append(self.headers.get("Authorization", ""))
        _DecisionsHandler.headers_seen.append(dict(self.headers))
        state = body.get("state")
        if state == "bad-request":
            self._send(400, {"error": {"message": "Model x does not exist", "code": 400}})
        elif state == "unauthorized":
            self._send(401, {"error": {"message": "Missing Authentication header", "code": 401}})
        elif state == "html-error":
            self._send_raw(502, b"<html>bad gateway</html>")
        elif state == "weird-status":
            self._send_raw(599, b"upstream exploded")
        elif state == "not-json":
            self._send_raw(200, b"this is not json")
        elif state == "no-answers":
            self._send(200, {"model": "typesafe/jev-1.13-20260917", "usage": {}})
        else:
            self._send(
                200,
                {
                    "model": "typesafe/jev-1.13-20260917",
                    "answers": LIVE_ANSWERS,
                    "usage": {"input_tokens": 382, "output_tokens": 69, "cost": 1.6044e-05},
                    "id": "gen-dec-test",
                    "provider": "TypeSafe",
                },
            )

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        self._send_raw(status, json.dumps(payload).encode())

    def _send_raw(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _decisions_server() -> Iterator[str]:
    """Serve :class:`_DecisionsHandler` on a free port; yield the API root."""
    server = HTTPServer(("127.0.0.1", 0), _DecisionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _DecisionsHandler.requests = []
    _DecisionsHandler.auth_headers = []
    _DecisionsHandler.headers_seen = []
    try:
        yield f"http://127.0.0.1:{server.server_port}/api"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _free_closed_port() -> int:
    """Return a port nothing listens on, so a connection to it is refused."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_question_builders_produce_wire_shapes() -> None:
    """The three builders emit exactly what the endpoint accepts."""
    assert noul("Is it raining?") == {"type": "noul", "instructions": "Is it raining?"}
    assert choice("Pick", {"a": "first", "b": "second"}) == {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {"a": "first", "b": "second"},
    }
    assert choice("Pick", ["a", "b"])["criteria"] == {"a": "a", "b": "b"}
    assert score("Rate", ["low", "high"]) == {
        "type": "score",
        "instructions": "Rate",
        "criteria": ["low", "high"],
    }


def test_api_model_id_strips_routing_prefix() -> None:
    """Only KISS's ``openrouter/`` prefix is removed; the ``~`` alias marker stays."""
    assert api_model_id(JEV) == "~typesafe/jev-latest"
    assert api_model_id("typesafe/jev-1.13") == "typesafe/jev-1.13"


def test_decide_round_trip_sends_documented_request() -> None:
    """decide() posts ``{model, state, questions}`` with the Bearer key and returns the body."""
    with _decisions_server() as root:
        m = DecisionsModel(JEV, base_url=root + "/", api_key="sk-test", model_config={"timeout": 5})
        assert m.endpoint_url == root + "/alpha/decisions"
        assert m.timeout == 5.0
        response = m.decide("The package arrived crushed.", QUESTIONS)
    assert response["answers"] == LIVE_ANSWERS
    assert response["model"] == "typesafe/jev-1.13-20260917"
    [request] = _DecisionsHandler.requests
    assert request["path"] == "/api/alpha/decisions"
    assert request["body"] == {
        "model": "~typesafe/jev-latest",
        "state": "The package arrived crushed.",
        "questions": QUESTIONS,
    }
    assert _DecisionsHandler.auth_headers == ["Bearer sk-test"]
    assert m.extract_input_output_token_counts_from_response(response) == (382, 69, 0, 0)


def test_decide_sends_extra_headers_but_keeps_auth_authoritative() -> None:
    """``model_config["extra_headers"]`` (as ``custom_model_config`` builds it) rides along.

    A custom ``MY_MODELS.json`` endpoint can add headers such as
    ``X-Title``; the adapter forwards them, while its own Bearer token
    still wins over a conflicting ``Authorization`` entry.
    """
    with _decisions_server() as root:
        m = DecisionsModel(
            JEV,
            base_url=root,
            api_key="sk-real",
            model_config={"extra_headers": {"X-Title": "kiss-test", "Authorization": "Bearer x"}},
        )
        m.decide("x", QUESTIONS)
    [headers] = _DecisionsHandler.headers_seen
    assert headers["X-Title"] == "kiss-test"
    assert headers["Authorization"] == "Bearer sk-real"


def test_decide_accepts_structured_state() -> None:
    """``state`` may be a JSON object or array, forwarded verbatim."""
    with _decisions_server() as root:
        m = DecisionsModel(JEV, base_url=root, api_key="k")
        m.decide({"page": {"url": "https://x", "text": "hi"}, "elements": [1, 2]}, QUESTIONS)
    assert _DecisionsHandler.requests[0]["body"]["state"] == {
        "page": {"url": "https://x", "text": "hi"},
        "elements": [1, 2],
    }


def test_decide_rejects_bad_questions_before_any_request() -> None:
    """Empty or mistyped question sets fail locally, without touching the network."""
    m = DecisionsModel(JEV, base_url=f"http://127.0.0.1:{_free_closed_port()}", api_key="k")
    with pytest.raises(KISSError, match="at least one question"):
        m.decide("x", {})
    with pytest.raises(KISSError, match="'q' has type 'yesno'; expected one of noul, choice"):
        m.decide("x", {"q": {"type": "yesno", "instructions": "?"}})
    with pytest.raises(KISSError, match="has type None"):
        m.decide("x", {"q": {"instructions": "?"}})


def test_decide_reports_http_errors_with_reason_phrase() -> None:
    """Non-2xx responses become KISSErrors carrying status, reason and the server message."""
    with _decisions_server() as root:
        m = DecisionsModel(JEV, base_url=root, api_key="k")
        with pytest.raises(KISSError) as bad:
            m.decide("bad-request", QUESTIONS)
        with pytest.raises(KISSError) as unauthorized:
            m.decide("unauthorized", QUESTIONS)
        with pytest.raises(KISSError) as html:
            m.decide("html-error", QUESTIONS)
        with pytest.raises(KISSError) as weird:
            m.decide("weird-status", QUESTIONS)
    assert str(bad.value).endswith(
        "Decisions request failed (HTTP 400 Bad Request): Model x does not exist"
    )
    # "Unauthorized" is what kiss_agent._is_retryable_error keys on.
    assert "HTTP 401 Unauthorized" in str(unauthorized.value)
    assert "Missing Authentication header" in str(unauthorized.value)
    assert str(html.value).endswith(
        "Decisions request failed (HTTP 502 Bad Gateway): <html>bad gateway</html>"
    )
    assert str(weird.value).endswith("Decisions request failed (HTTP 599 ): upstream exploded")


def test_decide_rejects_malformed_success_bodies() -> None:
    """A 200 without parseable JSON or without ``answers`` is an error, not an empty result."""
    with _decisions_server() as root:
        m = DecisionsModel(JEV, base_url=root, api_key="k")
        with pytest.raises(KISSError, match="non-JSON body: 'this is not json'"):
            m.decide("not-json", QUESTIONS)
        with pytest.raises(KISSError, match="no 'answers' object"):
            m.decide("no-answers", QUESTIONS)


def test_decide_wraps_connection_failures() -> None:
    """An unreachable endpoint surfaces as a KISSError naming the URL."""
    root = f"http://127.0.0.1:{_free_closed_port()}"
    m = DecisionsModel(JEV, base_url=root, api_key="k", model_config={"timeout": 2})
    with pytest.raises(KISSError, match=f"Decisions request to {root}/alpha/decisions failed"):
        m.decide("x", QUESTIONS)


def test_generate_answers_configured_questions(caplog: pytest.LogCaptureFixture) -> None:
    """generate() judges the conversation text with the configured questions, streaming once."""
    tokens: list[str] = []
    with _decisions_server() as root:
        m = DecisionsModel(
            JEV,
            base_url=root,
            api_key="k",
            model_config={"questions": QUESTIONS},
            token_callback=tokens.append,
        )
        with caplog.at_level(logging.WARNING):
            m.initialize(
                "The package arrived crushed.",
                attachments=[Attachment(data=b"\x89PNG", mime_type="image/png")],
            )
        m.add_message_to_conversation("user", "and the item inside is broken.")
        text, raw = m.generate()
    assert "dropping 1 attachment(s)" in caplog.text
    assert json.loads(text) == LIVE_ANSWERS
    assert tokens == [text]
    assert raw["usage"]["input_tokens"] == 382
    assert m.conversation[-1] == {"role": "assistant", "content": text}
    assert _DecisionsHandler.requests[0]["body"]["state"] == (
        "The package arrived crushed.\n\nand the item inside is broken."
    )


def test_generate_without_questions_explains_how_to_use_the_model() -> None:
    """A text-style generate() on a decisions model fails with guidance, not a server 400."""
    m = DecisionsModel(JEV, base_url=f"http://127.0.0.1:{_free_closed_port()}", api_key="k")
    m.initialize("hello")
    with pytest.raises(KISSError, match=r'model_config=\{"questions": \{\.\.\.\}\}'):
        m.generate()


def test_unsupported_model_operations_raise() -> None:
    """Tool calling and embeddings are not part of the decisions protocol."""
    m = DecisionsModel(JEV, api_key="k")
    assert m.endpoint_url == OPENROUTER_DECISIONS_BASE_URL + "/alpha/decisions"
    m.initialize("x")
    with pytest.raises(KISSError, match="cannot call tools"):
        m.generate_and_process_with_tools({"f": len})
    with pytest.raises(KISSError, match="cannot embed"):
        m.get_embedding("x")


def test_token_counts_default_to_zero_for_odd_responses() -> None:
    """Missing or malformed usage yields zeros rather than an exception."""
    m = DecisionsModel(JEV, api_key="k")
    assert m.extract_input_output_token_counts_from_response(None) == (0, 0, 0, 0)
    assert m.extract_input_output_token_counts_from_response({"usage": "n/a"}) == (0, 0, 0, 0)
    assert m.extract_input_output_token_counts_from_response({"usage": {"input_tokens": 7}}) == (
        7,
        0,
        0,
        0,
    )


def test_catalog_flags_jev_as_decisions_only() -> None:
    """Both Jev catalog entries are decisions models and nothing else."""
    for name in (JEV, "openrouter/typesafe/jev-1.13"):
        info = MODEL_INFO[name]
        assert info.is_decisions_supported is True
        assert (info.is_generation_supported, info.is_function_calling_supported) == (False, False)
        assert info.is_embedding_supported is False
        assert (info.context_length, info.input_price_per_1M, info.output_price_per_1M) == (
            32000,
            0.042,
            0.0,
        )
    assert MODEL_INFO["openrouter/openai/gpt-4o"].is_decisions_supported is False


def test_factory_routes_dec_models_and_honours_overrides() -> None:
    """model() builds a DecisionsModel for a ``dec`` entry; base_url/api_key are consumed."""
    with _decisions_server() as root:
        m = model(
            JEV,
            model_config={
                "base_url": root,
                "api_key": "sk-override",
                "questions": QUESTIONS,
                "timeout": 7,
            },
        )
        assert isinstance(m, DecisionsModel)
        assert m.endpoint_url == root + "/alpha/decisions"
        assert m.model_config == {"questions": QUESTIONS, "timeout": 7}
        assert m.timeout == 7.0
        m.initialize("The package arrived crushed.")
        text, _raw = m.generate()
    assert json.loads(text) == LIVE_ANSWERS
    assert _DecisionsHandler.auth_headers == ["Bearer sk-override"]


def test_factory_defaults_to_openrouter_endpoint_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without overrides the factory targets OpenRouter with the configured OPENROUTER_API_KEY."""
    from kiss.core import config as config_module

    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "sk-from-config")
    m = model(JEV)
    assert isinstance(m, DecisionsModel)
    assert m.endpoint_url == "https://openrouter.ai/api/alpha/decisions"
    assert m.api_key == "sk-from-config"
    assert m.model_config == {}
    text_model = model("openrouter/openai/gpt-4o")
    assert not isinstance(text_model, DecisionsModel)


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_live_jev_answers_all_three_question_types() -> None:
    """Round-trip against OpenRouter: live answer shapes match what the local server replays."""
    m = model(JEV)
    assert isinstance(m, DecisionsModel)
    response = m.decide("The package arrived crushed and the item inside is broken.", QUESTIONS)
    answers = response["answers"]
    assert answers["damaged"]["type"] == "noul"
    assert 0.5 <= answers["damaged"]["noul"] <= 1.0
    assert answers["issue"]["choice"] == "damage"
    assert set(answers["issue"]["probabilities"]) == {"damage", "delay", "other"}
    assert answers["severity"]["legend"] == {"0": "minor", "1": "moderate", "2": "severe"}
    assert 0.0 <= answers["severity"]["score"] <= 2.0
    in_tokens, out_tokens, _cr, _cw = m.extract_input_output_token_counts_from_response(response)
    assert in_tokens > 0 and out_tokens >= 0
