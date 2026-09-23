# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the automatic OpenRouter-twin fallback.

Production failure (``~/.kiss/sorcar.db``, task ``bcdb7b63``, 2026-09-22):
a 3 h 13 min / $24 task on ``claude-fable-5-1`` died with ``KISSError:
Non-retryable error from model: ... credit balance is too low`` because
that catalog entry declares no ``fallback`` (only 2 of 664 entries did).
``get_fallback_model`` now derives one: the ``openrouter/<vendor>/<name>``
entry serving the same model, used only when ``OPENROUTER_API_KEY`` is
configured.  An explicit ``fallback`` still wins.

The end-to-end case runs a real ``KISSAgent`` against two local HTTP
servers: the caller's ``base_url`` rejects the primary model with the
Anthropic credit error, and a stand-in for openrouter.ai (the OpenRouter
provider's ``base_url`` is re-pointed at it) answers the twin with a
``finish`` tool call.  The twin is a different provider, so the swap must
leave the caller's ``base_url``/``api_key`` behind and use the OpenRouter
route and key; a ``model_config`` copied wholesale sent the "fallback"
request back to the endpoint that had just rejected it.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from kiss.core import config as config_module
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models import model_info
from kiss.core.models.model_info import (
    MODEL_INFO,
    ModelInfo,
    get_fallback_model,
    openrouter_twin,
)


def _entry(fallback: str | None = None) -> ModelInfo:
    return ModelInfo(
        context_length=128_000,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
        is_function_calling_supported=True,
        is_embedding_supported=False,
        is_generation_supported=True,
        fallback=fallback,
    )


class TestOpenrouterTwin:
    """Name matching between direct-provider and OpenRouter catalog keys."""

    def test_dash_and_dot_versions_match(self) -> None:
        assert openrouter_twin("claude-fable-5-1") == "openrouter/anthropic/claude-fable-5.1"

    def test_openai_model_matches(self) -> None:
        assert openrouter_twin("gpt-5.6-sol") == "openrouter/openai/gpt-5.6-sol"

    def test_routed_names_have_no_twin(self) -> None:
        assert openrouter_twin("openrouter/openai/gpt-5.6-sol") is None
        assert openrouter_twin("cc/claude-fable-5") is None

    def test_harbor_prefix_is_stripped(self) -> None:
        assert openrouter_twin("anthropic/claude-fable-5-1") == (
            "openrouter/anthropic/claude-fable-5.1"
        )

    def test_unknown_name_has_no_twin(self) -> None:
        assert openrouter_twin("no-such-model-xyz") is None


class TestGetFallbackModelDerivesTwin:
    """``get_fallback_model`` policy: explicit > twin (with key) > None."""

    def test_twin_used_when_key_configured(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "sk-or-test")
        assert get_fallback_model("claude-fable-5-1") == "openrouter/anthropic/claude-fable-5.1"
        assert get_fallback_model("gpt-5.6-sol") == "openrouter/openai/gpt-5.6-sol"

    def test_no_twin_without_key(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "")
        assert get_fallback_model("claude-fable-5-1") is None

    def test_explicit_fallback_wins(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "sk-or-test")
        assert get_fallback_model("claude-fable-5") == "claude-opus-4-8"

    def test_unknown_model_is_none(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "sk-or-test")
        assert get_fallback_model("does-not-exist-xyz") is None


def _finish_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-twin",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_finish",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps({"result": "done on twin"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _send_json(handler: BaseHTTPRequestHandler, status: int, body: dict[str, Any]) -> None:
    raw = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


_CREDIT_ERROR = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "Your credit balance is too low to access the "
        "Anthropic API. Please go to Plans & Billing to upgrade.",
    },
}


def _recording_handler(
    seen: list[dict[str, Any]], status: int, body: dict[str, Any]
) -> type[BaseHTTPRequestHandler]:
    """Record every request's model and bearer token, then answer *status*/*body*."""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            seen.append({
                "model": payload.get("model", ""),
                "authorization": self.headers.get("Authorization"),
            })
            _send_json(self, status, body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    return _Handler


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


class TestTwinFallbackEndToEnd:
    """A credit-balance rejection on a model without an explicit fallback
    continues on its OpenRouter twin, through OpenRouter, instead of
    raising ``KISSError``."""

    def test_agent_switches_to_twin_on_the_openrouter_route(self, monkeypatch: Any) -> None:
        primary = "gpt-twin-primary-under-test"
        twin = "openrouter/synthetic/gpt-twin-primary-under-test"
        monkeypatch.setitem(MODEL_INFO, primary, _entry())
        monkeypatch.setitem(MODEL_INFO, twin, _entry())
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "sk-or-test")
        assert get_fallback_model(primary) == twin

        primary_seen: list[dict[str, Any]] = []
        twin_seen: list[dict[str, Any]] = []
        primary_server, primary_url = _serve(_recording_handler(primary_seen, 400, _CREDIT_ERROR))
        twin_server, twin_url = _serve(_recording_handler(twin_seen, 200, _finish_response()))
        # Stand in for openrouter.ai: the twin must arrive here, not at
        # the caller's ``base_url``.
        monkeypatch.setattr(
            model_info,
            "OPENAI_COMPATIBLE_PROVIDERS",
            tuple(
                dataclasses.replace(p, base_url=twin_url) if p.name == "openrouter" else p
                for p in model_info.OPENAI_COMPATIBLE_PROVIDERS
            ),
        )
        try:
            agent = KISSAgent("twin-fallback")
            result = agent.run(
                model_name=primary,
                prompt_template="hi",
                max_steps=5,
                max_budget=1.0,
                verbose=False,
                model_config={
                    "base_url": primary_url,
                    "api_key": "sk-test",
                    # A custom header is the other way a credential can
                    # ride along; it must not reach OpenRouter either.
                    "extra_headers": {"Authorization": "Bearer sk-primary-header"},
                },
            )
        finally:
            primary_server.shutdown()
            twin_server.shutdown()
        assert result == "done on twin"
        assert agent.model_name == twin
        assert [r["model"] for r in primary_seen] == [primary]
        assert primary_seen[0]["authorization"] == "Bearer sk-primary-header"
        assert [r["model"] for r in twin_seen] == ["synthetic/gpt-twin-primary-under-test"]
        assert twin_seen[0]["authorization"] == "Bearer sk-or-test"

    def test_declared_fallback_naming_the_twin_keeps_the_callers_endpoint(
        self, monkeypatch: Any
    ) -> None:
        """A fallback the user declared stays on the user's endpoint and key
        even when it happens to be spelled like the OpenRouter twin."""
        primary = "gpt-declared-twin-under-test"
        twin = "openrouter/synthetic/gpt-declared-twin-under-test"
        monkeypatch.setitem(MODEL_INFO, primary, _entry(fallback=twin))
        monkeypatch.setitem(MODEL_INFO, twin, _entry())
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "sk-or-test")
        assert get_fallback_model(primary) == twin == openrouter_twin(primary)

        seen: list[dict[str, Any]] = []
        openrouter_seen: list[dict[str, Any]] = []

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                seen.append({
                    "model": payload.get("model", ""),
                    "authorization": self.headers.get("Authorization"),
                })
                if payload["model"] == primary:
                    _send_json(self, 400, _CREDIT_ERROR)
                else:
                    _send_json(self, 200, _finish_response())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server, url = _serve(_Handler)
        openrouter_server, openrouter_url = _serve(
            _recording_handler(openrouter_seen, 200, _finish_response())
        )
        monkeypatch.setattr(
            model_info,
            "OPENAI_COMPATIBLE_PROVIDERS",
            tuple(
                dataclasses.replace(p, base_url=openrouter_url) if p.name == "openrouter" else p
                for p in model_info.OPENAI_COMPATIBLE_PROVIDERS
            ),
        )
        try:
            agent = KISSAgent("declared-twin-fallback")
            result = agent.run(
                model_name=primary,
                prompt_template="hi",
                max_steps=5,
                max_budget=1.0,
                verbose=False,
                model_config={"base_url": url, "api_key": "sk-test"},
            )
        finally:
            server.shutdown()
            openrouter_server.shutdown()
        assert result == "done on twin"
        assert agent.model_name == twin
        assert [r["model"] for r in seen] == [primary, "synthetic/gpt-declared-twin-under-test"]
        assert {r["authorization"] for r in seen} == {"Bearer sk-test"}
        assert openrouter_seen == []
