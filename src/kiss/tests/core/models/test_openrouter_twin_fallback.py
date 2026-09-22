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

The end-to-end case runs a real ``KISSAgent`` against a local HTTP server
that rejects the primary model with the Anthropic credit error and
answers the twin with a ``finish`` tool call.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from kiss.core import config as config_module
from kiss.core.kiss_agent import KISSAgent
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


def _credit_error_handler(primary: str, seen: list[str]) -> type[BaseHTTPRequestHandler]:
    """Reject *primary* with Anthropic's credit error; finish for any other model."""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            model_name = payload.get("model", "")
            seen.append(model_name)
            if model_name == primary:
                body = json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Your credit balance is too low to access the "
                            "Anthropic API. Please go to Plans & Billing to upgrade.",
                        },
                    }
                ).encode()
                self.send_response(400)
            else:
                body = json.dumps(_finish_response()).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    return _Handler


class TestTwinFallbackEndToEnd:
    """A credit-balance rejection on a model without an explicit fallback
    continues on its OpenRouter twin instead of raising ``KISSError``."""

    def test_agent_switches_to_twin(self, monkeypatch: Any) -> None:
        primary = "gpt-twin-primary-under-test"
        twin = "openrouter/synthetic/gpt-twin-primary-under-test"
        monkeypatch.setitem(MODEL_INFO, primary, _entry())
        monkeypatch.setitem(MODEL_INFO, twin, _entry())
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "sk-or-test")
        assert get_fallback_model(primary) == twin
        seen: list[str] = []
        server = HTTPServer(("127.0.0.1", 0), _credit_error_handler(primary, seen))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            agent = KISSAgent("twin-fallback")
            result = agent.run(
                model_name=primary,
                prompt_template="hi",
                max_steps=5,
                max_budget=1.0,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                    "api_key": "sk-test",
                },
            )
        finally:
            server.shutdown()
        assert result == "done on twin"
        assert agent.model_name == twin
        assert seen[0] == primary and len(seen) == 2
        assert seen[1] != primary
