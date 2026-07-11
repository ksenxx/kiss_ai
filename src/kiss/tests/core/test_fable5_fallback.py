# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``claude-fable-5`` non-retryable + fallback fix.

Bug reproduction (production failures in ``~/.kiss/sorcar.db``, all-time,
model = ``claude-fable-5``):

* 5+ tasks failed with Anthropic 404 whose body reads ``"Claude Fable 5
  is not available. Please use Opus 4.8"`` — KISSAgent retried 3× on the
  same non-retryable error and died.
* 10+ tasks died with ``"credit balance is too low"`` (Anthropic 400) —
  same problem, wasted retries against a dead credit balance.

Fix landed:

1. ``_NON_RETRYABLE_PHRASES`` now includes ``"is not available"``,
   ``"not_found_error"``, ``"credit balance is too low"`` so the retry
   loop shortcircuits.
2. ``MODEL_INFO`` entries may declare a ``fallback`` model name.
   ``claude-fable-5`` has ``"fallback": "claude-opus-4-8"``.
3. On a non-retryable error, ``KISSAgent._try_switch_to_fallback``
   rebuilds the model to the registered fallback (preserving the
   conversation history and the caller's ``model_config`` overrides),
   refreshes the cached tool schema, and the loop transparently
   continues on the fallback.  One-shot guard prevents A→B→A cycles.

These tests exercise the whole path via a local HTTP capture server
using the same pattern as ``test_empty_response_silent_death.py`` —
first request (from the primary model) returns a non-retryable error;
second request (from the fallback model) returns a normal ``finish``
tool call.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from kiss.core.kiss_agent import _NON_RETRYABLE_PHRASES, KISSAgent
from kiss.core.models import model_info as model_info_module
from kiss.core.models.model_info import (
    MODEL_INFO,
    ModelInfo,
    get_fallback_model,
)


class TestNonRetryablePhrases:
    """Direct unit checks on the ``_NON_RETRYABLE_PHRASES`` tuple."""

    def test_new_phrases_are_registered(self) -> None:
        """The three phrases fable-5 needs must be in the tuple."""
        assert "is not available" in _NON_RETRYABLE_PHRASES
        assert "not_found_error" in _NON_RETRYABLE_PHRASES
        assert "credit balance is too low" in _NON_RETRYABLE_PHRASES

    def test_phrases_are_lowercase(self) -> None:
        """``_is_retryable_error`` lowercases ``str(e)`` before checking,
        so every phrase must be stored lowercase to match reliably."""
        for phrase in _NON_RETRYABLE_PHRASES:
            assert phrase == phrase.lower(), phrase


class TestGetFallbackModel:
    """``get_fallback_model`` MODEL_INFO lookup."""

    def test_claude_fable_5_falls_back_to_opus_4_8(self) -> None:
        assert get_fallback_model("claude-fable-5") == "claude-opus-4-8"

    def test_unknown_model_returns_none(self) -> None:
        assert get_fallback_model("does-not-exist-xyz") is None

    def test_model_without_fallback_returns_none(self) -> None:
        """A registered model that does not declare ``fallback``
        returns ``None`` (not an error)."""
        # ``claude-opus-4-8`` itself has no ``fallback`` field.
        assert get_fallback_model("claude-opus-4-8") is None

    def test_harbor_prefix_is_stripped(self) -> None:
        """``get_fallback_model`` accepts harbor-style ``provider/name``
        input (matching the behavior of ``get_max_context_length``)."""
        assert get_fallback_model("anthropic/claude-fable-5") == "claude-opus-4-8"


def _finish_tool_call_response(summary: str = "done") -> dict[str, Any]:
    """OpenAI-compatible response invoking the built-in ``finish`` tool."""
    return {
        "id": "chatcmpl-fallback-ok",
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
                                "arguments": json.dumps({"result": summary}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _make_switching_handler(
    primary_model: str,
    fallback_model: str,
    error_status: int,
    error_body: dict[str, Any],
    seen: dict[str, list[str]],
) -> type[BaseHTTPRequestHandler]:
    """Build an HTTP handler that returns *error_body* for the primary model
    and a ``finish`` tool call for the fallback model.

    ``seen`` accumulates the ``"model"`` field of every request body so the
    test can assert the second request actually went to the fallback.
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            model_name = payload.get("model", "")
            seen.setdefault("models", []).append(model_name)
            if model_name == primary_model:
                body = json.dumps(error_body).encode()
                self.send_response(error_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            assert model_name == fallback_model, (
                f"Unexpected model in request: {model_name!r}"
            )
            body = json.dumps(_finish_tool_call_response()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    return _Handler


def _register_synthetic_pair(
    monkeypatch: Any, primary: str, fallback: str
) -> None:
    """Register two synthetic OpenAI-compatible model entries in
    ``MODEL_INFO`` for the duration of one test, with ``primary`` declaring
    ``fallback`` as its fallback.

    Using ``gpt-*`` names ensures the ``model()`` factory routes both
    entries through ``_openai_compatible`` (which honors the caller's
    ``model_config['base_url']``) — we deliberately avoid touching the
    real ``claude-fable-5`` entry so the test does not depend on
    Anthropic credentials or SDK routing.
    """
    for name, fb in ((primary, fallback), (fallback, None)):
        monkeypatch.setitem(
            MODEL_INFO,
            name,
            ModelInfo(
                context_length=128_000,
                input_price_per_million=0.0,
                output_price_per_million=0.0,
                is_function_calling_supported=True,
                is_embedding_supported=False,
                is_generation_supported=True,
                fallback=fb,
            ),
        )


class TestFallbackEndToEnd:
    """End-to-end swap on non-retryable provider errors."""

    def _run(
        self,
        monkeypatch: Any,
        error_status: int,
        error_body: dict[str, Any],
    ) -> tuple[KISSAgent, str, list[str]]:
        primary = "gpt-primary-under-test"
        fallback = "gpt-fallback-under-test"
        _register_synthetic_pair(monkeypatch, primary, fallback)
        seen: dict[str, list[str]] = {}
        handler = _make_switching_handler(
            primary, fallback, error_status, error_body, seen
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            agent = KISSAgent("test-fallback")
            result = agent.run(
                model_name=primary,
                prompt_template="hi",
                max_steps=5,
                max_budget=1.0,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "sk-test",
                },
            )
        finally:
            server.shutdown()
        return agent, result, seen["models"]

    def test_404_not_available_triggers_fallback(
        self, monkeypatch: Any
    ) -> None:
        """A 404 body containing ``not_found_error`` + ``is not
        available`` (Anthropic's actual response for gated fable-5)
        must trigger a swap to the fallback model, and the agent must
        complete on the fallback."""
        agent, result, models_seen = self._run(
            monkeypatch,
            error_status=404,
            error_body={
                "error": {
                    "type": "not_found_error",
                    "message": (
                        "Claude Fable 5 is not available. Please use "
                        "Opus 4.8"
                    ),
                }
            },
        )
        assert result == "done"
        assert agent.model_name == "gpt-fallback-under-test"
        assert agent._fallback_used is True
        assert models_seen[0] == "gpt-primary-under-test"
        assert "gpt-fallback-under-test" in models_seen[1:]

    def test_credit_balance_too_low_triggers_fallback(
        self, monkeypatch: Any
    ) -> None:
        """A 400 with ``"credit balance is too low"`` must also fall
        back (this was 10+ real production failures)."""
        agent, result, models_seen = self._run(
            monkeypatch,
            error_status=400,
            error_body={
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "Your credit balance is too low to access the "
                        "Anthropic API."
                    ),
                }
            },
        )
        assert result == "done"
        assert agent.model_name == "gpt-fallback-under-test"
        assert models_seen[0] == "gpt-primary-under-test"

    def test_no_fallback_registered_raises_kiss_error(
        self, monkeypatch: Any
    ) -> None:
        """If the failing model has no ``fallback`` registered, the
        agent must raise ``KISSError`` (preserving the previous
        behavior)."""
        from kiss.core.kiss_error import KISSError
        primary = "gpt-no-fallback-under-test"
        monkeypatch.setitem(
            MODEL_INFO,
            primary,
            ModelInfo(
                context_length=128_000,
                input_price_per_million=0.0,
                output_price_per_million=0.0,
                is_function_calling_supported=True,
                is_embedding_supported=False,
                is_generation_supported=True,
                fallback=None,
            ),
        )
        seen: dict[str, list[str]] = {}
        handler = _make_switching_handler(
            primary,
            "unused-fallback",
            404,
            {
                "error": {
                    "type": "not_found_error",
                    "message": "Model is not available.",
                }
            },
            seen,
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            agent = KISSAgent("test-no-fallback")
            import pytest as _pytest
            with _pytest.raises(KISSError) as excinfo:
                agent.run(
                    model_name=primary,
                    prompt_template="hi",
                    max_steps=5,
                    max_budget=1.0,
                    verbose=False,
                    model_config={
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-test",
                    },
                )
            assert "non-retryable" in str(excinfo.value).lower()
        finally:
            server.shutdown()

    def test_fallback_one_shot_guard(self, monkeypatch: Any) -> None:
        """When the fallback itself fails with a non-retryable error,
        the agent must NOT loop forever — it must raise ``KISSError``.

        We register ``A → B`` and make the server return a 404 for
        both.  Expected: primary hits 404 → swap to B → B hits 404 →
        ``_fallback_used`` is True so ``_try_switch_to_fallback``
        returns ``None`` and the loop raises ``KISSError``.
        """
        from kiss.core.kiss_error import KISSError
        primary = "gpt-loop-primary"
        fallback = "gpt-loop-fallback"
        _register_synthetic_pair(monkeypatch, primary, fallback)

        class _AlwaysErrorHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    self.rfile.read(length)
                body = json.dumps(
                    {
                        "error": {
                            "type": "not_found_error",
                            "message": "Model is not available.",
                        }
                    }
                ).encode()
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = HTTPServer(("127.0.0.1", 0), _AlwaysErrorHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            agent = KISSAgent("test-fallback-loop")
            import pytest as _pytest
            with _pytest.raises(KISSError):
                agent.run(
                    model_name=primary,
                    prompt_template="hi",
                    max_steps=5,
                    max_budget=1.0,
                    verbose=False,
                    model_config={
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-test",
                    },
                )
            # The swap must have happened exactly once.
            assert agent._fallback_used is True
            assert agent.model_name == fallback
        finally:
            server.shutdown()


class TestModelInfoJsonHasFallback:
    """Sanity: ``MODEL_INFO.json`` still declares the fable-5 fallback.

    Prevents accidental deletion during future JSON edits (the whole
    point of this feature)."""

    def test_claude_fable_5_entry_has_fallback(self) -> None:
        info = MODEL_INFO["claude-fable-5"]
        assert info.fallback == "claude-opus-4-8"

    def test_openrouter_fable_5_entry_has_fallback(self) -> None:
        """The OpenRouter mirror of fable-5 is a separate MODEL_INFO key
        (harbor prefix ``openrouter/`` is NOT stripped by
        ``_strip_provider_prefix``), so it needs its own ``fallback``
        entry.  Users routing through OpenRouter otherwise get no
        fallback at all."""
        info = MODEL_INFO["openrouter/anthropic/claude-fable-5"]
        assert info.fallback == "openrouter/anthropic/claude-opus-4.8"
        assert "openrouter/anthropic/claude-opus-4.8" in MODEL_INFO

    def test_module_reference_still_present(self) -> None:
        """Guard against accidental removal of ``get_fallback_model``."""
        assert hasattr(model_info_module, "get_fallback_model")
