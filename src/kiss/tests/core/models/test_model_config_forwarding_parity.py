# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: one policy for the public ``model_config`` dict.

``model_config`` is public API — ``Model(..., model_config={...})``, the
``sorcar`` CLI and ``speech_synthesis.py`` all set it — but each provider
adapter used to treat it differently: Anthropic forwarded every unclaimed
key into a keyword-only SDK call (``TypeError`` for anything it does not
declare), the two OpenAI transports forwarded whatever they had not
popped, and Gemini read a five-key allowlist and discarded the rest
without a word.  The same dict therefore behaved in three ways.

These tests pin one policy for every adapter:

1. Portable generation parameters (``temperature``, ``top_p``, ``stop``,
   ``max_tokens``) reach the wire in that provider's own spelling.
2. A parameter the provider supports but the adapter never named
   explicitly (``seed``) still reaches the wire.
3. A parameter the provider does **not** support is dropped with a
   warning that names the provider and the key — never a crash, and
   never in silence.

Every assertion is made against the JSON body a real local HTTP server
received from the real vendor SDK: no mocks, patches or doubles.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any

import anthropic
import pytest

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    serve,
    text_part,
)
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
)

_ANTHROPIC_MODEL = "claude-parity-under-test"
_GEMINI_MODEL = "gemini-parity-under-test"
_OPENAI_MODEL = "parity-under-test"


def _chat_reply(request: Request) -> Reply:
    """Answer a Chat Completions call with one short assistant message."""
    return Reply(
        json_body={
            "id": "chatcmpl-parity",
            "object": "chat.completion",
            "created": 0,
            "model": _OPENAI_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        }
    )


def _responses_reply(request: Request) -> Reply:
    """Answer a Responses call with one short assistant message."""
    return Reply(
        json_body={
            "id": "resp_parity",
            "object": "response",
            "created_at": 0,
            "model": _OPENAI_MODEL,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "ok", "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 4,
            },
        }
    )


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def anthropic_request_body(config: dict[str, Any]) -> dict[str, Any]:
    """Run one real Anthropic turn and return the recorded request body.

    Args:
        config: The ``model_config`` to hand the adapter.

    Returns:
        The JSON body the local ``/v1/messages`` endpoint received.
    """
    with ScriptedAnthropicServer() as server:
        model = AnthropicModel(
            _ANTHROPIC_MODEL, api_key="test-key", model_config=config
        )
        model.client = anthropic.Anthropic(
            api_key="test-key", base_url=server.base_url
        )
        model.conversation = [{"role": "user", "content": "Say ok."}]
        model.generate()
        return server.requests[-1]


def openai_chat_request_body(config: dict[str, Any]) -> dict[str, Any]:
    """Run one real Chat Completions turn and return the request body.

    Args:
        config: The ``model_config`` to hand the adapter.

    Returns:
        The JSON body the local ``/v1/chat/completions`` endpoint received.
    """
    with ScriptedOpenAIServer(_chat_reply) as server:
        model = OpenAICompatibleModel(
            _OPENAI_MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config=config,
        )
        model.initialize("Say ok.")
        model.generate()
        return server.requests[-1].body


def openai_responses_request_body(config: dict[str, Any]) -> dict[str, Any]:
    """Run one real Responses turn and return the request body.

    Args:
        config: The ``model_config`` to hand the adapter.

    Returns:
        The JSON body the local ``/v1/responses`` endpoint received.
    """
    with ScriptedOpenAIServer(_responses_reply) as server:
        model = OpenAICompatibleModel2(
            _OPENAI_MODEL,
            base_url=server.base_url,
            api_key="test-key",
            model_config=config,
        )
        model.initialize("Say ok.")
        model.generate()
        return server.requests[-1].body


def gemini_request_body(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: tuple[str, GeminiScript],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run one real Gemini turn and return the recorded request body.

    Args:
        monkeypatch: Fixture used to point the SDK at the local endpoint.
        endpoint: The ``(base_url, script)`` pair of the local endpoint.
        config: The ``model_config`` to hand the adapter.

    Returns:
        The JSON body the local ``generateContent`` endpoint received.
    """
    base_url, script = endpoint
    script.play([chunk([text_part("ok")])])
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(_GEMINI_MODEL, api_key="test-key", model_config=config)
    model.initialize("Say ok.")
    model.generate()
    return script.requests[-1]


class TestPortableParametersReachEveryProvider:
    """The four portable generation parameters must never be lost."""

    def test_anthropic_uses_its_own_spelling(self) -> None:
        """Anthropic gets ``stop_sequences`` and the rest verbatim."""
        body = anthropic_request_body(
            {"temperature": 0.25, "top_p": 0.9, "stop": ["END"], "max_tokens": 321}
        )
        assert body["temperature"] == 0.25
        assert body["top_p"] == 0.9
        assert body["stop_sequences"] == ["END"]
        assert body["max_tokens"] == 321

    def test_chat_completions_uses_its_own_spelling(self) -> None:
        """Chat Completions keeps ``stop`` as ``stop``."""
        body = openai_chat_request_body(
            {"temperature": 0.25, "top_p": 0.9, "stop": ["END"], "max_tokens": 321}
        )
        assert body["temperature"] == 0.25
        assert body["top_p"] == 0.9
        assert body["stop"] == ["END"]
        assert body["max_tokens"] == 321

    def test_gemini_uses_its_own_spelling(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """Gemini gets ``stopSequences`` / ``maxOutputTokens``."""
        body = gemini_request_body(
            monkeypatch,
            gemini_endpoint,
            {"temperature": 0.25, "top_p": 0.9, "stop": ["END"], "max_tokens": 321},
        )
        config = body["generationConfig"]
        assert config["temperature"] == 0.25
        assert config["topP"] == 0.9
        assert config["stopSequences"] == ["END"]
        assert config["maxOutputTokens"] == 321

    def test_responses_reports_the_parameter_it_cannot_send(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The Responses API has no ``stop``, so the drop must be announced."""
        with caplog.at_level(logging.WARNING):
            body = openai_responses_request_body(
                {
                    "temperature": 0.25,
                    "top_p": 0.9,
                    "stop": ["END"],
                    "max_tokens": 321,
                }
            )
        assert body["temperature"] == 0.25
        assert body["top_p"] == 0.9
        assert body["max_output_tokens"] == 321
        assert "stop" not in body
        assert "stop" in caplog.text


class TestSupportedButUnnamedParameterStillArrives:
    """A key the adapter never names must still reach a provider that takes it."""

    def test_chat_completions_forwards_seed(self) -> None:
        """``seed`` is a real Chat Completions parameter."""
        assert openai_chat_request_body({"seed": 7})["seed"] == 7

    def test_gemini_forwards_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """``seed`` is a real ``GenerateContentConfig`` field."""
        body = gemini_request_body(monkeypatch, gemini_endpoint, {"seed": 7})
        assert body["generationConfig"]["seed"] == 7

    def test_anthropic_reports_that_it_has_no_seed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Anthropic has no ``seed``: warn, do not raise ``TypeError``."""
        with caplog.at_level(logging.WARNING):
            body = anthropic_request_body({"seed": 7})
        assert "seed" not in body
        assert "seed" in caplog.text


class TestUnsupportedParameterIsAnnouncedNotCrashed:
    """An unknown key must produce a warning on every provider."""

    _UNKNOWN = {"totally_unknown_provider_param": 1}

    def _assert_reported(
        self, body: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Assert the key never shipped and the drop was announced."""
        assert "totally_unknown_provider_param" not in json.dumps(body)
        assert "totally_unknown_provider_param" in caplog.text

    def test_anthropic(self, caplog: pytest.LogCaptureFixture) -> None:
        """Anthropic must not raise ``TypeError`` on an unknown key."""
        with caplog.at_level(logging.WARNING):
            body = anthropic_request_body(dict(self._UNKNOWN))
        self._assert_reported(body, caplog)

    def test_chat_completions(self, caplog: pytest.LogCaptureFixture) -> None:
        """Chat Completions must not raise ``TypeError`` on an unknown key."""
        with caplog.at_level(logging.WARNING):
            body = openai_chat_request_body(dict(self._UNKNOWN))
        self._assert_reported(body, caplog)

    def test_responses(self, caplog: pytest.LogCaptureFixture) -> None:
        """Responses must not raise ``TypeError`` on an unknown key."""
        with caplog.at_level(logging.WARNING):
            body = openai_responses_request_body(dict(self._UNKNOWN))
        self._assert_reported(body, caplog)

    def test_gemini(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Gemini must announce the key it drops instead of staying silent."""
        with caplog.at_level(logging.WARNING):
            body = gemini_request_body(
                monkeypatch, gemini_endpoint, dict(self._UNKNOWN)
            )
        self._assert_reported(body, caplog)
