# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the group E/F model fixes of audit 2026-10-21.

* **F-F1 / F-F2** — :class:`GeminiModel` always streams.  The unary
  ``generate_content`` path shared the byte-level stall clock with the
  stream, so a response that took longer than ``stream_stall_timeout``
  to generate could never succeed when no token callback was bound, and
  an empty stream used to trigger a second billable request.
* **F-F3** — :class:`AnthropicModel.initialize` closes the SDK client it
  replaces instead of leaking its connection pool.
* **E-F2** — the text-based tools prompt a DeepSeek R1 tool turn sends
  is byte-identical to the one produced before the signature walk was
  shared with the OpenAI schema builder (captured output, not a
  re-implementation).
* **F-F5** — a bare-string ``stop`` reaches Gemini and Anthropic as a
  one-element list, and ``max_completion_tokens`` as their token cap.

Everything runs against real local HTTP servers and the real SDKs: no
mocks, patches or test doubles.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    serve,
    text_part,
)
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

_STALL = 1.0
_CHUNK_DELAY = 0.4
_TOKENS = ["one ", "two ", "three ", "four ", "five"]


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _gemini(monkeypatch: pytest.MonkeyPatch, base_url: str) -> GeminiModel:
    """A real ``GeminiModel`` with no token callback and a short stall clock."""
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(
        "gemini-under-test",
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL},
    )
    model.initialize("Count to five.")
    return model


def _slow_answer() -> list[dict[str, Any]]:
    """Five chunks whose total generation time exceeds the stall clock."""
    chunks = [chunk([text_part(token)]) for token in _TOKENS]
    chunks[-1]["usageMetadata"] = {"promptTokenCount": 10, "candidatesTokenCount": 5}
    return chunks


class TestGeminiAlwaysStreams:
    """F-F1/F-F2: no token callback must not mean a capped generation time."""

    def test_no_callback_turn_longer_than_the_stall_clock_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
    ) -> None:
        """Bytes keep arriving, so the stall clock never fires."""
        base_url, script = gemini_endpoint
        script.play(_slow_answer(), delay=_CHUNK_DELAY)
        assert len(_TOKENS) * _CHUNK_DELAY > _STALL
        model = _gemini(monkeypatch, base_url)

        started = time.monotonic()
        content, response = model.generate()

        assert content == "".join(_TOKENS)
        assert time.monotonic() - started >= len(_TOKENS) * _CHUNK_DELAY
        assert model.extract_input_output_token_counts_from_response(response) == (10, 5, 0, 0)
        assert [p.rsplit(":", 1)[-1].split("?")[0] for p in script.paths] == [
            "streamGenerateContent"
        ]

    def test_empty_stream_sends_exactly_one_request(
        self, monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
    ) -> None:
        """A stream yielding nothing is an empty turn, not a second request."""
        base_url, script = gemini_endpoint
        script.play([])
        model = _gemini(monkeypatch, base_url)

        content, response = model.generate()

        assert content == ""
        assert response is None
        assert model.extract_input_output_token_counts_from_response(response) == (0, 0, 0, 0)
        assert len(script.paths) == 1
        assert "streamGenerateContent" in script.paths[0]


class TestAnthropicInitializeClosesTheReplacedClient:
    """F-F3: a rebuilt SDK client must not leak the previous connection pool."""

    def test_changed_workspace_closes_the_old_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A new workspace id rebuilds the client and closes the old one."""
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_one")
        model = AnthropicModel("claude-under-test", api_key="key-1")
        model.initialize("first run")
        old = model.client
        assert old is not None and not old.is_closed()

        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_two")
        model.initialize("second run")

        assert model.client is not old
        assert old.is_closed()
        assert not model.client.is_closed()

    def test_unchanged_inputs_keep_the_open_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same inputs: the client is reused and stays open."""
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_one")
        model = AnthropicModel("claude-under-test", api_key="key-1")
        model.initialize("first run")
        client = model.client

        model.initialize("second run")

        assert model.client is client
        assert not client.is_closed()


class TestOpenAIInitializeClosesTheReplacedClient:
    """Same contract for the OpenAI adapter: a rebuilt client closes the old one."""

    def test_changed_api_key_closes_the_old_client(self) -> None:
        """A new API key rebuilds the client and closes the old pool."""
        model = OpenAICompatibleModel("gpt-under-test", base_url="http://127.0.0.1:9", api_key="k1")
        model.initialize("first run")
        old = model.client
        assert old is not None and not old.is_closed()

        model.api_key = "k2"
        model.initialize("second run")

        assert model.client is not old
        assert old.is_closed()
        assert not model.client.is_closed()

    def test_unchanged_inputs_keep_the_open_client(self) -> None:
        """Same inputs: the client is reused and stays open."""
        model = OpenAICompatibleModel("gpt-under-test", base_url="http://127.0.0.1:9", api_key="k1")
        model.initialize("first run")
        client = model.client

        model.initialize("second run")

        assert model.client is client
        assert not client.is_closed()


def _read_file(path: str, count: int = 3, tags: list[str] | None = None, extra=None) -> str:  # type: ignore[no-untyped-def]
    """Read a file.

    Second paragraph that must not appear.

    Args:
        path: The file path to read.
        count: How many lines.
    """
    return ""


def _bare():  # type: ignore[no-untyped-def]
    return None


def _pep563(path, items, flag):  # type: ignore[no-untyped-def]
    """Simulated PEP 563 tool."""
    return ""


# What ``from __future__ import annotations`` leaves on a function: strings.
_pep563.__annotations__ = {
    "path": "str",
    "items": "list[str]",
    "flag": "bool | None",
    "return": "str",
}

# Captured from ``_build_text_based_tools_prompt`` before the signature walk
# was shared with the OpenAI schema builder.  This module uses
# ``from __future__ import annotations``, so ``_read_file``'s annotations
# are strings and print verbatim, exactly as a SEA's tools do.
_CAPTURED_PROMPT = (
    "\n## Available Tools\n\nTo call a tool, output a JSON object in the following format:\n\n"
    '```json\n{"tool_calls": [{"name": "tool_name", "arguments": {"arg1": "value1", '
    '"arg2": "value2"}}]}\n```\n\nYou can call multiple tools at once by including '
    "multiple objects in the tool_calls array.\n\n### Tools:\n"
    "- **read_file**: Read a file.\n  Parameters:\n    - path (str)\n    - count (int)\n"
    "    - tags (list[str] | None)\n    - extra (any)\n"
    "- **no_doc**: Function no_doc\n  Parameters:\n    (no parameters)\n"
    "- **pep563**: Simulated PEP 563 tool.\n  Parameters:\n    - path (str)\n"
    "    - items (list[str])\n    - flag (bool | None)\n\n"
    "CRITICAL RULES:\n"
    "1. When you want to call a tool, output ONLY the JSON object with tool_calls.\n"
    "2. STOP IMMEDIATELY after outputting the tool_calls JSON. Do NOT continue generating.\n"
    "3. Do NOT predict, simulate, or hallucinate what the tool results will be.\n"
    "4. The system will execute the tools and provide actual results in the next message.\n"
    "5. When you have the final answer, call the `finish` tool with your result.\n"
)


def _plain_chat_reply(request: Request) -> Reply:
    """Answer any chat request with a short assistant text."""
    return Reply(
        json_body={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-r1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def test_text_tools_prompt_is_byte_identical_to_the_captured_output() -> None:
    """E-F2: the shared signature walk changes no byte of the transmitted prompt.

    DeepSeek R1 is the OpenAI-compatible model that uses text-based tool
    calling, so a tool turn against a local endpoint shows the prompt
    exactly as it leaves the process: appended to the first user message.
    """
    with ScriptedOpenAIServer(_plain_chat_reply) as server:
        model = OpenAICompatibleModel(
            "openrouter/deepseek/deepseek-r1", base_url=server.base_url, api_key="sk-test"
        )
        model.initialize("hi")
        model.generate_and_process_with_tools(
            {"read_file": _read_file, "no_doc": _bare, "pep563": _pep563}
        )
        (request,) = server.requests
    assert request.body["messages"][0] == {
        "role": "user",
        "content": "hi\n" + _CAPTURED_PROMPT,
    }


_STOP_CONFIG = {"stop": "END", "max_completion_tokens": 77}


class TestStopAliasReachesBothProviders:
    """F-F5: one alias resolver, so a bare-string ``stop`` is a list on the wire."""

    def test_gemini_wraps_a_bare_string_stop(
        self, monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
    ) -> None:
        """Gemini's ``stopSequences`` must be a list even for ``stop="END"``."""
        base_url, script = gemini_endpoint
        script.play([chunk([text_part("ok")])])
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
        model = GeminiModel("gemini-under-test", api_key="test-key", model_config=_STOP_CONFIG)
        model.initialize("hi")

        model.generate()

        (request,) = script.requests
        assert request["generationConfig"]["stopSequences"] == ["END"]
        assert request["generationConfig"]["maxOutputTokens"] == 77

    def test_anthropic_wraps_a_bare_string_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anthropic resolves the same aliases through the same helper."""
        with ScriptedAnthropicServer() as server:
            monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
            model = AnthropicModel("claude-under-test", api_key="key-1", model_config=_STOP_CONFIG)
            model.initialize("hi")

            model.generate()

            (request,) = server.requests
        assert request["stop_sequences"] == ["END"]
        assert request["max_tokens"] == 77
