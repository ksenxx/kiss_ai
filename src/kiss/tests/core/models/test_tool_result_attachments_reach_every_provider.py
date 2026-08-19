# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: what a tool returns must reach the model, or be declared.

A tool result travels as a plain string, so a tool that produces bytes —
``Read`` on a screenshot, a camera capture, a voice memo — embeds them in a
sentinel that :func:`parse_binary_attachments` lifts back out.  Whether the
model then *sees* those bytes was decided differently by every adapter: the
network transports re-attached them, while the two CLI-backed transports
inherited a base implementation that parsed the sentinel and threw the
attachments away, leaving the model with a ``[attached image/png, N bytes]`` placeholder it
could neither see nor know was missing.

The same divergence covered two smaller ones: a structured (non-string)
tool result crashed on two adapters and worked on the others, and only one
adapter honoured the ``tool_use_id`` its own public signature documents.

These tests pin all three across every adapter, driving the real vendor
SDKs against real local HTTP servers and the real CLI transports against
real stand-in executables on ``PATH``.  No mocks, patches or doubles.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import anthropic
import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model import encode_binary_attachment
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
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
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli
from kiss.tests.core.models.test_heif_attachment import _write_gradient_png


def _payload_is_present(body: dict[str, Any], data: bytes) -> bool:
    """Return whether *data* was carried in a recorded request body.

    Args:
        body: The decoded JSON body the provider received.
        data: The raw attachment bytes to look for.

    Returns:
        ``True`` when the bytes appear base64-encoded anywhere in the body.
        Both alphabets are checked because the Gemini SDK emits URL-safe
        base64 (``-``/``_``) for inline data while the others use the
        standard one.
    """
    serialized = json.dumps(body)
    return (
        base64.b64encode(data).decode("ascii") in serialized
        or base64.urlsafe_b64encode(data).decode("ascii") in serialized
    )


_ANTHROPIC_MODEL = "claude-tool-attachment-under-test"
_GEMINI_MODEL = "gemini-tool-attachment-under-test"
_OPENAI_MODEL = "tool-attachment-under-test"

_TOOL_NAME = "Read"
_TOOL_CALL_ID = "call_read_1"
_SECOND_CALL_ID = "call_read_2"

_CLAUDE_ECHOES_THE_PROMPT = """
    import json
    import os
    import pathlib
    import sys

    prompt = sys.stdin.read()
    pathlib.Path(os.environ["KISS_TEST_PROMPT_ECHO"]).write_text(prompt)
    print(json.dumps({"type": "content_block_delta",
                      "delta": {"type": "text_delta", "text": "seen"}}),
          flush=True)
    print(json.dumps({"type": "result", "result": "seen",
                      "usage": {"input_tokens": 5, "output_tokens": 1,
                                "cache_read_input_tokens": 0}}), flush=True)
"""

_CODEX_ECHOES_THE_PROMPT = """
    import json
    import os
    import pathlib
    import sys

    prompt = sys.stdin.read()
    pathlib.Path(os.environ["KISS_TEST_PROMPT_ECHO"]).write_text(prompt)
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "seen"}}),
          flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 5, "cached_input_tokens": 0,
                                "output_tokens": 1}}), flush=True)
"""


@pytest.fixture
def png_bytes(tmp_path: Path) -> bytes:
    """Return the bytes of a real PNG image."""
    path = tmp_path / "screenshot.png"
    _write_gradient_png(path)
    return path.read_bytes()


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _screenshot_result(png: bytes) -> dict[str, Any]:
    """Return the tool result a ``Read`` of a screenshot produces."""
    return {"result": "here it is:\n" + encode_binary_attachment("image/png", png)}


def _chat_reply(request: Request) -> Reply:
    """Answer a Chat Completions call with one short assistant message."""
    return Reply(
        json_body={
            "id": "chatcmpl-tool-attachment",
            "object": "chat.completion",
            "created": 0,
            "model": _OPENAI_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "seen"},
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
            "id": "resp_tool_attachment",
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
                        {"type": "output_text", "text": "seen", "annotations": []}
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


def _tool_call(call_id: str) -> dict[str, Any]:
    """Return one Chat-Completions-shaped tool call with *call_id*."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": _TOOL_NAME, "arguments": "{}"},
    }


def _chat_style_tool_call_turn(two_calls: bool = False) -> list[dict[str, Any]]:
    """Return a conversation whose last assistant message called tools.

    Args:
        two_calls: Whether the assistant made two calls instead of one.

    Returns:
        The conversation, ending in the assistant's tool call(s).
    """
    call_ids = [_TOOL_CALL_ID] + ([_SECOND_CALL_ID] if two_calls else [])
    return [
        {"role": "user", "content": "Show me the screenshot."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call(call_id) for call_id in call_ids],
        },
    ]


def anthropic_body_after_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Answer one tool result on a real Anthropic turn and record the body.

    Args:
        result: The ``result_dict`` a tool produced.

    Returns:
        The JSON body the local ``/v1/messages`` endpoint received.
    """
    with ScriptedAnthropicServer() as server:
        model = anthropic_model_with_tool_call(server.base_url)
        model.add_function_results_to_conversation_and_return([(_TOOL_NAME, result)])
        model.generate()
        return server.requests[-1]


def anthropic_model_with_tool_call(base_url: str, two_calls: bool = False) -> Any:
    """Return a real ``AnthropicModel`` mid-turn, pointed at *base_url*.

    Args:
        base_url: The local endpoint the SDK should talk to.
        two_calls: Whether the assistant made two tool calls.

    Returns:
        The model, its conversation ending in ``tool_use`` block(s).
    """
    from kiss.core.models.anthropic_model import AnthropicModel

    call_ids = [_TOOL_CALL_ID] + ([_SECOND_CALL_ID] if two_calls else [])
    model = AnthropicModel(_ANTHROPIC_MODEL, api_key="test-key")
    model.client = anthropic.Anthropic(api_key="test-key", base_url=base_url)
    model.conversation = [
        {"role": "user", "content": "Show me the screenshot."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": _TOOL_NAME,
                    "input": {},
                }
                for call_id in call_ids
            ],
        },
    ]
    return model


def chat_body_after_tool_results(
    results: list[tuple[str, dict[str, Any]]], two_calls: bool = False
) -> dict[str, Any]:
    """Answer tool results on a real Chat Completions turn.

    Args:
        results: The ``(function_name, result_dict)`` pairs to answer with.
        two_calls: Whether the assistant made two tool calls.

    Returns:
        The JSON body the local endpoint received.
    """
    with ScriptedOpenAIServer(_chat_reply) as server:
        model = OpenAICompatibleModel(
            _OPENAI_MODEL, base_url=server.base_url, api_key="test-key"
        )
        model.initialize("Show me the screenshot.")
        model.conversation = _chat_style_tool_call_turn(two_calls)
        model.add_function_results_to_conversation_and_return(results)
        model.generate()
        return server.requests[-1].body


def chat_body_after_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Answer a single tool result on a real Chat Completions turn.

    Args:
        result: The ``result_dict`` a tool produced.

    Returns:
        The JSON body the local endpoint received.
    """
    return chat_body_after_tool_results([(_TOOL_NAME, result)])


def responses_body_after_tool_results(
    results: list[tuple[str, dict[str, Any]]], two_calls: bool = False
) -> dict[str, Any]:
    """Answer tool results on a real Responses turn.

    Args:
        results: The ``(function_name, result_dict)`` pairs to answer with.
        two_calls: Whether the assistant made two tool calls.

    Returns:
        The JSON body the local endpoint received.
    """
    with ScriptedOpenAIServer(_responses_reply) as server:
        model = OpenAICompatibleModel2(
            _OPENAI_MODEL, base_url=server.base_url, api_key="test-key"
        )
        model.initialize("Show me the screenshot.")
        model.conversation = _chat_style_tool_call_turn(two_calls)
        model.add_function_results_to_conversation_and_return(results)
        model.generate()
        return server.requests[-1].body


def responses_body_after_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Answer a single tool result on a real Responses turn.

    Args:
        result: The ``result_dict`` a tool produced.

    Returns:
        The JSON body the local endpoint received.
    """
    return responses_body_after_tool_results([(_TOOL_NAME, result)])


def gemini_body_after_tool_result(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: tuple[str, GeminiScript],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Answer one tool result on a real Gemini turn.

    Args:
        monkeypatch: Fixture used to point the SDK at the local endpoint.
        endpoint: The ``(base_url, script)`` pair of the local endpoint.
        result: The ``result_dict`` a tool produced.

    Returns:
        The JSON body the local endpoint received.
    """
    base_url, script = endpoint
    script.play([chunk([text_part("seen")])])
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(_GEMINI_MODEL, api_key="test-key")
    model.initialize("Show me the screenshot.")
    model.conversation = _chat_style_tool_call_turn()
    model.add_function_results_to_conversation_and_return([(_TOOL_NAME, result)])
    model.generate()
    return script.requests[-1]


def claude_prompt_after_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]
) -> str:
    """Answer one tool result on a real Claude Code CLI turn.

    Args:
        tmp_path: Directory the stand-in executable is installed in.
        monkeypatch: Fixture used to set ``PATH``.
        result: The ``result_dict`` a tool produced.

    Returns:
        The prompt text the stand-in CLI received on stdin.
    """
    install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_ECHOES_THE_PROMPT)
    echo = tmp_path / "claude-prompt.txt"
    monkeypatch.setenv("KISS_TEST_PROMPT_ECHO", str(echo))
    model = ClaudeCodeModel("cc/opus")
    model.initialize("Show me the screenshot.")
    model.conversation = _chat_style_tool_call_turn()
    model.add_function_results_to_conversation_and_return([(_TOOL_NAME, result)])
    model.generate()
    return echo.read_text()


def codex_prompt_after_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]
) -> str:
    """Answer one tool result on a real Codex CLI turn.

    Args:
        tmp_path: Directory the stand-in executable is installed in.
        monkeypatch: Fixture used to set ``PATH``.
        result: The ``result_dict`` a tool produced.

    Returns:
        The prompt text the stand-in CLI received on stdin.
    """
    install_cli(tmp_path, monkeypatch, "codex", _CODEX_ECHOES_THE_PROMPT)
    echo = tmp_path / "codex-prompt.txt"
    monkeypatch.setenv("KISS_TEST_PROMPT_ECHO", str(echo))
    model = CodexModel("codex/default")
    model.initialize("Show me the screenshot.")
    model.conversation = _chat_style_tool_call_turn()
    model.add_function_results_to_conversation_and_return([(_TOOL_NAME, result)])
    model.generate()
    return echo.read_text()


class TestAnImageFromAToolReachesTheModel:
    """Bytes a tool produced must be carried to every network provider."""

    def test_anthropic_carries_the_image(self, png_bytes: bytes) -> None:
        """Anthropic re-attaches the image to the tool result."""
        body = anthropic_body_after_tool_result(_screenshot_result(png_bytes))
        assert _payload_is_present(body, png_bytes)

    def test_chat_completions_carries_the_image(self, png_bytes: bytes) -> None:
        """Chat Completions lifts the image into a follow-up user message."""
        body = chat_body_after_tool_result(_screenshot_result(png_bytes))
        assert _payload_is_present(body, png_bytes)

    def test_responses_carries_the_image(self, png_bytes: bytes) -> None:
        """The Responses transport lifts the image into a user message."""
        body = responses_body_after_tool_result(_screenshot_result(png_bytes))
        assert _payload_is_present(body, png_bytes)

    def test_gemini_carries_the_image(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
        png_bytes: bytes,
    ) -> None:
        """Gemini lifts the image into a follow-up user message."""
        body = gemini_body_after_tool_result(
            monkeypatch, gemini_endpoint, _screenshot_result(png_bytes)
        )
        assert _payload_is_present(body, png_bytes)


class TestATransportThatCannotCarryBytesSaysSo:
    """The CLI transports take one text prompt and must admit the loss."""

    def test_claude_code_states_the_image_was_not_shown(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        png_bytes: bytes,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The prompt says the attachment could not be shown, and it is logged."""
        prompt = claude_prompt_after_tool_result(
            tmp_path, monkeypatch, _screenshot_result(png_bytes)
        )
        assert "could not be shown" in prompt
        assert "image/png" in prompt
        assert base64.b64encode(png_bytes).decode("ascii") not in prompt
        assert "attachment" in caplog.text

    def test_codex_states_the_image_was_not_shown(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        png_bytes: bytes,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Codex must make the same statement Claude Code makes."""
        prompt = codex_prompt_after_tool_result(
            tmp_path, monkeypatch, _screenshot_result(png_bytes)
        )
        assert "could not be shown" in prompt
        assert "image/png" in prompt
        assert base64.b64encode(png_bytes).decode("ascii") not in prompt
        assert "attachment" in caplog.text


class TestAStructuredToolResultNeverCrashes:
    """A non-string ``result`` must be JSON-encoded, not fed to a regex."""

    _STRUCTURED = {"result": {"files": ["a.py"], "count": 1}}

    def test_anthropic(self) -> None:
        """Anthropic must accept a dict result."""
        body = anthropic_body_after_tool_result(dict(self._STRUCTURED))
        assert '"count": 1' in json.dumps(body).replace('\\"', '"')

    def test_chat_completions(self) -> None:
        """Chat Completions must accept a dict result."""
        body = chat_body_after_tool_result(dict(self._STRUCTURED))
        assert '"count": 1' in json.dumps(body).replace('\\"', '"')

    def test_responses(self) -> None:
        """The Responses transport must accept a dict result."""
        body = responses_body_after_tool_result(dict(self._STRUCTURED))
        assert '"count": 1' in json.dumps(body).replace('\\"', '"')

    def test_gemini(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gemini_endpoint: tuple[str, GeminiScript],
    ) -> None:
        """Gemini must accept a dict result."""
        body = gemini_body_after_tool_result(
            monkeypatch, gemini_endpoint, dict(self._STRUCTURED)
        )
        assert '"count": 1' in json.dumps(body).replace('\\"', '"')

    def test_claude_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Claude Code must accept a dict result."""
        prompt = claude_prompt_after_tool_result(
            tmp_path, monkeypatch, dict(self._STRUCTURED)
        )
        assert '"count": 1' in prompt

    def test_codex(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Codex must accept a dict result."""
        prompt = codex_prompt_after_tool_result(
            tmp_path, monkeypatch, dict(self._STRUCTURED)
        )
        assert '"count": 1' in prompt


class TestAnExplicitToolUseIdIsHonoured:
    """``result_dict["tool_use_id"]`` is part of the public signature.

    Results are otherwise matched to the assistant's tool calls by
    position, so answering two calls in reverse order is what tells the two
    mechanisms apart: with the id honoured each result reaches the call it
    belongs to, and without it the answers are swapped.
    """

    _REVERSED = [
        (_TOOL_NAME, {"result": "SECOND-ANSWER", "tool_use_id": _SECOND_CALL_ID}),
        (_TOOL_NAME, {"result": "FIRST-ANSWER", "tool_use_id": _TOOL_CALL_ID}),
    ]

    def test_anthropic_pairs_each_answer_with_its_call(self) -> None:
        """Anthropic's ``tool_result`` blocks carry the named ids."""
        with ScriptedAnthropicServer() as server:
            model = anthropic_model_with_tool_call(server.base_url, two_calls=True)
            model.add_function_results_to_conversation_and_return(list(self._REVERSED))
            model.generate()
            blocks = server.requests[-1]["messages"][-1]["content"]
        pairs = {block["tool_use_id"]: block["content"] for block in blocks}
        assert pairs[_SECOND_CALL_ID] == "SECOND-ANSWER"
        assert pairs[_TOOL_CALL_ID] == "FIRST-ANSWER"

    def test_chat_completions_pairs_each_answer_with_its_call(self) -> None:
        """Chat Completions ``tool`` messages carry the named ids."""
        body = chat_body_after_tool_results(list(self._REVERSED), two_calls=True)
        pairs = {
            msg["tool_call_id"]: msg["content"]
            for msg in body["messages"]
            if msg.get("role") == "tool"
        }
        assert pairs[_SECOND_CALL_ID] == "SECOND-ANSWER"
        assert pairs[_TOOL_CALL_ID] == "FIRST-ANSWER"

    def test_responses_pairs_each_answer_with_its_call(self) -> None:
        """``function_call_output`` items carry the named ids."""
        body = responses_body_after_tool_results(list(self._REVERSED), two_calls=True)
        pairs = {
            item["call_id"]: item["output"]
            for item in body["input"]
            if item.get("type") == "function_call_output"
        }
        assert pairs[_SECOND_CALL_ID] == "SECOND-ANSWER"
        assert pairs[_TOOL_CALL_ID] == "FIRST-ANSWER"

    def test_the_shared_base_pairs_each_answer_with_its_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The base implementation the CLI transports use honours it too."""
        install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_ECHOES_THE_PROMPT)
        model = ClaudeCodeModel("cc/opus")
        model.initialize("Show me the screenshots.")
        model.conversation = _chat_style_tool_call_turn(two_calls=True)
        model.add_function_results_to_conversation_and_return(list(self._REVERSED))
        pairs = {
            msg["tool_call_id"]: msg["content"]
            for msg in model.conversation
            if msg.get("role") == "tool"
        }
        assert pairs[_SECOND_CALL_ID] == "SECOND-ANSWER"
        assert pairs[_TOOL_CALL_ID] == "FIRST-ANSWER"
