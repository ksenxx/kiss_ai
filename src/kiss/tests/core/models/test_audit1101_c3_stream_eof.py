# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (C3/F1): a Chat Completions stream must end with a ``finish_reason``.

``_consume_chat_stream`` recorded the choice that carried a nonempty
``finish_reason`` and refused the cut-short ones (``length``,
``content_filter``, ``error``), but a stream whose HTTP body simply
ended -- EOF, or a ``[DONE]`` marker -- before any choice carried a
``finish_reason`` went through the unconditional return, and the public
generation methods appended the partial text (or the tool call whose
truncated JSON arguments had been parsed to ``{}``) as a finished turn.
A gateway can close its body cleanly before the upstream model is done,
so HTTP framing success is not completion success; the Responses
transport already refuses a stream without ``response.completed``.

Every case drives the real OpenAI SDK against a real local SSE
endpoint.  The partial usage a usage chunk reported before the cut
stays billable through ``take_partial_usage_response``.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_MODEL = "gpt-audit1101-eof"
_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
_PARTIAL_TEXT = "Partial answer: "
_TOOL_CALL_DELTA = {
    "index": 0,
    "id": "call_eof",
    "type": "function",
    "function": {"name": "Bash", "arguments": '{"command": "echo par'},
}
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
_DONE = b"data: [DONE]\n\n"


def _chunk(choices: list[dict[str, Any]], usage: dict[str, int] | None = None) -> bytes:
    payload: dict[str, Any] = {
        "id": "chatcmpl-eof",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": _MODEL,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return chat_chunk(payload)


def _delta_chunk(tools: bool) -> bytes:
    delta: dict[str, Any] = {"role": "assistant", "content": _PARTIAL_TEXT}
    if tools:
        delta["tool_calls"] = [_TOOL_CALL_DELTA]
    return _chunk([{"index": 0, "delta": delta, "finish_reason": None}])


def _finish_chunk(tools: bool) -> bytes:
    reason = "tool_calls" if tools else "stop"
    return _chunk([{"index": 0, "delta": {}, "finish_reason": reason}])


class _Stream:
    """Answer every request with the same scripted SSE chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def __call__(self, request: Request) -> Reply:
        return Reply(sse_chunks=self.chunks)


def _model(base_url: str) -> OpenAICompatibleModel:
    model = OpenAICompatibleModel(
        _MODEL, base_url=base_url, api_key="test-key", token_callback=lambda _t: None
    )
    model.initialize("Answer or run the tool.")
    return model


def _generate(model: OpenAICompatibleModel, tools: bool) -> Any:
    if tools:
        return model.generate_and_process_with_tools({}, tools_schema=_TOOLS)
    return model.generate()


@pytest.mark.parametrize("tools", [False, True], ids=["text", "tool-call"])
@pytest.mark.parametrize("done_marker", [False, True], ids=["eof", "done-marker"])
@pytest.mark.parametrize("usage_seen", [False, True], ids=["no-usage", "usage-chunk"])
def test_stream_without_finish_reason_is_refused(
    tools: bool, done_marker: bool, usage_seen: bool
) -> None:
    """The partial turn is refused, not appended; usage seen before the cut stays billable."""
    chunks = [_delta_chunk(tools)]
    if usage_seen:
        chunks.append(_chunk([], usage=_USAGE))
    if done_marker:
        chunks.append(_DONE)
    with ScriptedOpenAIServer(_Stream(chunks)) as server:
        model = _model(server.base_url)
        turns_before = list(model.conversation)
        with pytest.raises(KISSError) as excinfo:
            _generate(model, tools)
        assert len(server.requests) == 1
    assert "finish_reason" in str(excinfo.value)
    assert model.conversation == turns_before
    rejected = model.take_partial_usage_response()
    if usage_seen:
        assert model.extract_input_output_token_counts_from_response(rejected)[:2] == (11, 7)
    else:
        assert rejected is None
    assert model.take_partial_usage_response() is None


@pytest.mark.parametrize("tools", [False, True], ids=["text", "tool-call"])
@pytest.mark.parametrize("done_marker", [False, True], ids=["eof", "done-marker"])
def test_stream_with_finish_reason_still_succeeds(tools: bool, done_marker: bool) -> None:
    """Control: the same stream ended by a ``finish_reason`` chunk is a finished turn."""
    chunks = [_delta_chunk(tools), _finish_chunk(tools), _chunk([], usage=_USAGE)]
    if done_marker:
        chunks.append(_DONE)
    with ScriptedOpenAIServer(_Stream(chunks)) as server:
        model = _model(server.base_url)
        if tools:
            calls, content, response = model.generate_and_process_with_tools(
                {}, tools_schema=_TOOLS
            )
            assert content == _PARTIAL_TEXT
            # The arguments are the truncated JSON the stream carried: the
            # finish_reason says the model stopped there, so the turn is
            # complete as far as the transport can tell.
            assert [call["name"] for call in calls] == ["Bash"]
        else:
            content, response = model.generate()
            assert content == _PARTIAL_TEXT
    assert model.conversation[-1]["role"] == "assistant"
    assert model.extract_input_output_token_counts_from_response(response)[:2] == (11, 7)
    assert model.take_partial_usage_response() is None
