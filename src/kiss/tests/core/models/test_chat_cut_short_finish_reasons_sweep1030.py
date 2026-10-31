# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: Chat Completions refuses every completion the provider cut short.

``_raise_for_finish_reason`` refused only ``finish_reason="length"``.  A
gateway reports a provider that failed mid-generation as
``finish_reason="error"`` (OpenRouter's normalized value, the upstream
error riding on the choice or on the response) and a filtered output as
``"content_filter"``; both were taken as a finished turn, so the partial
text became the answer and a half-written tool call was parsed to
``{}`` and run.  The Responses transport already refuses the matching
``status="failed"`` / ``"incomplete"`` responses.

Every case runs through the real OpenAI SDK against a real local
endpoint, streamed (a token callback is set) and not.  A streamed chunk
with a top-level ``error`` is raised by the SDK itself, so the streamed
cases carry the error on the choice.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)
from kiss.tests.core.models.test_openai_chat_failed_response import (
    _MODEL,
    _TOOLS,
    _TRUNCATED_ARGS,
    _make_model,
)

_USAGE = {"prompt_tokens": 700, "completion_tokens": 300, "total_tokens": 1000}
_UPSTREAM_MESSAGE = "Provider disconnected unexpectedly"
_UPSTREAM_ERROR = {"code": 502, "message": _UPSTREAM_MESSAGE}
_PARTIAL_TEXT = "The files are: a.py, b"
_TOOL_CALL_DELTA = {
    "index": 0,
    "id": "call_1",
    "type": "function",
    "function": {"name": "Bash", "arguments": _TRUNCATED_ARGS},
}


def _completion_reply(choice: dict[str, Any], **extra: Any) -> Reply:
    """Return a non-streamed completion holding *choice*, billed usage and *extra* fields."""
    return Reply(
        json_body={
            "id": "chatcmpl-cut",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [choice],
            "usage": _USAGE,
            **extra,
        }
    )


def _streamed_reply(delta: dict[str, Any], final_choice: dict[str, Any]) -> Reply:
    """Return a stream of one content *delta*, the *final_choice* chunk and a usage chunk."""
    base = {"id": "chatcmpl-cut", "object": "chat.completion.chunk", "created": 0, "model": _MODEL}
    return Reply(
        sse_chunks=[
            chat_chunk({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}),
            chat_chunk({**base, "choices": [final_choice]}),
            chat_chunk({**base, "choices": [], "usage": _USAGE}),
            b"data: [DONE]\n\n",
        ]
    )


def _error_finish_reply(request: Request) -> Reply:
    """A text answer the provider broke off mid-generation, the error on the choice."""
    if request.body.get("stream"):
        return _streamed_reply(
            {"role": "assistant", "content": _PARTIAL_TEXT},
            {"index": 0, "delta": {}, "finish_reason": "error", "error": _UPSTREAM_ERROR},
        )
    return _completion_reply(
        {
            "index": 0,
            "message": {"role": "assistant", "content": _PARTIAL_TEXT},
            "finish_reason": "error",
            "error": _UPSTREAM_ERROR,
        }
    )


def _filtered_tool_call_reply(request: Request) -> Reply:
    """A tool call whose arguments the provider's content filter cut off."""
    if request.body.get("stream"):
        return _streamed_reply(
            {"role": "assistant", "tool_calls": [_TOOL_CALL_DELTA]},
            {"index": 0, "delta": {}, "finish_reason": "content_filter"},
        )
    tool_call = {key: value for key, value in _TOOL_CALL_DELTA.items() if key != "index"}
    return _completion_reply(
        {
            "index": 0,
            "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            "finish_reason": "content_filter",
        }
    )


@pytest.mark.parametrize("streaming", [False, True], ids=["non-streamed", "streamed"])
def test_an_upstream_error_finish_is_refused_and_still_billed(streaming: bool) -> None:
    """The partial answer is refused with the gateway's message; its usage stays billable."""
    with ScriptedOpenAIServer(_error_finish_reply) as server:
        model = _make_model(server.base_url, streaming=streaming)
        with pytest.raises(KISSError) as excinfo:
            model.generate()
    message = str(excinfo.value)
    assert "finish_reason='error'" in message
    assert _UPSTREAM_MESSAGE in message
    rejected = model.take_partial_usage_response()
    assert model.extract_input_output_token_counts_from_response(rejected)[:2] == (700, 300)


def test_a_response_level_upstream_error_is_reported() -> None:
    """A gateway that puts the upstream error on the response, not the choice, is heard too."""

    def responder(request: Request) -> Reply:
        """Answer with an ``error`` finish whose cause rides on the response."""
        return _completion_reply(
            {
                "index": 0,
                "message": {"role": "assistant", "content": _PARTIAL_TEXT},
                "finish_reason": "error",
            },
            error=_UPSTREAM_ERROR,
        )

    with ScriptedOpenAIServer(responder) as server:
        model = _make_model(server.base_url, streaming=False)
        with pytest.raises(KISSError, match=_UPSTREAM_MESSAGE):
            model.generate()


@pytest.mark.parametrize("streaming", [False, True], ids=["non-streamed", "streamed"])
def test_a_filtered_tool_call_is_never_run(streaming: bool) -> None:
    """A tool call cut off by the content filter is refused instead of run with ``{}``."""
    with ScriptedOpenAIServer(_filtered_tool_call_reply) as server:
        model = _make_model(server.base_url, streaming=streaming)
        with pytest.raises(KISSError) as excinfo:
            model.generate_and_process_with_tools({}, tools_schema=_TOOLS)
    message = str(excinfo.value)
    assert "finish_reason='content_filter'" in message
    assert "content filter" in message
    assert model.conversation[-1]["role"] == "user", "the cut-off tool call joined the conversation"
