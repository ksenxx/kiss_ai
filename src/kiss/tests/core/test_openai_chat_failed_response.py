# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: Chat Completions must refuse unusable responses.

Audit finding 02/I2.  The Responses transport guards every path with
``_raise_for_failed_response`` (``status="failed"`` / ``"incomplete"``),
but the Chat Completions transport never looked at ``finish_reason`` at
all — ``grep -n finish_reason openai_compatible_model.py`` returned zero
hits — and indexed ``response.choices[0]`` unconditionally.  Two failure
modes followed:

* **Truncation was invisible.**  ``finish_reason="length"`` carries a
  half-written ``tool_calls[0].function.arguments`` string;
  ``_build_tool_call_lists`` swallows the ``JSONDecodeError`` and yields
  ``arguments={}``, so the agent invokes ``Bash()`` with no command and
  the model is told the tool failed for a reason unrelated to the real
  cause.  The identical situation on the Responses path raises a clear
  ``KISSError`` naming ``max_output_tokens``.
* **An empty ``choices`` list was an ``IndexError``.**  Several
  OpenAI-compatible gateways answer an upstream failure with
  ``200 {"choices": [], "error": {...}}``; the user saw a stack trace
  instead of the gateway's message.

No mocks: a real ``ThreadingHTTPServer`` returns the genuine JSON and SSE
bodies these providers send, to the real OpenAI SDK.
"""

from __future__ import annotations

import pytest

from kiss.core.kiss_error import KISSError
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)
from kiss.tests.core.models.test_openai_chat_failed_response import (  # noqa: F401
    _MODEL,
    _TOOLS,
    _TRUNCATED_ARGS,
    _make_model,
    _run_tools,
    _truncated_tool_reply,
)


def _truncated_text_reply(request: Request) -> Reply:
    """Answer a tool-less turn with text cut off by max_tokens."""
    return Reply(
        json_body={
            "id": "chatcmpl-truncated-text",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "the answer is"},
                    "finish_reason": "length",
                }
            ],
        }
    )


def _empty_choices_reply(request: Request) -> Reply:
    """Answer the way a gateway reports an upstream failure with a 200."""
    return Reply(
        json_body={
            "id": "chatcmpl-empty",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [],
            "error": {"message": "upstream 502", "type": "server_error"},
        }
    )


def _streamed_truncated_tool_reply(request: Request) -> Reply:
    """Stream a tool call that is cut off mid-arguments."""
    return Reply(
        sse_chunks=[
            chat_chunk(
                {
                    "id": "chatcmpl-stream-truncated",
                    "object": "chat.completion.chunk",
                    "model": _MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "Bash",
                                            "arguments": _TRUNCATED_ARGS,
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            chat_chunk(
                {
                    "id": "chatcmpl-stream-truncated",
                    "object": "chat.completion.chunk",
                    "model": _MODEL,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "length"}
                    ],
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )


class TestTruncatedCompletionIsRefused:
    """``finish_reason="length"`` must never be treated as a success."""

    def test_non_streamed_truncated_tool_call(self) -> None:
        """Truncated arguments must raise, not silently become ``{}``."""
        with ScriptedOpenAIServer(_truncated_tool_reply) as server:
            model = _make_model(server.base_url, streaming=False)
            with pytest.raises(KISSError) as excinfo:
                _run_tools(model)
        assert "truncated" in str(excinfo.value)
        assert "finish_reason='length'" in str(excinfo.value)

    def test_streamed_truncated_tool_call(self) -> None:
        """The streamed agentic path needs the same guard."""
        with ScriptedOpenAIServer(_streamed_truncated_tool_reply) as server:
            model = _make_model(server.base_url, streaming=True)
            with pytest.raises(KISSError) as excinfo:
                _run_tools(model)
        assert "truncated" in str(excinfo.value)

    def test_non_streamed_truncated_text(self) -> None:
        """A tool-less turn must not report truncated text as complete."""
        with ScriptedOpenAIServer(_truncated_text_reply) as server:
            model = _make_model(server.base_url, streaming=False)
            with pytest.raises(KISSError) as excinfo:
                model.generate()
        assert "truncated" in str(excinfo.value)


class TestEmptyChoicesIsRefused:
    """A gateway's ``200 {"choices": []}`` must surface its own message."""

    def test_tool_turn_reports_the_gateway_error(self) -> None:
        """The agentic path must raise KISSError, not IndexError."""
        with ScriptedOpenAIServer(_empty_choices_reply) as server:
            model = _make_model(server.base_url, streaming=False)
            with pytest.raises(KISSError) as excinfo:
                _run_tools(model)
        assert "upstream 502" in str(excinfo.value)

    def test_toolless_turn_reports_the_gateway_error(self) -> None:
        """``generate()`` must do the same."""
        with ScriptedOpenAIServer(_empty_choices_reply) as server:
            model = _make_model(server.base_url, streaming=False)
            with pytest.raises(KISSError) as excinfo:
                model.generate()
        assert "upstream 502" in str(excinfo.value)

    def test_missing_error_field_still_raises(self) -> None:
        """An empty-choices body with no ``error`` must still be refused."""

        def responder(request: Request) -> Reply:
            """Answer with empty choices and no diagnostic at all."""
            reply = _empty_choices_reply(request)
            assert reply.json_body is not None
            reply.json_body.pop("error")
            return reply

        with ScriptedOpenAIServer(responder) as server:
            model = _make_model(server.base_url, streaming=False)
            with pytest.raises(KISSError, match="no choices"):
                model.generate()
