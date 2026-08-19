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

from typing import Any

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
)

_MODEL = "gpt-failed-response-under-test"


_TRUNCATED_ARGS = '{"command": "echo hel'


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def _make_model(base_url: str, streaming: bool) -> OpenAICompatibleModel:
    """Build a model against *base_url*.

    Args:
        base_url: The scripted server's ``/v1`` root.
        streaming: Attach a ``token_callback`` so the streaming path is
            taken (the transports branch on exactly that).

    Returns:
        An initialized model.
    """
    model = OpenAICompatibleModel(
        _MODEL,
        base_url=base_url,
        api_key="test-key",
        token_callback=(lambda _t: None) if streaming else None,
    )
    model.initialize("List the files.")
    return model


def _truncated_tool_reply(request: Request) -> Reply:
    """Answer with a tool call whose arguments were cut off by max_tokens."""
    return Reply(
        json_body={
            "id": "chatcmpl-truncated",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "Bash",
                                    "arguments": _TRUNCATED_ARGS,
                                },
                            }
                        ],
                    },
                    "finish_reason": "length",
                }
            ],
        }
    )


def _run_tools(model: OpenAICompatibleModel) -> Any:
    """Run one tool-bearing turn.

    Args:
        model: The model under test.

    Returns:
        The ``(function_calls, content, response)`` triple.
    """
    return model.generate_and_process_with_tools({}, tools_schema=_TOOLS)


class TestTruncatedCompletionIsRefused:
    """``finish_reason="length"`` must never be treated as a success."""

    def test_complete_response_is_untouched(self) -> None:
        """Only ``length`` is refused; a normal turn still works."""

        def responder(request: Request) -> Reply:
            """Answer with an ordinary, complete tool call."""
            reply = _truncated_tool_reply(request)
            assert reply.json_body is not None
            reply.json_body["choices"][0]["finish_reason"] = "tool_calls"
            reply.json_body["choices"][0]["message"]["tool_calls"][0][
                "function"
            ]["arguments"] = '{"command": "echo hello"}'
            return reply

        with ScriptedOpenAIServer(responder) as server:
            model = _make_model(server.base_url, streaming=False)
            function_calls, _content, _response = _run_tools(model)
        assert function_calls[0]["arguments"] == {"command": "echo hello"}
