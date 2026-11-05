# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An Anthropic stream is a finished turn only once its ``message_stop`` arrived.

The SDK's ``get_final_message()`` drains the iterator and returns the
accumulated snapshot whatever ended the stream, and its partial JSON
decoder turns a cut-off ``input_json_delta`` into a plausible value: a
string left open is dropped (``{"path": "target.txt"}`` without the
``content`` key), a number is cut (``12`` for ``123456``).  A connection
a gateway closed cleanly before the message's terminal events therefore
used to come back as a complete tool call, with ``stop_reason=None`` so
that the output-cap check did not apply either, and was appended to the
conversation and returned for execution.  The Chat Completions and
Responses transports already refuse a stream without its terminal event.

Every case drives the real SDK against a real local SSE endpoint; the
usage ``message_start`` reported before the cut stays billable through
``take_partial_usage_response``.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.tests.core.models.anthropic_sse_harness import (
    ScriptedAnthropicServer,
    sse,
    text_message_stream,
)

_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write text.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}
_SET_LIMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "set_limit",
        "description": "Set a numeric limit.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    },
}


def _tool_stream(name: str, partial_json: str) -> list[bytes]:
    """``message_start`` and a ``tool_use`` block whose input stops at *partial_json*."""
    return [
        text_message_stream()[0],
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_cut", "name": name, "input": {}},
            },
        ),
        sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": partial_json},
            },
        ),
    ]


_TERMINAL_EVENTS = [
    sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
    sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 25},
        },
    ),
    sse("message_stop", {"type": "message_stop"}),
]


def _generate(
    chunks: list[bytes], tools: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> tuple[AnthropicModel, list[dict[str, Any]] | KISSError]:
    """Run one tool turn against a server that streams *chunks*; return the model and outcome."""
    with ScriptedAnthropicServer(chunks) as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
        model = AnthropicModel("claude-under-test", api_key="local-only")
        model.initialize("Write the complete content.")
        try:
            calls, _text, _raw = model.generate_and_process_with_tools({}, tools)
            return model, calls
        except KISSError as error:
            return model, error
        finally:
            model.client.close()


@pytest.mark.parametrize(
    ("tools", "partial_json"),
    [
        ([_WRITE_TOOL], '{"path":"target.txt","content":"PARTIAL'),
        ([_SET_LIMIT_TOOL], '{"value":12'),
        ([], '{"path":"target.txt"'),
    ],
    ids=["open-string", "cut-number", "no-tools"],
)
def test_a_stream_cut_inside_a_tool_argument_is_refused(
    tools: list[dict[str, Any]], partial_json: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EOF after an ``input_json_delta``, before ``message_stop``, is not a turn."""
    name = tools[0]["function"]["name"] if tools else "write"
    model, outcome = _generate(_tool_stream(name, partial_json), tools, monkeypatch)
    assert isinstance(outcome, KISSError), f"the cut-off call was accepted: {outcome}"
    assert "message_stop" in str(outcome)
    # Nothing was appended: the retry re-sends the same conversation.
    assert [m["role"] for m in model.conversation] == ["user"]
    # The usage message_start reported is still billed.
    partial = model.take_partial_usage_response()
    assert partial is not None and partial.usage.input_tokens == 3


def test_the_same_stream_with_its_terminal_events_is_a_finished_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: ``content_block_stop``, ``message_delta`` and ``message_stop`` complete it."""
    chunks = _tool_stream("write", '{"path":"target.txt","content":"FULL"}') + _TERMINAL_EVENTS
    model, outcome = _generate(chunks, [_WRITE_TOOL], monkeypatch)
    assert outcome == [
        {"id": "toolu_cut", "name": "write", "arguments": {"path": "target.txt", "content": "FULL"}}
    ]
    assert [m["role"] for m in model.conversation] == ["user", "assistant"]
    assert model.take_partial_usage_response() is None
