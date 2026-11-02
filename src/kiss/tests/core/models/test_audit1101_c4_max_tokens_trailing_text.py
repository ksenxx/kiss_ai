# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (C4): the output cap discards only the block it actually cut off.

At ``stop_reason="max_tokens"`` the adapter drops a final ``tool_use``
block (its ``input`` may be a partial parse) and keeps the tool calls
completed before it.  It used to find that final block in the
NORMALIZED content, from which an empty or whitespace-only text block
has been dropped, so a completed tool followed by such a text block
(the cap landing in the text) was taken for the cut-off block, deleted,
and the turn raised ``OutputCapError`` although it carried a complete,
executable tool call.  Real SDK against a local scripted Messages
endpoint; nothing is mocked.
"""

from __future__ import annotations

from typing import Any

import pytest
from anthropic import Anthropic

from kiss.core.kiss_error import OutputCapError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer, sse

_NOOP_TOOL = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "A no-op",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _tool_block(index: int, call_id: str) -> list[bytes]:
    return [
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": call_id, "name": "noop", "input": {}},
            },
        ),
        sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
        ),
        sse("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def _text_block(index: int, text: str) -> list[bytes]:
    return [
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        sse("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def _capped_stream(blocks: list[list[bytes]]) -> list[bytes]:
    """A full assistant turn of *blocks* that ends with ``stop_reason="max_tokens"``."""
    head = sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_cap",
                "type": "message",
                "role": "assistant",
                "model": "claude-under-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        },
    )
    tail = [
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                "usage": {"output_tokens": 10},
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    ]
    return [head, *(event for block in blocks for event in block), *tail]


def _generate(chunks: list[bytes]) -> tuple[AnthropicModel, list[dict[str, Any]], str]:
    with ScriptedAnthropicServer(chunks) as server:
        model = AnthropicModel("claude-under-test", api_key="local-only")
        model.conversation = [{"role": "user", "content": "Call noop"}]
        model.client = Anthropic(api_key="local-only", base_url=server.base_url, max_retries=0)
        try:
            calls, text, _raw = model.generate_and_process_with_tools({}, [_NOOP_TOOL])
        finally:
            model.client.close()
    return model, calls, text


@pytest.mark.parametrize("tail", ["", " ", "\n\t "])
def test_completed_tool_before_a_blank_trailing_text_block_survives_the_cap(tail: str) -> None:
    model, calls, text = _generate(
        _capped_stream([_tool_block(0, "toolu_complete"), _text_block(1, tail)])
    )
    assert [c["id"] for c in calls] == ["toolu_complete"]
    assert text == ""
    # The completed call is in the conversation, the blank text block is not.
    assert model.conversation[-1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_complete", "name": "noop", "input": {}}],
    }


def test_every_completed_tool_before_a_blank_trailing_text_block_is_kept() -> None:
    _model, calls, _text = _generate(
        _capped_stream(
            [_tool_block(0, "toolu_first"), _tool_block(1, "toolu_second"), _text_block(2, " ")]
        )
    )
    assert [c["id"] for c in calls] == ["toolu_first", "toolu_second"]


def test_completed_tool_before_a_cut_off_text_block_is_kept() -> None:
    _model, calls, text = _generate(
        _capped_stream([_tool_block(0, "toolu_complete"), _text_block(1, "and nex")])
    )
    assert [c["id"] for c in calls] == ["toolu_complete"]
    assert text == "and nex"


def test_a_final_tool_use_block_is_still_discarded_at_the_cap() -> None:
    # The cap landed inside the last tool_use: its input is a partial parse,
    # so it goes; the tool completed before it is returned.
    _model, calls, _text = _generate(
        _capped_stream([_tool_block(0, "toolu_complete"), _tool_block(1, "toolu_cut")])
    )
    assert [c["id"] for c in calls] == ["toolu_complete"]
    # With no completed call left, a tool-bearing turn is unusable.
    with pytest.raises(OutputCapError):
        _generate(_capped_stream([_text_block(0, " "), _tool_block(1, "toolu_cut")]))
