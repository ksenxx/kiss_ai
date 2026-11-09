# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Claude Code tool-call turn keeps the final usage the CLI sends after the call.

The CLI (``--output-format stream-json --include-partial-messages``) sends
a text block's ``assistant`` message, whose usage holds a placeholder
``output_tokens`` (1), BEFORE the block's ``content_block_stop``, the
``message_delta`` with the message's real output count and the terminal
``result``.  The parser used to stop at the ``assistant`` message of a
turn ending in a JSON call, and at the ``</invoke>`` of a native call, so
every tool-call turn was counted (and, when an API key paid, billed) with
1 output token.  It now reads on for the usage and stops at the first new
content after the call (the model imagining the result).
"""

from __future__ import annotations

import json
from typing import Any

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.claude_code_model import ClaudeCodeModel, _is_call_epilogue
from kiss.core.models.model_info import calculate_cost

_MODEL = "claude-opus-5-5"
_START_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 1,
    "cache_read_input_tokens": 2000,
    "cache_creation_input_tokens": 0,
}
_JSON_CALL = '{"tool_calls": [{"name": "Read", "arguments": {"file_path": "/tmp/x"}}]}'
_XML_CALL = '<invoke name="Read">\n<parameter name="file_path">/tmp/x</parameter>\n</invoke>'


def _stream(text_deltas: list[str], tail: list[dict[str, Any]] | None = None) -> list[dict]:
    """Return the events the CLI sends for one text block, in its real order."""
    text = "".join(text_deltas)
    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": {"model": _MODEL, "usage": dict(_START_USAGE)}},
        {"type": "content_block_start", "content_block": {"type": "text"}},
    ]
    events += [
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": chunk}}
        for chunk in text_deltas
    ]
    events += tail if tail is not None else [
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": text}], "usage": dict(_START_USAGE)},
        },
        {"type": "content_block_stop"},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {**_START_USAGE, "output_tokens": 42},
        },
        {"type": "message_stop"},
        {"type": "result", "result": text, "usage": {**_START_USAGE, "output_tokens": 42}},
    ]
    return events


_STREAMED: list[str] = []
"""The text the last :func:`_parse` streamed to the token callback (the panel)."""


def _parse(events: list[dict[str, Any]]) -> tuple[str, dict[str, Any], int]:
    """Parse *events* as a tool-bearing turn; return content, response, events left unread."""
    model = ClaudeCodeModel("cc/" + _MODEL)
    model._api_billed = True
    _STREAMED.clear()
    model.token_callback = _STREAMED.append
    lines = iter([json.dumps(event) for event in events])
    content, response = model._parse_stream_events(lines, stop_on_tool_calls=True)
    return content, response, len(list(lines))


def _output_tokens(response: dict[str, Any]) -> int:
    return int(response["usage"]["output_tokens"])


def test_json_call_turn_keeps_final_output_tokens() -> None:
    content, response, left = _parse(_stream([_JSON_CALL[:30], _JSON_CALL[30:]]))
    assert content == _JSON_CALL
    assert _output_tokens(response) == 42
    assert left == 0


def test_xml_call_turn_keeps_final_output_tokens() -> None:
    for chunks in ([_XML_CALL], [_XML_CALL[:20], _XML_CALL[20:]], [_XML_CALL, "\n"]):
        content, response, left = _parse(_stream(chunks))
        assert content == _XML_CALL
        assert _output_tokens(response) == 42
        assert left == 0


def test_wrapped_xml_call_keeps_final_output_tokens() -> None:
    wrapped = ["<function_calls>\n", _XML_CALL, "\n</function", "_calls>\n"]
    content, response, _left = _parse(_stream(wrapped))
    assert content == "<function_calls>\n" + _XML_CALL
    assert _output_tokens(response) == 42


def test_fenced_calls_keep_final_output_tokens() -> None:
    for call in (_JSON_CALL, _XML_CALL):
        for chunks in (
            ["```json\n", call, "\n```"],
            ["```\n", call + "\n``", "`\n"],
            ["```\n" + call + "\n```\n"],
        ):
            content, response, left = _parse(_stream(chunks))
            assert content.endswith(call)
            assert _output_tokens(response) == 42
            assert left == 0


def test_text_after_a_closing_fence_stops_the_stream() -> None:
    for call in (_JSON_CALL, _XML_CALL):
        for after in (["\nThe file holds", " the answer."], ["\n```json\n", call]):
            events = _stream(["```\n", call, "\n```", *after])
            content, response, left = _parse(events)
            assert content.endswith(call)
            assert left > 0
            assert _output_tokens(response) == 1
            # Nothing after the closing fence reaches the panel (an XML
            # call drains from its ``</invoke>``, a JSON one from the fence).
            closed = "```\n" + call + ("\n```" if call == _JSON_CALL else "")
            assert "".join(_STREAMED) == closed


def test_imagined_continuation_still_stops_the_stream() -> None:
    for call in (_XML_CALL, _JSON_CALL):
        events = _stream([call, "\nThe file holds the answer.", " More imagined text."])
        content, response, left = _parse(events)
        assert content == call
        # Cut short: the read stopped inside the block, so the unread
        # events (the rest of the imagined text, the final usage) are left.
        assert left > 0
        assert _output_tokens(response) == 1


def test_new_content_block_after_the_call_stops_the_stream() -> None:
    tail: list[dict[str, Any]] = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": _JSON_CALL}]}},
        {"type": "content_block_stop"},
        {"type": "content_block_start", "content_block": {"type": "text"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "imagined"}},
        {"type": "message_delta", "usage": {**_START_USAGE, "output_tokens": 99}},
    ]
    content, _response, left = _parse(_stream([_JSON_CALL], tail=tail))
    assert content == _JSON_CALL
    assert left == 2


def test_whole_message_mode_reads_the_result_usage() -> None:
    events = [
        {"type": "message_start", "message": {"model": _MODEL, "usage": dict(_START_USAGE)}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": _JSON_CALL}]}},
        {"type": "result", "result": _JSON_CALL, "usage": {**_START_USAGE, "output_tokens": 42}},
    ]
    content, response, left = _parse(events)
    assert content == _JSON_CALL
    assert _output_tokens(response) == 42
    assert left == 0


def test_api_billed_cost_uses_the_final_output_tokens() -> None:
    _content, response, _left = _parse(_stream([_XML_CALL]))
    model = ClaudeCodeModel("cc/" + _MODEL)
    model._api_billed = True
    agent = KISSAgent("audit1108")
    agent.model = model
    agent._update_tokens_and_budget_from_response(response)
    expected = calculate_cost(
        _MODEL, num_input_tokens=1000, num_output_tokens=42, num_cache_read_tokens=2000
    )
    usage = agent.last_call_usage
    assert usage is not None
    assert usage["output_tokens"] == 42
    assert abs(agent.budget_used - expected) < 1e-12


def test_call_epilogue() -> None:
    assert _is_call_epilogue("")
    assert _is_call_epilogue(" \n")
    assert _is_call_epilogue("\n</func")
    assert _is_call_epilogue("\n</function_calls>\n")
    assert _is_call_epilogue("\n``")
    assert _is_call_epilogue("\n```\n</function_calls>")
    assert not _is_call_epilogue("\n```\nmore")
    assert not _is_call_epilogue("\n```json\n{")
    assert not _is_call_epilogue("\n</function_calls>\nmore")
    assert not _is_call_epilogue("\nThe result")
    assert not _is_call_epilogue("\0")
