# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking down offline-testable behavior of
``openai_compatible_model.py`` (v1) and ``openai_compatible_model2.py`` (v2)
before/after code simplification.

No mocks, patches, fakes, or monkeypatching: every test calls real
functions and constructs real objects (``types.SimpleNamespace`` instances
are plain Python objects standing in for SDK response payloads; dict-shaped
responses are natively supported by the code under test).  No network calls
are made (``base_url="http://localhost:1"`` is never contacted).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model import (
    OpenAICompatibleModel,
    _anthropic_media_block_to_openai_part,
    _audio_mime_to_format,
    _extract_deepseek_reasoning,
    _provider_model_name,
    _stringify_tool_call_arguments,
    _tool_result_block_text,
)
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

BASE_URL = "http://localhost:1"
API_KEY = "test"
MODEL = "totally-custom-model-not-in-catalog"


def make_v1(model_name: str = MODEL, **kw: Any) -> OpenAICompatibleModel:
    """Construct a v1 model pointing at an unreachable endpoint."""
    return OpenAICompatibleModel(model_name, BASE_URL, API_KEY, **kw)


def make_v2(model_name: str = MODEL, **kw: Any) -> OpenAICompatibleModel2:
    """Construct a v2 model pointing at an unreachable endpoint."""
    return OpenAICompatibleModel2(model_name, BASE_URL, API_KEY, **kw)


# ---------------------------------------------------------------------------
# Module-level helpers (v1)
# ---------------------------------------------------------------------------


def test_provider_model_name() -> None:
    assert _provider_model_name("openrouter/foo/bar") == "foo/bar"
    assert _provider_model_name("gpt-5.5-xhigh") == "gpt-5.5"
    assert _provider_model_name("openrouter/openai/gpt-5.5-xhigh") == "openai/gpt-5.5"
    assert _provider_model_name("plain-model") == "plain-model"


def test_audio_mime_to_format() -> None:
    assert _audio_mime_to_format("audio/mpeg") == "mp3"
    assert _audio_mime_to_format("audio/x-wav") == "wav"
    assert _audio_mime_to_format("audio/unknown-sub") == "unknown-sub"
    assert _audio_mime_to_format("noslash") == "noslash"


def test_extract_deepseek_reasoning() -> None:
    assert _extract_deepseek_reasoning("<think> why </think> ans") == ("why", "ans")
    assert _extract_deepseek_reasoning("no tags here") == ("", "no tags here")


def test_anthropic_media_block_to_openai_part() -> None:
    img = _anthropic_media_block_to_openai_part(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
        }
    )
    assert img == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAA"},
    }
    doc = _anthropic_media_block_to_openai_part(
        {"type": "document", "source": {"type": "url", "url": "http://x/y.pdf"}}
    )
    assert doc == {"type": "file", "file": {"file_data": "http://x/y.pdf"}}
    assert _anthropic_media_block_to_openai_part({"type": "image", "source": {}}) is None


def test_stringify_tool_call_arguments() -> None:
    out = _stringify_tool_call_arguments(
        [
            {"id": "1", "function": {"name": "f", "arguments": {"a": 1}}},
            {"id": "2", "function": {"name": "g", "arguments": '{"b":2}'}},
            "passthrough",
        ]
    )
    assert json.loads(out[0]["function"]["arguments"]) == {"a": 1}
    assert out[1]["function"]["arguments"] == '{"b":2}'
    assert out[2] == "passthrough"
    none_args = _stringify_tool_call_arguments([{"function": {"name": "h", "arguments": None}}])
    assert none_args[0]["function"]["arguments"] == "{}"


def test_tool_result_block_text() -> None:
    assert _tool_result_block_text({"content": "plain"}) == "plain"
    assert (
        _tool_result_block_text(
            {
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "image", "source": {}},
                    {"type": "text", "text": "b"},
                ]
            }
        )
        == "ab"
    )
    assert _tool_result_block_text({"content": None}) == ""
    assert _tool_result_block_text({"content": 42}) == "42"


# ---------------------------------------------------------------------------
# v1: message normalization
# ---------------------------------------------------------------------------


def test_normalize_message_for_api_plain_and_whitespace() -> None:
    n = OpenAICompatibleModel._normalize_message_for_api
    assert n({"role": "user", "content": "hi"}) == [{"role": "user", "content": "hi"}]
    assert n({"role": "user", "content": "   "}) == []
    kept = n({"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]})
    assert len(kept) == 1 and kept[0]["tool_calls"]


def test_normalize_message_for_api_anthropic_tool_use() -> None:
    msgs = OpenAICompatibleModel._normalize_message_for_api(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "using tool"},
                {"type": "thinking", "thinking": "secret"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}},
            ],
        }
    )
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "using tool"
    assert msg["tool_calls"][0]["id"] == "t1"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


def test_normalize_message_for_api_anthropic_tool_result() -> None:
    msgs = OpenAICompatibleModel._normalize_message_for_api(
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        }
    )
    assert msgs == [{"role": "tool", "tool_call_id": "t1", "content": "ok"}]


# ---------------------------------------------------------------------------
# v1: tool-call list building / stream finalization
# ---------------------------------------------------------------------------


def test_build_tool_call_lists_and_accum() -> None:
    fcs, raws = OpenAICompatibleModel._build_tool_call_lists(
        [("id1", "f", '{"a": 1}'), ("id2", "g", "not json")]
    )
    assert fcs == [
        {"id": "id1", "name": "f", "arguments": {"a": 1}},
        {"id": "id2", "name": "g", "arguments": {}},
    ]
    assert raws[1]["function"]["arguments"] == "not json"
    fcs2, _ = OpenAICompatibleModel._parse_tool_call_accum(
        {
            1: {"id": "b", "name": "g", "arguments": "{}"},
            0: {"id": "a", "name": "f", "arguments": "{}"},
        }
    )
    assert [fc["id"] for fc in fcs2] == ["a", "b"]


def test_finalize_stream_response() -> None:
    usage_chunk = SimpleNamespace(usage=object())
    last = SimpleNamespace(usage=None)
    assert OpenAICompatibleModel._finalize_stream_response(usage_chunk, last) is usage_chunk
    assert OpenAICompatibleModel._finalize_stream_response(None, last) is last
    with pytest.raises(KISSError):
        OpenAICompatibleModel._finalize_stream_response(None, None)


def test_v1_extract_token_counts() -> None:
    m = make_v1()
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20, cache_write_tokens=5),
        )
    )
    assert m.extract_input_output_token_counts_from_response(resp) == (75, 10, 20, 5)
    assert m.extract_input_output_token_counts_from_response(
        SimpleNamespace(usage=None)
    ) == (0, 0, 0, 0)


def test_v1_cache_control_openrouter_anthropic() -> None:
    m = make_v1("openrouter/anthropic/claude-fake")
    kwargs: dict[str, Any] = {}
    m._apply_cache_control_for_openrouter_anthropic(kwargs)
    assert kwargs["extra_body"]["cache_control"] == {"type": "ephemeral"}

    m2 = make_v1(
        "openrouter/anthropic/claude-fake", model_config={"enable_cache": False}
    )
    kwargs2: dict[str, Any] = {}
    m2._apply_cache_control_for_openrouter_anthropic(kwargs2)
    assert kwargs2 == {}

    m3 = make_v1()
    kwargs3: dict[str, Any] = {}
    m3._apply_cache_control_for_openrouter_anthropic(kwargs3)
    assert kwargs3 == {}


# ---------------------------------------------------------------------------
# v1: chat -> responses conversion helpers
# ---------------------------------------------------------------------------


def test_chat_parts_to_responses_parts() -> None:
    parts = OpenAICompatibleModel._chat_parts_to_responses_parts(
        [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "   "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,BB"}},
            {"type": "input_audio", "input_audio": {"data": "CC", "format": "mp3"}},
            {"type": "bogus"},
        ]
    )
    assert parts == [
        {"type": "input_text", "text": "hello"},
        {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "auto"},
        {
            "type": "input_file",
            "filename": "attachment.pdf",
            "file_data": "data:application/pdf;base64,BB",
        },
        {"type": "input_audio", "input_audio": {"data": "CC", "format": "mp3"}},
    ]


def test_chat_message_to_responses_items() -> None:
    conv = OpenAICompatibleModel._chat_message_to_responses_items
    assert conv({"role": "tool", "tool_call_id": "c1", "content": "res"}) == [
        {"type": "function_call_output", "call_id": "c1", "output": "res"}
    ]
    items = conv(
        {
            "role": "assistant",
            "content": "text",
            "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}
            ],
        }
    )
    assert items[0] == {"role": "assistant", "content": "text"}
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "c2"
    assert json.loads(items[1]["arguments"]) == {"a": 1}
    assert conv({"role": "user", "content": "hi"}) == [{"role": "user", "content": "hi"}]
    assert conv({"role": "user", "content": "  "}) == []


# ---------------------------------------------------------------------------
# v2: schema / config shaping helpers
# ---------------------------------------------------------------------------


def test_flatten_tools_schema() -> None:
    flat = OpenAICompatibleModel2._flatten_tools_schema(
        [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {}, "strict": True},
            },
            {"type": "function", "name": "already-flat", "parameters": {}},
        ]
    )
    assert flat[0] == {
        "type": "function",
        "name": "f",
        "description": "d",
        "parameters": {},
        "strict": True,
    }
    assert flat[1] == {"type": "function", "name": "already-flat", "parameters": {}}


def test_flatten_tool_choice() -> None:
    assert OpenAICompatibleModel2._flatten_tool_choice(
        {"type": "function", "function": {"name": "f"}}
    ) == {"type": "function", "name": "f"}
    assert OpenAICompatibleModel2._flatten_tool_choice("auto") == "auto"


def test_normalize_input() -> None:
    norm = OpenAICompatibleModel2._normalize_input(
        [
            {"type": "_kiss_pending_tool_result_attachment", "data": "x"},
            {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
            {"role": "user", "content": "   "},
            {"role": "user", "content": [{"type": "input_text", "text": " "}]},
            {"role": "user", "content": [{"type": "refusal", "refusal": ""}]},
            {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
            {"role": "assistant", "content": None},
            "not-a-dict",
        ]
    )
    assert norm == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
        },
        {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
        "not-a-dict",
    ]


def test_translate_response_format() -> None:
    tr = OpenAICompatibleModel2._translate_response_format_for_responses
    assert tr({"type": "json_schema", "json_schema": {"name": "s", "schema": {}}}) == {
        "type": "json_schema",
        "name": "s",
        "schema": {},
    }
    assert tr({"type": "json_object"}) == {"type": "json_object"}


def test_is_valid_json() -> None:
    v = OpenAICompatibleModel2._is_valid_json
    assert not v("")
    assert not v("{")
    assert v('{"a": 1}')


def test_reasoning_inner_index() -> None:
    r = OpenAICompatibleModel2._reasoning_inner_index
    assert r(SimpleNamespace(summary_index=0), ) == 0
    assert r(SimpleNamespace(summary_index=3)) == 3
    assert r(SimpleNamespace(summary_index=None, content_index=2)) == 2
    assert r(SimpleNamespace()) == 0
    assert r(SimpleNamespace(summary_index="bogus")) == 0
    assert r(SimpleNamespace(summary_index=None, content_index="bogus")) == 0


def test_shape_responses_kwargs() -> None:
    cfg = {
        "system_instruction": "sys",
        "reasoning_effort": "high",
        "max_tokens": 5,
        "stop": ["x"],
        "seed": 1,
        "temperature": 0.5,
        "response_format": {"type": "json_object"},
        "tool_choice": {"type": "function", "function": {"name": "f"}},
        "parallel_tool_calls": True,
        "enable_cache": True,
    }
    m = make_v2(model_config=dict(cfg))
    msgs = [{"role": "user", "content": "hi"}]
    kwargs = m._shape_responses_kwargs(input_items=msgs, tools=None)
    assert kwargs["model"] == MODEL
    assert kwargs["input"] == msgs
    assert kwargs["instructions"] == "sys"
    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
    assert kwargs["max_output_tokens"] == 5
    assert kwargs["text"] == {"format": {"type": "json_object"}}
    assert kwargs["temperature"] == 0.5
    for absent in (
        "stop",
        "seed",
        "system_instruction",
        "reasoning_effort",
        "response_format",
        "enable_cache",
        "tool_choice",
        "parallel_tool_calls",
        "tools",
        "max_tokens",
    ):
        assert absent not in kwargs

    tools = [{"type": "function", "name": "f", "parameters": {}}]
    kwargs2 = m._shape_responses_kwargs(input_items=msgs, tools=tools)
    assert kwargs2["tools"] == tools
    assert kwargs2["tool_choice"] == {"type": "function", "name": "f"}
    assert kwargs2["parallel_tool_calls"] is True

    # max_completion_tokens wins over max_tokens.
    m2 = make_v2(model_config={"max_tokens": 5, "max_completion_tokens": 9})
    assert m2._shape_responses_kwargs(input_items=msgs, tools=None)["max_output_tokens"] == 9

    # Empty conversation raises.
    with pytest.raises(KISSError):
        m._shape_responses_kwargs(input_items=[{"role": "user", "content": " "}], tools=None)

    # Caller config never mutated.
    m3 = make_v2(model_config={"reasoning": {"summary": "auto"}, "reasoning_effort": "low"})
    out3 = m3._shape_responses_kwargs(input_items=msgs, tools=None)
    assert out3["reasoning"] == {"summary": "auto", "effort": "low"}
    assert m3.model_config["reasoning"] == {"summary": "auto"}


# ---------------------------------------------------------------------------
# v2: function-call conversation contract
# ---------------------------------------------------------------------------


def _fc(call_id: str, name: str = "f") -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _fco(call_id: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": "ok"}


def test_trailing_function_call_ids() -> None:
    m = make_v2()
    m.initialize("hi")
    m.conversation.extend([_fc("c1", "f"), _fc("c2", "g")])
    assert m._trailing_function_call_ids() == [("f", "c1"), ("g", "c2")]


def test_add_function_results_uses_pending_call_id() -> None:
    m = make_v2()
    m.initialize("hi")
    m.conversation.append(_fc("c1"))
    m._pending_function_calls = [{"name": "f", "call_id": "c1"}]
    m.add_function_results_to_conversation_and_return([("f", {"result": "ok"})])
    assert m.conversation[-1] == _fco("c1")
    assert m._pending_function_calls == []


def test_add_function_results_mismatch_rolls_back() -> None:
    m = make_v2()
    m.initialize("hi")
    m.conversation.append(_fc("c1"))
    m._pending_function_calls = [{"name": "f", "call_id": "c1"}]
    before = list(m.conversation)
    with pytest.raises(KISSError):
        m.add_function_results_to_conversation_and_return([("WRONG", {"result": "x"})])
    assert m.conversation == before
    assert m._pending_function_calls == [{"name": "f", "call_id": "c1"}]


def test_add_function_results_fallback_unanswered() -> None:
    # Restored conversation: no pending queue, unanswered call in conversation.
    m = make_v2()
    m.initialize("hi")
    m.conversation.append(_fc("c9", "g"))
    m._pending_function_calls = []
    m.add_function_results_to_conversation_and_return([("g", {"result": "r"})])
    assert m.conversation[-1] == {
        "type": "function_call_output",
        "call_id": "c9",
        "output": "r",
    }


def test_validate_function_call_conversation() -> None:
    m = make_v2()
    m.initialize("hi")
    m.conversation.extend([_fc("c1"), _fco("c1"), _fc("c2")])
    assert m._validate_function_call_conversation() == ["c2"]

    m2 = make_v2()
    m2.initialize("hi")
    m2.conversation.extend([_fc("c1"), _fc("c1")])
    with pytest.raises(KISSError):
        m2._validate_function_call_conversation()

    m3 = make_v2()
    m3.initialize("hi")
    m3.conversation.append(_fco("nope"))
    with pytest.raises(KISSError):
        m3._validate_function_call_conversation()

    m4 = make_v2()
    m4.initialize("hi")
    m4.conversation.extend([_fc("c1"), {"role": "user", "content": "early"}])
    with pytest.raises(KISSError):
        m4._validate_function_call_conversation()


def test_ensure_no_pending_function_calls_raises() -> None:
    m = make_v2()
    m.initialize("hi")
    m.conversation.append(_fc("c1"))
    with pytest.raises(KISSError):
        m._build_request_kwargs(tools=None)


# ---------------------------------------------------------------------------
# v2: non-streaming response parsing
# ---------------------------------------------------------------------------


def _dict_response(output: list[dict[str, Any]], status: str = "completed") -> dict[str, Any]:
    return {"status": status, "output": output}


def test_parse_non_streaming_dict_response() -> None:
    resp = _dict_response(
        [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "hello"},
                    {"type": "refusal", "refusal": "no"},
                ],
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "f",
                "arguments": "{}",
                "id": "i1",
            },
        ]
    )
    content, tcs = OpenAICompatibleModel2._parse_non_streaming(resp)
    assert content == "hellono"
    assert tcs == [
        {"id": "c1", "name": "f", "arguments": "{}", "item_id": "i1", "output_index": "1"}
    ]


def test_build_function_calls() -> None:
    out = OpenAICompatibleModel2._build_function_calls(
        [
            {"id": "c1", "name": "f", "arguments": '{"a": 1}'},
            {"id": "c2", "name": "g", "arguments": "broken"},
        ]
    )
    assert out == [
        {"id": "c1", "name": "f", "arguments": {"a": 1}},
        {"id": "c2", "name": "g", "arguments": {}},
    ]


def test_response_has_message_text_and_failed() -> None:
    has = OpenAICompatibleModel2._response_has_message_text
    assert has(_dict_response(
        [{"type": "message", "content": [{"type": "output_text", "text": ""}]}]
    ))
    assert not has(_dict_response([{"type": "reasoning"}]))

    OpenAICompatibleModel2._raise_for_failed_response(_dict_response([]))
    with pytest.raises(KISSError, match="boom"):
        OpenAICompatibleModel2._raise_for_failed_response(
            {"status": "failed", "error": {"message": "boom"}}
        )
    with pytest.raises(KISSError, match="max_output_tokens"):
        OpenAICompatibleModel2._raise_for_failed_response(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        )


def test_v2_extract_token_counts_dict_usage() -> None:
    m = make_v2()
    resp = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 30},
        }
    }
    assert m.extract_input_output_token_counts_from_response(resp) == (70, 7, 30, 0)
    assert m.extract_input_output_token_counts_from_response({}) == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# v2: streaming (_consume_stream with real event objects)
# ---------------------------------------------------------------------------


def _completed(output: list[dict[str, Any]]) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed", response=_dict_response(output)
    )


def test_consume_stream_text_deltas() -> None:
    tokens: list[str] = []
    m = make_v2(token_callback=tokens.append)
    m.initialize("hi")
    events = [
        SimpleNamespace(
            type="response.output_text.delta", output_index=0, content_index=0, delta="Hel"
        ),
        SimpleNamespace(
            type="response.output_text.delta", output_index=0, content_index=0, delta="lo"
        ),
        _completed(
            [{"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}]
        ),
    ]
    content, tool_calls, response = m._consume_stream(events)
    assert content == "Hello"
    assert tool_calls == []
    assert tokens == ["Hel", "lo"]
    assert response["status"] == "completed"


def test_consume_stream_done_only_text() -> None:
    tokens: list[str] = []
    m = make_v2(token_callback=tokens.append)
    m.initialize("hi")
    events = [
        SimpleNamespace(
            type="response.output_text.done", output_index=0, content_index=0, text="Hi"
        ),
        _completed([]),
    ]
    content, tool_calls, _resp = m._consume_stream(events)
    assert content == "Hi"
    assert tokens == ["Hi"]
    assert tool_calls == []


def test_consume_stream_reasoning_brackets() -> None:
    tokens: list[str] = []
    thinking: list[bool] = []
    m = make_v2(token_callback=tokens.append, thinking_callback=thinking.append)
    m.initialize("hi")
    events = [
        SimpleNamespace(
            type="response.reasoning_summary_text.delta",
            output_index=0,
            summary_index=0,
            delta="mull",
        ),
        SimpleNamespace(
            type="response.output_text.delta", output_index=1, content_index=0, delta="ans"
        ),
        _completed(
            [{"type": "message", "content": [{"type": "output_text", "text": "ans"}]}]
        ),
    ]
    content, _tcs, _resp = m._consume_stream(events)
    assert content == "ans"
    assert tokens == ["mull", "ans"]
    assert thinking == [True, False]


def test_consume_stream_function_call() -> None:
    tokens: list[str] = []
    m = make_v2(token_callback=tokens.append)
    m.initialize("hi")
    fc_final = {
        "type": "function_call",
        "call_id": "c1",
        "name": "f",
        "arguments": '{"a": 1}',
        "id": "i1",
    }
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call", id="i1", call_id="c1", name="f", arguments=""
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="i1",
            output_index=0,
            delta='{"a"',
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="i1",
            output_index=0,
            delta=": 1}",
        ),
        SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="i1",
            output_index=0,
            arguments='{"a": 1}',
        ),
        _completed([fc_final]),
    ]
    content, tool_calls, _resp = m._consume_stream(events)
    assert content == ""
    assert len(tool_calls) == 1
    slot = tool_calls[0]
    assert slot["id"] == "c1"
    assert slot["name"] == "f"
    assert json.loads(slot["arguments"]) == {"a": 1}


def test_consume_stream_missing_completed_raises() -> None:
    m = make_v2(token_callback=lambda _t: None)
    m.initialize("hi")
    events = [
        SimpleNamespace(
            type="response.output_text.delta", output_index=0, content_index=0, delta="x"
        )
    ]
    with pytest.raises(KISSError):
        m._consume_stream(events)


def test_consume_stream_failed_event_raises() -> None:
    m = make_v2(token_callback=lambda _t: None)
    m.initialize("hi")
    events = [
        SimpleNamespace(type="response.failed", response=None, error={"message": "kaboom"})
    ]
    with pytest.raises(KISSError, match="kaboom"):
        m._consume_stream(events)


# ---------------------------------------------------------------------------
# v1: conversation init & attachments
# ---------------------------------------------------------------------------


def test_v1_initialize_with_system_instruction() -> None:
    m = make_v1(model_config={"system_instruction": "be nice"})
    m.initialize("hello")
    assert m.conversation == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
    ]


def test_v1_add_function_results() -> None:
    m = make_v1()
    m.initialize("hello")
    m.conversation.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "cid", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        }
    )
    m.add_function_results_to_conversation_and_return([("f", {"result": "done"})])
    assert m.conversation[-1] == {
        "role": "tool",
        "tool_call_id": "cid",
        "content": "done",
    }


def test_v2_initialize_conversation_shape() -> None:
    m = make_v2()
    m.initialize("hello")
    assert m.conversation == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
