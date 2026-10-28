"""Audit 2026-10-15 (core-models): behaviour pinned by the dedupe of
``OpenAICompatibleModel._chat_message_to_responses_items`` and of the
``OpenAICompatibleModel2`` stream helpers ``_part_final_text`` /
``_function_call_item``.

The helpers are pure (no network, no threads); every branch of each is
exercised here with real argument shapes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

_conv = OpenAICompatibleModel._chat_message_to_responses_items


def test_assistant_list_content_is_joined_with_and_without_tool_calls() -> None:
    parts = [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "data:x"}},
        "not-a-dict",
        {"type": "text", "text": "b"},
    ]
    assert _conv({"role": "assistant", "content": parts}) == [
        {"role": "assistant", "content": "ab"}
    ]
    items = _conv(
        {
            "role": "assistant",
            "content": parts,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"arguments": None}},
            ],
        }
    )
    assert items[0] == {"role": "assistant", "content": "ab"}
    assert items[1] == {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
    assert items[2] == {"type": "function_call", "call_id": "c2", "name": "", "arguments": "{}"}


def test_assistant_blank_or_missing_text_yields_only_tool_calls() -> None:
    assert _conv({"role": "assistant", "content": None}) == []
    assert _conv({"role": "assistant", "content": "   "}) == []
    assert _conv({"role": "assistant", "content": [{"type": "text", "text": " "}]}) == []
    assert _conv({"role": "assistant", "content": 42}) == []
    items = _conv(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c3", "type": "function", "function": {"name": "g", "arguments": {"k": 1}}}
            ],
        }
    )
    assert len(items) == 1
    assert items[0]["call_id"] == "c3"
    assert json.loads(items[0]["arguments"]) == {"k": 1}


def test_non_assistant_roles_ignore_tool_calls() -> None:
    items = _conv({"role": "user", "content": "hi", "tool_calls": [{"id": "x"}]})
    assert items == [{"role": "user", "content": "hi"}]


def test_part_final_text_handles_every_part_shape() -> None:
    f = OpenAICompatibleModel2._part_final_text
    assert f(SimpleNamespace(type="output_text", text="hello")) == "hello"
    assert f(SimpleNamespace(type="output_text", text=None)) == ""
    assert f(SimpleNamespace(type="refusal", refusal="no")) == "no"
    assert f(SimpleNamespace(type="refusal")) == ""
    assert f(SimpleNamespace(type="input_image")) is None
    assert f(SimpleNamespace()) is None
    assert f(None) is None


def test_function_call_item_keeps_item_id_only_when_present() -> None:
    f = OpenAICompatibleModel2._streamed_call_item
    assert f({"id": "call_1", "name": "fn", "arguments": '{"a":1}'}) == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "fn",
        "arguments": '{"a":1}',
    }
    assert f({"id": "call_1", "name": "fn", "arguments": "", "item_id": ""}) == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "fn",
        "arguments": "",
    }
    assert f({"id": "call_2", "name": "fn", "arguments": "{}", "item_id": "fc_9"}) == {
        "type": "function_call",
        "call_id": "call_2",
        "name": "fn",
        "arguments": "{}",
        "id": "fc_9",
    }
    assert f({}) == {"type": "function_call", "call_id": "", "name": "", "arguments": ""}
