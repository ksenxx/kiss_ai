"""Integration tests for Model._replace_last_assistant_with_tool_calls.

Verifies the shared helper method that replaces a plain assistant message
with one containing tool call metadata, used by ClaudeCodeModel and
OpenAIModel text-based tool calling paths.
"""

from __future__ import annotations

import json

from kiss.core.models.claude_code_model import ClaudeCodeModel


def _make_model() -> ClaudeCodeModel:
    """Create a concrete Model subclass for testing."""
    return ClaudeCodeModel("claude-code", model_config={})


def test_replace_with_tool_calls() -> None:
    """_replace_last_assistant_with_tool_calls builds correct conversation entry."""
    model = _make_model()
    model.conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "I'll call a tool"},
    ]
    function_calls = [
        {"id": "call_1", "name": "read_file", "arguments": {"path": "/tmp/f.txt"}},
        {"id": "call_2", "name": "write_file",
         "arguments": {"path": "/tmp/g.txt", "content": "hi"}},
    ]
    model._replace_last_assistant_with_tool_calls("I'll call a tool", function_calls)

    msg = model.conversation[-1]
    assert msg["role"] == "assistant"
    assert msg["content"] == "I'll call a tool"
    assert len(msg["tool_calls"]) == 2

    tc1 = msg["tool_calls"][0]
    assert tc1["id"] == "call_1"
    assert tc1["type"] == "function"
    assert tc1["function"]["name"] == "read_file"
    assert json.loads(tc1["function"]["arguments"]) == {"path": "/tmp/f.txt"}

    tc2 = msg["tool_calls"][1]
    assert tc2["id"] == "call_2"
    assert tc2["function"]["name"] == "write_file"
    assert json.loads(tc2["function"]["arguments"]) == {"path": "/tmp/g.txt", "content": "hi"}


def test_replace_preserves_earlier_messages() -> None:
    """The helper only replaces the last message, not earlier ones."""
    model = _make_model()
    model.conversation = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "tool call text"},
    ]
    model._replace_last_assistant_with_tool_calls(
        "tool call text",
        [{"id": "c1", "name": "bash", "arguments": {"cmd": "ls"}}],
    )
    assert model.conversation[0] == {"role": "user", "content": "first"}
    assert model.conversation[1] == {"role": "assistant", "content": "first reply"}
    assert model.conversation[2] == {"role": "user", "content": "second"}
    assert "tool_calls" in model.conversation[3]


def test_single_tool_call() -> None:
    """Works correctly with a single tool call."""
    model = _make_model()
    model.conversation = [{"role": "assistant", "content": "calling tool"}]
    model._replace_last_assistant_with_tool_calls(
        "calling tool",
        [{"id": "x", "name": "func", "arguments": {}}],
    )
    msg = model.conversation[-1]
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "func"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {}
