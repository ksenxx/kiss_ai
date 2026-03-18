"""Tests for code deduplication refactors.

Tests the shared helpers extracted during the deduplication pass:
- parse_result_yaml (shared YAML result parsing)
- _build_tool_call_lists (shared tool call list construction)
- Removal of trivial _handle_stream_event wrappers
"""

import unittest
from typing import Any

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.printer import parse_result_yaml


class TestParseResultYaml(unittest.TestCase):
    """Test the shared parse_result_yaml helper."""

    def test_valid_yaml_with_summary(self) -> None:
        raw = "success: true\nsummary: task done\n"
        result = parse_result_yaml(raw)
        assert result is not None
        assert result["success"] is True
        assert result["summary"] == "task done"

    def test_valid_yaml_without_summary(self) -> None:
        raw = "status: ok\nresult: done\n"
        result = parse_result_yaml(raw)
        assert result is None

    def test_invalid_yaml(self) -> None:
        raw = ":::invalid yaml{{{"
        result = parse_result_yaml(raw)
        assert result is None

    def test_non_dict_yaml(self) -> None:
        raw = "- item1\n- item2\n"
        result = parse_result_yaml(raw)
        assert result is None

    def test_empty_string(self) -> None:
        result = parse_result_yaml("")
        assert result is None

    def test_success_false_with_summary(self) -> None:
        raw = "success: false\nis_continue: true\nsummary: partial progress\n"
        result = parse_result_yaml(raw)
        assert result is not None
        assert result["success"] is False
        assert result["summary"] == "partial progress"


class TestBuildToolCallLists(unittest.TestCase):
    """Test the shared _build_tool_call_lists helper."""

    def test_single_valid_entry(self) -> None:
        entries = [("call_1", "finish", '{"result": "done"}')]
        fc, raw = OpenAICompatibleModel._build_tool_call_lists(entries)
        assert len(fc) == 1
        assert fc[0] == {"id": "call_1", "name": "finish", "arguments": {"result": "done"}}
        assert raw[0]["id"] == "call_1"
        assert raw[0]["type"] == "function"
        assert raw[0]["function"]["name"] == "finish"
        assert raw[0]["function"]["arguments"] == '{"result": "done"}'

    def test_multiple_entries(self) -> None:
        entries = [
            ("c1", "read", '{"path": "/tmp/f"}'),
            ("c2", "write", '{"path": "/tmp/g", "content": "hi"}'),
        ]
        fc, raw = OpenAICompatibleModel._build_tool_call_lists(entries)
        assert len(fc) == 2
        assert len(raw) == 2
        assert fc[0]["name"] == "read"
        assert fc[1]["name"] == "write"
        assert fc[1]["arguments"]["content"] == "hi"

    def test_invalid_json_fallback(self) -> None:
        entries = [("c1", "tool", "not valid json")]
        fc, raw = OpenAICompatibleModel._build_tool_call_lists(entries)
        assert len(fc) == 1
        assert fc[0]["arguments"] == {}
        assert raw[0]["function"]["arguments"] == "not valid json"

    def test_empty_entries(self) -> None:
        fc, raw = OpenAICompatibleModel._build_tool_call_lists([])
        assert fc == []
        assert raw == []


class TestParseToolCallAccum(unittest.TestCase):
    """Test _parse_tool_call_accum delegates to _build_tool_call_lists."""

    def test_sorted_order(self) -> None:
        accum = {
            2: {"id": "c2", "name": "write", "arguments": '{"x": 1}'},
            0: {"id": "c0", "name": "read", "arguments": '{"y": 2}'},
        }
        fc, raw = OpenAICompatibleModel._parse_tool_call_accum(accum)
        assert len(fc) == 2
        assert fc[0]["name"] == "read"
        assert fc[1]["name"] == "write"


class TestParseToolCallsFromMessage(unittest.TestCase):
    """Test _parse_tool_calls_from_message delegates to _build_tool_call_lists."""

    def test_no_tool_calls(self) -> None:
        from openai.types.chat.chat_completion_message import ChatCompletionMessage

        msg = ChatCompletionMessage(role="assistant", content="test", tool_calls=None)
        fc, raw = OpenAICompatibleModel._parse_tool_calls_from_message(msg)
        assert fc == []
        assert raw == []

    def test_with_tool_calls(self) -> None:
        from openai.types.chat.chat_completion_message import ChatCompletionMessage
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
            Function,
        )

        msg = ChatCompletionMessage(
            role="assistant",
            content="test",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="tc1",
                    type="function",
                    function=Function(name="bash", arguments='{"command": "ls"}'),
                ),
            ],
        )
        fc, raw = OpenAICompatibleModel._parse_tool_calls_from_message(msg)
        assert len(fc) == 1
        assert fc[0]["name"] == "bash"
        assert fc[0]["arguments"] == {"command": "ls"}


class TestStreamEventParseDirect(unittest.TestCase):
    """Verify parse_stream_event works directly (no _handle_stream_event wrapper)."""

    def test_browser_printer_parse_stream_event(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        events: list[dict[str, Any]] = []
        printer.broadcast = lambda ev: events.append(ev)  # type: ignore[assignment]

        class FakeEvent:
            event = {"type": "content_block_start", "content_block": {"type": "thinking"}}

        text = printer.parse_stream_event(FakeEvent())
        assert text == ""
        assert any(e.get("type") == "thinking_start" for e in events)

    def test_console_printer_parse_stream_event(self) -> None:
        from io import StringIO

        from kiss.core.print_to_console import ConsolePrinter

        printer = ConsolePrinter(file=StringIO())

        class FakeEvent:
            event = {"type": "content_block_start", "content_block": {"type": "thinking"}}

        text = printer.parse_stream_event(FakeEvent())
        assert text == ""


if __name__ == "__main__":
    unittest.main()
