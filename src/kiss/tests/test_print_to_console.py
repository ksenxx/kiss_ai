"""Tests for ConsolePrinter.

Tests verify correctness and accuracy of all terminal printing logic
extracted from ClaudeCodingAgent. Uses real objects with duck-typed
attributes (SimpleNamespace) as message inputs.
"""

import io
import unittest
from types import SimpleNamespace

from kiss.core.print_to_console import ConsolePrinter


class TestConsolePrinterInit(unittest.TestCase):
    def test_reset(self):
        p = ConsolePrinter(file=io.StringIO())
        p._mid_line = True
        p._current_block_type = "thinking"
        p._tool_name = "Read"
        p._tool_json_buffer = '{"path": "x"}'
        p.reset()
        assert p._mid_line is False
        assert p._current_block_type == ""
        assert p._tool_name == ""
        assert p._tool_json_buffer == ""


class TestStreamDelta(unittest.TestCase):
    def test_empty_string_no_change(self):
        p = ConsolePrinter(file=io.StringIO())
        p._stream_delta("")
        assert p._mid_line is False


class TestFormatToolCall(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_with_content(self):
        p, buf = self._make_printer()
        p._format_tool_call("Write", {"path": "test.py", "content": "print('hi')"})
        out = buf.getvalue()
        assert "Write" in out

    def test_with_description(self):
        p, buf = self._make_printer()
        p._format_tool_call("Bash", {"description": "Run tests", "command": "pytest"})
        out = buf.getvalue()
        assert "Run tests" in out


class TestPrintToolResult(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_success(self):
        p, buf = self._make_printer()
        p._print_tool_result("Success output", is_error=False)
        out = buf.getvalue()
        assert "OK" in out
        assert "Success output" in out


class TestPrintStreamEvent(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

    def test_content_block_delta_json(self):
        p, _ = self._make_printer()
        p._tool_json_buffer = ""
        text = p.print(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
            }),
            type="stream_event",
        )
        assert text == ""
        assert p._tool_json_buffer == '{"path":'

    def test_content_block_delta_thinking(self):
        p, _ = self._make_printer()
        text = p.print(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
            }),
            type="stream_event",
        )
        assert text == "Let me think..."

    def test_content_block_delta_unknown_delta_type(self):
        p, _ = self._make_printer()
        text = p.print(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "unknown_type"},
            }),
            type="stream_event",
        )
        assert text == ""

    def test_content_block_start_text(self):
        p, _ = self._make_printer()
        p.print(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "text"},
            }),
            type="stream_event",
        )
        assert p._current_block_type == "text"

    def test_content_block_start_thinking(self):
        p, buf = self._make_printer()
        text = p.print(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "thinking"},
            }),
            type="stream_event",
        )
        assert text == ""
        assert p._current_block_type == "thinking"
        assert "Thinking" in buf.getvalue()

    def test_content_block_stop_thinking(self):
        p, buf = self._make_printer()
        p._current_block_type = "thinking"
        p.print(self._event({"type": "content_block_stop"}), type="stream_event")
        assert p._current_block_type == ""

    def test_content_block_stop_tool_use_invalid_json(self):
        p, buf = self._make_printer()
        p._current_block_type = "tool_use"
        p._tool_name = "Bash"
        p._tool_json_buffer = "invalid json{"
        p.print(self._event({"type": "content_block_stop"}), type="stream_event")
        assert p._current_block_type == ""
        out = buf.getvalue()
        assert "Bash" in out

    def test_empty_event_dict(self):
        p, _ = self._make_printer()
        text = p.print(self._event({}), type="stream_event")
        assert text == ""


class TestPrintMessageSystem(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_other_subtype_ignored(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="other", data={"content": "should not appear"})
        p.print(msg, type="message")
        assert buf.getvalue() == ""

    def test_tool_output(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="tool_output", data={"content": "hello output"})
        p.print(msg, type="message")
        assert "hello output" in buf.getvalue()

    def test_tool_output_empty_content(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        p.print(msg, type="message")
        assert buf.getvalue() == ""


class TestPrintMessageUser(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_blocks_without_is_error_skipped(self):
        p, buf = self._make_printer()
        block = SimpleNamespace(text="just text")
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        out = buf.getvalue()
        assert "OK" not in out
        assert "FAILED" not in out


class TestPrintMessageDispatch(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_unknown_message_type_no_crash(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(unknown_attr="value")
        p.print(msg, type="message")
        assert buf.getvalue() == ""


class TestPrint(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_print_string(self):
        p, buf = self._make_printer()
        p.print("hello world")
        assert "hello world" in buf.getvalue()

    def test_print_flushes_mid_line(self):
        p, buf = self._make_printer()
        p._mid_line = True
        p.print("after flush")
        out = buf.getvalue()
        assert "\n" in out
        assert "after flush" in out


class TestTokenCallback(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_token_callback_streams_text(self):
        import asyncio
        p, buf = self._make_printer()
        asyncio.run(p.token_callback("hello"))
        assert "hello" in buf.getvalue()
        assert p._mid_line is True

    def test_token_callback_empty_string(self):
        import asyncio
        p, buf = self._make_printer()
        asyncio.run(p.token_callback(""))
        assert p._mid_line is False

    def test_token_callback_ending_newline(self):
        import asyncio
        p, buf = self._make_printer()
        asyncio.run(p.token_callback("line\n"))
        assert p._mid_line is False


if __name__ == "__main__":
    unittest.main()
