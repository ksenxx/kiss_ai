"""Tests for ConsolePrinter.

Tests verify correctness and accuracy of all terminal printing logic
extracted from ClaudeCodingAgent. Uses real objects with duck-typed
attributes (SimpleNamespace) as message inputs.
"""

import io
import unittest
from types import SimpleNamespace

from kiss.agents.coding_agents.print_to_console import ConsolePrinter
from kiss.agents.coding_agents.printer_common import lang_for_path


class TestLangForPath(unittest.TestCase):
    def test_c(self):
        assert lang_for_path("main.c") == "c"

    def test_sql(self):
        assert lang_for_path("query.sql") == "sql"

    def test_unknown_extension_returns_ext(self):
        assert lang_for_path("file.xyz") == "xyz"


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


class TestFlushNewline(unittest.TestCase):
    def test_no_op_when_not_mid_line(self):
        buf = io.StringIO()
        p = ConsolePrinter(file=buf)
        p._flush_newline()
        assert buf.getvalue() == ""


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
        text = p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
            })
        )
        assert text == ""
        assert p._tool_json_buffer == '{"path":'

    def test_content_block_delta_thinking(self):
        p, _ = self._make_printer()
        text = p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
            })
        )
        assert text == "Let me think..."

    def test_content_block_delta_unknown_delta_type(self):
        p, _ = self._make_printer()
        text = p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "unknown_type"},
            })
        )
        assert text == ""

    def test_content_block_start_text(self):
        p, _ = self._make_printer()
        p.print_stream_event(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "text"},
            })
        )
        assert p._current_block_type == "text"

    def test_content_block_start_thinking(self):
        p, buf = self._make_printer()
        text = p.print_stream_event(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "thinking"},
            })
        )
        assert text == ""
        assert p._current_block_type == "thinking"
        assert "Thinking" in buf.getvalue()

    def test_content_block_stop_thinking(self):
        p, buf = self._make_printer()
        p._current_block_type = "thinking"
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""

    def test_content_block_stop_tool_use_invalid_json(self):
        p, buf = self._make_printer()
        p._current_block_type = "tool_use"
        p._tool_name = "Bash"
        p._tool_json_buffer = "invalid json{"
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""
        out = buf.getvalue()
        assert "Bash" in out

    def test_empty_event_dict(self):
        p, _ = self._make_printer()
        text = p.print_stream_event(self._event({}))
        assert text == ""


class TestPrintMessageSystem(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_other_subtype_ignored(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="other", data={"content": "should not appear"})
        p.print_message(msg)
        assert buf.getvalue() == ""

    def test_tool_output(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="tool_output", data={"content": "hello output"})
        p.print_message(msg)
        assert "hello output" in buf.getvalue()

    def test_tool_output_empty_content(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        p.print_message(msg)
        assert buf.getvalue() == ""


class TestPrintMessageUser(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_blocks_without_is_error_skipped(self):
        p, buf = self._make_printer()
        block = SimpleNamespace(text="just text")
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
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
        p.print_message(msg)
        assert buf.getvalue() == ""


if __name__ == "__main__":
    unittest.main()
