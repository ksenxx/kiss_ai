"""Tests for ConsolePrinter.

Tests verify correctness and accuracy of all terminal printing logic
extracted from ClaudeCodingAgent. Uses real objects with duck-typed
attributes (SimpleNamespace) as message inputs.
"""

import io
import unittest
from types import SimpleNamespace

from kiss.agents.coding_agents.print_to_console import (
    _LANG_MAP,
    _MAX_RESULT_LEN,
    ConsolePrinter,
)


class TestLangForPath(unittest.TestCase):
    def test_python(self):
        assert ConsolePrinter._lang_for_path("foo.py") == "python"

    def test_javascript(self):
        assert ConsolePrinter._lang_for_path("bar.js") == "javascript"

    def test_typescript(self):
        assert ConsolePrinter._lang_for_path("/a/b/c.ts") == "typescript"

    def test_bash_sh(self):
        assert ConsolePrinter._lang_for_path("script.sh") == "bash"

    def test_bash_bash(self):
        assert ConsolePrinter._lang_for_path("script.bash") == "bash"

    def test_ruby(self):
        assert ConsolePrinter._lang_for_path("app.rb") == "ruby"

    def test_rust(self):
        assert ConsolePrinter._lang_for_path("main.rs") == "rust"

    def test_go(self):
        assert ConsolePrinter._lang_for_path("main.go") == "go"

    def test_java(self):
        assert ConsolePrinter._lang_for_path("App.java") == "java"

    def test_c(self):
        assert ConsolePrinter._lang_for_path("main.c") == "c"

    def test_cpp(self):
        assert ConsolePrinter._lang_for_path("main.cpp") == "cpp"

    def test_header(self):
        assert ConsolePrinter._lang_for_path("util.h") == "c"

    def test_json(self):
        assert ConsolePrinter._lang_for_path("data.json") == "json"

    def test_yaml(self):
        assert ConsolePrinter._lang_for_path("config.yaml") == "yaml"

    def test_yml(self):
        assert ConsolePrinter._lang_for_path("config.yml") == "yaml"

    def test_toml(self):
        assert ConsolePrinter._lang_for_path("pyproject.toml") == "toml"

    def test_html(self):
        assert ConsolePrinter._lang_for_path("index.html") == "html"

    def test_css(self):
        assert ConsolePrinter._lang_for_path("style.css") == "css"

    def test_sql(self):
        assert ConsolePrinter._lang_for_path("query.sql") == "sql"

    def test_markdown(self):
        assert ConsolePrinter._lang_for_path("README.md") == "markdown"

    def test_unknown_extension_returns_ext(self):
        assert ConsolePrinter._lang_for_path("file.xyz") == "xyz"

    def test_no_extension_returns_text(self):
        assert ConsolePrinter._lang_for_path("Makefile") == "text"

    def test_all_lang_map_entries(self):
        for ext, lang in _LANG_MAP.items():
            assert ConsolePrinter._lang_for_path(f"file.{ext}") == lang


class TestConsolePrinterInit(unittest.TestCase):
    def test_default_state(self):
        p = ConsolePrinter(file=io.StringIO())
        assert p._mid_line is False
        assert p._current_block_type == ""
        assert p._tool_name == ""
        assert p._tool_json_buffer == ""

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
    def test_flush_when_mid_line(self):
        buf = io.StringIO()
        p = ConsolePrinter(file=buf)
        p._mid_line = True
        p._flush_newline()
        assert p._mid_line is False
        assert "\n" in buf.getvalue()

    def test_no_op_when_not_mid_line(self):
        buf = io.StringIO()
        p = ConsolePrinter(file=buf)
        p._flush_newline()
        assert buf.getvalue() == ""


class TestStreamDelta(unittest.TestCase):
    def test_sets_mid_line_true(self):
        p = ConsolePrinter(file=io.StringIO())
        p._stream_delta("hello")
        assert p._mid_line is True

    def test_sets_mid_line_false_on_newline(self):
        p = ConsolePrinter(file=io.StringIO())
        p._stream_delta("hello\n")
        assert p._mid_line is False

    def test_empty_string_no_change(self):
        p = ConsolePrinter(file=io.StringIO())
        p._stream_delta("")
        assert p._mid_line is False


class TestFormatToolCall(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_with_path(self):
        p, buf = self._make_printer()
        p._format_tool_call("Read", {"path": "/tmp/test.py"})
        out = buf.getvalue()
        assert "Read" in out
        assert "/tmp/test.py" in out

    def test_with_file_path_key(self):
        p, buf = self._make_printer()
        p._format_tool_call("Read", {"file_path": "/tmp/test.rb"})
        out = buf.getvalue()
        assert "/tmp/test.rb" in out

    def test_with_command(self):
        p, buf = self._make_printer()
        p._format_tool_call("Bash", {"command": "ls -la"})
        out = buf.getvalue()
        assert "Bash" in out
        assert "ls -la" in out

    def test_with_content(self):
        p, buf = self._make_printer()
        p._format_tool_call("Write", {"path": "test.py", "content": "print('hi')"})
        out = buf.getvalue()
        assert "Write" in out

    def test_with_edit_strings(self):
        p, buf = self._make_printer()
        p._format_tool_call(
            "Edit",
            {"path": "test.py", "old_string": "old code", "new_string": "new code"},
        )
        out = buf.getvalue()
        assert "Edit" in out
        assert "old" in out.lower()
        assert "new" in out.lower()

    def test_with_description(self):
        p, buf = self._make_printer()
        p._format_tool_call("Bash", {"description": "Run tests", "command": "pytest"})
        out = buf.getvalue()
        assert "Run tests" in out

    def test_no_args(self):
        p, buf = self._make_printer()
        p._format_tool_call("SomeTool", {})
        out = buf.getvalue()
        assert "SomeTool" in out
        assert "no arguments" in out

    def test_truncates_long_extra_values(self):
        p, buf = self._make_printer()
        p._format_tool_call("Tool", {"extra_key": "x" * 300})
        out = buf.getvalue()
        assert "..." in out

    def test_extra_key_short_value(self):
        p, buf = self._make_printer()
        p._format_tool_call("Tool", {"pattern": "*.py"})
        out = buf.getvalue()
        assert "pattern" in out
        assert "*.py" in out

    def test_old_string_none_new_string_empty(self):
        p, buf = self._make_printer()
        p._format_tool_call("Edit", {"path": "f.py", "old_string": "a", "new_string": ""})
        out = buf.getvalue()
        assert "old" in out.lower()
        assert "new" in out.lower()


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

    def test_error(self):
        p, buf = self._make_printer()
        p._print_tool_result("Error message", is_error=True)
        out = buf.getvalue()
        assert "FAILED" in out
        assert "Error message" in out

    def test_truncation(self):
        p, buf = self._make_printer()
        long = "x" * (_MAX_RESULT_LEN * 2)
        p._print_tool_result(long, is_error=False)
        out = buf.getvalue()
        assert "... (truncated) ..." in out

    def test_no_truncation_for_short(self):
        p, buf = self._make_printer()
        p._print_tool_result("short", is_error=False)
        out = buf.getvalue()
        assert "... (truncated) ..." not in out

    def test_multiline_content(self):
        p, buf = self._make_printer()
        p._print_tool_result("line1\nline2\nline3", is_error=False)
        out = buf.getvalue()
        assert "line1" in out
        assert "line2" in out
        assert "line3" in out


class TestPrintStreamEvent(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

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

    def test_content_block_start_tool_use(self):
        p, buf = self._make_printer()
        text = p.print_stream_event(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Read"},
            })
        )
        assert text == ""
        assert p._current_block_type == "tool_use"
        assert p._tool_name == "Read"
        assert p._tool_json_buffer == ""
        assert p._mid_line is True

    def test_content_block_start_text(self):
        p, _ = self._make_printer()
        p.print_stream_event(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "text"},
            })
        )
        assert p._current_block_type == "text"

    def test_content_block_start_no_content_block(self):
        p, _ = self._make_printer()
        p.print_stream_event(
            self._event({"type": "content_block_start"})
        )
        assert p._current_block_type == ""

    def test_content_block_delta_text(self):
        p, _ = self._make_printer()
        text = p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello world"},
            })
        )
        assert text == "Hello world"

    def test_content_block_delta_thinking(self):
        p, _ = self._make_printer()
        text = p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
            })
        )
        assert text == "Let me think..."

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

    def test_content_block_delta_json_accumulates(self):
        p, _ = self._make_printer()
        p._tool_json_buffer = '{"path":'
        p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": ' "/tmp"}'},
            })
        )
        assert p._tool_json_buffer == '{"path": "/tmp"}'

    def test_content_block_delta_unknown_delta_type(self):
        p, _ = self._make_printer()
        text = p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "unknown_type"},
            })
        )
        assert text == ""

    def test_content_block_stop_thinking(self):
        p, buf = self._make_printer()
        p._current_block_type = "thinking"
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""

    def test_content_block_stop_tool_use_valid_json(self):
        p, buf = self._make_printer()
        p._current_block_type = "tool_use"
        p._tool_name = "Read"
        p._tool_json_buffer = '{"path": "/tmp/test.txt"}'
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""
        out = buf.getvalue()
        assert "Read" in out
        assert "/tmp/test.txt" in out

    def test_content_block_stop_tool_use_invalid_json(self):
        p, buf = self._make_printer()
        p._current_block_type = "tool_use"
        p._tool_name = "Bash"
        p._tool_json_buffer = "invalid json{"
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""
        out = buf.getvalue()
        assert "Bash" in out

    def test_content_block_stop_text(self):
        p, _ = self._make_printer()
        p._current_block_type = "text"
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""

    def test_content_block_stop_empty_block_type(self):
        p, _ = self._make_printer()
        p._current_block_type = ""
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""

    def test_unknown_event_type(self):
        p, _ = self._make_printer()
        text = p.print_stream_event(self._event({"type": "unknown_event"}))
        assert text == ""

    def test_empty_event_dict(self):
        p, _ = self._make_printer()
        text = p.print_stream_event(self._event({}))
        assert text == ""

    def test_full_streaming_sequence(self):
        p, buf = self._make_printer()
        p.print_stream_event(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "text"},
            })
        )
        t1 = p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello "},
            })
        )
        t2 = p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "world!"},
            })
        )
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert t1 == "Hello "
        assert t2 == "world!"
        assert p._current_block_type == ""

    def test_tool_use_full_sequence(self):
        p, buf = self._make_printer()
        p.print_stream_event(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Write"},
            })
        )
        p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
            })
        )
        p.print_stream_event(
            self._event({
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": ' "test.py", "content": "x"}',
                },
            })
        )
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""
        out = buf.getvalue()
        assert "Write" in out
        assert "test.py" in out


class TestPrintMessageSystem(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_tool_output(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="tool_output", data={"content": "hello output"})
        p.print_message(msg)
        assert "hello output" in buf.getvalue()

    def test_tool_output_tracks_mid_line(self):
        p, _ = self._make_printer()
        msg = SimpleNamespace(subtype="tool_output", data={"content": "no newline"})
        p.print_message(msg)
        assert p._mid_line is True

    def test_tool_output_ending_newline(self):
        p, _ = self._make_printer()
        msg = SimpleNamespace(subtype="tool_output", data={"content": "with newline\n"})
        p.print_message(msg)
        assert p._mid_line is False

    def test_tool_output_empty_content(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        p.print_message(msg)
        assert buf.getvalue() == ""

    def test_other_subtype_ignored(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="other", data={"content": "should not appear"})
        p.print_message(msg)
        assert buf.getvalue() == ""


class TestPrintMessageResult(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_result_message(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(result="Final answer here")
        p.print_message(msg, step_count=5, budget_used=0.0123, total_tokens_used=1500)
        out = buf.getvalue()
        assert "Result" in out
        assert "Final answer here" in out
        assert "steps=5" in out
        assert "tokens=1500" in out
        assert "$0.0123" in out

    def test_result_no_budget(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(result="Answer")
        p.print_message(msg, step_count=1, budget_used=0.0, total_tokens_used=100)
        out = buf.getvalue()
        assert "N/A" in out

    def test_result_none(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(result=None)
        p.print_message(msg)
        out = buf.getvalue()
        assert "(no result)" in out

    def test_result_empty_string(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(result="")
        p.print_message(msg)
        out = buf.getvalue()
        assert "(no result)" in out

    def test_result_defaults(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(result="done")
        p.print_message(msg)
        out = buf.getvalue()
        assert "steps=0" in out
        assert "tokens=0" in out
        assert "N/A" in out


class TestPrintMessageUser(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_tool_result_success(self):
        p, buf = self._make_printer()
        block = SimpleNamespace(is_error=False, content="Success output")
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
        out = buf.getvalue()
        assert "OK" in out
        assert "Success output" in out

    def test_tool_result_error(self):
        p, buf = self._make_printer()
        block = SimpleNamespace(is_error=True, content="Error message")
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
        out = buf.getvalue()
        assert "FAILED" in out
        assert "Error message" in out

    def test_multiple_tool_results(self):
        p, buf = self._make_printer()
        blocks = [
            SimpleNamespace(is_error=False, content="Result 1"),
            SimpleNamespace(is_error=True, content="Error 2"),
        ]
        msg = SimpleNamespace(content=blocks)
        p.print_message(msg)
        out = buf.getvalue()
        assert "Result 1" in out
        assert "Error 2" in out

    def test_non_string_content(self):
        p, buf = self._make_printer()
        block = SimpleNamespace(is_error=False, content=["a", "b"])
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
        out = buf.getvalue()
        assert "OK" in out

    def test_blocks_without_is_error_skipped(self):
        p, buf = self._make_printer()
        block = SimpleNamespace(text="just text")
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
        # Should not crash and should not print tool result
        out = buf.getvalue()
        assert "OK" not in out
        assert "FAILED" not in out

    def test_empty_content_list(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(content=[])
        p.print_message(msg)
        out = buf.getvalue()
        assert "OK" not in out


class TestPrintMessageDispatch(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_unknown_message_type_no_crash(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(unknown_attr="value")
        p.print_message(msg)
        assert buf.getvalue() == ""

    def test_dispatch_prefers_system_over_content(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(
            subtype="tool_output",
            data={"content": "system text"},
            content=[],
        )
        p.print_message(msg)
        out = buf.getvalue()
        assert "system text" in out

    def test_dispatch_prefers_result_over_content(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(result="result text", content=[])
        p.print_message(msg)
        out = buf.getvalue()
        assert "result text" in out


class TestConsolePrinterMidLineInteraction(unittest.TestCase):
    def test_stream_then_tool_result(self):
        buf = io.StringIO()
        p = ConsolePrinter(file=buf)
        p._stream_delta("partial")
        assert p._mid_line is True
        p._flush_newline()
        assert p._mid_line is False
        p._print_tool_result("done", is_error=False)
        out = buf.getvalue()
        assert "done" in out

    def test_system_output_then_flush(self):
        buf = io.StringIO()
        p = ConsolePrinter(file=buf)
        msg = SimpleNamespace(subtype="tool_output", data={"content": "output"})
        p.print_message(msg)
        assert p._mid_line is True
        p._flush_newline()
        assert p._mid_line is False


class TestConstants(unittest.TestCase):
    def test_lang_map_not_empty(self):
        assert len(_LANG_MAP) > 0

    def test_max_result_len_positive(self):
        assert _MAX_RESULT_LEN > 0

    def test_lang_map_values_are_strings(self):
        for k, v in _LANG_MAP.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


if __name__ == "__main__":
    unittest.main()
