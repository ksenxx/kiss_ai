"""Tests for ConsolePrinter.

Tests verify correctness and accuracy of all terminal printing logic.
Uses real objects with duck-typed attributes (SimpleNamespace) as
message inputs.
"""

import asyncio
import io
import unittest
from types import SimpleNamespace

from kiss.core.print_to_console import ConsolePrinter


def _make() -> tuple[ConsolePrinter, io.StringIO]:
    buf = io.StringIO()
    return ConsolePrinter(file=buf), buf


class TestInit(unittest.TestCase):
    def test_default_file_is_stdout(self):
        import sys
        p = ConsolePrinter()
        assert p._file is sys.stdout

    def test_custom_file(self):
        buf = io.StringIO()
        p = ConsolePrinter(file=buf)
        assert p._file is buf


class TestReset(unittest.TestCase):
    def test_reset_clears_mid_line(self):
        p, buf = _make()
        p._mid_line = True
        p._current_block_type = "thinking"
        p.reset()
        assert not p._mid_line
        assert p._current_block_type == ""


class TestPrintText(unittest.TestCase):
    def test_plain_text(self):
        p, buf = _make()
        result = p.print("hello world", type="text")
        assert "hello world" in buf.getvalue()
        assert result == ""

    def test_text_with_kwargs(self):
        p, buf = _make()
        p.print("styled", type="text", style="bold")
        assert "styled" in buf.getvalue()


class TestSystemPromptPanel(unittest.TestCase):
    def test_system_prompt_renders_panel(self):
        p, buf = _make()
        p.print("You are a helpful agent.", type="system_prompt")
        out = buf.getvalue()
        assert "System Prompt" in out
        assert "You are a helpful agent." in out

    def test_prompt_renders_panel(self):
        p, buf = _make()
        p.print("Do the task.", type="prompt")
        out = buf.getvalue()
        assert "Prompt" in out
        assert "Do the task." in out

    def test_system_prompt_flushes_newline(self):
        p, buf = _make()
        p._mid_line = True
        p.print("sys prompt", type="system_prompt")
        assert "System Prompt" in buf.getvalue()


class TestPrintStreamEvent(unittest.TestCase):
    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

    def test_thinking_block_start_and_end(self):
        p, buf = _make()
        # Start thinking
        p.print(self._event({
            "type": "content_block_start",
            "content_block": {"type": "thinking"},
        }), type="stream_event")
        out = buf.getvalue()
        assert "Thinking" in out

        # Thinking delta
        text = p.print(self._event({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        }), type="stream_event")
        assert text == "hmm"

        # Stop thinking
        p.print(self._event({
            "type": "content_block_stop",
        }), type="stream_event")

    def test_text_delta(self):
        p, buf = _make()
        # Start text block
        p.print(self._event({
            "type": "content_block_start",
            "content_block": {"type": "text"},
        }), type="stream_event")
        text = p.print(self._event({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"},
        }), type="stream_event")
        assert text == "hello"
        # End text block
        p.print(self._event({
            "type": "content_block_stop",
        }), type="stream_event")

    def test_tool_use_block(self):
        p, buf = _make()
        # Start tool_use
        p.print(self._event({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Bash"},
        }), type="stream_event")
        out1 = buf.getvalue()
        assert "Bash" in out1

        # JSON delta
        p.print(self._event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'},
        }), type="stream_event")

        # Stop tool_use
        p.print(self._event({
            "type": "content_block_stop",
        }), type="stream_event")
        out2 = buf.getvalue()
        assert "ls" in out2

    def test_tool_use_invalid_json(self):
        p, buf = _make()
        p.print(self._event({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Bad"},
        }), type="stream_event")
        p.print(self._event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "{invalid"},
        }), type="stream_event")
        p.print(self._event({
            "type": "content_block_stop",
        }), type="stream_event")
        # Should not crash

    def test_unknown_event_type(self):
        p, buf = _make()
        result = p.print(self._event({"type": "unknown"}), type="stream_event")
        assert result == ""

    def test_unknown_delta_type(self):
        p, buf = _make()
        p._current_block_type = "text"
        result = p.print(self._event({
            "type": "content_block_delta",
            "delta": {"type": "unknown_delta"},
        }), type="stream_event")
        assert result == ""


class TestPrintUsageInfo(unittest.TestCase):
    def test_usage_info(self):
        p, buf = _make()
        p.print("tokens: 500, cost: $0.01", type="usage_info")
        out = buf.getvalue()
        assert "tokens: 500" in out


class TestPrintBashStream(unittest.TestCase):
    def test_bash_stream_writes_directly(self):
        p, buf = _make()
        p.print("line1\n", type="bash_stream")
        assert buf.getvalue() == "line1\n"
        assert not p._mid_line

    def test_bash_stream_no_newline(self):
        p, buf = _make()
        p.print("partial", type="bash_stream")
        assert p._mid_line


class TestPrintToolCall(unittest.TestCase):
    def test_tool_call_with_file_and_command(self):
        p, buf = _make()
        p.print("Bash", type="tool_call", tool_input={
            "command": "ls -la",
            "description": "List files",
        })
        out = buf.getvalue()
        assert "Bash" in out
        assert "ls -la" in out
        assert "List files" in out

    def test_tool_call_with_content_and_path(self):
        p, buf = _make()
        p.print("Write", type="tool_call", tool_input={
            "file_path": "/tmp/test.py",
            "content": "print('hello')",
        })
        out = buf.getvalue()
        assert "Write" in out
        assert "/tmp/test.py" in out
        assert "print('hello')" in out

    def test_tool_call_with_old_new_string(self):
        p, buf = _make()
        p.print("Edit", type="tool_call", tool_input={
            "file_path": "/tmp/test.py",
            "old_string": "old code",
            "new_string": "new code",
        })
        out = buf.getvalue()
        assert "old" in out
        assert "new" in out

    def test_tool_call_with_extras(self):
        p, buf = _make()
        p.print("Custom", type="tool_call", tool_input={
            "custom_key": "custom_val",
        })
        out = buf.getvalue()
        assert "custom_key" in out

    def test_tool_call_empty_input(self):
        p, buf = _make()
        p.print("Empty", type="tool_call", tool_input={})
        out = buf.getvalue()
        assert "no arguments" in out

    def test_tool_call_with_path_key(self):
        p, buf = _make()
        p.print("Read", type="tool_call", tool_input={
            "path": "/tmp/file.js",
        })
        out = buf.getvalue()
        assert "/tmp/file.js" in out


class TestPrintToolResult(unittest.TestCase):
    def test_tool_result_success(self):
        p, buf = _make()
        p.print("All good", type="tool_result", is_error=False)
        out = buf.getvalue()
        assert "OK" in out
        assert "All good" in out

    def test_tool_result_error(self):
        p, buf = _make()
        p.print("Something failed", type="tool_result", is_error=True)
        out = buf.getvalue()
        assert "FAILED" in out
        assert "Something failed" in out

    def test_tool_result_truncation(self):
        p, buf = _make()
        long_content = "x" * 5000
        p.print(long_content, type="tool_result")
        out = buf.getvalue()
        assert "truncated" in out


class TestPrintResult(unittest.TestCase):
    def test_result_with_yaml_success(self):
        import yaml
        p, buf = _make()
        content = yaml.dump({"success": True, "summary": "Task completed"})
        p.print(content, type="result", cost="$0.05", step_count=3, total_tokens=1000)
        out = buf.getvalue()
        assert "PASSED" in out
        assert "Task completed" in out
        assert "steps=3" in out
        assert "tokens=1000" in out

    def test_result_with_yaml_failure(self):
        import yaml
        p, buf = _make()
        content = yaml.dump({"success": False, "summary": "Something broke"})
        p.print(content, type="result")
        out = buf.getvalue()
        assert "FAILED" in out
        assert "Something broke" in out

    def test_result_with_plain_text(self):
        p, buf = _make()
        p.print("just plain text", type="result")
        out = buf.getvalue()
        assert "just plain text" in out

    def test_result_no_content(self):
        p, buf = _make()
        p.print(None, type="result")
        out = buf.getvalue()
        assert "no result" in out


class TestPrintMessageSystem(unittest.TestCase):
    def test_tool_output_with_text(self):
        p, buf = _make()
        msg = SimpleNamespace(subtype="tool_output", data={"content": "output text\n"})
        p.print(msg, type="message")
        assert "output text" in buf.getvalue()
        assert not p._mid_line

    def test_tool_output_empty(self):
        p, buf = _make()
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        p.print(msg, type="message")
        assert buf.getvalue() == ""

    def test_tool_output_no_trailing_newline(self):
        p, buf = _make()
        msg = SimpleNamespace(subtype="tool_output", data={"content": "no newline"})
        p.print(msg, type="message")
        assert p._mid_line

    def test_other_subtype_ignored(self):
        p, buf = _make()
        msg = SimpleNamespace(subtype="other", data={"content": "should not appear"})
        p.print(msg, type="message")
        assert buf.getvalue() == ""


class TestPrintMessageResult(unittest.TestCase):
    def test_message_result_panel(self):
        import yaml
        p, buf = _make()
        result_str = yaml.dump({"success": True, "summary": "Done"})
        msg = SimpleNamespace(result=result_str)
        p.print(msg, type="message", step_count=5, budget_used=0.123, total_tokens_used=2000)
        out = buf.getvalue()
        assert "PASSED" in out
        assert "Done" in out
        assert "steps=5" in out
        assert "$0.1230" in out

    def test_message_result_no_budget(self):
        p, buf = _make()
        msg = SimpleNamespace(result="plain result")
        p.print(msg, type="message")
        out = buf.getvalue()
        assert "N/A" in out

    def test_message_result_none(self):
        p, buf = _make()
        msg = SimpleNamespace(result=None)
        p.print(msg, type="message")
        out = buf.getvalue()
        assert "no result" in out


class TestPrintMessageUser(unittest.TestCase):
    def test_blocks_with_is_error_true(self):
        p, buf = _make()
        block = SimpleNamespace(is_error=True, content="error content")
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        out = buf.getvalue()
        assert "FAILED" in out
        assert "error content" in out

    def test_blocks_with_is_error_false(self):
        p, buf = _make()
        block = SimpleNamespace(is_error=False, content="ok content")
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        out = buf.getvalue()
        assert "OK" in out
        assert "ok content" in out

    def test_blocks_with_non_string_content(self):
        p, buf = _make()
        block = SimpleNamespace(is_error=False, content=12345)
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        out = buf.getvalue()
        assert "12345" in out

    def test_blocks_without_is_error_skipped(self):
        p, buf = _make()
        block = SimpleNamespace(text="just text")
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        out = buf.getvalue()
        assert "OK" not in out
        assert "FAILED" not in out


class TestPrintMessageDispatch(unittest.TestCase):
    def test_unknown_message_type_no_crash(self):
        p, buf = _make()
        msg = SimpleNamespace(unknown_attr="value")
        p.print(msg, type="message")
        assert buf.getvalue() == ""


class TestPrintUnknownType(unittest.TestCase):
    def test_unknown_type_returns_empty(self):
        p, buf = _make()
        result = p.print("anything", type="nonexistent_type")
        assert result == ""


class TestTokenCallback(unittest.TestCase):
    def test_thinking_style(self):
        p, buf = _make()
        p._current_block_type = "thinking"
        asyncio.run(p.token_callback("thought"))
        assert "thought" in buf.getvalue()

    def test_text_style(self):
        p, buf = _make()
        p._current_block_type = "text"
        asyncio.run(p.token_callback("word"))
        assert "word" in buf.getvalue()

    def test_empty_token(self):
        p, buf = _make()
        asyncio.run(p.token_callback(""))
        assert buf.getvalue() == ""


class TestStreamDelta(unittest.TestCase):
    def test_mid_line_tracking_with_newline(self):
        p, buf = _make()
        p._stream_delta("hello\n")
        assert not p._mid_line

    def test_mid_line_tracking_without_newline(self):
        p, buf = _make()
        p._stream_delta("hello")
        assert p._mid_line


class TestFlushNewline(unittest.TestCase):
    def test_flush_when_mid_line(self):
        p, buf = _make()
        p._mid_line = True
        p._flush_newline()
        assert "\n" in buf.getvalue()
        assert not p._mid_line

    def test_no_flush_when_not_mid_line(self):
        p, buf = _make()
        p._mid_line = False
        p._flush_newline()
        assert buf.getvalue() == ""


class TestStreamingFlow(unittest.TestCase):
    """Test the full streaming flow: block_start -> token_callback -> block_stop."""

    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

    def test_full_thinking_flow(self):
        p, buf = _make()
        # Start thinking block
        p.print(self._event({
            "type": "content_block_start",
            "content_block": {"type": "thinking"},
        }), type="stream_event")

        # Stream tokens in thinking mode
        asyncio.run(p.token_callback("I think"))

        # End thinking
        p.print(self._event({"type": "content_block_stop"}), type="stream_event")

        out = buf.getvalue()
        assert "Thinking" in out
        assert "I think" in out

    def test_full_text_then_tool_flow(self):
        p, buf = _make()
        # Text block
        p.print(self._event({
            "type": "content_block_start",
            "content_block": {"type": "text"},
        }), type="stream_event")
        asyncio.run(p.token_callback("Let me help\n"))
        p.print(self._event({"type": "content_block_stop"}), type="stream_event")

        # Tool block
        p.print(self._event({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Read"},
        }), type="stream_event")
        p.print(self._event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"file_path": "/tmp/a.py"}'},
        }), type="stream_event")
        p.print(self._event({"type": "content_block_stop"}), type="stream_event")

        out = buf.getvalue()
        assert "Let me help" in out
        assert "Read" in out


class TestFormatResultContent(unittest.TestCase):
    def test_non_yaml_returns_raw(self):
        result = ConsolePrinter._format_result_content("just plain text")
        assert result == "just plain text"

    def test_yaml_without_summary_returns_raw(self):
        result = ConsolePrinter._format_result_content("key: value")
        assert result == "key: value"

    def test_invalid_yaml_returns_raw(self):
        result = ConsolePrinter._format_result_content(":: invalid: yaml: [")
        assert isinstance(result, str)
        assert "invalid" in result

    def test_yaml_with_summary_but_no_success(self):
        import yaml
        raw = yaml.dump({"summary": "Just a summary"})
        result = ConsolePrinter._format_result_content(raw)
        # Should return a Group (not raw string) since it has summary
        from rich.console import Group
        assert isinstance(result, Group)


if __name__ == "__main__":
    unittest.main()
