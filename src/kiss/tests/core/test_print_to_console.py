"""Tests for ConsolePrinter.

Tests verify correctness and accuracy of all terminal printing logic.
Uses real objects with duck-typed attributes (SimpleNamespace) as
message inputs.
"""

import io
import unittest
from types import SimpleNamespace

from kiss.core.print_to_console import ConsolePrinter


def _make() -> tuple[ConsolePrinter, io.StringIO]:
    buf = io.StringIO()
    return ConsolePrinter(file=buf), buf


class TestPrintStreamEvent(unittest.TestCase):
    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)


    def test_unknown_delta_type(self):
        p, buf = _make()
        p._current_block_type = "text"
        result = p.print(self._event({
            "type": "content_block_delta",
            "delta": {"type": "unknown_delta"},
        }), type="stream_event")
        assert result == ""


class TestPrintMessageSystem(unittest.TestCase):
    def test_other_subtype_ignored(self):
        p, buf = _make()
        msg = SimpleNamespace(subtype="other", data={"content": "should not appear"})
        p.print(msg, type="message")
        assert buf.getvalue() == ""


class TestPrintMessageUser(unittest.TestCase):
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


class TestStreamingFlow(unittest.TestCase):
    """Test the full streaming flow: block_start -> token_callback -> block_stop."""

    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

    def test_full_thinking_flow(self):
        p, buf = _make()
        p.print(self._event({
            "type": "content_block_start",
            "content_block": {"type": "thinking"},
        }), type="stream_event")

        p.token_callback("I think")

        p.print(self._event({"type": "content_block_stop"}), type="stream_event")

        out = buf.getvalue()
        assert "Thinking" in out
        assert "I think" in out

class TestBashStreamDedup(unittest.TestCase):
    """Test that tool_result doesn't repeat bash_stream output."""

    def test_tool_result_skips_content_after_bash_stream(self):
        p, buf = _make()
        p.print("line1\n", type="bash_stream")
        p.print("line2\n", type="bash_stream")
        buf_before = buf.getvalue()
        assert "line1" in buf_before
        # tool_result should not repeat the content
        p.print("line1\nline2\n", type="tool_result")
        buf_after = buf.getvalue()
        # The content after tool_result should only have OK rules, not repeated lines
        new_output = buf_after[len(buf_before):]
        assert "OK" in new_output
        assert "line1" not in new_output
        assert "line2" not in new_output

    def test_tool_result_shows_content_without_bash_stream(self):
        p, buf = _make()
        p.print("some output", type="tool_result")
        out = buf.getvalue()
        assert "some output" in out
        assert "OK" in out

    def test_tool_call_resets_bash_streamed(self):
        p, buf = _make()
        p.print("streamed\n", type="bash_stream")
        # tool_call resets the flag
        p.print("Read", type="tool_call", tool_input={"file_path": "/tmp/x"})
        buf.truncate(0)
        buf.seek(0)
        # Next tool_result should show content
        p.print("file contents", type="tool_result")
        out = buf.getvalue()
        assert "file contents" in out

    def test_reset_clears_bash_streamed(self):
        p, buf = _make()
        p.print("streamed\n", type="bash_stream")
        assert p._bash_streamed is True
        p.reset()
        assert p._bash_streamed is False


if __name__ == "__main__":
    unittest.main()
