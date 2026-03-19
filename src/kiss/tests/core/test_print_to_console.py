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

        asyncio.run(p.token_callback("I think"))

        p.print(self._event({"type": "content_block_stop"}), type="stream_event")

        out = buf.getvalue()
        assert "Thinking" in out
        assert "I think" in out

if __name__ == "__main__":
    unittest.main()
