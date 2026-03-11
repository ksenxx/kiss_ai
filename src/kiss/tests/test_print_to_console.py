"""Tests for ConsolePrinter.

Tests verify correctness and accuracy of all terminal printing logic.
Uses real objects with duck-typed attributes (SimpleNamespace) as
message inputs.
"""

import io
import unittest
from types import SimpleNamespace

from kiss.core.print_to_console import ConsolePrinter


class TestFormatToolCall(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf


class TestPrintToolResult(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf


class TestPrintStreamEvent(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

class TestPrintMessageSystem(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def test_other_subtype_ignored(self):
        p, buf = self._make_printer()
        msg = SimpleNamespace(subtype="other", data={"content": "should not appear"})
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


class TestTokenCallback(unittest.TestCase):
    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf


class TestStreamingFlow(unittest.TestCase):
    """Test the full streaming flow: block_start -> token_callback -> block_stop."""

    def _make_printer(self):
        buf = io.StringIO()
        return ConsolePrinter(file=buf), buf

    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)


if __name__ == "__main__":
    unittest.main()
