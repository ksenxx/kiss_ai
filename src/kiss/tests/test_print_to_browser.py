"""Tests for BaseBrowserPrinter.

Tests verify correctness and accuracy of all browser streaming logic.
Uses real objects with duck-typed attributes (SimpleNamespace) as
message inputs and real queue subscribers.
"""

import queue
import unittest
from types import SimpleNamespace

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.core.printer import MAX_RESULT_LEN as _MAX_RESULT_LEN


def _subscribe(printer: BaseBrowserPrinter) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    printer._clients.append(q)
    return q


def _drain(q: queue.Queue) -> list[dict]:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


class TestPrintStreamEvent(unittest.TestCase):
    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

    def test_tool_use_stop_invalid_json(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._current_block_type = "tool_use"
        p._tool_name = "Bash"
        p._tool_json_buffer = "invalid{"
        p.print(self._event({"type": "content_block_stop"}), type="stream_event")
        assert p._current_block_type == ""
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call"
        assert events[0]["name"] == "Bash"


class TestPrintToolResult(unittest.TestCase):
    def test_truncation(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        long = "x" * (_MAX_RESULT_LEN * 2)
        p.print(long, type="tool_result", is_error=False)
        events = _drain(q)
        assert "... (truncated) ..." in events[0]["content"]


class TestPrintMessageSystem(unittest.TestCase):
    def test_tool_output_empty(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        p.print(msg, type="message")
        assert _drain(q) == []


class TestPrintMessageDispatch(unittest.TestCase):
    def test_unknown_message_type_no_crash(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(unknown_attr="value")
        p.print(msg, type="message")
        assert _drain(q) == []


class TestStreamingFlow(unittest.TestCase):
    """Test the full streaming flow: block_start -> token_callback -> block_stop."""

    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

    def test_thinking_block_flow(self):
        import asyncio

        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print(
            self._event(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "thinking"},
                }
            ),
            type="stream_event",
        )
        assert p._current_block_type == "thinking"
        events = _drain(q)
        assert any(e["type"] == "thinking_start" for e in events)

        text = p.print(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "hmm"},
                }
            ),
            type="stream_event",
        )
        assert text == "hmm"
        assert _drain(q) == []

        asyncio.run(p.token_callback("hmm"))
        events = _drain(q)
        assert len(events) == 1
        assert events[0] == {"type": "thinking_delta", "text": "hmm"}

        p.print(self._event({"type": "content_block_stop"}), type="stream_event")
        assert p._current_block_type == ""
        events = _drain(q)
        assert any(e["type"] == "thinking_end" for e in events)

    def test_no_double_broadcast(self):
        import asyncio

        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print(
            self._event(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "text"},
                }
            ),
            type="stream_event",
        )
        _drain(q)
        p.print(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "unique_token"},
                }
            ),
            type="stream_event",
        )
        asyncio.run(p.token_callback("unique_token"))
        events = _drain(q)
        text_events = [e for e in events if e.get("text") == "unique_token"]
        assert len(text_events) == 1

        # Also verify block_stop resets state for text blocks
        p.print(self._event({"type": "content_block_stop"}), type="stream_event")
        assert p._current_block_type == ""


if __name__ == "__main__":
    unittest.main()
