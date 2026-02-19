"""Tests for BrowserPrinter.

Tests verify correctness and accuracy of all browser streaming logic
for ClaudeCodingAgent. Uses real objects with duck-typed attributes
(SimpleNamespace) as message inputs and real queue subscribers.
"""

import queue
import time
import unittest
from types import SimpleNamespace

from kiss.core.print_to_browser import BrowserPrinter
from kiss.core.printer import MAX_RESULT_LEN as _MAX_RESULT_LEN


def _subscribe(printer: BrowserPrinter) -> queue.Queue:
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

    def test_text_delta_empty(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": ""},
                }
            ),
            type="stream_event",
        )
        assert text == ""
        assert _drain(q) == []

    def test_text_delta_nonempty_does_not_broadcast(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello"},
                }
            ),
            type="stream_event",
        )
        assert text == "Hello"
        assert _drain(q) == []

    def test_thinking_delta_does_not_broadcast(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "Let me think"},
                }
            ),
            type="stream_event",
        )
        assert text == "Let me think"
        assert _drain(q) == []

    def test_tool_use_stop_invalid_json(self):
        p = BrowserPrinter()
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


class TestFormatToolCall(unittest.TestCase):
    def test_truncates_long_extra_values(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Tool", {"extra_key": "x" * 300})
        events = _drain(q)
        extras = events[0]["extras"]
        assert "extra_key" in extras
        assert extras["extra_key"].endswith("...")
        assert len(extras["extra_key"]) <= 204

    def test_with_description(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Bash", {"description": "Run tests", "command": "pytest"})
        events = _drain(q)
        assert events[0]["description"] == "Run tests"


class TestPrintToolResult(unittest.TestCase):
    def test_truncation(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        long = "x" * (_MAX_RESULT_LEN * 2)
        p.print(long, type="tool_result", is_error=False)
        events = _drain(q)
        assert "... (truncated) ..." in events[0]["content"]


class TestPrintMessageSystem(unittest.TestCase):
    def test_tool_output_empty(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        p.print(msg, type="message")
        assert _drain(q) == []


class TestPrintMessageUser(unittest.TestCase):
    def test_blocks_without_is_error_skipped(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        block = SimpleNamespace(text="just text")
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        assert _drain(q) == []


class TestPrintMessageDispatch(unittest.TestCase):
    def test_unknown_message_type_no_crash(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(unknown_attr="value")
        p.print(msg, type="message")
        assert _drain(q) == []


class TestServerLifecycle(unittest.TestCase):
    def test_events_endpoint_returns_sse_content_type(self):
        p = BrowserPrinter()
        p.start(open_browser=False)
        try:
            import http.client

            conn = http.client.HTTPConnection("127.0.0.1", p._port, timeout=3)
            conn.request("GET", "/events")
            resp = conn.getresponse()
            content_type = resp.getheader("content-type")
            assert content_type is not None and "text/event-stream" in content_type
            conn.close()
        finally:
            p.stop()
            time.sleep(0.5)

    def test_stop_broadcasts_done(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p.stop()
        events = _drain(q)
        assert any(e["type"] == "done" for e in events)


class TestPrint(unittest.TestCase):
    def test_print_broadcasts_text_delta(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p.print("hello world")
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert "hello world" in events[0]["text"]

    def test_print_empty_no_broadcast(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p.print("")
        assert _drain(q) == []


class TestTokenCallback(unittest.TestCase):
    def test_token_callback_broadcasts_text_delta(self):
        import asyncio
        p = BrowserPrinter()
        q = _subscribe(p)
        asyncio.run(p.token_callback("hello"))
        events = _drain(q)
        assert len(events) == 1
        assert events[0] == {"type": "text_delta", "text": "hello"}

    def test_token_callback_empty_string_no_broadcast(self):
        import asyncio
        p = BrowserPrinter()
        q = _subscribe(p)
        asyncio.run(p.token_callback(""))
        assert _drain(q) == []

    def test_token_callback_during_thinking_broadcasts_thinking_delta(self):
        import asyncio
        p = BrowserPrinter()
        p._current_block_type = "thinking"
        q = _subscribe(p)
        asyncio.run(p.token_callback("deep thought"))
        events = _drain(q)
        assert len(events) == 1
        assert events[0] == {"type": "thinking_delta", "text": "deep thought"}

    def test_token_callback_during_text_broadcasts_text_delta(self):
        import asyncio
        p = BrowserPrinter()
        p._current_block_type = "text"
        q = _subscribe(p)
        asyncio.run(p.token_callback("regular"))
        events = _drain(q)
        assert len(events) == 1
        assert events[0] == {"type": "text_delta", "text": "regular"}


class TestStreamingFlow(unittest.TestCase):
    """Test the full streaming flow: block_start -> token_callback -> block_stop."""

    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

    def test_thinking_block_flow(self):
        import asyncio
        p = BrowserPrinter()
        q = _subscribe(p)
        p.print(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "thinking"},
            }),
            type="stream_event",
        )
        assert p._current_block_type == "thinking"
        events = _drain(q)
        assert any(e["type"] == "thinking_start" for e in events)

        text = p.print(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "hmm"},
            }),
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
        p = BrowserPrinter()
        q = _subscribe(p)
        p.print(
            self._event({
                "type": "content_block_start",
                "content_block": {"type": "text"},
            }),
            type="stream_event",
        )
        _drain(q)
        p.print(
            self._event({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "unique_token"},
            }),
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
