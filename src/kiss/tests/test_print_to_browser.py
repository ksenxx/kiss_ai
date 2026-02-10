"""Tests for BrowserPrinter.

Tests verify correctness and accuracy of all browser streaming logic
for ClaudeCodingAgent. Uses real objects with duck-typed attributes
(SimpleNamespace) as message inputs and real queue subscribers.
"""

import queue
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

from kiss.agents.coding_agents.print_to_browser import BrowserPrinter
from kiss.agents.coding_agents.printer_common import MAX_RESULT_LEN as _MAX_RESULT_LEN


def _subscribe(printer: BrowserPrinter) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    with printer._clients_lock:
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


class TestBrowserPrinterInit(unittest.TestCase):
    def test_reset(self):
        p = BrowserPrinter()
        p._current_block_type = "thinking"
        p._tool_name = "Read"
        p._tool_json_buffer = '{"path": "x"}'
        p.reset()
        assert p._current_block_type == ""
        assert p._tool_name == ""
        assert p._tool_json_buffer == ""


class TestPrintStreamEvent(unittest.TestCase):
    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

    def test_text_delta_empty(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": ""},
                }
            )
        )
        assert text == ""
        assert _drain(q) == []

    def test_thinking_delta_empty(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": ""},
                }
            )
        )
        assert text == ""
        assert _drain(q) == []

    def test_tool_use_stop_invalid_json(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._current_block_type = "tool_use"
        p._tool_name = "Bash"
        p._tool_json_buffer = "invalid{"
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call"
        assert events[0]["name"] == "Bash"

    def test_unknown_delta_type(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "unknown_type"},
                }
            )
        )
        assert text == ""
        assert _drain(q) == []


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
        p._print_tool_result(long, is_error=False)
        events = _drain(q)
        assert "... (truncated) ..." in events[0]["content"]


class TestPrintMessageSystem(unittest.TestCase):
    def test_tool_output_empty(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        p.print_message(msg)
        assert _drain(q) == []

    def test_other_subtype_ignored(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(subtype="other", data={"content": "ignored"})
        p.print_message(msg)
        assert _drain(q) == []


class TestPrintMessageUser(unittest.TestCase):
    def test_blocks_without_is_error_skipped(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        block = SimpleNamespace(text="just text")
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
        assert _drain(q) == []


class TestPrintMessageDispatch(unittest.TestCase):
    def test_unknown_message_type_no_crash(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(unknown_attr="value")
        p.print_message(msg)
        assert _drain(q) == []


class TestServerLifecycle(unittest.TestCase):
    def test_start_and_stop(self):
        p = BrowserPrinter()
        p.start(open_browser=False)
        try:
            assert p._port > 0
            assert p._server is not None
            assert p._server_thread is not None
            assert p._server_thread.is_alive()

            url = f"http://127.0.0.1:{p._port}/"
            resp = urllib.request.urlopen(url, timeout=5)
            html = resp.read().decode()
            assert "KISS Agent" in html
            assert "EventSource" in html
            assert resp.status == 200
        finally:
            p.stop()
            time.sleep(0.5)

    def test_404_for_unknown_path(self):
        p = BrowserPrinter()
        p.start(open_browser=False)
        try:
            url = f"http://127.0.0.1:{p._port}/nonexistent"
            try:
                urllib.request.urlopen(url, timeout=5)
                assert False, "Expected HTTP error"
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            p.stop()
            time.sleep(0.5)

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


if __name__ == "__main__":
    unittest.main()
