"""Tests for BrowserPrinter.

Tests verify correctness and accuracy of all browser streaming logic
for ClaudeCodingAgent. Uses real objects with duck-typed attributes
(SimpleNamespace) as message inputs and real queue subscribers.
"""

import json
import queue
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

from kiss.agents.coding_agents.print_to_browser import (
    _LANG_MAP,
    _MAX_RESULT_LEN,
    BrowserPrinter,
    _find_free_port,
)


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


class TestFindFreePort(unittest.TestCase):
    def test_returns_positive_int(self):
        port = _find_free_port()
        assert isinstance(port, int)
        assert port > 0

    def test_returns_different_ports(self):
        ports = {_find_free_port() for _ in range(5)}
        assert len(ports) >= 2


class TestLangForPath(unittest.TestCase):
    def test_python(self):
        assert BrowserPrinter._lang_for_path("foo.py") == "python"

    def test_javascript(self):
        assert BrowserPrinter._lang_for_path("bar.js") == "javascript"

    def test_typescript(self):
        assert BrowserPrinter._lang_for_path("/a/b/c.ts") == "typescript"

    def test_unknown_extension(self):
        assert BrowserPrinter._lang_for_path("file.xyz") == "xyz"

    def test_no_extension(self):
        assert BrowserPrinter._lang_for_path("Makefile") == "text"

    def test_all_lang_map_entries(self):
        for ext, lang in _LANG_MAP.items():
            assert BrowserPrinter._lang_for_path(f"file.{ext}") == lang


class TestBrowserPrinterInit(unittest.TestCase):
    def test_default_state(self):
        p = BrowserPrinter()
        assert p._current_block_type == ""
        assert p._tool_name == ""
        assert p._tool_json_buffer == ""
        assert p._clients == []
        assert p._port == 0

    def test_reset(self):
        p = BrowserPrinter()
        p._current_block_type = "thinking"
        p._tool_name = "Read"
        p._tool_json_buffer = '{"path": "x"}'
        p.reset()
        assert p._current_block_type == ""
        assert p._tool_name == ""
        assert p._tool_json_buffer == ""


class TestBroadcast(unittest.TestCase):
    def test_single_client(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._broadcast({"type": "test", "data": "hello"})
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "test"
        assert events[0]["data"] == "hello"

    def test_multiple_clients(self):
        p = BrowserPrinter()
        q1 = _subscribe(p)
        q2 = _subscribe(p)
        p._broadcast({"type": "test"})
        assert _drain(q1) == [{"type": "test"}]
        assert _drain(q2) == [{"type": "test"}]

    def test_no_clients(self):
        p = BrowserPrinter()
        p._broadcast({"type": "test"})


class TestPrintStreamEvent(unittest.TestCase):
    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

    def test_thinking_start(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "thinking"},
                }
            )
        )
        assert text == ""
        assert p._current_block_type == "thinking"
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "thinking_start"

    def test_thinking_delta(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
                }
            )
        )
        assert text == "Let me think..."
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "thinking_delta"
        assert events[0]["text"] == "Let me think..."

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

    def test_thinking_end(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._current_block_type = "thinking"
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "thinking_end"

    def test_text_delta(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello world"},
                }
            )
        )
        assert text == "Hello world"
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello world"

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

    def test_text_end(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._current_block_type = "text"
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "text_end"

    def test_tool_use_start(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p.print_stream_event(
            self._event(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "tool_use", "name": "Read"},
                }
            )
        )
        assert p._current_block_type == "tool_use"
        assert p._tool_name == "Read"
        assert p._tool_json_buffer == ""
        assert _drain(q) == []

    def test_input_json_delta(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._tool_json_buffer = ""
        text = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
                }
            )
        )
        assert text == ""
        assert p._tool_json_buffer == '{"path":'
        assert _drain(q) == []

    def test_input_json_delta_accumulates(self):
        p = BrowserPrinter()
        p._tool_json_buffer = '{"path":'
        p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": ' "/tmp"}'},
                }
            )
        )
        assert p._tool_json_buffer == '{"path": "/tmp"}'

    def test_tool_use_stop_valid_json(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._current_block_type = "tool_use"
        p._tool_name = "Read"
        p._tool_json_buffer = '{"path": "/tmp/test.txt"}'
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call"
        assert events[0]["name"] == "Read"
        assert events[0]["path"] == "/tmp/test.txt"

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

    def test_unknown_event_type(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print_stream_event(self._event({"type": "unknown_event"}))
        assert text == ""
        assert _drain(q) == []

    def test_empty_event_dict(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        text = p.print_stream_event(self._event({}))
        assert text == ""
        assert _drain(q) == []

    def test_content_block_start_no_content_block(self):
        p = BrowserPrinter()
        p.print_stream_event(self._event({"type": "content_block_start"}))
        assert p._current_block_type == ""

    def test_content_block_start_text(self):
        p = BrowserPrinter()
        p.print_stream_event(
            self._event(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "text"},
                }
            )
        )
        assert p._current_block_type == "text"

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

    def test_content_block_stop_empty_type(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._current_block_type = ""
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert p._current_block_type == ""
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "text_end"

    def test_full_thinking_sequence(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p.print_stream_event(
            self._event(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "thinking"},
                }
            )
        )
        t1 = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "Part 1 "},
                }
            )
        )
        t2 = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "Part 2"},
                }
            )
        )
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert t1 == "Part 1 "
        assert t2 == "Part 2"
        events = _drain(q)
        types = [e["type"] for e in events]
        assert types == ["thinking_start", "thinking_delta", "thinking_delta", "thinking_end"]

    def test_full_text_sequence(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p.print_stream_event(
            self._event(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "text"},
                }
            )
        )
        t1 = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello "},
                }
            )
        )
        t2 = p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "world!"},
                }
            )
        )
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        assert t1 == "Hello "
        assert t2 == "world!"
        events = _drain(q)
        types = [e["type"] for e in events]
        assert types == ["text_delta", "text_delta", "text_end"]

    def test_full_tool_use_sequence(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p.print_stream_event(
            self._event(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "tool_use", "name": "Write"},
                }
            )
        )
        p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
                }
            )
        )
        p.print_stream_event(
            self._event(
                {
                    "type": "content_block_delta",
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": ' "test.py", "content": "x"}',
                    },
                }
            )
        )
        p.print_stream_event(self._event({"type": "content_block_stop"}))
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call"
        assert events[0]["name"] == "Write"
        assert events[0]["path"] == "test.py"


class TestFormatToolCall(unittest.TestCase):
    def test_with_path(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Read", {"path": "/tmp/test.py"})
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["name"] == "Read"
        assert events[0]["path"] == "/tmp/test.py"
        assert events[0]["lang"] == "python"

    def test_with_file_path_key(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Read", {"file_path": "/tmp/test.rb"})
        events = _drain(q)
        assert events[0]["path"] == "/tmp/test.rb"
        assert events[0]["lang"] == "ruby"

    def test_with_command(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Bash", {"command": "ls -la"})
        events = _drain(q)
        assert events[0]["command"] == "ls -la"

    def test_with_content(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Write", {"path": "test.py", "content": "print('hi')"})
        events = _drain(q)
        assert events[0]["content"] == "print('hi')"
        assert events[0]["path"] == "test.py"

    def test_with_edit_strings(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call(
            "Edit",
            {"path": "test.py", "old_string": "old code", "new_string": "new code"},
        )
        events = _drain(q)
        assert events[0]["old_string"] == "old code"
        assert events[0]["new_string"] == "new code"

    def test_with_description(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Bash", {"description": "Run tests", "command": "pytest"})
        events = _drain(q)
        assert events[0]["description"] == "Run tests"

    def test_no_args(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("SomeTool", {})
        events = _drain(q)
        assert events[0]["name"] == "SomeTool"
        assert "path" not in events[0]
        assert "command" not in events[0]

    def test_truncates_long_extra_values(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Tool", {"extra_key": "x" * 300})
        events = _drain(q)
        extras = events[0]["extras"]
        assert "extra_key" in extras
        assert extras["extra_key"].endswith("...")
        assert len(extras["extra_key"]) <= 204

    def test_extra_key_short_value(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Tool", {"pattern": "*.py"})
        events = _drain(q)
        assert events[0]["extras"]["pattern"] == "*.py"

    def test_old_string_none_new_string_empty(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Edit", {"path": "f.py", "old_string": "a", "new_string": ""})
        events = _drain(q)
        assert events[0]["old_string"] == "a"
        assert events[0]["new_string"] == ""

    def test_no_path_sets_text_lang(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Bash", {"command": "echo hi"})
        events = _drain(q)
        assert "lang" not in events[0]


class TestPrintToolResult(unittest.TestCase):
    def test_success(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._print_tool_result("Success output", is_error=False)
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "tool_result"
        assert events[0]["content"] == "Success output"
        assert events[0]["is_error"] is False

    def test_error(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._print_tool_result("Error message", is_error=True)
        events = _drain(q)
        assert events[0]["is_error"] is True

    def test_truncation(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        long = "x" * (_MAX_RESULT_LEN * 2)
        p._print_tool_result(long, is_error=False)
        events = _drain(q)
        assert "... (truncated) ..." in events[0]["content"]

    def test_no_truncation_for_short(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        p._print_tool_result("short", is_error=False)
        events = _drain(q)
        assert "... (truncated) ..." not in events[0]["content"]


class TestPrintMessageSystem(unittest.TestCase):
    def test_tool_output(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(subtype="tool_output", data={"content": "hello output"})
        p.print_message(msg)
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "system_output"
        assert events[0]["text"] == "hello output"

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


class TestPrintMessageResult(unittest.TestCase):
    def test_result_with_stats(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(result="Final answer")
        p.print_message(msg, step_count=5, budget_used=0.0123, total_tokens_used=1500)
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "result"
        assert events[0]["text"] == "Final answer"
        assert events[0]["step_count"] == 5
        assert events[0]["total_tokens"] == 1500
        assert events[0]["cost"] == "$0.0123"

    def test_result_no_budget(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(result="Answer")
        p.print_message(msg, step_count=1, budget_used=0.0, total_tokens_used=100)
        events = _drain(q)
        assert events[0]["cost"] == "N/A"

    def test_result_none(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(result=None)
        p.print_message(msg)
        events = _drain(q)
        assert events[0]["text"] == "(no result)"

    def test_result_empty_string(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(result="")
        p.print_message(msg)
        events = _drain(q)
        assert events[0]["text"] == "(no result)"

    def test_result_defaults(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(result="done")
        p.print_message(msg)
        events = _drain(q)
        assert events[0]["step_count"] == 0
        assert events[0]["total_tokens"] == 0
        assert events[0]["cost"] == "N/A"


class TestPrintMessageUser(unittest.TestCase):
    def test_tool_result_success(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        block = SimpleNamespace(is_error=False, content="Success output")
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "tool_result"
        assert events[0]["content"] == "Success output"
        assert events[0]["is_error"] is False

    def test_tool_result_error(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        block = SimpleNamespace(is_error=True, content="Error message")
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
        events = _drain(q)
        assert events[0]["is_error"] is True

    def test_multiple_tool_results(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        blocks = [
            SimpleNamespace(is_error=False, content="Result 1"),
            SimpleNamespace(is_error=True, content="Error 2"),
        ]
        msg = SimpleNamespace(content=blocks)
        p.print_message(msg)
        events = _drain(q)
        assert len(events) == 2
        assert events[0]["content"] == "Result 1"
        assert events[1]["content"] == "Error 2"

    def test_non_string_content(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        block = SimpleNamespace(is_error=False, content=["a", "b"])
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
        events = _drain(q)
        assert len(events) == 1

    def test_blocks_without_is_error_skipped(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        block = SimpleNamespace(text="just text")
        msg = SimpleNamespace(content=[block])
        p.print_message(msg)
        assert _drain(q) == []

    def test_empty_content_list(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(content=[])
        p.print_message(msg)
        assert _drain(q) == []


class TestPrintMessageDispatch(unittest.TestCase):
    def test_unknown_message_type_no_crash(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(unknown_attr="value")
        p.print_message(msg)
        assert _drain(q) == []

    def test_dispatch_prefers_system_over_content(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(
            subtype="tool_output",
            data={"content": "system text"},
            content=[],
        )
        p.print_message(msg)
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "system_output"

    def test_dispatch_prefers_result_over_content(self):
        p = BrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(result="result text", content=[])
        p.print_message(msg)
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "result"


class TestConstants(unittest.TestCase):
    def test_lang_map_not_empty(self):
        assert len(_LANG_MAP) > 0

    def test_max_result_len_positive(self):
        assert _MAX_RESULT_LEN > 0


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


class TestBrowserPrinterEndToEnd(unittest.TestCase):
    def test_full_agent_simulation(self):
        p = BrowserPrinter()
        q = _subscribe(p)

        p.print_stream_event(
            SimpleNamespace(
                event={
                    "type": "content_block_start",
                    "content_block": {"type": "thinking"},
                }
            )
        )
        p.print_stream_event(
            SimpleNamespace(
                event={
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "Planning..."},
                }
            )
        )
        p.print_stream_event(SimpleNamespace(event={"type": "content_block_stop"}))

        p.print_stream_event(
            SimpleNamespace(
                event={
                    "type": "content_block_start",
                    "content_block": {"type": "text"},
                }
            )
        )
        p.print_stream_event(
            SimpleNamespace(
                event={
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "I'll write a file."},
                }
            )
        )
        p.print_stream_event(SimpleNamespace(event={"type": "content_block_stop"}))

        p.print_stream_event(
            SimpleNamespace(
                event={
                    "type": "content_block_start",
                    "content_block": {"type": "tool_use", "name": "Write"},
                }
            )
        )
        p.print_stream_event(
            SimpleNamespace(
                event={
                    "type": "content_block_delta",
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"path": "test.py", "content": "print(1)"}',
                    },
                }
            )
        )
        p.print_stream_event(SimpleNamespace(event={"type": "content_block_stop"}))

        tool_block = SimpleNamespace(is_error=False, content="File written")
        p.print_message(SimpleNamespace(content=[tool_block]))

        p.print_message(
            SimpleNamespace(result="Task complete"),
            step_count=3,
            budget_used=0.05,
            total_tokens_used=2000,
        )

        events = _drain(q)
        types = [e["type"] for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types
        assert "text_delta" in types
        assert "text_end" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "result" in types

        tool_call_event = next(e for e in events if e["type"] == "tool_call")
        assert tool_call_event["name"] == "Write"
        assert tool_call_event["path"] == "test.py"

        result_event = next(e for e in events if e["type"] == "result")
        assert result_event["text"] == "Task complete"
        assert result_event["step_count"] == 3
        assert result_event["cost"] == "$0.0500"

    def test_events_are_valid_json(self):
        p = BrowserPrinter()
        q = _subscribe(p)

        p.print_stream_event(
            SimpleNamespace(
                event={
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": 'He said "hello"'},
                }
            )
        )
        p._format_tool_call("Read", {"path": "file with spaces.py"})
        p._print_tool_result("Line1\nLine2\tTab", is_error=False)

        events = _drain(q)
        for event in events:
            serialized = json.dumps(event)
            parsed = json.loads(serialized)
            assert parsed == event


if __name__ == "__main__":
    unittest.main()
