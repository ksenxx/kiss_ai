"""Tests for BaseBrowserPrinter.

Tests verify correctness and accuracy of all browser streaming logic.
Uses real objects with duck-typed attributes (SimpleNamespace) as
message inputs and real queue subscribers.
"""

import asyncio
import queue
import threading
import time
import unittest
from types import SimpleNamespace

from kiss.agents.sorcar.browser_ui import (
    _DISPLAY_EVENT_TYPES,
    BaseBrowserPrinter,
    _coalesce_events,
    find_free_port,
)


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


# ---------------------------------------------------------------------------
# find_free_port
# ---------------------------------------------------------------------------

class TestFindFreePort(unittest.TestCase):
    def test_returns_positive_int(self):
        port = find_free_port()
        assert isinstance(port, int) and port > 0

    def test_returns_different_ports(self):
        ports = {find_free_port() for _ in range(5)}
        # At least 2 different ports over 5 calls
        assert len(ports) >= 2


# ---------------------------------------------------------------------------
# _coalesce_events
# ---------------------------------------------------------------------------

class TestCoalesceEvents(unittest.TestCase):
    def test_empty_list(self):
        assert _coalesce_events([]) == []

    def test_single_event(self):
        events = [{"type": "tool_call", "name": "Bash"}]
        assert _coalesce_events(events) == events

    def test_merges_thinking_delta(self):
        events = [
            {"type": "thinking_delta", "text": "A"},
            {"type": "thinking_delta", "text": "B"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0]["text"] == "AB"

    def test_merges_text_delta(self):
        events = [
            {"type": "text_delta", "text": "Hello "},
            {"type": "text_delta", "text": "world"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"

    def test_merges_system_output(self):
        events = [
            {"type": "system_output", "text": "line1\n"},
            {"type": "system_output", "text": "line2\n"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0]["text"] == "line1\nline2\n"

    def test_no_merge_different_types(self):
        events = [
            {"type": "thinking_delta", "text": "A"},
            {"type": "text_delta", "text": "B"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

    def test_no_merge_non_delta_types(self):
        events = [
            {"type": "tool_call", "name": "A"},
            {"type": "tool_call", "name": "B"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

    def test_no_merge_missing_text_in_current(self):
        events = [
            {"type": "thinking_delta", "text": "A"},
            {"type": "thinking_delta"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

    def test_no_merge_missing_text_in_previous(self):
        events = [
            {"type": "thinking_delta"},
            {"type": "thinking_delta", "text": "B"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

    def test_preserves_extra_keys_in_merged(self):
        events = [
            {"type": "text_delta", "text": "A", "id": 1},
            {"type": "text_delta", "text": "B"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0]["text"] == "AB"
        assert result[0]["id"] == 1

    def test_interleaved_merge_non_merge(self):
        events = [
            {"type": "text_delta", "text": "a"},
            {"type": "text_delta", "text": "b"},
            {"type": "tool_call", "name": "X"},
            {"type": "text_delta", "text": "c"},
            {"type": "text_delta", "text": "d"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 3
        assert result[0]["text"] == "ab"
        assert result[1]["name"] == "X"
        assert result[2]["text"] == "cd"


# ---------------------------------------------------------------------------
# BaseBrowserPrinter init/reset
# ---------------------------------------------------------------------------

class TestBaseBrowserPrinterInit(unittest.TestCase):
    def test_init_state(self):
        p = BaseBrowserPrinter()
        assert p._clients == []
        assert p._recordings == {}
        assert p._bash_buffer == []
        assert p._bash_flush_timer is None
        assert not p.stop_event.is_set()

    def test_reset_clears_state(self):
        p = BaseBrowserPrinter()
        # Set up some state
        p._bash_buffer.append("data")
        p._bash_flush_timer = threading.Timer(10, lambda: None)
        p._bash_flush_timer.start()
        p.reset()
        assert p._bash_buffer == []
        assert p._bash_flush_timer is None

    def test_reset_without_timer(self):
        p = BaseBrowserPrinter()
        p._bash_buffer.append("data")
        p.reset()
        assert p._bash_buffer == []


# ---------------------------------------------------------------------------
# flush_bash
# ---------------------------------------------------------------------------

class TestFlushBash(unittest.TestCase):
    def test_flush_with_data(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._bash_buffer.append("output line")
        p._flush_bash()
        events = _drain(q)
        assert len(events) == 1
        assert events[0] == {"type": "system_output", "text": "output line"}

    def test_flush_empty_buffer(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._flush_bash()
        assert _drain(q) == []

    def test_flush_cancels_timer(self):
        p = BaseBrowserPrinter()
        p._bash_flush_timer = threading.Timer(10, lambda: None)
        p._bash_flush_timer.start()
        p._bash_buffer.append("x")
        q = _subscribe(p)
        p._flush_bash()
        assert p._bash_flush_timer is None
        events = _drain(q)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class TestRecording(unittest.TestCase):
    def test_start_stop_recording(self):
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "hello"})
        p.broadcast({"type": "tool_call", "name": "Bash"})
        events = p.stop_recording()
        assert len(events) == 2

    def test_stop_without_start_returns_empty(self):
        p = BaseBrowserPrinter()
        events = p.stop_recording()
        assert events == []

    def test_recording_filters_display_events(self):
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "yes"})
        p.broadcast({"type": "unknown_type", "text": "no"})
        events = p.stop_recording()
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"

    def test_recording_coalesces_deltas(self):
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "a"})
        p.broadcast({"type": "text_delta", "text": "b"})
        events = p.stop_recording()
        assert len(events) == 1
        assert events[0]["text"] == "ab"

    def test_independent_thread_recordings(self):
        p = BaseBrowserPrinter()
        results: dict[str, list] = {}

        def worker(name: str) -> None:
            p.start_recording()
            p.broadcast({"type": "text_delta", "text": name})
            results[name] = p.stop_recording()

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t1.join()
        t2.start()
        t2.join()
        # Each thread got its own event
        assert len(results["A"]) == 1
        assert len(results["B"]) == 1


# ---------------------------------------------------------------------------
# Broadcast / Client management
# ---------------------------------------------------------------------------

class TestBroadcast(unittest.TestCase):
    def test_broadcast_to_clients(self):
        p = BaseBrowserPrinter()
        q1 = _subscribe(p)
        q2 = _subscribe(p)
        p.broadcast({"type": "text_delta", "text": "hi"})
        assert _drain(q1) == [{"type": "text_delta", "text": "hi"}]
        assert _drain(q2) == [{"type": "text_delta", "text": "hi"}]

    def test_add_client(self):
        p = BaseBrowserPrinter()
        cq = p.add_client()
        p.broadcast({"type": "text_delta", "text": "test"})
        events = _drain(cq)
        assert len(events) == 1

    def test_remove_client(self):
        p = BaseBrowserPrinter()
        cq = p.add_client()
        p.remove_client(cq)
        p.broadcast({"type": "text_delta", "text": "test"})
        assert _drain(cq) == []

    def test_remove_unknown_client(self):
        p = BaseBrowserPrinter()
        unknown_q: queue.Queue = queue.Queue()
        # Should not crash
        p.remove_client(unknown_q)

    def test_has_clients(self):
        p = BaseBrowserPrinter()
        assert not p.has_clients()
        cq = p.add_client()
        assert p.has_clients()
        p.remove_client(cq)
        assert not p.has_clients()


# ---------------------------------------------------------------------------
# _broadcast_result
# ---------------------------------------------------------------------------

class TestBroadcastResult(unittest.TestCase):
    def test_plain_text_result(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._broadcast_result("some output", step_count=5, total_tokens=100, cost="$0.01")
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "result"
        assert events[0]["text"] == "some output"
        assert events[0]["step_count"] == 5

    def test_empty_result(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._broadcast_result("")
        events = _drain(q)
        assert events[0]["text"] == "(no result)"

    def test_yaml_result_with_summary(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        import yaml
        text = yaml.dump({"success": True, "summary": "Done well"})
        p._broadcast_result(text)
        events = _drain(q)
        assert events[0]["success"] is True
        assert events[0]["summary"] == "Done well"

    def test_yaml_result_failure(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        import yaml
        text = yaml.dump({"success": False, "summary": "Failed"})
        p._broadcast_result(text)
        events = _drain(q)
        assert events[0]["success"] is False

    def test_non_yaml_result(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._broadcast_result("just plain text")
        events = _drain(q)
        # No "summary" key if not parseable as result yaml
        assert "summary" not in events[0]


# ---------------------------------------------------------------------------
# _check_stop
# ---------------------------------------------------------------------------

class TestCheckStop(unittest.TestCase):
    def test_no_stop_no_raise(self):
        p = BaseBrowserPrinter()
        p._check_stop()  # should not raise

    def test_global_stop_event(self):
        p = BaseBrowserPrinter()
        p.stop_event.set()
        with self.assertRaises(KeyboardInterrupt):
            p._check_stop()

    def test_thread_local_stop_event(self):
        p = BaseBrowserPrinter()
        local_ev = threading.Event()
        p._thread_local.stop_event = local_ev
        p._check_stop()  # not set yet
        local_ev.set()
        with self.assertRaises(KeyboardInterrupt):
            p._check_stop()

    def test_thread_local_none_falls_through(self):
        p = BaseBrowserPrinter()
        p._thread_local.stop_event = None
        p.stop_event.set()
        with self.assertRaises(KeyboardInterrupt):
            p._check_stop()


# ---------------------------------------------------------------------------
# print() dispatcher
# ---------------------------------------------------------------------------

class TestPrintText(unittest.TestCase):
    def test_text_broadcasts_delta(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("Hello world", type="text")
        events = _drain(q)
        assert any(e["type"] == "text_delta" for e in events)

    def test_empty_text_no_broadcast(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("", type="text")
        events = _drain(q)
        assert not any(e.get("type") == "text_delta" for e in events)


class TestPrintSystemPrompt(unittest.TestCase):
    def test_system_prompt_broadcasts_event(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("You are a helpful agent.", type="system_prompt")
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "system_prompt"
        assert events[0]["text"] == "You are a helpful agent."


class TestPrintPrompt(unittest.TestCase):
    def test_prompt_broadcasts_event(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("Do the task.", type="prompt")
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "prompt"
        assert events[0]["text"] == "Do the task."


class TestPrintUsageInfo(unittest.TestCase):
    def test_usage_info_broadcasts(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("  Steps: 5, Tokens: 1000  ", type="usage_info")
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "usage_info"
        assert events[0]["text"] == "Steps: 5, Tokens: 1000"


class TestPrintBashStream(unittest.TestCase):
    def test_bash_stream_buffers_and_flushes(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        # Force immediate flush by setting last_flush to distant past
        p._bash_last_flush = 0.0
        p.print("line1\n", type="bash_stream")
        events = _drain(q)
        assert any(e["type"] == "system_output" for e in events)

    def test_bash_stream_schedules_timer(self):
        p = BaseBrowserPrinter()
        # Set recent flush time so timer path is taken
        p._bash_last_flush = time.monotonic()
        p.print("x", type="bash_stream")
        assert p._bash_flush_timer is not None
        # Clean up
        p._bash_flush_timer.cancel()

    def test_bash_stream_already_has_timer(self):
        p = BaseBrowserPrinter()
        p._bash_last_flush = time.monotonic()
        # First call schedules timer
        p.print("x", type="bash_stream")
        timer1 = p._bash_flush_timer
        # Second call reuses existing timer
        p.print("y", type="bash_stream")
        timer2 = p._bash_flush_timer
        assert timer1 is timer2
        assert timer1 is not None
        timer1.cancel()


class TestPrintToolCall(unittest.TestCase):
    def test_tool_call_broadcasts(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("Bash", type="tool_call", tool_input={"command": "ls"})
        events = _drain(q)
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "Bash"
        assert tool_calls[0]["command"] == "ls"

    def test_tool_call_flushes_bash_first(self):
        p = BaseBrowserPrinter()
        p._bash_buffer.append("pending output")
        q = _subscribe(p)
        p.print("Bash", type="tool_call", tool_input={})
        events = _drain(q)
        types = [e["type"] for e in events]
        # Should have system_output before tool_call
        assert "system_output" in types
        tc_idx = types.index("tool_call")
        so_idx = types.index("system_output")
        assert so_idx < tc_idx


class TestPrintToolResult(unittest.TestCase):
    def test_tool_result_success(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("output data", type="tool_result")
        events = _drain(q)
        results = [e for e in events if e["type"] == "tool_result"]
        assert len(results) == 1
        assert results[0]["is_error"] is False

    def test_tool_result_error(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("error msg", type="tool_result", is_error=True)
        events = _drain(q)
        results = [e for e in events if e["type"] == "tool_result"]
        assert results[0]["is_error"] is True

    def test_tool_result_flushes_bash(self):
        p = BaseBrowserPrinter()
        p._bash_buffer.append("pending")
        q = _subscribe(p)
        p.print("done", type="tool_result")
        events = _drain(q)
        types = [e["type"] for e in events]
        assert "system_output" in types


class TestPrintResult(unittest.TestCase):
    def test_result_with_cost(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("final output", type="result", step_count=10,
                total_tokens=500, cost="$0.05")
        events = _drain(q)
        results = [e for e in events if e["type"] == "result"]
        assert len(results) == 1
        assert results[0]["cost"] == "$0.05"
        assert results[0]["step_count"] == 10

    def test_result_sends_text_end_first(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("done", type="result")
        events = _drain(q)
        types = [e["type"] for e in events]
        assert types[0] == "text_end"


class TestPrintUnknownType(unittest.TestCase):
    def test_unknown_type_returns_empty(self):
        p = BaseBrowserPrinter()
        result = p.print("data", type="unknown_type_xyz")
        assert result == ""


class TestPrintStreamEvent(unittest.TestCase):
    def test_stream_event_delegates(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        # Create a content_block_start event for thinking
        event = SimpleNamespace(event={
            "type": "content_block_start",
            "content_block": {"type": "thinking"},
        })
        p.print(event, type="stream_event")
        events = _drain(q)
        assert any(e["type"] == "thinking_start" for e in events)


# ---------------------------------------------------------------------------
# token_callback
# ---------------------------------------------------------------------------

class TestTokenCallback(unittest.TestCase):
    def test_text_token(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        asyncio.get_event_loop().run_until_complete(p.token_callback("hello"))
        events = _drain(q)
        assert events[0] == {"type": "text_delta", "text": "hello"}

    def test_thinking_token(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._current_block_type = "thinking"
        asyncio.get_event_loop().run_until_complete(p.token_callback("thought"))
        events = _drain(q)
        assert events[0] == {"type": "thinking_delta", "text": "thought"}

    def test_empty_token(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        asyncio.get_event_loop().run_until_complete(p.token_callback(""))
        assert _drain(q) == []

    def test_stop_check_in_callback(self):
        p = BaseBrowserPrinter()
        p.stop_event.set()
        with self.assertRaises(KeyboardInterrupt):
            asyncio.get_event_loop().run_until_complete(p.token_callback("x"))


# ---------------------------------------------------------------------------
# _format_tool_call
# ---------------------------------------------------------------------------

class TestFormatToolCall(unittest.TestCase):
    def test_with_file_path(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Write", {"file_path": "/a/b.py", "content": "x=1"})
        events = _drain(q)
        tc = events[0]
        assert tc["path"] == "/a/b.py"
        assert tc["content"] == "x=1"

    def test_with_description(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Bash", {"command": "ls", "description": "list files"})
        events = _drain(q)
        tc = events[0]
        assert tc["description"] == "list files"
        assert tc["command"] == "ls"

    def test_with_old_new_strings(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Edit", {
            "file_path": "f.py",
            "old_string": "old",
            "new_string": "new",
        })
        events = _drain(q)
        tc = events[0]
        assert tc["old_string"] == "old"
        assert tc["new_string"] == "new"

    def test_with_only_old_string(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Edit", {"old_string": "old"})
        events = _drain(q)
        tc = events[0]
        assert tc["old_string"] == "old"
        assert "new_string" not in tc

    def test_with_only_new_string(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Edit", {"new_string": "new"})
        events = _drain(q)
        tc = events[0]
        assert "old_string" not in tc
        assert tc["new_string"] == "new"

    def test_minimal_tool_call(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("finish", {})
        events = _drain(q)
        tc = events[0]
        assert tc["name"] == "finish"
        assert "path" not in tc
        assert "command" not in tc

    def test_with_extras(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Bash", {
            "command": "ls",
            "timeout_seconds": "30",
        })
        events = _drain(q)
        tc = events[0]
        assert "extras" in tc

    def test_with_path_key(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._format_tool_call("Read", {"path": "/tmp/x.txt"})
        events = _drain(q)
        tc = events[0]
        assert tc["path"] == "/tmp/x.txt"


# ---------------------------------------------------------------------------
# Stream event callbacks
# ---------------------------------------------------------------------------

class TestStreamCallbacks(unittest.TestCase):
    def test_on_thinking_start(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._on_thinking_start()
        events = _drain(q)
        assert events[0]["type"] == "thinking_start"

    def test_on_thinking_end(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._on_thinking_end()
        events = _drain(q)
        assert events[0]["type"] == "thinking_end"

    def test_on_tool_use_end(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._on_tool_use_end("Bash", {"command": "pwd"})
        events = _drain(q)
        assert events[0]["type"] == "tool_call"
        assert events[0]["name"] == "Bash"

    def test_on_text_block_end(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._on_text_block_end()
        events = _drain(q)
        assert events[0]["type"] == "text_end"


# ---------------------------------------------------------------------------
# _handle_message
# ---------------------------------------------------------------------------

class TestHandleMessage(unittest.TestCase):
    def test_tool_output_message(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(
            subtype="tool_output",
            data={"content": "tool result text"},
        )
        p.print(msg, type="message")
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "system_output"
        assert events[0]["text"] == "tool result text"

    def test_tool_output_empty(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        p.print(msg, type="message")
        assert _drain(q) == []

    def test_result_message(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(result="All done")
        p.print(msg, type="message", budget_used=0.05, step_count=3,
                total_tokens_used=200)
        events = _drain(q)
        results = [e for e in events if e["type"] == "result"]
        assert len(results) == 1
        assert results[0]["cost"] == "$0.0500"

    def test_result_message_no_budget(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(result="Done")
        p.print(msg, type="message")
        events = _drain(q)
        results = [e for e in events if e["type"] == "result"]
        assert results[0]["cost"] == "N/A"

    def test_content_blocks_message(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        block = SimpleNamespace(is_error=False, content="success result")
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "tool_result"
        assert events[0]["is_error"] is False

    def test_content_blocks_error(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        block = SimpleNamespace(is_error=True, content="error!")
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        events = _drain(q)
        assert events[0]["is_error"] is True

    def test_subtype_not_tool_output(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(subtype="other", data={"content": "x"})
        p.print(msg, type="message")
        assert _drain(q) == []

    def test_empty_content_blocks(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(content=[])
        p.print(msg, type="message")
        assert _drain(q) == []

    def test_content_block_without_is_error(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        block = SimpleNamespace(text="just text, no is_error attr")
        msg = SimpleNamespace(content=[block])
        p.print(msg, type="message")
        assert _drain(q) == []

    def test_unknown_message_type_no_crash(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(unknown_attr="value")
        p.print(msg, type="message")
        assert _drain(q) == []


# ---------------------------------------------------------------------------
# _DISPLAY_EVENT_TYPES completeness
# ---------------------------------------------------------------------------

class TestDisplayEventTypes(unittest.TestCase):
    def test_expected_types_present(self):
        expected = {
            "clear", "thinking_start", "thinking_delta", "thinking_end",
            "text_delta", "text_end", "tool_call", "tool_result",
            "system_output", "result", "system_prompt", "prompt",
            "usage_info", "task_done", "task_error", "task_stopped",
            "followup_suggestion",
        }
        assert _DISPLAY_EVENT_TYPES == expected


# ---------------------------------------------------------------------------
# print() check_stop integration
# ---------------------------------------------------------------------------

class TestPrintCheckStop(unittest.TestCase):
    def test_print_raises_on_stop(self):
        p = BaseBrowserPrinter()
        p.stop_event.set()
        with self.assertRaises(KeyboardInterrupt):
            p.print("anything", type="text")


if __name__ == "__main__":
    unittest.main()
