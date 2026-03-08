"""Integration tests for chat history event recording and replay.

Tests the full flow: recording events via BaseBrowserPrinter, storing them
in task_history.json, and retrieving them for replay. No mocks or patches.
"""

import asyncio
import json
import queue
import tempfile
import threading
from pathlib import Path

import pytest

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar.browser_ui import (
    _DISPLAY_EVENT_TYPES,
    BaseBrowserPrinter,
    _coalesce_events,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _use_temp_history():
    """Redirect HISTORY_FILE to a temp file."""
    original = th.HISTORY_FILE
    tmp = Path(tempfile.mktemp(suffix=".json"))
    th.HISTORY_FILE = tmp
    th._history_cache = None
    return original, tmp


def _restore_history(original: Path, tmp: Path) -> None:
    th.HISTORY_FILE = original
    th._history_cache = None
    if tmp.exists():
        tmp.unlink()


def _subscribe(printer: BaseBrowserPrinter) -> queue.Queue:
    return printer.add_client()


def _drain(q: queue.Queue) -> list[dict]:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


# ── _coalesce_events tests ──────────────────────────────────────────────


class TestCoalesceEvents:
    def test_empty_list(self) -> None:
        assert _coalesce_events([]) == []

    def test_single_event_unchanged(self) -> None:
        events = [{"type": "tool_call", "name": "Bash"}]
        assert _coalesce_events(events) == events

    def test_merge_consecutive_text_delta(self) -> None:
        events = [
            {"type": "text_delta", "text": "hel"},
            {"type": "text_delta", "text": "lo"},
            {"type": "text_delta", "text": " world"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0] == {"type": "text_delta", "text": "hello world"}

    def test_merge_consecutive_thinking_delta(self) -> None:
        events = [
            {"type": "thinking_delta", "text": "Let me "},
            {"type": "thinking_delta", "text": "think..."},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0]["text"] == "Let me think..."

    def test_merge_consecutive_system_output(self) -> None:
        events = [
            {"type": "system_output", "text": "line1\n"},
            {"type": "system_output", "text": "line2\n"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0]["text"] == "line1\nline2\n"

    def test_no_merge_different_types(self) -> None:
        events = [
            {"type": "text_delta", "text": "a"},
            {"type": "thinking_delta", "text": "b"},
            {"type": "text_delta", "text": "c"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 3

    def test_no_merge_tool_call_events(self) -> None:
        events = [
            {"type": "tool_call", "name": "A"},
            {"type": "tool_call", "name": "B"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

    def test_merge_interspersed(self) -> None:
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

    def test_no_merge_without_text_field(self) -> None:
        events = [
            {"type": "text_delta"},
            {"type": "text_delta", "text": "a"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

    def test_no_merge_non_mergeable_type_with_text(self) -> None:
        events = [
            {"type": "tool_result", "text": "a"},
            {"type": "tool_result", "text": "b"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2


# ── Recording tests ─────────────────────────────────────────────────────


class TestPrinterRecording:
    def test_start_stop_empty(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        events = p.stop_recording()
        assert events == []

    def test_records_broadcast_events(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "hello"})
        p.broadcast({"type": "tool_call", "name": "Bash"})
        events = p.stop_recording()
        assert len(events) == 2
        assert events[0]["type"] == "text_delta"
        assert events[1]["type"] == "tool_call"

    def test_recording_filters_non_display_events(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "tasks_updated"})
        p.broadcast({"type": "text_delta", "text": "hello"})
        p.broadcast({"type": "proposed_updated"})
        p.broadcast({"type": "theme_changed", "bg": "#000"})
        p.broadcast({"type": "focus_chatbox"})
        events = p.stop_recording()
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"

    def test_recording_coalesces_deltas(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "hel"})
        p.broadcast({"type": "text_delta", "text": "lo"})
        events = p.stop_recording()
        assert len(events) == 1
        assert events[0]["text"] == "hello"

    def test_recording_does_not_affect_client_broadcast(self) -> None:
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "a"})
        p.broadcast({"type": "text_delta", "text": "b"})
        client_events = _drain(q)
        assert len(client_events) == 2  # Clients get uncoalesced events
        recorded = p.stop_recording()
        assert len(recorded) == 1  # Recording coalesces
        assert recorded[0]["text"] == "ab"
        p.remove_client(q)

    def test_not_recording_by_default(self) -> None:
        p = BaseBrowserPrinter()
        p.broadcast({"type": "text_delta", "text": "x"})
        events = p.stop_recording()
        assert events == []

    def test_stop_clears_buffer(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "x"})
        p.stop_recording()
        events = p.stop_recording()
        assert events == []

    def test_restart_recording(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "first"})
        p.stop_recording()
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "second"})
        events = p.stop_recording()
        assert len(events) == 1
        assert events[0]["text"] == "second"

    def test_records_all_display_event_types(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        for t in sorted(_DISPLAY_EVENT_TYPES):
            ev = {"type": t, "text": "x"}
            p.broadcast(ev)
        events = p.stop_recording()
        recorded_types = {e["type"] for e in events}
        # system_output, text_delta, thinking_delta may be coalesced
        # but their type should still appear
        assert recorded_types == _DISPLAY_EVENT_TYPES

    def test_thread_safety(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        errors: list[Exception] = []

        def broadcast_many():
            try:
                for i in range(50):
                    p.broadcast({"type": "text_delta", "text": f"t{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=broadcast_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = p.stop_recording()
        assert not errors
        # All events captured (may be coalesced)
        total_text = "".join(e.get("text", "") for e in events)
        for i in range(50):
            assert f"t{i}" in total_text

    def test_recording_full_task_simulation(self) -> None:
        """Simulate a full task lifecycle and verify events are recorded."""
        p = BaseBrowserPrinter()
        q = _subscribe(p)

        # Pre-recording events (not captured)
        p.broadcast({"type": "tasks_updated"})

        # Start recording (task begins)
        p.start_recording()
        p.broadcast({"type": "clear", "active_file": "/path/to/file.py"})

        # Simulate thinking
        p.broadcast({"type": "thinking_start"})
        p.broadcast({"type": "thinking_delta", "text": "Let me "})
        p.broadcast({"type": "thinking_delta", "text": "think..."})
        p.broadcast({"type": "thinking_end"})

        # Simulate text output
        p.broadcast({"type": "text_delta", "text": "Here's "})
        p.broadcast({"type": "text_delta", "text": "the answer"})
        p.broadcast({"type": "text_end"})

        # Simulate tool call
        p.broadcast({"type": "tool_call", "name": "Bash", "command": "ls"})
        p.broadcast({"type": "system_output", "text": "file1.py\n"})
        p.broadcast({"type": "system_output", "text": "file2.py\n"})
        p.broadcast({"type": "tool_result", "content": "OK", "is_error": False})

        # Simulate result
        p.broadcast(
            {
                "type": "result",
                "text": "Done",
                "step_count": 3,
                "total_tokens": 1000,
                "cost": "$0.01",
            }
        )

        # Non-display event (filtered out)
        p.broadcast({"type": "tasks_updated"})

        events = p.stop_recording()

        # Verify filtered and coalesced
        types = [e["type"] for e in events]
        assert "tasks_updated" not in types
        assert "clear" in types
        assert "thinking_start" in types
        assert "thinking_end" in types
        assert "text_end" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "result" in types

        # Thinking deltas should be coalesced
        thinking_deltas = [e for e in events if e["type"] == "thinking_delta"]
        assert len(thinking_deltas) == 1
        assert thinking_deltas[0]["text"] == "Let me think..."

        # Text deltas should be coalesced
        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0]["text"] == "Here's the answer"

        # System output should be coalesced
        sys_outputs = [e for e in events if e["type"] == "system_output"]
        assert len(sys_outputs) == 1
        assert sys_outputs[0]["text"] == "file1.py\nfile2.py\n"

        # Clear event preserves active_file
        clear_ev = [e for e in events if e["type"] == "clear"][0]
        assert clear_ev["active_file"] == "/path/to/file.py"

        p.remove_client(q)


# ── Task history storage tests ───────────────────────────────────────────


class TestTaskHistoryChatEvents:
    def setup_method(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_add_task_creates_empty_chat_events(self) -> None:
        th._add_task("test task")
        history = th._load_history()
        assert history[0]["task"] == "test task"
        assert history[0]["chat_events"] == []

    def test_set_latest_chat_events(self) -> None:
        th._add_task("test task")
        events: list[dict[str, object]] = [
            {"type": "text_delta", "text": "hello"},
            {"type": "tool_call", "name": "Bash"},
        ]
        th._set_latest_chat_events(events)
        history = th._load_history()
        assert history[0]["chat_events"] == events

    def test_set_latest_chat_events_removes_old_result_key(self) -> None:
        # Simulate old-format entry
        th._save_history([{"task": "old", "result": "old result"}])
        th._history_cache = None
        th._load_history()
        th._set_latest_chat_events([{"type": "text_delta", "text": "new"}])
        history = th._load_history()
        assert "result" not in history[0]
        assert history[0]["chat_events"] == [{"type": "text_delta", "text": "new"}]

    def test_set_latest_chat_events_empty_cache(self) -> None:
        th._history_cache = []
        th._set_latest_chat_events([{"type": "text_delta", "text": "x"}])
        assert th._history_cache == []

    def test_chat_events_persisted_to_disk(self) -> None:
        th._add_task("disk test")
        events = [{"type": "result", "text": "done", "step_count": 5}]
        th._set_latest_chat_events(events)

        # Clear cache and reload from disk
        th._history_cache = None
        history = th._load_history()
        assert history[0]["chat_events"] == events

    def test_backward_compat_old_format(self) -> None:
        """Old entries with 'result' key should still load fine."""
        old_data = [
            {"task": "old task", "result": "old result"},
            {"task": "another", "result": ""},
        ]
        self.tmp.write_text(json.dumps(old_data))
        th._history_cache = None
        history = th._load_history()
        assert len(history) == 2
        assert history[0]["task"] == "old task"
        assert history[0].get("result") == "old result"
        assert history[0].get("chat_events") is None

    def test_mixed_format_entries(self) -> None:
        """Mix of old and new format entries."""
        mixed_data = [
            {"task": "new task", "chat_events": [{"type": "text_delta", "text": "hi"}]},
            {"task": "old task", "result": "done"},
        ]
        self.tmp.write_text(json.dumps(mixed_data))
        th._history_cache = None
        history = th._load_history()
        assert len(history) == 2
        assert history[0].get("chat_events") == [{"type": "text_delta", "text": "hi"}]
        assert history[1].get("result") == "done"

    def test_sample_tasks_have_chat_events(self) -> None:
        for entry in th.SAMPLE_TASKS:
            assert "chat_events" in entry
            assert entry["chat_events"] == []

    def test_deduplication_preserves_latest(self) -> None:
        th._add_task("task A")
        th._set_latest_chat_events([{"type": "text_delta", "text": "first"}])
        th._add_task("task B")
        th._add_task("task A")  # Re-add A, should be at top with empty events
        history = th._load_history()
        assert history[0]["task"] == "task A"
        assert history[0]["chat_events"] == []  # Reset on re-add
        assert history[1]["task"] == "task B"


# ── Integration: recording -> storage -> retrieval ───────────────────────


class TestEndToEndRecordAndStore:
    def setup_method(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_record_store_retrieve(self) -> None:
        """Full integration: record events, store in history, retrieve."""
        printer = BaseBrowserPrinter()

        # Add task and start recording
        th._add_task("integration test task")
        printer.start_recording()
        printer.broadcast({"type": "clear", "active_file": "/test.py"})
        printer.broadcast({"type": "text_delta", "text": "Result: "})
        printer.broadcast({"type": "text_delta", "text": "success"})
        printer.broadcast({"type": "text_end"})
        printer.broadcast(
            {
                "type": "result",
                "text": "Done",
                "step_count": 1,
                "total_tokens": 100,
            }
        )
        events = printer.stop_recording()
        events.append({"type": "task_done"})

        # Store in history
        th._set_latest_chat_events(events)

        # Reload from disk
        th._history_cache = None
        history = th._load_history()
        stored_events: list[dict[str, object]] = history[0]["chat_events"]  # type: ignore[assignment]

        assert len(stored_events) > 0
        types = [e["type"] for e in stored_events]
        assert "clear" in types
        assert "text_delta" in types
        assert "text_end" in types
        assert "result" in types
        assert "task_done" in types

        # Text deltas should be coalesced
        text_deltas = [e for e in stored_events if e["type"] == "text_delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0]["text"] == "Result: success"

    def test_task_error_recorded(self) -> None:
        printer = BaseBrowserPrinter()
        th._add_task("error task")
        printer.start_recording()
        printer.broadcast({"type": "clear", "active_file": ""})
        printer.broadcast({"type": "text_delta", "text": "Working..."})
        events = printer.stop_recording()
        events.append({"type": "task_error", "text": "Something failed"})
        th._set_latest_chat_events(events)

        th._history_cache = None
        history = th._load_history()
        stored: list[dict[str, object]] = history[0]["chat_events"]  # type: ignore[assignment]
        error_events = [e for e in stored if e["type"] == "task_error"]
        assert len(error_events) == 1
        assert error_events[0]["text"] == "Something failed"

    def test_task_stopped_recorded(self) -> None:
        printer = BaseBrowserPrinter()
        th._add_task("stopped task")
        printer.start_recording()
        printer.broadcast({"type": "clear", "active_file": ""})
        printer.broadcast({"type": "task_stopped"})
        events = printer.stop_recording()
        th._set_latest_chat_events(events)

        th._history_cache = None
        history = th._load_history()
        stored: list[dict[str, object]] = history[0]["chat_events"]  # type: ignore[assignment]
        assert any(e["type"] == "task_stopped" for e in stored)

    def test_multiple_tasks_stored_independently(self) -> None:
        printer = BaseBrowserPrinter()

        # First task
        th._add_task("task 1")
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "result 1"})
        events1 = printer.stop_recording()
        events1.append({"type": "task_done"})
        th._set_latest_chat_events(events1)

        # Second task
        th._add_task("task 2")
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "result 2"})
        events2 = printer.stop_recording()
        events2.append({"type": "task_done"})
        th._set_latest_chat_events(events2)

        th._history_cache = None
        history = th._load_history()
        assert history[0]["task"] == "task 2"
        events0: list[dict[str, object]] = history[0]["chat_events"]  # type: ignore[assignment]
        assert events0[0]["text"] == "result 2"
        assert history[1]["task"] == "task 1"
        stored1: list[dict[str, object]] = history[1]["chat_events"]  # type: ignore[assignment]
        assert stored1[0]["text"] == "result 1"


# ── Display event types completeness ─────────────────────────────────────


class TestDisplayEventTypes:
    def test_all_event_types_documented(self) -> None:
        """Verify _DISPLAY_EVENT_TYPES contains the expected types."""
        expected = {
            "clear",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "text_delta",
            "text_end",
            "tool_call",
            "tool_result",
            "system_output",
            "result",
            "prompt",
            "usage_info",
            "task_done",
            "task_error",
            "task_stopped",
            "followup_suggestion",
        }
        assert _DISPLAY_EVENT_TYPES == expected

    def test_non_display_events_filtered(self) -> None:
        non_display = [
            "tasks_updated",
            "proposed_updated",
            "theme_changed",
            "focus_chatbox",
            "merge_started",
            "merge_ended",
        ]
        for t in non_display:
            assert t not in _DISPLAY_EVENT_TYPES


# ── Recording during print() calls ──────────────────────────────────────


class TestRecordingViaPrint:
    def test_text_print_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.print("Hello world")
        events = p.stop_recording()
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert "Hello world" in events[0]["text"]

    def test_tool_call_print_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.print("Bash", type="tool_call", tool_input={"command": "ls", "description": "list"})
        events = p.stop_recording()
        # tool_call triggers text_end + tool_call
        types = [e["type"] for e in events]
        assert "tool_call" in types

    def test_tool_result_print_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.print("output", type="tool_result", is_error=False)
        events = p.stop_recording()
        assert any(e["type"] == "tool_result" for e in events)

    def test_result_print_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.print("done", type="result", step_count=1, total_tokens=50, cost="$0.01")
        events = p.stop_recording()
        types = [e["type"] for e in events]
        assert "result" in types

    def test_prompt_print_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.print("prompt text", type="prompt")
        events = p.stop_recording()
        assert any(e["type"] == "prompt" for e in events)

    def test_usage_info_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.print("tokens: 100", type="usage_info")
        events = p.stop_recording()
        assert any(e["type"] == "usage_info" for e in events)

    def test_bash_stream_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.print("line1\n", type="bash_stream")
        p._flush_bash()
        events = p.stop_recording()
        assert any(e["type"] == "system_output" for e in events)

    def test_token_callback_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        asyncio.run(p.token_callback("hello"))
        events = p.stop_recording()
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"

    def test_thinking_token_callback_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p._current_block_type = "thinking"
        p.start_recording()
        asyncio.run(p.token_callback("hmm"))
        events = p.stop_recording()
        assert len(events) == 1
        assert events[0]["type"] == "thinking_delta"


# ── JSON serialization roundtrip ─────────────────────────────────────────


class TestJsonRoundtrip:
    def setup_method(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_complex_events_survive_json_roundtrip(self) -> None:
        events: list[dict[str, object]] = [
            {"type": "clear", "active_file": "/path/to/file.py"},
            {"type": "thinking_start"},
            {"type": "thinking_delta", "text": "Let me think..."},
            {"type": "thinking_end"},
            {"type": "text_delta", "text": "Here's the answer"},
            {"type": "text_end"},
            {
                "type": "tool_call",
                "name": "Edit",
                "path": "/file.py",
                "old_string": "old code",
                "new_string": "new code",
                "description": "Fix bug",
            },
            {"type": "tool_result", "content": "OK", "is_error": False},
            {
                "type": "result",
                "text": "success: true\nsummary: Done",
                "step_count": 5,
                "total_tokens": 2000,
                "cost": "$0.05",
                "success": True,
                "summary": "Done",
            },
            {"type": "task_done"},
        ]

        th._add_task("roundtrip test")
        th._set_latest_chat_events(events)

        # Force reload from disk
        th._history_cache = None
        history = th._load_history()
        retrieved: list[dict[str, object]] = history[0]["chat_events"]  # type: ignore[assignment]

        assert retrieved == events
        assert retrieved[6]["old_string"] == "old code"
        assert retrieved[6]["new_string"] == "new code"
        assert retrieved[8]["success"] is True


# ── /tasks endpoint format tests ─────────────────────────────────────────


def _tasks_endpoint_transform(history: list[dict]) -> list[dict]:
    """Replicate the /tasks endpoint list comprehension from sorcar.py."""
    return [
        {"task": e["task"], "has_events": bool(e.get("chat_events"))}
        for e in history
    ]


def _task_events_lookup(history: list[dict], idx: int) -> dict:
    """Replicate the /task-events endpoint logic from sorcar.py."""
    if 0 <= idx < len(history):
        entry = history[idx]
        events = entry.get("chat_events", [])
        return {"events": events, "task": entry["task"]}
    return {"events": [], "task": ""}


class TestTasksEndpointFormat:
    def setup_method(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_entries_with_events_have_has_events_true(self) -> None:
        th._add_task("task with events")
        th._set_latest_chat_events([{"type": "text_delta", "text": "hello"}])
        history = th._load_history()
        result = _tasks_endpoint_transform(history)
        assert result[0] == {"task": "task with events", "has_events": True}

    def test_entries_with_empty_events_have_has_events_false(self) -> None:
        th._add_task("task no events")
        history = th._load_history()
        result = _tasks_endpoint_transform(history)
        assert result[0] == {"task": "task no events", "has_events": False}

    def test_entries_without_chat_events_key_have_has_events_false(self) -> None:
        th._save_history([{"task": "old format", "result": "done"}])
        th._history_cache = None
        history = th._load_history()
        result = _tasks_endpoint_transform(history)
        assert result[0] == {"task": "old format", "has_events": False}

    def test_mixed_entries(self) -> None:
        mixed: list[dict[str, object]] = [
            {"task": "with events", "chat_events": [{"type": "text_delta", "text": "x"}]},
            {"task": "empty events", "chat_events": []},
            {"task": "old format", "result": "done"},
        ]
        th._save_history(mixed)
        th._history_cache = None
        history = th._load_history()
        result = _tasks_endpoint_transform(history)
        assert result[0]["has_events"] is True
        assert result[1]["has_events"] is False
        assert result[2]["has_events"] is False

    def test_sample_tasks_all_have_has_events_false(self) -> None:
        result = _tasks_endpoint_transform(th.SAMPLE_TASKS)
        for entry in result:
            assert entry["has_events"] is False
            assert "chat_events" not in entry

    def test_full_flow_record_and_check_format(self) -> None:
        printer = BaseBrowserPrinter()
        th._add_task("full flow")
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "output"})
        events = printer.stop_recording()
        events.append({"type": "task_done"})
        th._set_latest_chat_events(events)

        history = th._load_history()
        result = _tasks_endpoint_transform(history)
        assert result[0]["task"] == "full flow"
        assert result[0]["has_events"] is True


# ── /task-events endpoint logic tests ────────────────────────────────────


class TestTaskEventsEndpoint:
    def setup_method(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_valid_index_returns_events(self) -> None:
        events: list[dict[str, object]] = [{"type": "text_delta", "text": "hello"}]
        th._add_task("test task")
        th._set_latest_chat_events(events)
        history = th._load_history()
        result = _task_events_lookup(history, 0)
        assert result["task"] == "test task"
        assert result["events"] == events

    def test_index_out_of_range_returns_empty(self) -> None:
        th._add_task("single task")
        history = th._load_history()
        result = _task_events_lookup(history, 99)
        assert result == {"events": [], "task": ""}

    def test_negative_index_returns_empty(self) -> None:
        th._add_task("single task")
        history = th._load_history()
        result = _task_events_lookup(history, -1)
        assert result == {"events": [], "task": ""}

    def test_entry_without_chat_events_returns_empty_list(self) -> None:
        th._save_history([{"task": "old entry", "result": "done"}])
        th._history_cache = None
        history = th._load_history()
        result = _task_events_lookup(history, 0)
        assert result["task"] == "old entry"
        assert result["events"] == []

    def test_multiple_tasks_correct_index(self) -> None:
        th._add_task("first")
        th._set_latest_chat_events([{"type": "text_delta", "text": "r1"}])
        th._add_task("second")
        th._set_latest_chat_events([{"type": "text_delta", "text": "r2"}])
        history = th._load_history()
        r0 = _task_events_lookup(history, 0)
        r1 = _task_events_lookup(history, 1)
        assert r0["task"] == "second"
        assert r0["events"][0]["text"] == "r2"
        assert r1["task"] == "first"
        assert r1["events"][0]["text"] == "r1"

    def test_empty_history_returns_empty(self) -> None:
        result = _task_events_lookup([], 0)
        assert result == {"events": [], "task": ""}

    def test_full_flow_record_store_retrieve_by_index(self) -> None:
        printer = BaseBrowserPrinter()
        th._add_task("task A")
        printer.start_recording()
        printer.broadcast({"type": "clear", "active_file": "/a.py"})
        printer.broadcast({"type": "text_delta", "text": "answer A"})
        printer.broadcast({"type": "text_end"})
        events_a = printer.stop_recording()
        events_a.append({"type": "task_done"})
        th._set_latest_chat_events(events_a)

        th._add_task("task B")
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "answer B"})
        events_b = printer.stop_recording()
        events_b.append({"type": "task_done"})
        th._set_latest_chat_events(events_b)

        th._history_cache = None
        history = th._load_history()

        result_b = _task_events_lookup(history, 0)
        assert result_b["task"] == "task B"
        assert any(e.get("text") == "answer B" for e in result_b["events"])

        result_a = _task_events_lookup(history, 1)
        assert result_a["task"] == "task A"
        assert any(e.get("text") == "answer A" for e in result_a["events"])
        assert any(e.get("active_file") == "/a.py" for e in result_a["events"])


# ── JavaScript syntax validation ─────────────────────────────────────────


class TestChatbotJSSyntax:
    def test_render_tasks_balanced_braces(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function renderTasks(q){")
        depth = 0
        i = start
        while i < len(CHATBOT_JS):
            if CHATBOT_JS[i] == "{":
                depth += 1
            elif CHATBOT_JS[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        assert depth == 0, f"Unbalanced braces in renderTasks, depth={depth}"

    def test_render_tasks_single_for_each(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function renderTasks(q){")
        end_search = CHATBOT_JS.index("function replayTaskEvents(")
        render_tasks_js = CHATBOT_JS[start:end_search]
        count = render_tasks_js.count("allTasks.forEach")
        assert count == 1, f"Expected 1 allTasks.forEach, found {count}"

    def test_render_tasks_no_filtered_variable(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function renderTasks(q){")
        end_search = CHATBOT_JS.index("function replayTaskEvents(")
        render_tasks_js = CHATBOT_JS[start:end_search]
        assert "filtered" not in render_tasks_js

    def test_replay_task_events_balanced_braces(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function replayTaskEvents(")
        depth = 0
        i = start
        while i < len(CHATBOT_JS):
            if CHATBOT_JS[i] == "{":
                depth += 1
            elif CHATBOT_JS[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        assert depth == 0, f"Unbalanced braces in replayTaskEvents, depth={depth}"

    def test_full_js_balanced_braces(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        depth = 0
        for ch in CHATBOT_JS:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        assert depth == 0, f"Full JS has unbalanced braces, depth={depth}"

    def test_build_html_produces_valid_structure(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import _build_html

        html = _build_html("Test")
        assert "<script>" in html
        assert "</script>" in html
        assert "renderTasks" in html
        assert "loadWelcome" in html
        assert "loadTasks" in html
        assert "loadProposed" in html
        assert "replayTaskEvents" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
