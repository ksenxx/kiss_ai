"""Integration tests for chat history event recording and replay.

Tests the full flow: recording events via BaseBrowserPrinter, storing them
in task_history.json, and retrieving them for replay. No mocks or patches.
"""

import queue
import tempfile
from pathlib import Path
from typing import Any

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
    def test_no_merge_without_text_field(self) -> None:
        events = [
            {"type": "text_delta"},
            {"type": "text_delta", "text": "a"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

# ── Recording tests ─────────────────────────────────────────────────────


class TestPrinterRecording:
    def test_stop_clears_buffer(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "x"})
        p.stop_recording()
        events = p.stop_recording()
        assert events == []

# ── Task history storage tests ───────────────────────────────────────────


class TestTaskHistoryChatEvents:
    def setup_method(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_sample_tasks_have_chat_events(self) -> None:
        for entry in th.SAMPLE_TASKS:
            assert "chat_events" in entry
            assert entry["chat_events"] == []

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
    def test_tool_call_print_recorded(self) -> None:
        p = BaseBrowserPrinter()
        p.start_recording()
        p.print("Bash", type="tool_call", tool_input={"command": "ls", "description": "list"})
        events = p.stop_recording()
        # tool_call triggers text_end + tool_call
        types = [e["type"] for e in events]
        assert "tool_call" in types

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

# ── JSON serialization roundtrip ─────────────────────────────────────────


class TestJsonRoundtrip:
    def setup_method(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original, self.tmp)

# ── /tasks endpoint format tests ─────────────────────────────────────────


def _tasks_endpoint_transform(history: list[Any]) -> list[dict[str, Any]]:
    """Replicate the /tasks endpoint list comprehension from sorcar.py."""
    return [
        {"task": e["task"], "has_events": bool(e.get("chat_events"))}
        for e in history
    ]


class TestTasksEndpointFormat:
    def setup_method(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_sample_tasks_all_have_has_events_false(self) -> None:
        result = _tasks_endpoint_transform(th.SAMPLE_TASKS)
        for entry in result:
            assert "task" in entry
            assert entry["has_events"] is False

    def test_task_with_events_has_events_true(self) -> None:
        history = [
            {"task": "task with events", "chat_events": [{"type": "text_delta", "text": "hi"}]},
            {"task": "task without events", "chat_events": []},
        ]
        result = _tasks_endpoint_transform(history)
        assert result[0]["has_events"] is True
        assert result[1]["has_events"] is False

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
        end_search = CHATBOT_JS.index("\nhistSearch.addEventListener")
        render_tasks_js = CHATBOT_JS[start:end_search]
        count = render_tasks_js.count("allTasks.forEach")
        assert count == 1, f"Expected 1 allTasks.forEach, found {count}"

    def test_render_tasks_no_filtered_variable(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function renderTasks(q){")
        end_search = CHATBOT_JS.index("\nhistSearch.addEventListener")
        render_tasks_js = CHATBOT_JS[start:end_search]
        assert "filtered" not in render_tasks_js

    def test_render_tasks_copies_to_input_and_replays(self) -> None:
        """Verify clicking a task in history copies text and replays events."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function renderTasks(q){")
        end_search = CHATBOT_JS.index("histSearch.addEventListener")
        render_tasks_js = CHATBOT_JS[start:end_search]
        assert "inp.value=txt" in render_tasks_js
        assert "inp.focus()" in render_tasks_js
        assert "replayTaskEvents(idx,txt)" in render_tasks_js
        assert "hasEvents" in render_tasks_js

    def test_replay_task_events_function_exists(self) -> None:
        """Verify replayTaskEvents function is defined."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        assert "function replayTaskEvents(idx,txt){" in CHATBOT_JS
        # It should fetch from /task-events
        start = CHATBOT_JS.index("function replayTaskEvents(idx,txt){")
        end = CHATBOT_JS.index("function renderTasks(q){")
        replay_js = CHATBOT_JS[start:end]
        assert "/task-events" in replay_js
        assert "handleOutputEvent" in replay_js
        assert "showUserMsg" in replay_js

    def test_replay_task_events_does_not_open_sidebar(self) -> None:
        """Verify replayTaskEvents only closes sidebar if open, not toggles."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function replayTaskEvents(idx,txt){")
        end = CHATBOT_JS.index("function renderTasks(q){")
        replay_js = CHATBOT_JS[start:end]
        # Must NOT call toggleSidebar() unconditionally
        assert "toggleSidebar();" not in replay_js.replace(
            "if(sidebar.classList.contains('open')){toggleSidebar();}", ""
        )
        # Must guard toggleSidebar with sidebar open check
        assert "if(sidebar.classList.contains('open')){toggleSidebar();}" in replay_js

    def test_welcome_recent_clicks_replay_events(self) -> None:
        """Verify clicking a recent item in welcome replays events like sidebar."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function loadWelcome(){")
        end = CHATBOT_JS.index("\n}", start) + 2
        welcome_js = CHATBOT_JS[start:end]
        # Recent items must track hasEvents and idx
        assert "hasEvents" in welcome_js
        assert "item.idx" in welcome_js
        # Must call replayTaskEvents for recent items with events
        assert "replayTaskEvents(item.idx,item.text)" in welcome_js
        # Suggested items should just set input
        assert "inp.value=item.text" in welcome_js

    def test_full_js_balanced_braces(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        depth = 0
        for ch in CHATBOT_JS:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        assert depth == 0, f"Full JS has unbalanced braces, depth={depth}"

# ── /task-events endpoint logic tests ────────────────────────────────────


def _task_events_lookup(history: list[Any], idx: int) -> Any:
    """Replicate the /task-events endpoint logic from sorcar.py."""
    if idx < 0 or idx >= len(history):
        return {"error": "Index out of range"}
    return history[idx].get("chat_events", [])


class TestTaskEventsEndpoint:
    def setup_method(self) -> None:
        self.original, self.tmp = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original, self.tmp)

    def test_returns_empty_for_sample_tasks(self) -> None:
        result = _task_events_lookup(th.SAMPLE_TASKS, 0)
        assert result == []

    def test_returns_events_for_task_with_events(self) -> None:
        th._add_task("test task with events")
        events: list[dict[str, Any]] = [{"type": "text_delta", "text": "hello"}]
        th._set_latest_chat_events(events, task="test task with events")
        history = th._load_history()
        result = _task_events_lookup(history, 0)
        assert len(result) == 1
        assert result[0]["type"] == "text_delta"

    def test_out_of_range_returns_error(self) -> None:
        result = _task_events_lookup(th.SAMPLE_TASKS, 999)
        assert "error" in result

    def test_negative_index_returns_error(self) -> None:
        result = _task_events_lookup(th.SAMPLE_TASKS, -1)
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
