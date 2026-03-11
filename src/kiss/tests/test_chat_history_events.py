"""Integration tests for chat history event recording and replay.

Tests the full flow: recording events via BaseBrowserPrinter, storing them
in JSONL task history with separate chat event files, and retrieving them
for replay. No mocks or patches.
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
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _use_temp_history():
    """Redirect HISTORY_FILE and _CHAT_EVENTS_DIR to temp locations."""
    original_file = th.HISTORY_FILE
    original_events_dir = th._CHAT_EVENTS_DIR
    tmp_dir = Path(tempfile.mkdtemp())
    th.HISTORY_FILE = tmp_dir / "task_history.jsonl"
    th._CHAT_EVENTS_DIR = tmp_dir / "chat_events"
    th._history_cache = None
    return original_file, original_events_dir, tmp_dir


def _restore_history(original_file: Path, original_events_dir: Path, tmp_dir: Path) -> None:
    th.HISTORY_FILE = original_file
    th._CHAT_EVENTS_DIR = original_events_dir
    th._history_cache = None
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


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
        self.original_file, self.original_events_dir, self.tmp_dir = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original_file, self.original_events_dir, self.tmp_dir)

    def test_sample_tasks_have_task_key(self) -> None:
        for entry in th.SAMPLE_TASKS:
            assert "task" in entry

# ── Integration: recording -> storage -> retrieval ───────────────────────


class TestEndToEndRecordAndStore:
    def setup_method(self) -> None:
        self.original_file, self.original_events_dir, self.tmp_dir = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original_file, self.original_events_dir, self.tmp_dir)

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
        assert history[0]["has_events"] is True

        # Load events on demand
        stored_events = th._load_task_chat_events(str(history[0]["task"]))

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

class TestJsonRoundtrip:
    def setup_method(self) -> None:
        self.original_file, self.original_events_dir, self.tmp_dir = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original_file, self.original_events_dir, self.tmp_dir)

# ── /tasks endpoint format tests ─────────────────────────────────────────


def _tasks_endpoint_transform(history: list[Any]) -> list[dict[str, Any]]:
    """Replicate the /tasks endpoint list comprehension from sorcar.py."""
    return [
        {"task": e["task"], "has_events": bool(e.get("has_events"))}
        for e in history
    ]


class TestTasksEndpointFormat:
    def setup_method(self) -> None:
        self.original_file, self.original_events_dir, self.tmp_dir = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original_file, self.original_events_dir, self.tmp_dir)

    def test_sample_tasks_all_have_has_events_false(self) -> None:
        result = _tasks_endpoint_transform(th.SAMPLE_TASKS)
        for entry in result:
            assert "task" in entry
            assert entry["has_events"] is False

    def test_task_with_events_has_events_true(self) -> None:
        th._add_task("task with events")
        th._set_latest_chat_events(
            [{"type": "text_delta", "text": "hi"}], task="task with events"
        )
        th._add_task("task without events")
        history = th._load_history()
        result = _tasks_endpoint_transform(history)
        # task without events is first (most recent)
        assert result[0]["has_events"] is False
        assert result[1]["has_events"] is True

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


class TestTaskEventsEndpoint:
    def setup_method(self) -> None:
        self.original_file, self.original_events_dir, self.tmp_dir = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(self.original_file, self.original_events_dir, self.tmp_dir)

    def test_returns_empty_for_sample_tasks(self) -> None:
        history = th._load_history()
        result = th._load_task_chat_events(str(history[0]["task"]))
        assert result == []

    def test_out_of_range_returns_empty(self) -> None:
        result = th._load_task_chat_events("nonexistent_task_xyz")
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
