"""Integration tests for chat history event recording and replay.

Tests the full flow: recording events via BaseBrowserPrinter, storing them
in SQLite task history with events table, and retrieving them for replay.
No mocks or patches.
"""

import queue
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar.browser_ui import (
    _DISPLAY_EVENT_TYPES,
    BaseBrowserPrinter,
)


def _use_temp_history():
    """Redirect DB to a temp location and reset the singleton connection."""
    original_db_path = th._DB_PATH
    original_db_conn = th._db_conn
    original_kiss_dir = th._KISS_DIR
    tmp_dir = Path(tempfile.mkdtemp())
    kiss_dir = tmp_dir / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return original_db_path, original_db_conn, original_kiss_dir, tmp_dir


def _restore_history(
    original_db_path: Path, original_db_conn, original_kiss_dir: Path, tmp_dir: Path
) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
    th._DB_PATH = original_db_path
    th._db_conn = original_db_conn
    th._KISS_DIR = original_kiss_dir
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


class TestTaskHistoryChatEvents:
    def setup_method(self) -> None:
        self.saved = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(*self.saved)

    def test_sample_tasks_have_task_key(self) -> None:
        for entry in th.SAMPLE_TASKS:
            assert "task" in entry

    def test_first_run_seeds_sample_tasks(self) -> None:
        """On first run (empty DB), SAMPLE_TASKS are inserted."""
        history = th._load_history()
        assert len(history) == len(th.SAMPLE_TASKS)
        sample_tasks = [s["task"] for s in th.SAMPLE_TASKS]
        for entry in history:
            assert entry["task"] in sample_tasks
            assert entry["has_events"] == 0


class TestEndToEndRecordAndStore:
    def setup_method(self) -> None:
        self.saved = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(*self.saved)

    def test_record_store_retrieve(self) -> None:
        """Full integration: record events, store in history, retrieve."""
        printer = BaseBrowserPrinter()

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

        th._set_latest_chat_events(events)

        history = th._load_history()
        task_entry = next(e for e in history if e["task"] == "integration test task")
        assert task_entry["has_events"] == 1

        stored_events = th._load_task_chat_events(str(task_entry["task"]))

        assert len(stored_events) > 0
        types = [e["type"] for e in stored_events]
        assert "clear" in types
        assert "text_delta" in types
        assert "text_end" in types
        assert "result" in types
        assert "task_done" in types

        text_deltas = [e for e in stored_events if e["type"] == "text_delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0]["text"] == "Result: success"


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
            "system_prompt",
        }
        assert _DISPLAY_EVENT_TYPES == expected

    def test_non_display_events_filtered(self) -> None:
        non_display = [
            "tasks_updated",
            "theme_changed",
            "focus_chatbox",
            "merge_started",
            "merge_ended",
        ]
        for t in non_display:
            assert t not in _DISPLAY_EVENT_TYPES

def _tasks_endpoint_transform(history: list[Any]) -> list[dict[str, Any]]:
    """Replicate the /tasks endpoint list comprehension from sorcar.py."""
    return [
        {"task": e["task"], "has_events": bool(e.get("has_events"))}
        for e in history
    ]


class TestTasksEndpointFormat:
    def setup_method(self) -> None:
        self.saved = _use_temp_history()

    def teardown_method(self) -> None:
        _restore_history(*self.saved)

    def test_sample_tasks_all_have_has_events_false(self) -> None:
        result = _tasks_endpoint_transform(th.SAMPLE_TASKS)
        for entry in result:
            assert "task" in entry
            assert entry["has_events"] is False


class TestChatbotJSSyntax:
    def test_render_tasks_balanced_braces(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function renderSidebarTasks(tasks){")
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

        start = CHATBOT_JS.index("function renderSidebarTasks(tasks){")
        end_search = CHATBOT_JS.index("\nvar _histSearchTimer")
        render_tasks_js = CHATBOT_JS[start:end_search]
        count = render_tasks_js.count("tasks.forEach")
        assert count == 1, f"Expected 1 tasks.forEach, found {count}"

    def test_render_tasks_no_filtered_variable(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function renderSidebarTasks(tasks){")
        end_search = CHATBOT_JS.index("\nvar _histSearchTimer")
        render_tasks_js = CHATBOT_JS[start:end_search]
        assert "filtered" not in render_tasks_js

    def test_render_tasks_copies_to_input_and_replays(self) -> None:
        """Verify clicking a task in history copies text and replays events."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function renderSidebarTasks(tasks){")
        end_search = CHATBOT_JS.index("var _histSearchTimer")
        render_tasks_js = CHATBOT_JS[start:end_search]
        assert "inp.value=txt" in render_tasks_js
        assert "inp.focus()" in render_tasks_js
        assert "replayTaskEvents(idx,txt)" in render_tasks_js
        assert "hasEvents" in render_tasks_js

    def test_replay_task_events_function_exists(self) -> None:
        """Verify replayTaskEvents function is defined."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        assert "function replayTaskEvents(idx,txt){" in CHATBOT_JS
        start = CHATBOT_JS.index("function replayTaskEvents(idx,txt){")
        end = CHATBOT_JS.index("function renderSidebarTasks(tasks){")
        replay_js = CHATBOT_JS[start:end]
        assert "/task-events" in replay_js
        assert "processOutputEvent" in replay_js
        assert "showUserMsg" in replay_js

    def test_replay_task_events_does_not_open_sidebar(self) -> None:
        """Verify replayTaskEvents only closes sidebar if open, not toggles."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function replayTaskEvents(idx,txt){")
        end = CHATBOT_JS.index("function renderSidebarTasks(tasks){")
        replay_js = CHATBOT_JS[start:end]
        assert "toggleSidebar();" not in replay_js.replace(
            "if(sidebar.classList.contains('open')){toggleSidebar();}", ""
        )
        assert "if(sidebar.classList.contains('open')){toggleSidebar();}" in replay_js

    def test_welcome_recent_clicks_replay_events(self) -> None:
        """Verify clicking a recent item in welcome replays events like sidebar."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function loadWelcome(){")
        end = CHATBOT_JS.index("\n}", start) + 2
        welcome_js = CHATBOT_JS[start:end]
        assert "hasEvents" in welcome_js
        assert "replayTaskEvents" in welcome_js
        assert "inp.value=text" in welcome_js

    def test_full_js_balanced_braces(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        depth = 0
        for ch in CHATBOT_JS:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        assert depth == 0, f"Full JS has unbalanced braces, depth={depth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
