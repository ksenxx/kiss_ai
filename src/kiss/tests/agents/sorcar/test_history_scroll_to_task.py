"""Test that resumeSession with a taskId loads the specific task, not the latest.

When a user clicks a specific task in the history panel, the backend should
load that task's events (not the most recent task in the chat session) so
the frontend can scroll to the clicked task.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar import persistence as th
from kiss.agents.vscode.server import VSCodeServer


class TestResumeSessionWithTaskId:
    """resumeSession with taskId loads the specific task, not the latest."""

    def test_replay_session_loads_specific_task(self, tmp_path: Path) -> None:
        """When taskId is provided, _replay_session loads that task's events.

        Reproduces the bug: a chat session has two tasks (task_a older,
        task_b newer).  Without the fix, clicking task_a in the history
        would load task_b (the latest) and scroll to the end.  With the
        fix, task_a's events are loaded so the frontend shows the
        correct task.
        """
        orig_dir = th._KISS_DIR
        orig_db = th._DB_PATH
        orig_conn = th._db_conn
        try:
            th._db_conn = None
            th._KISS_DIR = tmp_path
            th._DB_PATH = tmp_path / "sorcar.db"

            # Create two tasks in the same chat session
            task_a_id, chat_id = th._add_task("task alpha", chat_id="0")
            events_a: list[dict[str, object]] = [
                {"type": "text_delta", "text": "alpha response"},
                {"type": "result", "summary": "alpha done"},
            ]
            for ev in events_a:
                th._append_chat_event(ev, task_id=task_a_id)

            task_b_id, _ = th._add_task("task beta", chat_id=chat_id)
            events_b: list[dict[str, object]] = [
                {"type": "text_delta", "text": "beta response"},
                {"type": "result", "summary": "beta done"},
            ]
            for ev in events_b:
                th._append_chat_event(ev, task_id=task_b_id)

            server = VSCodeServer()
            captured: list[dict[str, object]] = []
            orig_broadcast = server.printer.broadcast

            def capture(ev: dict[str, object]) -> None:
                captured.append(ev)
                orig_broadcast(ev)

            server.printer.broadcast = capture  # type: ignore[assignment]

            # --- Bug reproduction: without taskId, it loads the latest ---
            server._replay_session(chat_id, tab_id="tab-latest")
            te_latest = [e for e in captured if e.get("type") == "task_events"]
            assert len(te_latest) == 1
            # Without the fix, this always loads task_b (the latest)
            assert te_latest[0]["task"] == "task beta"

            # --- Fix verification: with taskId, load the specific task ---
            captured.clear()
            server._replay_session(
                chat_id, tab_id="tab-specific", task_id=task_a_id,
            )
            te_specific = [e for e in captured if e.get("type") == "task_events"]
            assert len(te_specific) == 1
            assert te_specific[0]["task"] == "task alpha"
            ev_list = te_specific[0].get("events", [])
            assert isinstance(ev_list, list)
            assert len(ev_list) == 2
            assert ev_list[0]["text"] == "alpha response"
            assert te_specific[0]["chat_id"] == chat_id
        finally:
            th._close_db()
            th._db_conn = orig_conn
            th._KISS_DIR = orig_dir
            th._DB_PATH = orig_db

    def test_cmd_resume_session_forwards_task_id(self, tmp_path: Path) -> None:
        """_cmd_resume_session passes taskId through to _replay_session."""
        orig_dir = th._KISS_DIR
        orig_db = th._DB_PATH
        orig_conn = th._db_conn
        try:
            th._db_conn = None
            th._KISS_DIR = tmp_path
            th._DB_PATH = tmp_path / "sorcar.db"

            task_id, chat_id = th._add_task("the task", chat_id="0")
            th._append_chat_event(
                {"type": "result", "summary": "ok"}, task_id=task_id,
            )

            server = VSCodeServer()
            captured: list[dict[str, object]] = []
            orig_broadcast = server.printer.broadcast

            def capture(ev: dict[str, object]) -> None:
                captured.append(ev)
                orig_broadcast(ev)

            server.printer.broadcast = capture  # type: ignore[assignment]

            server._handle_command({
                "type": "resumeSession",
                "chatId": chat_id,
                "taskId": task_id,
                "tabId": "tab-cmd",
            })
            te = [e for e in captured if e.get("type") == "task_events"]
            assert len(te) == 1
            assert te[0]["task"] == "the task"
            assert te[0]["task_id"] == task_id
        finally:
            th._close_db()
            th._db_conn = orig_conn
            th._KISS_DIR = orig_dir
            th._DB_PATH = orig_db

    def test_load_chat_events_by_task_id(self, tmp_path: Path) -> None:
        """_load_chat_events_by_task_id returns the correct task."""
        orig_dir = th._KISS_DIR
        orig_db = th._DB_PATH
        orig_conn = th._db_conn
        try:
            th._db_conn = None
            th._KISS_DIR = tmp_path
            th._DB_PATH = tmp_path / "sorcar.db"

            task_id, chat_id = th._add_task("specific task", chat_id="0")
            th._append_chat_event(
                {"type": "text_delta", "text": "hi"}, task_id=task_id,
            )

            result = th._load_chat_events_by_task_id(task_id)
            assert result is not None
            assert result["task"] == "specific task"
            assert result["task_id"] == task_id
            assert result["chat_id"] == chat_id
            evts = result["events"]
            assert isinstance(evts, list)
            assert len(evts) == 1

            # Non-existent task_id returns None
            assert th._load_chat_events_by_task_id(999999) is None
        finally:
            th._close_db()
            th._db_conn = orig_conn
            th._KISS_DIR = orig_dir
            th._DB_PATH = orig_db

    def test_replay_session_falls_back_without_task_id(
        self, tmp_path: Path,
    ) -> None:
        """When taskId is None, _replay_session loads the latest task."""
        orig_dir = th._KISS_DIR
        orig_db = th._DB_PATH
        orig_conn = th._db_conn
        try:
            th._db_conn = None
            th._KISS_DIR = tmp_path
            th._DB_PATH = tmp_path / "sorcar.db"

            task_a_id, chat_id = th._add_task("old task", chat_id="0")
            th._append_chat_event(
                {"type": "result", "summary": "old"}, task_id=task_a_id,
            )
            task_b_id, _ = th._add_task("new task", chat_id=chat_id)
            th._append_chat_event(
                {"type": "result", "summary": "new"}, task_id=task_b_id,
            )

            server = VSCodeServer()
            captured: list[dict[str, object]] = []
            orig_broadcast = server.printer.broadcast

            def capture(ev: dict[str, object]) -> None:
                captured.append(ev)
                orig_broadcast(ev)

            server.printer.broadcast = capture  # type: ignore[assignment]

            # task_id=None should fall back to latest
            server._replay_session(chat_id, tab_id="tab-fb", task_id=None)
            te = [e for e in captured if e.get("type") == "task_events"]
            assert len(te) == 1
            assert te[0]["task"] == "new task"
        finally:
            th._close_db()
            th._db_conn = orig_conn
            th._KISS_DIR = orig_dir
            th._DB_PATH = orig_db

    def test_replay_session_invalid_task_id_falls_back(
        self, tmp_path: Path,
    ) -> None:
        """When taskId doesn't exist, falls back to loading the latest task."""
        orig_dir = th._KISS_DIR
        orig_db = th._DB_PATH
        orig_conn = th._db_conn
        try:
            th._db_conn = None
            th._KISS_DIR = tmp_path
            th._DB_PATH = tmp_path / "sorcar.db"

            task_id, chat_id = th._add_task("only task", chat_id="0")
            th._append_chat_event(
                {"type": "result", "summary": "done"}, task_id=task_id,
            )

            server = VSCodeServer()
            captured: list[dict[str, object]] = []
            orig_broadcast = server.printer.broadcast

            def capture(ev: dict[str, object]) -> None:
                captured.append(ev)
                orig_broadcast(ev)

            server.printer.broadcast = capture  # type: ignore[assignment]

            # Non-existent task_id → should fall back to latest
            server._replay_session(
                chat_id, tab_id="tab-inv", task_id=999999,
            )
            te = [e for e in captured if e.get("type") == "task_events"]
            assert len(te) == 1
            assert te[0]["task"] == "only task"
        finally:
            th._close_db()
            th._db_conn = orig_conn
            th._KISS_DIR = orig_dir
            th._DB_PATH = orig_db
