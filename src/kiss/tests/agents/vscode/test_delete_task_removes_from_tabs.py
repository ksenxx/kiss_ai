"""Tests for: when a task is deleted from the history panel, the
corresponding task and its chat are removed from open tabs that show it.

These tests cover:

1. Backend (Python) — ``_handle_delete_task`` broadcasts a
   ``taskDeleted`` event carrying the deleted ``taskId``, the
   ``chatId`` of the chat that owned the task, and a
   ``chatHasMoreTasks`` flag so the frontend knows whether to drop
   only one adjacent block or close the whole tab.

2. Backend — ``adjacent_task_events`` and ``task_events`` broadcasts
   include ``task_id`` so the frontend can stamp ``data-task-id`` on
   adjacent-task DOM nodes and track each tab's current task id.

3. Frontend (static checks on ``media/main.js``) — the JS source
   defines a ``taskDeleted`` message handler that looks up tabs by
   ``backendChatId``, removes ``.adjacent-task[data-task-id]`` nodes
   from both the active output area and any saved ``outputFragment``,
   and closes the tab when its current task is the deleted one or
   when the chat has no remaining tasks.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.server import VSCodeServer

MAIN_JS = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    """Redirect the persistence DB to a temp dir; return saved state."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with an in-memory broadcast capture."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestDeleteTaskBackendBroadcast:
    """Backend ``_handle_delete_task`` broadcasts a ``taskDeleted``
    event with enough info for the frontend to update open tabs."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_delete_broadcasts_task_deleted_event(self) -> None:
        """Deleting a task broadcasts ``taskDeleted`` with task_id +
        chat_id so any open tab showing the chat can prune the entry.
        """
        id1, chat_id = th._add_task("first task")
        time.sleep(0.01)
        id2, _ = th._add_task("second task", chat_id=chat_id)
        server, events = _make_server()

        server._handle_delete_task(id1)

        deleted = [e for e in events if e.get("type") == "taskDeleted"]
        assert len(deleted) == 1, f"expected one taskDeleted event, got {events}"
        ev = deleted[0]
        assert ev["taskId"] == id1
        assert ev["chatId"] == chat_id
        assert ev["chatHasMoreTasks"] is True

    def test_delete_last_task_in_chat_signals_no_more(self) -> None:
        """When the deleted task was the only task in its chat,
        ``chatHasMoreTasks`` is False so the frontend can close the
        corresponding tab(s)."""
        task_id, chat_id = th._add_task("only task")
        server, events = _make_server()

        server._handle_delete_task(task_id)

        deleted = [e for e in events if e.get("type") == "taskDeleted"]
        assert len(deleted) == 1
        ev = deleted[0]
        assert ev["taskId"] == task_id
        assert ev["chatId"] == chat_id
        assert ev["chatHasMoreTasks"] is False

    def test_delete_nonexistent_task_does_not_broadcast(self) -> None:
        """Deleting a non-existent task must not broadcast anything."""
        server, events = _make_server()
        server._handle_delete_task(99999)
        assert [e for e in events if e.get("type") == "taskDeleted"] == []


class TestAdjacentTaskEventsCarryTaskId:
    """The adjacent_task_events broadcast must include ``task_id`` so
    the frontend can stamp ``data-task-id`` on the rendered
    ``.adjacent-task`` container."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_adjacent_task_event_includes_task_id(self) -> None:
        id1, chat_id = th._add_task("first")
        time.sleep(0.01)
        id2, _ = th._add_task("second", chat_id=chat_id)
        th._append_chat_event({"type": "text_delta", "text": "x"}, task_id=id1)
        server, events = _make_server()

        server._get_adjacent_task(chat_id, "second", "prev", "tab-1")

        adj = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(adj) == 1, f"expected one adjacent_task_events, got {events}"
        ev = adj[0]
        assert ev["task"] == "first"
        assert ev["task_id"] == id1


class TestReplaySessionIncludesTaskId:
    """``_replay_session`` broadcasts ``task_events`` carrying the
    current task's ``task_id`` so each tab can track which task it
    is currently displaying."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_replay_event_includes_task_id(self) -> None:
        task_id, chat_id = th._add_task("only")
        th._append_chat_event(
            {"type": "text_delta", "text": "hi"}, task_id=task_id,
        )
        server, events = _make_server()

        server._replay_session(chat_id, tab_id="tab-x")

        replays = [e for e in events if e.get("type") == "task_events"]
        assert len(replays) == 1, f"expected one task_events, got {events}"
        ev = replays[0]
        assert ev["task_id"] == task_id
        assert ev["chat_id"] == chat_id


class TestMainJsTaskDeletedHandler:
    """Static checks that ``media/main.js`` handles the
    ``taskDeleted`` message correctly: it must find tabs by
    ``backendChatId``, remove matching ``.adjacent-task[data-task-id]``
    nodes from both the live DOM and detached ``outputFragment``s,
    and close any tab whose current task was deleted (or whose chat
    has no remaining tasks)."""

    def _src(self) -> str:
        assert MAIN_JS.is_file(), f"main.js not found at {MAIN_JS}"
        return MAIN_JS.read_text()

    def _case_body(self, src: str, case_name: str) -> str:
        """Return the brace-balanced body of ``case '<name>': { ... }``.

        ``re.search`` with a non-greedy ``[\\s\\S]*?`` cannot extract
        a brace-balanced block when the body itself contains nested
        ``}`` characters.  This helper finds the opening ``{`` after
        the case label and walks forward counting braces.
        """
        m = re.search(
            r"case\s+'" + re.escape(case_name) + r"':\s*\{",
            src,
        )
        assert m, f"case '{case_name}' not found in main.js"
        start = m.end()
        depth = 1
        i = start
        while i < len(src) and depth > 0:
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        return src[start : i - 1]

    def test_render_adjacent_task_stamps_data_task_id(self) -> None:
        """``renderAdjacentTask`` must accept and stamp the row
        ``task_id`` onto the ``.adjacent-task`` container so the
        delete handler can find and remove specific blocks."""
        src = self._src()
        m = re.search(r"function\s+renderAdjacentTask\s*\(([^)]*)\)", src)
        assert m, "renderAdjacentTask not found in main.js"
        params = m.group(1)
        assert "taskId" in params, (
            "renderAdjacentTask must accept a taskId parameter"
        )
        m2 = re.search(
            r"function\s+renderAdjacentTask\s*\([^)]*\)\s*\{",
            src,
        )
        assert m2
        start = m2.end()
        depth = 1
        i = start
        while i < len(src) and depth > 0:
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        body = src[start:i]
        assert re.search(r"dataset\.taskId\s*=", body), (
            "renderAdjacentTask must set container.dataset.taskId"
        )

    def test_adjacent_task_events_handler_passes_task_id(self) -> None:
        """The 'adjacent_task_events' message handler must forward
        ``ev.task_id`` to ``renderAdjacentTask`` so the container
        is stamped with the row id."""
        src = self._src()
        m = re.search(
            r"case\s+'adjacent_task_events':[\s\S]*?renderAdjacentTask\s*\(([^)]*)\)",
            src,
        )
        assert m, "adjacent_task_events case must call renderAdjacentTask"
        args = m.group(1)
        assert "task_id" in args, (
            "adjacent_task_events handler must pass ev.task_id to "
            "renderAdjacentTask"
        )

    def test_task_events_handler_records_current_task_id(self) -> None:
        """The 'task_events' handler must record the resumed task's
        ``task_id`` on the tab object as ``currentTaskId`` so the
        delete handler can decide whether to close the tab."""
        src = self._src()
        body = self._case_body(src, "task_events")
        assert "currentTaskId" in body, (
            "task_events handler must set tab.currentTaskId from ev.task_id"
        )
        assert "task_id" in body, (
            "task_events handler must read ev.task_id"
        )

    def test_task_deleted_handler_exists_and_filters_by_chat(self) -> None:
        """A 'taskDeleted' message handler must exist and filter
        tabs by ``backendChatId``."""
        src = self._src()
        body = self._case_body(src, "taskDeleted")
        assert "backendChatId" in body, (
            "taskDeleted handler must filter tabs by backendChatId"
        )
        assert "chatId" in body, (
            "taskDeleted handler must read ev.chatId"
        )
        assert "taskId" in body, (
            "taskDeleted handler must read ev.taskId"
        )

    def test_task_deleted_handler_removes_adjacent_block(self) -> None:
        """The handler must remove ``.adjacent-task[data-task-id]``
        nodes from both the live output and saved fragments."""
        src = self._src()
        body = self._case_body(src, "taskDeleted")
        assert "adjacent-task" in body and "data-task-id" in body, (
            "taskDeleted handler must query "
            "'.adjacent-task[data-task-id=...]'"
        )
        assert "outputFragment" in body, (
            "taskDeleted handler must also prune detached "
            "outputFragment trees of inactive tabs"
        )

    def test_task_deleted_handler_closes_tab_when_current_or_empty(
        self,
    ) -> None:
        """The handler must close a tab whose currentTaskId equals
        the deleted id, or whose chat has no remaining tasks."""
        src = self._src()
        body = self._case_body(src, "taskDeleted")
        assert "currentTaskId" in body, (
            "taskDeleted handler must compare tab.currentTaskId to "
            "the deleted id to decide whether to close the tab"
        )
        assert "chatHasMoreTasks" in body, (
            "taskDeleted handler must check ev.chatHasMoreTasks to "
            "close tabs whose chat is now empty"
        )
        assert "closeTab" in body, (
            "taskDeleted handler must call closeTab() in those cases"
        )
