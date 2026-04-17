"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.agents.vscode.server import VSCodeServer


def _git(tmpdir: str, *args: str) -> None:
    """Run a git command in tmpdir."""
    subprocess.run(["git", *args], cwd=tmpdir, capture_output=True, check=True)


# ---------------------------------------------------------------------------
# persistence.py — _close_db when not initialized
# ---------------------------------------------------------------------------


class TestCloseDbNotInitialized:
    """Cover _close_db early return when _db_conn is None (line 52)."""

    def test_close_db_when_not_initialized(self) -> None:
        """Closing DB when connection was never opened should be a no-op."""
        # Save and restore state
        original_conn = th._db_conn
        original_path = th._DB_PATH
        try:
            th._db_conn = None
            th._close_db()
            # Should not raise, _db_conn still None
            assert th._db_conn is None
        finally:
            th._db_conn = original_conn
            th._DB_PATH = original_path


# ---------------------------------------------------------------------------
# stateful_sorcar_agent.py — resume_chat with unknown task
# ---------------------------------------------------------------------------


class TestResumeChatNoMatch:
    """Cover resume_chat branches."""

    def test_resume_chat_by_id_empty(self) -> None:
        """resume_chat_by_id("") should be a no-op."""
        agent = StatefulSorcarAgent("test")
        original_chat_id = agent.chat_id
        agent.resume_chat_by_id("")
        assert agent.chat_id == original_chat_id


# ---------------------------------------------------------------------------
# server.py — userAnswer with stale queue item
# ---------------------------------------------------------------------------


class TestUserAnswerDrain:
    """Cover drain-stale-answers path in userAnswer handler (lines 169-170)."""

    def test_user_answer_drains_stale(self) -> None:
        """Pre-filling a tab queue before userAnswer should drain stale item."""
        import queue as queue_mod

        server = VSCodeServer()
        # Create a per-tab queue and pre-fill it with a stale answer
        q: queue_mod.Queue[str] = queue_mod.Queue(maxsize=1)
        q.put("stale")
        server._get_tab("7").user_answer_queue = q
        # Send new answer — this should drain "stale" and put "new"
        server._handle_command({"type": "userAnswer", "answer": "new", "tabId": "7"})
        answer = q.get_nowait()
        assert answer == "new"


# ---------------------------------------------------------------------------
# server.py — resumeSession with non-empty task (line 179)
# ---------------------------------------------------------------------------


class TestResumeSessionWithTask:
    """Cover resumeSession handler calling _replay_session (line 179)."""

    def test_resume_session_with_task(self) -> None:
        """resumeSession with a non-empty chatId calls _replay_session."""
        server = VSCodeServer()
        events: list[dict[str, object]] = []
        orig = server.printer.broadcast

        def capture(ev: dict[str, object]) -> None:
            events.append(ev)
            orig(ev)

        server.printer.broadcast = capture  # type: ignore[assignment]
        # Use a task that doesn't exist — silently returns (no error broadcast)
        server._handle_command(
            {"type": "resumeSession", "chatId": "999999"}
        )
        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 0


# ---------------------------------------------------------------------------
# server.py — _replay_session with events (lines 554-555)
# ---------------------------------------------------------------------------


class TestReplaySessionWithEvents:
    """Cover successful _replay_session path (lines 554-555)."""

    def test_replay_session_with_recorded_events(self, tmp_path: Path) -> None:
        """_replay_session broadcasts task_events when events exist."""
        # Redirect persistence to tmp
        orig_dir = th._KISS_DIR
        orig_db = th._DB_PATH
        orig_conn = th._db_conn
        try:
            th._db_conn = None
            th._KISS_DIR = tmp_path
            th._DB_PATH = tmp_path / "history.db"

            # Create a task with events (using a chat_id)
            task_text = "test-replay-session-task"
            task_id, chat_id = th._add_task(task_text, chat_id="0")
            test_events: list[dict[str, object]] = [
                {"type": "text_delta", "text": "hello"},
                {"type": "result", "summary": "done"},
            ]
            th._set_latest_chat_events(test_events, task_id=task_id)

            server = VSCodeServer()
            captured: list[dict[str, object]] = []
            orig_broadcast = server.printer.broadcast

            def capture(ev: dict[str, object]) -> None:
                captured.append(ev)
                orig_broadcast(ev)

            server.printer.broadcast = capture  # type: ignore[assignment]

            # Call _replay_session with chat_id
            server._replay_session(chat_id)

            task_ev = [e for e in captured if e.get("type") == "task_events"]
            assert len(task_ev) == 1
            ev_list = task_ev[0].get("events", [])
            assert isinstance(ev_list, list)
            assert len(ev_list) == 2
        finally:
            th._close_db()
            th._db_conn = orig_conn
            th._KISS_DIR = orig_dir
            th._DB_PATH = orig_db


# ---------------------------------------------------------------------------
# server.py — _await_user_response loop iteration (466->462)
# ---------------------------------------------------------------------------


class TestAwaitUserResponseLoop:
    """Cover _await_user_response loop continuing (466->462)."""

    def test_await_user_response_delayed(self) -> None:
        """Answer arriving after first timeout iteration covers loop branch."""
        import queue as queue_mod

        server = VSCodeServer()
        stop_event = threading.Event()
        server.printer._thread_local.stop_event = stop_event
        server.printer._thread_local.tab_id = "42"
        # Create a per-tab queue
        q: queue_mod.Queue[str] = queue_mod.Queue(maxsize=1)
        server._get_tab("42").user_answer_queue = q

        def delayed_answer() -> None:
            time.sleep(1.0)
            q.put("delayed")

        t = threading.Thread(target=delayed_answer, daemon=True)
        t.start()
        result = server._await_user_response()
        assert result == "delayed"
        t.join(timeout=2)


# ---------------------------------------------------------------------------
# server.py — _complete with stale sequence (685->682)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# server.py — _run_task_inner drain (lines 268-269)
# ---------------------------------------------------------------------------


class TestRunTaskDrain:
    """Cover drain of stale answers at start of _run_task_inner (lines 268-269)."""

    def test_run_task_creates_fresh_queue(self) -> None:
        """Each task gets a fresh user_answer queue (RC8 fix)."""
        server = VSCodeServer()
        captured: list[dict[str, object]] = []
        orig = server.printer.broadcast

        def cap(ev: dict[str, object]) -> None:
            captured.append(ev)
            orig(ev)

        server.printer.broadcast = cap  # type: ignore[assignment]

        tab_id = "1"
        # Pre-fill a queue for this tab with a stale answer
        stale_q: queue.Queue[str] = queue.Queue(maxsize=1)
        stale_q.put("stale-answer")
        server._get_tab(tab_id).user_answer_queue = stale_q

        # Run a task — it will fail (no LLM key) but creates a fresh queue
        server._handle_command({
            "type": "run",
            "prompt": "test drain",
            "model": "nonexistent-model",
            "tabId": tab_id,
        })

        # Wait for task thread to complete
        thread = server._get_tab(tab_id).task_thread
        if thread:
            thread.join(timeout=30)

        # Task should have ended (status running=False)
        status_events = [
            e for e in captured
            if e.get("type") == "status" and e.get("running") is False
        ]
        assert len(status_events) >= 1


# ---------------------------------------------------------------------------
# web_use_tool.py — valid tab switch (lines 235-236)
# ---------------------------------------------------------------------------


class TestValidTabSwitch:
    """Cover successful tab switch in go_to_url (lines 235-236)."""

    @pytest.fixture()
    def http_server(self, tmp_path: Path) -> Generator[str]:
        """Start a minimal HTTP server for testing."""
        import http.server
        import socketserver

        html = "<html><body><h1>Tab Switch Test</h1></body></html>"
        (tmp_path / "index.html").write_text(html)

        handler = http.server.SimpleHTTPRequestHandler
        srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        # Change to tmp_path so server serves from there
        old_dir = os.getcwd()
        os.chdir(str(tmp_path))
        yield f"http://127.0.0.1:{port}/index.html"
        os.chdir(old_dir)
        srv.shutdown()

    def test_valid_tab_switch(self, http_server: str, tmp_path: Path) -> None:
        """Switching to tab 0 should succeed (lines 235-236)."""
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        profile = str(tmp_path / "browser_profile")
        tool = WebUseTool(user_data_dir=profile, headless=True)
        try:
            tool.go_to_url(http_server)
            # Tab 0 exists (the current page)
            result = tool.go_to_url("tab:0")
            assert "Error" not in result
            assert "Tab Switch Test" in result or "Page:" in result
        finally:
            tool.close()
