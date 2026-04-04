"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.agents.vscode.diff_merge import (
    _diff_files,
    _prepare_merge_view,
    _save_untracked_base,
    _snapshot_files,
)
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
        """resume_chat_by_id("") should be a no-op (branch 73->exit)."""
        agent = StatefulSorcarAgent("test")
        original_chat_id = agent.chat_id
        agent.resume_chat_by_id("")
        assert agent.chat_id == original_chat_id


# ---------------------------------------------------------------------------
# diff_merge.py — _diff_files uncovered branches
# ---------------------------------------------------------------------------


class TestDiffFilesBranches:
    """Cover all branches in _diff_files."""

    def test_nonexistent_base_file(self) -> None:
        """_diff_files with missing base file → base_lines=[] (line 238)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            current = Path(tmpdir) / "current.txt"
            current.write_text("line1\nline2\n")
            hunks = _diff_files("/nonexistent/base.txt", str(current))
            # Should produce an insertion hunk
            assert len(hunks) >= 1
            # Pure insertion: old_count should be 0
            assert hunks[0][1] == 0

    def test_nonexistent_current_file(self) -> None:
        """_diff_files with missing current file → current_lines=[] (lines 241-242)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base.txt"
            base.write_text("line1\nline2\n")
            hunks = _diff_files(str(base), "/nonexistent/current.txt")
            # Should produce a deletion hunk
            assert len(hunks) >= 1
            # Pure deletion: new_count should be 0
            assert hunks[0][3] == 0

# ---------------------------------------------------------------------------
# diff_merge.py — _prepare_merge_view untracked file branches
# ---------------------------------------------------------------------------


class TestPrepareMergeViewUntrackedBranches:
    """Cover untracked file handling in _prepare_merge_view."""

    def test_untracked_file_not_changed(self) -> None:
        """Pre-existing untracked file that wasn't changed → continue (line 384)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _git(tmpdir, "init")
            _git(tmpdir, "config", "user.email", "t@t.com")
            _git(tmpdir, "config", "user.name", "T")
            # Create a tracked file and commit
            Path(tmpdir, "tracked.txt").write_text("tracked")
            _git(tmpdir, "add", "tracked.txt")
            _git(tmpdir, "commit", "-m", "init")
            # Create an untracked file
            Path(tmpdir, "untracked.txt").write_text("untracked content")

            data_dir = os.path.join(tmpdir, ".kiss.artifacts", "merge_dir")
            pre_untracked = {"untracked.txt"}
            pre_hashes = _snapshot_files(tmpdir, pre_untracked)

            # File not changed → _file_changed returns False → continue (line 384)
            result = _prepare_merge_view(
                tmpdir, data_dir, {}, pre_untracked, pre_hashes
            )
            # No changes because untracked file wasn't modified
            assert result.get("error") == "No changes"

    def test_untracked_file_changed_empty_diff(self) -> None:
        """Pre-existing untracked file changed but diff produces no hunks → 386->380."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _git(tmpdir, "init")
            _git(tmpdir, "config", "user.email", "t@t.com")
            _git(tmpdir, "config", "user.name", "T")
            Path(tmpdir, "tracked.txt").write_text("tracked")
            _git(tmpdir, "add", "tracked.txt")
            _git(tmpdir, "commit", "-m", "init")
            # Create untracked file with content
            Path(tmpdir, "untracked.txt").write_text("content")

            # Save base before task
            _save_untracked_base(tmpdir, {"untracked.txt"})
            pre_untracked = {"untracked.txt"}
            pre_hashes = _snapshot_files(tmpdir, pre_untracked)

            # Now change the hash to pretend it changed, but the actual content
            # is the same as the saved base → diff produces 0 hunks
            # Force a different hash to trigger _file_changed=True
            pre_hashes["untracked.txt"] = "0000000000000000"

            data_dir = os.path.join(tmpdir, ".kiss.artifacts", "merge_dir")
            result = _prepare_merge_view(
                tmpdir, data_dir, {}, pre_untracked, pre_hashes
            )
            # The file is "changed" (hash mismatch) but diff against saved base
            # produces 0 hunks since content is identical. Result: no changes
            assert result.get("error") == "No changes"


# ---------------------------------------------------------------------------
# server.py — userAnswer with stale queue item
# ---------------------------------------------------------------------------


class TestUserAnswerDrain:
    """Cover drain-stale-answers path in userAnswer handler (lines 169-170)."""

    def test_user_answer_drains_stale(self) -> None:
        """Pre-filling queue before userAnswer should drain stale item."""
        server = VSCodeServer()
        # Pre-fill with a stale answer
        server._user_answer_queue.put("stale")
        # Send new answer — this should drain "stale" and put "new"
        server._handle_command({"type": "userAnswer", "answer": "new"})
        answer = server._user_answer_queue.get_nowait()
        assert answer == "new"


# ---------------------------------------------------------------------------
# server.py — resumeSession with non-empty task (line 179)
# ---------------------------------------------------------------------------


class TestResumeSessionWithTask:
    """Cover resumeSession handler calling _replay_session (line 179)."""

    def test_resume_session_with_task(self) -> None:
        """resumeSession with a non-empty sessionId calls _replay_session."""
        server = VSCodeServer()
        events: list[dict[str, object]] = []
        orig = server.printer.broadcast

        def capture(ev: dict[str, object]) -> None:
            events.append(ev)
            orig(ev)

        server.printer.broadcast = capture  # type: ignore[assignment]
        # Use a task that doesn't exist — will trigger "No recorded events" error
        server._handle_command(
            {"type": "resumeSession", "sessionId": "nonexistent-task-999"}
        )
        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 1
        assert "No recorded events" in str(err[0].get("text", ""))


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

            # Create a task with events
            task_text = "test-replay-session-task"
            task_id = th._add_task(task_text)
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

            # Call _replay_session — should find events and broadcast them
            server._replay_session(task_text)

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
        server = VSCodeServer()
        server._stop_event = threading.Event()
        server.printer._thread_local.stop_event = server._stop_event

        def delayed_answer() -> None:
            time.sleep(1.0)
            server._user_answer_queue.put("delayed")

        t = threading.Thread(target=delayed_answer, daemon=True)
        t.start()
        result = server._await_user_response()
        assert result == "delayed"
        t.join(timeout=2)


# ---------------------------------------------------------------------------
# server.py — _complete with stale sequence (685->682)
# ---------------------------------------------------------------------------


class TestCompleteFromActiveFileShorterSuffix:
    """Cover branch 685->682: shorter suffix not replacing best."""

    def test_multiple_candidates_shorter_not_chosen(self) -> None:
        """When multiple candidates match, shorter suffix doesn't replace best."""
        server = VSCodeServer()
        # Content has both 'method_long_name' and 'method_l' — partial is 'method_'
        # 'method_long_name' gives suffix 'long_name' (len=9)
        # 'method_l' gives suffix 'l' (len=1) — shorter, should NOT replace best
        content = "method_long_name method_l other_stuff"
        result = server._complete_from_active_file(
            "call method_", snapshot_content=content
        )
        # Should pick the longest suffix
        assert result == "long_name"


# ---------------------------------------------------------------------------
# server.py — _run_task_inner drain (lines 268-269)
# ---------------------------------------------------------------------------


class TestRunTaskDrain:
    """Cover drain of stale answers at start of _run_task_inner (lines 268-269)."""

    def test_run_task_drains_stale_answers(self) -> None:
        """Pre-filled queue is drained when task starts."""
        server = VSCodeServer()
        captured: list[dict[str, object]] = []
        orig = server.printer.broadcast

        def cap(ev: dict[str, object]) -> None:
            captured.append(ev)
            orig(ev)

        server.printer.broadcast = cap  # type: ignore[assignment]

        # Pre-fill queue with stale answer
        server._user_answer_queue.put("stale-answer")

        # Run a task — it will fail (no LLM key) but drain happens first
        server._handle_command({
            "type": "run",
            "prompt": "test drain",
            "model": "nonexistent-model",
        })

        # Wait for task thread to complete
        if server._task_thread:
            server._task_thread.join(timeout=30)

        # Queue should be empty (stale was drained)
        assert server._user_answer_queue.empty()

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
        tool = WebUseTool(user_data_dir=profile)
        try:
            tool.go_to_url(http_server)
            # Tab 0 exists (the current page)
            result = tool.go_to_url("tab:0")
            assert "Error" not in result
            assert "Tab Switch Test" in result or "Page:" in result
        finally:
            tool.close()
