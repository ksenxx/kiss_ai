# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects
and real function calls.
"""

from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiss.server import agent_state
from kiss.server.json_printer import (
    JsonPrinter,
    _coalesce_events,
)
from kiss.server.server import VSCodeServer


class TestJsonPrinterBranches:
    """Cover uncovered branches in JsonPrinter."""

    def test_reset_clears_bash_buffer_and_timer(self):
        p = JsonPrinter()
        p._thread_local.task_id = "0"
        with p._bash_lock:
            bs = p._bash_state
            bs.buffer.append("some text")
            bs.timer = threading.Timer(10.0, lambda: None)
            bs.timer.start()
        p.reset()
        with p._bash_lock:
            bs = p._bash_state
            assert bs.buffer == []
            assert bs.timer is None

    def test_check_stop_thread_local(self):
        """_check_stop uses thread_local stop_event."""
        p = JsonPrinter()
        p._thread_local.stop_event = threading.Event()
        p._check_stop()
        p._thread_local.stop_event.set()
        with pytest.raises(KeyboardInterrupt):
            p._check_stop()

    def test_print_text_blank_no_broadcast(self):
        """Text that is only whitespace should not be broadcast."""
        p = JsonPrinter()
        p.start_recording()
        p.print("   ", type="text")
        events = p.stop_recording()
        assert len(events) == 0

    def test_token_callback_stop(self):
        p = JsonPrinter()
        p._thread_local.stop_event = threading.Event()
        p._thread_local.stop_event.set()
        with pytest.raises(KeyboardInterrupt):
            p.token_callback("x")


class TestCoalesceEventsBranches:

    def test_no_merge_non_delta_type(self):
        events = [
            {"type": "tool_call", "name": "Read"},
            {"type": "tool_call", "name": "Write"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2


class TestVSCodeServerBranches:
    """Cover uncovered branches in VSCodeServer."""

    def _make_server(self):
        server = VSCodeServer()
        events: list[dict] = []
        def capture(event):
            events.append(event)
        server.printer.broadcast = capture  # type: ignore[assignment]
        return server, events

    def test_handle_command_run_already_running(self):
        """A run command while a task is alive queues the prompt (steering)."""
        server, events = self._make_server()
        t = threading.Thread(target=lambda: time.sleep(5), daemon=True)
        t.start()
        st = agent_state.AgentState(
            "w2-run-busy", tab_id="0", server_owned=True, task_thread=t,
        )
        agent_state.register(st)
        try:
            server._handle_command(
                {"type": "run", "prompt": "test", "tabId": "0"}
            )
            assert not any(e.get("type") == "error" for e in events)
            assert st.pending_user_messages == ["test"]
        finally:
            agent_state.unregister(st.task_id, st)
        t.join(timeout=0.1)

    def test_handle_command_stop_no_event(self):
        server, events = self._make_server()
        server._handle_command({"type": "stop"})

    def test_handle_command_get_history_with_query(self):
        server, events = self._make_server()
        server._handle_command({"type": "getHistory", "query": "test"})
        hist_events = [e for e in events if e["type"] == "history"]
        assert len(hist_events) == 1

    def test_handle_command_record_file_usage_empty(self):
        server, events = self._make_server()
        server._handle_command({"type": "recordFileUsage", "path": ""})

    def test_handle_command_resume_session(self):
        server, events = self._make_server()
        server._handle_command({"type": "resumeSession", "chatId": ""})

    def test_ask_user_question(self):
        server, events = self._make_server()
        stop_event = threading.Event()
        server.printer._thread_local.stop_event = stop_event
        task_id = "1"
        server.printer._thread_local.task_id = task_id
        user_q: queue.Queue[str] = queue.Queue(maxsize=1)
        st = agent_state.AgentState(
            task_id, tab_id=task_id, server_owned=True,
        )
        st.user_answer_queue = user_q
        agent_state.register(st)
        server.printer.subscribe_tab(task_id, task_id)
        try:
            def answer():
                time.sleep(0.1)
                user_q.put("my answer")

            t = threading.Thread(target=answer, daemon=True)
            t.start()
            result = server._ask_user_question("what?")
            t.join(timeout=1)
        finally:
            agent_state.unregister(task_id, st)
        assert result == "my answer"
        ask_events = [e for e in events if e["type"] == "askUser"]
        assert len(ask_events) == 1

    def test_emit_pending_worktree_not_a_repo(self, tmp_path):
        """_emit_pending_worktree does nothing when not in a git repo."""
        server, events = self._make_server()
        server.work_dir = str(tmp_path)
        server._emit_pending_worktree()
        wt_events = [e for e in events if e.get("type") == "worktree_done"]
        assert len(wt_events) == 0

    def test_handle_command_generate_commit_message_routing(self):
        """generateCommitMessage is routed properly - routes to thread."""
        with tempfile.TemporaryDirectory() as d:
            repo = os.path.join(d, "repo")
            os.makedirs(repo)
            subprocess.run(["git", "init"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=repo, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True)
            Path(repo, "f.txt").write_text("content\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
            server, events = self._make_server()
            server.work_dir = repo
            server._handle_command({"type": "generateCommitMessage"})
            time.sleep(1)
            commit_events = [e for e in events if e["type"] == "commitMessage"]
            assert len(commit_events) == 1
            assert commit_events[0]["message"] == ""
            assert "No staged changes" in commit_events[0]["error"]


class TestHandleMessageContentBlockNoIsError:
    """Cover the case where content block lacks is_error/content attributes."""

    def test_block_without_is_error(self):
        p = JsonPrinter()
        p.start_recording()
        block = SimpleNamespace(some_other_attr="value")
        msg = SimpleNamespace(content=[block])
        p._handle_message(msg)
        events = p.stop_recording()
        assert len(events) == 0


class TestVSCodeServerMoreBranches:
    def _make_server(self):
        server = VSCodeServer()
        events: list[dict] = []
        def capture(event):
            events.append(event)
        server.printer.broadcast = capture  # type: ignore[assignment]
        return server, events

    def test_handle_command_get_files(self):
        server, events = self._make_server()
        server._handle_command({"type": "getFiles", "prefix": ""})
        file_events = [e for e in events if e["type"] == "files"]
        assert len(file_events) == 1

    def test_run_task_with_attachments(self):
        """Test _run_task processes attachments."""
        import base64

        server, events = self._make_server()
        png_data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
        pdf_data = base64.b64encode(b"%PDF-1.4 fake").decode()

        with tempfile.TemporaryDirectory() as d:
            repo = os.path.join(d, "repo")
            os.makedirs(repo)
            subprocess.run(["git", "init"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=repo, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True)
            Path(repo, "f.txt").write_text("x")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)

            server.work_dir = repo
            server._run_task({
                "prompt": "test task",
                "model": "claude-opus-4-6",
                "workDir": repo,
                "activeFile": "/tmp/test.py",
                "attachments": [
                    {"data": png_data, "mimeType": "image/png"},
                    {"data": pdf_data, "mimeType": "application/pdf"},
                ],
            })
            types = [e.get("type") for e in events]
            assert "status" in types
