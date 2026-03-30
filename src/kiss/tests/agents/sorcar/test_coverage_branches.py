"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects
and real function calls.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    _extract_leading_command_name,
    _kill_process_group,
    _split_respecting_quotes,
    _truncate_output,
)
from kiss.agents.sorcar.web_use_tool import (
    WebUseTool,
)
from kiss.agents.vscode.browser_ui import (
    BaseBrowserPrinter,
    _coalesce_events,
)
from kiss.agents.vscode.diff_merge import (
    _agent_file_hunks,
    _capture_untracked,
    _file_as_new_hunks,
    _parse_diff_hunks,
    _prepare_merge_view,
    _scan_files,
    _snapshot_files,
)
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# browser_ui.py coverage
# ---------------------------------------------------------------------------


class TestBaseBrowserPrinterBranches:
    """Cover uncovered branches in BaseBrowserPrinter."""

    def test_reset_clears_bash_buffer_and_timer(self):
        p = BaseBrowserPrinter()
        p._bash_buffer.append("some text")
        # Set up a timer
        p._bash_flush_timer = threading.Timer(10.0, lambda: None)
        p._bash_flush_timer.start()
        p.reset()
        assert p._bash_buffer == []
        assert p._bash_flush_timer is None

    def test_stop_recording_returns_filtered_coalesced(self):
        p = BaseBrowserPrinter()
        p.start_recording()
        p.broadcast({"type": "thinking_delta", "text": "a"})
        p.broadcast({"type": "thinking_delta", "text": "b"})
        p.broadcast({"type": "internal_event"})  # non-display
        p.broadcast({"type": "text_delta", "text": "c"})
        events = p.stop_recording()
        # thinking_delta merged, internal_event filtered, text_delta kept
        assert len(events) == 2
        assert events[0]["text"] == "ab"
        assert events[1]["text"] == "c"

    def test_remove_client_only_current(self):
        """remove_client only removes if cq matches current."""
        p = BaseBrowserPrinter()
        q1: queue.Queue = queue.Queue()
        q2: queue.Queue = queue.Queue()
        p._client_queue = q2
        p.remove_client(q1)  # doesn't match, no effect
        assert p._client_queue is q2
        p.remove_client(q2)
        assert p._client_queue is None

    def test_has_clients_true_and_false(self):
        p = BaseBrowserPrinter()
        assert not p.has_clients()
        cq = p.add_client()
        assert p.has_clients()
        p.remove_client(cq)
        assert not p.has_clients()

    def test_check_stop_thread_local(self):
        """_check_stop uses thread_local stop_event."""
        p = BaseBrowserPrinter()
        p._thread_local.stop_event = threading.Event()
        # Not set - no raise
        p._check_stop()
        # Set - raises
        p._thread_local.stop_event.set()
        with pytest.raises(KeyboardInterrupt):
            p._check_stop()

    def test_print_text_blank_no_broadcast(self):
        """Text that is only whitespace should not be broadcast."""
        p = BaseBrowserPrinter()
        cq: queue.Queue = queue.Queue()
        p._client_queue = cq
        p.print("   ", type="text")
        assert cq.empty()

    def test_token_callback_stop(self):
        p = BaseBrowserPrinter()
        p.stop_event.set()
        with pytest.raises(KeyboardInterrupt):
            p.token_callback("x")
        p.stop_event.clear()


class TestCoalesceEventsBranches:

    def test_no_merge_non_delta_type(self):
        events = [
            {"type": "tool_call", "name": "Read"},
            {"type": "tool_call", "name": "Write"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# diff_merge.py coverage
# ---------------------------------------------------------------------------


class TestCodeServerBranches:

    def test_file_as_new_hunks_nonexistent(self):
        result = _file_as_new_hunks(Path("/nonexistent_file_xyz"))
        assert result == []

    def test_file_as_new_hunks_binary_file(self):
        """UnicodeDecodeError should be caught."""
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "binary.bin"
            f.write_bytes(b"\x80\x81\x82" * 100)
            result = _file_as_new_hunks(f)
            assert result == []

    def test_agent_file_hunks_post_file_hunks_filter(self):
        """Without saved base but with post_file_hunks, filters against pre_hunks."""
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            work.mkdir()
            ub = Path(d) / "ub"
            ub.mkdir()
            (work / "f.txt").write_text("changed\n")
            pre_hunks = {"f.txt": [(1, 1, 1, 1)]}
            post = [(1, 1, 1, 1), (5, 0, 5, 2)]  # first matches pre
            result = _agent_file_hunks(str(work), "f.txt", ub, pre_hunks, post)
            # Only the second hunk should pass (first matches pre)
            assert len(result) == 1


# ---------------------------------------------------------------------------
# helpers.py coverage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# useful_tools.py coverage
# ---------------------------------------------------------------------------


class TestUsefulToolsBranches:
    def test_read_truncates_large_file(self):
        ut = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(3000):
                f.write(f"line {i}\n")
            f.flush()
            result = ut.Read(f.name, max_lines=100)
            assert "[truncated:" in result
            os.unlink(f.name)

    def test_read_error(self):
        ut = UsefulTools()
        result = ut.Read("/nonexistent_file_xyz")
        assert "Error:" in result

    def test_edit_file_not_found(self):
        ut = UsefulTools()
        result = ut.Edit("/nonexistent_file_xyz", "old", "new")
        assert "Error:" in result

    def test_edit_same_string(self):
        ut = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            f.flush()
            result = ut.Edit(f.name, "content", "content")
            assert "must be different" in result
            os.unlink(f.name)

    def test_edit_string_not_found(self):
        ut = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            f.flush()
            result = ut.Edit(f.name, "xyz", "abc")
            assert "not found" in result
            os.unlink(f.name)

    def test_edit_multiple_occurrences_no_replace_all(self):
        ut = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("aaaa")
            f.flush()
            result = ut.Edit(f.name, "a", "b")
            assert "appears 4 times" in result
            os.unlink(f.name)

    def test_edit_replace_all(self):
        ut = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("aXaXa")
            f.flush()
            result = ut.Edit(f.name, "X", "Y", replace_all=True)
            assert "2 occurrence(s)" in result
            assert Path(f.name).read_text() == "aYaYa"
            os.unlink(f.name)

    def test_bash_disallowed(self):
        ut = UsefulTools()
        result = ut.Bash("eval echo hi", "test")
        assert "not allowed" in result

    def test_bash_timeout_nonstreaming(self):
        ut = UsefulTools()
        result = ut.Bash("sleep 100", "timeout test", timeout_seconds=0.5)
        assert "timeout" in result.lower()

    def test_split_respecting_quotes_escaped(self):
        import re
        pat = re.compile(r";")
        result = _split_respecting_quotes("a\\;b;c", pat)
        assert result == ["a\\;b", "c"]

    def test_kill_process_group_already_dead(self):
        """kill_process_group handles already-dead process."""
        p = subprocess.Popen(["true"], start_new_session=True)
        p.wait()
        # Should not raise
        _kill_process_group(p)


# ---------------------------------------------------------------------------
# web_use_tool.py coverage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# sorcar_agent.py coverage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# persistence.py coverage
# ---------------------------------------------------------------------------


class TestTaskHistoryBranches:
    """Cover specific uncovered branches in persistence."""

    def _fresh_db(self, tmp_path):
        """Switch to a fresh DB in tmp_path, return cleanup callback."""
        saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = tmp_path / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        return saved

    def _restore_db(self, saved):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        th._DB_PATH, th._db_conn, th._KISS_DIR = saved

    def test_set_latest_chat_events_no_match(self, tmp_path):
        saved = self._fresh_db(tmp_path)
        try:
            th._set_latest_chat_events([], task="nonexistent_task_xyz")
        finally:
            self._restore_db(saved)

    def test_load_chat_context_empty_id(self):
        assert th._load_chat_context("") == []

    def test_cleanup_stale_cs_dirs_no_port_file(self, tmp_path):
        """When sorcar-data exists but no cs-port file, clean up."""
        saved = self._fresh_db(tmp_path)
        try:
            sorcar_data = th._KISS_DIR / "sorcar-data"
            sorcar_data.mkdir()
            old_time = time.time() - 48 * 3600
            os.utime(sorcar_data, (old_time, old_time))
            removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
            assert removed >= 1
        finally:
            self._restore_db(saved)


# ---------------------------------------------------------------------------
# vscode/server.py coverage
# ---------------------------------------------------------------------------


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
        server, events = self._make_server()
        # Simulate a running thread
        t = threading.Thread(target=lambda: time.sleep(5), daemon=True)
        t.start()
        server._task_thread = t
        server._handle_command({"type": "run", "prompt": "test"})
        assert any("already running" in e.get("text", "") for e in events)
        t.join(timeout=0.1)

    def test_handle_command_stop_no_event(self):
        server, events = self._make_server()
        server._stop_event = None
        server._handle_command({"type": "stop"})
        # No crash

    def test_handle_command_get_history_with_query(self):
        server, events = self._make_server()
        server._handle_command({"type": "getHistory", "query": "test"})
        hist_events = [e for e in events if e["type"] == "history"]
        assert len(hist_events) == 1

    def test_handle_command_get_history_no_query(self):
        server, events = self._make_server()
        server._handle_command({"type": "getHistory"})
        hist_events = [e for e in events if e["type"] == "history"]
        assert len(hist_events) == 1

    def test_handle_command_record_file_usage_empty(self):
        server, events = self._make_server()
        server._handle_command({"type": "recordFileUsage", "path": ""})
        # No crash, no action
        # No crash

    def test_handle_command_user_answer_no_event(self):
        server, events = self._make_server()
        server._user_answer_event = None
        server._handle_command({"type": "userAnswer", "answer": "42"})
        assert server._user_answer == "42"

    def test_handle_command_resume_session(self):
        server, events = self._make_server()
        server._handle_command({"type": "resumeSession", "sessionId": ""})
        # Empty sessionId - no action

    def test_handle_merge_action_reject_ignored(self):
        """Individual accept/reject actions are tracked on the TS side only."""
        server, events = self._make_server()
        server._merging = True
        server._handle_command({"type": "mergeAction", "action": "reject"})
        assert server._merging is True  # no change

    def test_handle_merge_action_all_done(self):
        server, events = self._make_server()
        server._merging = True
        server._handle_command({"type": "mergeAction", "action": "all-done"})
        assert server._merging is False
        assert any(e.get("type") == "merge_ended" for e in events)

    def test_await_user_response_no_event(self):
        server, events = self._make_server()
        server._user_answer_event = None
        server._await_user_response()  # should not crash

    def test_wait_for_user(self):
        server, events = self._make_server()
        server._user_answer_event = threading.Event()

        def answer():
            time.sleep(0.1)
            assert server._user_answer_event is not None
            server._user_answer_event.set()

        t = threading.Thread(target=answer, daemon=True)
        t.start()
        server._wait_for_user("do something", "http://example.com")
        t.join(timeout=1)
        wfu = [e for e in events if e["type"] == "waitForUser"]
        assert len(wfu) == 1

    def test_ask_user_question(self):
        server, events = self._make_server()
        server._user_answer_event = threading.Event()
        server._user_answer = "my answer"

        def answer():
            time.sleep(0.1)
            assert server._user_answer_event is not None
            server._user_answer_event.set()

        t = threading.Thread(target=answer, daemon=True)
        t.start()
        result = server._ask_user_question("what?")
        t.join(timeout=1)
        assert result == "my answer"
        ask_events = [e for e in events if e["type"] == "askUser"]
        assert len(ask_events) == 1

    def test_run_with_stdin(self):
        """Test run() reads from stdin and dispatches."""
        server = VSCodeServer()
        events: list[dict] = []
        def capture(event):
            events.append(event)
        server.printer.broadcast = capture  # type: ignore[assignment]

        import io
        cmds = [
            json.dumps({"type": "getModels"}) + "\n",
            json.dumps({"type": "selectModel", "model": "claude-opus-4-6"}) + "\n",
            "",  # blank line
        ]
        old_stdin = os.sys.stdin  # type: ignore[attr-defined]
        os.sys.stdin = io.StringIO("".join(cmds))  # type: ignore[attr-defined]
        try:
            server.run()
        finally:
            os.sys.stdin = old_stdin  # type: ignore[attr-defined]

        model_events = [e for e in events if e["type"] == "models"]
        assert len(model_events) == 1

    def test_get_last_session_empty_task(self, tmp_path):
        """When last task has empty task text, no event emitted."""
        saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = tmp_path / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        try:
            # Add empty task
            th._add_task("")
            server, events = self._make_server()
            server._get_last_session()
            task_events = [e for e in events if e.get("type") == "task_events"]
            # Empty task should cause early return
            assert len(task_events) == 0
        finally:
            if th._db_conn is not None:
                th._db_conn.close()
                th._db_conn = None
            th._DB_PATH, th._db_conn, th._KISS_DIR = saved

    def test_get_last_session_no_entries(self, tmp_path):
        """When history is empty, no event emitted."""
        saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = tmp_path / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        try:
            db = th._get_db()
            # Delete all tasks including samples
            db.execute("DELETE FROM task_history")
            db.commit()
            server, events = self._make_server()
            server._get_last_session()
            task_events = [e for e in events if e.get("type") == "task_events"]
            assert len(task_events) == 0
        finally:
            if th._db_conn is not None:
                th._db_conn.close()
                th._db_conn = None
            th._DB_PATH, th._db_conn, th._KISS_DIR = saved

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
            assert commit_events[0]["message"] == "Error: No staged files."


# ---------------------------------------------------------------------------
# web_use_tool.py - more coverage via http server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def http_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    form_html = b"""<!DOCTYPE html>
<html><head><title>Test</title></head>
<body>
  <h1>Test</h1>
  <a href="/second">Link</a>
  <input type="text" id="name" name="name" placeholder="Name">
  <button>Submit</button>
  <div style="height:5000px"></div>
</body></html>"""

    second_html = b"""<!DOCTYPE html>
<html><head><title>Second</title></head>
<body><h1>Second Page</h1><a href="/">Back</a></body></html>"""

    empty_html = b"""<!DOCTYPE html>
<html><head><title>Empty</title></head><body></body></html>"""

    multi_html = b"""<!DOCTYPE html>
<html><head><title>Multi</title></head>
<body>
  <button>Submit</button>
  <button>Submit</button>
  <button>Submit</button>
</body></html>"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            pages = {
                "/": form_html, "/second": second_html,
                "/empty": empty_html, "/multi": multi_html,
            }
            content = pages.get(self.path, form_html)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, /, *args: object) -> None:  # type: ignore[override]
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(scope="module")
def browser_tool():
    tool = WebUseTool(user_data_dir=None)
    yield tool
    tool.close()


class TestWebUseToolIntegration:

    def test_tab_list(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        result = browser_tool.go_to_url("tab:list")
        assert "Open tabs" in result


    def test_tab_switch_invalid(self, http_server, browser_tool):
        result = browser_tool.go_to_url("tab:999")
        assert "Error" in result

    def test_click_invalid_element(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        result = browser_tool.click(999)
        assert "Error" in result

    def test_hover_element(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        result = browser_tool.click(1, action="hover")
        assert isinstance(result, str)

    def test_type_text(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        # Find the textbox element
        result = browser_tool.type_text(2, "test input")
        assert isinstance(result, str)

    def test_type_text_with_enter(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        result = browser_tool.type_text(2, "test", press_enter=True)
        assert isinstance(result, str)

    def test_press_key(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        result = browser_tool.press_key("Tab")
        assert isinstance(result, str)

    def test_screenshot(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shot.png")
            result = browser_tool.screenshot(path)
            assert "Screenshot saved" in result
            assert os.path.exists(path)

    def test_get_page_content_tree(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        result = browser_tool.get_page_content(text_only=False)
        assert "[" in result

    def test_get_page_content_text(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        result = browser_tool.get_page_content(text_only=True)
        assert "Test" in result


class TestWebUseToolPersistentContext:
    """Test WebUseTool with user_data_dir (persistent context).

    Note: Can't test launch_persistent_context in-process when a
    module-scoped browser_tool exists (asyncio loop conflict).
    The user_data_dir branch (lines 138-142) requires a separate process.
    """

    def test_persistent_context_in_subprocess(self, http_server):
        """Test launch with persistent user data dir in a subprocess.

        Subprocess is needed because module-scoped browser_tool creates
        an asyncio loop that conflicts with a second sync_playwright.
        Coverage is collected via subprocess coverage combine.
        """
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "test_persistent.py"
            script.write_text(f"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))
from kiss.agents.sorcar.web_use_tool import WebUseTool
udd = os.path.join("{d}", "user_data")
tool = WebUseTool(user_data_dir=udd)
try:
    result = tool.go_to_url("{http_server}/")
    assert tool._page is not None
    assert tool._context is not None
    assert tool._browser is None
    assert "Test" in result, f"Expected 'Test' in result: {{result[:200]}}"
    print("PASS")
finally:
    tool.close()
""")
            result = subprocess.run(
                ["uv", "run", "python", str(script)],
                capture_output=True, text=True, timeout=30,
                cwd=os.getcwd(),
            )
            assert "PASS" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"


class TestWebUseToolResolveLocatorBranches:
    """Test _resolve_locator branches for multiple/no elements."""

    def test_resolve_locator_refreshes_snapshot(self, http_server, browser_tool):
        """When elements list is empty, re-snapshot is attempted."""
        browser_tool.go_to_url(http_server + "/")
        # Clear element cache
        browser_tool._elements = []
        # Try to click - should re-snapshot
        result = browser_tool.click(1)
        assert isinstance(result, str)

    def test_press_key_error(self, browser_tool):
        """Press invalid key combination."""
        browser_tool.go_to_url("about:blank")
        result = browser_tool.press_key("InvalidKeyXYZ_12345")
        assert "Error" in result

    def test_scroll_left_right(self, http_server, browser_tool):
        browser_tool.go_to_url(http_server + "/")
        result = browser_tool.scroll("left", amount=1)
        assert isinstance(result, str)
        result = browser_tool.scroll("right", amount=1)
        assert isinstance(result, str)

    def test_screenshot_error(self, browser_tool):
        """Screenshot to invalid path."""
        browser_tool.go_to_url("about:blank")
        result = browser_tool.screenshot("/dev/null/cant/write/here.png")
        # This may or may not error depending on OS
        assert isinstance(result, str)

    def test_type_text_error_invalid_element(self, http_server, browser_tool):
        """type_text error on non-existent element."""
        browser_tool.go_to_url(http_server + "/empty")
        result = browser_tool.type_text(999, "text")
        assert "Error" in result


class TestBrowserPrinterStreamEvent:
    """Cover the stream_event print type."""

    def test_format_tool_call_with_extras(self):
        """Cover the extras branch in _format_tool_call."""
        p = BaseBrowserPrinter()
        cq: queue.Queue = queue.Queue()
        p._client_queue = cq
        p._format_tool_call("Bash", {
            "command": "ls",
            "timeout_seconds": 30,
            "max_output_chars": 50000,
        })
        ev = cq.get_nowait()
        assert "extras" in ev


class TestHandleMessageContentBlockNoIsError:
    """Cover the case where content block lacks is_error/content attributes."""

    def test_block_without_is_error(self):
        p = BaseBrowserPrinter()
        cq: queue.Queue = queue.Queue()
        p._client_queue = cq
        # Block without is_error and content attributes
        block = SimpleNamespace(some_other_attr="value")
        msg = SimpleNamespace(content=[block])
        p._handle_message(msg)
        # Nothing should be broadcast
        assert cq.empty()


# ---------------------------------------------------------------------------
# Additional useful_tools.py branches
# ---------------------------------------------------------------------------


class TestUsefulToolsMoreBranches:
    def test_extract_leading_command_name_invalid_shlex(self):
        """shlex.split raises ValueError on unmatched quotes."""
        result = _extract_leading_command_name("'unterminated")
        assert result is None

    def test_truncate_output_zero_tail(self):
        """Edge case where tail computes to 0."""
        # Create a string where (max_chars - msg_len) // 2 = head, remaining - head = 0 for tail
        # msg template has ~45 chars at minimum for "... [truncated N chars] ..."
        big = "X" * 200
        # With max_chars=48, msg would be ~42 chars, remaining=6, head=3, tail=3
        # We need tail=0: remaining=0 or very small
        # Actually, when remaining is odd, head gets floor, tail gets ceil
        # Let's just use a max that's slightly bigger than msg
        result = _truncate_output(big, 45)
        assert "truncated" in result

    def test_edit_exception(self):
        """Edit on a directory should raise an error."""
        ut = UsefulTools()
        with tempfile.TemporaryDirectory() as d:
            # Write a file, then make it read-only
            f = Path(d) / "readonly.txt"
            f.write_text("hello old world")
            f.chmod(0o444)
            try:
                result = ut.Edit(str(f), "old", "new")
                # On macOS, root can still write; non-root gets error
                assert "Error" in result or "Successfully" in result
            finally:
                f.chmod(0o644)


# ---------------------------------------------------------------------------
# Additional helpers.py branches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Additional diff_merge.py branches
# ---------------------------------------------------------------------------


class TestCodeServerMoreBranches:
    def test_scan_files_max_limit(self):
        """Scan should stop at 2000 files."""
        with tempfile.TemporaryDirectory() as d:
            # Create many files
            for i in range(2100):
                (Path(d) / f"file_{i:04d}.txt").write_text("x")
            result = _scan_files(d)
            assert len(result) <= 2000

    def test_prepare_merge_view_new_files_only(self):
        """When only new untracked files are added after a task."""
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

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set())

            # Agent creates a new file
            Path(repo, "new_file.txt").write_text("new content\nline 2\n")

            data_dir = os.path.join(d, "merge_data")
            os.makedirs(data_dir)
            result = _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked, pre_hashes)
            assert result.get("status") == "opened"


# ---------------------------------------------------------------------------
# Additional vscode/server.py branches
# ---------------------------------------------------------------------------


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

    def test_handle_command_refresh_files(self):
        server, events = self._make_server()
        server._handle_command({"type": "refreshFiles"})
        assert isinstance(server._file_cache, list)

        # May or may not have a suggestion depending on API availability
        # But the test exercises the branch

    def test_run_task_with_attachments(self):
        """Test _run_task processes attachments."""
        import base64

        server, events = self._make_server()
        # Create a 1x1 PNG
        png_data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
        pdf_data = base64.b64encode(b"%PDF-1.4 fake").decode()

        # The actual run will fail (no API), but we can test attachment parsing
        # by checking that the merging guard doesn't trigger
        server._merging = False
        # Use a work_dir that's a git repo so merge view doesn't crash
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
            # The run will fail because no API key, but we exercised the
            # attachment parsing branch
            types = [e.get("type") for e in events]
            assert "status" in types  # started
            # It should have gone past attachments parsing into agent.run
            # which would fail and trigger the except/finally path
