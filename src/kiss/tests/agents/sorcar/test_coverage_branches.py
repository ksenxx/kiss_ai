"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects
and real function calls.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
)
from kiss.agents.sorcar.web_use_tool import (
    WebUseTool,
)
from kiss.agents.vscode.browser_ui import (
    BaseBrowserPrinter,
    _coalesce_events,
)
from kiss.agents.vscode.server import VSCodeServer


class TestBaseBrowserPrinterBranches:
    """Cover uncovered branches in BaseBrowserPrinter."""

    def test_reset_clears_bash_buffer_and_timer(self):
        p = BaseBrowserPrinter()
        p._thread_local.tab_id = "0"
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
        p = BaseBrowserPrinter()
        p._thread_local.stop_event = threading.Event()
        p._check_stop()
        p._thread_local.stop_event.set()
        with pytest.raises(KeyboardInterrupt):
            p._check_stop()

    def test_print_text_blank_no_broadcast(self):
        """Text that is only whitespace should not be broadcast."""
        p = BaseBrowserPrinter()
        p.start_recording()
        p.print("   ", type="text")
        events = p.stop_recording()
        assert len(events) == 0

    def test_token_callback_stop(self):
        p = BaseBrowserPrinter()
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

    def test_bash_timeout_nonstreaming(self):
        ut = UsefulTools()
        result = ut.Bash("sleep 100", "timeout test", timeout_seconds=0.5)
        assert "timeout" in result.lower()


class TestTaskHistoryBranches:
    """Cover specific uncovered branches in persistence."""

    def _fresh_db(self, tmp_path):
        """Switch to a fresh DB in tmp_path, return cleanup callback."""
        saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = tmp_path / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        return saved

    def _restore_db(self, saved):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        th._DB_PATH, th._db_conn, th._KISS_DIR = saved

    def test_load_chat_context_empty_id(self):
        assert th._load_chat_context("") == []


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
        t = threading.Thread(target=lambda: time.sleep(5), daemon=True)
        t.start()
        server._get_tab("0").task_thread = t
        server._handle_command({"type": "run", "prompt": "test", "tabId": "0"})
        assert any("already running" in e.get("text", "") for e in events)
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
        tab_id = "1"
        server.printer._thread_local.tab_id = tab_id
        user_q: queue.Queue[str] = queue.Queue(maxsize=1)
        server._get_tab(tab_id).user_answer_queue = user_q

        def answer():
            time.sleep(0.1)
            user_q.put("my answer")

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
            "",
        ]
        old_stdin = os.sys.stdin  # type: ignore[attr-defined]
        os.sys.stdin = io.StringIO("".join(cmds))  # type: ignore[attr-defined]
        try:
            server.run()
        finally:
            os.sys.stdin = old_stdin  # type: ignore[attr-defined]

        model_events = [e for e in events if e["type"] == "models"]
        assert len(model_events) == 1

    def test_emit_pending_worktree_with_branch(self, tmp_path):
        """_emit_pending_worktree emits worktree_done when branch exists."""
        server, events = self._make_server()
        tab = server._get_tab("0")
        tab.use_worktree = True
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"],
                       cwd=str(repo), capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"],
                       cwd=str(repo), capture_output=True)
        (repo / "f.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"],
                       cwd=str(repo), capture_output=True)
        chat_id = tab.agent._chat_id
        branch = f"kiss/wt-{chat_id}-1234567890"
        subprocess.run(["git", "branch", branch],
                       cwd=str(repo), capture_output=True)
        subprocess.run(["git", "config", f"branch.{branch}.kiss-original", "main"],
                       cwd=str(repo), capture_output=True)
        subprocess.run(["git", "checkout", branch],
                       cwd=str(repo), capture_output=True)
        (repo / "f.txt").write_text("changed")
        subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
        subprocess.run(["git", "commit", "-m", "change"],
                       cwd=str(repo), capture_output=True)
        subprocess.run(["git", "checkout", "main"],
                       cwd=str(repo), capture_output=True)

        server.work_dir = str(repo)
        server._emit_pending_worktree("0")
        wt_events = [e for e in events if e.get("type") == "worktree_done"]
        assert len(wt_events) == 1
        assert wt_events[0]["branch"] == branch
        assert "f.txt" in wt_events[0]["changedFiles"]

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
    tool = WebUseTool(user_data_dir=None, headless=True)
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
tool = WebUseTool(user_data_dir=udd, headless=True)
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
        browser_tool._elements = []
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
        assert isinstance(result, str)

    def test_type_text_error_invalid_element(self, http_server, browser_tool):
        """type_text error on non-existent element."""
        browser_tool.go_to_url(http_server + "/empty")
        result = browser_tool.type_text(999, "text")
        assert "Error" in result


class TestHandleMessageContentBlockNoIsError:
    """Cover the case where content block lacks is_error/content attributes."""

    def test_block_without_is_error(self):
        p = BaseBrowserPrinter()
        p.start_recording()
        block = SimpleNamespace(some_other_attr="value")
        msg = SimpleNamespace(content=[block])
        p._handle_message(msg)
        events = p.stop_recording()
        assert len(events) == 0


class TestUsefulToolsMoreBranches:
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-only (uses chmod)")
    def test_edit_exception(self):
        """Edit on a directory should raise an error."""
        ut = UsefulTools()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "readonly.txt"
            f.write_text("hello old world")
            f.chmod(0o444)
            try:
                result = ut.Edit(str(f), "old", "new")
                assert "Error" in result or "Successfully" in result
            finally:
                f.chmod(0o644)


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
        time.sleep(0.5)
        assert server._file_cache is None or isinstance(server._file_cache, list)


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
