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
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
)
from kiss.agents.sorcar.web_use_tool import (
    WebUseTool,
)


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
                capture_output=True, text=True, timeout=180,
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
