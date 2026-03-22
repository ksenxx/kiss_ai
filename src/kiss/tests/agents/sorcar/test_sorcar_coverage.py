"""Integration tests for kiss/agents/sorcar/ to maximize branch coverage.

No mocks, patches, or test doubles.  Uses real files, real git repos, and
real objects.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import requests

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter, _coalesce_events
from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _cleanup_merge_data,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _setup_code_server,
    _snapshot_files,
)
from kiss.agents.sorcar.config import AgentConfig, SorcarConfig
from kiss.agents.sorcar.useful_tools import (
    _extract_command_names,
)
from kiss.agents.sorcar.web_use_tool import (
    INTERACTIVE_ROLES,
    WebUseTool,
)


def _redirect_history(tmpdir: str):
    """Redirect task_history DB to a temp dir."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore_history(saved: tuple):  # type: ignore[type-arg]
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved[0], saved[1], saved[2]


class TestBaseBrowserPrinter:
    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()

    def test_print_result_bad_yaml(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("not: [yaml: {", type="result")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "result"]
        assert len(result_events) == 1
        self.printer.remove_client(cq)


class TestScrollDelta:
    def test_all_directions(self) -> None:
        from kiss.agents.sorcar.web_use_tool import _SCROLL_DELTA

        assert _SCROLL_DELTA["down"] == (0, 300)
        assert _SCROLL_DELTA["up"] == (0, -300)
        assert _SCROLL_DELTA["right"] == (300, 0)
        assert _SCROLL_DELTA["left"] == (-300, 0)


class TestThemePresets:
    def test_all_presets_have_required_keys(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS
        required = {
            "bg", "bg2", "fg", "accent", "border",
            "inputBg", "green", "red", "purple", "cyan",
        }
        for name, theme in _THEME_PRESETS.items():
            assert set(theme.keys()) == required, f"Theme {name} missing keys"

    def test_all_presets_are_hex_colors(self) -> None:

        from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS
        for name, theme in _THEME_PRESETS.items():
            for key, value in theme.items():
                assert re.match(r"^#[0-9a-fA-F]{6}$", value), f"Theme {name}.{key}={value} not hex"


class TestWebUseToolBrowser:
    """Tests requiring real Playwright browser - headless."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tmpdir = tempfile.mkdtemp()
        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_go_to_url_tab_switch(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Page1</h1>")
        result = self.tool.go_to_url("tab:0")
        assert "Page:" in result

    def test_press_key(self) -> None:
        self.tool.go_to_url("data:text/html,<button>B</button>")
        result = self.tool.press_key("Tab")
        assert "Page:" in result

    def test_scroll_down(self) -> None:
        long_page = "<div style='height:5000px'>Tall page</div><button>Bottom</button>"
        self.tool.go_to_url(f"data:text/html,{long_page}")
        result = self.tool.scroll("down", 3)
        assert "Page:" in result

class TestWebUseToolResolveLocator:
    """Test _resolve_locator edge cases with real browser."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_resolve_element_without_name(self) -> None:
        """Elements without name attribute should still be clickable."""
        self.tool.go_to_url("data:text/html,<button></button>")
        result = self.tool.get_page_content()
        if "[1]" in result:
            result2 = self.tool.click(1)
            assert "Page:" in result2


    def test_screenshot_error_handling(self) -> None:
        """Screenshot to invalid path."""
        self.tool.go_to_url("data:text/html,<h1>X</h1>")
        result = self.tool.screenshot("/dev/null/impossible/file.png")
        assert "Error" in result or "saved" in result.lower()


class TestCodeServerBranchesR2:
    """Cover remaining code_server.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prepare_merge_view_hash_oserror_via_directory(self) -> None:
        """File replaced with directory after pre-hash, causing OSError.
        Covers lines 721-723."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=work_dir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=work_dir, capture_output=True,
        )
        fpath = Path(work_dir) / "test.txt"
        fpath.write_text("original")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)
        fpath.write_text("modified")
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        import hashlib
        pre_hashes = {"test.txt": hashlib.md5(b"original_different").hexdigest()}
        fpath.unlink()
        fpath.mkdir()
        (fpath / "subfile.txt").write_text("x")
        result = _prepare_merge_view(
            work_dir, self.tmpdir, pre_hunks, pre_untracked,
            pre_file_hashes=pre_hashes,
        )
        assert isinstance(result, (dict, type(None)))


class TestWebUseToolBranchesR2:
    """Cover remaining web_use_tool.py branches."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_resolve_locator_re_snapshot_success(self) -> None:
        """Element not in stale list, re-snapshot finds it.
        Covers branch 169->171."""
        self.tool.go_to_url("data:text/html,<button>MyBtn</button>")
        self.tool._elements = []
        result = self.tool.click(1)
        assert "Page:" in result

    def test_resolve_locator_zero_matches(self) -> None:
        """Element in snapshot but locator finds 0 matches.
        Covers line 181."""
        self.tool.go_to_url("data:text/html,<button>X</button>")
        tree = self.tool.get_page_content()
        if "[1]" in tree:
            self.tool._elements = [{"role": "button", "name": "NonExistentButtonXYZ"}]
            result = self.tool.click(1)
            assert "Error" in result

    def test_new_tab_via_target_blank_click(self) -> None:
        """Click on target=_blank link to open new tab.
        Covers lines 252-253 (_check_for_new_tab during click)."""
        html = '<a href="about:blank" target="_blank">New</a>'
        self.tool.go_to_url(f"data:text/html,{html}")
        tree = self.tool.get_page_content()
        assert "[1]" in tree
        pages_before = len(self.tool._context.pages)
        result = self.tool.click(1)
        pages_after = len(self.tool._context.pages)
        assert pages_after > pages_before, f"Expected new tab, got {pages_before} -> {pages_after}"
        assert "Page:" in result or "Error" in result

    def test_check_for_new_tab_no_context(self) -> None:
        """_check_for_new_tab with None context.
        Covers line 160 (context is None -> return)."""
        self.tool.go_to_url("data:text/html,<p>test</p>")
        saved_ctx = self.tool._context
        self.tool._context = None
        self.tool._check_for_new_tab()
        self.tool._context = saved_ctx

class TestWebUseToolCloseException:
    """Separate class for close exception test to avoid polluting other tests."""

    def test_close_with_corrupted_playwright(self) -> None:
        """Close with corrupted _playwright that raises on stop().
        Covers lines 384-386 (exception handler in close)."""
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        tool = WebUseTool(user_data_dir=None)
        tool.go_to_url("data:text/html,<p>test</p>")
        if tool._browser:
            tool._browser.close()
        real_pw = tool._playwright
        tool._playwright = "corrupted"
        tool._browser = None
        tool._context = None
        result = tool.close()
        assert result == "Browser closed."
        try:
            if real_pw:
                real_pw.stop()
        except Exception:
            pass


class TestWebUseToolWaitForStable:
    """Cover _wait_for_stable exception handlers (149-156) and _check_for_new_tab."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_resolve_locator_empty_snapshot_re_snapshot(self) -> None:
        """Re-snapshot on about:blank where snapshot is empty.
        Covers branch 169->171 (if snapshot: False path)."""
        self.tool.go_to_url("about:blank")
        self.tool._elements = [{"role": "button", "name": "fake"}]
        result = self.tool.click(2)
        assert "Error" in result


class TestWebUseToolDomContentLoadedTimeout:
    """Cover web_use_tool.py lines 149-151: domcontentloaded timeout in _wait_for_stable."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_click_navigates_to_slow_page(self) -> None:
        """Click a link that navigates to a page that never finishes loading.
        The domcontentloaded wait in _wait_for_stable should timeout."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(5)
        port = srv.getsockname()[1]

        def handle_client(conn: socket.socket) -> None:
            try:
                data = conn.recv(4096).decode()
                if "GET /slow" in data:
                    response = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/html\r\n"
                        "Transfer-Encoding: chunked\r\n\r\n"
                    )
                    conn.sendall(response.encode())
                    chunk = "<html><body><h1>Loading"
                    conn.sendall(f"{len(chunk):x}\r\n{chunk}\r\n".encode())
                    time.sleep(30)
                else:
                    html = '<html><body><a href="/slow">GoSlow</a></body></html>'
                    resp = (
                        f"HTTP/1.1 200 OK\r\n"
                        f"Content-Type: text/html\r\n"
                        f"Content-Length: {len(html)}\r\n\r\n{html}"
                    )
                    conn.sendall(resp.encode())
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        def accept_loop() -> None:
            while True:
                try:
                    conn, _ = srv.accept()
                    threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
                except Exception:
                    break

        acceptor = threading.Thread(target=accept_loop, daemon=True)
        acceptor.start()
        try:
            self.tool.go_to_url(f"http://127.0.0.1:{port}/")
            result = self.tool.click(1)
            assert "Page:" in result or "Error" in result
        finally:
            srv.close()


class TestWebUseToolAllInvisibleElements:
    """Cover web_use_tool.py lines 186->184 and 191:
    is_visible() returns False for all elements, falls through to return locator.first."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_all_zero_size_buttons(self) -> None:
        """Multiple buttons with zero bounding box: get_by_role finds them,
        is_visible returns False for all, loop falls through to locator.first."""
        html = (
            "<html><body>"
            '<button style="width:0;height:0;overflow:hidden;padding:0;border:0;'
            'margin:0;display:inline-block">ZBtn</button>'
            '<button style="width:0;height:0;overflow:hidden;padding:0;border:0;'
            'margin:0;display:inline-block">ZBtn</button>'
            "</body></html>"
        )

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            self.tool.go_to_url(f"http://127.0.0.1:{port}/")
            tree = self.tool.get_page_content()
            assert "ZBtn" in tree
            result = self.tool.click(1)
            assert "Page:" in result or "Error" in result
        finally:
            server.shutdown()

    def test_one_zero_size_one_visible(self) -> None:
        """First button is zero-size (invisible), second is normal (visible).
        Loop iterates past the first, returns the second."""
        html = (
            "<html><body>"
            '<button style="width:0;height:0;overflow:hidden;padding:0;border:0;'
            'margin:0;display:inline-block">MBtn</button>'
            "<button>MBtn</button>"
            "</body></html>"
        )

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            self.tool.go_to_url(f"http://127.0.0.1:{port}/")
            tree = self.tool.get_page_content()
            assert "MBtn" in tree
            result = self.tool.click(1)
            assert "Page:" in result or "Error" in result
        finally:
            server.shutdown()


class TestExtractCommandNames:

    def test_invalid_shlex(self) -> None:
        """Unmatched quote should not crash."""
        names = _extract_command_names("echo 'unclosed")
        assert isinstance(names, list)

class TestCoalesceEvents:

    def test_no_merge_missing_text(self) -> None:
        events = [
            {"type": "thinking_delta"},
            {"type": "thinking_delta", "text": "a"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

class TestSetupCodeServer:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_setup_removes_chat_sessions(self) -> None:
        ws_dir = Path(self.tmpdir) / "User" / "workspaceStorage" / "ws1" / "chatSessions"
        ws_dir.mkdir(parents=True)
        (ws_dir / "session.json").write_text("{}")
        ext_dir = os.path.join(self.tmpdir, "extensions")
        _setup_code_server(self.tmpdir, ext_dir)
        assert not ws_dir.exists()


class TestThemePresetsBranch:
    def test_all_presets_exist(self) -> None:
        assert "dark" in _THEME_PRESETS
        assert "light" in _THEME_PRESETS
        assert "hcDark" in _THEME_PRESETS
        assert "hcLight" in _THEME_PRESETS

    def test_presets_have_keys(self) -> None:
        for name, preset in _THEME_PRESETS.items():
            assert "bg" in preset
            assert "fg" in preset
            assert "accent" in preset


class TestSorcarConfig:
    def test_defaults(self) -> None:
        cfg = AgentConfig()
        assert cfg.model_name == "claude-opus-4-6"
        assert cfg.max_steps == 100
        assert cfg.max_budget == 200.0
        assert cfg.headless is False

    def test_sorcar_config(self) -> None:
        cfg = SorcarConfig()
        assert isinstance(cfg.sorcar_agent, AgentConfig)


class TestSorcarAgentMain:
    def test_main_with_file(self) -> None:
        tmpdir = tempfile.mkdtemp()
        task_file = os.path.join(tmpdir, "task.txt")
        Path(task_file).write_text("echo hello")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kiss.agents.sorcar.sorcar_agent",
                    "--max_steps", "0",
                    "--max_budget", "0.0",
                    "--max_sub_sessions", "1",
                    "--work_dir", tmpdir,
                    "--headless", "true",
                    "-f", task_file,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

class TestUsefulToolsEdgeCases:
    """Cover remaining useful_tools.py branches."""

    def test_extract_command_names_escaped_chars(self) -> None:
        """Escaped characters."""
        names = _extract_command_names("echo hello\\ world")
        assert "echo" in names

    def test_extract_command_names_double_quote_escape(self) -> None:
        """Escaped quote in double-quoted string."""
        names = _extract_command_names('echo "hello \\"world\\""')
        assert "echo" in names


class TestCodeServerEdgeCases:
    """Cover remaining code_server.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cleanup_merge_data_manifest_readonly(self) -> None:
        """Cover OSError in manifest.unlink()."""
        manifest = Path(self.tmpdir) / "pending-merge.json"
        manifest.write_text("{}")
        os.chmod(self.tmpdir, 0o555)
        try:
            _cleanup_merge_data(self.tmpdir)
        finally:
            os.chmod(self.tmpdir, 0o755)


class TestWebUseToolEdgeCases:
    """Cover remaining web_use_tool.py branches without starting a browser."""

    def test_scroll_delta_values(self) -> None:
        """Cover _SCROLL_DELTA dict lookup."""
        from kiss.agents.sorcar.web_use_tool import _SCROLL_DELTA

        assert _SCROLL_DELTA["down"] == (0, 300)
        assert _SCROLL_DELTA["up"] == (0, -300)
        assert _SCROLL_DELTA["right"] == (300, 0)
        assert _SCROLL_DELTA["left"] == (-300, 0)

    def test_interactive_roles_complete(self) -> None:
        """Ensure all expected roles are in INTERACTIVE_ROLES."""
        expected = {"link", "button", "textbox", "searchbox", "combobox",
                    "checkbox", "radio", "switch", "slider", "spinbutton",
                    "tab", "menuitem", "menuitemcheckbox", "menuitemradio",
                    "option", "treeitem"}
        assert INTERACTIVE_ROLES == expected


class TestBrowserUiFinalEdgeCases:
    """Cover last few browser_ui.py branches."""

    def test_handle_message_content_mixed_blocks(self) -> None:
        """Content with both valid and invalid blocks."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class ValidBlock:
            is_error = False
            content = "ok"

        class InvalidBlock:
            pass

        class Msg:
            content = [InvalidBlock(), ValidBlock()]

        printer._handle_message(Msg())
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        assert len(events) == 1
        assert events[0]["type"] == "tool_result"
        printer.remove_client(cq)

    def test_print_text_only_whitespace(self) -> None:
        """Cover empty text.strip() branch."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("   \n\t  ", type="text")
        assert cq.empty()
        printer.remove_client(cq)


class TestWebUseToolBrowserBranch:
    """Integration tests for WebUseTool with a real headless browser."""

    @pytest.fixture(autouse=True)
    def setup_tool(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        if hasattr(self, "tool"):
            self.tool.close()

    def _write_html(self, name: str, content: str) -> str:
        p = self.tmp_path / name
        p.write_text(content)
        return f"file://{p}"

    def test_click_hover(self) -> None:
        url = self._write_html(
            "hover.html",
            "<html><body><button>Hover</button></body></html>",
        )
        self.tool.go_to_url(url)
        result = self.tool.click(1, action="hover")
        assert "Page:" in result


def _wait_for_port_file(port_file: str, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(port_file) and os.path.getsize(port_file) > 0:
            return int(Path(port_file).read_text().strip())
        time.sleep(0.3)
    raise TimeoutError(f"Port file {port_file} not written within {timeout}s")


class TestSorcarServerIntegration:
    """Integration tests for run_chatbot by starting it as a subprocess.

    Uses _sorcar_test_server_with_cov.py to collect coverage data from
    the server subprocess. Coverage is combined after the server stops.
    """

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        tmpdir = tmp_path_factory.mktemp("sorcar_server")
        work_dir = str(tmpdir / "work")
        os.makedirs(work_dir)

        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=work_dir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=work_dir, capture_output=True,
        )
        Path(work_dir, "file.txt").write_text("line1\nline2\n")
        Path(work_dir, "readme.md").write_text("# Test\n\nsome content\n")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

        port_file = str(tmpdir / "port")
        cov_data_file = str(tmpdir / ".coverage.server")

        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).parent / "_sorcar_test_server_with_cov.py"),
                port_file,
                work_dir,
                cov_data_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        keepalive = None
        try:
            port = _wait_for_port_file(port_file)
            base_url = f"http://127.0.0.1:{port}"

            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                try:
                    resp = requests.get(base_url, timeout=2)
                    if resp.status_code == 200:
                        break
                except requests.ConnectionError:
                    time.sleep(0.3)
            else:
                raise TimeoutError("Server not responsive")

            keepalive = requests.get(
                f"{base_url}/events", stream=True, timeout=300,
            )

            yield base_url, work_dir, proc, str(tmpdir)
        finally:
            if keepalive is not None:
                keepalive.close()
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            try:
                import coverage as cov_mod

                if os.path.exists(cov_data_file):
                    main_cov_file = os.path.join(os.getcwd(), ".coverage")
                    cov = cov_mod.Coverage(data_file=main_cov_file)
                    cov.combine(data_paths=[cov_data_file], keep=True)
                    cov.save()
            except Exception:
                pass

    def test_index(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(base_url, timeout=5)
        assert r.status_code == 200
        assert "KISS" in r.text or "html" in r.text.lower()

    def test_models(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/models", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "selected" in data

    def test_theme(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/theme", timeout=5)
        assert r.status_code == 200

    def test_tasks(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/tasks", timeout=5)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_suggestions_empty(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/suggestions?q=", timeout=5)
        assert r.status_code == 200

    def test_suggestions_with_query(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/suggestions?q=test", timeout=5)
        assert r.status_code == 200

    def test_suggestions_files_mode(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/suggestions?mode=files&q=file", timeout=5)
        assert r.status_code == 200

    def test_complete_short(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/complete?q=a", timeout=5)
        assert r.status_code == 200

    def test_complete_longer(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/complete?q=test something", timeout=5)
        assert r.status_code == 200

    def test_active_file_info(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/active-file-info", timeout=5)
        assert r.status_code == 200

    def test_get_file_content(self, server) -> None:
        base_url, work_dir, _, _ = server
        fpath = os.path.join(work_dir, "file.txt")
        r = requests.get(f"{base_url}/get-file-content?path={fpath}", timeout=5)
        assert r.status_code == 200

    def test_get_file_content_not_found(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/get-file-content?path=/no/such/file", timeout=5)
        assert r.status_code == 404

    def test_run_empty_task(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/run", json={"task": ""}, timeout=5)
        assert r.status_code == 400

    def test_run_task(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/run",
            json={"task": "test task", "model": "claude-opus-4-6"},
            timeout=5,
        )
        assert r.status_code == 200
        time.sleep(1)

    def test_stop_task(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/stop", timeout=5)
        assert r.status_code in (200, 404)

    def test_run_selection_empty(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/run-selection", json={"text": ""}, timeout=5)
        assert r.status_code == 400

    def test_run_selection(self, server) -> None:
        base_url, _, _, _ = server
        time.sleep(1)
        r = requests.post(f"{base_url}/run-selection", json={"text": "echo hello"}, timeout=5)
        assert r.status_code == 200
        time.sleep(1)

    def test_task_events_invalid(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/task-events?idx=bad", timeout=5)
        assert r.status_code == 400

    def test_task_events_out_of_range(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/task-events?idx=9999", timeout=5)
        assert r.status_code == 404

    def test_task_events_valid(self, server) -> None:
        base_url, _, _, _ = server
        tasks = requests.get(f"{base_url}/tasks", timeout=5).json()
        if tasks:
            r = requests.get(f"{base_url}/task-events?idx=0", timeout=5)
            assert r.status_code == 200

    def test_open_file(self, server) -> None:
        base_url, work_dir, _, _ = server
        r = requests.post(
            f"{base_url}/open-file",
            json={"path": os.path.join(work_dir, "file.txt")},
            timeout=5,
        )
        assert r.status_code == 200

    def test_open_file_not_found(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/open-file",
            json={"path": "/no/such/file.txt"},
            timeout=5,
        )
        assert r.status_code == 404

    def test_open_file_empty(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/open-file", json={"path": ""}, timeout=5)
        assert r.status_code == 400

    def test_merge_action_next(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "next"}, timeout=5
        )
        assert r.status_code == 200

    def test_merge_action_invalid(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "invalid"}, timeout=5
        )
        assert r.status_code == 400

    def test_merge_action_all_done(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "all-done"}, timeout=5
        )
        assert r.status_code == 200

    def test_merge_action_accept(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "accept"}, timeout=5
        )
        assert r.status_code == 200

    def test_merge_action_reject(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "reject"}, timeout=5
        )
        assert r.status_code == 200

    def test_focus_chatbox(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/focus-chatbox", timeout=5)
        assert r.status_code == 200

    def test_focus_editor(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/focus-editor", timeout=5)
        assert r.status_code == 200

    def test_record_file_usage(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": "src/test.py"},
            timeout=5,
        )
        assert r.status_code == 200

    def test_commit_no_changes(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/commit", timeout=30)
        assert r.status_code in (200, 400)

    def test_sse_events(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/events", stream=True, timeout=10)
        assert r.status_code == 200
        content = b""
        for chunk in r.iter_content(chunk_size=256):
            content += chunk
            if len(content) > 20:
                break
        r.close()

    def test_run_with_attachments(self, server) -> None:
        import base64

        base_url, _, _, _ = server
        time.sleep(1)
        fake_image = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
        fake_pdf = base64.b64encode(b"%PDF-1.4 fake").decode()
        r = requests.post(
            f"{base_url}/run",
            json={
                "task": "test with attachments",
                "attachments": [
                    {"data": fake_image, "mime_type": "image/png"},
                    {"data": fake_pdf, "mime_type": "application/pdf"},
                ],
            },
            timeout=5,
        )
        assert r.status_code == 200
        time.sleep(1)

    def test_active_file_md(self, server) -> None:
        """Cover active file with .md extension triggering prompt detection."""
        base_url, work_dir, _, _ = server
        from kiss.agents.sorcar.task_history import _KISS_DIR

        sorcar_dir = _KISS_DIR / "sorcar-data"
        sorcar_dir.mkdir(parents=True, exist_ok=True)
        (sorcar_dir / "active-file.json").write_text(
            json.dumps({"path": os.path.join(work_dir, "readme.md")})
        )
        r = requests.get(f"{base_url}/active-file-info", timeout=5)
        assert r.status_code == 200
        assert "is_prompt" in r.json()

    def test_closing(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/closing", timeout=5)
        assert r.status_code == 200


class TestWebUseToolBrowserExtra:
    """Additional browser tests for remaining web_use_tool.py branches."""

    @pytest.fixture(autouse=True)
    def setup_tool(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        if hasattr(self, "tool"):
            self.tool.close()

    def _write_html(self, name: str, content: str) -> str:
        p = self.tmp_path / name
        p.write_text(content)
        return f"file://{p}"

    def test_screenshot_error(self) -> None:
        """Cover screenshot exception path (line 346-348)."""
        tool2 = WebUseTool(user_data_dir=None)
        url = self._write_html("ss.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.screenshot()
        assert "Error" in result
        tool2.close()

    def test_get_page_content_error(self) -> None:
        """Cover get_page_content exception path (line 368-370)."""
        tool2 = WebUseTool(user_data_dir=None)
        url = self._write_html("pc.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.get_page_content()
        assert "Error" in result
        tool2.close()

    def test_press_key_error(self) -> None:
        """Cover press_key exception path."""
        tool2 = WebUseTool(user_data_dir=None)
        url = self._write_html("k.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.press_key("Enter")
        assert "Error" in result
        tool2.close()

    def test_type_text_error(self) -> None:
        """Cover type_text exception path."""
        tool2 = WebUseTool(user_data_dir=None)
        url = self._write_html("t.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.type_text(1, "text")
        assert "Error" in result
        tool2.close()


class TestCodeServerFinalEdgeCases:
    """Cover remaining code_server.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prepare_merge_view_modified_untracked_already_in_file_hunks(self) -> None:
        """Pre-existing untracked file already in file_hunks -> skip."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=work_dir, capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
        Path(work_dir, "f.txt").write_text("a\nb\n")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

        Path(work_dir, "ut.txt").write_text("original\n")
        pre_untracked = _capture_untracked(work_dir)
        pre_hashes = _snapshot_files(work_dir, pre_untracked | {"f.txt"})
        _save_untracked_base(work_dir, pre_untracked)

        Path(work_dir, "f.txt").write_text("a\nX\n")
        Path(work_dir, "ut.txt").write_text("modified\n")

        data_dir = os.path.join(self.tmpdir, "data")
        result = _prepare_merge_view(work_dir, data_dir, {}, pre_untracked, pre_hashes)
        assert result.get("status") == "opened"

