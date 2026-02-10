"""Tests for useful_tools.py module."""

import os
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kiss.core.useful_tools import (
    UsefulTools,
    _extract_directory,
    _extract_search_results,
    _render_page_with_playwright,
    fetch_url,
    parse_bash_command_paths,
)


@pytest.fixture
def temp_test_dir():
    test_dir = Path(tempfile.mkdtemp()).resolve()
    original_dir = Path.cwd()
    os.chdir(test_dir)
    yield test_dir
    os.chdir(original_dir)
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def tools_sandbox(temp_test_dir):
    readable_dir = temp_test_dir / "readable"
    writable_dir = temp_test_dir / "writable"
    readable_dir.mkdir()
    writable_dir.mkdir()
    tools = UsefulTools(
        base_dir=str(temp_test_dir),
        readable_paths=[str(readable_dir)],
        writable_paths=[str(writable_dir)],
    )
    return tools, readable_dir, writable_dir, temp_test_dir


@pytest.fixture
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/notfound":
                self.send_response(404)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<html><body>Not found</body></html>")
                return
            if self.path == "/slow":
                time.sleep(0.2)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<html><main>Slow content</main></html>")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><main>Hello from server</main></html>")

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


class TestExtractDirectory:
    def test_invalid_path_returns_none(self):
        assert _extract_directory("\0") is None


class TestParseBashCommandPaths:
    def test_single_file_cat_redirect_in(self, temp_test_dir):
        test_file = temp_test_dir / "test.txt"
        test_file.write_text("content")
        readable, writable = parse_bash_command_paths(f"cat < {test_file}")
        assert readable == [str(test_file)]
        assert writable == []

    def test_single_file_tee(self, temp_test_dir):
        test_file = temp_test_dir / "test.txt"
        readable, writable = parse_bash_command_paths(f"tee {test_file}")
        assert readable == []
        assert str(test_file) in writable

    def test_dd_command(self, temp_test_dir):
        input_file = temp_test_dir / "input.bin"
        output_file = temp_test_dir / "output.bin"
        input_file.write_bytes(b"data")

        readable, writable = parse_bash_command_paths(f"dd if={input_file} of={output_file}")
        assert readable == [str(input_file)]
        assert writable == [str(output_file)]

    def test_safe_special_paths_filtered(self):
        readable, writable = parse_bash_command_paths("echo hello > /dev/null")
        assert readable == [] and writable == []

        readable, writable = parse_bash_command_paths("cat /dev/zero")
        assert readable == []

        readable, writable = parse_bash_command_paths("dd if=/dev/urandom of=/dev/null")
        assert readable == [] and writable == []

    def test_safe_prefix_paths_filtered(self):
        _, writable = parse_bash_command_paths("echo hello > /dev/fd/3")
        assert writable == []
        readable, _ = parse_bash_command_paths("cat < /proc/self/status")
        assert readable == []

    def test_flags_ignored(self, temp_test_dir):
        test_file = temp_test_dir / "test.txt"
        test_file.write_text("content")

        readable, writable = parse_bash_command_paths(f"grep -i -n pattern {test_file}")
        assert readable == [str(test_file)]
        assert writable == []

    def test_unterminated_quotes_fallback_split(self):
        readable, writable = parse_bash_command_paths('echo "unterminated')
        assert readable == [] and writable == []

    def test_empty_pipe_parts(self):
        readable, writable = parse_bash_command_paths(" | ")
        assert readable == [] and writable == []

    def test_invalid_command_type(self):
        readable, writable = parse_bash_command_paths(1)  # type: ignore[arg-type]
        assert readable == [] and writable == []


class TestSearchResultExtraction:
    def test_skips_invalid_and_blocked_domains(self):
        from bs4 import BeautifulSoup

        html = """
        <html><body>
            <a class="r" href="https://example.com">Good Result</a>
            <a class="r" href="https://youtube.com/watch?v=123">YouTube</a>
            <a class="r" href="/relative/path">Relative</a>
            <a class="r" href="http://example.net"></a>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        results = _extract_search_results(soup, "a.r", max_results=10)
        assert results == [("Good Result", "https://example.com")]


class TestUsefulTools:
    def test_permission_denied(self, tools_sandbox):
        tools, _, _, test_dir = tools_sandbox
        outside_file = test_dir / "outside.txt"
        cmd = f"touch {outside_file}"
        result = tools.Bash(cmd, "Test permission")
        assert "Error: Access denied for writing" in result

    def test_bash_read_allowed(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        test_file = readable_dir / "test.txt"
        test_file.write_text("readable content")
        assert "readable content" in tools.Bash(f"cat {test_file}", "Read allowed")

    def test_bash_write_allowed(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "output.txt"

        result = tools.Bash(f"echo 'writable content' > {test_file}", "Write allowed")
        assert "Error:" not in result
        assert test_file.read_text().strip() == "writable content"

    def test_bash_allows_safe_special_paths(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        assert "Error:" not in tools.Bash("echo hello > /dev/null", "Redirect to /dev/null")
        assert "Error: Access denied" not in tools.Bash(
            "dd if=/dev/urandom bs=8 count=1 2>/dev/null | od -An -tx1", "Read /dev/urandom"
        )

    def test_bash_timeout(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        result = tools.Bash("sleep 1", "Timeout test", timeout_seconds=0.01)
        assert result == "Error: Command execution timeout"

    def test_bash_nonzero_exit(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        assert tools.Bash("false", "Failure test").startswith("Error:")

    def test_edit_string_not_found(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "missing.txt"
        test_file.write_text("alpha beta")

        result = tools.Edit(str(test_file), "gamma", "delta")
        assert result.startswith("Error:")
        assert "String not found" in result

    def test_edit_timeout(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "timeout_edit.txt"
        test_file.write_text("a" * 5_000_000)

        result = tools.Edit(str(test_file), "a", "b", replace_all=True, timeout_seconds=0.0001)
        assert result == "Error: Command execution timeout"

    def test_multiedit_delegates_to_edit(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "multiedit_test.txt"
        test_file.write_text("Alpha Beta\nGamma Delta\n")

        tools.MultiEdit(file_path=str(test_file), old_string="Alpha Beta", new_string="Alpha Omega")
        assert "Alpha Omega" in test_file.read_text()
        assert "Alpha Beta" not in test_file.read_text()


class TestRead:
    def test_read_nonexistent_file(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        result = tools.Read(str(readable_dir / "missing.txt"))
        assert "Error:" in result

    def test_read_max_lines_truncation(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        test_file = readable_dir / "big.txt"
        test_file.write_text("\n".join(f"line{i}" for i in range(100)))
        result = tools.Read(str(test_file), max_lines=10)
        assert "[truncated: 90 more lines]" in result
        assert "line9" in result
        assert "line10" not in result


class TestFetchUrl:
    def test_http_error(self, http_server):
        result = fetch_url(f"{http_server}/notfound", {"User-Agent": "Test Agent"})
        assert "Failed to fetch content: HTTP 404" in result

    def test_timeout(self, http_server):
        headers = {"User-Agent": "Test Agent"}
        result = fetch_url(f"{http_server}/slow", headers, timeout_seconds=0.01)
        assert result == "Failed to fetch content: Request timed out."

    def test_connection_refused(self):
        result = fetch_url("http://127.0.0.1:1", {"User-Agent": "Test Agent"}, timeout_seconds=0.1)
        assert result.startswith("Failed to fetch content:")

    def test_invalid_headers(self):
        result = fetch_url("http://example.com", 1, timeout_seconds=0.1)  # type: ignore[arg-type]
        assert result.startswith("Failed to fetch content:")


class TestRenderPageWithPlaywright:
    @pytest.mark.timeout(60)
    def test_render_local_file(self, tmp_path):
        html_file = tmp_path / "page.html"
        html_file.write_text("<html><main>Hello Playwright</main></html>")
        content = _render_page_with_playwright(f"file://{html_file}", wait_selector="#missing")
        assert "Hello Playwright" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
