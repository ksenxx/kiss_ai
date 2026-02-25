"""Tests for useful_tools.py module."""

import os
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.assistant.useful_tools import (
    UsefulTools,
    _extract_command_names,
    _extract_leading_command_name,
    _extract_search_results,
    _render_page_with_playwright,
    _strip_heredocs,
    fetch_url,
    search_web,
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
def tools(temp_test_dir):
    return UsefulTools(), temp_test_dir


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
            if self.path == "/empty":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<html><body></body></html>")
                return
            if self.path == "/article":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><script>var x=1;</script>"
                    b"<nav>Nav</nav><article>Article content here</article></body></html>"
                )
                return
            if self.path == "/long":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                body = "A" * 200
                self.wfile.write(f"<html><main>{body}</main></html>".encode())
                return
            if self.path == "/role-main":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b'<html><body><div role="main">Role main content</div></body></html>'
                )
                return
            if self.path == "/id-content":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b'<html><body><div id="content">ID content area</div></body></html>'
                )
                return
            if self.path == "/class-content":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b'<html><body><div class="main-wrapper">Class content area</div></body></html>'
                )
                return
            if self.path == "/body-only":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<html><body>Body only content</body></html>")
                return
            if self.path == "/no-body":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<p>Bare paragraph</p>")
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

    def test_max_results_limit(self):
        from bs4 import BeautifulSoup

        html = """
        <html><body>
            <a class="r" href="https://a.com">A</a>
            <a class="r" href="https://b.com">B</a>
            <a class="r" href="https://c.com">C</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        results = _extract_search_results(soup, "a.r", max_results=1)
        assert len(results) == 1
        assert results[0] == ("A", "https://a.com")

    def test_href_list_value(self):
        from bs4 import BeautifulSoup

        html = '<html><body><a class="r" href="https://example.com">Link</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        link_tag = cast(Any, soup.find("a"))
        link_tag["href"] = ["https://first.com", "https://second.com"]
        results = _extract_search_results(soup, "a.r", max_results=10)
        assert results == [("Link", "https://first.com")]

    def test_href_empty_list(self):
        from bs4 import BeautifulSoup

        html = '<html><body><a class="r" href="https://example.com">Link</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        link_tag = cast(Any, soup.find("a"))
        link_tag["href"] = []
        results = _extract_search_results(soup, "a.r", max_results=10)
        assert results == []


class TestUsefulTools:
    def test_bash_read_file(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "test.txt"
        test_file.write_text("readable content")
        assert "readable content" in ut.Bash(f"cat {test_file}", "Read file")

    def test_bash_write_file(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "output.txt"
        result = ut.Bash(f"echo 'writable content' > {test_file}", "Write file")
        assert "Error:" not in result
        assert test_file.read_text().strip() == "writable content"

    def test_bash_timeout(self, tools):
        ut, _ = tools
        result = ut.Bash("sleep 1", "Timeout test", timeout_seconds=0.01)
        assert result == "Error: Command execution timeout"

    def test_bash_output_truncation(self, tools):
        ut, test_dir = tools
        big_file = test_dir / "big.txt"
        big_file.write_text("X" * 200)
        result = ut.Bash(f"cat {big_file}", "Cat big", max_output_chars=50)
        assert "truncated" in result

    def test_bash_called_process_error(self, tools):
        ut, _ = tools
        result = ut.Bash("false", "Failing command")
        assert "Error:" in result

    def test_bash_disallowed_command(self, tools):
        ut, _ = tools
        result = ut.Bash("eval echo hi", "Disallowed")
        assert "Error: Command 'eval' is not allowed" in result

    def test_edit_string_not_found(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "missing.txt"
        test_file.write_text("alpha beta")
        result = ut.Edit(str(test_file), "gamma", "delta")
        assert result.startswith("Error:")
        assert "String not found" in result

    def test_edit_timeout(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "timeout_edit.txt"
        test_file.write_text("a" * 5_000_000)
        result = ut.Edit(str(test_file), "a", "b", replace_all=True, timeout_seconds=0.0001)
        assert result == "Error: Command execution timeout"

    def test_edit_success(self, tools):
        ut, test_dir = tools
        f = test_dir / "edit_me.txt"
        f.write_text("hello world")
        result = ut.Edit(str(f), "hello", "goodbye")
        assert "Successfully replaced" in result
        assert f.read_text() == "goodbye world"

    def test_edit_replace_all(self, tools):
        ut, test_dir = tools
        f = test_dir / "multi.txt"
        f.write_text("aaa bbb aaa")
        result = ut.Edit(str(f), "aaa", "ccc", replace_all=True)
        assert "Successfully replaced" in result
        assert f.read_text() == "ccc bbb ccc"

    def test_edit_not_unique(self, tools):
        ut, test_dir = tools
        f = test_dir / "dup.txt"
        f.write_text("aaa\naaa\n")
        result = ut.Edit(str(f), "aaa", "ccc")
        assert "Error:" in result
        assert "not unique" in result

    def test_multi_edit(self, tools):
        ut, test_dir = tools
        f = test_dir / "multi_edit.txt"
        f.write_text("foo bar")
        result = ut.MultiEdit(str(f), "foo", "baz")
        assert "Successfully replaced" in result
        assert f.read_text() == "baz bar"

    def test_read_success(self, tools):
        ut, test_dir = tools
        f = test_dir / "hello.txt"
        f.write_text("hello world")
        result = ut.Read(str(f))
        assert result == "hello world"

    def test_read_nonexistent_file(self, tools):
        ut, test_dir = tools
        result = ut.Read(str(test_dir / "missing.txt"))
        assert "Error:" in result

    def test_read_max_lines_truncation(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "big.txt"
        test_file.write_text("\n".join(f"line{i}" for i in range(100)))
        result = ut.Read(str(test_file), max_lines=10)
        assert "[truncated: 90 more lines]" in result
        assert "line9" in result
        assert "line10" not in result

    def test_write_success(self, tools):
        ut, test_dir = tools
        f = test_dir / "new_file.txt"
        result = ut.Write(str(f), "new content")
        assert "Successfully wrote" in result
        assert f.read_text() == "new content"

    def test_write_creates_parent_dirs(self, tools):
        ut, test_dir = tools
        f = test_dir / "sub" / "deep" / "file.txt"
        result = ut.Write(str(f), "nested content")
        assert "Successfully wrote" in result
        assert f.read_text() == "nested content"

    def test_write_to_directory_path(self, tools):
        ut, test_dir = tools
        subdir = test_dir / "subdir"
        subdir.mkdir()
        result = ut.Write(str(subdir), "content")
        assert "Error:" in result


class TestFetchUrl:
    def test_http_error(self, http_server):
        result = fetch_url(f"{http_server}/notfound", {"User-Agent": "Test Agent"})
        assert "Failed to fetch content: HTTP 404" in result

    def test_timeout(self, http_server):
        headers = {"User-Agent": "Test Agent"}
        result = fetch_url(f"{http_server}/slow", headers, timeout_seconds=0.01)
        assert result == "Failed to fetch content: Request timed out."

    def test_invalid_headers(self):
        result = fetch_url("http://example.com", 1, timeout_seconds=0.1)  # type: ignore[arg-type]
        assert result.startswith("Failed to fetch content:")

    def test_success_main_tag(self, http_server):
        result = fetch_url(f"{http_server}/", {"User-Agent": "Test"})
        assert "Hello from server" in result

    def test_article_tag_with_script_removed(self, http_server):
        result = fetch_url(f"{http_server}/article", {"User-Agent": "Test"})
        assert "Article content here" in result
        assert "var x" not in result

    def test_truncation(self, http_server):
        result = fetch_url(f"{http_server}/long", {"User-Agent": "Test"}, max_characters=50)
        assert "... [truncated]" in result

    def test_empty_content(self, http_server):
        result = fetch_url(f"{http_server}/empty", {"User-Agent": "Test"})
        assert result == "No readable content found."

    def test_request_exception_connection(self):
        result = fetch_url("http://invalid.invalid:1", {"User-Agent": "Test"}, timeout_seconds=5)
        assert result.startswith("Failed to fetch content:")

    def test_role_main(self, http_server):
        result = fetch_url(f"{http_server}/role-main", {"User-Agent": "Test"})
        assert "Role main content" in result

    def test_id_content(self, http_server):
        result = fetch_url(f"{http_server}/id-content", {"User-Agent": "Test"})
        assert "ID content area" in result

    def test_class_content(self, http_server):
        result = fetch_url(f"{http_server}/class-content", {"User-Agent": "Test"})
        assert "Class content area" in result

    def test_body_only(self, http_server):
        result = fetch_url(f"{http_server}/body-only", {"User-Agent": "Test"})
        assert "Body only content" in result

    def test_no_body_fallback(self, http_server):
        result = fetch_url(f"{http_server}/no-body", {"User-Agent": "Test"})
        assert "Bare paragraph" in result


class TestRenderPageWithPlaywright:
    def test_render_basic_page(self, http_server):
        html = _render_page_with_playwright(http_server + "/")
        assert "Hello from server" in html

    def test_render_with_wait_selector(self, http_server):
        html = _render_page_with_playwright(http_server + "/article", wait_selector="article")
        assert "Article content" in html

    def test_render_with_invalid_wait_selector(self, http_server):
        html = _render_page_with_playwright(http_server + "/", wait_selector="#nonexistent-element")
        assert "Hello from server" in html


class TestSearchWeb:
    def test_search_web_real(self):
        result = search_web("python programming language", max_results=1)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExtractLeadingCommandName:
    def test_unterminated_quote_returns_none(self):
        assert _extract_leading_command_name('"unterminated') is None

    def test_empty_string_returns_none(self):
        assert _extract_leading_command_name("") is None

    def test_only_env_vars_returns_none(self):
        assert _extract_leading_command_name("FOO=bar BAZ=qux") is None


class TestExtractCommandNames:
    def test_only_env_vars_segment(self):
        assert _extract_command_names("FOO=bar") == []

    def test_unterminated_quote_segment(self):
        assert _extract_command_names('"unterminated') == []

    def test_empty_pipe_segment(self):
        assert _extract_command_names("echo hi | | cat") == ["echo", "cat"]

    def test_heredoc_stripping(self):
        cmd = "cat << EOF\nhello world\nEOF"
        result = _strip_heredocs(cmd)
        assert "hello world" not in result


@pytest.fixture
def streaming_tools(temp_test_dir):
    streamed: list[str] = []
    ut = UsefulTools(stream_callback=streamed.append)
    return ut, temp_test_dir, streamed


class TestBashStreaming:
    def test_streaming_captures_output_lines(self, streaming_tools):
        ut, test_dir, streamed = streaming_tools
        test_file = test_dir / "lines.txt"
        test_file.write_text("line1\nline2\nline3\n")
        result = ut.Bash(f"cat {test_file}", "Stream cat")
        assert "line1" in result
        assert "line2" in result
        assert len(streamed) >= 3
        joined = "".join(streamed)
        assert "line1" in joined
        assert "line2" in joined
        assert "line3" in joined

    def test_streaming_handles_error(self, streaming_tools):
        ut, _, streamed = streaming_tools
        result = ut.Bash("false", "Failing command")
        assert "Error:" in result

    def test_streaming_timeout(self, streaming_tools):
        ut, _, _ = streaming_tools
        result = ut.Bash("sleep 10", "Slow command", timeout_seconds=0.1)
        assert result == "Error: Command execution timeout"

    def test_streaming_output_truncation(self, streaming_tools):
        ut, test_dir, streamed = streaming_tools
        big_file = test_dir / "big.txt"
        big_file.write_text("X" * 200)
        result = ut.Bash(f"cat {big_file}", "Cat big", max_output_chars=50)
        assert "truncated" in result
        assert len(streamed) >= 1

    def test_streaming_stderr_captured(self, streaming_tools):
        ut, _, streamed = streaming_tools
        ut.Bash("echo out && echo err >&2", "Mixed output")
        joined = "".join(streamed)
        assert "out" in joined
        assert "err" in joined

    def test_no_streaming_without_callback(self, temp_test_dir):
        ut = UsefulTools()
        assert ut.stream_callback is None
        result = ut.Bash("echo normal", "No streaming")
        assert "normal" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
