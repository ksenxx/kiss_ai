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

from kiss.core.useful_tools import (
    INLINE_CODE_FLAGS,
    UsefulTools,
    _extract_command_names,
    _extract_leading_command_name,
    _extract_paths_from_code,
    _extract_search_results,
    _is_safe_special_path,
    _render_page_with_playwright,
    _resolve_path,
    _strip_heredocs,
    fetch_url,
    parse_bash_command_paths,
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


class TestResolvePath:
    def test_invalid_path_returns_none(self):
        assert _resolve_path("\0") is None


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

    def test_cp_mv_source_and_dest(self, temp_test_dir):
        src = temp_test_dir / "src.txt"
        dst = temp_test_dir / "dst.txt"
        src.write_text("data")
        readable, writable = parse_bash_command_paths(f"cp {src} {dst}")
        assert str(src) in readable
        assert str(dst) in writable

    def test_mv_with_multiple_sources(self, temp_test_dir):
        s1 = temp_test_dir / "a.txt"
        s2 = temp_test_dir / "b.txt"
        dst = temp_test_dir / "dest_dir"
        s1.write_text("a")
        s2.write_text("b")
        dst.mkdir()
        readable, writable = parse_bash_command_paths(f"mv {s1} {s2} {dst}")
        assert str(s1) in readable
        assert str(s2) in readable
        assert str(dst) in writable

    def test_chmod_skips_mode(self, temp_test_dir):
        target = temp_test_dir / "script.sh"
        target.write_text("#!/bin/bash")
        _, writable = parse_bash_command_paths(f"chmod +x {target}")
        assert str(target) in writable

    def test_chmod_numeric_mode(self, temp_test_dir):
        target = temp_test_dir / "script.sh"
        target.write_text("#!/bin/bash")
        _, writable = parse_bash_command_paths(f"chmod 755 {target}")
        assert str(target) in writable

    def test_heredoc_stripping(self):
        cmd = "cat << EOF\nhello world\nEOF"
        result = _strip_heredocs(cmd)
        assert "hello world" not in result

    def test_shell_operator_tokens(self, temp_test_dir):
        f = temp_test_dir / "x.txt"
        f.write_text("x")
        readable, _ = parse_bash_command_paths(f"cat {f} && echo done")
        assert str(f) in readable

    def test_redirect_operator_token_skip(self, temp_test_dir):
        f = temp_test_dir / "out.txt"
        _, writable = parse_bash_command_paths(f"sort > {f}")
        assert str(f) in writable

    def test_write_command_other(self, temp_test_dir):
        f = temp_test_dir / "newdir"
        _, writable = parse_bash_command_paths(f"mkdir {f}")
        assert str(f) in writable

    def test_rm_write_command(self, temp_test_dir):
        f = temp_test_dir / "delete_me.txt"
        f.write_text("data")
        _, writable = parse_bash_command_paths(f"rm {f}")
        assert str(f) in writable

    def test_flag_with_path_argument(self, temp_test_dir):
        f = temp_test_dir / "file.txt"
        f.write_text("content")
        readable, _ = parse_bash_command_paths(f"grep -f /tmp/patterns {f}")
        assert str(f) in readable

    def test_input_redirect_safe_special_path(self):
        readable, _ = parse_bash_command_paths("cat < /dev/stdin")
        assert readable == []

    def test_write_redirect_empty_path(self):
        readable, writable = parse_bash_command_paths("echo hello >")
        assert writable == []


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

    def test_bash_read_denied(self, tools_sandbox):
        tools, _, _, test_dir = tools_sandbox
        outside = test_dir / "secret.txt"
        outside.write_text("secret")
        result = tools.Bash(f"cat {outside}", "Read outside")
        assert "Error: Access denied for reading" in result

    def test_bash_output_truncation(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        big_file = readable_dir / "big.txt"
        big_file.write_text("X" * 200)
        result = tools.Bash(f"cat {big_file}", "Cat big", max_output_chars=50)
        assert "truncated" in result

    def test_bash_called_process_error(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        result = tools.Bash("false", "Failing command")
        assert "Error:" in result

    def test_read_success(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        f = readable_dir / "hello.txt"
        f.write_text("hello world")
        result = tools.Read(str(f))
        assert result == "hello world"

    def test_read_access_denied(self, tools_sandbox):
        tools, _, _, test_dir = tools_sandbox
        outside = test_dir / "outside.txt"
        outside.write_text("forbidden")
        result = tools.Read(str(outside))
        assert "Error: Access denied for reading" in result

    def test_write_success(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        f = writable_dir / "new_file.txt"
        result = tools.Write(str(f), "new content")
        assert "Successfully wrote" in result
        assert f.read_text() == "new content"

    def test_write_access_denied(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        f = readable_dir / "forbidden.txt"
        result = tools.Write(str(f), "data")
        assert "Error: Access denied for writing" in result

    def test_write_creates_parent_dirs(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        f = writable_dir / "sub" / "deep" / "file.txt"
        result = tools.Write(str(f), "nested content")
        assert "Successfully wrote" in result
        assert f.read_text() == "nested content"

    def test_edit_success(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        f = writable_dir / "edit_me.txt"
        f.write_text("hello world")
        result = tools.Edit(str(f), "hello", "goodbye")
        assert "Successfully replaced" in result
        assert f.read_text() == "goodbye world"

    def test_edit_replace_all(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        f = writable_dir / "multi.txt"
        f.write_text("aaa bbb aaa")
        result = tools.Edit(str(f), "aaa", "ccc", replace_all=True)
        assert "Successfully replaced" in result
        assert f.read_text() == "ccc bbb ccc"

    def test_edit_access_denied(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        f = readable_dir / "nope.txt"
        f.write_text("data")
        result = tools.Edit(str(f), "data", "new")
        assert "Error: Access denied for writing" in result

    def test_multi_edit(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        f = writable_dir / "multi_edit.txt"
        f.write_text("foo bar")
        result = tools.MultiEdit(str(f), "foo", "baz")
        assert "Successfully replaced" in result
        assert f.read_text() == "baz bar"

    def test_edit_not_unique(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        f = writable_dir / "dup.txt"
        f.write_text("aaa\naaa\n")
        result = tools.Edit(str(f), "aaa", "ccc")
        assert "Error:" in result
        assert "not unique" in result

    def test_edit_not_unique_same_line(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        f = writable_dir / "dup_same_line.txt"
        f.write_text("aaa bbb aaa\n")
        result = tools.Edit(str(f), "aaa", "ccc")
        assert "Error:" in result
        assert "not unique" in result

    def test_bash_blocks_noclobber_redirect_outside_writable(self, tools_sandbox):
        tools, _, _, test_dir = tools_sandbox
        outside = test_dir / "outside_noclobber.txt"
        result = tools.Bash(f"echo blocked >| {outside}", "No-clobber redirect outside")
        assert "Error: Access denied for writing" in result
        assert not outside.exists()

    def test_bash_allows_numeric_append_redirect_in_writable(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        output = writable_dir / "append.txt"
        result = tools.Bash(f"echo hi 1>> {output}", "Append in writable path")
        assert "Error:" not in result
        assert output.read_text().strip() == "hi"


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


class TestIsSafeSpecialPath:
    def test_safe_paths(self):
        assert _is_safe_special_path("/dev/null")
        assert _is_safe_special_path("/dev/fd/3")
        assert _is_safe_special_path("/proc/self/fd/0")

    def test_unsafe_paths(self):
        assert not _is_safe_special_path("/tmp/file.txt")
        assert not _is_safe_special_path("/home/user/data")
        assert not _is_safe_special_path("/proc/self/root/etc/passwd")


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


class TestWriteError:
    def test_write_to_directory_path(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        subdir = writable_dir / "subdir"
        subdir.mkdir()
        result = tools.Write(str(subdir), "content")
        assert "Error:" in result


class TestBashEdgeCases:
    def test_bash_read_command_no_path_args(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        result = tools.Bash("echo hello", "Echo only")
        assert "hello" in result

    def test_bash_write_redirect_to_safe_special(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        result = tools.Bash("echo test > /dev/null", "Write to dev null")
        assert "Error:" not in result

    def test_bash_blocks_inline_python_outside_readable(self, tools_sandbox):
        tools, _, _, temp_test_dir = tools_sandbox
        outside = temp_test_dir / "outside_inline.txt"
        outside.write_text("secret")
        result = tools.Bash(
            f"python3 -c \"print(open('{outside}').read())\"",
            "Inline python outside",
        )
        assert "Error: Access denied" in result

    def test_bash_allows_inline_python_in_readable_dir(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        test_file = readable_dir / "data.txt"
        test_file.write_text("hello")
        result = tools.Bash(
            f"python3 -c \"print(open('{test_file}').read().strip())\"",
            "Inline python read",
        )
        assert "hello" in result

    def test_bash_blocks_inline_python_write_outside_readable(self, tools_sandbox):
        tools, _, _, temp_test_dir = tools_sandbox
        target = temp_test_dir / "outside_write.txt"
        result = tools.Bash(
            f"python3 -c \"open('{target}', 'w').write('x')\"",
            "Inline python write outside readable",
        )
        assert "Error: Access denied for reading" in result

    def test_bash_allows_inline_node_in_readable_dir(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        test_file = readable_dir / "data.txt"
        test_file.write_text("node-hello")
        result = tools.Bash(
            f"node -e \"console.log(require('fs').readFileSync('{test_file}', 'utf8').trim())\"",
            "Inline node read",
        )
        assert "node-hello" in result

    def test_bash_blocks_cd_to_non_readable_dir(self, tools_sandbox):
        tools, _, _, temp_test_dir = tools_sandbox
        outside_dir = temp_test_dir / "outside_dir"
        outside_dir.mkdir()
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret")
        result = tools.Bash(f"cd {outside_dir} && cat secret.txt", "cd bypass")
        assert "Error: Access denied for reading" in result

    def test_bash_allows_cd_to_readable_dir(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        result = tools.Bash(f"cd {readable_dir} && pwd", "cd to readable")
        assert str(readable_dir) in result

    def test_bash_allows_dynamic_variable_expansion(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        result = tools.Bash('echo "$HOME"', "dynamic shell expansion")
        assert "Error: Command contains unsafe shell expansion" not in result

    def test_bash_allows_interpreter_without_inline_exec(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        result = tools.Bash("python --version", "python version")
        assert "Error: Command 'python' is not allowed" not in result


class TestParseBashEdgeCases:
    def test_cd_extracts_readable_path(self, temp_test_dir):
        target = temp_test_dir / "subdir"
        target.mkdir()
        readable, writable = parse_bash_command_paths(f"cd {target}")
        assert str(target) in readable
        assert writable == []

    def test_inline_python_extracts_paths(self, temp_test_dir):
        target = temp_test_dir / "file.txt"
        cmd = f"python3 -c \"open('{target}').read()\""
        readable, writable = parse_bash_command_paths(cmd)
        assert str(target) in readable

    def test_inline_node_eval_extracts_paths(self, temp_test_dir):
        target = temp_test_dir / "data.json"
        cmd = f"node --eval \"require('fs').readFileSync('{target}')\""
        readable, writable = parse_bash_command_paths(cmd)
        assert str(target) in readable

    def test_inline_code_no_paths(self):
        readable, writable = parse_bash_command_paths("python3 -c \"print('hello')\"")
        assert readable == [] and writable == []

    def test_read_command_with_no_file_args(self):
        readable, writable = parse_bash_command_paths("ls")
        assert readable == [] and writable == []

    def test_cp_single_arg(self, temp_test_dir):
        f = temp_test_dir / "single.txt"
        f.write_text("data")
        readable, writable = parse_bash_command_paths(f"cp {f}")
        assert str(f) in writable

    def test_rsync_source_and_dest(self, temp_test_dir):
        src = temp_test_dir / "rsrc"
        dst = temp_test_dir / "rdst"
        src.mkdir()
        dst.mkdir()
        readable, writable = parse_bash_command_paths(f"rsync {src}/ {dst}/")
        assert str(src) in readable
        assert str(dst) in writable

    def test_token_with_equals_skipped(self):
        readable, writable = parse_bash_command_paths("make CC=gcc")
        assert readable == [] and writable == []

    def test_redirect_token_in_command_tokens(self, temp_test_dir):
        f = temp_test_dir / "redir.txt"
        readable, writable = parse_bash_command_paths(f"sort -o {f} << EOF\ndata\nEOF")
        assert isinstance(readable, list)

    def test_null_byte_path_in_write_redirect(self):
        readable, writable = parse_bash_command_paths("echo hello > '\x00invalid'")
        assert writable == []

    def test_null_byte_path_in_input_redirect(self):
        readable, writable = parse_bash_command_paths("cat < '\x00invalid'")
        assert readable == []

    def test_null_byte_in_read_command_path(self):
        readable, writable = parse_bash_command_paths("cat '\x00invalid'")
        assert readable == []

    def test_null_byte_in_write_command_path(self):
        readable, writable = parse_bash_command_paths("mkdir '\x00invalid'")
        assert writable == []

    def test_null_byte_in_cp_source_and_dest(self):
        readable, writable = parse_bash_command_paths("cp '\x00a' '\x00b'")
        assert readable == [] and writable == []

    def test_null_byte_in_dd_paths(self):
        readable, writable = parse_bash_command_paths("dd if=/\x00a of=/\x00b")
        assert readable == [] and writable == []

    def test_null_byte_in_tee_path(self):
        readable, writable = parse_bash_command_paths("tee '\x00invalid'")
        assert writable == []

    def test_shell_operator_in_token_loop(self):
        readable, writable = parse_bash_command_paths("cat ; echo done")
        assert readable == [] and writable == []

    def test_redirect_operator_as_last_token(self):
        readable, writable = parse_bash_command_paths("sort >")
        assert writable == []

    def test_noclobber_redirect_path_is_parsed(self, temp_test_dir):
        output = temp_test_dir / "noclobber.txt"
        _, writable = parse_bash_command_paths(f"echo hi >| {output}")
        assert writable == [str(output)]

    def test_numeric_append_redirect_path_is_parsed_once(self, temp_test_dir):
        output = temp_test_dir / "append.txt"
        _, writable = parse_bash_command_paths(f"echo hi 1>> {output}")
        assert writable == [str(output)]

    def test_background_operator_in_read_cmd(self, temp_test_dir):
        f = temp_test_dir / "bg.txt"
        f.write_text("data")
        readable, _ = parse_bash_command_paths(f"cat {f} &")
        assert str(f) in readable

    def test_write_command_no_path_args(self):
        readable, writable = parse_bash_command_paths("touch")
        assert writable == []


class TestEnvVarPrefixBypass:
    """Tests for Bug #8: env-var prefix assignments must not bypass path detection."""

    def test_env_prefix_cat_detects_readable_path(self, temp_test_dir):
        f = temp_test_dir / "secret.txt"
        f.write_text("secret")
        readable, _ = parse_bash_command_paths(f"FOO=bar cat {f}")
        assert str(f) in readable

    def test_env_prefix_touch_detects_writable_path(self, temp_test_dir):
        f = temp_test_dir / "new.txt"
        _, writable = parse_bash_command_paths(f"FOO=bar touch {f}")
        assert str(f) in writable

    def test_multiple_env_prefixes(self, temp_test_dir):
        f = temp_test_dir / "data.txt"
        f.write_text("data")
        readable, _ = parse_bash_command_paths(f"A=1 B=2 cat {f}")
        assert str(f) in readable

    def test_env_prefix_only_no_command(self):
        readable, writable = parse_bash_command_paths("FOO=bar BAZ=qux")
        assert readable == [] and writable == []

    def test_env_prefix_cp_source_and_dest(self, temp_test_dir):
        src = temp_test_dir / "src.txt"
        dst = temp_test_dir / "dst.txt"
        src.write_text("data")
        readable, writable = parse_bash_command_paths(f"VAR=x cp {src} {dst}")
        assert str(src) in readable
        assert str(dst) in writable

    def test_env_prefix_write_redirect(self, temp_test_dir):
        f = temp_test_dir / "out.txt"
        _, writable = parse_bash_command_paths(f"VAR=x sort > {f}")
        assert str(f) in writable

    def test_bash_blocks_env_prefix_read_outside_sandbox(self, tools_sandbox):
        tools, _, _, test_dir = tools_sandbox
        outside = test_dir / "secret.txt"
        outside.write_text("secret")
        result = tools.Bash(f"FOO=bar cat {outside}", "env prefix read bypass")
        assert "Error: Access denied for reading" in result

    def test_bash_blocks_env_prefix_write_outside_sandbox(self, tools_sandbox):
        tools, _, _, test_dir = tools_sandbox
        outside = test_dir / "outside.txt"
        result = tools.Bash(f"FOO=bar touch {outside}", "env prefix write bypass")
        assert "Error: Access denied for writing" in result


class TestResolvePathExtended:
    def test_resolves_relative_path(self, temp_test_dir):
        result = _resolve_path("somefile.txt")
        assert result is not None
        assert str(temp_test_dir) in result

    def test_resolves_absolute_path(self):
        result = _resolve_path("/tmp/test.txt")
        assert result is not None
        # On macOS /tmp -> /private/tmp
        assert "tmp/test.txt" in result

    def test_resolves_dotdot_path(self, temp_test_dir):
        result = _resolve_path(str(temp_test_dir / "sub" / ".." / "file.txt"))
        assert result is not None
        assert ".." not in result


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


class TestExtractPathsFromCode:
    def test_single_quoted_absolute_path(self):
        paths = _extract_paths_from_code("open('/tmp/foo.txt').read()")
        assert len(paths) == 1
        assert "tmp/foo.txt" in paths[0]

    def test_double_quoted_absolute_path(self):
        paths = _extract_paths_from_code('open("/tmp/bar.txt").read()')
        assert len(paths) == 1
        assert "tmp/bar.txt" in paths[0]

    def test_relative_path_dot_slash(self, temp_test_dir):
        paths = _extract_paths_from_code("open('./data/file.txt')")
        assert len(paths) == 1
        assert "data/file.txt" in paths[0]

    def test_relative_path_dotdot_slash(self, temp_test_dir):
        paths = _extract_paths_from_code("open('../parent/file.txt')")
        assert len(paths) == 1
        assert "parent/file.txt" in paths[0]

    def test_multiple_paths(self):
        code = "open('/tmp/a.txt'); open('/tmp/b.txt')"
        paths = _extract_paths_from_code(code)
        assert len(paths) == 2

    def test_no_paths(self):
        assert _extract_paths_from_code("print('hello')") == []

    def test_safe_special_path_filtered(self):
        assert _extract_paths_from_code("open('/dev/null')") == []

    def test_inline_code_flags_has_expected_keys(self):
        assert "python" in INLINE_CODE_FLAGS
        assert "python3" in INLINE_CODE_FLAGS
        assert "node" in INLINE_CODE_FLAGS
        assert "-c" in INLINE_CODE_FLAGS["python"]
        assert "-e" in INLINE_CODE_FLAGS["node"]
        assert "--eval" in INLINE_CODE_FLAGS["node"]


class TestEditScriptLiteralGrep:
    def test_edit_with_regex_special_chars_in_new_string(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        f = writable_dir / "regex_test.txt"
        f.write_text("hello world")
        result = tools.Edit(str(f), "hello", "he.*llo")
        assert "Successfully replaced" in result
        assert f.read_text() == "he.*llo world"


@pytest.fixture
def streaming_sandbox(temp_test_dir):
    readable_dir = temp_test_dir / "readable"
    writable_dir = temp_test_dir / "writable"
    readable_dir.mkdir()
    writable_dir.mkdir()
    streamed: list[str] = []
    tools = UsefulTools(
        base_dir=str(temp_test_dir),
        readable_paths=[str(readable_dir)],
        writable_paths=[str(writable_dir)],
        stream_callback=streamed.append,
    )
    return tools, readable_dir, writable_dir, temp_test_dir, streamed


class TestBashStreaming:
    def test_streaming_captures_output_lines(self, streaming_sandbox):
        tools, readable_dir, _, _, streamed = streaming_sandbox
        test_file = readable_dir / "lines.txt"
        test_file.write_text("line1\nline2\nline3\n")
        result = tools.Bash(f"cat {test_file}", "Stream cat")
        assert "line1" in result
        assert "line2" in result
        assert len(streamed) >= 3
        joined = "".join(streamed)
        assert "line1" in joined
        assert "line2" in joined
        assert "line3" in joined

    def test_streaming_returns_full_output(self, streaming_sandbox):
        tools, _, _, _, streamed = streaming_sandbox
        result = tools.Bash("echo hello && echo world", "Two echoes")
        assert "hello" in result
        assert "world" in result
        assert len(streamed) >= 2

    def test_streaming_handles_error(self, streaming_sandbox):
        tools, _, _, _, streamed = streaming_sandbox
        result = tools.Bash("false", "Failing command")
        assert "Error:" in result

    def test_streaming_timeout(self, streaming_sandbox):
        tools, _, _, _, _ = streaming_sandbox
        result = tools.Bash("sleep 10", "Slow command", timeout_seconds=0.1)
        assert result == "Error: Command execution timeout"

    def test_streaming_output_truncation(self, streaming_sandbox):
        tools, readable_dir, _, _, streamed = streaming_sandbox
        big_file = readable_dir / "big.txt"
        big_file.write_text("X" * 200)
        result = tools.Bash(f"cat {big_file}", "Cat big", max_output_chars=50)
        assert "truncated" in result
        assert len(streamed) >= 1

    def test_streaming_permission_denied(self, streaming_sandbox):
        tools, _, _, test_dir, streamed = streaming_sandbox
        outside = test_dir / "secret.txt"
        outside.write_text("secret")
        result = tools.Bash(f"cat {outside}", "Read outside")
        assert "Error: Access denied for reading" in result
        assert len(streamed) == 0

    def test_streaming_echo_no_file_access(self, streaming_sandbox):
        tools, _, _, _, streamed = streaming_sandbox
        result = tools.Bash("echo streaming_test", "Simple echo")
        assert "streaming_test" in result
        joined = "".join(streamed)
        assert "streaming_test" in joined

    def test_streaming_stderr_captured(self, streaming_sandbox):
        tools, _, _, _, streamed = streaming_sandbox
        tools.Bash("echo out && echo err >&2", "Mixed output")
        joined = "".join(streamed)
        assert "out" in joined
        assert "err" in joined

    def test_no_streaming_without_callback(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        assert tools.stream_callback is None
        result = tools.Bash("echo normal", "No streaming")
        assert "normal" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
