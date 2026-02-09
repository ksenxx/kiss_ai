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
    SAFE_SPECIAL_PATHS,
    SAFE_SPECIAL_PREFIXES,
    UsefulTools,
    _extract_directory,
    _extract_search_results,
    _is_safe_special_path,
    _render_page_with_playwright,
    _strip_heredocs,
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
    @pytest.mark.parametrize(
        "path_type,setup_func",
        [
            ("existing_file", lambda d: (d / "test.txt", "write_text")),
            ("existing_dir", lambda d: (d / "subdir", "mkdir")),
            ("nonexistent_with_ext", lambda d: (d / "nonexistent.txt", None)),
            ("nonexistent_no_ext", lambda d: (d / "nonexistent", None)),
        ],
    )
    def test_path_types(self, temp_test_dir, path_type, setup_func):
        path, action = setup_func(temp_test_dir)
        if action == "write_text":
            path.write_text("content")
        elif action == "mkdir":
            path.mkdir()
        assert _extract_directory(str(path)) == str(path)

    def test_trailing_slash(self, temp_test_dir):
        result = _extract_directory(str(temp_test_dir / "newdir/"))
        assert result == str(temp_test_dir / "newdir")

    def test_relative_path(self, temp_test_dir):
        result = _extract_directory("relative/path.txt")
        assert result == str((temp_test_dir / "relative/path.txt").resolve())

    def test_empty_string(self, temp_test_dir):
        assert _extract_directory("") == str(temp_test_dir)

    def test_invalid_path_returns_none(self):
        assert _extract_directory("\0") is None


class TestParseBashCommandPaths:
    @pytest.mark.parametrize(
        "cmd_template,creates_file,expected_readable,expected_writable",
        [
            ("cat {file}", True, ["{file}"], []),
            ("echo hello > {file}", False, [], ["{file}"]),
            ("echo hello >> {file}", False, [], ["{file}"]),
            ("cat < {file}", True, ["{file}"], []),
            ("touch {file}", False, [], ["{file}"]),
            ("mkdir {file}", False, [], ["{file}"]),
            ("rm {file}", True, [], ["{file}"]),
            ("tee {file}", False, [], ["{file}"]),
        ],
    )
    def test_single_file_commands(
        self, temp_test_dir, cmd_template, creates_file, expected_readable, expected_writable
    ):
        test_file = temp_test_dir / "test.txt"
        if creates_file:
            test_file.write_text("content")

        cmd = cmd_template.format(file=test_file)
        readable, writable = parse_bash_command_paths(cmd)

        expected_r = [str(test_file)] if "{file}" in str(expected_readable) else []
        expected_w = [str(test_file)] if "{file}" in str(expected_writable) else []

        assert readable == expected_r
        if expected_w:
            assert str(test_file) in writable

    @pytest.mark.parametrize("cmd_name", ["cp", "mv"])
    def test_copy_move_commands(self, temp_test_dir, cmd_name):
        src = temp_test_dir / "source.txt"
        dst = temp_test_dir / "dest.txt"
        src.write_text("content")

        readable, writable = parse_bash_command_paths(f"{cmd_name} {src} {dst}")
        assert readable == [str(src)]
        assert writable == [str(dst)]

    def test_dd_command(self, temp_test_dir):
        input_file = temp_test_dir / "input.bin"
        output_file = temp_test_dir / "output.bin"
        input_file.write_bytes(b"data")

        readable, writable = parse_bash_command_paths(f"dd if={input_file} of={output_file}")
        assert readable == [str(input_file)]
        assert writable == [str(output_file)]

    def test_pipe_command(self, temp_test_dir):
        file1 = temp_test_dir / "file1.txt"
        file2 = temp_test_dir / "file2.txt"
        file1.write_text("content")

        readable, writable = parse_bash_command_paths(f"cat {file1} | grep pattern > {file2}")
        assert str(file1) in readable
        assert writable == [str(file2)]

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

    def test_safe_paths_in_subshell(self):
        readable, writable = parse_bash_command_paths("$(echo hello > /dev/null)")
        assert readable == [] and writable == []
        readable, writable = parse_bash_command_paths("echo hello 2>/dev/null)")
        assert readable == [] and writable == []

    def test_multiple_files(self, temp_test_dir):
        file1 = temp_test_dir / "file1.txt"
        file2 = temp_test_dir / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        readable, writable = parse_bash_command_paths(f"cat {file1} {file2}")
        assert sorted(readable) == sorted([str(file1), str(file2)])
        assert writable == []

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

    def test_heredoc_paths_not_extracted(self, temp_test_dir):
        cmd = (
            f"cat > {temp_test_dir}/out.sh << 'EOF'\n"
            "#!/bin/bash\necho /etc/passwd\ngrep /var/log/syslog\nEOF"
        )
        readable, writable = parse_bash_command_paths(cmd)
        assert "/etc/passwd" not in " ".join(readable)
        assert "/var/log/syslog" not in " ".join(readable)

    def test_heredoc_redirect_still_detected(self, temp_test_dir):
        cmd = f"cat > {temp_test_dir}/out.sh << 'EOF'\nsome content\nEOF"
        _, writable = parse_bash_command_paths(cmd)
        assert str(temp_test_dir / "out.sh") in writable

    def test_and_separator(self, temp_test_dir):
        dir_path = temp_test_dir / "newdir"
        readable, writable = parse_bash_command_paths(f"mkdir -p {dir_path} && ls {dir_path}")
        assert str(dir_path) in writable
        assert str(dir_path) in readable
        # && itself should NOT appear as a path
        for p in writable + readable:
            assert "&&" not in p

    def test_or_separator(self, temp_test_dir):
        file1 = temp_test_dir / "a.txt"
        file2 = temp_test_dir / "b.txt"
        readable, writable = parse_bash_command_paths(f"cat {file1} || cat {file2}")
        assert str(file1) in readable
        assert str(file2) in readable
        for p in readable:
            assert "||" not in p

    def test_semicolon_separator(self, temp_test_dir):
        dir_path = temp_test_dir / "mydir"
        file_path = temp_test_dir / "out.txt"
        readable, writable = parse_bash_command_paths(
            f"mkdir {dir_path}; echo hello > {file_path}"
        )
        assert str(dir_path) in writable
        assert str(file_path) in writable

    def test_chained_and_operators(self, temp_test_dir):
        d = temp_test_dir / "target"
        readable, writable = parse_bash_command_paths(
            f"mkdir -p {d} && touch {d}/file.txt && ls {d}"
        )
        assert str(d) in writable
        assert str(d) in readable
        for p in writable + readable:
            assert "&&" not in p


class TestStripHeredocs:
    def test_strips_single_quoted_heredoc(self):
        cmd = "cat > file.sh << 'EOF'\n#!/bin/bash\necho /etc/passwd\nEOF"
        assert "/etc/passwd" not in _strip_heredocs(cmd)
        assert "cat > file.sh" in _strip_heredocs(cmd)

    def test_strips_unquoted_heredoc(self):
        cmd = "cat > file.sh << EOF\necho /var/log/syslog\nEOF"
        assert "/var/log/syslog" not in _strip_heredocs(cmd)

    def test_strips_double_quoted_heredoc(self):
        cmd = 'cat > file.sh << "EOF"\necho /secret/path\nEOF'
        assert "/secret/path" not in _strip_heredocs(cmd)

    def test_strips_indented_heredoc(self):
        cmd = "cat > file.sh <<- MARKER\n\techo /private/data\n\tMARKER"
        assert "/private/data" not in _strip_heredocs(cmd)

    def test_no_heredoc_unchanged(self):
        cmd = "echo hello > /dev/null"
        assert _strip_heredocs(cmd) == cmd

    def test_multiple_heredocs(self):
        cmd = "cat << 'A'\n/path/one\nA\ncat << 'B'\n/path/two\nB"
        result = _strip_heredocs(cmd)
        assert "/path/one" not in result
        assert "/path/two" not in result


class TestIsSafeSpecialPath:
    @pytest.mark.parametrize("path", sorted(SAFE_SPECIAL_PATHS))
    def test_exact_safe_paths(self, path):
        assert _is_safe_special_path(path) is True

    @pytest.mark.parametrize("prefix", SAFE_SPECIAL_PREFIXES)
    def test_prefix_safe_paths(self, prefix):
        assert _is_safe_special_path(prefix + "something") is True

    def test_safe_paths_with_trailing_punctuation(self):
        assert _is_safe_special_path("/dev/null)") is True
        assert _is_safe_special_path("/dev/null')") is True
        assert _is_safe_special_path('/dev/null")') is True
        assert _is_safe_special_path("/dev/null;") is True
        assert _is_safe_special_path("/dev/fd/3)") is True

    def test_non_safe_paths(self):
        assert _is_safe_special_path("/etc/passwd") is False
        assert _is_safe_special_path("/dev/sda") is False
        assert _is_safe_special_path("/tmp/file") is False
        assert _is_safe_special_path("/proc/1/status") is False
        assert _is_safe_special_path("") is False


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
    def test_init_creates_base_dir(self, temp_test_dir):
        new_base = temp_test_dir / "new_base"
        readable = temp_test_dir / "readable"
        writable = temp_test_dir / "writable"
        readable.mkdir()
        writable.mkdir()

        tools = UsefulTools(
            base_dir=str(new_base),
            readable_paths=[str(readable)],
            writable_paths=[str(writable)],
        )
        assert new_base.exists()
        assert tools.base_dir == str(new_base.resolve())

    def test_bash_safe_command(self, tools_sandbox):
        tools, _, _, _ = tools_sandbox
        assert "hello" in tools.Bash("echo hello", "Test echo")

    @pytest.mark.parametrize(
        "cmd,error_type",
        [
            ("cat /etc/passwd", "reading"),
            ("cat {outside}", "reading"),
            ("touch {outside}", "writing"),
        ],
    )
    def test_permission_denied(self, tools_sandbox, cmd, error_type):
        tools, _, _, test_dir = tools_sandbox
        if "{outside}" in cmd:
            outside_file = test_dir / "outside.txt"
            if "cat" in cmd:
                outside_file.write_text("secret")
            cmd = cmd.format(outside=outside_file)

        result = tools.Bash(cmd, "Test permission")
        assert f"Error: Access denied for {error_type}" in result

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

    @pytest.mark.parametrize("replace_all", [False, True])
    def test_edit(self, tools_sandbox, replace_all):
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "edit_test.txt"

        if replace_all:
            test_file.write_text("foo bar foo baz foo\n")
            old, new, expected_count = "foo", "qux", 3
        else:
            test_file.write_text("Hello World\nGoodbye World\n")
            old, new, expected_count = "Hello World", "Hi World", 1

        tools.Edit(
            file_path=str(test_file), old_string=old,
            new_string=new, replace_all=replace_all,
        )

        content = test_file.read_text()
        assert content.count(new) == expected_count
        assert old not in content

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
    def test_read_allowed_file(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        test_file = readable_dir / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")
        result = tools.Read(str(test_file))
        assert result == "line1\nline2\nline3\n"

    def test_read_access_denied(self, tools_sandbox):
        tools, _, _, test_dir = tools_sandbox
        outside_file = test_dir / "outside.txt"
        outside_file.write_text("secret")
        result = tools.Read(str(outside_file))
        assert "Error: Access denied" in result

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

    def test_read_empty_file(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        test_file = readable_dir / "empty.txt"
        test_file.write_text("")
        result = tools.Read(str(test_file))
        assert result == ""

    def test_read_writable_dir_denied(self, tools_sandbox):
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "file.txt"
        test_file.write_text("content")
        result = tools.Read(str(test_file))
        assert "Error: Access denied" in result

    def test_read_within_max_lines(self, tools_sandbox):
        tools, readable_dir, _, _ = tools_sandbox
        test_file = readable_dir / "small.txt"
        test_file.write_text("a\nb\nc\n")
        result = tools.Read(str(test_file), max_lines=100)
        assert result == "a\nb\nc\n"
        assert "[truncated" not in result


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
