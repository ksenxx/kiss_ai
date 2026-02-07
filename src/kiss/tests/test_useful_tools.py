"""Tests for useful_tools.py module.

These tests run in a temporary directory and do not use any mocking.
"""

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
    fetch_url,
    parse_bash_command_paths,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_test_dir():
    """Provides a temporary directory for tests."""
    test_dir = Path(tempfile.mkdtemp()).resolve()
    original_dir = Path.cwd()
    os.chdir(test_dir)
    yield test_dir
    os.chdir(original_dir)
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def tools_sandbox(temp_test_dir):
    """Provides a UsefulTools instance with readable/writable directories."""
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
    """Starts a local HTTP server for fetch_url tests."""

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


# =============================================================================
# _extract_directory tests
# =============================================================================


class TestExtractDirectory:
    """Test the _extract_directory function."""

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
        """Test _extract_directory with various path types."""
        path, action = setup_func(temp_test_dir)
        if action == "write_text":
            path.write_text("content")
        elif action == "mkdir":
            path.mkdir()

        result = _extract_directory(str(path))
        assert result == str(path)

    def test_trailing_slash(self, temp_test_dir):
        """Test that trailing slashes are normalized."""
        test_path = temp_test_dir / "newdir/"
        result = _extract_directory(str(test_path))
        assert result == str(temp_test_dir / "newdir")

    def test_relative_path(self, temp_test_dir):
        """Test that relative paths are resolved to absolute."""
        result = _extract_directory("relative/path.txt")
        expected = str((temp_test_dir / "relative/path.txt").resolve())
        assert result == expected

    def test_empty_string(self, temp_test_dir):
        """Test that empty string resolves to current directory."""
        result = _extract_directory("")
        assert result == str(temp_test_dir)

    def test_invalid_path_returns_none(self):
        """Test that invalid paths return None."""
        assert _extract_directory("\0") is None


# =============================================================================
# parse_bash_command_paths tests
# =============================================================================


class TestParseBashCommandPaths:
    """Test parse_bash_command_paths function."""

    @pytest.mark.parametrize(
        "cmd_template,creates_file,expected_readable,expected_writable",
        [
            # Read commands
            ("cat {file}", True, ["{file}"], []),
            # Write redirections
            ("echo hello > {file}", False, [], ["{file}"]),
            ("echo hello >> {file}", False, [], ["{file}"]),
            # Input redirection
            ("cat < {file}", True, ["{file}"], []),
            # Write commands
            ("touch {file}", False, [], ["{file}"]),
            ("mkdir {file}", False, [], ["{file}"]),
            ("rm {file}", True, [], ["{file}"]),
            ("tee {file}", False, [], ["{file}"]),
        ],
    )
    def test_single_file_commands(
        self, temp_test_dir, cmd_template, creates_file, expected_readable, expected_writable
    ):
        """Test parsing commands with single file arguments."""
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

    @pytest.mark.parametrize(
        "cmd_name",
        ["cp", "mv"],
    )
    def test_copy_move_commands(self, temp_test_dir, cmd_name):
        """Test cp and mv commands identify src as readable, dst as writable."""
        src = temp_test_dir / "source.txt"
        dst = temp_test_dir / "dest.txt"
        src.write_text("content")

        cmd = f"{cmd_name} {src} {dst}"
        readable, writable = parse_bash_command_paths(cmd)

        assert readable == [str(src)]
        assert writable == [str(dst)]

    def test_dd_command(self, temp_test_dir):
        """Test dd command with if= and of= parameters."""
        input_file = temp_test_dir / "input.bin"
        output_file = temp_test_dir / "output.bin"
        input_file.write_bytes(b"data")

        cmd = f"dd if={input_file} of={output_file}"
        readable, writable = parse_bash_command_paths(cmd)

        assert readable == [str(input_file)]
        assert writable == [str(output_file)]

    def test_pipe_command(self, temp_test_dir):
        """Test piped commands with mixed read/write."""
        file1 = temp_test_dir / "file1.txt"
        file2 = temp_test_dir / "file2.txt"
        file1.write_text("content")

        cmd = f"cat {file1} | grep pattern > {file2}"
        readable, writable = parse_bash_command_paths(cmd)

        assert str(file1) in readable
        assert writable == [str(file2)]

    def test_dev_null_ignored(self):
        """Test that /dev/null is filtered out."""
        readable, writable = parse_bash_command_paths("echo hello > /dev/null")
        assert readable == []
        assert writable == []

    @pytest.mark.parametrize("path", sorted(SAFE_SPECIAL_PATHS))
    def test_safe_special_paths_ignored_in_redirect(self, path):
        """Test that all safe special paths are filtered from write redirects."""
        readable, writable = parse_bash_command_paths(f"echo hello > {path}")
        assert writable == []

    @pytest.mark.parametrize("path", sorted(SAFE_SPECIAL_PATHS))
    def test_safe_special_paths_ignored_in_input_redirect(self, path):
        """Test that all safe special paths are filtered from input redirects."""
        readable, writable = parse_bash_command_paths(f"cat < {path}")
        assert readable == []

    @pytest.mark.parametrize("path", sorted(SAFE_SPECIAL_PATHS))
    def test_safe_special_paths_ignored_in_command_args(self, path):
        """Test that safe special paths are filtered from command arguments."""
        readable, writable = parse_bash_command_paths(f"cat {path}")
        assert readable == []

    def test_safe_prefix_paths_ignored_in_redirect(self):
        """Test that safe prefix paths (/dev/fd/*, /proc/self/*) are filtered."""
        readable, writable = parse_bash_command_paths("echo hello > /dev/fd/3")
        assert writable == []
        readable, writable = parse_bash_command_paths("cat < /proc/self/status")
        assert readable == []

    def test_safe_special_paths_ignored_in_dd(self):
        """Test that safe special paths are filtered from dd if= and of=."""
        readable, writable = parse_bash_command_paths("dd if=/dev/zero of=/dev/null")
        assert readable == []
        assert writable == []

        cmd = "dd if=/dev/urandom of=/dev/null bs=1024 count=10"
        readable, writable = parse_bash_command_paths(cmd)
        assert readable == []
        assert writable == []

    def test_multiple_files(self, temp_test_dir):
        """Test command with multiple file arguments."""
        file1 = temp_test_dir / "file1.txt"
        file2 = temp_test_dir / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        readable, writable = parse_bash_command_paths(f"cat {file1} {file2}")
        assert sorted(readable) == sorted([str(file1), str(file2)])
        assert writable == []

    def test_flags_ignored(self, temp_test_dir):
        """Test that command flags are not treated as paths."""
        test_file = temp_test_dir / "test.txt"
        test_file.write_text("content")

        readable, writable = parse_bash_command_paths(f"grep -i -n pattern {test_file}")
        assert readable == [str(test_file)]
        assert writable == []

    def test_unterminated_quotes_fallback_split(self):
        """Test shlex failure falls back to basic splitting."""
        readable, writable = parse_bash_command_paths('echo "unterminated')
        assert readable == []
        assert writable == []

    def test_empty_pipe_parts(self):
        """Test empty pipe parts are ignored."""
        readable, writable = parse_bash_command_paths(" | ")
        assert readable == []
        assert writable == []

    def test_invalid_command_type(self):
        """Test invalid command input returns empty paths."""
        readable, writable = parse_bash_command_paths(1)  # type: ignore[arg-type]
        assert readable == []
        assert writable == []


class TestIsSafeSpecialPath:
    """Test the _is_safe_special_path function."""

    @pytest.mark.parametrize("path", sorted(SAFE_SPECIAL_PATHS))
    def test_exact_safe_paths(self, path):
        assert _is_safe_special_path(path) is True

    @pytest.mark.parametrize("prefix", SAFE_SPECIAL_PREFIXES)
    def test_prefix_safe_paths(self, prefix):
        assert _is_safe_special_path(prefix + "something") is True

    def test_non_safe_paths(self):
        assert _is_safe_special_path("/etc/passwd") is False
        assert _is_safe_special_path("/dev/sda") is False
        assert _is_safe_special_path("/tmp/file") is False
        assert _is_safe_special_path("/proc/1/status") is False
        assert _is_safe_special_path("") is False


class TestSearchResultExtraction:
    """Tests for _extract_search_results."""

    def test_skips_invalid_and_blocked_domains(self):
        """Test that invalid and blocked domains are skipped."""
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


# =============================================================================
# UsefulTools tests
# =============================================================================


class TestUsefulTools:
    """Test the UsefulTools class."""

    def test_init_creates_base_dir(self, temp_test_dir):
        """Test that UsefulTools creates base_dir if it doesn't exist."""
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
        assert new_base.is_dir()
        assert tools.base_dir == str(new_base.resolve())

    def test_bash_safe_command(self, tools_sandbox):
        """Test that safe commands execute successfully."""
        tools, _, _, _ = tools_sandbox
        result = tools.Bash("echo hello", "Test echo")
        assert "hello" in result

    @pytest.mark.parametrize(
        "cmd,error_type",
        [
            ("cat /etc/passwd", "reading"),
            ("cat {outside}", "reading"),
            ("touch {outside}", "writing"),
        ],
    )
    def test_permission_denied(self, tools_sandbox, cmd, error_type):
        """Test that access outside allowed paths is denied."""
        tools, _, _, test_dir = tools_sandbox

        # Create outside file if needed for read test
        if "{outside}" in cmd:
            outside_file = test_dir / "outside.txt"
            if "cat" in cmd:
                outside_file.write_text("secret")
            cmd = cmd.format(outside=outside_file)

        result = tools.Bash(cmd, "Test permission")
        assert f"Error: Access denied for {error_type}" in result

    def test_bash_read_allowed(self, tools_sandbox):
        """Test that reading from readable_paths is allowed."""
        tools, readable_dir, _, _ = tools_sandbox
        test_file = readable_dir / "test.txt"
        test_file.write_text("readable content")

        result = tools.Bash(f"cat {test_file}", "Read allowed")
        assert "readable content" in result

    def test_bash_write_allowed(self, tools_sandbox):
        """Test that writing to writable_paths is allowed."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "output.txt"

        result = tools.Bash(f"echo 'writable content' > {test_file}", "Write allowed")
        assert "Error:" not in result
        assert test_file.exists()
        assert test_file.read_text().strip() == "writable content"

    def test_bash_allows_dev_null_redirect(self, tools_sandbox):
        """Test that redirecting to /dev/null is allowed without whitelisting."""
        tools, _, _, _ = tools_sandbox
        result = tools.Bash("echo hello > /dev/null", "Redirect to /dev/null")
        assert "Error:" not in result

    def test_bash_allows_reading_dev_urandom(self, tools_sandbox):
        """Test that reading from /dev/urandom is allowed without whitelisting."""
        tools, _, _, _ = tools_sandbox
        cmd = "dd if=/dev/urandom bs=8 count=1 2>/dev/null | od -An -tx1"
        result = tools.Bash(cmd, "Read /dev/urandom")
        assert "Error: Access denied" not in result

    def test_bash_allows_dev_zero(self, tools_sandbox):
        """Test that reading from /dev/zero is allowed without whitelisting."""
        tools, _, _, _ = tools_sandbox
        cmd = "dd if=/dev/zero bs=8 count=1 2>/dev/null | od -An -tx1"
        result = tools.Bash(cmd, "Read /dev/zero")
        assert "Error: Access denied" not in result

    def test_bash_timeout(self, tools_sandbox):
        """Test Bash timeout handling."""
        tools, _, _, _ = tools_sandbox
        result = tools.Bash("sleep 1", "Timeout test", timeout_seconds=0.01)
        assert result == "Error: Command execution timeout"

    def test_bash_called_process_error(self, tools_sandbox):
        """Test Bash returns errors on non-zero exit."""
        tools, _, _, _ = tools_sandbox
        result = tools.Bash("false", "Failure test")
        assert result.startswith("Error:")

    def test_edit_called_process_error(self, tools_sandbox):
        """Test Edit returns error when string is missing."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "missing.txt"
        test_file.write_text("alpha beta")

        result = tools.Edit(str(test_file), "gamma", "delta")
        assert result.startswith("Error:")
        assert "String not found" in result

    def test_edit_general_exception(self, tools_sandbox):
        """Test Edit returns error on invalid arguments."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "bad.txt"
        test_file.write_text("alpha")

        result = tools.Edit(str(test_file), "alpha\0", "beta")
        assert result.startswith("Error:")

    def test_edit_timeout(self, tools_sandbox):
        """Test Edit timeout handling."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "timeout_edit.txt"
        test_file.write_text("a" * 5_000_000)

        result = tools.Edit(
            str(test_file),
            "a",
            "b",
            replace_all=True,
            timeout_seconds=0.0001,
        )
        assert result == "Error: Command execution timeout"

    def test_multiedit_called_process_error(self, tools_sandbox):
        """Test MultiEdit returns error when string is missing."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "missing_multi.txt"
        test_file.write_text("alpha beta")

        result = tools.MultiEdit(str(test_file), "gamma", "delta")
        assert result.startswith("Error:")
        assert "String not found" in result

    def test_multiedit_general_exception(self, tools_sandbox):
        """Test MultiEdit returns error on invalid arguments."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "bad_multi.txt"
        test_file.write_text("alpha")

        result = tools.MultiEdit(str(test_file), "alpha\0", "beta")
        assert result.startswith("Error:")

    def test_multiedit_timeout(self, tools_sandbox):
        """Test MultiEdit timeout handling."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "timeout_multiedit.txt"
        test_file.write_text("a" * 5_000_000)

        result = tools.MultiEdit(
            str(test_file),
            "a",
            "b",
            replace_all=True,
            timeout_seconds=0.0001,
        )
        assert result == "Error: Command execution timeout"

    def test_edit_unlink_failure(self, tools_sandbox, tmp_path, monkeypatch):
        """Test Edit cleanup handles unlink errors gracefully."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "unlink_fail.txt"
        test_file.write_text("alpha beta")

        temp_dir = tmp_path / "tempdir"
        temp_dir.mkdir()
        monkeypatch.setenv("TMPDIR", str(temp_dir))

        def lock_dir_once():
            end_time = time.time() + 2
            while time.time() < end_time:
                if any(temp_dir.iterdir()):
                    os.chmod(temp_dir, 0o500)
                    return
                time.sleep(0.01)

        thread = threading.Thread(target=lock_dir_once, daemon=True)
        thread.start()
        try:
            result = tools.Edit(str(test_file), "alpha", "gamma")
            assert "Successfully" in result or result.startswith("Error:")
        finally:
            thread.join(timeout=2)
            os.chmod(temp_dir, 0o700)

    def test_multiedit_unlink_failure(self, tools_sandbox, tmp_path, monkeypatch):
        """Test MultiEdit cleanup handles unlink errors gracefully."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "unlink_fail_multi.txt"
        test_file.write_text("alpha beta")

        temp_dir = tmp_path / "tempdir_multi"
        temp_dir.mkdir()
        monkeypatch.setenv("TMPDIR", str(temp_dir))

        def lock_dir_once():
            end_time = time.time() + 2
            while time.time() < end_time:
                if any(temp_dir.iterdir()):
                    os.chmod(temp_dir, 0o500)
                    return
                time.sleep(0.01)

        thread = threading.Thread(target=lock_dir_once, daemon=True)
        thread.start()
        try:
            result = tools.MultiEdit(str(test_file), "alpha", "gamma")
            assert "Successfully" in result or result.startswith("Error:")
        finally:
            thread.join(timeout=2)
            os.chmod(temp_dir, 0o700)


class TestFetchUrl:
    """Tests for fetch_url error handling."""

    def test_http_error(self, http_server):
        """Test HTTP error handling."""
        headers = {"User-Agent": "Test Agent"}
        result = fetch_url(f"{http_server}/notfound", headers)
        assert "Failed to fetch content: HTTP 404" in result

    def test_timeout(self, http_server):
        """Test timeout handling."""
        headers = {"User-Agent": "Test Agent"}
        result = fetch_url(
            f"{http_server}/slow",
            headers,
            timeout_seconds=0.01,
        )
        assert result == "Failed to fetch content: Request timed out."

    def test_request_exception(self):
        """Test request exception handling."""
        headers = {"User-Agent": "Test Agent"}
        result = fetch_url("http://127.0.0.1:1", headers, timeout_seconds=0.1)
        assert result.startswith("Failed to fetch content:")

    def test_generic_exception(self):
        """Test generic exception handling."""
        result = fetch_url("http://example.com", 1, timeout_seconds=0.1)  # type: ignore[arg-type]
        assert result.startswith("Failed to fetch content:")


class TestRenderPageWithPlaywright:
    """Tests for _render_page_with_playwright."""

    @pytest.mark.timeout(60)
    def test_render_local_file(self, tmp_path):
        """Test rendering a local file with Playwright."""
        html_file = tmp_path / "page.html"
        html_file.write_text("<html><main>Hello Playwright</main></html>")
        content = _render_page_with_playwright(f"file://{html_file}", wait_selector="#missing")
        assert "Hello Playwright" in content

    @pytest.mark.parametrize("replace_all", [False, True])
    def test_edit(self, tools_sandbox, replace_all):
        """Test Edit method with replace_all parameter."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "edit_test.txt"

        if replace_all:
            test_file.write_text("foo bar foo baz foo\n")
            old, new = "foo", "qux"
            expected_count = 3
        else:
            test_file.write_text("Hello World\nGoodbye World\n")
            old, new = "Hello World", "Hi World"
            expected_count = 1

        tools.Edit(
            file_path=str(test_file),
            old_string=old,
            new_string=new,
            replace_all=replace_all,
        )

        content = test_file.read_text()
        assert content.count(new) == expected_count
        assert old not in content

    @pytest.mark.parametrize("replace_all", [False, True])
    def test_multiedit(self, tools_sandbox, replace_all):
        """Test MultiEdit method with replace_all parameter."""
        tools, _, writable_dir, _ = tools_sandbox
        test_file = writable_dir / "multiedit_test.txt"

        if replace_all:
            test_file.write_text("test test test\n")
            old, new = "test", "pass"
            expected_count = 3
        else:
            test_file.write_text("Alpha Beta\nGamma Delta\n")
            old, new = "Alpha Beta", "Alpha Omega"
            expected_count = 1

        tools.MultiEdit(
            file_path=str(test_file),
            old_string=old,
            new_string=new,
            replace_all=replace_all,
        )

        content = test_file.read_text()
        assert content.count(new) == expected_count
        assert old not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
