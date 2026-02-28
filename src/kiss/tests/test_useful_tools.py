"""Tests for useful_tools.py module."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from kiss.agents.assistant.useful_tools import (
    UsefulTools,
    _extract_command_names,
    _extract_leading_command_name,
    _strip_heredocs,
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

    def test_edit_replace_all(self, tools):
        ut, test_dir = tools
        f = test_dir / "replace_all.txt"
        f.write_text("foo bar foo")
        result = ut.Edit(str(f), "foo", "baz", replace_all=True)
        assert "Successfully replaced" in result
        assert f.read_text() == "baz bar baz"

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
