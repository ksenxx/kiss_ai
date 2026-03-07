"""Tests for useful_tools.py module."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import (
    DISALLOWED_BASH_COMMANDS,
    UsefulTools,
    _extract_command_names,
    _strip_heredocs,
    _truncate_output,
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
        assert result.startswith("Error (exit code")

    def test_bash_error_includes_output(self, tools):
        ut, test_dir = tools
        script = test_dir / "fail.sh"
        script.write_text("#!/bin/bash\necho partial_output\nexit 1\n")
        script.chmod(0o755)
        result = ut.Bash(str(script), "Failing with output")
        assert "partial_output" in result
        assert "exit code 1" in result

    def test_bash_stderr_visible(self, tools):
        ut, test_dir = tools
        script = test_dir / "stderr.sh"
        script.write_text("#!/bin/bash\necho err_msg >&2\nexit 1\n")
        script.chmod(0o755)
        result = ut.Bash(str(script), "stderr test")
        assert "err_msg" in result

    def test_bash_stderr_visible_on_success(self, tools):
        ut, test_dir = tools
        script = test_dir / "warn.sh"
        script.write_text("#!/bin/bash\necho out_data\necho warn_data >&2\n")
        script.chmod(0o755)
        result = ut.Bash(str(script), "stderr on success")
        assert "out_data" in result
        assert "warn_data" in result

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
        assert "characters" in result
        assert f.read_text() == "new content"

    def test_write_to_directory_path(self, tools):
        ut, test_dir = tools
        subdir = test_dir / "subdir"
        subdir.mkdir()
        result = ut.Write(str(subdir), "content")
        assert "Error:" in result


class TestExtractCommandNames:
    def test_only_env_vars_segment(self):
        assert _extract_command_names("FOO=bar") == []

    def test_unterminated_quote_segment(self):
        assert _extract_command_names('"unterminated') == []

    def test_empty_pipe_segment(self):
        assert _extract_command_names("echo hi | | cat") == ["echo", "cat"]


@pytest.fixture
def streaming_tools(temp_test_dir):
    streamed: list[str] = []
    ut = UsefulTools(stream_callback=streamed.append)
    return ut, temp_test_dir, streamed


class TestBashStreaming:

    def test_streaming_handles_error(self, streaming_tools):
        ut, _, streamed = streaming_tools
        result = ut.Bash("false", "Failing command")
        assert result.startswith("Error (exit code")

    def test_streaming_error_includes_output(self, streaming_tools):
        ut, test_dir, streamed = streaming_tools
        script = test_dir / "sfail.sh"
        script.write_text("#!/bin/bash\necho stream_partial\nexit 1\n")
        script.chmod(0o755)
        result = ut.Bash(str(script), "Streaming fail with output")
        assert any("stream_partial" in s for s in streamed)
        assert "stream_partial" in result
        assert "exit code 1" in result

    def test_streaming_timeout(self, streaming_tools):
        ut, _, _ = streaming_tools
        result = ut.Bash("sleep 10", "Slow command", timeout_seconds=0.1)
        assert result == "Error: Command execution timeout"

    def test_streaming_timeout_compound_command(self, streaming_tools):
        """Compound commands spawn child processes that must also be killed on timeout."""
        ut, _, _ = streaming_tools
        result = ut.Bash(
            "sleep 30; echo done",
            "Compound cmd timeout",
            timeout_seconds=0.5,
        )
        assert result == "Error: Command execution timeout"

    def test_streaming_output(self, streaming_tools):
        ut, _, streamed = streaming_tools
        result = ut.Bash("echo hello", "echo test")
        assert "hello" in result
        assert any("hello" in s for s in streamed)

    def test_streaming_output_truncation(self, streaming_tools):
        ut, test_dir, _ = streaming_tools
        big_file = test_dir / "big.txt"
        big_file.write_text("X" * 200)
        result = ut.Bash(
            f"cat {big_file}", "Cat big", max_output_chars=50
        )
        assert "truncated" in result


class TestBashTimeoutNonStreaming:

    def test_timeout_compound_command(self, tools):
        """Non-streaming: compound commands with child processes timeout correctly."""
        ut, _ = tools
        result = ut.Bash(
            "sleep 30; echo done",
            "Compound cmd timeout",
            timeout_seconds=0.5,
        )
        assert result == "Error: Command execution timeout"


class TestAdversarial:
    """Adversarial tests to try to break the Popen/killpg changes."""

    def test_timeout_kills_background_children(self, tools):
        """Background child processes must be killed on timeout, not just the shell."""
        ut, test_dir = tools
        pid_file = test_dir / "bg_child.pid"
        script = test_dir / "bg_child.sh"
        script.write_text(
            f"#!/bin/bash\nsleep 100 &\necho $! > {pid_file}\nwait\n"
        )
        script.chmod(0o755)
        result = ut.Bash(str(script), "bg child timeout", timeout_seconds=1)
        assert result == "Error: Command execution timeout"
        import time
        time.sleep(0.5)
        if pid_file.exists():
            child_pid = int(pid_file.read_text().strip())
            try:
                os.kill(child_pid, 0)
                os.kill(child_pid, signal.SIGKILL)
                raise AssertionError(
                    f"Background child {child_pid} survived timeout!"
                )
            except ProcessLookupError:
                pass

    def test_streaming_timeout_kills_background_children(self, streaming_tools):
        """Streaming: background child processes must be killed on timeout."""
        ut, test_dir, _ = streaming_tools
        pid_file = test_dir / "sbg_child.pid"
        script = test_dir / "sbg_child.sh"
        script.write_text(
            f"#!/bin/bash\nsleep 100 &\necho $! > {pid_file}\nwait\n"
        )
        script.chmod(0o755)
        result = ut.Bash(str(script), "streaming bg child", timeout_seconds=1)
        assert result == "Error: Command execution timeout"
        import time
        time.sleep(0.5)
        if pid_file.exists():
            child_pid = int(pid_file.read_text().strip())
            try:
                os.kill(child_pid, 0)
                os.kill(child_pid, signal.SIGKILL)
                raise AssertionError(
                    f"Streaming bg child {child_pid} survived timeout!"
                )
            except ProcessLookupError:
                pass

    def test_nonstreaming_interrupt_kills_child(self, tools):
        """KeyboardInterrupt during non-streaming Bash must kill the child process group."""
        import _thread
        import time

        ut, test_dir = tools
        pid_file = test_dir / "interrupt_child.pid"
        script = test_dir / "interrupt_target.sh"
        script.write_text(
            f"#!/bin/bash\necho $$ > {pid_file}\nsleep 100\n"
        )
        script.chmod(0o755)

        child_pid = None

        def send_interrupt():
            nonlocal child_pid
            for _ in range(20):
                time.sleep(0.1)
                if pid_file.exists():
                    child_pid = int(pid_file.read_text().strip())
                    break
            if child_pid:
                _thread.interrupt_main()

        import threading
        t = threading.Thread(target=send_interrupt, daemon=True)
        t.start()
        try:
            ut.Bash(str(script), "interruptible", timeout_seconds=30)
        except KeyboardInterrupt:
            pass
        t.join(timeout=5)

        if child_pid is None:
            pytest.skip("Script didn't start in time")

        time.sleep(0.3)
        try:
            os.kill(child_pid, 0)
            os.kill(child_pid, signal.SIGKILL)
            raise AssertionError(
                f"Child process {child_pid} survived KeyboardInterrupt!"
            )
        except ProcessLookupError:
            pass

    def test_streaming_interrupt_kills_child(self, streaming_tools):
        """Streaming path properly kills child on KeyboardInterrupt (has BaseException handler)."""
        import _thread
        import time

        ut, test_dir, _ = streaming_tools
        pid_file = test_dir / "sinterrupt_child.pid"
        script = test_dir / "sinterrupt_target.sh"
        script.write_text(
            f"#!/bin/bash\necho $$ > {pid_file}\nsleep 100\n"
        )
        script.chmod(0o755)

        child_pid = None

        def send_interrupt():
            nonlocal child_pid
            for _ in range(20):
                time.sleep(0.1)
                if pid_file.exists():
                    child_pid = int(pid_file.read_text().strip())
                    break
            if child_pid:
                _thread.interrupt_main()

        import threading
        t = threading.Thread(target=send_interrupt, daemon=True)
        t.start()
        try:
            ut.Bash(str(script), "streaming interruptible", timeout_seconds=30)
        except KeyboardInterrupt:
            pass
        t.join(timeout=5)

        if child_pid is None:
            pytest.skip("Script didn't start in time")

        time.sleep(0.3)
        alive = False
        try:
            os.kill(child_pid, 0)
            alive = True
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        assert not alive, (
            f"Streaming child {child_pid} survived KeyboardInterrupt!"
        )

    def test_nonstreaming_communicate_no_timeout_after_kill(self, tools):
        """After killpg, the second process.communicate() has no timeout.

        This test verifies the timeout path works, but documents that if killpg
        were to fail silently, the second communicate() would hang forever.
        """
        ut, test_dir = tools
        script = test_dir / "deep_tree.sh"
        script.write_text(
            "#!/bin/bash\n"
            "for i in $(seq 5); do sleep 100 & done\n"
            "wait\n"
        )
        script.chmod(0o755)
        result = ut.Bash(str(script), "deep tree timeout", timeout_seconds=1)
        assert result == "Error: Command execution timeout"

    def test_error_message_no_trailing_newline_on_empty_output(self, tools):
        """Commands with no output should not have a trailing newline in the error."""
        ut, _ = tools
        result = ut.Bash("false", "empty output error")
        assert result == "Error (exit code 1):"

    def test_no_zombie_processes_after_rapid_calls(self, tools):
        """Rapid successive Bash calls shouldn't leak zombie processes."""
        import time
        ut, _ = tools
        for _ in range(20):
            ut.Bash("echo ok", "rapid fire")
        time.sleep(0.5)
        zombies = os.popen("ps aux | grep defunct | grep -v grep").read().strip()
        assert zombies == "", f"Zombie processes found:\n{zombies}"


import signal


class TestBugs:
    """Tests that expose bugs in useful_tools.py."""

    # --- Bug 1: Disallowed-command bypass via quoted delimiters ---
    # _extract_command_names splits on ; && || BEFORE considering shell
    # quoting. A disallowed command whose quoted argument happens to contain
    # one of those delimiters is torn apart mid-quote, both halves fail
    # shlex.split, and the command name is never seen.

    def test_eval_bypass_semicolon_in_quotes(self):
        names = _extract_command_names("eval 'echo hi; echo bye'")
        assert "eval" in names, (
            "eval should be detected even when its argument contains ';' in quotes"
        )

    def test_eval_bypass_and_in_quotes(self):
        names = _extract_command_names("eval 'a && b'")
        assert "eval" in names, (
            "eval should be detected even when its argument contains '&&' in quotes"
        )

    def test_exec_bypass_semicolon_in_double_quotes(self):
        names = _extract_command_names('exec "cmd; other"')
        assert "exec" in names, (
            "exec should be detected even when its argument contains ';' in double quotes"
        )

    def test_dot_bypass_or_in_quotes(self):
        names = _extract_command_names(". 'script || exit'")
        assert "." in names, (
            ". should be detected even when its argument contains '||' in quotes"
        )

    # --- Bug 2: `source` not in DISALLOWED_BASH_COMMANDS ---

    def test_source_is_blocked(self):
        assert "source" in DISALLOWED_BASH_COMMANDS, (
            "source is the bash synonym of . and should be disallowed"
        )

    def test_source_detected_by_extract(self):
        names = _extract_command_names("source /etc/profile")
        assert "source" in names

    # --- Bug 3: _strip_heredocs fails with trailing tokens on heredoc line ---

    def test_strip_heredoc_with_pipe_after_marker(self):
        cmd = "cat <<EOF | grep foo\nhello world\nEOF"
        result = _strip_heredocs(cmd)
        assert "hello world" not in result, (
            "heredoc body should be stripped even when pipe follows <<EOF"
        )

    def test_strip_heredoc_with_semicolon_after_marker(self):
        cmd = "cat <<EOF; echo done\nhello\nEOF"
        result = _strip_heredocs(cmd)
        assert "hello" not in result, (
            "heredoc body should be stripped even when ; follows <<EOF"
        )

    # --- Bug 4: _truncate_output exceeds max_chars ---

    def test_truncate_output_respects_limit(self):
        big = "X" * 200
        result = _truncate_output(big, 50)
        assert len(result) <= 50, (
            f"_truncate_output returned {len(result)} chars, exceeding max_chars=50"
        )

    def test_truncate_output_accurate_dropped_count(self):
        big = "A" * 100 + "Z" * 100
        result = _truncate_output(big, 60)
        assert "truncated" in result
        import re
        m = re.search(r"truncated (\d+) chars", result)
        assert m is not None
        claimed_dropped = int(m.group(1))
        kept_chars = len(result) - len(m.group(0)) - len("\n\n... [] ...\n\n")
        actual_dropped = 200 - kept_chars
        assert claimed_dropped == actual_dropped, (
            f"Message says {claimed_dropped} dropped but actually {actual_dropped}"
        )

    def test_truncate_output_tiny_limit(self):
        big = "X" * 200
        result = _truncate_output(big, 5)
        assert len(result) <= 5

    def test_truncate_output_limit_smaller_than_message(self):
        big = "X" * 1000
        result = _truncate_output(big, 10)
        assert len(result) <= 10

    # --- Bug 5: _strip_heredocs empty heredoc ---

    def test_strip_heredoc_empty_body(self):
        cmd = "cat <<EOF\nEOF"
        result = _strip_heredocs(cmd)
        assert "EOF" not in result or result.strip() == "cat"

    # --- Bug 6: _strip_heredocs delimiter mid-line match ---

    def test_strip_heredoc_delimiter_in_body_line(self):
        cmd = "cat <<EOF\nEOF is a marker\nactual body\nEOF"
        result = _strip_heredocs(cmd)
        assert "EOF is a marker" not in result
        assert "actual body" not in result

    # --- Bug 7: subshell bypass of disallowed commands ---

    def test_subshell_eval_detected(self):
        names = _extract_command_names("(eval foo)")
        assert "eval" in names

    def test_brace_group_eval_detected(self):
        names = _extract_command_names("{ eval foo; }")
        assert "eval" in names

    def test_subshell_exec_detected(self):
        names = _extract_command_names("(exec /bin/sh)")
        assert "exec" in names

    # --- Bug 8: redirect bypass of disallowed commands ---

    def test_redirect_before_eval(self):
        names = _extract_command_names("< /dev/null eval foo")
        assert "eval" in names

    def test_fd_redirect_before_source(self):
        names = _extract_command_names("2>/dev/null source script.sh")
        assert "source" in names

    def test_redirect_output_before_exec(self):
        names = _extract_command_names("> /tmp/log exec cmd")
        assert "exec" in names

    # --- Bug 9: EDIT_SCRIPT grep without -- separator ---

    def test_edit_new_string_starts_with_dash(self, tools):
        ut, test_dir = tools
        f = test_dir / "dash_edit.txt"
        f.write_text("old_value")
        result = ut.Edit(str(f), "old_value", "-e new_value")
        assert "Successfully replaced" in result
        assert f.read_text() == "-e new_value"

    # --- Bug 10: & (background) as command separator bypass ---

    def test_background_eval_detected(self):
        names = _extract_command_names("echo hi & eval foo")
        assert "eval" in names

    def test_background_source_detected(self):
        names = _extract_command_names("sleep 1 & source script.sh")
        assert "source" in names

    def test_double_ampersand_still_works(self):
        names = _extract_command_names("echo hi && echo bye")
        assert names == ["echo", "echo"]

    def test_redirect_ampersand_not_split(self):
        names = _extract_command_names("echo hi &>/dev/null")
        assert names == ["echo"]

    def test_fd_redirect_not_split(self):
        names = _extract_command_names("echo hi 2>&1")
        assert names == ["echo"]

    # --- Bug 11: newline as command separator bypass ---

    def test_newline_separator_eval_detected(self):
        names = _extract_command_names("echo hi\neval foo")
        assert "eval" in names

    def test_newline_separator_exec_detected(self):
        names = _extract_command_names("ls\nexec /bin/sh")
        assert "exec" in names

    # --- Bug 12: |& pipe bypass ---

    def test_pipe_stderr_eval_detected(self):
        names = _extract_command_names("echo hi |& eval")
        assert "eval" in names

    # --- Bug 13: Write says bytes not characters ---

    def test_write_unicode_reports_characters(self, tools):
        ut, test_dir = tools
        f = test_dir / "unicode.txt"
        content = "hello \U0001f600 world"
        result = ut.Write(str(f), content)
        assert "characters" in result
        assert str(len(content)) in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
