"""Tests for useful_tools.py module."""

import os
import shutil
import signal
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import (
    DISALLOWED_BASH_COMMANDS,
    UsefulTools,
    _extract_command_names,
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

    def test_write_to_directory_path(self, tools):
        ut, test_dir = tools
        subdir = test_dir / "subdir"
        subdir.mkdir()
        result = ut.Write(str(subdir), "content")
        assert "Error:" in result


class TestExtractCommandNames:
    def test_unterminated_quote_segment(self):
        assert _extract_command_names('"unterminated') == []

    def test_empty_pipe_segment(self):
        assert _extract_command_names("echo hi | | cat") == ["echo", "cat"]


@pytest.fixture
def streaming_tools(temp_test_dir):
    streamed: list[str] = []
    ut = UsefulTools(stream_callback=streamed.append)
    return ut, temp_test_dir, streamed


@pytest.fixture(params=[False, True], ids=["nonstreaming", "streaming"])
def any_tools(request, temp_test_dir):
    if request.param:
        return UsefulTools(stream_callback=lambda _: None), temp_test_dir
    return UsefulTools(), temp_test_dir


class TestBashBothPaths:
    """Tests that apply identically to both streaming and non-streaming Bash paths."""

    def test_error_exit_code(self, any_tools):
        ut, _ = any_tools
        result = ut.Bash("false", "Failing command")
        assert result.startswith("Error (exit code")

    def test_output_truncation(self, any_tools):
        ut, test_dir = any_tools
        big_file = test_dir / "big.txt"
        big_file.write_text("X" * 200)
        result = ut.Bash(f"cat {big_file}", "Cat big", max_output_chars=50)
        assert "truncated" in result

    def test_timeout_compound_command(self, any_tools):
        ut, _ = any_tools
        result = ut.Bash(
            "sleep 30; echo done",
            "Compound cmd timeout",
            timeout_seconds=0.5,
        )
        assert result == "Error: Command execution timeout"


class TestAdversarial:
    """Adversarial tests to try to break the Popen/killpg changes."""

    def test_interrupt_kills_child(self, any_tools):
        """KeyboardInterrupt must kill the child process group."""
        import _thread
        import threading
        import time

        ut, test_dir = any_tools
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
        alive = False
        try:
            os.kill(child_pid, 0)
            alive = True
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        assert not alive, f"Child {child_pid} survived KeyboardInterrupt!"

class TestBugs:
    """Tests that expose bugs in useful_tools.py."""

    # --- Bug 1: Disallowed-command bypass via quoted delimiters ---
    # _extract_command_names splits on ; && || BEFORE considering shell
    # quoting. A disallowed command whose quoted argument happens to contain
    # one of those delimiters is torn apart mid-quote, both halves fail
    # shlex.split, and the command name is never seen.

    # --- Bug 2: `source` not in DISALLOWED_BASH_COMMANDS ---

    def test_source_is_blocked(self):
        assert "source" in DISALLOWED_BASH_COMMANDS, (
            "source is the bash synonym of . and should be disallowed"
        )

    # --- Bug 3: _strip_heredocs fails with trailing tokens on heredoc line ---

    # --- Bug 4: _truncate_output exceeds max_chars ---

    def test_truncate_output_tiny_limit(self):
        big = "X" * 200
        result = _truncate_output(big, 5)
        assert len(result) <= 5

    # --- Bug 5: _strip_heredocs empty heredoc ---

    # --- Bug 6: _strip_heredocs delimiter mid-line match ---

    # --- Bug 7: subshell bypass of disallowed commands ---

    def test_brace_group_eval_detected(self):
        names = _extract_command_names("{ eval foo; }")
        assert "eval" in names

    # --- Bug 8: redirect bypass of disallowed commands ---

    def test_fd_redirect_before_source(self):
        names = _extract_command_names("2>/dev/null source script.sh")
        assert "source" in names

    def test_redirect_output_before_exec(self):
        names = _extract_command_names("> /tmp/log exec cmd")
        assert "exec" in names

    # --- Bug 9: EDIT_SCRIPT grep without -- separator ---

    # --- Bug 10: & (background) as command separator bypass ---

    # --- Bug 11: newline as command separator bypass ---

    # --- Bug 12: |& pipe bypass ---

    # --- Bug 13: Write says bytes not characters ---

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
