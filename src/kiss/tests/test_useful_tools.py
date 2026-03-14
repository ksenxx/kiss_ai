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
    _detect_shell_prefix,
    _extract_command_names,
    _extract_leading_command_name,
    _rewrite_windows_bash_command,
    _strip_heredocs,
    _truncate_output,
)

_WINDOWS_WITH_GIT = os.name == "nt" and shutil.which("git") is not None

# Build shell commands with Python so tests are portable across cmd/sh.


def _py_cmd(code: str) -> str:
    escaped = code.replace('"', '\\"')
    # Use interpreter from PATH so command works in both cmd and Git Bash.
    return f'python -c "{escaped}"'


def _read_file_cmd(path: Path) -> str:
    return _py_cmd(
        f"from pathlib import Path; print(Path({path.as_posix()!r}).read_text(), end='')"
    )


def _write_text_cmd(path: Path, text: str) -> str:
    return _py_cmd(f"from pathlib import Path; Path({path.as_posix()!r}).write_text({text!r})")


def _to_git_bash_path(path: Path) -> str:
    resolved = str(path.resolve())
    drive, tail = os.path.splitdrive(resolved)
    if not drive:
        return resolved.replace("\\", "/")
    return f"/{drive[0].lower()}{tail.replace('\\', '/')}"


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
        assert "readable content" in ut.Bash(_read_file_cmd(test_file), "Read file")

    def test_bash_write_file(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "output.txt"
        result = ut.Bash(_write_text_cmd(test_file, "writable content"), "Write file")
        assert "Error:" not in result
        assert test_file.read_text().strip() == "writable content"

    def test_bash_timeout(self, tools):
        ut, _ = tools
        result = ut.Bash(
            _py_cmd("import time; time.sleep(1)"),
            "Timeout test",
            timeout_seconds=0.01,
        )
        assert result == "Error: Command execution timeout"

    def test_bash_output_truncation(self, tools):
        ut, test_dir = tools
        big_file = test_dir / "big.txt"
        big_file.write_text("X" * 200)
        result = ut.Bash(_read_file_cmd(big_file), "Cat big", max_output_chars=50)
        assert "truncated" in result

    def test_bash_called_process_error(self, tools):
        ut, _ = tools
        result = ut.Bash(_py_cmd("import sys; sys.exit(1)"), "Failing command")
        assert "Error:" in result

    def test_bash_disallowed_command(self, tools):
        ut, _ = tools
        result = ut.Bash("eval echo hi", "Disallowed")
        assert "Error: Command 'eval' is not allowed" in result

    def test_edit_string_not_found(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "missing.txt"
        test_file.write_text("alpha beta")
        ut.Read(str(test_file))
        result = ut.Edit(str(test_file), "gamma", "delta")
        assert result.startswith("Error:")
        assert "String not found" in result

    def test_edit_timeout(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "timeout_edit.txt"
        test_file.write_text("a" * 5_000_000)
        ut.Read(str(test_file))
        result = ut.Edit(str(test_file), "a", "b", replace_all=True, timeout_seconds=0.0001)
        assert result == "Error: Command execution timeout"

    def test_edit_success(self, tools):
        ut, test_dir = tools
        f = test_dir / "edit_me.txt"
        f.write_text("hello world")
        ut.Read(str(f))
        result = ut.Edit(str(f), "hello", "goodbye")
        assert "Successfully replaced" in result
        assert f.read_text() == "goodbye world"

    def test_edit_replace_all(self, tools):
        ut, test_dir = tools
        f = test_dir / "multi.txt"
        f.write_text("aaa bbb aaa")
        ut.Read(str(f))
        result = ut.Edit(str(f), "aaa", "ccc", replace_all=True)
        assert "Successfully replaced" in result
        assert f.read_text() == "ccc bbb ccc"

    def test_edit_not_unique(self, tools):
        ut, test_dir = tools
        f = test_dir / "dup.txt"
        f.write_text("aaa\naaa\n")
        ut.Read(str(f))
        result = ut.Edit(str(f), "aaa", "ccc")
        assert "Error:" in result
        assert "not unique" in result

    def test_edit_replace_all_different_content(self, tools):
        ut, test_dir = tools
        f = test_dir / "replace_all.txt"
        f.write_text("foo bar foo")
        ut.Read(str(f))
        result = ut.Edit(str(f), "foo", "baz", replace_all=True)
        assert "Successfully replaced" in result
        assert f.read_text() == "baz bar baz"

    def test_edit_requires_read_first(self, tools):
        ut, test_dir = tools
        f = test_dir / "needs_read.txt"
        f.write_text("hello world")

        result = ut.Edit(str(f), "hello", "goodbye")

        assert result.startswith("Error:")
        assert "must be read with Read()" in result

    def test_edit_fails_if_file_changed_since_read(self, tools):
        ut, test_dir = tools
        f = test_dir / "stale.txt"
        f.write_text("hello world")
        ut.Read(str(f))
        f.write_text("hello there")

        result = ut.Edit(str(f), "hello", "goodbye")

        assert result.startswith("Error:")
        assert "changed since last Read" in result

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

    def test_read_non_utf8_text_uses_lossy_fallback(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "legacy.md"
        test_file.write_bytes(b"alpha\x9dbeta")

        result = ut.Read(str(test_file))

        assert "alpha" in result
        assert "beta" in result
        assert "Error:" not in result

    def test_read_binary_file_reports_error(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02")

        result = ut.Read(str(test_file))

        assert result == "Error: Binary file cannot be displayed as text"

    @pytest.mark.skipif(not _WINDOWS_WITH_GIT, reason="requires Windows with git installed")
    def test_read_non_utf8_text_no_charmap_error_on_windows(self, tools):
        ut, test_dir = tools
        test_file = test_dir / "windows-legacy.md"
        test_file.write_bytes(b"alpha\x9dbeta")

        result = ut.Read(str(test_file))

        assert "Error:" not in result
        assert "alpha" in result
        assert "beta" in result

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

    def test_write_refuses_existing_file(self, tools):
        ut, test_dir = tools
        f = test_dir / "existing.txt"
        f.write_text("old content")

        result = ut.Write(str(f), "new content")

        assert result.startswith("Error:")
        assert "only creates new files" in result
        assert f.read_text() == "old content"

    def test_write_to_directory_path(self, tools):
        ut, test_dir = tools
        subdir = test_dir / "subdir"
        subdir.mkdir()
        result = ut.Write(str(subdir), "content")
        assert "Error:" in result

    @pytest.mark.skipif(os.name != "nt", reason="Windows path normalization behavior")
    def test_write_accepts_git_bash_style_drive_path(self, tools):
        ut, test_dir = tools
        target = test_dir / "git_bash_write.txt"
        result = ut.Write(_to_git_bash_path(target), "hello")
        assert "Successfully wrote" in result
        assert target.read_text() == "hello"

    @pytest.mark.skipif(os.name != "nt", reason="Windows path normalization behavior")
    def test_read_accepts_git_bash_style_drive_path(self, tools):
        ut, test_dir = tools
        target = test_dir / "git_bash_read.txt"
        target.write_text("from git bash path")
        result = ut.Read(_to_git_bash_path(target))
        assert result == "from git bash path"

    @pytest.mark.skipif(os.name != "nt", reason="Windows path normalization behavior")
    def test_edit_accepts_git_bash_style_drive_path(self, tools):
        ut, test_dir = tools
        target = test_dir / "git_bash_edit.txt"
        target.write_text("alpha beta")
        ut.Read(_to_git_bash_path(target))
        result = ut.Edit(_to_git_bash_path(target), "alpha", "omega")
        assert "Successfully replaced" in result
        assert target.read_text() == "omega beta"

    @pytest.mark.skipif(os.name != "nt", reason="Windows line-ending preservation behavior")
    def test_write_preserves_lf_bytes_on_windows(self, tools):
        ut, test_dir = tools
        target = test_dir / "write_lf.txt"
        result = ut.Write(str(target), "a\nb\n")
        assert "Successfully wrote" in result
        assert target.read_bytes() == b"a\nb\n"

    @pytest.mark.skipif(os.name != "nt", reason="Windows line-ending preservation behavior")
    def test_write_preserves_crlf_bytes_on_windows(self, tools):
        ut, test_dir = tools
        target = test_dir / "write_crlf.txt"
        result = ut.Write(str(target), "a\r\nb\r\n")
        assert "Successfully wrote" in result
        assert target.read_bytes() == b"a\r\nb\r\n"

    @pytest.mark.skipif(os.name != "nt", reason="Windows line-ending preservation behavior")
    def test_edit_preserves_existing_lf_endings_on_windows(self, tools):
        ut, test_dir = tools
        target = test_dir / "edit_lf.txt"
        target.write_bytes(b"alpha\nbeta\n")
        ut.Read(str(target))
        result = ut.Edit(str(target), "alpha", "omega")
        assert "Successfully replaced" in result
        assert target.read_bytes() == b"omega\nbeta\n"

    @pytest.mark.skipif(os.name != "nt", reason="Windows line-ending preservation behavior")
    def test_edit_preserves_existing_crlf_endings_on_windows(self, tools):
        ut, test_dir = tools
        target = test_dir / "edit_crlf.txt"
        target.write_bytes(b"alpha\r\nbeta\r\n")
        ut.Read(str(target))
        result = ut.Edit(str(target), "alpha", "omega")
        assert "Successfully replaced" in result
        assert target.read_bytes() == b"omega\r\nbeta\r\n"

    def test_overwrite_requires_read_first(self, tools):
        ut, test_dir = tools
        target = test_dir / "overwrite.txt"
        target.write_text("old")

        result = ut.Overwrite(str(target), "new")

        assert result.startswith("Error:")
        assert "must be read with Read()" in result

    def test_overwrite_fails_if_file_changed_since_read(self, tools):
        ut, test_dir = tools
        target = test_dir / "overwrite_stale.txt"
        target.write_text("old")
        ut.Read(str(target))
        target.write_text("changed")

        result = ut.Overwrite(str(target), "new")

        assert result.startswith("Error:")
        assert "changed since last Read" in result

    def test_overwrite_small_replacement_succeeds_without_confirmation(self, tools):
        ut, test_dir = tools
        target = test_dir / "overwrite_small.txt"
        target.write_text("\n".join(f"line {i}" for i in range(20)) + "\n")
        ut.Read(str(target))

        result = ut.Overwrite(
            str(target),
            "\n".join(["line 0 updated", *[f"line {i}" for i in range(1, 20)]]) + "\n",
        )

        assert "Successfully overwrote" in result
        assert target.read_text().startswith("line 0 updated\n")

    def test_overwrite_large_replacement_requires_confirmation(self, temp_test_dir):
        asked: list[str] = []

        def ask(question: str) -> str:
            asked.append(question)
            return "yes"

        ut = UsefulTools(ask_user_question_callback=ask)
        target = temp_test_dir / "overwrite_large.txt"
        target.write_text("\n".join(f"old {i}" for i in range(120)) + "\n")
        ut.Read(str(target))

        result = ut.Overwrite(
            str(target),
            "\n".join(f"new {i}" for i in range(120)) + "\n",
        )

        assert "Successfully overwrote" in result
        assert asked
        assert "Confirm large overwrite" in asked[0]

    def test_overwrite_large_replacement_can_be_denied(self, temp_test_dir):
        def ask(_question: str) -> str:
            return "no"

        ut = UsefulTools(ask_user_question_callback=ask)
        target = temp_test_dir / "overwrite_denied.txt"
        target.write_text("\n".join(f"old {i}" for i in range(120)) + "\n")
        original = target.read_text()
        ut.Read(str(target))

        result = ut.Overwrite(
            str(target),
            "\n".join(f"new {i}" for i in range(120)) + "\n",
        )

        assert result == "Error: Overwrite cancelled by user"
        assert target.read_text() == original

    def test_overwrite_large_replacement_fails_closed_without_callback(self, tools):
        ut, test_dir = tools
        target = test_dir / "overwrite_no_callback.txt"
        target.write_text("\n".join(f"old {i}" for i in range(120)) + "\n")
        original = target.read_text()
        ut.Read(str(target))

        result = ut.Overwrite(
            str(target),
            "\n".join(f"new {i}" for i in range(120)) + "\n",
        )

        assert result.startswith("Error:")
        assert "refused a large rewrite" in result
        assert target.read_text() == original


class TestExtractLeadingCommandName:
    def test_unterminated_quote_returns_none(self):
        assert _extract_leading_command_name('"unterminated') is None

    def test_empty_string_returns_none(self):
        assert _extract_leading_command_name("") is None

    def test_only_env_vars_returns_none(self):
        assert _extract_leading_command_name("FOO=bar BAZ=qux") is None


class TestShellSelection:
    def test_windows_prefers_git_bash_when_available(self):
        shell = _detect_shell_prefix(
            os_name="nt",
            environ={"COMSPEC": r"C:\Windows\System32\cmd.exe"},
            which=lambda name: r"C:\Program Files\Git\bin\bash.exe" if name == "bash" else None,
        )
        assert shell == [r"C:\Program Files\Git\bin\bash.exe", "-lc"]

    def test_windows_falls_back_to_cmd_when_no_bash(self):
        shell = _detect_shell_prefix(
            os_name="nt",
            environ={"COMSPEC": r"C:\Windows\System32\cmd.exe"},
            which=lambda _name: None,
        )
        assert shell == [r"C:\Windows\System32\cmd.exe", "/c"]

    def test_posix_uses_shell_env(self):
        shell = _detect_shell_prefix(
            os_name="posix",
            environ={"SHELL": "/bin/zsh"},
            which=lambda _name: "/bin/sh",
        )
        assert shell == ["/bin/zsh", "-c"]

    def test_posix_falls_back_to_sh(self):
        shell = _detect_shell_prefix(
            os_name="posix",
            environ={},
            which=lambda name: "/bin/sh" if name == "sh" else None,
        )
        assert shell == ["/bin/sh", "-c"]


class TestWindowsBashRewriting:
    def test_rewrites_cd_drive_prefix_for_git_bash(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        command = r"cd /d D:\github\graphedit && python - <<'PY'"
        rewritten = _rewrite_windows_bash_command(
            command,
            [r"C:\Program Files\Git\bin\bash.exe", "-lc"],
        )
        assert rewritten.startswith("cd /d/github/graphedit &&")

    def test_rewrites_quoted_cd_drive_prefix_for_git_bash(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        command = r'cd /d "D:\github\my repo" && pwd'
        rewritten = _rewrite_windows_bash_command(
            command,
            [r"C:\Program Files\Git\bin\bash.exe", "-lc"],
        )
        assert rewritten.startswith("cd '/d/github/my repo' &&")

    def test_keeps_cd_drive_prefix_for_cmd_shell(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        command = r"cd /d D:\github\graphedit && dir"
        rewritten = _rewrite_windows_bash_command(
            command,
            [r"C:\Windows\System32\cmd.exe", "/c"],
        )
        assert rewritten == command

    @pytest.mark.skipif(not _WINDOWS_WITH_GIT, reason="requires Windows with git installed")
    def test_bash_executes_cmd_style_cd_prefix_under_windows_git_bash(self, tools):
        ut, test_dir = tools
        if ut.shell_prefix[-1] != "-lc":
            pytest.skip("requires Git Bash shell selection")

        result = ut.Bash(
            f'cd /d {test_dir} && python -c "print(123)"',
            "windows git bash rewrite",
        )

        assert "Error:" not in result
        assert "123" in result


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

    def test_heredoc_stripping_crlf(self):
        cmd = "cat << EOF\r\nhello world\r\nEOF\r\n"
        result = _strip_heredocs(cmd)
        assert "hello world" not in result


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


class TestBashBothPaths:
    """Tests that apply identically to both streaming and non-streaming Bash paths."""

    def test_error_exit_code(self, any_tools):
        ut, _ = any_tools
        result = ut.Bash("false", "Failing command")
        assert "Error (exit code" in result


class TestBashStreaming:
    def test_streaming_captures_output_lines(self, streaming_tools):
        ut, test_dir, streamed = streaming_tools
        test_file = test_dir / "lines.txt"
        test_file.write_text("line1\nline2\nline3\n")
        result = ut.Bash(
            _py_cmd(
                "from pathlib import Path; "
                f"[print(line) for line in Path({test_file.as_posix()!r}).read_text().splitlines()]"
            ),
            "Stream cat",
        )
        assert "line1" in result
        assert "line2" in result
        assert len(streamed) >= 3
        joined = "".join(streamed)
        assert "line1" in joined
        assert "line2" in joined
        assert "line3" in joined

    def test_streaming_handles_error(self, streaming_tools):
        ut, _, streamed = streaming_tools
        result = ut.Bash(_py_cmd("import sys; sys.exit(1)"), "Failing command")
        assert "Error:" in result

    def test_streaming_timeout(self, streaming_tools):
        ut, _, _ = streaming_tools
        result = ut.Bash(
            _py_cmd("import time; time.sleep(10)"),
            "Slow command",
            timeout_seconds=0.1,
        )
        assert result == "Error: Command execution timeout"

    def test_streaming_output_truncation(self, streaming_tools):
        ut, test_dir, streamed = streaming_tools
        big_file = test_dir / "big.txt"
        big_file.write_text("X" * 200)
        result = ut.Bash(_read_file_cmd(big_file), "Cat big", max_output_chars=50)
        assert "truncated" in result
        assert len(streamed) >= 1

    def test_streaming_stderr_captured(self, streaming_tools):
        ut, _, streamed = streaming_tools
        ut.Bash(_py_cmd("import sys; print('out'); print('err', file=sys.stderr)"), "Mixed output")
        joined = "".join(streamed)
        assert "out" in joined
        assert "err" in joined

    def test_no_streaming_without_callback(self, temp_test_dir):
        ut = UsefulTools()
        assert ut.stream_callback is None
        result = ut.Bash("echo normal", "No streaming")
        assert "normal" in result


class TestDocumentation:
    def test_docs_describe_safe_write_and_overwrite(self):
        root = Path(__file__).resolve().parents[3]

        api_doc = (root / "API.md").read_text(encoding="utf-8")
        readme = (root / "README.md").read_text(encoding="utf-8")

        assert "Write content to a new file" in api_doc
        assert "Overwrite" in api_doc
        assert "explicit `Overwrite`" in readme


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
