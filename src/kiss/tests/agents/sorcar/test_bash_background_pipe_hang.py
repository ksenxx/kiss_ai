"""E2E tests: streaming Bash must not hang when background children hold stdout.

Reproduces the frozen sub-agent bug: a command like ``(sleep 30) & echo go``
exits immediately, but the backgrounded subshell inherits the stdout pipe.
The old ``_bash_streaming`` timer callback saw ``process.poll() is not None``
and returned without killing anything, so ``readline()`` blocked until every
background child exited — 94 minutes in the observed incident.
"""

from __future__ import annotations

import os
import shlex
import signal
import sys
import time
from pathlib import Path

import pytest

from kiss.core.useful_tools import UsefulTools

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell/process-group semantics"
)


def test_background_child_in_group_does_not_block_past_timeout(
    tmp_path: Path,
) -> None:
    """Shell exits fast, in-group background child holds the pipe.

    The timeout must kill the process group so the pipe closes, and the
    completed command's real output must be returned (not a timeout error,
    since the command itself finished successfully).
    """
    lines: list[str] = []
    tools = UsefulTools(stream_callback=lines.append, work_dir=str(tmp_path))
    start = time.monotonic()
    out = tools.Bash(
        "(sleep 30) & echo launched",
        "background child holds pipe",
        timeout_seconds=2,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"Bash blocked for {elapsed:.1f}s on background child"
    assert "launched" in out
    assert "timeout" not in out.lower()


def test_out_of_group_child_does_not_block_forever(tmp_path: Path) -> None:
    """A descendant that escapes the process group survives the group kill.

    ``os.setsid()`` is used (the ``setsid`` binary does not exist on macOS)
    so the child genuinely leaves the process group while inheriting the
    stdout pipe.  The tool must still return within timeout + grace instead
    of blocking until the escaped child exits, and the output streamed
    before the deadline must be preserved.
    """
    pid_file = tmp_path / "escaped.pid"
    escaper = (
        "import os,time; os.setsid(); "
        f"open({str(pid_file)!r},'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    escape = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(escaper)} & echo escaped"
    )
    lines: list[str] = []
    tools = UsefulTools(stream_callback=lines.append, work_dir=str(tmp_path))
    start = time.monotonic()
    try:
        out = tools.Bash(escape, "escaped child holds pipe", timeout_seconds=2)
        elapsed = time.monotonic() - start
        assert elapsed < 20, f"Bash blocked for {elapsed:.1f}s on escaped child"
        assert "escaped" in out
        # Prove the scenario was real: the escaped child outlived the call.
        assert pid_file.exists(), "escaped child never started"
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass


def test_non_streaming_background_child_returns_real_output(
    tmp_path: Path,
) -> None:
    """Non-streaming path: background child holds the pipe past the deadline.

    ``communicate(timeout=...)`` unblocks bounded either way, but the
    completed command's real output must be returned — not misreported
    as ``Error: Command execution timeout`` — matching the streaming path.
    """
    tools = UsefulTools(work_dir=str(tmp_path))
    start = time.monotonic()
    out = tools.Bash(
        "(sleep 30) & echo launched",
        "non-streaming background child",
        timeout_seconds=2,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"Bash blocked for {elapsed:.1f}s on background child"
    assert "launched" in out
    assert "timeout" not in out.lower()


def test_non_streaming_genuine_timeout_still_reported(tmp_path: Path) -> None:
    """Non-streaming path: a still-running command reports the timeout."""
    tools = UsefulTools(work_dir=str(tmp_path))
    start = time.monotonic()
    out = tools.Bash("sleep 30", "non-streaming timeout", timeout_seconds=1)
    elapsed = time.monotonic() - start
    assert elapsed < 12, f"timeout kill took {elapsed:.1f}s"
    assert out == "Error: Command execution timeout"


def test_genuine_timeout_of_running_command_still_reported(
    tmp_path: Path,
) -> None:
    """A command whose shell is still running at the deadline reports timeout."""
    lines: list[str] = []
    tools = UsefulTools(stream_callback=lines.append, work_dir=str(tmp_path))
    start = time.monotonic()
    out = tools.Bash(
        "echo started; sleep 30",
        "genuine timeout",
        timeout_seconds=1,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 12, f"timeout kill took {elapsed:.1f}s"
    assert out == "Error: Command execution timeout"


def test_fast_command_streams_and_returns_output(tmp_path: Path) -> None:
    """The happy path is unaffected: full output streamed and returned."""
    lines: list[str] = []
    tools = UsefulTools(stream_callback=lines.append, work_dir=str(tmp_path))
    out = tools.Bash("echo one; echo two", "fast path", timeout_seconds=10)
    assert "one\n" in lines
    assert "two\n" in lines
    assert "one" in out
    assert "two" in out


def test_raising_stream_callback_propagates_and_kills_command(
    tmp_path: Path,
) -> None:
    """A raising stream callback aborts the command and propagates.

    Pre-existing semantics (callback ran on the tool's calling thread):
    the exception escapes ``Bash`` and the process group is killed.  The
    reader-thread rewrite must preserve both.
    """
    marker = tmp_path / "kept-running"

    def boom(_line: str) -> None:
        raise RuntimeError("callback exploded")

    tools = UsefulTools(stream_callback=boom, work_dir=str(tmp_path))
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="callback exploded"):
        tools.Bash(
            f"echo first; sleep 30; touch {shlex.quote(str(marker))}",
            "callback failure",
            timeout_seconds=60,
        )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"Bash blocked for {elapsed:.1f}s after callback error"
    time.sleep(0.5)
    assert not marker.exists(), "command survived the callback failure"
