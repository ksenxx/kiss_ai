"""E2E tests: Bash must neither hang on nor kill pipe-holding background jobs.

Covers two historical bugs around background children that inherit the
tool's stdout pipe:

* The frozen sub-agent bug: a command like ``(sleep 30) & echo go`` exits
  immediately, but the backgrounded subshell inherits the stdout pipe.
  The old ``_bash_streaming`` timer callback saw ``process.poll() is not
  None`` and returned without unblocking anything, so ``readline()``
  blocked until every background child exited — 94 minutes in the
  observed incident.  The tool must return at the deadline instead.
* The silent background-job kill: the first fix unblocked the call by
  SIGKILLing the WHOLE process group at the deadline even when the shell
  itself had already exited successfully.  A deliberately detached job —
  ``cd d && nohup job > log 2>&1 < /dev/null &`` — wraps ``job`` in a
  pipe-holding ``&&``-subshell, so minutes after Bash returned
  success-looking output the framework SIGKILLed the user's nohup'd job
  (observed as whole pytest runs dying ~300 s after launch).  The group
  may be killed only on a GENUINE timeout (shell still running at the
  deadline); children of an exited shell must be left running.
"""

from __future__ import annotations

import os
import shlex
import signal
import sys
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell/process-group semantics"
)


def _assert_alive_and_kill(pid_file: Path) -> None:
    """Assert the pid recorded in *pid_file* is alive, then SIGKILL it.

    Used by the background-job survival tests: the recorded child must
    have outlived the Bash call (the old deadline group-kill would have
    SIGKILLed it before Bash returned), and is then reaped so no test
    process leaks past the suite.

    Bash returns at pipe-EOF, which the launching subshell delivers the
    moment it forks the job — typically a few milliseconds BEFORE the
    job's ``echo $$ > pid_file`` runs.  Checking existence immediately
    would race that startup (observed: file appears ~5 ms after Bash
    returns), so the pid file is polled briefly first.  The survival
    property is still fully checked: a child SIGKILLed by the old
    deadline group-kill either never writes the file or fails the
    liveness probe below.
    """
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_file.exists(), "background child never started"
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pytest.fail(f"background child {pid} was killed by the Bash deadline")
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ValueError):
            pass


def _background_job_command(pid_file: Path) -> str:
    """Shell snippet: record the job's pid in *pid_file*, then sleep."""
    return (
        f"echo $$ > {shlex.quote(str(pid_file))}; exec sleep 30"
    )


def test_background_child_in_group_does_not_block_past_timeout(
    tmp_path: Path,
) -> None:
    """Shell exits fast, in-group background child holds the pipe.

    The tool must return at the deadline with the completed command's
    real output (not a timeout error, since the command itself finished
    successfully), and the background child must be LEFT RUNNING — the
    shell exited on time, so nothing timed out and nothing may be
    killed.
    """
    pid_file = tmp_path / "child.pid"
    lines: list[str] = []
    tools = UsefulTools(stream_callback=lines.append, work_dir=str(tmp_path))
    start = time.monotonic()
    out = tools.Bash(
        f"sh -c {shlex.quote(_background_job_command(pid_file))} & echo launched",
        "background child holds pipe",
        timeout_seconds=2,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"Bash blocked for {elapsed:.1f}s on background child"
    assert "launched" in out
    assert "timeout" not in out.lower()
    _assert_alive_and_kill(pid_file)


def test_cd_prefixed_nohup_background_job_survives_deadline(
    tmp_path: Path,
) -> None:
    """The exact silent-kill incident: ``cd d && nohup job … & echo ok``.

    ``&`` backgrounds the whole ``&&`` list, so an intermediate subshell
    keeps the tool's stdout pipe open even though the job's own stdio is
    fully redirected.  The shell exits instantly, Bash returns "ok" at
    the deadline — and the nohup'd job must still be alive afterwards.
    The pre-fix code SIGKILLed the job's process group at the deadline
    while returning this very success-looking output (observed killing
    whole background pytest runs ~300 s after launch).
    """
    pid_file = tmp_path / "job.pid"
    job = _background_job_command(pid_file)
    log = tmp_path / "job.log"
    lines: list[str] = []
    tools = UsefulTools(stream_callback=lines.append, work_dir=str(tmp_path))
    start = time.monotonic()
    out = tools.Bash(
        f"cd {shlex.quote(str(tmp_path))} && "
        f"nohup sh -c {shlex.quote(job)} > {shlex.quote(str(log))} 2>&1 "
        "< /dev/null & echo ok",
        "cd && nohup launch",
        timeout_seconds=2,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"Bash blocked for {elapsed:.1f}s on nohup launch"
    assert "ok" in out
    assert "timeout" not in out.lower()
    _assert_alive_and_kill(pid_file)


def test_non_streaming_cd_prefixed_nohup_job_survives_deadline(
    tmp_path: Path,
) -> None:
    """Non-streaming path of the same incident: the job must survive."""
    pid_file = tmp_path / "job.pid"
    job = _background_job_command(pid_file)
    tools = UsefulTools(work_dir=str(tmp_path))
    start = time.monotonic()
    out = tools.Bash(
        f"cd {shlex.quote(str(tmp_path))} && "
        f"nohup sh -c {shlex.quote(job)} > /dev/null 2>&1 "
        "< /dev/null & echo ok",
        "non-streaming cd && nohup launch",
        timeout_seconds=2,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"Bash blocked for {elapsed:.1f}s on nohup launch"
    assert "ok" in out
    assert "timeout" not in out.lower()
    _assert_alive_and_kill(pid_file)


@pytest.mark.slow
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
