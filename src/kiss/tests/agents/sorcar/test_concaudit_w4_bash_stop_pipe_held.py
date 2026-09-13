"""E2E: a ``stop_event`` must end ``Bash``'s wait even when the shell has exited.

``UsefulTools.Bash`` waits for stdout EOF up to ``timeout_seconds``.  When
the shell exits at once but a background child inherited the stdout pipe
(``server.py &`` without redirection is the classic shape), EOF never
comes and the call waits out the whole deadline.  ``_stop_monitor`` is
meant to cut that short on a user stop, but it only kills the shell's
process group — and refuses to (correctly: the PGID may be recycled)
once the shell has already been reaped.  So a stop in that state changed
nothing: Bash kept blocking until the deadline (300 s by default).

The consume loop now observes the stop event itself, so a stopped call
returns within the short EOF grace period instead.
"""

from __future__ import annotations

import os
import shlex
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell/process-group semantics"
)


def _kill_recorded_pid(pid_file: Path) -> None:
    """SIGKILL the pid recorded in *pid_file* so no test process leaks."""
    if not pid_file.exists():
        return
    try:
        os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
    except (OSError, ValueError):
        pass


@pytest.mark.parametrize("streaming", [True, False])
def test_stop_returns_promptly_when_exited_shell_left_pipe_held(
    tmp_path: Path, streaming: bool,
) -> None:
    """A stop while a background child holds the pipe must not wait for the deadline."""
    pid_file = tmp_path / "bg.pid"
    command = (
        f"sleep 30 & echo $! > {shlex.quote(str(pid_file))}; echo started"
    )
    stop_event = threading.Event()
    streamed: list[str] = []
    tools = UsefulTools(
        stream_callback=streamed.append if streaming else None,
        stop_event=stop_event,
        work_dir=str(tmp_path),
    )
    timer = threading.Timer(0.5, stop_event.set)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    try:
        out = tools.Bash(command, "bg child holds pipe", timeout_seconds=20)
    finally:
        timer.cancel()
        _kill_recorded_pid(pid_file)
    elapsed = time.monotonic() - started
    # 0.5 s until the stop + the 5 s EOF grace, with slack; the pre-fix
    # code returned only at the 20 s deadline.
    assert elapsed < 12, f"Bash ignored the stop for {elapsed:.1f}s"
    # The shell itself exited successfully, so its output is returned.
    assert "started" in out
    assert "timeout" not in out.lower()


def test_stop_of_running_shell_still_kills_it_and_reports_exit(
    tmp_path: Path,
) -> None:
    """Regression guard: a stop of a RUNNING shell keeps its original result shape."""
    pid_file = tmp_path / "loop.pid"
    command = (
        f"echo $$ > {shlex.quote(str(pid_file))}; "
        "while true; do echo tick; sleep 0.2; done"
    )
    stop_event = threading.Event()
    tools = UsefulTools(stop_event=stop_event, work_dir=str(tmp_path))
    timer = threading.Timer(0.7, stop_event.set)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    try:
        out = tools.Bash(command, "stoppable loop", timeout_seconds=30)
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started
    assert elapsed < 10
    assert pid_file.exists()
    pid = int(pid_file.read_text().strip())
    time.sleep(0.3)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("shell survived the stop event")
    # Killed by SIGKILL: reported as a failed command, never as a timeout.
    assert out.startswith("Error (exit code")
    assert "tick" in out


def test_no_stop_event_keeps_deadline_semantics(tmp_path: Path) -> None:
    """Without a stop event the exited-shell case still returns at the deadline."""
    pid_file = tmp_path / "bg2.pid"
    command = (
        f"sleep 30 & echo $! > {shlex.quote(str(pid_file))}; echo started"
    )
    tools = UsefulTools(work_dir=str(tmp_path))
    started = time.monotonic()
    try:
        out = tools.Bash(command, "bg child holds pipe", timeout_seconds=1)
    finally:
        _kill_recorded_pid(pid_file)
    elapsed = time.monotonic() - started
    assert 1 <= elapsed < 10
    assert "started" in out
