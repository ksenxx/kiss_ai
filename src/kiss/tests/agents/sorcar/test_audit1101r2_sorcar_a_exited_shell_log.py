# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 rounds 2 and 3 (sorcar_a F4, N2): an exited shell's growing log, the deadline.

The wait of a foreground call run as a job (``_follow_job``) enforced
its deadline only while the shell was running, taking an exited shell's
log for final.  A shell that started a background writer on its stdout
and exited leaves a log that keeps growing; when the writer outpaces
the stream callback the wait never reaches the end of the log and
ignored the deadline forever.  The deadline now holds between pieces
whatever the shell's state, and a wait that did not reach the end of
the log keeps the job and its log for ``bash_job``.

The schedule is synchronized, not raced: the writer fills the log with
more than the wait can take before the deadline and only then lets the
shell exit, so the wait cannot find the log at its end; the first piece
taken waits for the shell's exit, so the deadline is met on an exited
shell.  The writer's pid is kept in a file, a cleanup handle that does
not depend on the job registry.  Real shells in ``tmp_path``, nothing
mocked.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import BackgroundJob, UsefulTools
from kiss.core.config import DEFAULT_CONFIG
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the host Bash tool runs bash loops and sleeps")

#: The shell starts a writer on its stdout (the job's log): it writes 4 MiB, signals
#: ``ready`` and keeps writing.  The shell records the writer's pid and exits on the signal.
EXIT_LEAVING_A_WRITER = (
    "( head -c 4194304 /dev/zero | tr '\\0' x; touch ready; "
    "while :; do head -c 65536 /dev/zero | tr '\\0' x; sleep 0.01; done ) & "
    "echo $! > writer.pid; while [ ! -f ready ]; do sleep 0.01; done; exit 0"
)


def _wait_exit(job: BackgroundJob) -> int | None:
    deadline = time.monotonic() + 10
    while job.returncode() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    return job.returncode()


def _kill_writer(tmp_path: Path) -> None:
    """Kill the writer by the pid the shell recorded, whether or not its job is still registered."""
    with contextlib.suppress(FileNotFoundError, ValueError, ProcessLookupError):
        os.kill(int((tmp_path / "writer.pid").read_text()), signal.SIGKILL)


def test_the_deadline_holds_against_a_writer_the_exited_shell_left_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 1)
    jobs: dict[str, BackgroundJob] = {}
    taken: list[int] = []
    shell_exited = threading.Event()

    def slow_sink(piece: str) -> None:
        """Takes the pieces several times slower than the writer makes them.

        The first piece is taken only once the shell has exited, so the
        pieces after it are read from an exited shell's log.
        """
        taken.append(len(piece))
        if not shell_exited.is_set():
            (job,) = jobs.values()
            if _wait_exit(job) is not None:
                shell_exited.set()
        time.sleep(0.2)

    tools = UsefulTools(stream_callback=slow_sink, work_dir=str(tmp_path), jobs=jobs)
    outcome: list[str] = []
    thread = threading.Thread(
        target=lambda: outcome.append(
            tools.Bash(EXIT_LEAVING_A_WRITER, "flood", timeout_seconds=60)
        ),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=20)
    try:
        assert not thread.is_alive(), f"Bash still draining after 20 s; {sum(taken)} chars taken"
    finally:
        _kill_writer(tmp_path)  # the log stops growing, so a stuck wait ends too
        for job_id in list(jobs):
            tools.bash_job(job_id, action="kill")
        thread.join(timeout=60)
    assert shell_exited.is_set(), "the shell did not exit within 10 s of the first piece"
    (out,) = outcome
    # The deadline held, and the job and its log stay for the model to follow.
    assert out.startswith("Still running after 1s; left running as background job j"), out
    assert "Output so far:\n" in out and taken and all(n <= 1 << 16 for n in taken)
    (job,) = jobs.values()
    assert job.log_path.exists() and job.job_id in out
