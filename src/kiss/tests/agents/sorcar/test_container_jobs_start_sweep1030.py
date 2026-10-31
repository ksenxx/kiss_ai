# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end test: starting a container background job returns at once.

``ContainerJobs.start`` sends one script that launches the job detached
and then polls for its pid file.  The launch used to be
``mkdir ... && printf ... && env ... setsid nohup sh -c ... > /dev/null &``:
bash backgrounds that whole AND-list as a subshell whose stdout is still
the exec's pipe (only the last command's own redirections point at
``/dev/null``), and the subshell waits for the job.  A reader that waits
for EOF on the exec's output (the exec bridge's providers) then blocked
until the job ended or the start timed out, and Docker paid its 2 s
stream-drain timeout on every start.

The host's bash runs the start script exactly as a container shell does,
and :func:`_bash_to_eof` reads its output to EOF like a provider's exec.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.container_jobs import JOBS_DIR, QUICK_TIMEOUT, ContainerJobs

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("setsid") is None,
    reason="needs bash and setsid",
)


def _bash_to_eof(command: str, description: str, timeout_seconds: int, max_chars: int) -> str:
    """Run *command* with the host's bash and return its output once stdout and stderr hit EOF."""
    try:
        done = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Error: {description} timed out after {timeout_seconds}s"
    return (done.stdout + done.stderr)[-max_chars:]


def test_start_returns_while_the_job_still_runs() -> None:
    """The start's output reaches EOF as soon as the pid is printed, not when the job ends."""
    jobs_dir = Path(JOBS_DIR)
    if jobs_dir.exists() and not os.access(jobs_dir, os.W_OK):
        pytest.skip(f"{JOBS_DIR} belongs to another user")
    jobs = ContainerJobs(_bash_to_eof)
    started = time.monotonic()
    job_id, report = jobs.start("sleep 300")
    elapsed = time.monotonic() - started
    try:
        assert job_id is not None, report
        assert elapsed < QUICK_TIMEOUT / 2, report
        assert "status: running" in jobs.bash_job(job_id, "tail")
        assert "status: exited killed" in jobs.bash_job(job_id, "kill")
    finally:
        if job_id is not None:
            for path in jobs_dir.glob(f"{job_id}.*"):
                path.unlink()
        with contextlib.suppress(OSError):  # other jobs' files are still there
            jobs_dir.rmdir()
