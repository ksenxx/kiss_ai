# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A cron run that cannot even begin is still a recorded outcome.

Two ways a job's occurrence can be consumed by a tick (``last_run_at``
set, ``next_run_at`` advanced) without the job running:

* ``_execute_job`` cannot create the run's scratch directory (here:
  a regular file sits where ``$KISS_HOME/cron/runs`` must be).  The
  failure is recorded as the job's ``last_status`` / ``last_summary``
  and delivered like any other error, on the scheduler path and on
  the ``cron_job("run_now")`` path alike.
* ``tick`` cannot start the job's thread.  Reproduced for real in a
  subprocess that lowers its ``RLIMIT_NPROC`` below its own task count,
  which makes every ``Thread.start()`` raise ``RuntimeError: can't
  start new thread``: each due job is recorded as "could not start",
  no entry stays in the running registry, and the tick returns the
  number of jobs it actually started.

Everything runs against the real JSON job store under an isolated
``KISS_HOME``; no mocks or test doubles.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import cron_job, load_jobs, tick
from kiss.tests.conftest import thread_start_can_be_starved


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KISS_HOME at a per-test temp dir so the job store is isolated."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    return tmp_path


def _create_due_command_job(name: str) -> str:
    """Create a repeating command job whose next occurrence is already due; return its id."""
    created = yaml.safe_load(cron_job(
        "create", name=name, command=f"echo {name}", schedule="every 1m", deliver="none",
    ))
    job_id = str(created["created"]["id"])
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job["next_run_at"] = time.time() - 1
    cron_agent.save_jobs(jobs)
    return job_id


def _stored(job_id: str) -> dict:
    """Return the stored record of *job_id*."""
    return next(job for job in load_jobs() if job["id"] == job_id)


def _block_runs_dir() -> None:
    """Put a regular file where the per-run scratch directories must go."""
    runs = cron_agent._runs_dir()
    assert not runs.exists()
    runs.write_text("not a directory", encoding="utf-8")


def test_tick_records_a_run_whose_scratch_dir_cannot_be_created() -> None:
    job_id = _create_due_command_job("blocked")
    _block_runs_dir()
    assert tick(wait=True) == 1
    stored = _stored(job_id)
    assert stored["last_status"] == "error", stored
    assert stored["last_summary"].startswith("FileExistsError:"), stored["last_summary"]
    assert stored["last_run_at"] is not None
    assert cron_agent.running_job_ids() == set()


def test_run_now_reports_a_run_whose_scratch_dir_cannot_be_created() -> None:
    job_id = _create_due_command_job("blocked")
    _block_runs_dir()
    reply = yaml.safe_load(cron_job("run_now", job_id=job_id))
    assert reply["ran"]["last_status"] == "error", reply
    stored = _stored(job_id)
    assert stored["last_status"] == "error", stored
    assert stored["last_summary"].startswith("FileExistsError:"), stored["last_summary"]


_TICK_WITHOUT_THREADS = textwrap.dedent("""
    import resource
    from kiss.agents.sorcar import cron_agent
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    launched = cron_agent.tick(wait=True)
    print(launched, sorted(cron_agent.running_job_ids()))
""")


def test_tick_records_every_due_job_whose_thread_cannot_start(tmp_path: Path) -> None:
    if not thread_start_can_be_starved():
        # Windows, root, or macOS (which does not count threads against RLIMIT_NPROC).
        pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
    first = _create_due_command_job("first")
    second = _create_due_command_job("second")
    proc = subprocess.run(
        [sys.executable, "-c", _TICK_WITHOUT_THREADS],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "KISS_HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0 []", proc.stdout
    assert "could not start" in proc.stderr
    for job_id in (first, second):
        stored = _stored(job_id)
        assert stored["last_status"] == "error", stored
        assert stored["last_summary"].startswith("could not start: RuntimeError:"), stored
        assert stored["last_run_at"] is not None
        assert stored["enabled"] is True
    # The occurrences were consumed: the job runs again at its next time.
    assert all(_stored(job_id)["next_run_at"] > time.time() for job_id in (first, second))
    assert not (tmp_path / "cron" / "runs").exists()
