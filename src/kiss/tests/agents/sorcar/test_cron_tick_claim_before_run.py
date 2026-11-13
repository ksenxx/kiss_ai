# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a cron tick persists its claim before it runs anything.

``tick`` advances ``next_run_at`` (and disables a one-shot) and must
save that claim BEFORE launching the job.  When the save fails, nothing
may run: a job launched before an unsaved claim executes, stays enabled
in the store, and executes again on the next tick.  The store is made
unwritable with ``chmod`` (no mocks): ``save_jobs`` stages a sibling
temp file in the cron directory, so a read-only cron directory makes
the save raise ``PermissionError`` while the job's own scratch, output
and lock directories, created beforehand, stay writable.  The Schedule
subpanel's Run now (``apply_job_action``) consumes no occurrence: its
claim is the job's run lock, taken before its thread starts.
"""

from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import cron_job, load_jobs, running_job_ids, tick

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="chmod cannot make a directory unwritable on Windows"
)


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KISS_HOME at a per-test temp dir so the job store is isolated."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    return tmp_path


def _wait_until_idle() -> None:
    deadline = time.time() + 30
    while running_job_ids() and time.time() < deadline:
        time.sleep(0.05)
    assert not running_job_ids()


def test_one_shot_whose_claim_cannot_be_saved_never_runs_twice(tmp_path: Path) -> None:
    runs = tmp_path / "runs.txt"
    created = yaml.safe_load(cron_job(
        "create", name="once", command=f"echo ran >> {runs}", schedule="1h",
    ))["created"]
    assert created["one_shot"] is True
    jobs = load_jobs()
    jobs[0]["next_run_at"] = 1.0
    cron_agent.save_jobs(jobs)
    # The run's scratch, output and lock directories must exist and stay
    # writable: only the job store itself becomes unwritable.
    cron_agent._runs_dir().mkdir()
    cron_agent._output_dir().mkdir()
    cron_agent._locks_dir().mkdir()
    cron_dir = cron_agent._cron_dir()
    cron_dir.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            tick(2_000_000_000.0)
        _wait_until_idle()
        # The claim could not be persisted, so the occurrence was not run.
        assert not runs.exists()
        assert load_jobs()[0]["enabled"] is True
    finally:
        cron_dir.chmod(0o700)
    # The store is writable again: the occurrence runs exactly once.
    assert tick(2_000_000_000.0) == 1
    _wait_until_idle()
    assert runs.read_text(encoding="utf-8").split() == ["ran"]
    stored = load_jobs()[0]
    assert stored["enabled"] is False and stored["next_run_at"] is None
    assert tick(2_000_000_001.0) == 0
    assert runs.read_text(encoding="utf-8").split() == ["ran"]


def test_run_now_that_cannot_claim_the_job_starts_nothing(tmp_path: Path) -> None:
    """Run now started the job's thread before the job was claimed: when the claim then
    failed the caller got the error while the job ran unclaimed, for a scheduler in
    another process to run again on top of it.  The claim is the job's run lock, whose
    directory cannot be created here; the job's own command is the other process."""
    runs = tmp_path / "runs.txt"
    job_id = yaml.safe_load(cron_job(
        "create", name="now", schedule="every 1h", command="true",
    ))["created"]["id"]
    again = (
        "from kiss.agents.sorcar.cron_agent import apply_job_action; "
        f"print(apply_job_action('run_now', {job_id!r}))"
    )
    jobs = load_jobs()
    jobs[0]["command"] = f"{shlex.quote(sys.executable)} -c {shlex.quote(again)} >> {runs}"
    cron_agent.save_jobs(jobs)
    cron_agent._runs_dir().mkdir()
    cron_agent._output_dir().mkdir()
    cron_dir = cron_agent._cron_dir()
    cron_dir.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            cron_agent.apply_job_action("run_now", job_id)
        _wait_until_idle()
        assert not runs.exists(), "the job ran although it could not be claimed"
    finally:
        cron_dir.chmod(0o700)
    # The directory is writable again: the job runs, claimed by then.
    assert cron_agent.apply_job_action("run_now", job_id) == ""
    _wait_until_idle()
    assert runs.read_text(encoding="utf-8").strip() == f"job {job_id!r} is already running"
