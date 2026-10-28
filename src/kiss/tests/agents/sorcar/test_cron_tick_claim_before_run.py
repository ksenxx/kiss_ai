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
the save raise ``PermissionError`` while the job's own scratch and
output directories, created beforehand, stay writable.
"""

from __future__ import annotations

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
    # The run's scratch and output directories must exist and stay
    # writable: only the job store itself becomes unwritable.
    cron_agent._runs_dir().mkdir()
    cron_agent._output_dir().mkdir()
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
