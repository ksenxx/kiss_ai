# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for the 2026-10-30 sweep of the cron automations (cron_agent).

Everything runs against the real JSON job store under an isolated
``KISS_HOME``; ``monkeypatch`` only sets environment variables.  The
connector pre-flight calls the real ``connection_status()`` of each SEA,
which reports "not connected" (``""``) here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import cron_job, load_jobs, tick


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point KISS_HOME at a per-test temp dir so the job store is isolated."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))


def _create(**fields: Any) -> dict:
    """Create a job with ``cron_job("create", ...)`` and return the created-job view."""
    reply = yaml.safe_load(cron_job("create", **fields))
    assert "created" in reply, reply
    return dict(reply["created"])


def _set_job_fields(job_id: str, **fields: object) -> None:
    """Rewrite stored fields of one job directly in the JSON store."""
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job.update(fields)
    cron_agent.save_jobs(jobs)


def test_adoption_carries_a_malformed_job_over_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job whose schedule cannot be parsed no longer blocks the hand-over of the others.

    It is adopted disabled, with the error ``tick`` records for a
    malformed job, and keeps its stored next run.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    kiss, s10s = tmp_path / ".kiss", tmp_path / ".s10s"
    monkeypatch.setenv("KISS_HOME", str(s10s))
    assert cron_agent.adopt_previous_home_jobs() == []
    tests = _create(name="Run the tests", command="pytest -q", schedule="every 2h")
    broken = _create(name="Broken", command="true", schedule="every 1h")
    _set_job_fields(broken["id"], schedule="every fortnight", next_run_at=5.0)
    monkeypatch.setenv("KISS_HOME", str(kiss))
    now = time.time()
    assert cron_agent.adopt_previous_home_jobs(now=now) == ["Run the tests", "Broken"]
    adopted_tests, adopted_broken = load_jobs()
    assert adopted_tests["id"] == tests["id"] and adopted_tests["enabled"] is True
    assert adopted_tests["next_run_at"] == now + 2 * 3600
    assert adopted_broken["id"] == broken["id"] and adopted_broken["enabled"] is False
    assert adopted_broken["last_status"] == "error"
    assert adopted_broken["last_summary"].startswith(
        "disabled: malformed job: Unsupported schedule 'every fortnight'"
    )
    assert adopted_broken["next_run_at"] == 5.0


def test_connector_auth_words_must_start_a_word() -> None:
    """Words such as "blog in" or "design in" do not skip the pre-flight; real auth wording does."""
    for prompt in (
        "Summarize my Gmail threads about the new design in Figma",
        "Draft a post for my blog in Google Docs",
        "Archive the product catalog in Google Drive",
    ):
        problem = cron_agent.connector_problem(prompt)
        assert problem is not None and " is not connected: " in problem, prompt
    for prompt in (
        "Log into Gmail and archive the spam",
        "Sign in to Google Calendar",
        "Reauthenticate Gmail",
        "re-authenticate gmail",
        "Check the Gmail login",
        "Re-connect Google Sheets",
    ):
        assert cron_agent.connector_problem(prompt) is None, prompt


def test_create_refuses_a_budget_every_run_would_reject() -> None:
    """A negative or non-finite max_budget is refused when the job is created, not on each run."""
    for bad in ("-1", "nan", "inf", "1e999"):
        reply = yaml.safe_load(cron_job(
            "create", name="x", schedule="every 5m", prompt="p", max_budget=bad,
        ))
        assert reply["error"] == f"max_budget {bad!r} must be a positive number", bad
    assert load_jobs() == []
    assert _create(name="default", schedule="every 5m", prompt="p", max_budget="0")
    assert _create(name="capped", schedule="every 6m", prompt="p", max_budget="2.5")
    assert [job["max_budget"] for job in load_jobs()] == [0.0, 2.5]


def test_a_huge_failure_streak_still_records_the_run() -> None:
    """The back-off stays a day for any streak, so the 1025th repeat is recorded like the 2nd."""
    assert cron_agent.failure_backoff_seconds(1025) == cron_agent.FAILURE_BACKOFF_MAX_SECONDS
    assert cron_agent.failure_backoff_seconds(10**9) == cron_agent.FAILURE_BACKOFF_MAX_SECONDS
    job = _create(name="broken", command="echo same-error >&2; exit 3", schedule="every 1m")
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    _set_job_fields(job["id"], next_run_at=1.0, failure_streak=1024)
    before = time.time()
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["failure_streak"] == 1025
    assert stored["last_delivery"][0].startswith("repeat of the previous failure (1025)")
    assert stored["next_run_at"] >= before + cron_agent.FAILURE_BACKOFF_MAX_SECONDS
