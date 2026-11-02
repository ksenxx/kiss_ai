# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A run whose news was not delivered must not advance the job's window.

``_execute_job`` keeps an ``until_delivered`` job alive when a delivery
target fails, so the next tick retries.  It still moved the coverage
watermark ``covered_until`` to that run's start because the run itself
succeeded, so the retry's run context told the next run to cover only
the time after the undelivered news: a recovered target was never
sent the news when nothing newer happened.  The watermark now advances
on a silent run or on a delivered result only; a failed delivery
leaves the window open, like a failed run.

Real job store, real scheduler tick, real delivery path (the first
target is a channel nobody ships, so its delivery fails for real).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import cron_job, load_jobs, tick


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point KISS_HOME at a per-test temp dir so the job store is isolated."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)


def _set_job_fields(job_id: str, **fields: object) -> None:
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job.update(fields)
    cron_agent.save_jobs(jobs)


def test_a_failed_delivery_keeps_the_window_open_until_the_news_is_delivered() -> None:
    """The watermark stays put after an undelivered result and moves once it is delivered."""
    created = yaml.safe_load(cron_job(
        "create", name="release news", schedule="every 1m", command="echo 'released!'",
        until_delivered=True, deliver="nosuchchannel:inbox",
    ))["created"]
    stored = load_jobs()[0]
    watermark = stored["covered_until"]
    assert watermark == stored["created_at"]

    _set_job_fields(created["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "ok" and stored["last_summary"] == "released!"
    assert stored["last_delivery"] == ["error: unknown channel 'nosuchchannel'"]
    assert stored["enabled"] is True, "an undelivered poll retries next tick"
    assert stored["covered_until"] == watermark, "the window moved past the undelivered news"
    assert "cover everything since" in cron_agent.run_context(stored)

    # The target is repaired: the retry delivers the news and retires the poll.
    _set_job_fields(created["id"], deliver="local", next_run_at=1.0)
    assert tick(3.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "ok" and stored["last_delivery"] == []
    assert stored["enabled"] is False and stored["next_run_at"] is None
    assert stored["covered_until"] > watermark
