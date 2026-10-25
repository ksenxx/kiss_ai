# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Hermes-style cron automations (cron_agent).

Everything runs against the real JSON job store under an isolated
``KISS_HOME`` — no mocks or test doubles (``monkeypatch`` is used
only to isolate environment variables, ``sys.argv``, and the
module-level daemon-socket default between tests, and to capture the
daemon submission a prompt job would perform).  A real LLM run of a
prompt job needs a live kiss-web daemon and LLM endpoint, which are
unavailable (and non-deterministic) in unit tests: the failure path
is covered via ``_execute_job``'s exception handling, and the
successful / silent paths — including the generated SEA the daemon
would apply — via ``test_prompt_job_runs_as_generated_sea`` and
``test_prompt_sea_defaults_and_silent_result`` (timeouts are covered
in ``test_cron_concurrent_runs``).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas.base.base_sea import ChannelSea
from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import (
    SCHEDULE_TZ,
    compute_next_run,
    cron_job,
    is_one_shot,
    load_jobs,
    main,
    start_scheduler_thread,
    tick,
)
from kiss.agents.sorcar.sea_settings import WORKER_DEFAULTS
from kiss.core.config import kiss_home


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KISS_HOME at a per-test temp dir so the job store is isolated.

    Also resets the module-level daemon endpoint default so a scheduler
    started by one test cannot redirect another test's prompt jobs.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    return tmp_path


def _ts(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    """Return the epoch timestamp of a Pacific-time (schedule-time) datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=SCHEDULE_TZ).timestamp()


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    """Return the epoch timestamp of a UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()


def _create(yaml_text: str) -> dict:
    """Parse a cron_job YAML reply and return the created-job dict."""
    parsed = yaml.safe_load(yaml_text)
    assert "created" in parsed, parsed
    return dict(parsed["created"])


def _set_job_fields(job_id: str, **fields: object) -> None:
    """Rewrite stored fields of one job directly in the JSON store."""
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job.update(fields)
    cron_agent.save_jobs(jobs)


def _run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
             *argv: str) -> str:
    monkeypatch.setattr(sys, "argv", ["kiss-cron", *argv])
    main()
    return str(capsys.readouterr().out)


def _cron_thread() -> threading.Thread | None:
    """Return the live kiss-cron scheduler thread, if any."""
    for thread in threading.enumerate():
        if thread.name == "kiss-cron-scheduler" and thread.is_alive():
            return thread
    return None


def _stop_scheduler(stop_event: threading.Event) -> None:
    """Stop the scheduler thread and wait for it to exit."""
    stop_event.set()
    thread = _cron_thread()
    if thread is not None:
        thread.join(timeout=10)
        assert not thread.is_alive()


def test_interval_schedules() -> None:
    assert compute_next_run("every 30m", 1000.0) == 1000.0 + 1800.0
    assert compute_next_run("every 2h", 0.0) == 7200.0
    assert compute_next_run("Every 1d", 5.0) == 5.0 + 86400.0
    assert compute_next_run("every  45 s", 0.0) == 45.0
    assert not is_one_shot("every 30m")


def test_one_shot_duration_and_iso() -> None:
    assert compute_next_run("30m", 100.0) == 100.0 + 1800.0
    assert compute_next_run("2 h", 0.0) == 7200.0
    assert is_one_shot("30m")
    future = compute_next_run("2999-01-15T14:00", 0.0)
    assert future == _ts(2999, 1, 15, 14, 0)
    assert is_one_shot("2999-01-15T14:00")
    assert compute_next_run("2001-01-15T14:00", _ts(2020, 1, 1)) is None


def test_schedule_times_are_pacific_whatever_the_machine_zone() -> None:
    # "0 9 * * *" is 9:00 Pacific: 16:00 UTC under PDT, 17:00 UTC under PST.
    summer = compute_next_run("0 9 * * *", _utc(2026, 9, 26, 12, 0))
    assert summer == _utc(2026, 9, 26, 16, 0)
    winter = compute_next_run("0 9 * * *", _utc(2026, 12, 1, 12, 0))
    assert winter == _utc(2026, 12, 1, 17, 0)
    # After 9:00 PDT the next run is tomorrow's 9:00 PDT.
    assert compute_next_run("0 9 * * *", _utc(2026, 9, 26, 16, 1)) == _utc(2026, 9, 27, 16, 0)
    # Offset-less ISO timestamps are Pacific; explicit offsets are honored.
    assert compute_next_run("2999-07-01T09:00", 0.0) == datetime(
        2999, 7, 1, 16, 0, tzinfo=UTC
    ).timestamp()
    assert compute_next_run("2999-07-01T09:00+00:00", 0.0) == datetime(
        2999, 7, 1, 9, 0, tzinfo=UTC
    ).timestamp()
    assert cron_agent.format_schedule_time(_utc(2026, 9, 27, 12, 0)) == "2026-09-27 05:00:00 PDT"
    assert cron_agent.format_schedule_time(_utc(2026, 12, 1, 17, 0)) == "2026-12-01 09:00:00 PST"


def test_repeated_fall_back_hour_fires_once() -> None:
    # 2026-11-01 01:00 PST is the second pass through 01:xx; the 01:30
    # PDT run already happened, so the next one is tomorrow's 01:30 PST.
    second_pass = datetime.fromisoformat("2026-11-01T01:00:00-08:00").timestamp()
    assert compute_next_run("30 1 * * *", second_pass) == datetime.fromisoformat(
        "2026-11-02T01:30:00-08:00"
    ).timestamp()
    # During the first pass (PDT) the 01:30 PDT occurrence is next.
    first_pass = datetime.fromisoformat("2026-11-01T01:00:00-07:00").timestamp()
    assert compute_next_run("30 1 * * *", first_pass) == first_pass + 1800


def test_listing_and_run_log_show_pacific_times() -> None:
    created = _create(cron_job("create", name="brief", schedule="0 5 * * *", command="echo hi"))
    assert created["next_run_at"].endswith((" PDT", " PST"))
    assert created["next_run_at"].split()[1] == "05:00:00"
    listed = yaml.safe_load(cron_job("list"))
    assert listed["jobs"] == [created]
    cron_job("run_now", job_id=created["id"])
    log = (cron_agent._output_dir() / f"{created['id']}.md").read_text(encoding="utf-8")
    assert log.startswith("## ") and log.split(" — ")[0].endswith((" PDT", " PST"))


def test_cron_daily_and_step() -> None:
    now = _ts(2026, 1, 15, 8, 30)  # Thursday
    assert compute_next_run("0 9 * * *", now) == _ts(2026, 1, 15, 9, 0)
    assert compute_next_run("*/15 * * * *", now) == _ts(2026, 1, 15, 8, 45)
    assert not is_one_shot("0 9 * * *")


def test_cron_weekday_range_from_weekend() -> None:
    saturday = _ts(2026, 1, 17, 12, 0)
    assert compute_next_run("0 9 * * 1-5", saturday) == _ts(2026, 1, 19, 9, 0)


def test_cron_sunday_as_seven() -> None:
    thursday = _ts(2026, 1, 15, 0, 0)
    assert compute_next_run("0 0 * * 7", thursday) == _ts(2026, 1, 18, 0, 0)


def test_cron_dom_dow_or_rule() -> None:
    # Both day fields restricted: fire on the 13th OR on Friday,
    # whichever comes first.  From Thu 2026-01-15 the next Friday is
    # Jan 16, before the next 13th (Feb 13).
    now = _ts(2026, 1, 15, 1, 0)
    assert compute_next_run("0 0 13 * 5", now) == _ts(2026, 1, 16, 0, 0)


def test_cron_lists_ranges_and_month() -> None:
    now = _ts(2026, 1, 15, 10, 0)
    assert compute_next_run("30 6,18 * * *", now) == _ts(2026, 1, 15, 18, 30)
    assert compute_next_run("0 0 1 3 *", now) == _ts(2026, 3, 1, 0, 0)
    assert compute_next_run("0-5/2 12 * * *", now) == _ts(2026, 1, 15, 12, 0)


def test_cron_never_matches_returns_none() -> None:
    # February 30th does not exist, so no match within the scan horizon.
    assert compute_next_run("0 0 30 2 *", _ts(2026, 1, 1)) is None


def test_cron_leap_day_found_across_years() -> None:
    # Regression: the scan horizon must cover the gap to the next leap day.
    assert compute_next_run("0 0 29 2 *", _ts(2025, 3, 1)) == _ts(2028, 2, 29)


def test_cron_full_range_dom_is_still_restricted() -> None:
    # Regression: Vixie cron keys the dom/dow OR rule on the literal '*',
    # so '1-31' counts as restricted and the OR rule applies.
    tuesday = _ts(2026, 1, 6, 1, 0)
    assert compute_next_run("0 0 1-31 * 1", tuesday) == _ts(2026, 1, 7, 0, 0)
    # A literal '*' dom with restricted dow requires the dow to match.
    assert compute_next_run("0 0 * * 1", tuesday) == _ts(2026, 1, 12, 0, 0)


@pytest.mark.parametrize(
    "schedule",
    [
        "whenever I feel like it",
        "* * * *",
        "61 * * * *",
        "* 24 * * *",
        "*/0 * * * *",
        "5-1 * * * *",
        "a,b * * * *",
        "every 5 weeks",
    ],
)
def test_invalid_schedules_raise(schedule: str) -> None:
    with pytest.raises(ValueError, match="Unsupported schedule"):
        compute_next_run(schedule, 0.0)


def test_create_list_pause_resume_remove(tmp_path: Path) -> None:
    job = _create(cron_job(
        "create", name="hello", command="echo hi", schedule="every 5m",
    ))
    assert job["enabled"] is True and not job.get("one_shot")
    stored = load_jobs()
    assert len(stored) == 1 and stored[0]["name"] == "hello"
    assert (tmp_path / "cron" / "jobs.json").exists()

    listing = yaml.safe_load(cron_job("list"))
    assert listing["jobs"][0]["id"] == job["id"]
    assert "next_run_at" in listing["jobs"][0]

    assert yaml.safe_load(cron_job("pause", job_id=job["id"])) == {"pause": job["id"]}
    assert load_jobs()[0]["enabled"] is False
    assert yaml.safe_load(cron_job("resume", job_id=job["id"])) == {"resume": job["id"]}
    assert load_jobs()[0]["enabled"] is True

    assert yaml.safe_load(cron_job("remove", job_id=job["id"])) == {"remove": job["id"]}
    assert load_jobs() == []


def test_default_jobs_are_seeded_once_per_home() -> None:
    """A fresh home gets the 6-hourly /recommendations and the 8am /good_morning jobs, once.

    The seed is recorded in the cron directory, so a job the user later
    removes, pauses or reschedules is never re-created or reset; a job
    the user has under a default's name (even paused) stands untouched;
    a corrupt or missing record re-seeds without duplicating; a store
    that cannot be written is left alone without an exception.
    """
    names = [spec["name"] for spec in cron_agent.DEFAULT_JOBS]
    assert cron_agent.seed_default_jobs() == names
    recommend, morning = load_jobs()
    assert recommend["name"] == (
        "Recommendations: ten suggested tasks from your recent work and apps"
    )
    assert recommend["prompt"] == "/recommendations now" and recommend["command"] == ""
    assert recommend["schedule"] == "0 */6 * * *"
    assert recommend["deliver"] == "none" and recommend["enabled"] is True
    assert recommend["max_budget"] == 2.0 and recommend["timeout"] == 900.0
    assert recommend["in_chat"] is False
    assert morning["prompt"] == "/good_morning now" and morning["schedule"] == "0 8 * * *"
    assert morning["in_chat"] is True and morning["deliver"] == "none"
    assert morning["max_budget"] == 0.5 and morning["timeout"] == 300.0
    assert cron_agent._job_view(morning)["in_chat"] is True
    assert "in_chat" not in cron_agent._job_view(recommend)
    assert json.loads(cron_agent._seeded_defaults_path().read_text()) == sorted(names)
    # Idempotent; the user's edits and removal stand.
    assert cron_agent.seed_default_jobs() == []
    assert len(load_jobs()) == 2
    assert cron_job("pause", job_id=recommend["id"])
    assert cron_agent.seed_default_jobs() == []
    assert load_jobs()[0]["enabled"] is False
    cron_job("remove", job_id=recommend["id"])
    cron_job("remove", job_id=morning["id"])
    assert cron_agent.seed_default_jobs() == [] and load_jobs() == []
    # A corrupt or missing record seeds again, but a job the user has
    # under a default's name is kept as it is, paused included.
    mine = _create(cron_job(
        "create", name=names[0], prompt="my own prompt", schedule="every 2h",
    ))
    cron_job("pause", job_id=mine["id"])
    cron_agent._seeded_defaults_path().write_text("{not json")
    assert cron_agent.seed_default_jobs() == names
    kept, morning2 = load_jobs()
    assert kept["id"] == mine["id"] and kept["prompt"] == "my own prompt"
    assert kept["enabled"] is False
    assert morning2["prompt"] == "/good_morning now"
    cron_agent._seeded_defaults_path().unlink()
    assert cron_agent.seed_default_jobs() == names and len(load_jobs()) == 2
    # A default renamed by a later version is the same default: a job
    # under its former name is kept and nothing is seeded beside it, and
    # a record naming the former name means it was seeded, so removing
    # or editing that job never brings the default back.
    (former,) = cron_agent.FORMER_DEFAULT_NAMES
    assert cron_agent.FORMER_DEFAULT_NAMES[former] == names[0]
    for job in load_jobs():
        cron_job("remove", job_id=job["id"])
    cron_agent._seeded_defaults_path().unlink()
    old_name = _create(cron_job(
        "create", name=former, prompt="/recommendations now", schedule="0 */6 * * *",
    ))
    assert cron_agent.seed_default_jobs() == names
    old_kept, morning3 = load_jobs()
    assert old_kept["id"] == old_name["id"] and old_kept["max_budget"] == 0.0
    assert morning3["prompt"] == "/good_morning now"
    cron_agent._seeded_defaults_path().write_text(json.dumps([former, names[1]]))
    cron_job("remove", job_id=old_kept["id"])
    assert cron_agent.seed_default_jobs() == []
    assert [j["id"] for j in load_jobs()] == [morning3["id"]]
    # A one-shot with a default's prompt under its own name is not that
    # default: the recurring job is seeded beside it.
    cron_agent._seeded_defaults_path().unlink()
    once = _create(cron_job(
        "create", name="once", prompt="/recommendations now", schedule="30m",
    ))
    assert cron_agent.seed_default_jobs() == names
    assert sorted(j["name"] for j in load_jobs()) == sorted([*names, "once"])
    assert next(j for j in load_jobs() if j["name"] == names[0])["id"] != once["id"]
    # A store that exists but cannot be read or parsed is the user's
    # schedules (load_jobs() reads it as empty): it is left as it is.
    cron_agent._seeded_defaults_path().unlink()
    store = cron_agent._jobs_path()
    user_store = store.read_text()
    store.write_text("{not a list")
    assert cron_agent.seed_default_jobs() == []
    assert store.read_text() == "{not a list"
    store.write_text(user_store)
    if sys.platform != "win32":  # chmod cannot make a file unreadable on Windows
        store.chmod(0o000)
        try:
            assert cron_agent.seed_default_jobs() == []
        finally:
            store.chmod(0o600)
        assert store.read_text() == user_store
        assert not cron_agent._seeded_defaults_path().exists()
    # A malformed job in the store (``_job_view`` cannot render it) is
    # a warning, not a daemon-startup exception.
    for job in load_jobs():
        cron_job("remove", job_id=job["id"])
    broken = _create(cron_job(
        "create", name="odd", prompt="/recommendations now", schedule="0 */6 * * *",
        deliver="none",
    ))
    jobs = load_jobs()
    jobs[0]["next_run_at"] = "invalid"
    cron_agent.save_jobs(jobs)
    assert cron_agent.seed_default_jobs() == []
    assert [j["id"] for j in load_jobs()] == [broken["id"]]
    cron_job("remove", job_id=broken["id"])
    # An unwritable store: nothing seeded, nothing raised.
    cron_agent._seeded_defaults_path().unlink(missing_ok=True)
    if sys.platform != "win32":  # chmod cannot make a directory unwritable on Windows
        cron_agent._cron_dir().chmod(0o500)
        try:
            assert cron_agent.seed_default_jobs() == []
            assert load_jobs() == []
        finally:
            cron_agent._cron_dir().chmod(0o700)


def test_in_chat_prompt_job_is_submitted_as_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``in_chat`` job's prompt reaches the daemon verbatim as a chat message.

    No preamble, no pinned SEA (the daemon's own slash dispatch runs
    ``/good_morning now``), the job's model, budget, work directory and
    a stop at the timeout; the chat's answer is the job's summary, a
    [SILENT] one is silent, a failed run or an unreachable daemon is an
    error, and an unconfirmed stop is the TimeoutError the scheduler
    keeps the scratch directory for.
    """
    from kiss.agents.sorcar import daemon_client

    captured: list[dict[str, object]] = []
    outcome: dict[str, object] = {}

    def capture_run(prompt: str, **kwargs: object) -> daemon_client.TaskResult:
        captured.append({"prompt": prompt, **kwargs})
        if isinstance(outcome.get("raise"), BaseException):
            raise outcome["raise"]  # type: ignore[misc]
        return daemon_client.TaskResult(
            text=str(outcome.get("text", "")), success=bool(outcome.get("success", True)),
            cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    job = {
        "id": "abcd1234", "name": "morning", "prompt": "/good_morning now",
        "model_name": "some-model", "max_budget": 0.5, "in_chat": True,
    }
    outcome.update(text="<p>Good morning!</p>", success=True)
    assert cron_agent._run_prompt_job(job, work_dir, work_dir, 300.0) == (
        "ok", "<p>Good morning!</p>",
    )
    sent = captured[0]
    assert sent["prompt"] == "/good_morning now"
    assert "sea_path" not in sent and "parent_task_id" not in sent
    assert sent["model"] == "some-model" and sent["max_budget"] == 0.5
    assert sent["work_dir"] == str(work_dir)
    assert sent["timeout"] == 300.0 and sent["stop_on_timeout"] is True
    assert not list(work_dir.iterdir())

    outcome.update(text="[SILENT]", success=True)
    assert cron_agent._run_prompt_job(job, work_dir, work_dir, 300.0) == ("silent", None)
    outcome.update(text="", success=False)
    assert cron_agent._run_prompt_job(job, work_dir, work_dir, 300.0) == (
        "error", "Task failed",
    )
    outcome["raise"] = ConnectionError("daemon down")
    assert cron_agent._run_prompt_job(job, work_dir, work_dir, 300.0) == (
        "error", "Error: daemon down",
    )
    outcome["raise"] = daemon_client.StopUnconfirmedTimeoutError("no confirmation")
    with pytest.raises(TimeoutError, match="MAY STILL BE RUNNING"):
        cron_agent._run_prompt_job(job, work_dir, work_dir, 300.0)
    assert captured[-1]["max_budget"] == 0.5
    job["max_budget"] = 0
    outcome.pop("raise")
    outcome.update(text="hi", success=True)
    cron_agent._run_prompt_job(job, work_dir, work_dir, 300.0)
    assert captured[-1]["max_budget"] is None


def test_resume_completed_one_shot_is_rejected() -> None:
    job = _create(cron_job("create", name="once", command="true", schedule="1h"))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    reply = yaml.safe_load(cron_job("resume", job_id=job["id"]))
    assert "already ran" in reply["error"]
    assert load_jobs()[0]["enabled"] is False


def test_resume_recomputes_missing_next_run() -> None:
    job = _create(cron_job(
        "create", name="r", command="true", schedule="every 1h",
    ))
    _set_job_fields(job["id"], enabled=False, next_run_at=None)
    cron_job("resume", job_id=job["id"])
    resumed = load_jobs()[0]
    assert resumed["enabled"] is True
    assert resumed["next_run_at"] is not None


def test_create_validation_errors() -> None:
    def err(reply: str) -> str:
        return str(yaml.safe_load(reply)["error"])

    assert "name and schedule" in err(cron_job("create", name="x"))
    assert "exactly one of" in err(cron_job(
        "create", name="x", schedule="every 5m",
    ))
    assert "exactly one of" in err(cron_job(
        "create", name="x", schedule="every 5m", prompt="p", command="c",
    ))
    assert "Unsupported schedule" in err(cron_job(
        "create", name="x", schedule="sometimes", command="c",
    ))
    assert "never fires" in err(cron_job(
        "create", name="x", schedule="2001-01-01T00:00", command="c",
    ))
    assert "not a number" in err(cron_job(
        "create", name="x", schedule="every 5m", prompt="p", max_budget="lots",
    ))


def test_create_refuses_duplicate_of_scheduled_job() -> None:
    first = _create(cron_job(
        "create", name="brief", command="echo hi", schedule="every 1h",
        deliver="telegram:1",
    ))
    # Same work, schedule and delivery under a different name, with
    # incidental whitespace and a different model/budget: still a duplicate.
    reply = yaml.safe_load(cron_job(
        "create", name="morning brief", command="  echo hi ", schedule=" every 1h",
        deliver="telegram:1 ", model_name="other", max_budget="9",
    ))
    assert "already scheduled" in reply["error"]
    assert "same command" in reply["error"]
    assert "remove it first" in reply["error"]
    assert reply["existing"]["id"] == first["id"]
    assert len(load_jobs()) == 1

    # Any difference in the work, schedule or delivery is a new job.
    _create(cron_job(
        "create", name="brief", command="echo hi", schedule="every 2h",
        deliver="telegram:1",
    ))
    _create(cron_job(
        "create", name="brief", command="echo hi", schedule="every 1h",
        deliver="telegram:2",
    ))
    _create(cron_job(
        "create", name="brief", command="echo bye", schedule="every 1h",
        deliver="telegram:1",
    ))
    assert len(load_jobs()) == 4


def test_create_refuses_duplicate_of_paused_prompt_job() -> None:
    first = _create(cron_job(
        "create", name="llm", prompt="say hi", schedule="0 9 * * 1-5",
    ))
    cron_job("pause", job_id=first["id"])
    reply = yaml.safe_load(cron_job(
        "create", name="llm again", prompt="say hi", schedule="0 9 * * 1-5",
    ))
    assert "already paused" in reply["error"]
    assert "same prompt" in reply["error"]
    assert f"job_id={first['id']!r}" in reply["error"]
    assert reply["existing"]["id"] == first["id"]
    assert len(load_jobs()) == 1


def test_create_allows_recreating_finished_job() -> None:
    once = _create(cron_job("create", name="once", command="true", schedule="1h"))
    _set_job_fields(once["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    assert load_jobs()[0]["next_run_at"] is None
    again = _create(cron_job("create", name="once", command="true", schedule="1h"))
    assert again["id"] != once["id"]
    assert len(load_jobs()) == 2

    # A legacy store entry without the compared keys is never a match
    # and never crashes the comparison.
    jobs = load_jobs()
    jobs.append({"id": "legacy", "enabled": True, "next_run_at": 1.0})
    cron_agent.save_jobs(jobs)
    _create(cron_job("create", name="other", command="echo x", schedule="every 1m"))
    assert len(load_jobs()) == 4


def test_cli_create_reports_duplicate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    _run_cli(monkeypatch, capsys, "--create", "a", "--schedule", "every 1m",
             "--command", "echo 1")
    out = _run_cli(monkeypatch, capsys, "--create", "b", "--schedule", "every 1m",
                   "--command", "echo 1")
    assert "already scheduled" in yaml.safe_load(out)["error"]
    assert len(load_jobs()) == 1


def test_unknown_ids_and_actions() -> None:
    assert "no job with id" in cron_job("remove", job_id="nope")
    assert "no job with id" in cron_job("pause", job_id="nope")
    assert "no job with id" in cron_job("run_now", job_id="nope")
    assert "requires job_id" in cron_job("pause")
    assert "unknown action" in cron_job("explode")


def test_load_jobs_tolerates_bad_store(tmp_path: Path) -> None:
    store = tmp_path / "cron" / "jobs.json"
    assert load_jobs() == []  # missing file
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("not json", encoding="utf-8")
    assert load_jobs() == []
    store.write_text('{"a": 1}', encoding="utf-8")
    assert load_jobs() == []
    store.write_text('[{"no_id": true}, 42]', encoding="utf-8")
    assert load_jobs() == []


def test_writers_refuse_a_corrupt_store(tmp_path: Path) -> None:
    """A store that exists but is not a JSON list is never overwritten.

    ``load_jobs()`` reads a damaged file as ``[]`` for listing; a writer
    must not mistake that for an empty store and replace the user's
    schedules with the one job it was about to save.
    """
    mine = _create(cron_job("create", name="mine", command="echo 1", schedule="every 1m"))
    store = tmp_path / "cron" / "jobs.json"
    damaged = store.read_text(encoding="utf-8")[:-5]  # truncated mid-JSON
    store.write_text(damaged, encoding="utf-8")
    error = cron_agent.apply_job_action(
        "create", "", name="new", schedule="every 1m", what="echo 2", kind="command",
    )
    assert "not a JSON list" in error
    assert store.read_text(encoding="utf-8") == damaged
    assert "not a JSON list" in cron_job(
        "create", name="new", command="echo 2", schedule="every 1m",
    )
    assert store.read_text(encoding="utf-8") == damaged
    assert "not a JSON list" in cron_job("remove", job_id=mine["id"])
    cron_agent._execute_job(mine)  # the run happens; the record step is skipped
    assert store.read_text(encoding="utf-8") == damaged
    assert tick(2_000_000_000.0) == 0  # a damaged store reads as []: nothing due, no save
    assert store.read_text(encoding="utf-8") == damaged
    store.write_text('{"a": 1}', encoding="utf-8")
    created = cron_job("create", name="new", command="echo 2", schedule="every 1m")
    assert "not a JSON list" in created
    assert store.read_text(encoding="utf-8") == '{"a": 1}'


def test_tick_runs_due_command_job(tmp_path: Path) -> None:
    job = _create(cron_job(
        "create", name="greeter", command="echo hello from cron",
        schedule="every 1m",
    ))
    now = 2_000_000_000.0
    _set_job_fields(job["id"], next_run_at=now - 5)
    assert tick(now) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "ok"
    assert stored["last_summary"] == "hello from cron"
    assert stored["last_run_at"] == now
    assert stored["next_run_at"] == now + 60
    log = (tmp_path / "cron" / "output" / f"{job['id']}.md").read_text(encoding="utf-8")
    assert "hello from cron" in log and "greeter" in log


def test_tick_skips_not_due_and_disabled_jobs() -> None:
    job = _create(cron_job(
        "create", name="later", command="echo x", schedule="every 1h",
    ))
    assert tick() == 0  # not due yet
    _set_job_fields(job["id"], next_run_at=1.0, enabled=False)
    assert tick() == 0  # due but disabled


def test_tick_one_shot_disables_job(tmp_path: Path) -> None:
    job = _create(cron_job(
        "create", name="once", command="echo one shot ran", schedule="1h",
    ))
    assert job["one_shot"] is True
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2_000_000_000.0) == 1
    stored = load_jobs()[0]
    assert stored["enabled"] is False
    assert stored["next_run_at"] is None
    assert "one shot ran" in (
        tmp_path / "cron" / "output" / f"{job['id']}.md"
    ).read_text(encoding="utf-8")


def test_tick_after_downtime_skips_missed_repeating_runs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A scheduler starting a day late does not fire every missed job at once.

    This is what happened after ``install.sh`` restarted the kiss-web
    daemon: every daily job whose time had passed while the daemon was
    down ran together on the first tick.  Missed occurrences of
    repeating jobs are now skipped (rescheduled from now, like cron);
    a one-shot still runs late, once (``test_tick_one_shot_disables_job``
    ticks a one-shot two billion seconds late).
    """
    schedules = {"nightly": "0 5 * * *", "sync": "every 4h", "poll": "every 1m"}
    jobs = {
        name: _create(cron_job(
            "create", name=name, command=f"echo {name} ran", schedule=schedule,
        ))
        for name, schedule in schedules.items()
    }
    now = _ts(2026, 10, 4, 11, 12)
    for job in jobs.values():
        _set_job_fields(job["id"], next_run_at=now - 26 * 3600)
    with caplog.at_level("WARNING", logger="kiss.agents.sorcar.cron_agent"):
        assert tick(now) == 0
    for name, job in jobs.items():
        stored = next(j for j in load_jobs() if j["id"] == job["id"])
        assert stored["enabled"] is True
        assert stored["last_run_at"] is None, name
        assert stored["next_run_at"] == compute_next_run(schedules[name], now), name
        assert not (tmp_path / "cron" / "output" / f"{job['id']}.md").exists(), name
        assert f"skipping the run of {name!r}" in caplog.text
    assert "due 1 day, 2:00:00 ago, past the 0:10:00 grace window" in caplog.text
    # The jobs resume their normal cadence: only the minutely poll is due
    # a minute later, and it runs.
    assert tick(now + 60) == 1
    assert "poll ran" in (
        tmp_path / "cron" / "output" / f"{jobs['poll']['id']}.md"
    ).read_text(encoding="utf-8")


def test_tick_grace_window_boundary(tmp_path: Path) -> None:
    """A run late by at most the grace window still happens; one second more is missed."""
    late = _create(cron_job("create", name="late", command="echo late", schedule="every 1h"))
    missed = _create(cron_job(
        "create", name="missed", command="echo missed", schedule="every 1h",
    ))
    now = 2_000_000_000.0
    _set_job_fields(late["id"], next_run_at=now - cron_agent.MISSED_RUN_GRACE_SECONDS)
    _set_job_fields(missed["id"], next_run_at=now - cron_agent.MISSED_RUN_GRACE_SECONDS - 1)
    assert tick(now) == 1
    by_id = {job["id"]: job for job in load_jobs()}
    assert by_id[late["id"]]["last_status"] == "ok"
    assert by_id[late["id"]]["last_run_at"] == now
    assert by_id[missed["id"]]["last_run_at"] is None
    assert by_id[missed["id"]]["next_run_at"] == now + 3600
    assert not (tmp_path / "cron" / "output" / f"{missed['id']}.md").exists()


def test_missed_run_of_never_matching_cron_logs_never(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missed job whose expression has no future occurrence is logged as ``never``."""
    job = _create(cron_job("create", name="feb31", command="echo x", schedule="0 0 1 1 *"))
    now = 2_000_000_000.0
    _set_job_fields(job["id"], schedule="0 0 31 2 *", next_run_at=now - 86400)
    with caplog.at_level("WARNING", logger="kiss.agents.sorcar.cron_agent"):
        assert tick(now) == 0
    assert "next run at never" in caplog.text
    assert load_jobs()[0]["next_run_at"] is None


def test_infinitely_overdue_job_is_disabled_and_the_rest_run() -> None:
    """A ``-Infinity`` timestamp (json accepts it) is malformed, not a tick-wide crash."""
    bad = _create(cron_job("create", name="bad", command="echo bad", schedule="every 1h"))
    good = _create(cron_job("create", name="good", command="echo good", schedule="every 1h"))
    now = 2_000_000_000.0
    _set_job_fields(bad["id"], next_run_at=float("-inf"))
    _set_job_fields(good["id"], next_run_at=now - 1)
    assert tick(now) == 1
    by_id = {job["id"]: job for job in load_jobs()}
    assert by_id[bad["id"]]["enabled"] is False
    assert by_id[bad["id"]]["last_summary"].startswith("disabled: malformed job:")
    assert by_id[good["id"]]["last_summary"] == "good"


def test_silent_command_delivers_nothing(tmp_path: Path) -> None:
    job = _create(cron_job("create", name="quiet", command="true", schedule="every 1m"))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "silent"
    assert stored["last_summary"] == ""
    assert not (tmp_path / "cron" / "output" / f"{job['id']}.md").exists()


def test_failing_command_reports_error(tmp_path: Path) -> None:
    job = _create(cron_job(
        "create", name="broken", command="echo boom >&2; exit 3",
        schedule="every 1m",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "error"
    assert "exited 3" in stored["last_summary"]
    assert "boom" in stored["last_summary"]
    # Errors are still delivered (locally) so the user learns about them.
    log = (tmp_path / "cron" / "output" / f"{job['id']}.md").read_text(encoding="utf-8")
    assert "exited 3" in log


def test_run_now_ignores_schedule_state() -> None:
    job = _create(cron_job(
        "create", name="manual", command="echo manual run", schedule="every 1d",
    ))
    cron_job("pause", job_id=job["id"])
    reply = yaml.safe_load(cron_job("run_now", job_id=job["id"]))
    assert reply["ran"]["last_status"] == "ok"
    assert load_jobs()[0]["last_summary"] == "manual run"


def test_tick_disables_malformed_job_and_runs_the_rest() -> None:
    bad = _create(cron_job("create", name="bad", command="echo no", schedule="every 1m"))
    good = _create(cron_job("create", name="good", command="echo yes", schedule="every 1m"))
    _set_job_fields(bad["id"], next_run_at="not-a-number")
    _set_job_fields(good["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    by_id = {job["id"]: job for job in load_jobs()}
    assert by_id[bad["id"]]["enabled"] is False
    assert by_id[bad["id"]]["last_status"] == "error"
    assert "malformed" in by_id[bad["id"]]["last_summary"]
    assert by_id[good["id"]]["last_summary"] == "yes"


def test_tick_skips_when_lock_held() -> None:
    job = _create(cron_job("create", name="locked", command="echo x", schedule="every 1m"))
    _set_job_fields(job["id"], next_run_at=1.0)
    with cron_agent._jobs_lock(blocking=True):
        assert tick(2.0) == 0
    assert tick(2.0) == 1


def test_prompt_job_runs_the_cron_prompt_sea(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt job is launched as the bundled ``cron_prompt_sea.py`` through ``run_agent``.

    The job's prompt (preamble-prefixed) is the task, and its model,
    budget, work directory and git flags travel as the tool's arguments
    and ``options`` — nothing is written to disk.  The real path is
    exercised up to the daemon-client boundary; only that boundary call
    is captured, and the SEA it names is then applied with the daemon's
    own loader to read the effective run settings.
    """
    from kiss.agents.sorcar import daemon_client
    from kiss.agents.sorcar.sea_apply import apply_sea

    captured: list[dict[str, object]] = []

    def capture_run(prompt: str, **kwargs: object) -> daemon_client.TaskResult:
        captured.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(
            text="hello", success=True, cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    job = {
        "id": "abcd1234", "name": "greeter", "prompt": "say 'hi'\n",
        "model_name": "some-model", "max_budget": 1.5,
    }
    status, text = cron_agent._run_prompt_job(
        job, work_dir, work_dir, cron_agent.PROMPT_TIMEOUT_SECONDS,
    )
    assert (status, text) == ("ok", "hello")
    assert not list(work_dir.iterdir())
    sent = captured[0]
    assert sent["sea_path"] == str(cron_agent.PROMPT_SEA_PATH)
    assert str(sent["prompt"]).startswith(cron_agent.PROMPT_PREAMBLE + "# Run context\n- Now: ")
    assert str(sent["prompt"]).endswith("itself: do not write a log.\n\nsay 'hi'\n")
    assert sent["model"] == "some-model"
    assert sent["max_budget"] == 1.5
    assert sent["work_dir"] == str(work_dir)
    assert sent["record_timeout"] == cron_agent.PROMPT_TIMEOUT_SECONDS
    # Top-level task: no parent, so no reviewer sub-tree marking.
    assert sent["parent_task_id"] == ""
    cmd: dict[str, object] = {
        "seaPath": str(cron_agent.PROMPT_SEA_PATH), "prompt": str(sent["prompt"]),
        "classifyTasks": None,
    }
    assert apply_sea(cmd) == {"classifyTasks"}
    assert str(cmd["prompt"]).endswith("\n\nsay 'hi'\n")
    assert cmd["classifyTasks"] is False


def test_slash_command_prompt_job_runs_that_sea(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt that is a slash command (``/recommendations now``) runs that SEA directly.

    The SEA named is the command's own file (not ``cron_prompt_sea.py``),
    the task is the preamble plus the trailing text, and the classifier
    is off as it is for the generic prompt SEA; a bare ``/recommendations``
    (no text) runs that SEA too, on the preamble alone, since the SEA's
    ``prompt()`` ignores the task text.
    """
    from kiss.agents.seas.recommendations import recommendations_sea
    from kiss.agents.sorcar import daemon_client

    captured: list[dict[str, object]] = []

    def capture_run(prompt: str, **kwargs: object) -> daemon_client.TaskResult:
        captured.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(
            text="<ul><li>five</li></ul>", success=True, cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    job = {"id": "abcd1234", "name": "Recommendations", "prompt": "/recommendations now"}
    status, text = cron_agent._run_prompt_job(
        job, work_dir, work_dir, cron_agent.PROMPT_TIMEOUT_SECONDS,
    )
    assert (status, text) == ("ok", "<ul><li>five</li></ul>")
    sent = captured[0]
    assert sent["sea_path"] == str(Path(recommendations_sea.__file__).resolve())
    assert str(sent["prompt"]).startswith(cron_agent.PROMPT_PREAMBLE + "# Run context")
    assert str(sent["prompt"]).endswith("\n\nnow")
    assert sent["auto_classify"] is False
    assert sent["work_dir"] == str(work_dir)
    assert sent["parent_task_id"] == ""

    job["prompt"] = "/recommendations"
    cron_agent._run_prompt_job(job, work_dir, work_dir, cron_agent.PROMPT_TIMEOUT_SECONDS)
    assert captured[1]["sea_path"] == str(Path(recommendations_sea.__file__).resolve())
    assert str(captured[1]["prompt"]).endswith("do not write a log.\n\n")
    assert captured[1]["auto_classify"] is False
    # A prompt that names no command still runs the generic prompt SEA.
    job["prompt"] = "list my open pull requests"
    cron_agent._run_prompt_job(job, work_dir, work_dir, cron_agent.PROMPT_TIMEOUT_SECONDS)
    assert captured[2]["sea_path"] == str(cron_agent.PROMPT_SEA_PATH)
    assert str(captured[2]["prompt"]).endswith("\n\nlist my open pull requests")
    assert captured[2]["auto_classify"] is None


def test_prompt_job_defaults_and_silent_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset model/budget leave the daemon defaults in force; [SILENT] is silent."""
    from kiss.agents.sorcar import daemon_client

    captured: list[dict[str, object]] = []

    def silent_run(prompt: str, **kwargs: object) -> daemon_client.TaskResult:
        captured.append(kwargs)
        return daemon_client.TaskResult(
            text="[SILENT]", success=True, cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", silent_run)
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    job = {"id": "abcd1234", "prompt": "say hi", "max_budget": 0}
    assert cron_agent._run_prompt_job(
        job, work_dir, work_dir, cron_agent.PROMPT_TIMEOUT_SECONDS,
    ) == ("silent", None)
    assert captured[0]["max_budget"] is None
    assert captured[0]["model"] == ""


def test_prompt_job_text_survives_adversarial_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quotes, triple quotes, backslashes and unicode in job fields round-trip."""
    from kiss.agents.sorcar import daemon_client

    captured: list[dict[str, object]] = []

    def capture_run(prompt: str, **kwargs: object) -> daemon_client.TaskResult:
        captured.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(text="ok", success=True, cost=0.0, tokens=0, steps=0)

    monkeypatch.setattr(daemon_client, "run", capture_run)
    name = 'bad """ name ' + "''' with \\ backslash"
    prompt = 'line1\n"""\n' + "'''\n\\n ünïcode \x00 {braces} #comment"
    job = {"id": "abcd1234", "name": name, "prompt": prompt, "model_name": name}
    cron_agent._run_prompt_job(job, tmp_path, tmp_path, cron_agent.PROMPT_TIMEOUT_SECONDS)
    assert str(captured[0]["prompt"]).startswith(cron_agent.PROMPT_PREAMBLE + "# Run context")
    assert str(captured[0]["prompt"]).endswith("\n\n" + prompt)
    assert captured[0]["model"] == name


def test_prompt_job_marker_in_endpoint_path_is_not_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the exact unconfirmed-stop reply is treated as a timeout.

    A connection error naming an endpoint file whose path happens to
    contain the unconfirmed-stop wording must stay an ordinary error
    (scratch directory removed), not be mistaken for a possibly live task.
    """
    endpoint = tmp_path / "MAY STILL BE RUNNING" / "no.json"
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(endpoint))
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    job = {"id": "abcd1234", "name": "hi", "prompt": "say hi", "max_budget": 0}
    status, text = cron_agent._run_prompt_job(
        job, work_dir, work_dir, cron_agent.PROMPT_TIMEOUT_SECONDS,
    )
    assert status == "error"
    assert text is not None and str(endpoint) in text


def test_prompt_job_failure_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no reachable kiss-web daemon (the isolated KISS_HOME contains
    # no sorcar-local.json endpoint file), the run_agent dispatch of the
    # job's SEA fails and _execute_job records the error end-to-end.
    monkeypatch.delenv("KISS_SORCAR_LOCAL", raising=False)
    job = _create(cron_job(
        "create", name="llm", prompt="say hi", schedule="every 1m",
        deliver="none",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "error"
    # The error explains what is missing (the endpoint file) and the
    # remedy (start kiss-web) instead of a bare traceback.
    assert "Cannot connect to the sorcar daemon" in stored["last_summary"]
    assert "no endpoint file at" in stored["last_summary"]
    assert "kiss-web" in stored["last_summary"]


def test_cli_usage_exits_without_args(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(sys, "argv", ["kiss-cron"])
    with pytest.raises(SystemExit):
        main()
    assert "Usage: kiss-cron" in capsys.readouterr().out


def test_cli_create_list_tick_manage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    out = _run_cli(
        monkeypatch, capsys, "--create", "cli job", "--schedule", "every 1m",
        "--command", "echo from cli",
    )
    job_id = str(yaml.safe_load(out)["created"]["id"])
    assert "cli job" in _run_cli(monkeypatch, capsys, "--list")
    _set_job_fields(job_id, next_run_at=time.time() - 1)
    assert "ran 1 job(s)" in _run_cli(monkeypatch, capsys, "--tick")
    assert "ran:" in _run_cli(monkeypatch, capsys, "--run", job_id)
    assert load_jobs()[0]["last_summary"] == "from cli"
    assert "pause" in _run_cli(monkeypatch, capsys, "--pause", job_id)
    assert "resume" in _run_cli(monkeypatch, capsys, "--resume", job_id)
    assert "remove" in _run_cli(monkeypatch, capsys, "--remove", job_id)
    assert load_jobs() == []


def test_cli_nothing_to_do(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(sys, "argv", ["kiss-cron", "--interval", "5"])
    with pytest.raises(SystemExit):
        main()
    assert "Nothing to do" in capsys.readouterr().out


def test_agent_script_getters(tmp_path: Path) -> None:
    """The SEA contract used by ``run_agent(agent="cron", ...)``.

    The SEA is a ``ChannelSea`` in ``~/.kiss/cron/work``,
    so the resolved settings (what the dispatcher and the daemon read)
    turn the git lifecycle, classification and inheritance off; the
    cron guidance reaches the session through ``system_prompt()``
    (the daemon applies it as the run's system-prompt hook), not the task.
    """
    from kiss.agents.sorcar.sea_apply import CHANNEL_PREAMBLE, apply_sea
    from kiss.agents.sorcar.sea_commands import sea_settings

    work_dir = cron_agent.cron_work_dir()
    assert work_dir == str(tmp_path / "cron" / "work")
    assert Path(work_dir).is_dir()
    sea = cron_agent.CronAgentSea()
    assert isinstance(sea, ChannelSea)
    assert sea.settings({}) == {"work_dir": work_dir}
    resolved = sea_settings(Path(cron_agent.__file__))
    # A channel is a worker whose worker keys and work_dir are locked
    # (CHANNEL_BEHAVIOURS "worker"): a call cannot give cron a worktree.
    assert resolved == {
        **WORKER_DEFAULTS,
        "work_dir": work_dir, "locked": sorted({"work_dir", *WORKER_DEFAULTS}),
    }
    assert "kind" not in resolved and "channel" not in resolved
    assert resolved["auto_classify"] is False
    assert sea.system_prompt("BASE") == "BASE\n\n" + cron_agent.CRON_DISPATCH_PREAMBLE
    assert "cron_job" in cron_agent.CRON_DISPATCH_PREAMBLE
    assert [tool.__name__ for tool in sea.tools([])] == ["cron_job", "gateway_command"]
    # Applied by the daemon: the channel preamble (named after the script)
    # lands on the system prompt suffix after the caller's own text, the
    # cron guidance is the system-prompt hook the run applies to the
    # assembled prompt, the tools hook adds the cron tools to the run's
    # built-ins; the task prompt is untouched.
    cmd: dict[str, Any] = {
        "seaPath": cron_agent.__file__, "prompt": "schedule it",
        "appendToSystemPrompt": "CALLER",
    }
    overridden = apply_sea(cmd)
    # The hooks are always staged, so only settings fields are reported.
    assert {"appendToSystemPrompt", "workDir", "classifyTasks"} <= overridden
    assert not {"systemPromptHook", "toolsHook", "prompt"} & overridden
    assert cmd["prompt"] == "schedule it"
    assert cmd["workDir"] == work_dir
    assert cmd["classifyTasks"] is False
    assert cmd["appendToSystemPrompt"] == (
        "CALLER\n\n" + CHANNEL_PREAMBLE.format(name="cron_agent")
    )
    assert cmd["systemPromptHook"]("ASSEMBLED") == (
        "ASSEMBLED\n\n" + cron_agent.CRON_DISPATCH_PREAMBLE
    )
    assert [tool.__name__ for tool in cmd["toolsHook"]([print])] == [
        "print", "cron_job", "gateway_command",
    ]


def test_store_is_plain_json_list(tmp_path: Path) -> None:
    _create(cron_job("create", name="a", command="echo 1", schedule="every 1m"))
    raw = json.loads((tmp_path / "cron" / "jobs.json").read_text())
    assert isinstance(raw, list) and raw[0]["name"] == "a"


def test_silence_tokens() -> None:
    assert cron_agent._is_silent("[SILENT]")
    assert cron_agent._is_silent("<p>[SILENT]</p>\n")
    assert cron_agent._is_silent("NO_REPLY")
    assert not cron_agent._is_silent("all good")


def test_run_scheduler_returns_when_stop_event_preset() -> None:
    stop_event = threading.Event()
    stop_event.set()
    cron_agent.run_scheduler(stop_event, interval=0.01)  # returns immediately


def test_run_now_uses_daemon_endpoint_file(tmp_path: Path) -> None:
    # A scheduler started for a daemon with a custom endpoint file
    # records that file as the module default, so run_now prompt jobs
    # target that daemon instead of $KISS_HOME/sorcar-local.json.
    custom_endpoint = tmp_path / "custom-daemon.json"
    stop_event = start_scheduler_thread(
        interval=999.0, endpoint_file=str(custom_endpoint),
    )
    try:
        job = _create(cron_job(
            "create", name="llm", prompt="say hi", schedule="every 1h",
            deliver="none",
        ))
        reply = yaml.safe_load(cron_job("run_now", job_id=job["id"]))
        assert reply["ran"]["last_status"] == "error"
        assert "custom-daemon.json" in load_jobs()[0]["last_summary"]
    finally:
        _stop_scheduler(stop_event)


def test_scheduler_thread_runs_due_jobs_and_stops() -> None:
    job = _create(cron_job(
        "create", name="sched", command="echo scheduled", schedule="every 1h",
    ))
    _set_job_fields(job["id"], next_run_at=time.time() - 1)
    stop_event = start_scheduler_thread(interval=0.05)
    try:
        deadline = time.time() + 10
        while time.time() < deadline and load_jobs()[0]["last_status"] != "ok":
            time.sleep(0.05)
    finally:
        _stop_scheduler(stop_event)
    assert load_jobs()[0]["last_summary"] == "scheduled"


def test_scheduler_thread_survives_tick_failure(tmp_path: Path) -> None:
    # A cron state path that cannot be a directory makes every tick
    # raise; the loop must log and keep going instead of dying.
    (tmp_path / "cron").write_text("not a directory", encoding="utf-8")
    stop_event = start_scheduler_thread(interval=0.02)
    try:
        time.sleep(0.3)  # several failing ticks
        thread = _cron_thread()
        assert thread is not None and thread.is_alive()
    finally:
        _stop_scheduler(stop_event)


def test_until_delivered_job_disables_itself_after_first_news(tmp_path: Path) -> None:
    """A "notify me when ..." poll stops after its first non-silent delivery."""
    flag = tmp_path / "released"
    job = _create(cron_job(
        "create", name="release poll", schedule="every 1m",
        command=f"test -f {flag.as_posix()} && echo 'released!'", until_delivered=True,
    ))
    assert job["until_delivered"] is True
    _set_job_fields(job["id"], next_run_at=1.0)
    # Nothing to report yet (the command fails/exits 1 → error status, not
    # a delivery); an error is recorded but does not retire the job.
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["enabled"] is True and stored["last_status"] == "error"
    _set_job_fields(job["id"], next_run_at=1.0)
    flag.write_text("yes")
    assert tick(3.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "ok" and stored["last_summary"] == "released!"
    assert stored["enabled"] is False and stored["next_run_at"] is None
    assert tick(4.0) == 0


def test_repeating_job_without_until_delivered_keeps_running() -> None:
    job = _create(cron_job(
        "create", name="heartbeat", schedule="every 1m", command="echo beat",
    ))
    assert job["until_delivered"] is False
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["enabled"] is True and stored["next_run_at"] is not None


def test_prompt_sea_carries_job_work_dir_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job with work_dir/timeout configures its run.

    This is the shape of a "run the tests in my repo nightly" job: the
    ``work_dir`` option names the project instead of the scratch
    directory, and ``run_agent`` waits for the job's own timeout instead
    of :data:`PROMPT_TIMEOUT_SECONDS`.
    """
    from kiss.agents.sorcar import daemon_client

    captured: list[dict[str, object]] = []

    def capture_run(prompt: str, **kwargs: object) -> daemon_client.TaskResult:
        captured.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(
            text="done", success=True, cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)
    project = tmp_path / "project"
    project.mkdir()
    job = _create(cron_job(
        "create", name="nightly", prompt="run the tests", schedule="every 1d",
        model_name="some-model", work_dir=str(project), timeout="21600",
    ))
    assert job["work_dir"] == str(project.resolve())
    assert job["timeout"] == 21600.0
    # The full run path: _execute_job derives the run directory and
    # timeout from the stored job and hands them to the dispatch.
    cron_agent._execute_job(load_jobs()[0])
    ran = load_jobs()[0]
    assert (ran["last_status"], ran["last_summary"]) == ("ok", "done")
    sent = captured[0]
    assert sent["sea_path"] == str(cron_agent.PROMPT_SEA_PATH)
    assert sent["record_timeout"] == 21600.0
    assert sent["work_dir"] == str(project.resolve())
    assert sent["auto_classify"] is None  # pinned off by the SEA on the daemon
    assert sent["model"] == "some-model"


def test_create_validates_work_dir_and_timeout(tmp_path: Path) -> None:
    """Bad work_dir/timeout values are refused."""
    def create(**fields: object) -> dict:
        return dict(yaml.safe_load(cron_job(
            "create", name="x", schedule="every 5m", **fields,  # type: ignore[arg-type]
        )))

    assert "not a directory" in create(prompt="p", work_dir=str(tmp_path / "no"))["error"]
    assert "not a number" in create(prompt="p", timeout="soon")["error"]
    for bad in ("-5", "nan", "inf"):
        assert "must be a positive number" in create(prompt="p", timeout=bad)["error"]
    # Without the new fields a job stores the defaults and lists none of them.
    job = create(prompt="p")["created"]
    assert not any(key in job for key in ("work_dir", "timeout"))
    stored = load_jobs()[0]
    assert (stored["work_dir"], stored["timeout"]) == ("", 0.0)
    assert load_jobs() == [stored]


def test_command_job_honours_work_dir_and_timeout(tmp_path: Path) -> None:
    """A command job runs in its work_dir and is stopped after its own timeout."""
    project = tmp_path / "project"
    project.mkdir()
    # The command prints its cwd through Python, not the shell's ``pwd``:
    # on Windows the job runs under Git bash, whose ``pwd`` reports the
    # MSYS translation (``/tmp/...``) of the native ``C:\...`` work dir.
    python = Path(sys.executable).as_posix()
    job = _create(cron_job(
        "create", name="pwd", schedule="every 1m", work_dir=str(project),
        command=f'"{python}" -c "import os; print(os.getcwd())"',
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    assert Path(load_jobs()[0]["last_summary"]).resolve() == project.resolve()
    # The command sees the daemon's state directory, so a script it runs
    # with another interpreter reads the same files as the daemon does.
    home_job = _create(cron_job(
        "create", name="home", schedule="every 1m",
        command=f'"{python}" -c "import os; print(os.environ[\'KISS_HOME\'])"',
    ))
    _set_job_fields(home_job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored_home = {j["id"]: j for j in load_jobs()}[home_job["id"]]
    assert Path(stored_home["last_summary"]).resolve() == kiss_home().resolve()
    slow = _create(cron_job(
        "create", name="slow", command="sleep 30", schedule="every 1m",
        timeout="0.5",
    ))
    _set_job_fields(slow["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = {j["id"]: j for j in load_jobs()}[slow["id"]]
    assert stored["last_status"] == "error"
    assert "command timed out after 0.5s" in stored["last_summary"]


def test_same_prompt_in_another_work_dir_is_not_a_duplicate(tmp_path: Path) -> None:
    """The same prompt scheduled for two projects is two jobs, not a duplicate."""
    def create(work_dir: Path) -> str:
        return cron_job(
            "create", name="tests", prompt="run the tests", schedule="every 1d",
            work_dir=str(work_dir),
        )

    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    _create(create(first))
    _create(create(second))
    assert "duplicate" in yaml.safe_load(create(second))["error"]
    assert len(load_jobs()) == 2


def test_unconfirmed_prompt_timeout_names_the_job_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kept-directory note points at the job's work_dir, where the task runs.

    Only the SEA lives in the scratch directory; the possibly still
    running task works in the job's ``work_dir``, so that is the
    directory the error names, with the kept scratch directory beside it.
    The stand-in daemon wait never finishes and, once the job's 0.5-s
    ``run_agent`` bound hands the task back and cron kills it, reports
    the stop as unconfirmed.
    """
    from kiss.agents.sorcar import daemon_client

    def hang(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        kwargs["running"].set()
        kwargs["cancel"].wait()
        raise daemon_client.CancelledError(
            "no terminal status",
            daemon_client.TaskResult(text="", success=False, cost=0.0, tokens=0, steps=0),
            confirmed=False,
        )

    monkeypatch.setattr(daemon_client, "run", hang)
    project = tmp_path / "project"
    project.mkdir()
    job = _create(cron_job(
        "create", name="slow", prompt="think hard", schedule="every 1d",
        work_dir=str(project), timeout="0.5",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "error"
    assert "timed out after 0.5s" in stored["last_summary"]
    assert f"running in {project.resolve()}" in stored["last_summary"]
    assert "MAY STILL BE RUNNING" in stored["last_summary"]
    kept = [p for p in (tmp_path / "cron" / "runs").iterdir() if p.is_dir()]
    assert len(kept) == 1 and kept[0].is_dir()
    assert f"scratch directory {kept[0]}" in stored["last_summary"]


def test_cli_create_with_work_dir_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """kiss-cron --create passes --work-dir/--timeout through."""
    project = tmp_path / "project"
    project.mkdir()
    out = _run_cli(
        monkeypatch, capsys, "--create", "nightly", "--schedule", "0 12 * * *",
        "--prompt", "run the tests", "--work-dir", str(project),
        "--timeout", "7200", "-m", "some-model",
    )
    created = yaml.safe_load(out)["created"]
    assert created["work_dir"] == str(project.resolve())
    assert created["timeout"] == 7200.0
    assert created["model_name"] == "some-model"
    listed = yaml.safe_load(_run_cli(monkeypatch, capsys, "--list"))["jobs"][0]
    assert listed["work_dir"] == str(project.resolve())


def _ensure(prompt: str = "do the thing") -> dict:
    """Parse the reply of ``cron_job("ensure")`` for the test's weekly job spec."""
    return dict(yaml.safe_load(
        cron_job("ensure", name="weekly thing", schedule="0 1 * * 6", prompt=prompt)
    ))


def test_ensure_creates_once_keeps_an_enabled_job_and_resumes_a_paused_one() -> None:
    """``ensure`` is keyed by name: created, then ``exists``, then ``resumed`` after a pause.

    A same-named job with another prompt is still one job (the name is
    the identity), while ``create`` of the identical spec is refused as a
    duplicate; the store's lock makes racing ``ensure`` calls create one.
    """
    assert yaml.safe_load(cron_job("ensure", command="x")) == {
        "error": "ensure requires name and schedule",
    }
    created = _ensure()
    assert "created" in created, created
    job_id = created["created"]["id"]
    assert _ensure() == {"exists": created["created"]}
    assert _ensure("do it differently")["exists"]["id"] == job_id
    refused = cron_job("create", name="weekly thing", schedule="0 1 * * 6", prompt="do the thing")
    assert "duplicate" in yaml.safe_load(refused)["error"]
    cron_job("pause", job_id=job_id)
    assert load_jobs()[0]["enabled"] is False
    assert _ensure()["resumed"]["id"] == job_id
    assert load_jobs()[0]["enabled"] is True
    assert len(load_jobs()) == 1
    # A hand-renamed equivalent job (same prompt, schedule, delivery) is the job too.
    _set_job_fields(job_id, name="my renamed weekly thing")
    assert _ensure()["exists"]["id"] == job_id
    cron_job("pause", job_id=job_id)
    assert _ensure()["resumed"]["id"] == job_id
    assert len(load_jobs()) == 1

    barrier = threading.Barrier(6)
    replies: list[str] = []

    def ensure(i: int) -> None:
        barrier.wait()
        replies.append(cron_job("ensure", name="raced", schedule="every 1h", prompt=f"p{i}"))

    threads = [threading.Thread(target=ensure, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    raced = [job for job in load_jobs() if job["name"] == "raced"]
    assert len(raced) == 1, raced
    assert sum("created" in yaml.safe_load(reply) for reply in replies) == 1, replies


def test_ensure_of_a_finished_one_shot_reports_the_resume_error() -> None:
    """A paused one-shot job that already ran cannot be re-enabled by ``ensure`` either."""
    created = _create(cron_job("create", name="once", schedule="5m", prompt="ping"))
    _set_job_fields(created["id"], enabled=False, next_run_at=None)
    reply = yaml.safe_load(cron_job("ensure", name="once", schedule="5m", prompt="ping"))
    assert reply == {
        "error": f"one-shot job {created['id']!r} already ran; create a new job instead",
    }


def _wait_for_run(job_id: str, timeout: float = 20.0) -> dict[str, Any]:
    """Wait until the job's run thread finished and return the stored job."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job_id not in cron_agent.running_job_ids():
            return next(job for job in load_jobs() if job["id"] == job_id)
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


def test_apply_job_action_pause_resume_update_remove(tmp_path: Path) -> None:
    """The Schedule subpanel's buttons edit the stored job in place."""
    job = _create(cron_job(
        "create", name="hello", command="echo hi", schedule="every 5m",
    ))
    job_id = job["id"]
    assert cron_agent.apply_job_action("pause", job_id) == ""
    assert load_jobs()[0]["enabled"] is False
    assert cron_agent.apply_job_action("resume", job_id) == ""
    assert load_jobs()[0]["enabled"] is True

    # An update keeps the job's kind (a command job stays one) and
    # recomputes the next run only when the schedule changed.
    before = load_jobs()[0]["next_run_at"]
    assert cron_agent.apply_job_action(
        "update", job_id, name=" greeter ", schedule="every 5m", what="echo bye",
    ) == ""
    updated = load_jobs()[0]
    assert updated["name"] == "greeter"
    assert updated["command"] == "echo bye" and not updated.get("prompt")
    assert updated["next_run_at"] == before
    assert cron_agent.apply_job_action(
        "update", job_id, name="greeter", schedule="2h", what="echo bye",
    ) == ""
    updated = load_jobs()[0]
    assert updated["schedule"] == "2h" and updated["one_shot"] is True
    assert updated["next_run_at"] > before

    assert cron_agent.apply_job_action("remove", job_id) == ""
    assert load_jobs() == []


def test_apply_job_action_errors() -> None:
    job = _create(cron_job("create", name="p", prompt="summarize", schedule="every 1h"))
    job_id = job["id"]
    assert cron_agent.apply_job_action("pause", "nope") == "no job with id 'nope'"
    assert cron_agent.apply_job_action("explode", job_id) == "unknown action 'explode'"
    assert cron_agent.apply_job_action(
        "update", job_id, name="", schedule="every 1h", what="x",
    ) == "the name, the schedule and the task must not be empty"
    assert cron_agent.apply_job_action(
        "update", job_id, name="p", schedule="whenever", what="x",
    ).startswith("Unsupported schedule 'whenever':")
    assert cron_agent.apply_job_action(
        "update", job_id, name="p", schedule="2000-01-01T00:00:00", what="x",
    ) == "schedule '2000-01-01T00:00:00' never fires (in the past?)"
    stored = load_jobs()[0]
    assert stored["schedule"] == "every 1h" and stored["prompt"] == "summarize"
    # A finished one-shot cannot be resumed.
    _set_job_fields(job_id, enabled=False, one_shot=True, next_run_at=None)
    assert "already ran" in cron_agent.apply_job_action("resume", job_id)
    # A prompt job's update replaces its prompt.
    _set_job_fields(job_id, enabled=True, one_shot=False, next_run_at=1.0)
    assert cron_agent.apply_job_action(
        "update", job_id, name="p", schedule="every 1h", what="digest",
    ) == ""
    assert load_jobs()[0]["prompt"] == "digest"


def test_apply_job_action_create_schedules_a_job() -> None:
    """The Schedule header's + form: a prompt or command job with the tool's defaults."""
    create = cron_agent.apply_job_action
    assert create("create", "", "Digest", "every 1d", "summarize email") == ""
    (job,) = load_jobs()
    assert (job["name"], job["schedule"], job["prompt"], job["command"]) == (
        "Digest", "every 1d", "summarize email", "",
    )
    assert job["deliver"] == "local" and job["enabled"] is True
    assert job["next_run_at"] > time.time()
    assert create(
        "create", "ignored", " Backup ", " 0 3 * * * ", "rsync -a src dst", kind="command",
    ) == ""
    backup = load_jobs()[1]
    assert (backup["name"], backup["schedule"], backup["prompt"], backup["command"]) == (
        "Backup", "0 3 * * *", "", "rsync -a src dst",
    )
    # Refusals: a blank field, an unknown kind, a bad schedule and a
    # duplicate leave the two jobs alone.
    blank = "the name, the schedule and the task must not be empty"
    assert create("create", "", "", "every 1d", "x") == blank
    assert create("create", "", "N", "  ", "x") == blank
    assert create("create", "", "N", "every 1d", " ") == blank
    assert create("create", "", "N", "every 1d", "x", kind="both") == "unknown job kind 'both'"
    assert create("create", "", "N", "whenever", "x").startswith("Unsupported schedule 'whenever':")
    assert create("create", "", "Digest", "every 1d", "summarize email").startswith(
        "duplicate: job "
    )
    assert len(load_jobs()) == 2


def test_apply_job_action_run_now_starts_a_thread_once(tmp_path: Path) -> None:
    """``run_now`` returns at once; the job runs in its own thread; a second start is refused."""
    job = _create(cron_job(
        "create", name="slow", command="sleep 1; echo ran now", schedule="every 1d",
    ))
    job_id = job["id"]
    cron_job("pause", job_id=job_id)
    assert cron_agent.apply_job_action("run_now", job_id) == ""
    assert job_id in cron_agent.running_job_ids()
    assert cron_agent.apply_job_action("run_now", job_id) == f"job {job_id!r} is already running"
    stored = _wait_for_run(job_id)
    assert stored["last_status"] == "ok"
    assert stored["last_summary"] == "ran now"
    assert stored["enabled"] is False, "a run now does not resume a paused job"
    assert cron_agent.apply_job_action("run_now", "nope") == "no job with id 'nope'"


def test_run_context_names_time_window_and_previous_result() -> None:
    """The block carries now, the schedule, the window start, the previous attempt, its result."""
    job = {
        "id": "abcd1234", "schedule": "every 1h", "previous_run_at": _ts(2026, 3, 2, 9, 0),
        "covered_until": _ts(2026, 3, 2, 8, 0), "last_status": "failed",
        "last_summary": "x" * 2000 + "\nlast line",
    }
    block = cron_agent.run_context(job, now=_ts(2026, 3, 2, 10, 0))
    lines = block.splitlines()
    assert lines[0] == "# Run context"
    assert lines[1] == "- Now: 2026-03-02 10:00:00 PST (2026-03-02 18:00 UTC)"
    assert lines[2] == "- Schedule: every 1h"
    assert lines[3] == (
        "- Window start: 2026-03-02 08:00:00 PST (the last successful run; cover everything since)"
    )
    assert lines[4] == "- Previous attempt: 2026-03-02 09:00:00 PST (failed)"
    assert lines[5] == "- Previous result (failed, tail):"
    assert lines[-2] == "last line" and "x" * 1400 in lines[-3]
    assert lines[-1].endswith("/abcd1234.md itself: do not write a log.")
    first = cron_agent.run_context({"id": "x"}, now=_ts(2026, 3, 2, 10, 0))
    assert "- Window start: not set (no successful run yet; use your judgement)" in first
    assert "- Previous attempt: none (first run)" in first
    assert "Previous result" not in first
    interval_job = {
        "id": "y", "last_run_at": _ts(2026, 3, 2, 9, 0), "next_run_at": _ts(2026, 3, 2, 10, 0),
        "previous_run_at": _ts(2026, 3, 2, 8, 0), "last_status": "failed",
    }
    second = cron_agent.run_context(interval_job, now=_ts(2026, 3, 2, 10, 0))
    assert "- Window start: 2026-03-02 09:00:00 PST (one schedule interval back)" in second
    # A new job's watermark is its creation until a run succeeds.
    fresh = {"id": "z", "created_at": _ts(2026, 3, 2, 7, 0), "covered_until": _ts(2026, 3, 2, 7, 0)}
    assert (
        "- Window start: 2026-03-02 07:00:00 PST (the job was created; no run has "
        "succeeded yet; cover everything since)"
    ) in cron_agent.run_context(fresh, now=_ts(2026, 3, 2, 10, 0))


def test_covered_until_advances_on_success_or_silence_and_survives_failures(
    tmp_path: Path,
) -> None:
    """The watermark starts at creation, moves to the start of an ok/silent run, never on error."""
    job = _create(cron_job("create", name="cov", command="echo fine", schedule="every 1d"))
    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    assert stored["covered_until"] == stored["created_at"]
    before = time.time()
    cron_agent.apply_job_action("run_now", job["id"])  # a run now never sets last_run_at
    stored = _wait_for_run(job["id"])
    assert stored["last_status"] == "ok" and stored["last_run_at"] is None
    assert before <= stored["covered_until"] <= time.time()
    covered = stored["covered_until"]
    _set_job_fields(job["id"], command="exit 3")
    cron_agent.apply_job_action("run_now", job["id"])
    failed = _wait_for_run(job["id"])
    assert failed["last_status"] == "error" and failed["covered_until"] == covered
    _set_job_fields(job["id"], command="true", last_status="")  # empty output: silent
    cron_agent.apply_job_action("run_now", job["id"])
    silent = _wait_for_run(job["id"])
    assert silent["last_status"] == "silent" and silent["covered_until"] > covered


def test_failure_backoff_doubles_to_a_day() -> None:
    assert cron_agent.failure_backoff_seconds(0) == 0
    assert [cron_agent.failure_backoff_seconds(n) for n in (1, 2, 3, 5, 6)] == [
        3600, 7200, 14400, 57600, 86400,
    ]


def test_repeated_identical_failure_is_delivered_once_and_backs_off(tmp_path: Path) -> None:
    """The second identical failure is not written to the log and pushes next_run_at by an hour."""
    job = _create(cron_job(
        "create", name="broken", command="echo same-error >&2; exit 3",
        schedule="every 1m",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    first = load_jobs()[0]
    assert first["last_status"] == "error" and first["failure_streak"] == 0
    log_path = tmp_path / "cron" / "output" / f"{job['id']}.md"
    assert log_path.read_text(encoding="utf-8").count("same-error") == 1
    _set_job_fields(job["id"], next_run_at=1.0)
    before = time.time()
    assert tick(2.0) == 1
    second = load_jobs()[0]
    assert second["failure_streak"] == 1
    assert second["last_delivery"][0].startswith("repeat of the previous failure (1)")
    assert second["next_run_at"] >= before + 3600
    assert log_path.read_text(encoding="utf-8").count("same-error") == 1
    # A different failure (or a success) resets the streak and is delivered.
    _set_job_fields(job["id"], next_run_at=1.0, command="echo other-error >&2; exit 4")
    assert tick(2.0) == 1
    third = load_jobs()[0]
    assert third["failure_streak"] == 0
    assert log_path.read_text(encoding="utf-8").count("other-error") == 1


def test_prompt_job_naming_an_unconnected_connector_does_not_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt that names Gmail with no Gmail connection fails before any daemon call."""
    from kiss.agents.sorcar import daemon_client

    def never(prompt: str, **kwargs: object) -> daemon_client.TaskResult:
        raise AssertionError("the run must not be dispatched")

    monkeypatch.setattr(daemon_client, "run", never)
    monkeypatch.delenv("KISS_SORCAR_LOCAL", raising=False)
    job = _create(cron_job(
        "create", name="mail", prompt="Summarise my Gmail inbox", schedule="every 1h",
        deliver="none",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "error"
    assert stored["last_summary"].startswith("Gmail is not connected: this scheduled job")
    assert "/gmail authenticate" in stored["last_summary"]
    assert cron_agent.connector_problem("list my open pull requests") is None


def test_every_connector_word_resolves_to_a_sea_with_a_status_hook() -> None:
    """Each SEA named by CONNECTOR_WORDS exposes connection_status(); none is connected here."""
    import importlib

    for word, (sea, label) in cron_agent.CONNECTOR_WORDS.items():
        module = importlib.import_module(f"kiss.agents.third_party_agents.{sea}.{sea}_sea")
        assert module.connection_status() == "", (word, sea)
        problem = cron_agent.connector_problem(f"check {word} now")
        assert problem is not None and problem.startswith(f"{label} is not connected"), word
        assert f"/{sea} authenticate" in problem, word
    assert cron_agent.CONNECTOR_WORDS["googlechat"][0] == "googlechat"


def test_connector_preflight_skips_authentication_and_in_chat_jobs() -> None:
    """A job that reconnects a connector, or runs in a chat with a person, is never held back."""
    assert cron_agent.connector_problem("Summarise my Gmail inbox") is not None
    assert cron_agent.connector_problem("/gmail authenticate") is None
    assert cron_agent.connector_problem("Reconnect Google Drive for me") is None
    assert cron_agent.connector_problem("Summarise my Gmail inbox", in_chat=True) is None


def test_cron_prompt_sea_swaps_the_full_base_for_the_lite_one() -> None:
    """A scheduled run gets the lite base plus the scheduled-run rules; the suffix survives."""
    from kiss.agents.sorcar.cron_prompt_sea import CRON_RUN_RULES, CronPromptSea
    from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE

    sea = CronPromptSea()
    rewritten = sea.system_prompt(SYSTEM_PROMPT + "\n\n# House rule\nbe brief")
    assert rewritten == SYSTEM_PROMPT_LITE + CRON_RUN_RULES + "\n\n# House rule\nbe brief"
    assert len(rewritten) < len(SYSTEM_PROMPT) // 2
    assert "never call ask_user_question" in rewritten
    assert sea.system_prompt("custom base") == "custom base" + CRON_RUN_RULES
    assert sea.settings({"auto_classify": True})["auto_classify"] is False
