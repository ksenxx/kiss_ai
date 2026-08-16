# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Hermes-style cron automations (cron_agent).

Everything runs against the real JSON job store under an isolated
``KISS_HOME`` — no mocks, patches, or test doubles.  The only branch
not exercised here is ``_run_prompt_job``'s successful LLM path: it
submits a task to the kiss-web daemon and requires a live LLM
endpoint, which is unavailable (and non-deterministic) in unit tests;
its failure path is covered via ``_execute_job``'s exception handling.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from kiss.agents.third_party_agents import cron_agent
from kiss.agents.third_party_agents._channel_agent_utils import channel_state_lock
from kiss.agents.third_party_agents.cron_agent import (
    compute_next_run,
    cron_job,
    is_one_shot,
    load_jobs,
    main,
    tick,
)


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KISS_HOME at a per-test temp dir so the job store is isolated."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    return tmp_path


def _ts(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    """Return the epoch timestamp of a local datetime."""
    return datetime(year, month, day, hour, minute).timestamp()


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


# ---------------------------------------------------------------- schedules


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


# ------------------------------------------------------------ tool actions


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


# --------------------------------------------------------- tick + delivery


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
    log = (tmp_path / "cron" / "output" / f"{job['id']}.md").read_text()
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
    ).read_text()


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
    log = (tmp_path / "cron" / "output" / f"{job['id']}.md").read_text()
    assert "exited 3" in log


def test_delivery_error_notes() -> None:
    job = _create(cron_job(
        "create", name="multi", command="echo payload",
        schedule="every 1m",
        deliver="local,nosuchchannel:1,homeassistant:x,telegram:123",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    notes = load_jobs()[0]["last_delivery"]
    assert len(notes) == 3
    assert "unknown channel 'nosuchchannel'" in notes[0]
    assert "does not support delivery" in notes[1]
    # Telegram module exists and has _make_backend, but no credentials
    # exist under the isolated KISS_HOME, so its factory sys.exit(1)s.
    assert "not authenticated" in notes[2]


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
    with channel_state_lock(cron_agent._jobs_path(), blocking=True):
        assert tick(2.0) == 0
    assert tick(2.0) == 1


def test_prompt_job_failure_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no reachable kiss-web daemon socket, a prompt job's launch
    # raises and _execute_job records the error end-to-end.
    job = _create(cron_job(
        "create", name="llm", prompt="say hi", schedule="every 1m",
        deliver="none",
    ))
    import kiss.agents.third_party_agents._kiss_web_launcher as launcher

    monkeypatch.setattr(launcher, "_SOCK_PATH_OVERRIDE", "/nonexistent/cron.sock")
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "error"
    assert stored["last_summary"]


# ------------------------------------------------------------------- CLI


def _run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
             *argv: str) -> str:
    monkeypatch.setattr(sys, "argv", ["kiss-cron", *argv])
    main()
    return str(capsys.readouterr().out)


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
    _set_job_fields(job_id, next_run_at=1.0)
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


# --------------------------------------------------------------- wiring


def test_get_tools_and_sorcar_wiring() -> None:
    assert cron_agent.get_tools() == [cron_job]
    # The default Sorcar toolset registers the tool.
    source = Path(
        cron_agent.__file__
    ).parent.parent / "sorcar" / "sorcar_agent.py"
    assert "tools.append(cron_job)" in source.read_text(encoding="utf-8")
    # The kiss-cron CLI entry point is wired in pyproject.toml.
    pyproject = Path(cron_agent.__file__).parents[4] / "pyproject.toml"
    assert (
        'kiss-cron = "kiss.agents.third_party_agents.cron_agent:main"'
        in pyproject.read_text(encoding="utf-8")
    )


def test_store_is_plain_json_list(tmp_path: Path) -> None:
    _create(cron_job("create", name="a", command="echo 1", schedule="every 1m"))
    raw = json.loads((tmp_path / "cron" / "jobs.json").read_text())
    assert isinstance(raw, list) and raw[0]["name"] == "a"
