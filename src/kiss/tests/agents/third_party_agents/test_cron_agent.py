# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Hermes-style cron automations (cron_agent).

Everything runs against the real JSON job store under an isolated
``KISS_HOME`` — no mocks or test doubles (``monkeypatch`` is used
only to isolate environment variables, ``sys.argv``, and the
module-level daemon-socket default between tests).  The only
branches not exercised here are ``_run_prompt_job``'s successful /
silent / timed-out LLM paths: they submit a task to the kiss-web
daemon and require a live LLM endpoint, which is unavailable (and
non-deterministic) in unit tests; the failure path is covered via
``_execute_job``'s exception handling.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import (
    compute_next_run,
    cron_job,
    is_one_shot,
    load_jobs,
    main,
    start_scheduler_thread,
    tick,
)


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KISS_HOME at a per-test temp dir so the job store is isolated.

    Also resets the module-level daemon socket default so a scheduler
    started by one test cannot redirect another test's prompt jobs.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "_daemon_sock_path", None)
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
    with cron_agent._jobs_lock(blocking=True):
        assert tick(2.0) == 0
    assert tick(2.0) == 1


def test_prompt_job_failure_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no reachable kiss-web daemon socket (the isolated KISS_HOME
    # contains no sorcar.sock), a prompt job's sorcar.run raises and
    # _execute_job records the error end-to-end.
    monkeypatch.delenv("KISS_SORCAR_SOCK", raising=False)
    job = _create(cron_job(
        "create", name="llm", prompt="say hi", schedule="every 1m",
        deliver="none",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert stored["last_status"] == "error"
    # The error explains what is missing instead of a bare traceback.
    assert "kiss-web daemon" in stored["last_summary"]


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
    # The module lives in the sorcar package and never imports from
    # kiss.agents.third_party_agents at module scope.
    source_text = Path(cron_agent.__file__).read_text(encoding="utf-8")
    assert "/agents/sorcar/" in cron_agent.__file__
    assert "from kiss.agents.third_party_agents" not in source_text
    assert "import kiss.agents.third_party_agents" not in source_text
    # cron_job is NOT a built-in tool of the default Sorcar toolset:
    # scheduling requests go through run_agent("cron", ...), which
    # dispatches this module as an agent script.
    agent_source = Path(cron_agent.__file__).parent / "sorcar_agent.py"
    agent_text = agent_source.read_text(encoding="utf-8")
    assert "tools.append(cron_job)" not in agent_text
    assert "from kiss.agents.sorcar.cron_agent import cron_job" not in agent_text
    dispatch_source = Path(cron_agent.__file__).parent / "agent_dispatch.py"
    assert (
        "cron_agent.CRON_DISPATCH_PREAMBLE + task"
        in dispatch_source.read_text(encoding="utf-8")
    )
    # The system prompt directs scheduling requests to run_agent("cron").
    system_md = Path(cron_agent.__file__).parents[2] / "SYSTEM.md"
    assert 'run_agent tool with "cron"' in system_md.read_text(encoding="utf-8")
    # The kiss-cron CLI entry point is wired in pyproject.toml.
    pyproject = Path(cron_agent.__file__).parents[4] / "pyproject.toml"
    assert (
        'kiss-cron = "kiss.agents.sorcar.cron_agent:main"'
        in pyproject.read_text(encoding="utf-8")
    )


def test_agent_script_getters(tmp_path: Path) -> None:
    # The agent-script contract used by run_agent("cron", ...): the
    # dispatched session runs in ~/.kiss/cron/work with no git
    # lifecycle.
    work_dir = cron_agent.get_work_dir()
    assert work_dir == str(tmp_path / "cron" / "work")
    assert Path(work_dir).is_dir()
    assert cron_agent.get_use_worktree() is False
    assert cron_agent.get_auto_commit() is False
    assert "cron_job" in cron_agent.CRON_DISPATCH_PREAMBLE


def test_store_is_plain_json_list(tmp_path: Path) -> None:
    _create(cron_job("create", name="a", command="echo 1", schedule="every 1m"))
    raw = json.loads((tmp_path / "cron" / "jobs.json").read_text())
    assert isinstance(raw, list) and raw[0]["name"] == "a"


def test_silence_tokens() -> None:
    assert cron_agent._is_silent("[SILENT]")
    assert cron_agent._is_silent("<p>[SILENT]</p>\n")
    assert cron_agent._is_silent("NO_REPLY")
    assert not cron_agent._is_silent("all good")


# ------------------------------------------------------ scheduler thread


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


def test_run_scheduler_returns_when_stop_event_preset() -> None:
    stop_event = threading.Event()
    stop_event.set()
    cron_agent.run_scheduler(stop_event, interval=0.01)  # returns immediately


def test_run_now_uses_daemon_sock_path(tmp_path: Path) -> None:
    # A scheduler started for a custom-UDS daemon records its socket as
    # the module default, so run_now prompt jobs target that daemon
    # instead of $KISS_HOME/sorcar.sock.
    custom_sock = tmp_path / "custom-daemon.sock"
    stop_event = start_scheduler_thread(interval=999.0, sock_path=str(custom_sock))
    try:
        job = _create(cron_job(
            "create", name="llm", prompt="say hi", schedule="every 1h",
            deliver="none",
        ))
        reply = yaml.safe_load(cron_job("run_now", job_id=job["id"]))
        assert reply["ran"]["last_status"] == "error"
        assert "custom-daemon.sock" in load_jobs()[0]["last_summary"]
    finally:
        _stop_scheduler(stop_event)


def test_tools_file_loaded_run_now_uses_daemon_sock_path(
    tmp_path: Path,
) -> None:
    # A run_agent("cron", ...) session gets its cron_job tool from a
    # FRESH synthetic module (the daemon's tools-file loader re-executes
    # this file), whose own _daemon_sock_path global is never set:
    # run_now must still target the socket recorded in the canonical
    # module by the daemon's scheduler thread.
    from kiss.server.tools_file import ToolsFileError, execute_python_file

    custom_sock = tmp_path / "custom-daemon.sock"
    stop_event = start_scheduler_thread(interval=999.0, sock_path=str(custom_sock))
    try:
        namespace = execute_python_file(
            cron_agent.__file__, ToolsFileError, "tools file",
        )
        loaded_cron_job = namespace["get_tools"]()[0]
        # A distinct module copy — the very situation the canonical
        # lookup exists for.
        assert loaded_cron_job is not cron_job
        assert namespace["_daemon_sock_path"] is None
        job = _create(loaded_cron_job(
            "create", name="llm", prompt="say hi", schedule="every 1h",
            deliver="none",
        ))
        reply = yaml.safe_load(loaded_cron_job("run_now", job_id=job["id"]))
        assert reply["ran"]["last_status"] == "error"
        assert "custom-daemon.sock" in load_jobs()[0]["last_summary"]
    finally:
        _stop_scheduler(stop_event)


def test_scheduler_thread_runs_due_jobs_and_stops() -> None:
    job = _create(cron_job(
        "create", name="sched", command="echo scheduled", schedule="every 1h",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
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


def test_kiss_web_daemon_runs_scheduler_thread(tmp_path: Path) -> None:
    """The kiss-web daemon starts the cron thread, the thread executes a
    due job, and shutdown stops the thread."""
    import asyncio

    from kiss.server.web_server import RemoteAccessServer

    job = _create(cron_job(
        "create", name="boot job", command="echo ran inside daemon",
        schedule="every 1h",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)  # due on the first tick

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    async def _run() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=port,
            use_tunnel=False,
            work_dir=str(tmp_path),
            uds_path=str(tmp_path / "sorcar.sock"),
        )
        task = asyncio.ensure_future(server._serve_async())
        try:
            for _ in range(200):
                if (
                    load_jobs()[0]["last_status"] == "ok"
                    and server._shutdown_future is not None
                ):
                    break
                await asyncio.sleep(0.05)
            assert _cron_thread() is not None, "cron scheduler thread not started"
            assert load_jobs()[0]["last_summary"] == "ran inside daemon"
        finally:
            for _ in range(200):
                if server._shutdown_future is not None:
                    break
                await asyncio.sleep(0.05)
            assert server._shutdown_future is not None
            if not server._shutdown_future.done():
                server._shutdown_future.set_result(None)
            await task
        for _ in range(200):
            if _cron_thread() is None:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("cron scheduler thread still alive after shutdown")

    asyncio.run(_run())
