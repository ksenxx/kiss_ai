# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for concurrent cron runs, per-run scratch dirs, and gateways.

Covers the scheduler behaviour added on top of the base cron agent:

- jobs due at the same tick run concurrently, each in its own scratch
  directory under ``~/.kiss/cron/runs/`` that is removed when the run
  ends (also on failure and when the run left files behind);
- a tick that overlaps runs from a previous tick is not skipped: it
  starts the due jobs that are idle and leaves the still-running one
  alone, which runs again on the first tick after it finishes;
- the scheduler thread never blocks on a long job;
- ``gateway_command`` turns a channel + chat into the channel CLI's
  tick command so an always-on gateway is scheduled as a command job.

Everything runs against the real JSON store under an isolated
``KISS_HOME`` (fixture reused from ``test_cron_agent``); only the
daemon-client boundary of prompt jobs is captured, as in
``test_cron_agent.test_prompt_job_runs_as_generated_sea``.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import (
    cron_job,
    gateway_command,
    load_jobs,
    running_job_ids,
    start_scheduler_thread,
    tick,
)
from kiss.tests.agents.sorcar.test_cron_agent import (  # noqa: F401
    _create,
    _isolated_kiss_home,
    _set_job_fields,
    _stop_scheduler,
)
from kiss.tests.conftest import IS_WINDOWS


@pytest.fixture(autouse=True)
def _join_job_threads() -> Iterator[None]:
    """Wait for every job thread this test launched, even after a failure.

    Runs launched with ``tick(wait=False)`` or by the scheduler thread
    write to the job store through the ``KISS_HOME`` environment; they
    must be over before the isolated-home fixture is torn down.
    """
    yield
    for thread in list(cron_agent._running.values()):
        thread.join(timeout=10)
    cron_agent._running.clear()


def _runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "cron" / "runs"


def _stored(job_id: str) -> dict:
    return next(job for job in load_jobs() if job["id"] == job_id)


def _wait_until(predicate, timeout: float = 5.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


def _due_command_job(name: str, command: str) -> dict:
    job = _create(cron_job(
        "create", name=name, command=command, schedule="every 1m", deliver="none",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    return job


# Command jobs run under Git bash on Windows, whose plain ``pwd`` prints the
# MSYS view of the directory (``/tmp/...`` for ``%TEMP%``); ``pwd -W`` prints
# the native path the test compares against.
_PWD = "pwd -W" if IS_WINDOWS else "pwd"


def test_due_jobs_run_concurrently_in_private_work_dirs(tmp_path: Path) -> None:
    # Distinct commands: an identical command on the same schedule would be
    # rejected by cron_job("create") as a duplicate.
    first = _due_command_job("first", f"sleep 1; {_PWD}")
    second = _due_command_job("second", f"sleep 1 && {_PWD}")
    started = time.monotonic()
    assert tick(2.0) == 2
    # Two 1 s jobs finishing in under 2 s can only have overlapped.
    assert time.monotonic() - started < 1.8
    cwds = [Path(_stored(job["id"])["last_summary"]) for job in (first, second)]
    assert cwds[0] != cwds[1]
    for job, cwd in zip((first, second), cwds, strict=True):
        assert cwd.parent.resolve() == _runs_dir(tmp_path).resolve()
        assert cwd.name.startswith(job["id"] + "-")
        assert not cwd.exists()
    assert list(_runs_dir(tmp_path).iterdir()) == []


def test_work_dir_removed_after_failure_and_leftover_files(tmp_path: Path) -> None:
    failing = _due_command_job("fails", "touch leftover; echo boom >&2; exit 3")
    assert tick(2.0) == 1
    stored = _stored(failing["id"])
    assert stored["last_status"] == "error"
    assert "boom" in stored["last_summary"]
    assert list(_runs_dir(tmp_path).iterdir()) == []


def test_prompt_job_runs_in_private_work_dir_and_stops_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kiss.agents.sorcar import daemon_client

    captured: list[dict[str, object]] = []
    seen_dirs: list[bool] = []

    def capture_run(prompt: str, **kwargs: object) -> daemon_client.TaskResult:
        captured.append(kwargs)
        seen_dirs.append(Path(str(kwargs["work_dir"])).is_dir())
        if len(captured) == 2:
            raise TimeoutError("late")
        if len(captured) == 3:
            raise daemon_client.StopUnconfirmedTimeoutError("no terminal status")
        return daemon_client.TaskResult(
            text="hello", success=True, cost=0.0, tokens=0, steps=0,
        )

    monkeypatch.setattr(daemon_client, "run", capture_run)
    job = _create(cron_job(
        "create", name="llm", prompt="say hi", schedule="every 1m", deliver="none",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    work_dir = Path(str(captured[0]["work_dir"]))
    assert work_dir.parent.resolve() == _runs_dir(tmp_path).resolve()
    assert work_dir.name.startswith(job["id"] + "-")
    assert seen_dirs == [True]
    assert not work_dir.exists()
    assert captured[0]["stop_on_timeout"] is True
    assert _stored(job["id"])["last_summary"] == "hello"

    # A timed-out run is reported as stopped and its directory removed too.
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(3.0) == 1
    stored = _stored(job["id"])
    assert stored["last_status"] == "error"
    assert "did not finish within" in stored["last_summary"]
    assert "was stopped" in stored["last_summary"]
    assert list(_runs_dir(tmp_path).iterdir()) == []

    # A stop the daemon never confirmed means the task may still be using
    # the directory: it is kept, and the summary says so instead of
    # claiming the task was stopped.
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(4.0) == 1
    stored = _stored(job["id"])
    kept = Path(str(captured[2]["work_dir"]))
    assert stored["last_status"] == "error"
    assert "was not confirmed" in stored["last_summary"]
    assert "MAY STILL BE RUNNING" in stored["last_summary"]
    assert str(kept) in stored["last_summary"]
    assert "was stopped" not in stored["last_summary"]
    assert kept.is_dir()
    assert [path.name for path in _runs_dir(tmp_path).iterdir()] == [kept.name]


def test_overlapping_tick_runs_idle_jobs_and_skips_the_running_one(tmp_path: Path) -> None:
    slow = _due_command_job("slow", "sleep 1.5; echo slow-done")
    quick = _due_command_job("quick", "echo quick-done")
    started = time.monotonic()
    assert tick(2.0, wait=False) == 2
    # Without ``wait`` the tick returns while the slow job is still running.
    assert time.monotonic() - started < 1.0
    _wait_until(lambda: running_job_ids() == {slow["id"]})
    assert _stored(quick["id"])["last_summary"] == "quick-done"

    # Both jobs are due again; the overlapping tick is not skipped: it
    # runs the quick job and leaves the still-running slow job alone.
    for job in (slow, quick):
        _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(3.0) == 1
    assert _stored(quick["id"])["last_run_at"] == 3.0
    assert _stored(slow["id"])["last_run_at"] == 2.0
    assert _stored(slow["id"])["next_run_at"] == 1.0
    assert running_job_ids() == {slow["id"]}

    # Once the slow job finishes it is due again and the next tick runs it.
    _wait_until(lambda: running_job_ids() == set())
    assert _stored(slow["id"])["last_summary"] == "slow-done"
    assert tick(4.0) == 1
    assert _stored(slow["id"])["last_run_at"] == 4.0
    assert list(_runs_dir(tmp_path).iterdir()) == []


def test_scheduler_thread_does_not_block_on_long_job(tmp_path: Path) -> None:
    slow = _due_command_job("slow", "sleep 1.5; echo slow-done")
    stop_event = start_scheduler_thread(interval=0.05)
    try:
        _wait_until(lambda: running_job_ids() == {slow["id"]})
        # Created while the slow job runs: the scheduler must still pick it
        # up on its next tick instead of waiting for the slow job.
        quick = _create(cron_job(
            "create", name="quick", command="echo quick-done", schedule="every 1m",
            deliver="none",
        ))
        with cron_agent._jobs_lock(blocking=True):
            _set_job_fields(quick["id"], next_run_at=1.0)
        _wait_until(lambda: _stored(quick["id"]).get("last_status") == "ok", timeout=1.0)
        assert slow["id"] in running_job_ids()
        _wait_until(lambda: _stored(slow["id"]).get("last_status") == "ok")
    finally:
        _stop_scheduler(stop_event)
    _wait_until(lambda: running_job_ids() == set())
    assert list(_runs_dir(tmp_path).iterdir()) == []


def test_running_registry_prunes_finished_threads() -> None:
    job = _due_command_job("fast", "echo done")
    assert tick(2.0) == 1
    assert running_job_ids() == set()
    assert job["id"] not in cron_agent._running
    # A registered thread that already exited is treated as not running.
    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join()
    cron_agent._running["deadbeef"] = finished
    assert running_job_ids() == set()
    assert "deadbeef" not in cron_agent._running


def test_gateway_command_builds_channel_cli_tick() -> None:
    assert gateway_command("Telegram", "-100123") == (
        "kiss-telegram --channel=-100123 --pairing --quiet"
    )
    # Console-script names differ from module names for some channels; the
    # chat and workspace are shell-quoted.
    assert gateway_command("google chat", "spaces/AAA", pairing=False, workspace="w s") == (
        "kiss-gchat --channel=spaces/AAA --workspace='w s' --quiet"
    )
    assert gateway_command("nosuchchannel", "x").startswith("error: unknown channel")
    assert gateway_command("slack", "   ") == "error: chat is required (a chat id or channel name)"
    # A channel without a poll backend has no gateway mode.
    assert gateway_command("homeassistant", "x") == (
        "error: channel 'homeassistant' has no gateway (poll) mode"
    )
    assert cron_agent._channel_cli_name("notinstalled") == "kiss-notinstalled"
    assert "gateway_command" in cron_agent.CRON_DISPATCH_PREAMBLE
    assert "gateway_command" in str(cron_job.__doc__)


def test_gateway_command_is_schedulable_as_command_job() -> None:
    # The produced command is a plain shell line that cron_job("create")
    # accepts as a command job (never a prompt job).  The real tick of that
    # command is exercised in
    # tests/agents/third_party_agents/test_cron_agent.py.
    command = gateway_command("telegram", "-100123")
    job = _create(cron_job("create", name="gw", command=command, schedule="every 2m"))
    assert _stored(job["id"])["command"] == command
    assert _stored(job["id"]).get("prompt", "") == ""


def test_gateway_described_as_prompt_job_is_refused() -> None:
    # The failure mode behind the "tick-OK" posts: an LLM session scheduled
    # to "run a gateway tick" posts a status line into the chat on every
    # run.  create and ensure both refuse it and point at gateway_command.
    prompt = (
        "Run one Slack gateway tick on #sorcar with pairing enabled and "
        "report whether any user messages are waiting."
    )
    for action in ("create", "ensure"):
        parsed = yaml.safe_load(cron_job(
            action, name="gw-prompt", prompt=prompt, schedule="every 5m",
            deliver="slack:C0AKYSNLB7W",
        ))
        assert "gateway_command" in parsed["error"], parsed
        assert "tick-OK" in parsed["error"]
    assert load_jobs() == []
    # Ordinary jobs about an API gateway (even its heartbeat) or a poll of a
    # chat are normal prompt jobs: all three word groups must be present.
    _create(cron_job(
        "create", name="api",
        prompt="Poll the API gateway health endpoint and alert me if it is down.",
        schedule="every 1h",
    ))
    _create(cron_job(
        "create", name="logs",
        prompt="Summarize heartbeat failures in the API gateway logs.",
        schedule="every 1h",
    ))
    _create(cron_job(
        "create", name="chat", prompt="Summarize the new messages in #general.",
        schedule="every 1h",
    ))
    assert {job["name"] for job in load_jobs()} == {"api", "logs", "chat"}


_CHATTY_GATEWAY_CLI = """#!/usr/bin/env python3
print("Processed 1 message(s).")
"""


def test_gateway_command_job_never_delivers_to_a_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A gateway tick that served messages (or failed) prints a line; with a
    # channel delivery target that line would be posted into the very chat
    # the gateway serves.  cron_job stores the job with deliver "none"
    # instead, says so, and the run is logged locally only.
    from kiss.tests.conftest import install_cli_script

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    install_cli_script(bin_dir / "kiss-fakechat", _CHATTY_GATEWAY_CLI)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])

    command = "kiss-fakechat --channel=C0AKYSNLB7W --pairing --quiet"
    parsed = yaml.safe_load(cron_job(
        "create", name="gw", command=command, schedule="every 1m",
        deliver="slack:C0AKYSNLB7W,local",
    ))
    assert "delivery set to none" in parsed["note"]
    job = parsed["created"]
    assert _stored(job["id"])["deliver"] == "none"
    # The absolute-path, $(...) and quoted-option forms real gateways use
    # are recognised too; a command that only mentions a launcher as data,
    # or has none, keeps its delivery targets.
    assert cron_agent._is_gateway_command(
        "/opt/kiss/.venv/bin/kiss-slack --workspace w --channel C0AKYSNLB7W --quiet"
    )
    assert cron_agent._is_gateway_command(
        '"$(ls -d "$HOME"/ext/kiss-sorcar-*/.venv/bin/kiss-slack | sort -V | tail -n1)" '
        "--channel=seamless-loop --pairing --quiet --allow-users=ksen"
    )
    assert cron_agent._is_gateway_command(
        '"$(ls -d "C:/Users/First Last"/ext/kiss-sorcar-*/.venv/Scripts/kiss-slack.exe '
        '| sort -V | tail -n1)" --channel=C0AKYSNLB7W --quiet'
    )
    assert cron_agent._is_gateway_command('kiss-slack "--channel=C0AKYSNLB7W" --quiet')
    assert not cron_agent._is_gateway_command("kiss-slack --channels C0AKYSNLB7W")
    assert not cron_agent._is_gateway_command("echo 'unbalanced")
    other = _create(cron_job(
        "create", name="poll", command="curl -s https://example.com/channel",
        schedule="every 1m", deliver="slack:C0AKYSNLB7W",
    ))
    assert _stored(other["id"])["deliver"] == "slack:C0AKYSNLB7W"
    reminder = _create(cron_job(
        "create", name="reminder",
        command="printf '%s\\n' 'Run kiss-slack --channel=C0AKYSNLB7W --quiet'",
        schedule="every 1m", deliver="slack:C0AKYSNLB7W",
    ))
    assert _stored(reminder["id"])["deliver"] == "slack:C0AKYSNLB7W"
    _set_job_fields(reminder["id"], enabled=False)

    # A gateway job stored by a version without this rule (channel delivery
    # on disk) is normalised on load, so it is listed with deliver none,
    # recognised as the duplicate of a fresh create, and found by ensure.
    _set_job_fields(job["id"], deliver="slack:C0AKYSNLB7W")
    assert _stored(job["id"])["deliver"] == "none"
    duplicate = yaml.safe_load(cron_job(
        "create", name="gw-again", command=command, schedule="every 1m",
        deliver="slack:C0AKYSNLB7W",
    ))
    assert duplicate["error"].startswith(f"duplicate: job {job['id']!r}")
    ensured = yaml.safe_load(cron_job(
        "ensure", name="gw", command=command, schedule="every 1m", deliver="none",
    ))
    assert ensured["exists"]["id"] == job["id"]
    assert ensured["exists"]["deliver"] == "none"

    _set_job_fields(job["id"], next_run_at=1.0)
    _set_job_fields(other["id"], enabled=False)
    assert tick(2.0) == 1
    stored = _stored(job["id"])
    assert stored["last_status"] == "ok"
    assert stored["last_summary"] == "Processed 1 message(s)."
    # With a slack target the run would have recorded a delivery note
    # (sent or error); none was attempted.
    assert stored["last_delivery"] == []
    log = (tmp_path / "cron" / "output" / f"{job['id']}.md").read_text(encoding="utf-8")
    assert "Processed 1 message(s)." in log
