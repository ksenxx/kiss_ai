# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The wait primitive of the Bash tools (P9 of the 2026-10-24 trajectory audit).

``bash_job(action="wait", until=<regex>)`` returns as soon as the job's
log matches; a foreground ``sleep N`` over 30 s is refused;
``Bash(on_timeout="background")`` leaves a command that outlives its
deadline running as a job; ``Bash(background=True, machine=...)`` needs
a registered SSH machine.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools, sleep_guard


def _job_id(started: str) -> str:
    assert started.startswith("Started background job "), started
    return started.split()[3]


def test_wait_until_returns_when_the_log_matches(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    job = _job_id(
        tools.Bash("echo starting; sleep 0.5; echo READY now; sleep 20", "job", background=True)
    )
    before = time.monotonic()
    report = tools.bash_job(job, action="wait", until=r"READY \w+", timeout_seconds=15)
    assert time.monotonic() - before < 10
    assert "running (pid" in report and "READY now" in report
    assert tools.bash_job(job, action="wait", until="[", timeout_seconds=1).startswith(
        "Error: until is not a valid regular expression"
    )
    tools.bash_job(job, action="kill")


def test_wait_until_still_ends_at_exit_or_deadline(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    job = _job_id(tools.Bash("echo done-quickly", "job", background=True))
    report = tools.bash_job(job, action="wait", until="never-printed", timeout_seconds=10)
    assert "exited with code 0" in report
    slow = _job_id(tools.Bash("sleep 20", "slow", background=True))
    before = time.monotonic()
    report = tools.bash_job(slow, action="wait", until="never", timeout_seconds=0.5)
    assert time.monotonic() - before < 5 and "running (pid" in report
    tools.bash_job(slow, action="kill")


def test_long_foreground_sleep_is_refused(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    out = tools.Bash("sleep 295; echo x", "poll")
    assert out.startswith("REFUSED: the Bash call did NOT run: a foreground sleep of 295s")
    assert 'until="<regex' in out
    assert tools.Bash("sleep 0.1 && echo short", "short") == "short\n"
    assert sleep_guard("echo sleep 100") is None
    assert sleep_guard("make && sleep 60") is not None
    assert sleep_guard("sleep 30") is None
    assert sleep_guard("sleep 30.5") is not None


def test_on_timeout_background_keeps_the_command_running(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    quick = tools.Bash("echo fast; exit 2", "quick", timeout_seconds=5, on_timeout="background")
    assert quick.startswith("Error (exit code 2):\nfast")
    out = tools.Bash(
        "echo begun; sleep 20; echo late", "slow", timeout_seconds=0.5, on_timeout="background",
    )
    assert out.startswith("Still running after 0.5s; left running as background job ")
    job = out.split("background job ")[1].split()[0]
    assert "begun" in out and job in tools._jobs
    assert tools._jobs[job].process.poll() is None
    tools.bash_job(job, action="kill")
    assert tools.Bash("echo x", "bad", on_timeout="later").startswith("Error: on_timeout must be")


def test_machine_requires_background_and_a_registered_ssh_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    tools = UsefulTools(work_dir=str(tmp_path))
    assert tools.Bash("echo x", "fg", machine="bench") == (
        "Error: machine= applies to background=True only."
    )
    out = tools.Bash("echo x", "bg", background=True, machine="bench")
    assert out.startswith("Error: no registered machine named 'bench'")
