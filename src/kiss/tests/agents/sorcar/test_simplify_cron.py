# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the cron scheduler race and leak fixes.

- ``run_now`` registers its run in ``cron_agent._running`` like a tick
  does, so a tick skips the job meanwhile and a second ``run_now`` is
  refused instead of overlapping the run.
- ``stop_scheduler_thread`` joins the scheduler thread before it
  forgets the daemon endpoint, so a tick in flight finishes with the
  endpoint still recorded.
- A command job whose timed-out tree left a double-detached straggler
  holding the output pipes is still reaped (no zombie shell, no
  ``ResourceWarning``) once the bounded drain gives up.

Not covered: the ``finally`` that forgets a detached prompt job after
``kill_agent_job`` is only reached when the kill itself is interrupted
(a Stop landing during its bounded join), which needs an interrupt
injected into the thread — a test double — to reproduce.
"""

from __future__ import annotations

import os
import shlex
import threading
import time
import warnings
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import cron_job, load_jobs, save_jobs


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    with cron_agent._running_lock:
        cron_agent._running.clear()
    return tmp_path


def _wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), f"{path.name} never appeared"


def test_run_now_is_registered_and_refused_while_running(tmp_path: Path) -> None:
    """A run_now run blocks a tick and a second run_now of the same job."""
    started = tmp_path / "started"
    release = tmp_path / "release"
    runs = tmp_path / "runs.log"
    command = (
        f"echo run >> {shlex.quote(str(runs))}; touch {shlex.quote(str(started))}; "
        f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.05; done; echo done"
    )
    created = yaml.safe_load(
        cron_job(
            "create",
            name="slow",
            command=command,
            schedule="every 1h",
            deliver="local",
        )
    )
    job_id = created["created"]["id"]
    # Make the job due so a tick during the run would launch it again.
    jobs = load_jobs()
    jobs[0]["next_run_at"] = time.time() - 1
    save_jobs(jobs)

    results: list[str] = []
    runner = threading.Thread(target=lambda: results.append(cron_job("run_now", job_id=job_id)))
    runner.start()
    try:
        _wait_for(started)
        assert job_id in cron_agent.running_job_ids()
        refused = yaml.safe_load(cron_job("run_now", job_id=job_id))
        assert refused == {"error": f"job {job_id!r} is already running"}
        assert cron_agent.tick(wait=True) == 0
        assert runs.read_text().count("run") == 1
    finally:
        release.touch()
        runner.join(timeout=20)
    assert not runner.is_alive()
    assert yaml.safe_load(results[0])["ran"]["last_status"] == "ok"
    # The entry is gone, so the job can run again.
    assert job_id not in cron_agent.running_job_ids()
    assert cron_agent.tick(wait=True) == 1
    assert runs.read_text().count("run") == 2


def test_run_now_forgets_its_registration_when_the_run_raises(tmp_path: Path) -> None:
    """A run that fails is still unregistered, so the job is not stuck as running."""
    created = yaml.safe_load(
        cron_job(
            "create",
            name="x",
            command="echo hi",
            schedule="every 1h",
            deliver="local",
        )
    )
    job_id = created["created"]["id"]
    # A file where the runs directory belongs makes _execute_job fail
    # before it runs anything.
    cron_agent._runs_dir().write_text("")
    with pytest.raises(FileExistsError):
        cron_job("run_now", job_id=job_id)
    assert job_id not in cron_agent.running_job_ids()
    cron_agent._runs_dir().unlink()
    assert yaml.safe_load(cron_job("run_now", job_id=job_id))["ran"]["last_status"] == "ok"
    assert load_jobs()[0]["last_summary"] == "hi"


def test_run_now_from_a_dispatched_cron_session_registers_canonically(tmp_path: Path) -> None:
    """A run_now through the SEA's module copy is visible to the canonical tick.

    A ``run_agent(agent="cron")`` session gets its ``cron_job`` tool from a
    synthetic copy of the module (the way ``load_sea`` builds it), whose
    own ``_running`` dict the daemon's scheduler never reads; the copy
    must register in the canonical registry.
    """
    from kiss.agents.sorcar.sea_commands import load_sea

    loaded_cron_job = load_sea(Path(cron_agent.__file__)).tools([])[0]
    assert loaded_cron_job is not cron_job
    started = tmp_path / "started"
    release = tmp_path / "release"
    runs = tmp_path / "runs.log"
    command = (
        f"echo run >> {shlex.quote(str(runs))}; touch {shlex.quote(str(started))}; "
        f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.05; done"
    )
    created = yaml.safe_load(
        loaded_cron_job("create", name="slow", command=command, schedule="every 1h")
    )
    job_id = created["created"]["id"]
    jobs = load_jobs()
    jobs[0]["next_run_at"] = time.time() - 1
    save_jobs(jobs)
    runner = threading.Thread(target=loaded_cron_job, args=("run_now",), kwargs={"job_id": job_id})
    runner.start()
    try:
        _wait_for(started)
        assert job_id in cron_agent.running_job_ids()
        assert "already running" in cron_job("run_now", job_id=job_id)
        assert cron_agent.tick(wait=True) == 0
        assert runs.read_text().count("run") == 1
    finally:
        release.touch()
        runner.join(timeout=20)
    assert not runner.is_alive()
    assert job_id not in cron_agent.running_job_ids()


def test_malformed_stored_timeout_is_recorded_not_raised(tmp_path: Path) -> None:
    """A hand-edited store with a non-numeric timeout records the error and cleans up."""
    created = yaml.safe_load(
        cron_job(
            "create",
            name="x",
            command="echo hi",
            schedule="every 1h",
            deliver="local",
        )
    )
    jobs = load_jobs()
    jobs[0]["timeout"] = "1h"
    save_jobs(jobs)
    cron_agent._execute_job(load_jobs()[0])
    stored = load_jobs()[0]
    assert stored["id"] == created["created"]["id"]
    assert stored["last_status"] == "error"
    assert stored["last_summary"] == "ValueError: could not convert string to float: '1h'"
    assert not list(cron_agent._runs_dir().iterdir())


def test_stop_scheduler_thread_joins_the_loop_before_forgetting_the_endpoint(
    tmp_path: Path,
) -> None:
    """After stop returns the scheduler thread has exited and the endpoint is cleared."""
    endpoint = tmp_path / "sorcar-local.json"
    for _ in range(10):
        stop = cron_agent.start_scheduler_thread(interval=0.01, endpoint_file=str(endpoint))
        assert stop.thread.name == "kiss-cron-scheduler"
        assert cron_agent._daemon_endpoint_file == str(endpoint)
        cron_agent.stop_scheduler_thread(stop)
        assert not stop.thread.is_alive()
        assert cron_agent._daemon_endpoint_file is None


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as f:
            fields = f.read().rpartition(")")[2].split()
    except OSError:
        return False  # gone (reaped)
    return bool(fields) and fields[0] == "Z"


def test_timed_out_command_with_pipe_holding_straggler_is_reaped(tmp_path: Path) -> None:
    """A straggler that keeps the pipes open does not leave the shell unreaped.

    The command double-detaches a child (``( setsid sh -c ... & )``: the
    subshell exits at once, so the child re-parents away from the tree
    before the kill's ``/proc`` walk and escapes it, as documented for
    :func:`cron_agent._kill_command_tree`) that inherits the job's
    stdout/stderr pipes.  The bounded drain after the kill then times
    out; the runner must close its pipe ends and reap the killed shell
    instead of leaving it a zombie with a ``ResourceWarning``.
    """
    if os.name == "nt" or not Path("/usr/bin/setsid").exists() or not os.path.isdir("/proc"):
        pytest.skip("POSIX setsid and /proc required")
    shell_pid_file = tmp_path / "shell.pid"
    orphan_pid_file = tmp_path / "orphan.pid"
    inner = f"echo $$ > {shlex.quote(str(orphan_pid_file))}; exec sleep 300"
    command = (
        f"echo $$ > {shlex.quote(str(shell_pid_file))}; "
        f"( setsid sh -c {shlex.quote(inner)} & ); sleep 300"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.monotonic()
        status, text = cron_agent._run_command_job(
            {"id": "t-straggler", "command": command},
            tmp_path,
            0.5,
        )
        elapsed = time.monotonic() - start
    try:
        assert status == "error"
        assert text == "command timed out after 0.5s"
        assert elapsed < 30
        _wait_for(orphan_pid_file, timeout=1.0)
        assert not _is_zombie(int(shell_pid_file.read_text().strip()))
        assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
    finally:
        try:
            os.kill(int(orphan_pid_file.read_text().strip()), 9)
        except (OSError, ValueError):
            pass
