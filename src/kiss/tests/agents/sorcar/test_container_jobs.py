# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of the container background jobs (``ContainerJobs``).

A real ``DockerManager`` attached to a live ``python:3.11-slim`` container
runs every job script, so the tests cover what the agent sees in a trial:
``Bash(background=True)`` returns at once, ``bash_job(wait, until=...)``
returns as soon as the log matches, the job survives the deadline sweep
of a timed-out foreground command, ``kill`` stops the whole process
group, and the exit code of a finished job is reported.  Skipped without
a Docker daemon.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from kiss.agents.sorcar.container_jobs import JOBS_DIR, ContainerJobs
from kiss.agents.sorcar.docker_manager import DockerManager

IMAGE = "python:3.11-slim"


def _live_container() -> Any:
    """A running container of :data:`IMAGE`, or ``None`` without Docker."""
    import docker

    try:
        client = docker.from_env()
        client.ping()
    except Exception:
        return None
    return client.containers.run(IMAGE, "sleep infinity", detach=True)


def _ps(manager: DockerManager, needle: str) -> str:
    """The process table lines whose command line contains *needle* (``sleep NNN``).

    The listing shell's own command line holds the pattern, so the
    pattern's last character is bracketed (``sleep 30[0]`` matches
    ``sleep 300`` but not the pattern text itself).
    """
    pattern = f"{needle[:-1]}[{needle[-1]}]"
    return manager.Bash(
        f"for d in /proc/[0-9]*; do tr '\\0' ' ' < $d/cmdline 2>/dev/null | grep -E -- {pattern!r} "
        "&& echo; done; true", "ps", 30, 5000)


def test_background_jobs_in_a_container() -> None:
    """Start, wait-until, tail, timeout-survival, kill and exit codes, all in the container."""
    live = _live_container()
    if live is None:
        pytest.skip("Docker is not available")
    manager = DockerManager(f"container:{live.id}", workdir="/tmp")
    try:
        manager.open()
        jobs = ContainerJobs(manager.Bash)

        # Unknown ids and bad arguments are errors, not reports.
        assert jobs.bash_job("nope").startswith("Error: unknown job id 'nope'. Known jobs: none.")

        # A job that prints a marker, then keeps running: the wait ends at the marker.
        job_id, report = jobs.start("echo starting; sleep 1; echo READY; sleep 300; echo nev''er")
        assert job_id is not None and f"Started background job {job_id} (pid " in report
        assert f"log: {JOBS_DIR}/{job_id}.log" in report
        t0 = time.monotonic()
        waited = jobs.bash_job(job_id, "wait", timeout_seconds=60, until="^READY$")
        assert time.monotonic() - t0 < 30, waited
        assert "status: running" in waited and "READY" in waited and "never" not in waited
        assert waited.startswith(f"Job {job_id}: echo starting;")
        assert waited.rstrip().endswith("--- last 50 log lines ---\nstarting\nREADY")
        # The pattern is checked by the engine that runs it (grep -E), not by
        # Python's re: a Python-only construct is refused at once instead of
        # silently never matching for the whole timeout.
        assert jobs.bash_job(job_id, "wait", timeout_seconds=60, until="[").startswith(
            "Error: until is not a valid grep -E regex: grep: ")
        t0 = time.monotonic()
        python_only = jobs.bash_job(job_id, "wait", timeout_seconds=60, until="(?i)ready")
        assert time.monotonic() - t0 < 30, python_only
        assert python_only.startswith("Error: until is not a valid grep -E regex: grep: ")
        assert jobs.bash_job(job_id, "dance").startswith("Error: action must be one of")

        # The job outlives a timed-out foreground command: the deadline sweep
        # kills only the tagged tree, and the job's environment has no tag.
        timed_out = manager.Bash("sleep 30", "hang", 2, 5000)
        assert "timed out" in timed_out
        assert "sleep 300" in _ps(manager, "sleep 300")
        tail = jobs.bash_job(job_id, "tail", tail_lines=1)
        assert "status: running" in tail and tail.rstrip().endswith("READY")

        # kill stops the whole process group (the sleep child too) and marks the job.
        killed = jobs.bash_job(job_id, "kill")
        assert "status: exited killed" in killed
        deadline = time.monotonic() + 10
        while "sleep 300" in _ps(manager, "sleep 300"):
            assert time.monotonic() < deadline, "the job's sleep survived the kill"
            time.sleep(0.2)
        unknown = f"Error: unknown job id 'zzz'. Known jobs: {job_id}."
        assert jobs.bash_job("zzz").startswith(unknown)

        # A finished job reports its exit code and its output; waiting on it returns at once.
        job2, _ = jobs.start("echo out; echo err >&2; exit 3")
        assert job2 is not None
        t0 = time.monotonic()
        done = jobs.bash_job(job2, "wait", timeout_seconds=30)
        assert time.monotonic() - t0 < 15
        assert "status: exited 3" in done and "out" in done and "err" in done
        # A wait whose deadline passes reports the job still running.
        job3, _ = jobs.start("sleep 20; echo late")
        assert job3 is not None
        slow = jobs.bash_job(job3, "wait", timeout_seconds=2)
        assert "status: running" in slow and slow.rstrip().endswith("log lines ---")
        assert "status: exited killed" in jobs.bash_job(job3, "kill")
        # A fresh registry (a later session of the task) still addresses the job
        # through its files in the container; a malformed id never reaches a script.
        later = ContainerJobs(manager.Bash)
        assert later.bash_job("x; rm -rf /").startswith("Error: unknown job id 'x; rm -rf /'")
        assert later.bash_job("deadbeef").startswith("Error: unknown job id 'deadbeef'")
        assert later.bash_job(job2).startswith(f"Job {job2}: echo out; echo err >&2; exit 3\n")
        assert "status: exited 3" in later.bash_job(job2)
        # A job that cannot start (a file sits where the jobs dir goes) is an error, not a job.
        try:
            manager.Bash(f"mv {JOBS_DIR} {JOBS_DIR}.bak && touch {JOBS_DIR}", "block", 30, 500)
            none, error = jobs.start("echo x")
            assert none is None and error.startswith("Error: could not start the background job")
        finally:
            manager.Bash(f"rm -f {JOBS_DIR}; mv {JOBS_DIR}.bak {JOBS_DIR}", "unblock", 30, 500)
    finally:
        manager.close()
        live.remove(force=True)


def test_sorcar_docker_bash_offers_background_jobs() -> None:
    """In Docker mode ``Bash(background=True)``, ``on_timeout="background"`` and ``bash_job``
    work."""
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    live = _live_container()
    if live is None:
        pytest.skip("Docker is not available")
    agent = SorcarAgent("docker-jobs")
    agent._use_web_tools = False
    agent.docker_manager = DockerManager(f"container:{live.id}", workdir="/tmp")
    try:
        agent.docker_manager.open()
        tools = {t.__name__: t for t in agent._get_tools()}
        names = [t.__name__ for t in agent._get_tools()]
        assert names.index("Bash") + 1 == names.index("bash_job")
        assert "background" in tools["Bash"].__code__.co_varnames
        started = tools["Bash"]("echo one; sleep 30", "bg", background=True)
        assert started.startswith("Started background job ")
        job_id = started.split()[3]
        report = tools["bash_job"](job_id, action="wait", timeout_seconds=20, until="one")
        assert "status: running" in report and report.rstrip().endswith("one")
        assert "status: exited killed" in tools["bash_job"](job_id, action="kill")
        # on_timeout="background": the command keeps running past the deadline as a job
        slow = tools["Bash"]("echo early; sleep 20; echo late", "slow", timeout_seconds=2,
                             on_timeout="background")
        assert "status: running" in slow and "early" in slow and slow.startswith("Job ")
        later_id = slow.split()[1].rstrip(":")
        # the registry outlives a rebuilt tool list
        rebuilt = {t.__name__: t for t in agent._get_tools()}
        assert "status: running" in rebuilt["bash_job"](later_id)
        assert "status: exited killed" in rebuilt["bash_job"](later_id, action="kill")
        # a plain foreground command is unchanged
        assert tools["Bash"]("echo fg", "fg").strip() == "fg"
        # a command the guards refuse cannot start as a job either
        assert tools["Bash"]("kill -9 -1", "bad", background=True).startswith(
            "Error: could not start the background job")
    finally:
        agent.docker_manager.close()
        live.remove(force=True)
