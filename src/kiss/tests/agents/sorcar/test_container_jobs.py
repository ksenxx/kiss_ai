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

import shlex
import time
from typing import Any

import pytest

from kiss.agents.sorcar.container_jobs import (
    GONE_STATUS,
    JOB_ID,
    JOBS_DIR,
    KILL_ALL_SCRIPT,
    LOST_STATUS,
    ContainerJobs,
)
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
    # ``2>`` before ``<``: a process gone since the glob listed it is no line of the answer.
    return manager.Bash(
        f"for d in /proc/[0-9]*; do tr '\\0' ' ' 2>/dev/null < $d/cmdline | grep -E -- {pattern!r} "
        "&& echo; done; true", "ps", 30, 5000)


def _lose(manager: DockerManager, jobs: ContainerJobs, seconds: int) -> str:
    """Start ``sleep <seconds>`` as a job and kill every process naming it, as ``pkill -f`` would.

    The job's shell is among them (its command line holds the command), so no exit file is
    written.  The image has no ``pkill``: the processes are found by their command lines,
    the pattern's last character bracketed so the killing shell is not one of them.
    """
    job_id, report = jobs.start(f"sleep {seconds}")
    assert job_id is not None, report
    manager.Bash(
        f"for d in /proc/[0-9]*; do tr '\\0' ' ' < $d/cmdline 2>/dev/null | grep -qE -- "
        f"'sleep {str(seconds)[:-1]}[{str(seconds)[-1]}]' && kill -KILL ${{d#/proc/}}; done; true",
        "pkill -f", 30, 500)
    assert _ps(manager, f"sleep {seconds}").strip() == ""
    return job_id


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
        # Only a wait reads the pattern: a tail with one is a tail, not a check of it.
        assert "status: running" in jobs.bash_job(job_id, "tail", until="[")
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
        # The job's shell publishes its exit code by opening the marker, then
        # writing it: an empty marker (the open landed, the status not yet) is
        # not a completed job, for the registry, a tail and a wait alike.
        manager.Bash(f": > {JOBS_DIR}/{job3}.exit", "a marker not written yet", 30, 500)
        assert jobs.last_running() == job3
        assert "status: running" in jobs.bash_job(job3, "tail")
        t0 = time.monotonic()
        assert "status: running" in jobs.bash_job(job3, "wait", timeout_seconds=1)
        assert time.monotonic() - t0 >= 1
        assert "status: exited killed" in jobs.bash_job(job3, "kill")
        # Audit 2026-11-10 (C13-F5): a job stopped from outside (``pkill -f`` and its command:
        # the job's shell dies with it and writes no exit file) is not running for ever.  The
        # registry, a tail and a wait each settle it themselves, a job per script here.
        lost = _lose(manager, jobs, 297)
        assert jobs.last_running() is None
        lost_status = f"status: exited {LOST_STATUS}"
        assert lost_status in jobs.bash_job(lost) and lost_status in jobs.bash_job(_lose(
            manager, jobs, 296))
        t0 = time.monotonic()
        assert jobs.wait(_lose(manager, jobs, 295), 30) == f"exited {LOST_STATUS}"
        assert time.monotonic() - t0 < 15
        # So is one whose recorded pid a stranger holds now: it is settled, not signalled.
        recycled = _lose(manager, jobs, 293)
        manager.Bash(f"echo 1 > {JOBS_DIR}/{recycled}.pid", "recycle the pid", 30, 500)
        assert lost_status in jobs.bash_job(recycled)
        # A job whose shell alone was killed (a zombie here: pid 1 reaps nothing) still has
        # its child in the process group: running as before, until bash_job kills it.
        orphaned, _ = jobs.start("sleep 294")
        assert orphaned is not None
        manager.Bash(f"kill -KILL $(cat {JOBS_DIR}/{orphaned}.pid)", "the shell only", 30, 500)
        assert "status: running" in jobs.bash_job(orphaned) and jobs.last_running() == orphaned
        assert "sleep 294" in _ps(manager, "sleep 294")
        assert "status: exited killed" in jobs.bash_job(orphaned, "kill")
        # A job whose shell ended leaves a child running in its process group:
        # the sweep before a checkpoint restore still ends it.
        job4, _ = jobs.start("sleep 298 > /dev/null 2>&1 < /dev/null & exit 0")
        assert job4 is not None
        assert "status: exited 0" in jobs.bash_job(job4, "wait", timeout_seconds=30)
        assert "sleep 298" in _ps(manager, "sleep 298")
        # A recycled pid: the finished job's record names a pid that now belongs
        # to an unrelated session.  Neither kill signals it.
        other = manager.Bash(
            "setsid sleep 299 > /dev/null 2>&1 < /dev/null & echo $!", "unrelated", 30, 500,
        ).strip().splitlines()[-1]
        assert other.isdigit() and "sleep 299" in _ps(manager, "sleep 299")
        assert "sleep 299" in manager.Bash(
            f"tr '\\0' ' ' < /proc/{other}/cmdline", "the session leader", 30, 500,
        )
        manager.Bash(f"echo {other} > {JOBS_DIR}/{job2}.pid", "recycle the pid", 30, 500)
        assert "status: exited 3" in jobs.bash_job(job2, "kill")
        assert "sleep 299" in _ps(manager, "sleep 299")
        swept = manager.Bash(KILL_ALL_SCRIPT, "kill every job", 30, 5000).split()
        assert job2 not in swept and job4 in swept, swept
        assert "sleep 299" in _ps(manager, "sleep 299")
        deadline = time.monotonic() + 10
        while "sleep 298" in _ps(manager, "sleep 298"):
            assert time.monotonic() < deadline, "the finished job's child survived the sweep"
            time.sleep(0.2)
        manager.Bash(f"kill -KILL {other}", "end the unrelated session", 30, 500)
        # A recorded pid with no /proc entry at all: neither kill prints the
        # shell's complaint about it (the sweep's caller reads job ids).
        manager.Bash(f"echo 999999 > {JOBS_DIR}/{job2}.pid", "a pid nobody has", 30, 500)
        assert "No such file" not in jobs.bash_job(job2, "kill")
        swept = manager.Bash(KILL_ALL_SCRIPT, "kill every job", 30, 5000).split()
        assert all(JOB_ID.fullmatch(word) for word in swept), swept
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


def _pid_of(manager: DockerManager, cmdline: str) -> int:
    """The pid of the one process whose command line is exactly *cmdline*."""
    pattern = f"^{cmdline[:-1]}[{cmdline[-1]}] $"
    out = manager.Bash(
        f"for d in /proc/[0-9]*; do tr '\\0' ' ' < $d/cmdline 2>/dev/null | grep -qE -- "
        f"{pattern!r} && echo ${{d#/proc/}}; done; true", "pid", 30, 500)
    pids = [int(line) for line in out.split() if line.isdigit()]
    assert len(pids) == 1, out
    return pids[0]


def test_adopted_process_is_a_job() -> None:
    """A ``nohup … &`` survivor adopted as a job is found, tailed, waited on and killed like one."""
    live = _live_container()
    if live is None:
        pytest.skip("Docker is not available")
    manager = DockerManager(f"container:{live.id}", workdir="/tmp")
    try:
        manager.open()
        jobs = ContainerJobs(manager.Bash)
        assert jobs.last_running() is None
        manager.Bash("nohup sh -c 'echo begun; exec sleep 30' > /tmp/x.log 2>&1 &", "leave", 30,
                     500)
        pid = _pid_of(manager, "sleep 30")
        job_id, report = jobs.adopt(pid, "sleep 30", "/tmp/x.log")
        assert job_id is not None
        assert report == f"left running as job {job_id} (sleep 30, log /tmp/x.log)"
        assert jobs.last_running() == job_id
        # The job's log is the file the redirection named: a tail shows it and a
        # wait-until matches it at once; a wait on the running process times out.
        tail = jobs.bash_job(job_id, "tail", tail_lines=1)
        assert tail.startswith(
            f"Job {job_id}: sleep 30\nstatus: running\nlog: {JOBS_DIR}/{job_id}.log")
        assert tail.rstrip().endswith("begun")
        t0 = time.monotonic()
        matched = jobs.bash_job(job_id, "wait", timeout_seconds=30, until="^begun$")
        assert "status: running" in matched and time.monotonic() - t0 < 15
        assert "status: running" in jobs.bash_job(job_id, "wait", timeout_seconds=1)
        # A survivor without a redirection gets an empty log; its exit is seen by
        # the watcher within seconds of the process ending, and its code is unknown.
        manager.Bash("nohup sleep 3 > /dev/null 2>&1 &", "leave another", 30, 500)
        short_id, report = jobs.adopt(_pid_of(manager, "sleep 3"), "sleep 3", "")
        assert short_id is not None
        assert report == f"left running as job {short_id} (sleep 3, no log)"
        assert jobs.last_running() == short_id
        t0 = time.monotonic()
        done = jobs.bash_job(short_id, "wait", timeout_seconds=30)
        assert time.monotonic() - t0 < 20, done
        assert "status: exited (exit code unknown: adopted process)" in done
        assert done.rstrip().endswith("--- last 50 log lines ---")
        assert jobs.last_running() == job_id
        # kill stops an adopted pid (no process group of its own) and marks the job killed.
        assert "status: exited killed" in jobs.bash_job(job_id, "kill")
        deadline = time.monotonic() + 10
        while "sleep 30" in _ps(manager, "sleep 30"):
            assert time.monotonic() < deadline, "the adopted sleep survived the kill"
            time.sleep(0.2)
        assert jobs.last_running() is None
        # A later session addresses the adopted job through its files in the container.
        later = ContainerJobs(manager.Bash)
        assert later.bash_job(job_id).startswith(f"Job {job_id}: sleep 30\nstatus: exited killed")
        # A pid that is gone is not adopted; nor is anything while the jobs dir cannot be made.
        assert jobs.adopt(pid, "sleep 30", "/tmp/x.log") == (
            None, f"Error: process {pid} is not running")
        manager.Bash("nohup sleep 31 > /dev/null 2>&1 &", "leave a third", 30, 500)
        third = _pid_of(manager, "sleep 31")
        try:
            manager.Bash(f"mv {JOBS_DIR} {JOBS_DIR}.bak && touch {JOBS_DIR}", "block", 30, 500)
            none, error = jobs.adopt(third, "sleep 31", "")
            assert none is None and error.startswith(
                f"Error: process {third} could not be adopted: mkdir: ")
        finally:
            manager.Bash(f"rm -f {JOBS_DIR}; mv {JOBS_DIR}.bak {JOBS_DIR}", "unblock", 30, 500)
        # The end-of-task kill counts an adopted pid that leads no group among the killed
        # (the earlier jobs' pids count too: pid 1 of this container reaps no zombie).
        third_id, _ = jobs.adopt(third, "sleep 31", "")
        assert third_id is not None
        killed = manager.Bash(KILL_ALL_SCRIPT, "kill all", 30, 2000).split()
        assert third_id in killed
        deadline = time.monotonic() + 10
        while "sleep 31" in _ps(manager, "sleep 31"):
            assert time.monotonic() < deadline, "the adopted sleep survived the end-of-task kill"
            time.sleep(0.2)
        # A survivor that execs after adoption (its command line changes) and
        # spawns a child after adoption (it leads no process group, so no group
        # kill reaches the child) is stopped whole, child included, by the
        # individual kill and by the end-of-task kill alike, even when both
        # ignore TERM: an adopted job is its pid plus start time, not its
        # command line, and its descendants are found by their parent pids
        # and signalled by the same identity.
        survive = ("while [ ! -e $3 ]; do sleep 0.1; done\ntrap '' TERM\nsleep $1 &\n"
                   "exec sleep $2\n")
        manager.Bash(f"printf '%s' {shlex.quote(survive)} > /tmp/survive.sh", "script", 30, 500)
        parents: list[tuple[int, str]] = []
        for child, grown, flag in ((32, 33, "/tmp/go"), (34, 35, "/tmp/go2")):
            command = f"sh /tmp/survive.sh {child} {grown} {flag}"
            out = manager.Bash(f"nohup {command} > /dev/null 2>&1 & echo $!; "
                               "sed 's/.*) //' /proc/$!/stat | cut -d' ' -f3", "leave", 30, 500)
            pid, pgid = (int(line) for line in out.split()[-2:])
            assert pgid != pid, out
            adopted, _ = jobs.adopt(pid, command, "")
            assert adopted is not None
            parents.append((pid, adopted))
            manager.Bash(f"touch {flag}", "let it grow", 30, 500)
            deadline = time.monotonic() + 10
            while f"sleep {child}" not in _ps(manager, f"sleep {child}"):
                assert time.monotonic() < deadline, f"sleep {child} never started"
                time.sleep(0.2)
            assert _pid_of(manager, f"sleep {grown}") == pid
            # The exec is not an exit to the watcher: the job still runs.
            assert "status: running" in jobs.bash_job(adopted, "tail")
        assert "status: exited killed" in jobs.bash_job(parents[0][1], "kill")
        deadline = time.monotonic() + 10
        while "sleep 33" in _ps(manager, "sleep 33") or "sleep 32" in _ps(manager, "sleep 32"):
            assert time.monotonic() < deadline, "the exec'd parent or its child survived the kill"
            time.sleep(0.2)
        assert "sleep 35" in _ps(manager, "sleep 35") and "sleep 34" in _ps(manager, "sleep 34")
        killed = manager.Bash(KILL_ALL_SCRIPT, "kill all", 30, 2000).split()
        assert parents[1][1] in killed and parents[0][1] not in killed, killed
        deadline = time.monotonic() + 10
        while "sleep 35" in _ps(manager, "sleep 35") or "sleep 34" in _ps(manager, "sleep 34"):
            assert time.monotonic() < deadline, "the exec'd parent or its child survived the sweep"
            time.sleep(0.2)
        # The watcher and the kill compare the start time too: a pid that no
        # longer holds the process adopted (here: a tampered start time, and no
        # token on record: a stranger carries neither) is an ended job to the
        # watcher and nobody's to signal for the kill.
        manager.Bash("nohup sleep 36 > /dev/null 2>&1 &", "leave a stranger", 30, 500)
        stranger = _pid_of(manager, "sleep 36")
        stranger_id, _ = jobs.adopt(stranger, "sleep 36", "")
        assert stranger_id is not None
        manager.Bash(f"echo 0 > {JOBS_DIR}/{stranger_id}.owner; : > {JOBS_DIR}/{stranger_id}.token",
                     "another process now", 30, 500)
        deadline = time.monotonic() + 10
        while "status: running" in jobs.bash_job(stranger_id, "tail"):
            assert time.monotonic() < deadline, "the watcher took the stranger for the job"
            time.sleep(0.5)
        assert "status: exited (exit code unknown: adopted process)" in jobs.bash_job(
            stranger_id, "kill")
        assert "sleep 36" in _ps(manager, "sleep 36")
        manager.Bash(f"kill -KILL {stranger}", "end the stranger", 30, 500)
        # A survivor whose job files cannot be written is not adopted: no job
        # is registered or reported running, and the process is left as it was.
        manager.Bash("nohup sleep 37 > /dev/null 2>&1 &", "leave a fourth", 30, 500)
        fourth = _pid_of(manager, "sleep 37")
        known = dict(jobs.commands)
        try:
            manager.Bash(f"mv {JOBS_DIR} {JOBS_DIR}.bak && ln -s /proc/1 {JOBS_DIR}", "block", 30,
                         500)
            none, error = jobs.adopt(fourth, "sleep 37", "")
            assert none is None and error.startswith(
                f"Error: process {fourth} could not be adopted: ") and ".cmd" in error, error
            # the sentinel is printed once every file is in place, not before: the
            # failed registration's output is the shell's complaint alone
            assert "adopted" not in error.split("could not be adopted: ", 1)[1].splitlines()
        finally:
            manager.Bash(f"rm -f {JOBS_DIR}; mv {JOBS_DIR}.bak {JOBS_DIR}", "unblock", 30, 500)
        assert jobs.commands == known and jobs.last_running() is None
        assert "sleep 37" in _ps(manager, "sleep 37")
        # A job whose files are gone (the agent cleared /tmp) has nothing left to
        # record: its watcher ends, writing no exit file for the job that is no
        # more, instead of retrying a write into the missing jobs dir for ever.
        for clear in (f"rm -f {JOBS_DIR}/{{id}}.*", f"rm -rf {JOBS_DIR}"):
            gone_id, _ = jobs.adopt(fourth, "sleep 37", "")
            assert gone_id is not None
            watcher = f": {JOBS_DIR}/{gone_id}.exit"
            assert watcher in _ps(manager, watcher)
            manager.Bash(clear.format(id=gone_id), "clear the job's files", 30, 500)
            deadline = time.monotonic() + 15
            while _ps(manager, watcher).strip():
                assert time.monotonic() < deadline, f"the watcher outlived its job ({clear})"
                time.sleep(0.5)
            assert manager.Bash(f"ls {JOBS_DIR}/{gone_id}.exit 2>&1; true", "ls", 30,
                                500).strip().endswith("No such file or directory")
        assert "sleep 37" in _ps(manager, "sleep 37")
    finally:
        manager.close()
        live.remove(force=True)


def test_a_job_whose_files_are_gone_is_unknown_not_running() -> None:
    """Review of 2026-11-11 (S6, finding 2): a job whose files are gone (the agent cleared
    ``/tmp``) counted as running for ever, one that had exited 0 included: ``last_running``
    chose it and every wait on it ran to its deadline.  What became of it cannot be known:
    every action says so at once, the registry passes it over, and a kill, which has no pid
    to signal, marks nothing killed."""
    live = _live_container()
    if live is None:
        pytest.skip("Docker is not available")
    manager = DockerManager(f"container:{live.id}", workdir="/tmp")
    try:
        manager.open()
        jobs = ContainerJobs(manager.Bash)
        for seconds, clear in ((291, f"rm -f {JOBS_DIR}/*"), (293, f"rm -rf {JOBS_DIR}")):
            commands = ("exit 0", f"sleep {seconds}", f"sleep {seconds + 1}")
            ended, _ = jobs.start(commands[0])
            started, _ = jobs.start(commands[1])
            manager.Bash(f"nohup {commands[2]} > /dev/null 2>&1 &", "leave", 30, 500)
            adopted, _ = jobs.adopt(_pid_of(manager, commands[2]), commands[2], "")
            assert ended is not None and started is not None and adopted is not None
            assert "status: exited 0" in jobs.bash_job(ended, "wait", timeout_seconds=30)
            assert jobs.last_running() == adopted
            manager.Bash(clear, "clear the jobs' files", 30, 500)
            assert jobs.last_running() is None
            for job, command in zip((ended, started, adopted), commands, strict=True):
                gone = (f"Job {job}: {command}\nstatus: {GONE_STATUS}\nlog: {JOBS_DIR}/{job}.log\n"
                        "--- last 50 log lines ---")
                assert jobs.bash_job(job, "tail").rstrip() == gone
                t0 = time.monotonic()
                assert jobs.wait(job, 30) == GONE_STATUS
                assert time.monotonic() - t0 < 15
                assert jobs.bash_job(job, "kill").rstrip() == gone
            # Nothing was signalled and nothing written: the jobs' pids are not known any more.
            assert all(command in _ps(manager, command) for command in commands[1:])
            assert manager.Bash(f"ls {JOBS_DIR} 2>/dev/null; true", "ls", 30, 500).strip() == ""
    finally:
        manager.close()
        live.remove(force=True)


def _leader(manager: DockerManager, seconds: int) -> int:
    """Start a session leader that, once told to go, leaves a ``sleep <seconds>`` in its
    process group and exits (its waiting parent reaps it); return the leader's pid.  It
    carries no exec token: its group is the only handle on what it leaves."""
    script = (f"echo \\$\\$ > /tmp/lead{seconds}; while [ ! -e /tmp/go{seconds} ]; do sleep 0.1; "
              f"done; sleep {seconds} & exit 0")
    return int(manager.Bash(
        f"sh -c 'env -u KISS_EXEC_TOKEN setsid sh -c \"{script}\" & wait' "
        "> /dev/null 2>&1 < /dev/null & "
        f"for i in $(seq 50); do [ -s /tmp/lead{seconds} ] && break; sleep 0.1; done; "
        f"cat /tmp/lead{seconds}", "lead", 30, 500).split()[-1])


def test_the_group_of_an_adopted_leader_that_ended_is_still_killed() -> None:
    """Audit 2026-11-10 (B/S2-F4): an adopted process that led its own process group and
    ended leaves the group the job's: a child it created after its adoption is killed by
    ``bash_job`` and by the end-of-task sweep, and a stranger given the pid is not."""
    live = _live_container()
    if live is None:
        pytest.skip("Docker is not available")
    manager = DockerManager(f"container:{live.id}", workdir="/tmp")
    try:
        manager.open()
        jobs = ContainerJobs(manager.Bash)
        orphaned = {}
        for seconds in (281, 282):
            pid = _leader(manager, seconds)
            job, _ = jobs.adopt(pid, f"leader {seconds}", "")
            assert job is not None
            manager.Bash(f"touch /tmp/go{seconds}; for i in $(seq 50); do [ -e /proc/{pid} ] "
                         "|| break; sleep 0.1; done", "the leader ends", 30, 500)
            assert f"sleep {seconds}" in _ps(manager, f"sleep {seconds}")
            orphaned[seconds] = (job, pid)
        # a live process with another start time under the recorded pid is a stranger
        manager.Bash("setsid sleep 283 > /dev/null 2>&1 < /dev/null &", "a bystander", 30, 500)
        (job, pid), stranger = orphaned[281], _pid_of(manager, "sleep 283")
        manager.Bash(f"echo {stranger} > {JOBS_DIR}/{job}.pid", "pid", 30, 500)
        jobs.bash_job(job, "kill")
        assert "sleep 283" in _ps(manager, "sleep 283") and "sleep 281" in _ps(manager, "sleep 281")
        manager.Bash(f"echo {pid} > {JOBS_DIR}/{job}.pid", "pid", 30, 500)
        jobs.bash_job(job, "kill")
        assert _ps(manager, "sleep 281") == "" and "sleep 282" in _ps(manager, "sleep 282")
        manager.Bash(f"echo {stranger} > {JOBS_DIR}/{job}.pid", "pid", 30, 500)
        swept = manager.Bash(KILL_ALL_SCRIPT, "sweep", 30, 500).split()
        assert job not in swept and orphaned[282][0] in swept
        assert _ps(manager, "sleep 282") == "" and "sleep 283" in _ps(manager, "sleep 283")
    finally:
        manager.close()
        live.remove(force=True)


def _launcher(seconds: int) -> str:
    """A command leaving a shell that leads no process group and, once told to go, starts a
    ``sleep <seconds>`` and ends at once; the command prints the shell's pid."""
    script = f"while [ ! -e /tmp/go{seconds} ]; do sleep 0.1; done; sleep {seconds} & exit 0"
    return f"nohup sh -c {shlex.quote(script)} > /dev/null 2>&1 & echo $!"


def test_what_an_adopted_process_that_ended_left_running_is_still_killed() -> None:
    """An adopted process that led no process group and ended by itself is no parent to find
    its children by.  They carry the token of the command that left it: ``bash_job`` kills
    them when that command left no other job (one of two is not told from the other by it),
    and the end-of-task sweep whatever the jobs' commands left."""
    live = _live_container()
    if live is None:
        pytest.skip("Docker is not available")
    manager = DockerManager(f"container:{live.id}", workdir="/tmp")
    try:
        manager.open()
        jobs = ContainerJobs(manager.Bash)
        manager.Bash("nohup sleep 270 > /dev/null 2>&1 &", "a bystander", 30, 500)
        ended = {}
        kept: str | None = None
        for seconds, command in ((271, _launcher(271)), (272, _launcher(272)), (
                273, f"nohup sleep 274 > /dev/null 2>&1 & {_launcher(273)}")):
            pid = int(manager.Bash(command, "leave", 30, 500).split()[-1])
            job, _ = jobs.adopt(pid, f"launcher {seconds}", "")
            assert job is not None
            assert manager.Bash(f"ls {JOBS_DIR}/{job}.group 2>&1; true", "ls", 30,
                                500).strip().endswith("No such file or directory")
            if seconds == 273:
                kept, _ = jobs.adopt(_pid_of(manager, "sleep 274"), "sleep 274", "")
                assert kept is not None
            manager.Bash(f"touch /tmp/go{seconds}", "the launcher starts its child and ends", 30,
                         500)
            assert "status: exited (exit code unknown: adopted process)" in jobs.bash_job(
                job, "wait", timeout_seconds=30)
            assert f"sleep {seconds}" in _ps(manager, f"sleep {seconds}")
            ended[seconds] = job
        jobs.bash_job(ended[271], "kill")
        assert _ps(manager, "sleep 271") == ""
        jobs.bash_job(ended[273], "kill")
        assert all(f"sleep {n}" in _ps(manager, f"sleep {n}") for n in (270, 272, 273, 274))
        swept = manager.Bash(KILL_ALL_SCRIPT, "sweep", 30, 500).split()
        assert swept == [kept]
        assert all(_ps(manager, f"sleep {n}") == "" for n in (272, 273, 274))
        assert "sleep 270" in _ps(manager, "sleep 270")
    finally:
        manager.close()
        live.remove(force=True)


def test_sorcar_docker_bash_offers_background_jobs() -> None:
    """In Docker mode ``Bash(background=True)``, ``on_timeout="background"``, ``bash_job`` and
    ``Bash(until=)`` work, and the tool's schema shows the ``until`` arguments."""
    from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent
    from kiss.core.models.model_info import model

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
        # Bash(until=) reaches the container tool: the schema the model sees names
        # both arguments, a wait returns when the output matches, and until with
        # background is refused (a job's log is waited on with bash_job).
        schema = model(DEFAULT_DECISIONS_MODEL)._function_to_openai_tool(tools["Bash"])
        properties = schema["function"]["parameters"]["properties"]
        assert "re-run every" in properties["until"]["description"]
        assert properties["poll_seconds"]["type"] == "number"
        tools["Bash"]("(sleep 1; echo UP > /tmp/up) >/dev/null 2>&1 &", "raise a flag")
        waited = tools["Bash"]("cat /tmp/up 2>/dev/null; true", "poll", timeout_seconds=20,
                               until="^UP$", poll_seconds=1)
        assert waited.startswith("UP\n") and "[until '^UP$': matched on run " in waited
        assert tools["Bash"]("echo x", "both", background=True, until="x") == (
            "Error: until cannot be combined with background=True; start the job and wait on "
            "its log with bash_job(job_id, 'wait', until=...).")
    finally:
        agent.docker_manager.close()
        live.remove(force=True)
