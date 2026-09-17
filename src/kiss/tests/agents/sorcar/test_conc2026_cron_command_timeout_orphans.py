# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A timed-out cron command job must not orphan its process tree.

``cron_agent._run_command_job`` runs the job's shell command with
``shell=True``.  Before the fix it used ``subprocess.run(...,
timeout=...)``, which on timeout kills ONLY the shell process:
every child the command spawned (a build, a watcher, a background
``&`` job) survived the timeout, kept running — and writing — for
ever, and a repeating schedule spawned a fresh orphan tree on every
tick.  The fix launches the command in its own session and, on
timeout, kills the whole process TREE: the command's process group
plus — via a ``/proc`` walk snapshotted before the group kill — every
surviving descendant that escaped the group by calling ``setsid`` or
daemonizing (``cron_agent._kill_command_tree``).

End-to-end: real shell commands spawn real background children (one
ordinary, one ``setsid``-detached) that record their pids; after the
timeout the whole tree must be dead.  ``monkeypatch`` is used only for
environment/config values (``KISS_HOME``, the timeout constant), never
for test doubles.
"""

import os
import shlex
import subprocess
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar._concurrency import pid_alive


def _wait_pid_dead(pid: int, timeout_s: float = 5.0) -> bool:
    """Poll until *pid* no longer exists (reaped or re-parented+dead)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_alive(pid) or _is_zombie(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid) or _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    """Whether *pid* is a zombie (dead but not yet reaped by init)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as f:
            fields = f.read().rpartition(")")[2].split()
    except OSError:
        return True
    return bool(fields) and fields[0] == "Z"


def test_timed_out_command_job_kills_descendants(tmp_path: Path) -> None:
    """The background child of a timed-out command must be killed."""
    pid_file = tmp_path / "child.pid"
    marker = tmp_path / "still-alive"
    # The command backgrounds a long sleep (recording its pid), then
    # blocks in a foreground sleep so the job genuinely times out.
    # The background child would prove it survived by touching the
    # marker after the timeout window.
    command = (
        f"(sleep 2; touch {pid_file.with_name('late-write')}) & "
        f"echo $! > {pid_file}; sleep 300"
    )
    job = {"id": "t1", "command": command}
    start = time.monotonic()
    status, text = cron_agent._run_command_job(job, timeout_seconds=1.0)
    elapsed = time.monotonic() - start
    assert status == "error"
    assert text is not None and "timed out" in text
    # The call itself must return promptly (no hang draining pipes).
    assert elapsed < 30
    # The recorded background child must be dead shortly after.
    deadline = time.monotonic() + 5.0
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_file.exists(), "the command never ran"
    child_pid = int(pid_file.read_text().strip())
    try:
        assert _wait_pid_dead(child_pid), (
            f"background child {child_pid} of the timed-out cron command "
            f"survived the timeout (orphaned process tree)"
        )
        # And it must not have kept acting after the timeout.
        time.sleep(2.2)
        assert not pid_file.with_name("late-write").exists(), (
            "the orphaned child kept running and wrote after the timeout"
        )
        assert not marker.exists()
    finally:
        # Never leak a sleeping tree out of the test run.
        for pid in (child_pid,):
            try:
                os.kill(pid, 9)
            except (ProcessLookupError, PermissionError):
                pass


def test_command_job_success_and_failure_paths(tmp_path: Path) -> None:
    """Normal command jobs behave exactly as before the fix."""
    ok_status, ok_text = cron_agent._run_command_job(
        {"id": "t2", "command": "echo hello"},
    )
    assert (ok_status, ok_text) == ("ok", "hello")

    silent_status, silent_text = cron_agent._run_command_job(
        {"id": "t3", "command": "true"},
    )
    assert (silent_status, silent_text) == ("silent", None)

    err_status, err_text = cron_agent._run_command_job(
        {"id": "t4", "command": "echo out; echo err >&2; exit 3"},
    )
    assert err_status == "error"
    assert err_text is not None
    assert "exited 3" in err_text
    assert "out" in err_text and "err" in err_text


def test_setsid_descendant_is_killed_on_timeout(tmp_path: Path) -> None:
    """A ``setsid``-daemonizing descendant must not survive the timeout.

    ``killpg`` alone cannot reach a child that created its own session:
    the escapee here would prove survival by touching a marker file 2s
    after the 0.5s timeout.  The tree kill must find it through the
    ``/proc`` parent chain and kill its new group too.
    """
    if os.name == "nt" or not Path("/usr/bin/setsid").exists():
        pytest.skip("POSIX setsid required")
    pid_file = tmp_path / "escaped.pid"
    marker = tmp_path / "escaped-wrote"
    inner = (
        f"echo $$ > {shlex.quote(str(pid_file))}; "
        f"sleep 2; touch {shlex.quote(str(marker))}; sleep 300"
    )
    command = (
        f"setsid sh -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 & "
        "sleep 300"
    )
    status, text = cron_agent._run_command_job(
        {"id": "t-setsid", "command": command}, timeout_seconds=0.5,
    )
    assert status == "error"
    assert text is not None and "timed out" in text
    deadline = time.monotonic() + 5.0
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_file.exists(), "the command never ran"
    escaped_pid = int(pid_file.read_text().strip())
    try:
        assert _wait_pid_dead(escaped_pid), (
            f"setsid descendant {escaped_pid} of the timed-out cron "
            f"command survived the process-tree kill"
        )
        time.sleep(2.2)
        assert not marker.exists(), (
            "the setsid descendant kept running and wrote after the timeout"
        )
    finally:
        # Never leak a detached sleeping tree out of the test run.
        for kill in (os.killpg, os.kill):
            try:
                kill(escaped_pid, 9)
            except (ProcessLookupError, PermissionError):
                pass


def test_timed_out_command_reports_error_via_execute_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_execute_job`` records a timed-out command's error in the store.

    Drives the REAL production path — ``_execute_job`` →
    ``_run_command_job`` with the module timeout constant → ``_deliver``
    (local log) → persisted ``last_status``/``last_summary`` — against
    an isolated ``KISS_HOME`` job store, with the constant shortened so
    the command genuinely times out and its background child dies.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "COMMAND_TIMEOUT_SECONDS", 1.0)
    pid_file = tmp_path / "child.pid"
    job = {
        "id": "t-exec",
        "name": "timeout-job",
        "command": f"(sleep 300) & echo $! > {pid_file}; sleep 300",
        "deliver": "local",
        "enabled": True,
        "schedule": "every 1h",
        "next_run_at": 0.0,
    }
    cron_agent.save_jobs([job])
    start = time.monotonic()
    cron_agent._execute_job(dict(job))
    assert time.monotonic() - start < 30
    stored = {j["id"]: j for j in cron_agent.load_jobs()}["t-exec"]
    assert stored["last_status"] == "error"
    assert "timed out after 1s" in stored["last_summary"]
    # The error was delivered to the job's local output log.
    log = tmp_path / "cron" / "output" / "t-exec.md"
    assert log.exists() and "timed out after 1s" in log.read_text()
    # And the command's background child died with the tree.
    assert pid_file.exists(), "the command never ran"
    child_pid = int(pid_file.read_text().strip())
    try:
        assert _wait_pid_dead(child_pid)
    finally:
        try:
            os.kill(child_pid, 9)
        except (ProcessLookupError, PermissionError):
            pass


def test_proc_descendants_sees_multi_level_tree() -> None:
    """The /proc walk finds grandchildren, not only direct children.

    A real shell spawns a nested subshell tree; every level must appear
    in the descendant set used by the tree kill.
    """
    if not Path("/proc").exists():
        pytest.skip("/proc required")
    proc = subprocess.Popen(
        "sh -c 'sh -c \"sleep 30\" & sleep 30' & sleep 30",
        shell=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if len(cron_agent._proc_descendants(proc.pid)) >= 3:
                break
            time.sleep(0.05)
        descendants = cron_agent._proc_descendants(proc.pid)
        assert len(descendants) >= 3, descendants
        cron_agent._kill_command_tree(proc)
        for pid in descendants:
            assert _wait_pid_dead(pid), f"descendant {pid} survived"
    finally:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)
