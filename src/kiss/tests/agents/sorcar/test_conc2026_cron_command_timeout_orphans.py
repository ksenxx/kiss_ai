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
daemonizing, and — by the mark the run puts in the command's
environment — one whose shell has exited by then
(``cron_agent._kill_command_tree``).  A Stop of the run kills the
same tree, by the mark alone once ``Popen.communicate`` has reaped a
shell that had exited.

End-to-end: real shell commands spawn real background children (one
ordinary, one ``setsid``-detached) that record their pids; after the
timeout the whole tree must be dead.  ``monkeypatch`` is used only for
environment/config values (``KISS_HOME``, the timeout constant), never
for test doubles.

Branches of the kill not covered here, and why (unreachable without
test doubles): ``_kill_marked`` on a system without ``/proc`` (macOS,
BSD), its signal failing because the process died between the read of
its environment and the signal, and ``_kill_command_tree`` giving up on
a marked process that outlives its bound (one stuck in the kernel).
Its skip of a descendant reaped since the snapshot is a matter of
timing: taken in some runs, never forced.
"""

import contextlib
import os
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import IO, Any

import pytest

from kiss.agents.sorcar import cron_agent
from kiss.core.processes import pid_alive, start_process_group
from kiss.tests.conftest import IS_WINDOWS, posix_only
from kiss.tests.lock_injection import inject

# The pid the tests record and probe must be an OS pid.  Under Git bash on
# Windows ``$!`` is the MSYS-internal pid, which differs from the Windows
# pid; ``/proc/<pid>/winpid`` maps it.  Probing (or killing) an MSYS pid
# as if it were a Windows pid targets an unrelated process.
_BG_PID = "$(cat /proc/$!/winpid)" if IS_WINDOWS else "$!"
# The job timeout that must fire while the command sits in ``sleep 300``.
# Git bash on Windows needs well over a second to start and fork its
# subshells on a loaded machine, so 1 s expired before ``echo`` ran there.
_JOB_TIMEOUT = 5.0 if IS_WINDOWS else 1.0

# A pytest worker launched as an asynchronous shell job inherits SIG_IGN.
# Establish the interactive CLI precondition the interrupt test exercises explicitly.
_CLI_WITH_INTERRUPT_HANDLER = (
    "import signal\n"
    "from kiss.agents.sorcar import cron_agent\n"
    "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
    "cron_agent.main()\n"
)


def _wait_pid_dead(pid: int, timeout_s: float = 5.0) -> bool:
    """Poll until *pid* no longer exists (reaped or re-parented+dead)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_alive(pid) or _is_zombie(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid) or _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    """Whether *pid* is a zombie (dead but not yet reaped by init).

    Without ``/proc`` (Windows, macOS) nothing can be a zombie here and
    ``pid_alive`` alone decides; a missing ``/proc/<pid>`` where ``/proc``
    exists means the process is gone.
    """
    if not os.path.isdir("/proc"):
        return False
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
    late_write_delay = int(_JOB_TIMEOUT) + 1
    command = (
        f"(sleep {late_write_delay}; touch {pid_file.with_name('late-write').as_posix()}) & "
        f"echo {_BG_PID} > {pid_file.as_posix()}; sleep 300"
    )
    job = {"id": "t1", "command": command}
    start = time.monotonic()
    status, text = cron_agent._run_command_job(job, tmp_path, _JOB_TIMEOUT)
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
        time.sleep(late_write_delay + 0.2)
        assert not pid_file.with_name("late-write").exists(), (
            "the orphaned child kept running and wrote after the timeout"
        )
        assert not marker.exists()
    finally:
        # Never leak a sleeping tree out of the test run -- but only
        # signal a pid that is still alive: a dead pid may already have
        # been reused by an unrelated process.
        if pid_alive(child_pid):
            try:
                os.kill(child_pid, 9)
            except OSError:  # gone between the probe and the kill
                pass


def test_command_job_success_and_failure_paths(tmp_path: Path) -> None:
    """Normal command jobs behave exactly as before the fix."""
    ok_status, ok_text = cron_agent._run_command_job(
        {"id": "t2", "command": "echo hello"}, tmp_path, 5.0,
    )
    assert (ok_status, ok_text) == ("ok", "hello")

    silent_status, silent_text = cron_agent._run_command_job(
        {"id": "t3", "command": "true"}, tmp_path, 5.0,
    )
    assert (silent_status, silent_text) == ("silent", None)

    err_status, err_text = cron_agent._run_command_job(
        {"id": "t4", "command": "echo out; echo err >&2; exit 3"}, tmp_path, 5.0,
    )
    assert err_status == "error"
    assert err_text is not None
    assert "exited 3" in err_text
    assert "out" in err_text and "err" in err_text

    # A command that cannot even start (its directory is gone) raises to
    # the caller, which records the failure: there is nothing to kill.
    # So does one that is never spawned (a NUL byte in the command).
    with pytest.raises(OSError):
        cron_agent._run_command_job({"id": "t5", "command": "true"}, tmp_path / "gone", 5.0)
    with pytest.raises(ValueError):
        cron_agent._run_command_job({"id": "t6", "command": "true \0"}, tmp_path, 5.0)


def test_command_job_does_not_read_the_schedulers_stdin(tmp_path: Path) -> None:
    """A scheduled command gets an empty stdin, not the scheduler's own.

    Nobody answers an unattended command.  One that reads stdin (``cat``,
    ``ssh``, ``ffmpeg``) inherited the scheduler's: the terminal of a
    ``kiss-cron --daemon``, the pipe a launcher keeps to the daemon.  It
    sat there until the job's timeout and took input meant for the
    scheduler.  The job runs here in a child whose stdin is a pipe that
    stays open and silent until the job is over.
    """
    code = (
        "from pathlib import Path\n"
        "from kiss.agents.sorcar import cron_agent\n"
        "print(cron_agent._run_command_job("
        f"{{'id': 't-stdin', 'command': 'cat'}}, Path({str(tmp_path)!r}), 3.0))\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.wait(timeout=120) == 0
        assert child.stdout is not None
        assert child.stdout.read().strip().splitlines()[-1] == "('silent', None)"
    finally:
        child.kill()
        child.communicate()


@pytest.mark.parametrize(
    "launch",
    ["setsid sh -c {inner} </dev/null >/dev/null 2>&1 & sleep 300", "setsid sh -c {inner} &"],
    ids=["shell-waits", "shell-exited"],
)
def test_setsid_descendant_is_killed_on_timeout(tmp_path: Path, launch: str) -> None:
    """A ``setsid``-daemonizing descendant must not survive the timeout.

    ``killpg`` alone cannot reach a child that created its own session:
    the escapee here would prove survival by touching a marker file 2s
    after the 0.5s timeout.  The tree kill must find it through the
    ``/proc`` parent chain and kill its new group too.  The chain ends
    with the shell: one that starts the child and exits at once leaves
    it to init, and only the output pipes the child inherited keep the
    run open until the timeout.  That child ran on and wrote; the kill
    must find it by the mark the run put in its environment.
    """
    if os.name == "nt" or not Path("/usr/bin/setsid").exists() or not Path("/proc").exists():
        pytest.skip("POSIX setsid and /proc required")
    pid_file = tmp_path / "escaped.pid"
    marker = tmp_path / "escaped-wrote"
    inner = (
        f"echo $$ > {shlex.quote(str(pid_file))}; "
        f"sleep 2; touch {shlex.quote(str(marker))}; sleep 300"
    )
    command = launch.format(inner=shlex.quote(inner))
    status, text = cron_agent._run_command_job(
        {"id": "t-setsid", "command": command}, tmp_path, 0.5,
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
    monkeypatch.setattr(cron_agent, "COMMAND_TIMEOUT_SECONDS", _JOB_TIMEOUT)
    pid_file = tmp_path / "child.pid"
    job = {
        "id": "t-exec",
        "name": "timeout-job",
        "command": f"(sleep 300) & echo {_BG_PID} > {pid_file.as_posix()}; sleep 300",
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
    assert f"timed out after {_JOB_TIMEOUT:g}s" in stored["last_summary"]
    # The error was delivered to the job's local output log.
    log = tmp_path / "cron" / "output" / "t-exec.md"
    assert log.exists() and f"timed out after {_JOB_TIMEOUT:g}s" in log.read_text()
    # And the command's background child died with the tree.
    assert pid_file.exists(), "the command never ran"
    child_pid = int(pid_file.read_text().strip())
    try:
        assert _wait_pid_dead(child_pid)
    finally:
        if pid_alive(child_pid):
            try:
                os.kill(child_pid, 9)
            except OSError:  # gone between the probe and the kill
                pass


@posix_only(
    "the /proc parent-chain walk, sh subshells and os.killpg; Windows kills the "
    "tree through the Job Object (covered by the timeout tests above)",
)
def test_proc_descendants_sees_multi_level_tree() -> None:
    """The /proc walk finds grandchildren, not only direct children.

    A real shell spawns a nested subshell tree; every level must appear
    in the descendant set used by the tree kill.
    """
    if not Path("/proc").exists():
        pytest.skip("/proc required")
    mark = "KISS_CRON_RUN_multi_level_tree"
    proc = subprocess.Popen(
        "sh -c 'sh -c \"sleep 30\" & sleep 30' & sleep 30",
        shell=True,
        start_new_session=True,
        env={**os.environ, mark: "1"},
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if len(cron_agent._proc_descendants(proc.pid)) >= 3:
                break
            time.sleep(0.05)
        descendants = cron_agent._proc_descendants(proc.pid)
        assert len(descendants) >= 3, descendants
        cron_agent._kill_command_tree(proc, mark)
        for pid in descendants:
            assert _wait_pid_dead(pid), f"descendant {pid} survived"
    finally:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)


@posix_only("the shell's $$/$! are OS pids and the kill is a POSIX killpg")
@pytest.mark.parametrize(
    "tail",
    ["while true; do echo tick; sleep 0.05; done", "wait", "exit"],
    ids=["chatty", "quiet", "shell-exited"],
)
def test_interrupted_command_job_kills_tree(tmp_path: Path, tail: str) -> None:
    """A Stop landing in ``_run_command_job``'s wait must kill the command tree.

    ``cron_job("run_now")`` runs ``_execute_job`` on the cron agent's
    task thread, where the daemon's Stop arrives as the asynchronously
    injected ``KeyboardInterrupt`` (and ``kiss-cron --run`` gets Ctrl-C
    the same way).  Before the fix only a ``TimeoutExpired`` killed the
    tree: the interrupt propagated out of ``communicate`` and the
    command, in its own process group, ran on unattended.  The command
    here records the shell's and a background child's pids, then either
    prints a line every 50 ms, so ``communicate``'s selector loop keeps
    returning to Python bytecode where the interrupt can land, or sits
    quietly in ``wait``: a single ``communicate(timeout=300)`` then slept
    in its selector for the whole job timeout and the Stop stayed
    pending until then, so the wait must be taken in short slices.
    Or the shell exits at once and leaves the child the output pipes
    that keep the run open: ``communicate`` reaps that shell as the
    interrupt passes through it, and the tree of a reaped shell was left
    alone.  It is found by the run's mark, which needs ``/proc``.
    """
    if tail == "exit" and not Path("/proc").exists():
        pytest.skip("/proc required")
    pid_file = tmp_path / "pids"
    command = f"sleep 300 & echo $$ $! > {pid_file.as_posix()}; {tail}"
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = cron_agent._run_command_job(
                {"id": "t-int", "command": command}, tmp_path, 300.0,
            )
        except BaseException as exc:  # noqa: BLE001 — capture for assert
            outcome["exc"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    deadline = time.monotonic() + 5.0
    pids: list[int] = []
    while len(pids) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
        if pid_file.exists():
            pids = [int(p) for p in pid_file.read_text().split()]
    assert len(pids) == 2, "the command never ran"
    shell_pid, child_pid = pids
    try:
        while tail == "exit" and not _is_zombie(shell_pid):
            assert time.monotonic() < deadline, "the shell never exited"
            time.sleep(0.02)
        assert inject(worker) == 1
        worker.join(timeout=10)
        assert not worker.is_alive(), "the interrupt never ended the wait"
        assert isinstance(outcome.get("exc"), KeyboardInterrupt), outcome
        for pid in (shell_pid, child_pid):
            assert _wait_pid_dead(pid), (
                f"process {pid} of the interrupted cron command survived (orphaned tree)"
            )
    finally:
        for pid in (shell_pid, child_pid):
            if pid_alive(pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, 9)


@posix_only("the command's pid is probed and killed as an OS process group")
@pytest.mark.parametrize(
    ("where", "when", "timeout"),
    [
        (start_process_group, "return", 300.0),
        (subprocess.Popen._communicate, "call", 300.0),  # type: ignore[attr-defined]
        (cron_agent._kill_command_tree, "call", _JOB_TIMEOUT),
    ],
    ids=["as-the-spawn-returns", "as-the-wait-begins", "as-the-timeout-kill-begins"],
)
def test_a_stop_at_the_edges_of_the_wait_kills_the_command(
    tmp_path: Path, where: Any, when: str, timeout: float,
) -> None:
    """A Stop raised just before, at the start of or after the wait must end the command.

    The injected ``KeyboardInterrupt`` is raised at the check that ends
    a call, before the caller binds its result.  ``_run_command_job``
    spawned the command ahead of its ``try``, so an interrupt landing
    there skipped the handler that kills the tree: the command, in its
    own session, ran on with nobody to stop it, from a scratch directory
    ``_execute_job`` then removed.  The trace function raises the
    interrupt on the return of the spawn itself.  Or as the first
    ``communicate`` begins: the ``Popen`` is then marked as
    communicating before it has set that up, a later ``communicate``
    fails on it, and the reap, which called one, raised that failure in
    place of the interrupt, so the Stop was lost.  Or as the kill of a
    timed-out command begins: that kill and the reap take seconds in
    the timeout's handler, and an interrupt raised there was not caught
    by the handler that kills an interrupted command, its sibling, so
    whatever was not killed yet ran on.
    """
    spawned: list[int] = []

    def stop_there(frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is not where.__code__:
            return None
        if event == when:
            sys.settrace(None)
            spawned.append((frame.f_locals.get("proc") or frame.f_locals["self"]).pid)
            raise KeyboardInterrupt
        return stop_there

    sys.settrace(stop_there)
    try:
        with pytest.raises(KeyboardInterrupt):
            cron_agent._run_command_job(
                {"id": "t-spawn", "command": "sleep 300"}, tmp_path, timeout,
            )
    finally:
        sys.settrace(None)
    (pid,) = spawned
    try:
        assert _wait_pid_dead(pid), "the command the Stop landed around ran on unattended"
    finally:
        if pid_alive(pid):
            with contextlib.suppress(OSError):
                os.killpg(pid, 9)


def _read_to_eof(pipe: IO[str], buffer: list[str]) -> None:
    """The body of ``subprocess.Popen._readerthread``, which only Windows has."""
    buffer.append(pipe.read())
    pipe.close()


@posix_only("the reader threads Windows starts by itself are set up on a POSIX Popen")
def test_the_reap_leaves_a_pipe_to_the_thread_still_reading_it(tmp_path: Path) -> None:
    """The reap must not wait for a process the kill missed that holds an output pipe.

    On Windows ``Popen.communicate`` reads each pipe in a thread of its
    own (``stdout_thread``, ``stderr_thread``), blocked in ``read()``
    until every process that holds the pipe has exited, and a
    ``close()`` of the pipe waits for that read to return.  The reap
    closed both pipes, so with such a process left (nothing is killed
    there once ``communicate`` has reaped the shell as a Stop passes
    through it) the Stop hung until that process exited.  The threads
    are set up here the way ``Popen._communicate`` sets them up on
    Windows; the shell has exited and left a child holding stdout.
    Stderr's thread has not run yet, as when a Stop cut its start
    short: such a thread is not alive, and reads all the same.
    """
    pid_file = tmp_path / "pid"
    proc = subprocess.Popen(
        f"sleep 300 2>/dev/null & echo $! > {pid_file.as_posix()}; echo out",
        shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None and proc.stderr is not None
    stdout_fd, stderr_fd = proc.stdout.fileno(), proc.stderr.fileno()
    output: dict[str, list[str]] = {"stdout": [], "stderr": []}
    readers = {
        name: threading.Thread(target=_read_to_eof, args=(getattr(proc, name), buffer), daemon=True)
        for name, buffer in output.items()
    }
    for name, reader in readers.items():
        setattr(proc, f"{name}_thread", reader)
    readers["stdout"].start()
    straggler = 0
    try:
        # Only the shell held stderr: it is at its end once the shell has exited.
        assert select.select([stderr_fd], [], [], 10.0)[0], "the shell never exited"
        straggler = int(pid_file.read_text())
        # The shell's line no longer waits in the pipe: stdout's reader took
        # it, so it is inside the read that a close would wait for.
        deadline = time.monotonic() + 10.0
        while select.select([stdout_fd], [], [], 0)[0]:
            assert time.monotonic() < deadline, "stdout was never read"
            time.sleep(0.01)
        reap = threading.Thread(target=cron_agent._reap_killed_command, args=(proc,), daemon=True)
        reap.start()
        reap.join(cron_agent._REAP_SECONDS + 5.0)
        assert not reap.is_alive(), "the reap waits for the process that holds the pipe"
        assert pid_alive(straggler) and readers["stdout"].is_alive() and not proc.stdout.closed
        assert not proc.stderr.closed, "closed under a thread that is about to read it"
        assert proc.returncode == 0, "the shell was not reaped"
        # Once that process is gone the readers end, and the reap waits for them.
        readers["stderr"].start()
        os.kill(straggler, 9)
        cron_agent._reap_killed_command(proc)
        assert not any(reader.is_alive() for reader in readers.values())
        assert proc.stdout.closed and proc.stderr.closed
        assert output == {"stdout": ["out\n"], "stderr": [""]}
    finally:
        if straggler and pid_alive(straggler):
            with contextlib.suppress(OSError):
                os.kill(straggler, 9)
        proc.wait(timeout=10)


@posix_only("the CLI is interrupted with SIGINT and the shell's $$/$! are OS pids")
@pytest.mark.parametrize(
    "mode", [["--tick"], ["--daemon", "--interval", "1"]], ids=["tick", "daemon"],
)
def test_interrupted_cli_scheduler_kills_its_running_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: list[str],
) -> None:
    """Ctrl-C on ``kiss-cron --tick`` / ``--daemon`` must not leave its running commands behind.

    The jobs run in daemon threads, which the exiting process abandoned: a
    command, in its own process group (out of the terminal's SIGINT), ran on
    unattended, past the timeout whose watchdog died with the process (audit
    2026-11-10, S3 F3).  The job's timeout here is far away, so only the
    interrupt can end the command, and the CLI still dies of its SIGINT.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    pid_pipe = tmp_path / "pids"
    os.mkfifo(pid_pipe)
    # Read-write, so the pipe has a writer from the start: readable once the command wrote.
    pid_reader = os.open(pid_pipe, os.O_RDWR | os.O_NONBLOCK)
    cron_agent.save_jobs([{
        "id": "t-cli", "name": "cli", "deliver": "none", "enabled": True,
        "schedule": "every 1h", "next_run_at": time.time(), "timeout": 300,
        "command": f"sleep 300 & echo $$ $! > {pid_pipe.as_posix()}; wait",
    }])
    cli = subprocess.Popen(
        [sys.executable, "-c", _CLI_WITH_INTERRUPT_HANDLER, *mode],
        env={**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    pids: list[int] = []
    try:
        if select.select([pid_reader], [], [], 60.0)[0]:
            pids = [int(p) for p in os.read(pid_reader, 100).split()]
        assert len(pids) == 2, "the command never ran"
        cli.send_signal(signal.SIGINT)
        assert cli.wait(timeout=60) == -signal.SIGINT
        for pid in pids:
            assert _wait_pid_dead(pid), (
                f"process {pid} of the interrupted CLI's command outlived it (orphaned tree)"
            )
    finally:
        os.close(pid_reader)
        cli.kill()
        cli.wait(timeout=10)
        for pid in pids:
            if pid_alive(pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, 9)
