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

import ctypes
import functools
import inspect
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar.useful_tools import (
    _SUPERVISE_LOG_MATCHER,
    UsefulTools,
    _log_matches,
    _run_log_matcher,
    _start_log_matcher,
    sleep_guard,
)
from kiss.core import tool_interrupt
from kiss.core.processes import kill_process_tree, popen_process_group


def _job_id(started: str) -> str:
    assert started.startswith("Started background job "), started
    return started.split()[3]


def _start_noted(log: Path, pattern: str, children: list[subprocess.Popen]) -> subprocess.Popen:
    """Start the matcher of *log* for *pattern*, noting the child in *children*."""
    children.append(_start_log_matcher(log, re.compile(pattern)))
    return children[-1]


def test_wait_until_returns_when_the_log_matches(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    job = _job_id(
        tools.Bash("echo starting; sleep 0.5; echo READY now; sleep 20", "job", background=True)
    )
    before = time.monotonic()
    report = tools.bash_job(job, action="wait", until=r"READY \w+", timeout_seconds=15)
    assert time.monotonic() - before < 10
    assert "running (pid" in report and "READY now" in report
    # "[" does not compile; a NUL or a lone surrogate cannot reach the matcher child as a
    # program argument (the child was never started and the wait ran to its deadline).
    for bad in ("[", "READY|\x00", "READY|\ud800"):
        assert tools.bash_job(job, action="wait", until=bad, timeout_seconds=1).startswith(
            "Error: until is not a valid regular expression"
        )
    # A launch the arguments themselves refuse is reported like one the system refuses.
    failed: list[str] = []
    start = functools.partial(_start_log_matcher, tmp_path / "x.log", re.compile("\x00"))
    _run_log_matcher(start, threading.Event(), [], time.monotonic() + 5, threading.Event(), failed)
    assert len(failed) == 1 and failed[0].startswith("embedded null ")
    # The owner of a matcher closes the child's stdin pipe (its lifeline) itself: left to the
    # collector, the daemon kept the pipe open for as long as anything referred to the child.
    children: list[subprocess.Popen] = []
    matched = threading.Event()
    (tmp_path / "x.log").write_text("READY\n")
    start = functools.partial(_start_noted, tmp_path / "x.log", "READY", children)
    _run_log_matcher(start, threading.Event(), [], time.monotonic() + 15, matched, [])
    assert matched.is_set() and children[0].stdin is not None and children[0].stdin.closed
    tools.bash_job(job, action="kill")


def test_wait_until_finds_a_match_that_straddles_two_scans(tmp_path: Path) -> None:
    """The log is scanned incrementally (once a second); a match split across scans is found."""
    tools = UsefulTools(work_dir=str(tmp_path))
    job = _job_id(
        tools.Bash("printf READY-; sleep 1.5; echo GO; sleep 20", "job", background=True)
    )
    before = time.monotonic()
    report = tools.bash_job(job, action="wait", until="READY-GO", timeout_seconds=15)
    assert time.monotonic() - before < 10
    assert "running (pid" in report and "READY-GO" in report
    tools.bash_job(job, action="kill")


def test_log_matches_searches_the_whole_log(tmp_path: Path) -> None:
    """Anchored and multi-line patterns keep their meaning across appends."""
    log = tmp_path / "job.log"
    log.write_bytes(b"START\n" + b"x" * 10_000)
    assert _log_matches(log, re.compile(r"START[\s\S]*END")) is False
    assert _log_matches(log, re.compile(r"^END", re.MULTILINE)) is False
    with log.open("ab") as appended:
        appended.write(b"\nEND\n")
    assert _log_matches(log, re.compile(r"START[\s\S]*END")) is True
    assert _log_matches(log, re.compile(r"^END", re.MULTILINE)) is True
    # Anchored to the start of the whole log, not of the appended tail.
    assert _log_matches(log, re.compile(r"\AEND")) is False
    assert _log_matches(tmp_path / "missing.log", re.compile("END")) is False


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


def _cpu_seconds(ps_time: str) -> float:
    """Return the seconds of a ``ps`` ``time`` field (``[[dd-]hh:]mm:ss`` or ``mm:ss.hh``)."""
    days, _, clock = ps_time.rpartition("-")
    seconds = 0.0
    for part in clock.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds + float(days or 0) * 86400


def _processes_naming(text: str) -> dict[int, float]:
    """Return ``{pid: CPU seconds}`` of the live processes whose command line holds *text*.

    ``ps`` (procps and BSD alike) rather than ``/proc``, which macOS
    lacks; Windows has no ``ps``, so there ``Win32_Process`` is listed
    with CPU seconds already computed.  A listing that fails raises
    instead of reporting no process.
    """
    if sys.platform == "win32":
        command = [
            "powershell", "-NoProfile", "-Command",
            # [double]: a process whose times are not readable reports $null;
            # one line per process: a ``-c`` program's newlines are joined.
            "Get-CimInstance Win32_Process | ForEach-Object { '{0} {1:F3} {2}' -f "
            "$_.ProcessId, [double](($_.UserModeTime + $_.KernelModeTime) / 1e7), "
            "($_.CommandLine -replace '[\\r\\n]+', ' ') }",
        ]
    else:
        command = ["ps", "-A", "-ww", "-o", "pid=,time=,args="]
    listing = subprocess.run(command, capture_output=True, text=True, check=True)
    found = {}
    for line in listing.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and text in fields[2]:
            found[int(fields[0])] = _cpu_seconds(fields[1])
    return found


def _kill_naming(log: Path) -> None:
    """Emergency cleanup: kill every live process naming *log* (``TerminateProcess`` on Windows)."""
    for pid in _processes_naming(str(log)):
        with suppress(OSError):
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


@pytest.mark.timeout(120)
def test_wait_until_with_a_catastrophic_pattern_still_ends_at_the_deadline(
    tmp_path: Path,
) -> None:
    """A backtracking ``until`` (2026-11-04 audit S5 F1) no longer holds the task thread.

    ``(a+)+$`` against 28 ``a`` and a ``!`` keeps ``re`` busy for tens of
    seconds (every two more ``a`` quadruple it); the wait must still end
    at its deadline, and at a Stop, with no matcher process left behind.
    """
    stop = threading.Event()
    tools = UsefulTools(stop_event=stop, work_dir=str(tmp_path))
    job = _job_id(tools.Bash("printf '%s\\n' '" + "a" * 28 + "!'; sleep 30", "job",
                             background=True))
    log = tools._jobs[job].log_path
    deadline = time.monotonic() + 10
    while "!" not in log.read_text() and time.monotonic() < deadline:
        time.sleep(0.05)
    before = time.monotonic()
    report = tools.bash_job(job, action="wait", until=r"(a+)+$", timeout_seconds=1)
    assert time.monotonic() - before < 8, "the regex held the wait past its deadline"
    assert "running (pid" in report
    assert _processes_naming(str(log)) == {}, "the matcher outlived the wait"
    threading.Timer(0.5, stop.set).start()
    before = time.monotonic()
    tools.bash_job(job, action="wait", until=r"(a+)+$", timeout_seconds=30)
    assert time.monotonic() - before < 8, "the regex held the wait past the Stop"
    assert _processes_naming(str(log)) == {}
    tools.bash_job(job, action="kill")


@pytest.mark.timeout(60)
@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="counts open fds through /proc")
def test_a_failed_matcher_launch_ends_the_wait_with_an_error(tmp_path: Path) -> None:
    """A matcher that cannot be started (no file descriptor left) fails the wait at once.

    The launch ran before the matcher thread's ``try``: an ``OSError``
    there (the daemon at its fd or process limit) ended the thread
    silently, so the wait ran to its full deadline without ever
    searching the log and returned a normal report (A14-3 claude).
    The limit is lowered for real: ``RLIMIT_NOFILE`` set to the second
    fd number the listing left unused makes the matcher's pipes fail
    with ``EMFILE``. Only that fd and the listing's own stay usable,
    whatever holes earlier tests left in the fd table (the open count
    plus one left a free number in every hole).
    """
    import resource  # POSIX only, like the /proc this test needs

    tools = UsefulTools(work_dir=str(tmp_path))
    job = _job_id(tools.Bash("sleep 30", "job", background=True))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    listed = {int(name) for name in os.listdir("/proc/self/fd")}
    unused = [fd for fd in range(max(listed) + 3) if fd not in listed]
    resource.setrlimit(resource.RLIMIT_NOFILE, (unused[1], hard))
    before = time.monotonic()
    try:
        report = tools.bash_job(job, action="wait", until="NEVER", timeout_seconds=20)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert time.monotonic() - before < 5, "the wait ran on without a matcher"
    assert report.startswith("Error: could not start the log matcher for until='NEVER':"), report
    tools.bash_job(job, action="kill")


@pytest.mark.timeout(120)
@pytest.mark.skipif(sys.platform != "linux", reason="ulimit -v (RLIMIT_AS) is enforced on Linux")
def test_a_supervisor_that_dies_early_leaves_no_worker_behind(tmp_path: Path) -> None:
    """The matcher's worker ends with its supervisor, however that one ended (B/S13-F6).

    First a supervisor that cannot start its watcher thread, for real:
    a 1 GB default thread stack (``ulimit -s``) cannot be mapped in
    512 MB of address space (``ulimit -v``).  It exited with the worker
    left running, watched by nothing.  Then the supervisor of a running
    wait killed from outside: the wait's owner reaped it and left the
    worker searching for good, and the wait ran on to its deadline
    with nothing searching the log; it ends with an error now.
    """
    log = tmp_path / "alone.log"
    try:
        with (tmp_path / "stderr.txt").open("w") as errors:
            subprocess.run(
                ["bash", "-c", 'ulimit -s 1048576 && ulimit -v 524288 && exec "$@"', "bash",
                 sys.executable, "-c", _SUPERVISE_LOG_MATCHER,
                 sys.executable, "-c", "import time; time.sleep(60)", str(log)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                timeout=60, check=False,
            )
        assert "can't start new thread" in (tmp_path / "stderr.txt").read_text()
        assert _nothing_names(log), "the worker outlived a supervisor that had no watcher"
    finally:
        _kill_naming(log)
    tools = UsefulTools(work_dir=str(tmp_path))
    job = _job_id(tools.Bash("sleep 30", "job", background=True))
    log = tools._jobs[job].log_path
    pool = ThreadPoolExecutor(1)
    waiting = pool.submit(tools.bash_job, job, action="wait", until="NEVER", timeout_seconds=30)
    try:
        deadline = time.monotonic() + 20
        while len(_processes_naming(str(log))) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        supervisors = set(_processes_naming("kill_at_eof")) & set(_processes_naming(str(log)))
        assert len(supervisors) == 1, supervisors
        os.kill(supervisors.pop(), signal.SIGKILL)
        assert _nothing_names(log), "the worker outlived its killed supervisor"
        assert waiting.result(10).startswith(
            "Error: could not start the log matcher for until='NEVER': it exited with status -9"
        )
    finally:
        _kill_naming(log)
        tools.bash_job(job, action="kill")
        pool.shutdown()


# Searches 100 kB with files limited to 1 kB (the limit is the process's, hence a child):
# the spool cannot be written.  Prints the files left in the temporary directory.
_SEARCH_ON_A_FULL_DISK = """
import os, re, resource, sys, tempfile, threading, time
from kiss.agents.sorcar.useful_tools import _search_output
tempfile.tempdir = sys.argv[1]
resource.setrlimit(resource.RLIMIT_FSIZE, (1000, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
try:
    _search_output("x" * 100000, re.compile("x"), threading.Event(), [], time.monotonic() + 5,
                   threading.Event())
except OSError:
    print("left:", os.listdir(sys.argv[1]))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_FSIZE is POSIX")
def test_a_spool_that_cannot_be_written_is_removed(tmp_path: Path) -> None:
    """An ``until`` search whose spool file fails mid-write leaves no file behind (A/S13-F7)."""
    run = subprocess.run(
        [sys.executable, "-c", _SEARCH_ON_A_FULL_DISK, str(tmp_path)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert run.stdout.strip() == "left: []", run.stdout + run.stderr


@pytest.mark.timeout(60)
def test_stop_at_the_matcher_launch_leaves_no_matcher_behind(tmp_path: Path) -> None:
    """A Stop landing as the matcher comes into existence (2026-11-04 review R1) still ends it.

    The runner's Stop lands at any check point of the waiting thread's
    bytecode, including the one right after the matcher's ``Popen``
    returned and before any handle a ``finally`` could kill is bound.
    The hook delivers ``KeyboardInterrupt`` to that thread exactly as the
    matcher's process appears (raised in place when the spawn runs on it,
    injected as the runner does when it runs elsewhere) and keeps the
    process to check that its owner killed and reaped it.
    """
    tools = UsefulTools(work_dir=str(tmp_path))
    job = _job_id(tools.Bash("sleep 30", "job", background=True))
    log = tools._jobs[job].log_path
    waiting = threading.get_ident()
    spawned: list[subprocess.Popen[Any]] = []

    def hook(frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is not popen_process_group.__code__:
            return None
        if event == "return":
            spawned.append(arg)
            if threading.get_ident() == waiting:
                raise KeyboardInterrupt("Stop at the matcher launch")
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(waiting), ctypes.py_object(KeyboardInterrupt)
            )
        return hook

    threading.settrace(hook)
    sys.settrace(hook)
    try:
        with pytest.raises(KeyboardInterrupt):
            tools.bash_job(job, action="wait", until="NEVER", timeout_seconds=30)
    finally:
        sys.settrace(None)
        threading.settrace(None)  # type: ignore[arg-type]
    try:
        assert spawned, "no matcher was started"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (
            spawned[0].returncode is None or _processes_naming(str(log))
        ):
            time.sleep(0.05)
        assert spawned[0].returncode is not None, "the matcher was not reaped by its owner"
        assert _processes_naming(str(log)) == {}, "the matcher outlived the Stop"
    finally:
        for process in spawned:
            if process.poll() is None:
                kill_process_tree(process)
            process.wait(timeout=5)
        tools.bash_job(job, action="kill")


@pytest.mark.timeout(60)
def test_stop_at_the_cleanup_entry_leaves_no_matcher_behind(tmp_path: Path) -> None:
    """A Stop landing as the wait's cleanup begins (2026-11-04 review R2) still ends the matcher.

    The first statement of the wait's ``finally`` tells the matcher's
    owner to end it; a forced Stop landing right there skips the whole
    cleanup, so the owner must end the matcher on its own.  First a real
    ``bash_job`` tool call, interrupted through the production
    ``interrupt_tool_call`` once the waiting thread is held at that
    statement (the watchdog's injection lands there), which the owner
    must learn of through the event captured on the waiting thread;
    then, with no tool call and no task stop event, a ``KeyboardInterrupt``
    raised there, which leaves the owner the deadline alone.  The test
    sets the skipped notification only in its own emergency cleanup.
    """
    tools = UsefulTools(work_dir=str(tmp_path))
    source, first = inspect.getsourcelines(UsefulTools._wait_for_job)
    cleanup = first + next(i for i, line in enumerate(source) if line.strip() == "done.set()")
    token = tool_interrupt.new_tool_call("bash_job")
    owned: dict[str, Any] = {}

    def hook(frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is not UsefulTools._wait_for_job.__code__:
            return None
        if event == "line" and frame.f_lineno == cleanup:
            owned.update(done=frame.f_locals["done"], owner=frame.f_locals["matcher"])
            if tool_interrupt.current_tool_call() is None:
                raise KeyboardInterrupt("Stop at the cleanup entry")
            assert tool_interrupt.interrupt_tool_call(token.thread_ident, call_id=token.call_id)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:  # the watchdog's injection lands here
                time.sleep(0.02)
            raise AssertionError("the Stop was not injected")
        return hook

    def stopped_at_cleanup(command: str, stop: type[BaseException], timeout: float) -> None:
        job = _job_id(tools.Bash(command, "job", background=True))
        log = tools._jobs[job].log_path
        owned.clear()
        sys.settrace(hook)
        try:
            with pytest.raises(stop):
                tools.bash_job(job, action="wait", until="NEVER", timeout_seconds=timeout)
        finally:
            sys.settrace(None)
        try:
            assert owned and not owned["done"].is_set(), "the Stop did not skip the cleanup"
            owned["owner"].join(timeout=5)
            assert not owned["owner"].is_alive(), "the matcher's owner outlived the Stop"
            assert _nothing_names(log), "the matcher outlived the Stop"
        finally:
            owned["done"].set()
            _kill_naming(log)
            tools.bash_job(job, action="kill")

    tool_interrupt.register_tool_call(token)
    try:
        stopped_at_cleanup("sleep 0.3", tool_interrupt.ToolCallInterrupted, 30)
    finally:
        tool_interrupt.unregister_tool_call(token)
    stopped_at_cleanup("sleep 30", KeyboardInterrupt, 0.5)


# A process that starts a matcher for its log and pattern, then lives for the given seconds.
# It keeps the Popen: on Windows a collected Popen closes the matcher's stdin pipe at once
# (POSIX keeps a running child's Popen alive in subprocess._active), ending the matcher.
_OWNER = """
import re, sys, time
from pathlib import Path
from kiss.agents.sorcar.useful_tools import _start_log_matcher
matcher = _start_log_matcher(Path(sys.argv[1]), re.compile(sys.argv[2]))
time.sleep(float(sys.argv[3]))
"""


def _nothing_names(log: Path) -> bool:
    """Return whether no live process names *log* on its command line, waiting up to 3 s."""
    deadline = time.monotonic() + 3
    while _processes_naming(str(log)) and time.monotonic() < deadline:
        time.sleep(0.05)
    return _processes_naming(str(log)) == {}


def _matcher_cpu(log: Path, owner: int) -> float:
    """Return the CPU seconds the processes naming *log* other than *owner* have used."""
    return sum(cpu for pid, cpu in _processes_naming(str(log)).items() if pid != owner)


@pytest.mark.timeout(90)
def test_matcher_dies_with_its_owner(tmp_path: Path) -> None:
    """The matcher ends once the process that started it is gone (2026-11-04 review R2).

    An owner that exits while the matcher is still starting up (before
    the old code looked up its parent), then an owner killed while the
    matcher backtracks in ``(a+)+$`` (where no Python-level check runs
    for hours): in both cases nothing naming the log may remain.
    """
    log = tmp_path / "job.log"
    log.write_text("no match\n")
    try:
        owner = subprocess.run(
            [sys.executable, "-c", _OWNER, str(log), "NEVER", "0"], timeout=30, check=False
        )
        assert owner.returncode == 0
        assert _nothing_names(log), "the matcher outlived an owner that exited during its start"

        log.write_text("a" * 34 + "!\n")
        living = subprocess.Popen([sys.executable, "-c", _OWNER, str(log), r"(a+)+$", "60"])
        try:
            deadline = time.monotonic() + 20
            while _matcher_cpu(log, living.pid) < 1 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert _matcher_cpu(log, living.pid) >= 1, "the matcher never backtracked"
        finally:
            living.kill()
            living.wait(timeout=5)
        assert _nothing_names(log), "the matcher outlived an owner killed while it backtracked"
    finally:
        _kill_naming(log)


def test_long_foreground_sleep_is_refused(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    # Since 2026-10-31 only a sleep longer than the call's timeout is refused
    # (a shorter one runs, or waits on a running job instead).
    out = tools.Bash("sleep 295; echo x", "poll", timeout_seconds=200)
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
    assert tools._jobs[job].returncode() is None
    tools.bash_job(job, action="kill")
    assert tools.Bash("echo x", "bad", on_timeout="later").startswith("Error: on_timeout must be")
    # Checked before a shell rewrite is applied as its Edit: the file stays as it was.
    target = tmp_path / "f.txt"
    target.write_text("old\n")
    tools.Read(str(target))
    out = tools.Bash(f"sed -i 's/old/new/' {target}", "bad rewrite", on_timeout="later")
    assert out.startswith("Error: on_timeout must be"), out
    assert target.read_text() == "old\n"


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
