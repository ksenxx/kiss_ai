# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of :func:`kiss.core.processes.run_capped`.

``run_capped`` replaced two copies of "run a command, keep at most N
bytes of stdout, kill on timeout" (``fs_actions._run_capped`` and
``git_worktree._git_stdout_head``'s body).  The cap/timeout/grandchild
behaviour the old copies pinned is still tested where it was
(``tests/server/test_fs_actions_cap_vs_timeout.py``,
``tests/server/test_fixes3_explorer_fs_file_index.py``,
``tests/agents/sorcar/test_staged_diff_bounded.py``); this file covers
the options the merged helper added and the branches those suites do
not reach.  Every child is a real subprocess.

Not reproducible end to end: ``returncode == -1`` (a ``SIGKILL``ed
child that is still not reaped 5 s later needs a process stuck in the
kernel) and, anywhere but on Windows, the Windows poll of a pipe.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import FrameType
from typing import IO, Any

import pytest

from kiss.core import processes
from kiss.core.locked import locked_call
from kiss.core.processes import (
    IS_WINDOWS,
    PIPE_GIVE_UP_SECONDS,
    SIGKILL,
    CappedOutput,
    _capped_timeout,
    _read_some,
    kill_process_group,
    kill_process_tree,
    pid_alive,
    pin_process,
    popen_process_group,
    process_identity,
    run_capped,
    run_process_group,
    start_process_group,
)
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.conftest import posix_only
from kiss.tests.lock_injection import inject
from kiss.tests.server.pause import Pause


def _py(code: str) -> list[str]:
    """A bare interpreter (no site) running *code*, writing ``\\n`` for a newline on every OS.

    Windows opens ``sys.stdout`` with ``\\r\\n`` translation; the tests pin
    the bytes ``run_capped`` keeps, so the child turns it off.
    """
    newline_as_is = 'import sys; sys.stdout.reconfigure(newline="\\n"); '
    return [sys.executable, "-I", "-S", "-c", newline_as_is + code]


def test_output_under_the_cap_is_complete() -> None:
    out = run_capped(_py("print('a'); print('b')"), limit=1000, timeout=10)
    assert (out.stdout, out.truncated, out.timed_out) == (b"a\nb\n", False, False)
    assert (out.returncode, out.stderr) == (0, "")


def test_output_of_exactly_the_cap_is_not_truncated() -> None:
    """The probe read after *limit* bytes sees EOF, so nothing was lost."""
    out = run_capped(_py("import sys; sys.stdout.write('x' * 10)"), limit=10, timeout=10)
    assert out.stdout == b"x" * 10
    assert out.truncated is False


def test_longer_output_is_capped_and_the_child_killed() -> None:
    """One byte past the cap decides ``truncated`` and the child dies early."""
    code = "import sys, time; sys.stdout.write('y' * 11); sys.stdout.flush(); time.sleep(30)"
    started = time.monotonic()
    out = run_capped(_py(code), limit=10, timeout=60)
    assert time.monotonic() - started < 10
    assert out.stdout == b"y" * 10
    assert out.truncated is True
    assert out.timed_out is False
    assert out.returncode != 0


def test_stderr_is_discarded_without_keep_stderr() -> None:
    out = run_capped(_py("import sys; sys.stderr.write('oops')"), limit=100, timeout=10)
    assert out.stderr == ""
    assert out.returncode == 0


def test_stderr_is_kept_up_to_keep_stderr() -> None:
    """Stderr is read to the end but only its first *keep_stderr* bytes are kept."""
    out = run_capped(
        _py("import sys; sys.stderr.write('e' * 300000)"),
        limit=100,
        timeout=10,
        keep_stderr=7,
    )
    assert out.stderr == "e" * 7
    assert out.returncode == 0
    short = run_capped(
        _py("import sys; sys.stderr.write('short')"),
        limit=100,
        timeout=10,
        keep_stderr=65536,
    )
    assert short.stderr == "short"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # A complete line is kept, the partial one after it dropped.
        ("import sys; sys.stdout.write('line1\\nline2\\nrest')", b"line1\nline2"),
        # No newline in the head: the raw head is kept.
        ("import sys; sys.stdout.write('x' * 50)", b"x" * 15),
    ],
)
def test_last_full_line_cuts_a_capped_head(code: str, expected: bytes) -> None:
    out = run_capped(_py(code), limit=15, timeout=10, last_full_line=True)
    assert out.truncated is True
    assert out.stdout == expected


def test_last_full_line_leaves_complete_output_alone() -> None:
    out = run_capped(_py("print('a'); print('b')"), limit=1000, timeout=10, last_full_line=True)
    assert out.stdout == b"a\nb\n"
    assert out.truncated is False


def test_without_last_full_line_the_capped_head_is_raw() -> None:
    code = "import sys; sys.stdout.write('line1\\nline2\\nrest')"
    out = run_capped(_py(code), limit=8, timeout=10)
    assert out.stdout == b"line1\nli"
    assert out.truncated is True


def test_last_full_line_also_cuts_a_timed_out_head() -> None:
    code = "import sys, time; sys.stdout.write('done\\npart'); sys.stdout.flush(); time.sleep(30)"
    out = run_capped(_py(code), limit=1000, timeout=0.5, last_full_line=True)
    assert out.timed_out is True
    assert out.truncated is False
    assert out.stdout == b"done"


def test_cwd_and_env_reach_the_child(tmp_path: Path) -> None:
    out = run_capped(
        _py("import os; print(os.getcwd()); print(os.environ.get('SWEEP1026', '-'))"),
        limit=10_000,
        timeout=10,
        cwd=tmp_path,
        env={**os.environ, "SWEEP1026": "set"},
    )
    cwd_line, env_line = out.stdout.decode().splitlines()
    assert Path(cwd_line).resolve() == tmp_path.resolve()
    assert env_line == "set"


def test_a_command_that_cannot_start_raises() -> None:
    with pytest.raises(OSError):
        run_capped(["kiss-sweep1026-no-such-command"], limit=10, timeout=10)


@posix_only("on Windows communicate's reader threads own the pipes and close them at EOF")
def test_a_timed_out_run_process_group_closes_the_pipes_it_opened() -> None:
    """The pipes of a killed child are closed with it, not left to the garbage collector.

    ``run_process_group`` killed and reaped a child that outlived its
    timeout but closed neither of its pipes, so the two descriptors
    stayed open for as long as the ``Popen`` lived: the ``TimeoutExpired``
    it raises keeps the frame (and the ``Popen``) alive through its
    traceback, and a caller that stores the error (a future, an error
    list) kept both descriptors with it, one pair per failed probe.
    """
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run_process_group(_py("import time; time.sleep(60)"), timeout=0.2)
    assert raised.value.timeout == 0.2, "the error names a slice of the wait, not the timeout"
    frame = raised.value.__traceback__
    while frame is not None and frame.tb_frame.f_code.co_name != "run_process_group":
        frame = frame.tb_next
    assert frame is not None
    proc = frame.tb_frame.f_locals["proc"]
    assert proc.returncode is not None, "the child was not reaped"
    assert proc.stdout.closed and proc.stderr.closed, "the child's pipes were left open"


# A child interpreter in which no thread can be started: a tebibyte of
# stack per thread under a 64 GiB address-space limit makes
# ``pthread_create`` fail.  Both settings are process-wide, hence the
# child.  It reports the state of the child ``run_capped`` had spawned,
# read from the raised error's traceback (which keeps the ``Popen`` and
# so any open pipe alive, exactly as a caller storing the error would).
_NO_THREADS = """
import json, resource, sys, threading
from kiss.core.processes import run_capped
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
cap = 1 << 36 if hard == resource.RLIM_INFINITY else min(1 << 36, hard)
resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
threading.stack_size(1 << 40)
try:
    run_capped(
        [sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
        limit=100, timeout=60, keep_stderr=100,
    )
except RuntimeError as exc:
    tb = exc.__traceback__
    while tb.tb_frame.f_code.co_name != "run_capped":
        tb = tb.tb_next
    proc = tb.tb_frame.f_locals["proc"]
    print(json.dumps({
        "error": str(exc), "reaped": proc.returncode is not None,
        "stdout_closed": proc.stdout.closed, "stderr_closed": proc.stderr.closed,
    }))
"""


@posix_only("RLIMIT_AS")
@pytest.mark.skipif(
    sys.platform == "darwin", reason="macOS refuses to lower RLIMIT_AS from unlimited"
)
def test_a_drain_thread_that_cannot_start_leaves_no_pipe_open() -> None:
    """``run_capped`` closes the child's stderr itself when its drain thread cannot be started.

    The handler of ``run_capped`` left every non-null stderr pipe to its
    drain thread, which closes it at EOF or on the stop event; a
    ``Thread.start`` that raised (no thread to be had, or a Stop landing
    before the thread ran) left no drain, so the child was killed and
    reaped but its stderr pipe stayed open for as long as the error
    lived.  The pipe is the caller's until the drain has taken it.
    """
    result = subprocess.run(
        [sys.executable, "-c", _NO_THREADS], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "can't start new thread" in report.pop("error")
    assert report == {"reaped": True, "stdout_closed": True, "stderr_closed": True}


def test_timeout_fired_after_the_result_is_decided_is_ignored() -> None:
    """The watchdog finds ``_finished`` set and leaves the child alone.

    The timer thread reaches :func:`_capped_timeout` between the
    reader's verdict and ``timer.cancel()`` only in a window a few
    bytecodes wide, so the callback is driven here directly against a
    live child.
    """
    proc = popen_process_group(_py("import time; time.sleep(30)"))
    try:
        out = CappedOutput()
        out._finished = True
        _capped_timeout(proc, out)
        assert out.timed_out is False
        assert proc.poll() is None, "a decided result must not kill the child"
        out._finished = False
        _capped_timeout(proc, out)
        assert out.timed_out is True
        proc.wait(timeout=10)
    finally:
        with contextlib.suppress(OSError):
            proc.kill()
        proc.wait(timeout=10)


# Publishes its pid atomically (rename) and sleeps; the pid file tells the
# test when the child is up and which process must be gone afterwards.
_PID_SLEEPER = """
import os, sys, time
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(60)
"""


def _wait_for_pid_file(pid_file: Path) -> int:
    deadline = time.monotonic() + 20
    while not pid_file.exists():
        assert time.monotonic() < deadline, "the child never published its pid"
        time.sleep(0.02)
    return int(pid_file.read_text())


@pytest.mark.parametrize("pid", [2**31, 2**32, 2**64])
def test_a_number_too_large_for_a_pid_names_no_process(pid: int) -> None:
    """A number beyond the OS's pid type (a corrupt pidfile's) names no process and no group.

    The system calls refuse such a number with an error that is no
    ``OSError`` (``OverflowError`` from ``os.kill`` and ``os.killpg``,
    ``ctypes.ArgumentError`` from ``OpenProcess`` past a DWORD): it went
    through every caller's handler, and the daemon's sweep of an
    orphaned tunnel raised instead of finding the pidfile stale.
    """
    assert pid_alive(pid) is False
    assert process_identity(pid) is None
    assert pin_process(pid) is None
    with pytest.raises(ProcessLookupError):
        kill_process_group(pid, 0)


@pytest.fixture
def sigint_raises_keyboard_interrupt() -> Iterator[None]:
    """Python's SIGINT handler installed for the test, whatever the launcher left.

    A runner started as a background job of a non-interactive shell
    inherits SIGINT ignored, and Python then installs no handler for
    it at start-up: a SIGINT sent to the process would be dropped.
    """
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@posix_only("SIGINT is delivered to the waiting thread with pthread_kill")
def test_an_interrupted_wait_kills_and_reaps_the_child(
    tmp_path: Path, sigint_raises_keyboard_interrupt: None
) -> None:
    """A ``KeyboardInterrupt`` while reading the pipe does not leak the child.

    A real ``SIGINT`` reaches this (main) thread while ``run_capped``
    is blocked on the child's stdout, as a Stop landing in a task
    thread does; the exception propagates, and the child (which never
    writes and would sleep for 60 s) must have been killed and reaped
    by then, with its stdout pipe closed, as the git runner this helper
    replaced guaranteed in its ``finally``.
    """
    pid_file = tmp_path / "sleeper.pid"
    main_thread = threading.main_thread().ident
    assert main_thread is not None

    def interrupt_once_the_child_is_up() -> None:
        _wait_for_pid_file(pid_file)
        signal.pthread_kill(main_thread, signal.SIGINT)

    threading.Thread(target=interrupt_once_the_child_is_up, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        run_capped(
            [sys.executable, "-I", "-S", "-c", _PID_SLEEPER, str(pid_file)],
            limit=1000,
            timeout=60,
            keep_stderr=1000,
        )
    elapsed = time.monotonic() - started
    pid = int(pid_file.read_text())
    try:
        assert elapsed < 20, "the interrupt must not wait out the timeout"
        assert not pid_alive(pid), "the child outlived the interrupted wait"
        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)
    finally:
        with contextlib.suppress(OSError):
            os.killpg(pid, signal.SIGKILL)


class _InterruptsAsTheChildIsCreated:
    """Profile hook: a real ``SIGINT`` inside ``Popen``, its child created and not yet published.

    POSIX: as ``_fork_exec`` returns the pid that ``Popen`` has yet to
    store.  Windows: as ``_close_pipe_fds`` returns to
    ``_execute_child``, whose locals alone hold the handles of the child
    that ``CreateProcess`` made; ``handles`` keeps them (process, thread)
    for the test to end a child the spawn lost.
    """

    def __init__(self) -> None:
        self.handles: tuple[int, int] | None = None

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if IS_WINDOWS:
            if event != "return" or frame.f_code.co_name != "_close_pipe_fds":
                return
            created = frame.f_back.f_locals if frame.f_back is not None else {}
            if "hp" not in created:
                return
            self.handles = (created["hp"], created["ht"])
        elif event != "c_return" or getattr(arg, "__name__", None) != "fork_exec":
            return
        sys.setprofile(None)
        signal.raise_signal(signal.SIGINT)


def test_an_interrupt_while_the_child_is_created_is_raised_with_the_child_in_hand(
    sigint_raises_keyboard_interrupt: None,
) -> None:
    """A ``KeyboardInterrupt`` arriving while ``Popen`` creates the child does not lose the child.

    Python raises the interrupt of a ``SIGINT`` on the main thread at
    its next check, and the deferral of the Stop injector does not hold
    it back: raised inside ``Popen.__init__`` between the call that
    creates the child and the statement that stores its pid, it left a
    child nobody could kill or reap, and on Windows, where the child is
    created suspended, one that never ran and never ended.  The spawn
    now has the signal recorded until the child is published (in its
    job and resumed, on Windows) and calls the handler then: the
    interrupt is raised where the child (which exits at once, so a
    survivor is a lost one) is killed and reaped.
    """
    proc: subprocess.Popen[bytes] = object.__new__(subprocess.Popen)
    hook = _InterruptsAsTheChildIsCreated()
    sys.setprofile(hook)
    try:
        with pytest.raises(KeyboardInterrupt):
            start_process_group(proc, _py("pass"))
    finally:
        sys.setprofile(None)
        lost = getattr(proc, "pid", None) is None
        if lost and hook.handles is not None:  # Windows: suspended for good unless ended here
            kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
            kernel32.TerminateProcess(ctypes.c_void_p(hook.handles[0]), 1)
            for handle in hook.handles:
                kernel32.CloseHandle(ctypes.c_void_p(handle))
    assert not lost, "the interrupt was raised before the spawn held its child"
    assert proc.returncode is not None, "the child was not reaped"
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def _raise_system_exit(signum: int, frame: FrameType | None) -> None:
    """A signal handler that raises, as Python's own ``SIGINT`` handler does."""
    raise SystemExit(signum)


class _InterruptedStdin:
    """A stdin for the child: ``Popen`` gets its descriptor with a real ``SIGINT`` raised first."""

    def __init__(self, file: IO[bytes]) -> None:
        self.file = file

    def fileno(self) -> int:
        signal.raise_signal(signal.SIGINT)
        return self.file.fileno()


class _TerminatedAsSigintIsReinstalled:
    """:class:`Pause` hook that never holds: a real ``SIGTERM`` as ``signal.signal`` starts.

    Not at its first call, which keeps the ``SIGINT`` handler from
    running, but at the *failures* next ones, which are to put it back.
    (Such a hook stays in place when it raises, which a profile function
    does not.)
    """

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, _locals: dict[str, Any]) -> bool:
        self.calls += 1
        if 1 < self.calls <= 1 + self.failures:
            signal.raise_signal(signal.SIGTERM)
        return False


@pytest.mark.parametrize("failures", [1, 2])
def test_sigint_is_handled_again_whatever_interrupts_the_reinstallation_of_its_handler(
    failures: int, sigint_raises_keyboard_interrupt: None
) -> None:
    """What other signals' handlers raise as the spawn ends leaves ``SIGINT`` handled.

    The spawn keeps the ``SIGINT`` handler from running until its child
    is published, then puts it back.  ``signal.signal`` first runs the
    handlers of pending signals, and what one of them raises ends the
    call before it has installed anything: the spawn calls it once
    more, and a ``SIGINT`` is handled again even where that fails as
    well (the stand-in, still installed, hands it on), or every later
    one would only be noted.  The ``SIGINT`` noted during the spawn is
    handled too, and the child killed and reaped as on any other
    exception.
    """
    proc: subprocess.Popen[bytes] = object.__new__(subprocess.Popen)
    sigterm = signal.signal(signal.SIGTERM, _raise_system_exit)
    try:
        with (
            open(os.devnull, "rb") as null,
            Pause(signal.signal.__code__, _TerminatedAsSigintIsReinstalled(failures)),
            pytest.raises(KeyboardInterrupt) as interrupt,
        ):
            start_process_group(proc, _py("pass"), stdin=_InterruptedStdin(null))
    finally:
        signal.signal(signal.SIGTERM, sigterm)
    assert isinstance(interrupt.value.__context__, SystemExit), "SIGTERM's handler did not run"
    assert (signal.getsignal(signal.SIGINT) is signal.default_int_handler) == (failures == 1)
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)
    assert proc.returncode is not None, "the child was not reaped"


_ESCAPED_HOLDER = """
import os, subprocess, sys
child = subprocess.Popen(
    [sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"], start_new_session=True,
)
with open(sys.argv[1], "w") as f:
    f.write(str(child.pid))
sys.stdout.write("hi\\n")
sys.stderr.write("err\\n")
"""

_ESCAPED_SPEWER = """
import os, subprocess, sys
code = (
    "import sys, time\\n"
    "while True:\\n"
    "    sys.stdout.write('x' * 100 + '\\\\n'); sys.stdout.flush(); time.sleep(0.2)"
)
child = subprocess.Popen([sys.executable, "-I", "-S", "-c", code], start_new_session=True)
with open(sys.argv[1], "w") as f:
    f.write(str(child.pid))
"""


def _kill_pid_file(pid_file: Path) -> None:
    with contextlib.suppress(OSError, ValueError):
        os.kill(int(pid_file.read_text()), signal.SIGKILL)


@pytest.mark.parametrize(("give_up_in", "stopped"), [(-1.0, False), (60.0, True)])
def test_a_silent_open_pipe_is_not_read_past_the_deadline_or_a_stop(
    give_up_in: float, stopped: bool
) -> None:
    """A pipe that stays open and silent is given up at once past the deadline or on a stop.

    The Windows branch read it plainly, whatever the two said: behind
    a writer that outlived the kill (a child the Job Object never held)
    the caller, or the stderr drain, sat in that read until the writer
    left.  Both platforms share the loop now; this is the regression on
    Windows, whose own part (the poll of the pipe) only Windows runs.
    """
    read_fd, write_fd = os.pipe()
    stop = threading.Event()
    if stopped:
        stop.set()
    with open(read_fd, "rb") as stream, ThreadPoolExecutor(1) as pool:
        read = pool.submit(_read_some, stream, 1, time.monotonic() + give_up_in, stop)
        try:
            assert read.result(timeout=10) == b""
        finally:
            os.close(write_fd)  # EOF releases a read that heeded neither


@posix_only("a descendant in its own session survives the group kill")
def test_pipes_held_by_an_escaped_descendant_are_given_up(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stdout and stderr both stop waiting ``PIPE_GIVE_UP_SECONDS`` after the kill.

    The command itself exits at once, so the deadline is what sets
    ``timed_out``; the reads end when the shared give-up deadline
    passes, not when the holder (asleep for 60 s) exits.
    """
    pid_file = tmp_path / "holder.pid"
    started = time.monotonic()
    try:
        with caplog.at_level(logging.WARNING, logger="kiss.core.processes"):
            out = run_capped(
                [sys.executable, "-I", "-S", "-c", _ESCAPED_HOLDER, str(pid_file)],
                limit=1000,
                timeout=0.5,
                keep_stderr=65536,
            )
    finally:
        _kill_pid_file(pid_file)
    elapsed = time.monotonic() - started
    assert 0.5 + PIPE_GIVE_UP_SECONDS <= elapsed < 0.5 + PIPE_GIVE_UP_SECONDS + 4.0
    assert out.stdout == b"hi\n"
    assert out.stderr == "err\n"
    assert out.timed_out is True
    assert out.truncated is False
    assert sum("giving up" in r.getMessage() for r in caplog.records) >= 2


# Writes to stderr, hands only stderr to a descendant in its own session,
# and exits at once: the command is done while a stray still holds stderr.
_ESCAPED_STDERR_HOLDER = """
import os, subprocess, sys
sys.stderr.write("err\\n")
sys.stderr.flush()
child = subprocess.Popen(
    [sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
    start_new_session=True, stdout=subprocess.DEVNULL,
)
with open(sys.argv[1], "w") as f:
    f.write(str(child.pid))
"""


@posix_only("a descendant in its own session survives the group kill")
def test_stderr_read_before_a_stray_holder_stalls_the_drain_is_kept(tmp_path: Path) -> None:
    """Stderr written by the command is kept even when its drain never reaches EOF.

    The command exits at once, so the result is decided long before
    the timeout; only its stderr pipe stays open, in a descendant that
    escaped the group.  The caller stops waiting for the drain thread
    after ``PIPE_GIVE_UP_SECONDS`` and must still report the stderr
    read so far, not an empty string.  The drain thread is told to stop
    at that point: left to its own deadline (the command's timeout plus
    the give-up), it would keep the pipe and the returned result for
    another 25 s and overwrite ``stderr`` with whatever the stray wrote
    meanwhile.
    """
    pid_file = tmp_path / "stderr-holder.pid"
    threads_before = set(threading.enumerate())
    started = time.monotonic()
    try:
        out = run_capped(
            [sys.executable, "-I", "-S", "-c", _ESCAPED_STDERR_HOLDER, str(pid_file)],
            limit=1000,
            timeout=30,
            keep_stderr=65536,
        )
    finally:
        _kill_pid_file(pid_file)
    elapsed = time.monotonic() - started
    assert elapsed < 30, "the finished command must not wait out the timeout"
    assert out.stderr == "err\n"
    assert out.returncode == 0
    assert out.timed_out is False
    assert out.truncated is False
    for thread in threading.enumerate():
        if isinstance(thread, threading.Timer):
            thread.join(5)  # the cancelled watchdog is on its way out
    stray = [t for t in threading.enumerate() if t not in threads_before and t.is_alive()]
    assert stray == [], "the stderr drain thread outlived the call"


@posix_only("a descendant in its own session survives the group kill")
def test_an_escaped_descendant_that_keeps_writing_cannot_extend_the_read(
    tmp_path: Path,
) -> None:
    """The give-up deadline is absolute: a read past it returns nothing even
    though output is ready, so a chatty escapee cannot hold the caller."""
    pid_file = tmp_path / "spewer.pid"
    started = time.monotonic()
    try:
        out = run_capped(
            [sys.executable, "-I", "-S", "-c", _ESCAPED_SPEWER, str(pid_file)],
            limit=10_000_000,
            timeout=0.5,
        )
    finally:
        _kill_pid_file(pid_file)
    elapsed = time.monotonic() - started
    assert 0.5 + PIPE_GIVE_UP_SECONDS <= elapsed < 0.5 + PIPE_GIVE_UP_SECONDS + 4.0
    assert out.timed_out is True
    assert out.stdout.startswith(b"x" * 100 + b"\n")
    assert len(out.stdout) < 10_000_000


windows_only = pytest.mark.skipif(not IS_WINDOWS, reason="Job Objects are Windows'")

# Starts a grandchild (``_PID_SLEEPER``) as its first act, then sleeps.
_SPAWNS_AT_ONCE = """
import subprocess, sys, time
subprocess.Popen([sys.executable, "-I", "-S", "-c", sys.argv[1], sys.argv[2]])
time.sleep(60)
"""


class _RunsBeforeItsJob:
    """Profile hook: as the spawn goes on to put its child in a Job Object, let the child run.

    The child's first act is a grandchild that publishes *pid_file*;
    ``ran`` tells whether it did in the two seconds waited here.
    """

    def __init__(self, pid_file: Path) -> None:
        self.pid_file = pid_file
        self.ran: bool | None = None

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "call" and frame.f_code is processes._windows_enrol_in_job.__code__:
            deadline = time.monotonic() + 2
            while not self.pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.ran = self.pid_file.exists()


@windows_only
def test_a_child_starts_no_descendant_before_it_is_in_its_job(tmp_path: Path) -> None:
    """A child runs only once it is in its Job Object, so a kill takes every descendant.

    Windows puts a new process in the jobs of its parent, and in none
    its parent joins later.  The child used to run from its creation
    and was put in its job afterwards: a launcher quick to start its
    worker (it has two seconds here, the spawn held at that point) left
    it outside the job, where the kill of the group did not reach it.
    The child is now created suspended and resumed once it is in the job.
    """
    pid_file = tmp_path / "grandchild.pid"
    hook = _RunsBeforeItsJob(pid_file)
    sys.setprofile(hook)
    try:
        proc = popen_process_group(
            [sys.executable, "-I", "-S", "-c", _SPAWNS_AT_ONCE, _PID_SLEEPER, str(pid_file)]
        )
    finally:
        sys.setprofile(None)
    try:
        assert hook.ran is False, "the child ran before it was in its Job Object"
        grandchild = _wait_for_pid_file(pid_file)
        kill_process_tree(proc)
        proc.wait(timeout=10)
        deadline = time.monotonic() + 10
        while pid_alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pid_alive(grandchild), "a descendant outlived the kill of its group"
    finally:
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(pid_file.read_text()), SIGKILL)
        with contextlib.suppress(OSError):
            proc.kill()
        proc.wait(timeout=10)


class _Spawner(threading.Thread):
    """Start a minute's sleeper with :func:`popen_process_group`, keeping what the call raised."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.raised: BaseException | None = None

    def run(self) -> None:
        try:
            popen_process_group(_py("import time; time.sleep(60)"))
        except BaseException as exc:  # the injected Stop is the expected outcome
            self.raised = exc


def _registering_its_job(worker: threading.Thread) -> FrameType:
    """Wait until *worker* waits for the job registry's lock; its frame enrolling the child."""
    deadline = time.monotonic() + 30
    while True:
        frame = sys._current_frames().get(worker.ident or 0)
        if frame is not None and frame.f_code is locked_call.__code__:
            caller = frame.f_back
            if caller is not None and caller.f_code is processes._windows_enrol_in_job.__code__:
                return caller
        assert time.monotonic() < deadline, "the spawn never reached the job registry"
        time.sleep(0.005)


def _is_open_job(handle: int) -> bool:
    """Whether *handle* is an open handle on a Job Object."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    accounting = ctypes.create_string_buffer(48)  # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
    return bool(
        kernel32.QueryInformationJobObject(
            ctypes.c_void_p(handle), 1, accounting, len(accounting), None
        )
    )


@windows_only
@pytest.mark.parametrize("deferred", [True, False], ids=["the task's Stop", "another exception"])
def test_a_stop_while_the_job_is_registered_leaves_its_handle_owned_or_closed(
    deferred: bool,
) -> None:
    """A Stop landing while the child's Job Object is registered does not leak the job's handle.

    The registration waits for the registry's lock (held here, as by
    another task's spawn or kill) with the child already in the job.
    The spawn stopped deferring the Stop once ``Popen`` had returned,
    so it was raised in that wait: the child was killed, and the handle,
    which the registry did not hold yet, stayed open for good.  The Stop
    is now deferred until the job is registered and the child killed
    with it; any other exception raised in the wait (set here the way
    the interpreter sets one, past the deferral) closes the handle it
    leaves unregistered.
    """
    worker = _Spawner()
    processes._WINDOWS_JOBS_LOCK.acquire()
    try:
        worker.start()
        enrolment = _registering_its_job(worker)
        job, proc = enrolment.f_locals["job"], enrolment.f_locals["proc"]
        if deferred:
            assert inject_async_exception(worker.ident or 0, KeyboardInterrupt) == 1
        else:
            assert inject(worker) == 1
    finally:
        processes._WINDOWS_JOBS_LOCK.release()
    try:
        worker.join(30)
        assert isinstance(worker.raised, KeyboardInterrupt)
        assert proc.returncode is not None, "the child was left running after the Stop"
        assert processes._WINDOWS_JOBS.get(proc.pid) == job or not _is_open_job(job), (
            "the job's handle is open and nothing owns it"
        )
    finally:
        if proc.returncode is None:
            kill_process_tree(proc)
            proc.wait(timeout=10)


class _RaisesOnceTheJobExists:
    """Trace function: has *raise_here* raise in the spawning thread once its job exists.

    That is the first line of ``_windows_enrol_in_job`` reached with the
    new job's handle in a local (``jobs``, or ``job``): the child is not
    in the job yet, and no registry holds the handle (kept in ``job``).
    With *registered*, it is the return of the function that registers
    the job instead.
    """

    def __init__(self, raise_here: Callable[[], object], registered: bool) -> None:
        self.raise_here = raise_here
        where = processes._windows_record_job if registered else processes._windows_enrol_in_job
        self.code = where.__code__
        self.event = "return" if registered else "line"
        self.job: int | None = None

    def __call__(self, frame: FrameType, event: str, arg: object) -> Any:
        if frame.f_code is not self.code:
            return None
        if event == self.event and self.job is None:
            made = [*frame.f_locals.get("jobs", ()), frame.f_locals.get("job")]
            if made[0]:
                self.job = made[0]
                self.raise_here()
        return self


@windows_only
@pytest.mark.parametrize(
    ("deferred", "registered"),
    [(True, False), (False, False), (False, True)],
    ids=["an interrupt", "another exception", "another exception, the job registered"],
)
def test_an_exception_once_the_job_exists_leaves_its_handle_owned_or_closed(
    deferred: bool, registered: bool, sigint_raises_keyboard_interrupt: None
) -> None:
    """An exception raised once the child's Job Object exists does not leak the job's handle.

    The handler that closes an unregistered handle covered the
    registration only: an exception raised between the call that
    creates the job and that registration (the interrupt of a
    ``SIGINT``, which the Stop injector's deferral does not hold back)
    killed the child and left the handle open with no owner.  The
    interrupt is now raised once the job is registered, and the handle
    is owned from the call that creates it, so any other exception
    raised from there on (set here the way the interpreter sets one)
    closes it, unless the registry holds it by then: a registered job
    stays open for the kill of the child's group.
    """
    proc: subprocess.Popen[bytes] = object.__new__(subprocess.Popen)
    raise_here = (
        partial(signal.raise_signal, signal.SIGINT)
        if deferred
        else partial(inject, threading.main_thread())
    )
    hook = _RaisesOnceTheJobExists(raise_here, registered)
    tracing = sys.gettrace()
    sys.settrace(hook)
    try:
        with pytest.raises(KeyboardInterrupt):
            start_process_group(proc, _py("pass"))
    finally:
        sys.settrace(tracing)
    try:
        assert hook.job is not None, "the spawn made no Job Object"
        assert proc.returncode is not None, "the child was left running after the exception"
        assert (processes._WINDOWS_JOBS.get(proc.pid) == hook.job) == _is_open_job(hook.job), (
            "the job's handle is open with no owner, or closed with one"
        )
    finally:
        if proc.returncode is None:
            kill_process_tree(proc)
            proc.wait(timeout=10)
