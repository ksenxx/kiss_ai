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
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kiss.core.processes import (
    PIPE_GIVE_UP_SECONDS,
    CappedOutput,
    _capped_timeout,
    _read_some,
    pid_alive,
    popen_process_group,
    run_capped,
    run_process_group,
)
from kiss.tests.conftest import posix_only


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
