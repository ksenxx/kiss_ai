# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``pty_spawn`` helper used by sorcar CLI tests.

These tests pin down two contractual guarantees of
:func:`kiss.tests.agents.sorcar._pty_helper.pty_spawn` that the rest of
the sorcar CLI/REPL test suite depends on:

1. The child runs with the PTY slave as its standard streams, so writes
   to the master fd appear on the child's stdin and the child's stdout
   appears as readable bytes on the master fd. This proves the basic
   replacement of ``pty.fork()`` is wired correctly.
2. Writing the ``\\x03`` byte to the master fd delivers ``SIGINT`` to
   the child's foreground process group. This proves the child's
   ``TIOCSCTTY`` ioctl succeeded — without a controlling terminal the
   line discipline would silently drop the interrupt character and the
   sorcar bughunt tests for Ctrl+C would become silently broken.

The tests are kept intentionally narrow (single short-lived child each)
so they execute in well under a second and do not need the ``slow``
marker that the heavier sorcar PTY-driven tests carry.
"""

from __future__ import annotations

import os
import select
import sys
import time

from kiss.tests.agents.sorcar._pty_helper import pty_spawn


def _drain(fd: int, deadline: float) -> bytes:
    """Read all currently-available bytes from *fd* until *deadline*.

    Stops when ``select`` reports no readable data within a short poll
    window, or when the absolute ``deadline`` (``time.monotonic()``) is
    reached, or when the child closes the master fd (``OSError`` /
    empty read). Returning early on quiet is intentional — these tests
    read complete short outputs in one or two iterations.
    """
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        ready, _, _ = select.select([fd], [], [], min(0.1, remaining))
        if not ready:
            if chunks:
                return b"".join(chunks)
            continue
        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def _reap(pid: int, fd: int, timeout: float = 5.0) -> int:
    """Wait for *pid* to exit, returning the raw ``os.waitpid`` status.

    Polls with ``WNOHANG`` rather than blocking so that a stuck child
    cannot wedge the test forever — once the polling deadline elapses
    the child is force-killed with ``SIGKILL`` so the test framework
    reports a real failure rather than hanging in a slot.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            try:
                os.close(fd)
            except OSError:
                pass
            return status
        time.sleep(0.02)
    # Last-resort cleanup so the worker process slot is freed.
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass
    _, status = os.waitpid(pid, 0)
    try:
        os.close(fd)
    except OSError:
        pass
    return status


def test_pty_spawn_runs_child_and_captures_stdout() -> None:
    """Child stdout written through the PTY slave is readable on the master fd."""
    pid, fd = pty_spawn(
        [sys.executable, "-c", "import sys; sys.stdout.write('hello\\n')"],
    )
    try:
        output = _drain(fd, time.monotonic() + 5.0)
    finally:
        status = _reap(pid, fd)
    assert b"hello" in output, output
    assert os.WIFEXITED(status), status
    assert os.WEXITSTATUS(status) == 0, os.WEXITSTATUS(status)


def test_pty_spawn_delivers_sigint_via_etx_byte() -> None:
    """Writing ``\\x03`` to the master fd delivers ``SIGINT`` to the child.

    The child installs a ``SIGINT`` handler *before* printing ``ready``
    (so there is no window in which the signal could arrive with the
    default disposition and kill the child uncaught — the source of a
    rare flake under heavy parallel test load), then parks in a long
    ``time.sleep``. The handler prints a sentinel and exits 0. If
    ``TIOCSCTTY`` failed in the helper, the line discipline would
    discard the ETX byte and the sentinel would never appear — the test
    would then fail on timeout.
    """
    child_code = (
        "import signal, sys, time\n"
        "def _on_sigint(signum, frame):\n"
        "    sys.stdout.write('interrupted\\n'); sys.stdout.flush()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGINT, _on_sigint)\n"
        "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    pid, fd = pty_spawn([sys.executable, "-c", child_code])
    try:
        ready = _drain(fd, time.monotonic() + 5.0)
        assert b"ready" in ready, ready
        # ETX byte triggers SIGINT to the child's foreground PG via the
        # line discipline because TIOCSCTTY made the slave PTY the
        # controlling terminal.
        os.write(fd, b"\x03")
        tail = _drain(fd, time.monotonic() + 5.0)
    finally:
        status = _reap(pid, fd)
    assert b"interrupted" in (ready + tail), ready + tail
    assert os.WIFEXITED(status), status
    assert os.WEXITSTATUS(status) == 0, os.WEXITSTATUS(status)
