# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of :func:`kiss.core.processes.signal_unreaped_child`.

The helper signals a child only while it is this process's unreaped
child, so a pid another thread's wait has reaped (and the kernel may
have handed to a stranger) is never signalled.  Every child is a real
subprocess.

Not reproducible end to end: the window between the ownership check
and the signal, which the pidfd closes on Linux (a stranger would have
to take the pid in that microsecond); ``pidfd_open`` failing with
``ENOSYS``/``EPERM`` on a kernel or seccomp profile without it; and
that a zombie is not signalled, since a signal to a zombie has no
effect to observe (the zombie test checks what it can: the helper
still reports the child as the caller's to reap).
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time

import pytest

from kiss.core.processes import exit_status_unreaped, signal_unreaped_child

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX wait semantics")


def test_a_running_child_is_signalled_and_left_for_the_caller_to_reap() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert signal_unreaped_child(proc.pid, signal.SIGKILL) is True
        assert proc.wait(timeout=10) == -signal.SIGKILL
    finally:
        if proc.returncode is None:
            proc.kill()
            proc.wait()


def test_a_stopped_child_is_running_not_exited_and_is_signalled() -> None:
    """A SIGSTOPped child has not exited: it is signalled and its status stays None.

    macOS answers ``waitid(WEXITED | WNOHANG | WNOWAIT)`` for a stopped
    child too (``CLD_STOPPED``); taken for an exit, the frozen Playwright
    driver was never killed and the browser close waited for it.
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        os.kill(proc.pid, signal.SIGSTOP)
        time.sleep(0.2)
        assert exit_status_unreaped(proc) is None, "a stopped child counted as exited"
        assert signal_unreaped_child(proc.pid, signal.SIGKILL) is True
        assert proc.wait(timeout=10) == -signal.SIGKILL
    finally:
        if proc.returncode is None:
            proc.kill()
            proc.wait()


def test_an_exited_unreaped_child_is_still_the_callers_to_reap() -> None:
    """A zombie keeps its pid, so the helper reports it as the caller's, with its status."""
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    deadline = time.monotonic() + 10
    while exit_status_unreaped(proc) is None:
        assert time.monotonic() < deadline, "the child did not exit"
        time.sleep(0.02)
    assert signal_unreaped_child(proc.pid, signal.SIGKILL) is True
    pid, status = os.waitpid(proc.pid, 0)
    assert pid == proc.pid and os.waitstatus_to_exitcode(status) == 3
    proc.returncode = 3  # reaped here, not through Popen


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="pidfds are Linux-only")
def test_without_a_descriptor_for_the_pidfd_the_helper_raises_and_signals_nothing() -> None:
    """Out of descriptors, the bare pid is not signalled in the pidfd's place.

    Run in a subprocess so that the descriptor limit of this process is
    left alone: the child lowers its own soft ``RLIMIT_NOFILE`` to zero,
    calls the helper on a sleeping grandchild and reports whether the
    helper raised and whether the grandchild is still running.
    """
    code = """
import resource, signal, subprocess, sys
from kiss.core.processes import signal_unreaped_child
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (0, hard))
try:
    print("returned", signal_unreaped_child(child.pid, signal.SIGKILL))
except OSError as exc:
    print("raised", exc.errno)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
print("alive" if child.poll() is None else "dead")
child.kill()
child.wait()
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert run.stdout.split() == ["raised", str(errno.EMFILE), "alive"], run.stdout


def test_a_reaped_child_is_not_signalled() -> None:
    """Once reaped the pid may belong to anyone: the helper reports it gone."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    assert proc.wait(timeout=10) == 0
    assert signal_unreaped_child(proc.pid, signal.SIGKILL) is False
