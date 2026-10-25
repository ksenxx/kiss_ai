# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``processes.terminate_process_group``.

The helper is the one SIGTERM-then-SIGKILL sequence behind
``git_worktree._git`` (so git removes its ``.lock`` files) and
``web_stealth`` (the Xvfb wrapper).  It must reap a child that honours
SIGTERM, escalate to SIGKILL for one that ignores it, take the whole
process group with it, and be a no-op on a child that already exited.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time

from kiss.core.processes import (
    kill_process_tree,
    pid_alive,
    popen_process_group,
    terminate_process_group,
)
from kiss.tests.conftest import posix_only

pytestmark = posix_only("SIGTERM / SIGKILL semantics and bash traps")


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid)


def test_child_honouring_sigterm_is_reaped() -> None:
    proc = popen_process_group([sys.executable, "-c", "import time; time.sleep(60)"])
    started = time.monotonic()
    terminate_process_group(proc)
    assert proc.returncode is not None
    assert time.monotonic() - started < 4


def test_child_ignoring_sigterm_is_killed_with_its_group() -> None:
    # The shell ignores SIGTERM and so does its sleep grandchild; only
    # the SIGKILL round, sent to the whole group, ends both.
    proc = popen_process_group(
        ["bash", "-c", "trap '' TERM; sleep 60 & echo $!; wait"],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    grandchild = int(proc.stdout.readline().strip())
    assert pid_alive(grandchild)
    terminate_process_group(proc, grace=0.5)
    assert proc.returncode is not None
    assert _wait_gone(grandchild)


def test_descendant_ignoring_sigterm_is_killed_after_leader_exits() -> None:
    # Mixed disposition: the shell leader dies of SIGTERM at once, its
    # Python grandchild ignores it.  Escalation must look at the group,
    # not just the leader, or the grandchild lives on for its full 60 s.
    code = (
        "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(os.getpid(), flush=True); time.sleep(60)"
    )
    proc = popen_process_group(
        ["bash", "-c", '"$1" -c "$2" & wait', "mixed", sys.executable, code],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    try:
        grandchild = int(proc.stdout.readline().strip())  # printed after SIG_IGN is set
        assert pid_alive(grandchild)
        started = time.monotonic()
        terminate_process_group(proc, grace=0.5)
        assert proc.returncode == -signal.SIGTERM
        assert time.monotonic() - started < 4
        assert _wait_gone(grandchild)
    finally:
        # On failure the grandchild still sleeps in the (still valid) group.
        kill_process_tree(proc)
        proc.wait(timeout=5)
        proc.stdout.close()


def test_already_exited_child_is_a_no_op() -> None:
    proc = popen_process_group([sys.executable, "-c", "pass"])
    proc.wait()
    terminate_process_group(proc)
    assert proc.returncode == 0
