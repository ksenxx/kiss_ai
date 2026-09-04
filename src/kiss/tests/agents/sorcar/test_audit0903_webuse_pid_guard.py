# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Branch coverage for ``web_use_tool._killable_live_pid``.

The predicate is the consolidated arming guard of the browser
watchdogs (audit 0903 redundancy fix: ``_close_browser_only`` and
``_input_hang_watchdog`` previously duplicated it inverted, a drift
hazard for a check whose failure mode is signalling the wrong
process).  Probed here against real OS processes — no doubles: a live
child process, the same child after it exited, this process itself,
and the never-a-single-process pid values ``None``, ``0``, and ``-1``.
"""

from __future__ import annotations

import os
import subprocess
import sys

from kiss.agents.sorcar.web_use_tool import _killable_live_pid


def test_live_foreign_process_is_killable() -> None:
    """A live child process is a valid watchdog target."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    try:
        assert _killable_live_pid(child.pid)
    finally:
        child.kill()
        child.wait(timeout=10)


def test_exited_process_is_not_killable() -> None:
    """A dead pid must not arm a watchdog (PID could be recycled)."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    # The zombie is reaped by wait(); the pid no longer exists.
    assert not _killable_live_pid(child.pid)


def test_own_pid_and_sentinel_pids_are_refused() -> None:
    """``None``, group-addressing pids, and our own pid never arm.

    ``0`` signals the caller's process group and negative values whole
    groups; a corrupted record must never make a watchdog SIGKILL this
    process or its group.
    """
    assert not _killable_live_pid(None)
    assert not _killable_live_pid(0)
    assert not _killable_live_pid(-1)
    assert not _killable_live_pid(os.getpid())
