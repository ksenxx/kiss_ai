# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop reaches a task thread waiting for ``WebUseTool._profile_lock``.

``_launch_browser`` takes ``<profile>.lock`` on the task thread.  An
unbounded ``exclusive_file_lock`` sits in one kernel ``flock(LOCK_EX)``
call, which neither the cooperative stop nor the daemon's forced Stop
(``PyThreadState_SetAsyncExc``) can interrupt: a Stop injected while
another kiss process held the lock landed only when that holder let go
(audit 10-17, F1).  With ``timeout=_PROFILE_LOCK_TIMEOUT_SECS`` the wait
polls the non-blocking lock in Python, where the injection lands within
one poll interval.

Both tests use a real second OS process holding the real lock file; no
mocks, patches or fakes.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.conftest import posix_only

pytestmark = posix_only("fcntl.flock and PyThreadState_SetAsyncExc on a thread")

_HOLDER = """
import fcntl, sys, time
fd = open(sys.argv[1], "w")
fcntl.flock(fd, fcntl.LOCK_EX)
print("held", flush=True)
time.sleep(float(sys.argv[2]))
"""


def _hold_lock_in_other_process(lock_path: Path, seconds: float) -> subprocess.Popen[str]:
    """Start a process that holds *lock_path* for *seconds*; return once it does."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(lock_path), str(seconds)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "held"
    return proc


def _wait_for_lock(tool: WebUseTool, result: dict[str, float | str]) -> None:
    """Enter ``tool._profile_lock()``; record how long it took and how it ended."""
    result["tid"] = threading.get_ident()
    started = time.monotonic()
    try:
        with tool._profile_lock():
            result["outcome"] = "acquired"
    except KeyboardInterrupt:
        result["outcome"] = "interrupted"
    result["elapsed"] = time.monotonic() - started


def test_stop_lands_while_other_process_holds_profile_lock(tmp_path) -> None:
    """An injected Stop interrupts the wait long before the holder releases."""
    profile = tmp_path / "browser_profile"
    holder = _hold_lock_in_other_process(Path(f"{profile}.lock"), 8.0)
    try:
        tool = WebUseTool(user_data_dir=str(profile), headless=True)
        result: dict[str, float | str] = {}
        waiter = threading.Thread(target=_wait_for_lock, args=(tool, result))
        waiter.start()
        time.sleep(0.5)
        assert inject_async_exception(int(result["tid"]), KeyboardInterrupt) == 1
        waiter.join(timeout=3.0)
        assert not waiter.is_alive(), "Stop did not reach the thread waiting for the lock"
        assert result["outcome"] == "interrupted"
        assert float(result["elapsed"]) < 2.0, result
    finally:
        holder.kill()
        holder.wait()


def test_waiter_gets_profile_lock_once_holder_releases(tmp_path) -> None:
    """The bounded wait still admits the waiter after a legitimate hold."""
    profile = tmp_path / "browser_profile"
    holder = _hold_lock_in_other_process(Path(f"{profile}.lock"), 1.0)
    try:
        tool = WebUseTool(user_data_dir=str(profile), headless=True)
        result: dict[str, float | str] = {}
        _wait_for_lock(tool, result)
        assert result["outcome"] == "acquired"
        assert 0.5 < float(result["elapsed"]) < 5.0, result
    finally:
        holder.wait(timeout=10)
