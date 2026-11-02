# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A timed-out default-browser probe leaves no descendant behind (audit 2026-11-01, C5 F2).

``default_browser._run`` bounds ``xdg-settings`` / ``xdg-mime`` /
``osascript`` by ``_PROBE_TIMEOUT``.  With a plain ``subprocess.run`` the
timeout killed only the direct child, so a helper the probe had spawned
lived on after ``_run`` returned ``""``.  The probe here is a Python
parent that starts a sleeping grandchild, records its pid and sleeps past
the timeout; once ``_run`` gives up the grandchild must be gone too.
Linux only: liveness is read from ``/proc``.
"""

import contextlib
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from kiss.core import default_browser

PARENT_THAT_SPAWNS_A_SLEEPER = (
    "import pathlib, subprocess, sys, time; "
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
    "time.sleep(60)"
)


def _running(pid: int) -> bool:
    """True while *pid* is a live, non-zombie process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_timed_out_probe_kills_its_descendants(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    started = time.monotonic()
    result = default_browser._run(
        [sys.executable, "-c", PARENT_THAT_SPAWNS_A_SLEEPER, str(pid_file)]
    )
    elapsed = time.monotonic() - started
    grandchild = int(pid_file.read_text(encoding="utf-8"))
    try:
        assert result == ""
        assert default_browser._PROBE_TIMEOUT <= elapsed < default_browser._PROBE_TIMEOUT + 15
        deadline = time.monotonic() + 5
        while _running(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(grandchild), f"grandchild {grandchild} survived the probe timeout"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(grandchild, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(grandchild, 0)
