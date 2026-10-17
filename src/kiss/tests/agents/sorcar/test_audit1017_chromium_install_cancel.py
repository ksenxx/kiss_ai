# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cancelling ``_install_chromium`` ends the installer's whole process tree.

``python -m playwright install chromium`` is only a launcher: it spawns
the Node CLI that performs the download.  Killing and reaping just the
launcher on a Stop or timeout left that CLI (and its download worker)
running and holding the browser-cache lock (audit 10-17, review #3).
The installer now runs in its own process group and the group is killed.

The tests run ``_install_chromium`` against a stand-in launcher (a
Python script in ``tmp_path``) that spawns a sleeping grandchild and
records its pid; no mocks, patches or fakes of the code under test.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.web_use_tool import _install_chromium
from kiss.core.processes import pid_alive
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.conftest import posix_only

pytestmark = posix_only("process groups and PyThreadState_SetAsyncExc on a thread")

# argv[1]: file receiving the grandchild's pid; argv[2]: launcher exit status.
_LAUNCHER = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(sys.argv[1], "w") as f:
    f.write(str(child.pid))
if sys.argv[2] != "wait":
    sys.exit(int(sys.argv[2]))
child.wait()
"""


def _stand_in(tmp_path: Path, mode: str) -> tuple[list[str], Path]:
    """Write the stand-in launcher and return its command and pid file."""
    script = tmp_path / "launcher.py"
    script.write_text(_LAUNCHER)
    pid_file = tmp_path / "grandchild.pid"
    return [sys.executable, str(script), str(pid_file), mode], pid_file


def _grandchild_pid(pid_file: Path) -> int:
    """Wait for the launcher to record its grandchild's pid."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if pid_file.exists() and pid_file.read_text():
            return int(pid_file.read_text())
        time.sleep(0.02)
    raise AssertionError("stand-in launcher never spawned its grandchild")


def _gone(pid: int) -> bool:
    """Return whether *pid* exited, allowing the orphan reaper a few seconds.

    A killed grandchild is re-parented to init and stays a zombie until
    init reaps it; a zombie still answers ``os.kill(pid, 0)``, so the
    ``/proc`` state is consulted too.
    """
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except OSError:
            return True
        time.sleep(0.05)
    return False


def test_timeout_kills_installer_grandchild(tmp_path: Path) -> None:
    """A timed-out installer is killed together with its descendants."""
    command, pid_file = _stand_in(tmp_path, "wait")
    with pytest.raises(subprocess.TimeoutExpired):
        _install_chromium(command, timeout=1.0)
    grandchild = _grandchild_pid(pid_file)
    assert _gone(grandchild), f"grandchild {grandchild} survived the timeout"


def test_stop_kills_installer_grandchild(tmp_path: Path) -> None:
    """A Stop injected into the installing thread ends the whole tree."""
    command, pid_file = _stand_in(tmp_path, "wait")
    outcome: list[BaseException] = []

    def run() -> None:
        try:
            _install_chromium(command, timeout=60.0)
        except BaseException as exc:  # noqa: BLE001 — the Stop is a KeyboardInterrupt
            outcome.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    grandchild = _grandchild_pid(pid_file)
    assert pid_alive(grandchild)
    assert worker.ident is not None
    assert inject_async_exception(worker.ident, KeyboardInterrupt) == 1
    worker.join(10)
    assert not worker.is_alive(), "installer did not return after the Stop"
    assert len(outcome) == 1 and isinstance(outcome[0], KeyboardInterrupt)
    assert _gone(grandchild), f"grandchild {grandchild} survived the Stop"


def test_installer_exit_status(tmp_path: Path) -> None:
    """A clean exit returns; a failing installer raises ``RuntimeError``."""
    command, pid_file = _stand_in(tmp_path, "0")
    _install_chromium(command, timeout=30.0)
    grandchild = _grandchild_pid(pid_file)
    assert pid_alive(grandchild)  # a finished installer's descendants are left alone
    os.kill(grandchild, signal.SIGKILL)
    (tmp_path / "fail").mkdir()
    command, pid_file = _stand_in(tmp_path / "fail", "3")
    with pytest.raises(RuntimeError, match="exited with status 3"):
        _install_chromium(command, timeout=30.0)
    os.kill(_grandchild_pid(pid_file), signal.SIGKILL)
