# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``virtual_display`` and an Xvfb wrapper that exits before reporting a display.

``_read_display_number`` watches the wrapper without reaping it and
returns ``None`` as soon as it has exited.  An exited wrapper does not
mean an empty process group: killed from outside it could not take its
Xvfb along, and what Xvfb forked lives on in the group either way.  The
leader being a zombie still, the group id cannot be anyone else's, so
``virtual_display`` ends the group and only then reaps the wrapper
(audit S17/F2).  Until then the wrapper was reaped at the first look and
its group left alone, a reaped leader's group id being possibly a
stranger's, and this test required the ``sleep`` the fake Xvfb leaves
behind in the wrapper's process group to survive; it must now be gone
when ``virtual_display`` returns.  The test also pins the branch around
it: an Xvfb that dies at once yields no display, promptly, with the
wrapper reaped and the module state clean.
The companion branch (a wrapper still alive at the timeout or at a Stop
is terminated) is ``test_web_stealth.py::
test_stop_during_xvfb_startup_leaves_no_wrapper_behind``.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import web_stealth

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="virtual_display only runs Xvfb on Linux",
)


def _runs_the_stray_sleep(pid: int) -> bool:
    """Whether *pid* runs the fake Xvfb's ``sleep 30``: not gone, a zombie, or a stranger."""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes() == b"sleep\x0030\x00"
    except OSError:
        return False


def test_an_xvfb_that_exits_at_once_yields_no_display_and_no_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_stealth.stop_virtual_display()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "Xvfb"
    # The stray sleep stays in the wrapper's process group (a
    # non-interactive sh has no job control) and inherits the displayfd
    # write end, so ``_read_display_number`` can only return through its
    # look at the wrapper, i.e. once the wrapper has exited.
    pid_file = tmp_path / "stray.pid"
    fake.write_text(f"#!/bin/sh\nsleep 30 &\necho $! > '{pid_file}'\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    started = time.monotonic()
    try:
        assert web_stealth.virtual_display() is None
        elapsed = time.monotonic() - started
        assert elapsed < web_stealth._XVFB_START_TIMEOUT_SECS, f"waited {elapsed:.1f}s"
        assert web_stealth._XVFB_PROC is None and web_stealth._XVFB_DISPLAY is None
        # The wrapper shell was reaped: no child of ours still runs the fake.
        # The pattern is the wrapper's own ``sh -c`` text, anchored: a bare
        # ``xvfb`` also matched any forked-but-not-exec'd child of this
        # process (its command line is pytest's, which names this test
        # file) and failed the assertion under a parallel full-suite run.
        children = subprocess.run(
            ["pgrep", "-P", str(os.getpid()), "-f", "^sh -c Xvfb -displayfd "],
            capture_output=True,
            text=True,
        ).stdout.split()
        assert children == [], f"wrapper shells left behind: {children}"
        # The exited wrapper's group was ended while its leader, not yet
        # reaped, still reserved the group id: the sleep it left behind is gone.
        stray = int(pid_file.read_text())
        assert not _runs_the_stray_sleep(stray), "the exited wrapper's group was left running"
    finally:
        # On failure: no stray left behind.
        with contextlib.suppress(OSError, ValueError):
            stray = int(pid_file.read_text())
            if _runs_the_stray_sleep(stray):
                os.kill(stray, signal.SIGKILL)
