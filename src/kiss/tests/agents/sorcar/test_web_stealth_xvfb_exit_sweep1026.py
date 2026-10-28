# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``virtual_display`` and an Xvfb wrapper that exits before reporting a display.

``_read_display_number`` ``poll()``s the wrapper and returns ``None`` as
soon as it has exited; the wrapper is then reaped and its process group
gone (it kills its Xvfb before exiting), so ``virtual_display`` must not
signal that group: :func:`terminate_process_group` documents that a
caller which already saw the child exit skips it, since the pgid may
belong to an unrelated process by now.  The misdirected signal is made
observable without pid reuse: the fake Xvfb leaves a ``sleep`` behind in
the wrapper's process group before exiting, so a group signal sent after
the wrapper's exit would kill that sleep, while the guard leaves it
alive.  The test also pins the branch around it: an Xvfb that dies at
once yields no display, promptly, with the wrapper reaped and the module
state clean.
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
from kiss.core.processes import pid_alive

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="virtual_display only runs Xvfb on Linux",
)


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
    # ``poll()``, i.e. once the wrapper is reaped.
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
        # The reaped wrapper's group was not signalled: the sleep it left
        # behind (a stand-in for whoever owns that pgid by now) survived.
        stray = int(pid_file.read_text())
        assert pid_alive(stray), "virtual_display signalled the exited wrapper's process group"
    finally:
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
