# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cloudflared cleanup paths (sweep 10/30).

* ``_terminate_orphan_cloudflared`` given a pidfile that names a dead
  process only removes the stale pidfile.  It used to hand the dead pid
  to the identity check, which ran ``ps`` and logged the pid as
  "recycled".
* ``_start_named_tunnel`` that reads no registration line from a
  cloudflared that keeps running terminates it, as the quick-tunnel
  path does.  Left running, the tunnel could register later and pass
  every watchdog health check while only the local URL was ever
  advertised.  The real trigger is the 30 s stderr read timeout; here
  the fake closes its stderr and keeps running, which makes
  ``_read_url_from_stderr`` return the same ``None`` at once.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kiss.server import web_server as ws
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import install_fake_cloudflared, posix_only

# Records its pid, prints one non-registration line, closes stderr and
# keeps running.
SILENT_CLOUDFLARED = """
import os, sys, time
with open({pidfile!r}, "w") as f:
    f.write(str(os.getpid()))
sys.stderr.write("INF Starting tunnel tunnelID=abc\\n")
sys.stderr.flush()
os.close(2)
time.sleep(60)
"""


class _PidfileTestBase(unittest.TestCase):
    """A private temp dir holding this test's cloudflared pidfile."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._old_pidfile = ws._CLOUDFLARED_PIDFILE
        ws._CLOUDFLARED_PIDFILE = self.dir / "cloudflared.pid"

    def tearDown(self) -> None:
        ws._CLOUDFLARED_PIDFILE = self._old_pidfile
        self._tmp.cleanup()


class TestOrphanPidfileOfADeadProcess(_PidfileTestBase):
    """A pidfile left by a cloudflared that already died is just removed."""

    def test_dead_pid_is_not_reported_as_recycled(self) -> None:
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        ws._save_cloudflared_pidfile(dead.pid, 45678, "https://old.trycloudflare.com")
        self.assertTrue(ws._cloudflared_pidfile().exists())

        with self.assertNoLogs(ws.logger, level="INFO"):
            ws._terminate_orphan_cloudflared()

        self.assertFalse(ws._cloudflared_pidfile().exists())


@posix_only("the fake cloudflared closes its stderr file descriptor")
class TestNamedTunnelThatNeverRegisters(_PidfileTestBase):
    """A named tunnel whose cloudflared never registers is not left running."""

    def setUp(self) -> None:
        super().setUp()
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.dir}{os.pathsep}{self._old_path}"

    def tearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        super().tearDown()

    def test_live_cloudflared_without_registration_is_terminated(self) -> None:
        pidfile = self.dir / "fake.pid"
        install_fake_cloudflared(self.dir, SILENT_CLOUDFLARED.format(pidfile=str(pidfile)))
        srv = RemoteAccessServer(
            use_tunnel=False,
            tunnel_token="dummy-token",
            tunnel_url="https://kiss.example.com",
        )
        try:
            self.assertIsNone(srv._start_named_tunnel())
            self.assertIsNone(srv._tunnel_proc, "the silent cloudflared was left running")
            self.assertFalse(ws._is_pid_alive(int(pidfile.read_text())))
        finally:
            srv._stop_tunnel()


if __name__ == "__main__":
    unittest.main()
