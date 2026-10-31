# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A SIGTERM during the startup wait for ``remote_password`` ends the daemon at once.

With the tunnel on and no password saved yet, ``start()`` waits up to
30 s for one to appear before deciding about the tunnel.  The wait used
to run in one executor thread: a SIGTERM (a supervisor restart, ``systemctl
stop``) cancelled the startup coroutine, but ``asyncio.run`` joins its
executor threads on the way out, so the process lived on until the
30 s poll ended (or the 30 s shutdown failsafe force-exited it).

The daemon runs in a child process with its own empty ``KISS_HOME``
(no password) and is signalled once its local endpoint file, written
right before the wait, exists.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import find_free_port

DAEMON = """
import sys
from pathlib import Path
from kiss.server.web_server import RemoteAccessServer
tmp = Path(sys.argv[1])
RemoteAccessServer(
    host="127.0.0.1", port=int(sys.argv[2]), use_tunnel=True,
    url_file=tmp / "remote-url.json", local_endpoint_file=tmp / "sorcar-local.json",
    scheduler=False,
).start()
"""


@posix_only("delivers a real SIGTERM with os.kill")
class TestSigtermDuringPasswordWait(unittest.TestCase):
    """The daemon exits promptly when stopped while it waits for a password."""

    def test_sigterm_during_the_password_wait_exits_promptly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            env = dict(os.environ, KISS_HOME=str(home), HOME=tmp)
            with open(Path(tmp) / "daemon.log", "wb") as log:
                proc = subprocess.Popen(
                    [sys.executable, "-c", DAEMON, tmp, str(find_free_port())],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
            try:
                endpoint = Path(tmp) / "sorcar-local.json"
                deadline = time.monotonic() + 60.0
                while not endpoint.exists():
                    self.assertIsNone(proc.poll(), "the daemon exited before the wait")
                    self.assertLess(time.monotonic(), deadline, "no endpoint file")
                    time.sleep(0.05)
                time.sleep(1.0)  # well inside the 30 s wait
                self.assertIsNone(proc.poll())
                signalled = time.monotonic()
                os.kill(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    self.fail("the daemon was still running 15 s after SIGTERM")
                self.assertLess(time.monotonic() - signalled, 10.0)
                self.assertEqual(proc.returncode, 0)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    unittest.main()
