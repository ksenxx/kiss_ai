# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: a second SIGTERM during shutdown must not crash kiss-web.

The remote-access web server installs a SIGTERM handler that raises
``KeyboardInterrupt`` so ``asyncio.run`` in :meth:`RemoteAccessServer.start`
unwinds cleanly into its ``finally`` cleanup.  That cleanup calls
:meth:`RemoteAccessServer._stop_tunnel`, which blocks in
``subprocess.wait`` (an internal ``time.sleep`` loop) while terminating
``cloudflared``.

The original bug: when a *second* SIGTERM arrived during that wait — e.g.
an impatient ``pkill``/supervisor restart loop sending SIGTERM twice in
quick succession — the handler re-raised ``KeyboardInterrupt`` inside the
sleep.  That exception escaped the ``finally`` block uncaught, propagated
out of ``start()`` and ``main()``, and crashed the process with an
unhandled traceback.  Any in-flight agent task was killed abruptly
("process killed").

This test reproduces the exact sequence in a real subprocess: it stands up
a tunnel child that ignores SIGTERM (so ``subprocess.wait`` genuinely
blocks), installs the production signal handlers, enters the same
try/except/finally shutdown structure as ``start()``, and then has the
parent deliver two SIGTERMs in rapid succession.  The driver must shut
down cleanly (exit 0, print the sentinel, emit no traceback).
"""

from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Driver executed in a fresh subprocess.  It mirrors the structure of
# RemoteAccessServer.start(): install the real signal handlers, then run
# the blocking section inside try/except KeyboardInterrupt/finally cleanup.
# A SIGTERM-ignoring child stands in for cloudflared so that
# _stop_tunnel() -> _terminate_tunnel_proc() -> proc.wait(timeout=5)
# blocks long enough for a second SIGTERM to land mid-cleanup.
_DRIVER = r"""
import os, signal, subprocess, sys, time
from pathlib import Path

from kiss.server.web_server import RemoteAccessServer

url_file = sys.argv[1]
uds_path = sys.argv[2]
ready_file = sys.argv[3]

server = RemoteAccessServer(
    use_tunnel=False, url_file=url_file, uds_path=uds_path,
)

# A child that ignores SIGTERM and sleeps, standing in for cloudflared.
# proc.terminate() (SIGTERM) is ignored, so proc.wait(timeout=5) inside
# _terminate_tunnel_proc must block for the full grace period -- the
# window during which the second SIGTERM is delivered.  A Python child
# installing SIG_IGN is used (rather than a shell ``trap``) because the
# shell hands SIGTERM to its foreground ``sleep`` child and exits.  The
# child touches *ready_file* only AFTER installing SIG_IGN; we wait for
# that marker before announcing READY so the first SIGTERM cannot race
# the child's handler installation (which would let it die early and
# make proc.wait() return before the second SIGTERM lands).
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal, time, sys; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').close(); "
        "time.sleep(30)",
        ready_file,
    ],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
server._tunnel_proc = child

deadline = time.monotonic() + 15.0
while not Path(ready_file).exists() and time.monotonic() < deadline:
    time.sleep(0.02)

server._install_signal_handlers()

print("READY", flush=True)
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("FIRST_SIGTERM_CAUGHT", flush=True)
finally:
    server._stop_tunnel()
print("CLEAN_EXIT", flush=True)
sys.exit(0)
"""


class TestDoubleSigtermShutdown(unittest.TestCase):
    """``kiss-web`` survives two SIGTERMs delivered during shutdown."""

    def test_second_sigterm_during_cleanup_does_not_crash(self) -> None:
        """A SIGTERM landing mid-cleanup is ignored; shutdown stays clean."""
        with tempfile.TemporaryDirectory() as tmp:
            url_file = str(Path(tmp) / "remote-url.json")
            uds_path = str(Path(tmp) / "sorcar.sock")
            ready_file = str(Path(tmp) / "tunnel-child-ready")
            driver_path = Path(tmp) / "driver.py"
            driver_path.write_text(_DRIVER)

            proc = subprocess.Popen(
                [
                    sys.executable, str(driver_path),
                    url_file, uds_path, ready_file,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # New process group so we signal only the driver, not
                # the pytest runner.
                start_new_session=True,
            )

            # Wait for the driver to finish setup and print READY.
            ready = self._wait_for_line(proc, "READY", timeout=30.0)
            self.assertTrue(ready, "driver never reached READY")

            # First SIGTERM: unwinds the blocking loop into the finally
            # cleanup, where _stop_tunnel() begins blocking in
            # subprocess.wait(timeout=5).
            proc.send_signal(signal.SIGTERM)
            # Give the driver a beat to enter the cleanup / proc.wait.
            time.sleep(0.5)
            # Second SIGTERM: lands while subprocess.wait is sleeping.
            # Pre-fix this re-raised KeyboardInterrupt and crashed the
            # process; post-fix it is ignored.
            proc.send_signal(signal.SIGTERM)

            try:
                stdout, stderr = proc.communicate(timeout=30.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                self.fail(
                    f"driver did not exit after double SIGTERM\n"
                    f"stdout={stdout!r}\nstderr={stderr!r}"
                )

            self.assertEqual(
                proc.returncode,
                0,
                f"driver exited uncleanly (rc={proc.returncode})\n"
                f"stdout={stdout!r}\nstderr={stderr!r}",
            )
            self.assertIn(
                "CLEAN_EXIT",
                stdout,
                f"driver did not complete cleanup\nstderr={stderr!r}",
            )
            self.assertNotIn(
                "Traceback",
                stderr,
                f"shutdown produced an unhandled traceback\nstderr={stderr!r}",
            )
            self.assertNotIn("KeyboardInterrupt", stderr)

    @staticmethod
    def _wait_for_line(
        proc: subprocess.Popen[str], needle: str, timeout: float,
    ) -> bool:
        """Block until *proc* prints a stdout line containing *needle*.

        Args:
            proc: The running subprocess (stdout must be a text pipe).
            needle: Substring to look for in a single output line.
            timeout: Maximum seconds to wait.

        Returns:
            True if a matching line appeared before the timeout.
        """
        deadline = time.monotonic() + timeout
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    return False
                continue
            if needle in line:
                return True
        return False


if __name__ == "__main__":
    unittest.main()
