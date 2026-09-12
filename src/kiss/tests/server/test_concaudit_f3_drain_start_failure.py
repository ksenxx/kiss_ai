# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_read_url_from_stderr`` must not leave a reader-less live child behind.

Concurrency audit (F3, reviewer R3 finding 2): the drain thread is the
ONLY reader of ``cloudflared``'s stderr pipe.  When ``Thread.start()``
itself raised (``RuntimeError: can't start new thread`` under thread
exhaustion) the helper propagated the error, ``_start_tunnel`` logged
it and returned ``None`` — and the live cloudflared stayed in
``_tunnel_proc`` with nobody reading its pipe.  Once the 64 KiB pipe
buffer filled, cloudflared blocked on its next log write (a Go-wide
logging deadlock), so even the metrics endpoint the watchdog probes
went silent.

The exhaustion is produced for real by lowering ``RLIMIT_NPROC`` to 1
after the child has been spawned (threads count against the limit for
a non-root uid).  No mocks or patches.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import unittest

import pytest

from kiss.server.web_server import (
    _parse_quick_tunnel_url,
    _read_url_from_stderr,
)
from kiss.tests.server.test_concaudit_w6_commit_msg_claim import (
    _thread_start_can_be_starved,
)


def _spawn_long_lived_child() -> subprocess.Popen[str]:
    """Start a child that would live for minutes if nobody stopped it."""
    return subprocess.Popen(
        [
            sys.executable, "-u", "-c",
            "import sys, time\n"
            "sys.stderr.write('INF starting\\n')\n"
            "time.sleep(300)\n",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


class TestDrainStartFailureTerminatesChild(unittest.TestCase):
    """A failed drain-thread spawn kills and reaps the child, then re-raises."""

    def test_child_killed_and_reaped_when_thread_start_fails(self) -> None:
        if not _thread_start_can_be_starved():
            pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
        proc = _spawn_long_lived_child()
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
            try:
                with self.assertRaises(RuntimeError):
                    _read_url_from_stderr(
                        proc, _parse_quick_tunnel_url, timeout=0.2,
                    )
            finally:
                resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))
            self.assertIsNotNone(
                proc.returncode,
                "child must be killed and reaped before the error propagates",
            )
            assert proc.stderr is not None
            self.assertTrue(proc.stderr.closed, "stderr pipe must be released")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
