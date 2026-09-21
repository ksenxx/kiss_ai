# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The fake ``cloudflared`` must pass the product's process-identity check.

``web_server._looks_like_cloudflared`` guards every signal sent to a
pidfile-recorded PID by matching the process's executable basename
against ``cloudflared``.  The test fake therefore has to *be* a process
named ``cloudflared`` on every platform, not merely a script called
that: on macOS ``ps -o comm=`` reports the interpreter of a shebang
script, which made the adoption/decline tests spawn a fake the product
refused to recognise.  This exercises ``install_fake_cloudflared``
end to end: spawned by bare name from ``PATH`` exactly as the product
does, the fake is recognised, receives its arguments, writes to stderr
and propagates its exit code.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from kiss.server import web_server as ws
from kiss.tests.conftest import install_fake_cloudflared


class TestFakeCloudflaredIdentity(unittest.TestCase):
    """``install_fake_cloudflared`` produces a process the product recognises."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.env = dict(os.environ, PATH=f"{self.dir}{os.pathsep}{os.environ.get('PATH', '')}")

    def test_long_lived_fake_is_recognised_by_bare_name_spawn(self) -> None:
        """A running fake matches ``_looks_like_cloudflared``; a plain python does not."""
        marker = self.dir / "started"
        install_fake_cloudflared(
            self.dir,
            "import pathlib, sys, time\n"
            f"pathlib.Path({str(marker)!r}).write_text(' '.join(sys.argv[1:]))\n"
            "time.sleep(60)\n",
        )
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", "http://localhost:1"],
            env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.02)
        self.assertEqual(marker.read_text(), "tunnel --url http://localhost:1")
        self.assertTrue(ws._looks_like_cloudflared(proc.pid))
        self.assertFalse(ws._looks_like_cloudflared(os.getpid()), "the test runner is not it")

    def test_fake_propagates_exit_code_and_stderr(self) -> None:
        """``sys.exit(code)`` and stderr output reach the parent as for a real binary."""
        install_fake_cloudflared(
            self.dir,
            "import sys\nsys.stderr.write('ERR rate limited\\n')\nsys.exit(3)\n",
        )
        result = subprocess.run(
            ["cloudflared", "tunnel"], env=self.env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stderr, "ERR rate limited\n")

    def test_reinstall_in_same_directory_replaces_body(self) -> None:
        """Installing twice into one directory runs the newest body."""
        install_fake_cloudflared(self.dir, "import sys\nsys.exit(1)\n")
        fake = install_fake_cloudflared(self.dir, "import sys\nsys.exit(2)\n")
        result = subprocess.run([str(fake)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr, "")
