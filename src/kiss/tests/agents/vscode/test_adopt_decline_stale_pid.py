# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: declined cloudflared adoption must never kill a reused PID.

Commit 3fe9b33d added ``_terminate_declined_cloudflared`` so a
declined-but-alive cloudflared is not orphaned.  But the pidfile
records only a bare integer PID — if kiss-web (or the machine) died
without unlinking the pidfile and the OS later recycled that PID for
an UNRELATED process, the decline path would SIGTERM (and after ~2s
SIGKILL) an innocent process.  The pre-commit code never signalled the
recorded PID, so this was a regression.

The fix verifies the process behind the PID still looks like a
cloudflared binary (``ps -o comm=`` basename) before sending any
signal; a mismatch only unlinks the stale pidfile.

These tests use a real subprocess, a real HTTP metrics server, and a
real pidfile.  No mocks.
"""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from kiss.agents.vscode import web_server as ws


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    """Serves /ready reporting a confirmed-unhealthy tunnel."""

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path.startswith("/ready"):
            body = json.dumps({"readyConnections": 0})
        else:
            body = ""
        data = body.encode()
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


class TestAdoptDeclineStalePid(unittest.TestCase):
    """A reused (non-cloudflared) PID in the pidfile must not be killed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old_pidfile = ws._CLOUDFLARED_PIDFILE
        ws._CLOUDFLARED_PIDFILE = Path(self._tmp.name) / "cloudflared.pid"
        self._procs: list[subprocess.Popen[bytes]] = []
        self._httpds: list[http.server.HTTPServer] = []

    def tearDown(self) -> None:
        ws._CLOUDFLARED_PIDFILE = self._old_pidfile
        for p in self._procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        for h in self._httpds:
            h.shutdown()
        self._tmp.cleanup()

    def _start_unhealthy_metrics_server(self) -> int:
        """Start a real metrics server reporting readyConnections=0."""
        httpd = http.server.HTTPServer(("127.0.0.1", 0), _MetricsHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._httpds.append(httpd)
        return httpd.server_address[1]

    def _write_pidfile(self, pid: int, metrics_port: int) -> None:
        ws._cloudflared_pidfile().write_text(json.dumps({
            "pid": pid,
            "metrics_port": metrics_port,
            "url": "https://saved.trycloudflare.com",
        }))

    def test_unrelated_process_with_reused_pid_survives(self) -> None:
        """Decline path must not signal a PID that is not cloudflared."""
        port = self._start_unhealthy_metrics_server()
        # A real, long-lived, UNRELATED process (plays the role of
        # whatever the OS gave the recycled PID to — a user's editor,
        # a build, anything).  Its executable is python, not
        # cloudflared.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        self._write_pidfile(proc.pid, port)

        result = ws._try_adopt_existing_cloudflared()

        self.assertIsNone(result, "unhealthy tunnel must not be adopted")
        # Give any wrongly-sent SIGTERM time to land.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.assertIsNone(
                proc.poll(),
                "decline path killed an unrelated process whose PID was "
                "recycled from a stale cloudflared pidfile",
            )
            time.sleep(0.05)
        self.assertFalse(
            ws._cloudflared_pidfile().exists(),
            "stale (non-cloudflared) pidfile must still be unlinked",
        )

    def test_similarly_named_process_survives(self) -> None:
        """An exact-name check: ``cloudflared-helper`` must NOT be killed."""
        port = self._start_unhealthy_metrics_server()
        link = Path(self._tmp.name) / "cloudflared-helper"
        os.symlink(sys.executable, link)
        proc = subprocess.Popen(
            [str(link), "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        self._write_pidfile(proc.pid, port)

        result = ws._try_adopt_existing_cloudflared()

        self.assertIsNone(result)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.assertIsNone(
                proc.poll(),
                "decline path killed a similarly-named but unrelated "
                "process (prefix match instead of exact match)",
            )
            time.sleep(0.05)

    def test_real_cloudflared_lookalike_is_terminated(self) -> None:
        """A genuine cloudflared process on the decline path IS terminated."""
        port = self._start_unhealthy_metrics_server()
        # Exec python through a symlink named ``cloudflared`` so the
        # process's comm / argv[0] basename is ``cloudflared`` — the
        # same identity a real cloudflared binary presents.
        link = Path(self._tmp.name) / "cloudflared"
        os.symlink(sys.executable, link)
        proc = subprocess.Popen(
            [str(link), "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        self._write_pidfile(proc.pid, port)

        result = ws._try_adopt_existing_cloudflared()

        self.assertIsNone(result)
        deadline = time.monotonic() + 4.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(
            proc.poll(),
            "declined cloudflared was left running (orphan process leak)",
        )
        self.assertFalse(
            ws._cloudflared_pidfile().exists(),
            "pidfile of the declined cloudflared was not unlinked",
        )


if __name__ == "__main__":
    unittest.main()
