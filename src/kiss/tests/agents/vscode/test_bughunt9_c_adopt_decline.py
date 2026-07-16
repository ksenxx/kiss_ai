# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for cloudflared adoption decline handling.

Two real bugs in ``_try_adopt_existing_cloudflared``:

1. A cloudflared whose adoption is declined (confirmed-unhealthy or no
   discoverable URL) was left running forever — an orphan process leak,
   since the caller then spawns a fresh cloudflared (rotating the
   public URL) while the old one keeps a metrics port bound.
2. ``_probe_tunnel_ready`` is 3-valued (``None`` = "no information",
   e.g. metrics endpoint slow to bind after wake) but the adoption path
   treated ``None`` like ``False`` and declined immediately, needlessly
   rotating the public tunnel URL.

These tests use a real HTTP metrics server, a real subprocess standing
in for cloudflared, and a real pidfile (module-constant monkeypatch,
same pattern as the other web_server tests).  No mocks.
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
    """Serves /ready and /quicktunnel with per-server scripted replies."""

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        srv = self.server
        if self.path.startswith("/ready"):
            replies: list[tuple[int, str]] = srv.ready_replies  # type: ignore[attr-defined]
            status, body = replies.pop(0) if len(replies) > 1 else replies[0]
        elif self.path.startswith("/quicktunnel"):
            status, body = 200, json.dumps(
                {"hostname": "adopted.trycloudflare.com"},
            )
        else:
            status, body = 404, ""
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


def _start_metrics_server(
    ready_replies: list[tuple[int, str]],
) -> tuple[http.server.HTTPServer, int]:
    """Start a real metrics HTTP server on a free port; return (srv, port)."""
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _MetricsHandler)
    httpd.ready_replies = ready_replies  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


class TestAdoptDecline(unittest.TestCase):
    """Declined adoption must terminate the old cloudflared; None re-probes."""

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

    def _spawn_fake_cloudflared(self) -> subprocess.Popen[bytes]:
        """Spawn a real long-lived subprocess standing in for cloudflared.

        Exec'd through a symlink named ``cloudflared`` so the process
        presents the same ``ps -o comm=`` identity a real cloudflared
        binary does — the decline path verifies identity before
        signalling (stale-pidfile PID-reuse protection).
        """
        link = Path(self._tmp.name) / "cloudflared"
        if not link.exists():
            os.symlink(sys.executable, link)
        proc = subprocess.Popen(
            [str(link), "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        return proc

    def _write_pidfile(self, pid: int, metrics_port: int) -> None:
        ws._cloudflared_pidfile().write_text(json.dumps({
            "pid": pid,
            "metrics_port": metrics_port,
            "url": "https://saved.trycloudflare.com",
        }))

    def test_declined_unhealthy_cloudflared_is_terminated(self) -> None:
        """Confirmed-unhealthy (readyConnections=0) -> decline AND terminate."""
        httpd, port = _start_metrics_server(
            [(200, json.dumps({"readyConnections": 0}))],
        )
        self._httpds.append(httpd)
        proc = self._spawn_fake_cloudflared()
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

    def test_none_probe_reprobes_then_adopts(self) -> None:
        """Transient no-information probes re-probe; adoption succeeds."""
        # First two /ready replies are garbage (probe -> None), then the
        # endpoint reports a healthy tunnel.  Pre-fix the first None
        # probe declined adoption outright.
        httpd, port = _start_metrics_server([
            (503, "not json"),
            (503, "not json"),
            (200, json.dumps({"readyConnections": 4})),
        ])
        self._httpds.append(httpd)
        proc = self._spawn_fake_cloudflared()
        self._write_pidfile(proc.pid, port)

        result = ws._try_adopt_existing_cloudflared()

        self.assertIsNotNone(
            result,
            "adoption was declined on a transient no-information probe",
        )
        pid, metrics_port, url = result  # type: ignore[misc]
        self.assertEqual(pid, proc.pid)
        self.assertEqual(metrics_port, port)
        self.assertEqual(url, "https://adopted.trycloudflare.com")
        self.assertIsNone(proc.poll(), "adopted cloudflared must stay alive")


if __name__ == "__main__":
    unittest.main()
