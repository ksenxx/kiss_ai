# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: the quick-tunnel metrics fallback must query OUR cloudflared.

When ``_start_quick_tunnel`` fails to parse the URL from stderr, it
falls back to the metrics ``/quicktunnel`` endpoint.  Pre-fix it went
straight to ``_discover_tunnel_url_from_metrics``, which scans ALL
running cloudflared processes plus a hardcoded port range
(20240-20259) — and could therefore adopt a FOREIGN tunnel's URL.
Post-fix the just-spawned process's own ``--metrics`` port is queried
first.

A fake ``cloudflared`` (prepended to PATH, same pattern as
``test_named_tunnel_url.py``) parses its ``--metrics`` argument, closes
stderr (so no URL is ever parsed from logs), and serves a real
``/quicktunnel`` endpoint reporting ``own.trycloudflare.com``.  A
second, FOREIGN metrics server on a port in the hardcoded scan range
reports ``foreign.trycloudflare.com``.
"""

from __future__ import annotations

import http.server
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path

import pytest

from kiss.agents.vscode import web_server as ws

_FAKE_CLOUDFLARED = """#!/bin/sh
port=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--metrics" ]; then port="${a##*:}"; fi
  prev="$a"
done
exec python3 -c '
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"hostname": "own.trycloudflare.com"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
' "$port" 2>/dev/null
"""


class _ForeignHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        body = json.dumps({"hostname": "foreign.trycloudflare.com"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class TestQuickTunnelPrefersOwnMetricsPort(unittest.TestCase):
    """The stderr-less fallback must return OUR URL, not a foreign one."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(self._tmp.name)
        script = tmpdir / "cloudflared"
        script.write_text(_FAKE_CLOUDFLARED)
        script.chmod(
            script.stat().st_mode
            | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
        )
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{tmpdir}{os.pathsep}{self._old_path}"
        self._old_pidfile = ws._CLOUDFLARED_PIDFILE
        ws._CLOUDFLARED_PIDFILE = tmpdir / "cloudflared.pid"
        # Foreign metrics server inside the hardcoded scan range.
        self._foreign: http.server.HTTPServer | None = None
        for port in range(20240, 20260):
            try:
                self._foreign = http.server.HTTPServer(
                    ("127.0.0.1", port), _ForeignHandler,
                )
                break
            except OSError:
                continue
        if self._foreign is None:
            self.skipTest("no free port in the 20240-20259 scan range")
        threading.Thread(
            target=self._foreign.serve_forever, daemon=True,
        ).start()

    def tearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        ws._CLOUDFLARED_PIDFILE = self._old_pidfile
        if self._foreign is not None:
            self._foreign.shutdown()
        self._tmp.cleanup()

    @pytest.mark.slow
    def test_fallback_returns_own_url_not_foreign(self) -> None:
        """_start_quick_tunnel adopts OUR /quicktunnel hostname."""
        with tempfile.TemporaryDirectory() as tmp:
            srv = ws.RemoteAccessServer(
                host="127.0.0.1",
                use_tunnel=False,
                url_file=Path(tmp) / "remote-url.json",
                uds_path=Path(tmp) / "sorcar.sock",
            )
            try:
                url = srv._start_quick_tunnel()
            finally:
                srv._stop_tunnel()
        self.assertEqual(
            url,
            "https://own.trycloudflare.com",
            "quick-tunnel fallback adopted a foreign tunnel's URL "
            "instead of querying the just-spawned cloudflared's own "
            "metrics port",
        )


if __name__ == "__main__":
    unittest.main()
