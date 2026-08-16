# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: no tunnel URL rotation across ``stop_async()``.

``RemoteAccessServer.stop_async()`` is the embedder/test shutdown path.
Unlike ``start()``'s shutdown ``finally`` (which *detaches* the tunnel),
it calls ``_stop_tunnel()``.  A review of the install-restart fix (the
third gpt-5.6-sol finding) flagged that ``_stop_tunnel()`` treats an
ADOPTED cloudflared differently from a self-spawned one, and that this
asymmetry was left unpinned by tests — a future "cleanup" of either
branch could silently start rotating the public URL on shutdown paths
that go through ``stop_async()``.

These tests pin both halves of the contract with real subprocesses and
real servers (no mocks):

1. **Adopted tunnel — the URL must NOT rotate.**  A cloudflared adopted
   from a previous kiss-web is left running by ``stop_async()`` with its
   pidfile intact, so the next daemon re-adopts it and serves the exact
   same public URL.  Killing it (or unlinking the pidfile) would rotate
   a healthy URL on every embedder shutdown — the churn this file
   exists to detect.

2. **Self-spawned tunnel — rotation is deliberate and must be CLEAN.**
   ``stop_async()`` kills a cloudflared this server spawned (embedders
   own their server's full lifecycle and must not leak background
   processes) and unlinks the pidfile, so the next daemon cannot adopt
   a dead pid — or worse, a recycled one — and resurrect a stale URL.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import web_server as ws
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server._ntfy_emulator import unroutable_base_url

_PINNED_URL = "https://stop-async-pinned.trycloudflare.com"


def _find_free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    """Fake cloudflared metrics endpoint: always healthy, one fixed URL."""

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        """Serve ``/ready`` (2 ready connections) and ``/quicktunnel``."""
        if self.path.startswith("/ready"):
            body = json.dumps({"readyConnections": 2})
        elif self.path.startswith("/quicktunnel"):
            body = json.dumps(
                {"hostname": _PINNED_URL.removeprefix("https://")},
            )
        else:
            body = ""
        data = body.encode()
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence per-request logging."""


class TestAdoptedTunnelUrlStableAcrossStopAsync(IsolatedAsyncioTestCase):
    """An adopted cloudflared's URL survives a full stop_async() cycle."""

    async def asyncSetUp(self) -> None:
        """Stand up one healthy fake cloudflared for two daemons to adopt."""
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)

        # Fake cloudflared metrics endpoint (healthy, fixed hostname).
        self._httpd = http.server.HTTPServer(
            ("127.0.0.1", 0), _MetricsHandler,
        )
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        metrics_port = self._httpd.server_address[1]

        # Fake adopted cloudflared: a live process whose argv[0] is
        # literally "cloudflared" (python behind a symlink), exactly
        # like the harness in test_tunnel_survives_install_restart.py.
        link = tmp / "cloudflared"
        os.symlink(sys.executable, link)
        self._cf_proc = subprocess.Popen(
            [str(link), "-c", "import time; time.sleep(300)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Redirect the pidfile the adoption machinery reads/writes.
        self._pidfile = tmp / "cloudflared.pid"
        self._old_pidfile = ws._CLOUDFLARED_PIDFILE
        ws._CLOUDFLARED_PIDFILE = self._pidfile
        ws._save_cloudflared_pidfile(self._cf_proc.pid, metrics_port, None)

        # A remote password is what authorizes the tunnel machinery.
        self._config_snapshot = (
            CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        )
        save_config({"remote_password": "stop-async-url-regr"})

        # Regression guard: if a future change makes stop_async() (or
        # the second start) decline/kill the adopted tunnel, the server
        # falls back to spawning `cloudflared` from PATH.  Put a
        # fail-fast stub FIRST on PATH so that fallback can never spawn
        # a real tunnel on the machine running the tests.
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "cloudflared"
        stub.write_text("#!/bin/sh\nexit 1\n")
        stub.chmod(0o755)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{self._old_path}"

        self._url_file = tmp / "remote-url.json"
        self._uds_path = tmp / "kiss-web-test.sock"

    async def asyncTearDown(self) -> None:
        """Restore globals, reap the fake cloudflared, stop the metrics server."""
        os.environ["PATH"] = self._old_path
        ws._CLOUDFLARED_PIDFILE = self._old_pidfile
        if self._config_snapshot is None:
            CONFIG_PATH.unlink(missing_ok=True)
        else:
            CONFIG_PATH.write_text(self._config_snapshot)
        if self._cf_proc.poll() is None:
            self._cf_proc.kill()
            self._cf_proc.wait()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._tmp.cleanup()

    def _make_server(self) -> RemoteAccessServer:
        """Build a tunnel-enabled server fully isolated to the temp dir."""
        return RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            work_dir=self._tmp.name,
            use_tunnel=True,
            url_file=self._url_file,
            uds_path=self._uds_path,
            ntfy_base_url=unroutable_base_url(),
        )

    async def test_url_identical_across_stop_async_restart_cycle(self) -> None:
        """daemon A adopts → stop_async() → daemon B re-adopts the SAME URL."""
        server_a = self._make_server()
        await server_a.start_async()
        try:
            self.assertEqual(
                server_a._tunnel_adopted_pid, self._cf_proc.pid,
                "first daemon failed to adopt the healthy cloudflared",
            )
            url_a = server_a._active_url
            self.assertEqual(url_a, _PINNED_URL)
        finally:
            await server_a.stop_async()

        # The core of the pinned contract: stop_async() must leave an
        # ADOPTED cloudflared running with its pidfile intact.
        self.assertIsNone(
            self._cf_proc.poll(),
            "stop_async() killed the ADOPTED cloudflared — the next "
            "daemon must mint a fresh URL (tunnel URL rotation)",
        )
        pidfile = json.loads(self._pidfile.read_text())
        self.assertEqual(
            pidfile["pid"], self._cf_proc.pid,
            "stop_async() removed or rewrote the cloudflared pidfile — "
            "the next daemon cannot re-adopt and will rotate the URL",
        )
        self.assertEqual(
            pidfile.get("url"), _PINNED_URL,
            "the adopted URL was not persisted for the next daemon",
        )

        server_b = self._make_server()
        await server_b.start_async()
        try:
            self.assertEqual(
                server_b._tunnel_adopted_pid, self._cf_proc.pid,
                "second daemon did not re-adopt the surviving cloudflared",
            )
            self.assertEqual(
                server_b._active_url, url_a,
                "tunnel URL ROTATED across stop_async(): the second "
                "daemon serves a different public URL",
            )
            saved = json.loads(self._url_file.read_text())
            self.assertEqual(
                saved.get("tunnel"), url_a,
                "the URL file advertises a rotated tunnel URL",
            )
        finally:
            await server_b.stop_async()
        self.assertIsNone(
            self._cf_proc.poll(),
            "the second stop_async() killed the adopted cloudflared",
        )


class TestSpawnedTunnelStopAsyncLeavesNoStaleState(IsolatedAsyncioTestCase):
    """stop_async() on a SELF-SPAWNED tunnel kills it and clears the pidfile."""

    async def asyncSetUp(self) -> None:
        """Start a real tunnel-less server plus an isolated pidfile."""
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self._pidfile = tmp / "cloudflared.pid"
        self._old_pidfile = ws._CLOUDFLARED_PIDFILE
        ws._CLOUDFLARED_PIDFILE = self._pidfile
        self._config_snapshot = (
            CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        )
        save_config({"remote_password": ""})
        self._server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            work_dir=self._tmp.name,
            use_tunnel=False,
            url_file=tmp / "remote-url.json",
            uds_path=tmp / "kiss-web-test.sock",
            ntfy_base_url=unroutable_base_url(),
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        """Restore globals and clean up the temp dir."""
        ws._CLOUDFLARED_PIDFILE = self._old_pidfile
        if self._config_snapshot is None:
            CONFIG_PATH.unlink(missing_ok=True)
        else:
            CONFIG_PATH.write_text(self._config_snapshot)
        self._tmp.cleanup()

    async def test_spawned_tunnel_killed_and_pidfile_cleared(self) -> None:
        """A killed spawned tunnel must not leave a pidfile to mis-adopt.

        The intentional half of the ``stop_async()`` contract: a
        cloudflared THIS server spawned dies with the server.  The
        pidfile must go with it — a stale pidfile would let the next
        daemon "adopt" a dead (or recycled) pid and briefly resurrect a
        URL that no longer routes anywhere.
        """
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(300)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self._server._tunnel_proc = proc
        self._server._tunnel_metrics_port = 19999
        self._server._active_url = "https://doomed.trycloudflare.com"
        ws._save_cloudflared_pidfile(
            proc.pid, 19999, "https://doomed.trycloudflare.com",
        )
        try:
            await self._server.stop_async()
            self.assertIsNotNone(
                proc.poll(),
                "stop_async() leaked the SELF-SPAWNED cloudflared",
            )
            self.assertFalse(
                self._pidfile.exists(),
                "stop_async() left a stale pidfile behind — the next "
                "daemon may adopt a dead/recycled pid and advertise a "
                "URL that routes nowhere",
            )
            self.assertIsNone(
                ws._try_adopt_existing_cloudflared(),
                "a daemon starting after stop_async() adopted the "
                "killed tunnel's stale state",
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    unittest.main()
