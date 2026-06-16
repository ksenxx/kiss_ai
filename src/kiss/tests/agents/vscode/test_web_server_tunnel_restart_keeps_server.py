# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Restarting the cloudflared tunnel must never restart the WSS server.

When :mod:`kiss.agents.vscode.web_server` restarts the ``cloudflared``
tunnel — whether because the tunnel subprocess died, because the
Cloudflare edge deregistered it (``readyConnections == 0``), or because
:meth:`RemoteAccessServer._restart_tunnel_url` is invoked directly — it
must only respawn ``cloudflared``.  The long-lived HTTPS/WSS listener
(``_ws_server``) that browsers and the VS Code extension are connected
to MUST stay up: tearing it down would drop every in-flight chat
session and force every client to reconnect for a problem that lives
entirely on the public-tunnel side.

These are real integration tests: a genuine ``websockets`` WSS server
is bound on a free localhost port (no tunnel, so no real
``cloudflared`` is needed), each tunnel-restart path is driven, and the
same server object is asserted to still be serving afterwards.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.agents.vscode.web_server as ws_mod
from kiss.agents.vscode.web_server import (
    _TUNNEL_STARTUP_GRACE,
    _TUNNEL_UNHEALTHY_LIMIT_QUICK,
    RemoteAccessServer,
)


def _free_port() -> int:
    """Return a currently-free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestTunnelRestartKeepsServer(IsolatedAsyncioTestCase):
    """Every tunnel-restart code path must leave ``_ws_server`` serving."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_free_port(),
            use_tunnel=False,
            url_file=tmp / "remote-url.json",
            uds_path=tmp / "sorcar.sock",
        )
        # Bind the real HTTPS/WSS + UDS listeners (no tunnel yet).
        await self.server._setup_server()
        # The watchdog / version-check loops are not needed here and
        # would otherwise run their own checks during the test.
        for task in (
            self.server._watchdog_task,
            self.server._version_check_task,
        ):
            if task is not None:
                task.cancel()

        self._ws_server = self.server._ws_server
        self.assertIsNotNone(self._ws_server)
        self.assertTrue(self._ws_server.is_serving())

        # Now pretend we are running in tunnel mode so the tunnel
        # restart paths are active, but never spawn a real cloudflared.
        self.server.use_tunnel = True
        self._start_calls = 0

        def fake_start_tunnel() -> str:
            self._start_calls += 1
            return f"https://fake-{self._start_calls}.trycloudflare.com"

        self.server._start_tunnel = fake_start_tunnel  # type: ignore[method-assign]

        # Keep ntfy.sh posting (network) out of the test.
        self._orig_post = ws_mod._post_url_to_message_board
        ws_mod._post_url_to_message_board = lambda *_a, **_k: None  # type: ignore[assignment]

        # A non-empty remote_password lets the no-process / dead-process
        # restart branch proceed (it refuses to start a tunnel without
        # one).  Patch the loader so the test does not touch the real
        # ``~/.kiss/config.json``.
        self._orig_load_config = ws_mod.load_config
        ws_mod.load_config = lambda: {"remote_password": "test-pw"}  # type: ignore[assignment]

        self._dummy_procs: list[subprocess.Popen[str]] = []

    async def asyncTearDown(self) -> None:
        ws_mod._post_url_to_message_board = self._orig_post  # type: ignore[assignment]
        ws_mod.load_config = self._orig_load_config  # type: ignore[assignment]
        for proc in self._dummy_procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        await self.server.stop_async()
        self._tmp.cleanup()

    def _spawn_dummy(self) -> subprocess.Popen[str]:
        """Spawn a long-lived dummy process standing in for cloudflared."""
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._dummy_procs.append(proc)
        return proc

    def _assert_server_alive(self) -> None:
        """The original WSS server object is unchanged and still serving."""
        self.assertIs(self.server._ws_server, self._ws_server)
        self.assertTrue(self._ws_server.is_serving())

    async def test_direct_restart_tunnel_url_keeps_server(self) -> None:
        """``_restart_tunnel_url`` respawns the tunnel, not the server."""
        await self.server._restart_tunnel_url()
        self.assertEqual(self._start_calls, 1)
        self.assertEqual(
            self.server._active_url, "https://fake-1.trycloudflare.com",
        )
        self._assert_server_alive()

    async def test_dead_tunnel_process_restart_keeps_server(self) -> None:
        """A dead cloudflared subprocess triggers a tunnel-only restart."""
        proc = self._spawn_dummy()
        proc.terminate()
        proc.wait(timeout=2)
        self.server._tunnel_proc = proc
        self.server._tunnel_metrics_port = _free_port()
        self.server._tunnel_started_at = time.monotonic()

        await self.server._check_and_restart_tunnel()

        # Tunnel was restarted (fresh cloudflared) ...
        self.assertGreaterEqual(self._start_calls, 1)
        # ... but the WSS server was untouched.
        self._assert_server_alive()

    async def test_deregistered_force_restart_keeps_server(self) -> None:
        """A confirmed ``readyConnections == 0`` force-restart spares the server."""
        proc = self._spawn_dummy()
        self.server._tunnel_proc = proc
        self.server._tunnel_metrics_port = _free_port()
        self.server._tunnel_started_at = (
            time.monotonic() - _TUNNEL_STARTUP_GRACE - 1
        )
        self.server._tunnel_unhealthy_ticks = 0

        orig_probe = ws_mod._probe_tunnel_ready
        ws_mod._probe_tunnel_ready = lambda _port: False  # type: ignore[assignment]
        try:
            for _ in range(_TUNNEL_UNHEALTHY_LIMIT_QUICK):
                await self.server._check_and_restart_tunnel()
        finally:
            ws_mod._probe_tunnel_ready = orig_probe  # type: ignore[assignment]

        # The force-restart fired (cloudflared killed + respawned) ...
        self.assertIsNotNone(proc.poll())
        self.assertGreaterEqual(self._start_calls, 1)
        self.assertEqual(self.server._tunnel_force_restart_count, 1)
        # ... and the WSS server stayed up the whole time.
        self._assert_server_alive()
