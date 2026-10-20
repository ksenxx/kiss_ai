# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-20, finding C2-01: a slow tunnel restart must not starve the watchdog tick.

The watchdog used to await :meth:`RemoteAccessServer._check_and_restart_tunnel`
inline, first thing in its tick, ahead of the client pings that end
it.  A restart spawns ``cloudflared`` and waits up to 30 s for its URL,
so one slow restart pushed the next ``heartbeat`` frame past the 45 s
after which every idle page drops its socket and reconnects, each
reconnect a full ``ready``, while the daemon was already busy.  The
restart is now a single-flight background task of the tick.

A real TLS daemon with one browser peer; the tunnel subprocess has died
and the ``cloudflared`` on PATH is a fake that never names a URL and
takes seconds to exit, so the restart takes seconds.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import kiss.server.web_server as ws
from kiss.tests.server.test_f9_web_server_audit import _LoopbackServerCase
from kiss.tests.server.test_tunnel_grace_and_backoff import (
    _FAKE_CLOUDFLARED_IS_A_SHELL_SCRIPT,
    _write_fake_cloudflared,
)

FAKE_RESTART_SECONDS = 4.0
"""How long the fake ``cloudflared`` lives: the restart takes at least this long."""


@_FAKE_CLOUDFLARED_IS_A_SHELL_SCRIPT
class TestHungTunnelRestartKeepsTheHeartbeat(_LoopbackServerCase):
    """Pings go out on every tick while a tunnel restart is in flight."""

    password = "watchdog-pw"

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self._fake_dir = tempfile.mkdtemp(prefix="kiss-fake-cloudflared-")
        _write_fake_cloudflared(
            Path(self._fake_dir),
            [],
            exit_code=1,
            sleep_after=FAKE_RESTART_SECONDS,
        )
        self._saved_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self._fake_dir}{os.pathsep}{self._saved_path}"
        self._saved_interval = ws.TUNNEL_CHECK_INTERVAL
        ws.TUNNEL_CHECK_INTERVAL = 0.5

    async def asyncTearDown(self) -> None:
        ws.TUNNEL_CHECK_INTERVAL = self._saved_interval
        os.environ["PATH"] = self._saved_path
        await super().asyncTearDown()

    def _tunnel_has_died(self) -> None:
        """Put the server in tunnel mode with a tunnel subprocess that has exited."""
        proc = subprocess.Popen(["sh", "-c", "exit 1"], stdout=subprocess.DEVNULL, text=True)
        proc.wait()
        self.server.use_tunnel = True
        self.server._tunnel_proc = proc
        self.server._tunnel_metrics_port = 1
        self.server._tunnel_started_at = time.monotonic() - 600

    async def test_heartbeat_arrives_while_the_restart_runs(self) -> None:
        client, _endpoint = await self._connect()
        self._tunnel_has_died()
        started = time.monotonic()
        self.server._watchdog_task = asyncio.create_task(self.server._watchdog())
        # The first tick (0.5 s) starts the restart; its ping must not
        # wait the seconds the restart takes.
        frame = await self._recv(client)
        self.assertEqual(frame, {"type": "heartbeat"})
        self.assertLess(time.monotonic() - started, FAKE_RESTART_SECONDS / 2)
        # The restart is in flight, on its own task, and ends with the
        # fake's failure recorded; a tick meanwhile does not start a second one.
        task = self.server._tunnel_check_task
        assert task is not None
        self.assertFalse(task.done())
        frame = await self._recv(client)
        self.assertEqual(frame, {"type": "heartbeat"})
        self.assertIs(self.server._tunnel_check_task, task)
        await asyncio.wait_for(task, FAKE_RESTART_SECONDS + 30)
        self.assertEqual(self.server._tunnel_failure_count, 1)
        self.assertGreater(self.server._tunnel_next_retry, time.monotonic())
        # ``stop_async`` cancels the restart task along with the watchdog.
        await self.server.stop_async()
        self.stopped = True
        self.assertIsNone(self.server._tunnel_check_task)
        self.assertIsNone(self.server._watchdog_task)


if __name__ == "__main__":
    unittest.main()
