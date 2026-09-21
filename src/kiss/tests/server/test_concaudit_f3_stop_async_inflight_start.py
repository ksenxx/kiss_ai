# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``stop_async`` must not leak a cloudflared whose start was in flight.

Concurrency audit (F3, reviewer R3 finding 6): the tunnel watchdog
starts cloudflared through ``run_in_executor(self._start_tunnel)``.
``stop_async`` cancels the watchdog *task*, but cancelling an asyncio
future never stops the executor function: ``_stop_tunnel`` then saw
``_tunnel_proc is None`` and returned, and a moment later the executor
published a fresh, live cloudflared into ``_tunnel_proc`` — after the
server had promised it was down — while the cancelled coroutine never
reached its post-await cleanup.

The test runs a real fake ``cloudflared`` (a long-lived child that
records its pid) and calls ``stop_async`` while the spawn is inside its
fail-fast window, i.e. before the process is published.  No mocks.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path

from kiss.core.processes import pid_alive
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import install_fake_cloudflared


class TestStopAsyncKillsInFlightTunnelStart(unittest.IsolatedAsyncioTestCase):
    """A cloudflared spawned after ``_stop_tunnel`` ran must be killed."""

    async def test_no_tunnel_published_or_alive_after_stop(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_dir = Path(tmp.name)
        pid_marker = tmp_dir / "fake-cloudflared.pid"
        install_fake_cloudflared(
            tmp_dir,
            "import os, pathlib, sys, time\n"
            f"pathlib.Path({str(pid_marker)!r}).write_text(str(os.getpid()))\n"
            "sys.stderr.write('INF registering, no url yet\\n')\n"
            "sys.stderr.flush()\n"
            "time.sleep(300)\n",
        )
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{tmp_dir}{os.pathsep}{old_path}"
        self.addCleanup(os.environ.__setitem__, "PATH", old_path)

        server = RemoteAccessServer(
            use_tunnel=False,
            url_file=tmp_dir / "remote-url.json",
            uds_path=tmp_dir / "kiss.sock",
        )
        server._vscode_server.use_private_tab_registry(tmp_dir / "tabs.json")
        server._loop = asyncio.get_running_loop()
        # Stand in for the watchdog tick that (re)starts the tunnel;
        # stop_async cancels this task exactly like the real watchdog.
        server._watchdog_task = asyncio.create_task(
            server._restart_tunnel_url(),
        )
        # Wait until the fake cloudflared is running, i.e. the spawn is
        # inside its fail-fast window and _tunnel_proc is still unset.
        # Poll for parseable *content*, not mere existence: write_text
        # creates the file empty before writing, so an existence check
        # can win the race and read ''.
        deadline = time.monotonic() + 10
        pid_text = ""
        while time.monotonic() < deadline:
            if pid_marker.exists():
                pid_text = pid_marker.read_text().strip()
                if pid_text.isdigit():
                    break
            await asyncio.sleep(0.02)
        self.assertTrue(pid_text.isdigit(), "fake cloudflared never started")
        child_pid = int(pid_text)
        self.addCleanup(self._kill, child_pid)
        self.assertIsNone(server._tunnel_proc)

        await server.stop_async()

        # Give the still-running executor start time to finish its
        # fail-fast window and reach the publish point.
        deadline = time.monotonic() + 5
        while pid_alive(child_pid) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        self.assertIsNone(
            server._tunnel_proc,
            "a cloudflared started before stop_async was published after it",
        )
        self.assertFalse(
            pid_alive(child_pid),
            "cloudflared spawned during shutdown must be killed, not leaked",
        )

    @staticmethod
    def _kill(pid: int) -> None:
        try:
            os.kill(pid, 9)
        except OSError:  # already gone (Windows raises a plain OSError)
            pass


if __name__ == "__main__":
    unittest.main()
