# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``RemoteAccessServer.stop_async`` must not block the event loop on the tunnel.

Concurrency audit (W5) regression: ``stop_async`` called
``_stop_tunnel()`` inline, and that path blocks in
``Popen.wait(timeout=5)`` (then ``kill``) while ``cloudflared`` shuts
down.  Every other blocking step of ``stop_async`` already runs in
``asyncio.to_thread``; the inline tunnel stop froze the loop — and
with it every connected client's final frames — for the whole grace
period when the tunnel ignored SIGTERM.

The test installs a real child that ignores SIGTERM as the tunnel
process and measures the event loop's longest stall while
``stop_async`` runs.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from kiss.server.web_server import RemoteAccessServer


class TestStopAsyncTunnelStopOffLoop(unittest.IsolatedAsyncioTestCase):
    """The loop keeps ticking while the tunnel's 5 s grace period elapses."""

    async def test_loop_stays_responsive_during_tunnel_stop(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_dir = Path(tmp.name)
        server = RemoteAccessServer(
            use_tunnel=False,
            url_file=tmp_dir / "remote-url.json",
            uds_path=tmp_dir / "kiss.sock",
        )
        server._vscode_server.use_private_tab_registry(tmp_dir / "tabs.json")
        server._loop = asyncio.get_running_loop()
        # The child prints "ready" only AFTER ignoring SIGTERM, so
        # stop_async's terminate() is guaranteed to hit a child that
        # ignores it (otherwise the 5 s blocking wait under test may
        # never be exercised and the test passes vacuously).  Windows
        # cannot ignore terminate(); there the test still checks that
        # the tunnel child is reaped without the loop stalling.  The
        # base interpreter is used because a venv's python.exe is a
        # launcher whose pid is not the interpreter's.
        proc = subprocess.Popen(
            [
                getattr(sys, "_base_executable", sys.executable), "-c",
                "import signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "print('ready', flush=True)\n"
                "time.sleep(120)\n",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.addCleanup(self._reap, proc)
        assert proc.stdout is not None
        self.assertEqual(proc.stdout.readline().strip(), "ready")
        proc.stdout.close()
        server._tunnel_proc = proc

        stalls: list[float] = []
        stop_ticking = asyncio.Event()

        async def heartbeat() -> None:
            last = time.monotonic()
            while not stop_ticking.is_set():
                await asyncio.sleep(0.05)
                now = time.monotonic()
                stalls.append(now - last)
                last = now

        ticker = asyncio.create_task(heartbeat())
        try:
            await server.stop_async()
        finally:
            stop_ticking.set()
            await ticker

        self.assertIsNotNone(proc.poll(), "tunnel child must be killed")
        self.assertIsNone(server._tunnel_proc)
        self.assertTrue(stalls, "heartbeat never ticked during stop_async")
        self.assertLess(
            max(stalls), 2.0,
            f"event loop stalled for {max(stalls):.2f}s during stop_async",
        )

    @staticmethod
    def _reap(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
