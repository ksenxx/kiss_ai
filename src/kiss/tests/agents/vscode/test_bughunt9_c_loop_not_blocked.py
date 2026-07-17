# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: slow disk/network helpers must not stall the event loop.

M10 convention: blocking calls (directory scans, IP discovery) run via
``asyncio.to_thread`` when invoked from coroutines.  Pre-fix,
``_broadcast_update_available`` called ``_read_version`` (a directory
scan) synchronously on the loop, and the watchdog called
``_get_local_ips`` synchronously each tick.

These tests replace the module-level helpers with REAL slow functions
(the same module-attribute override pattern used by
``test_web_server_ip_watchdog_debounce.py``) and verify concurrent
loop progress / the new pre-fetched-ips call contract.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from kiss.server import web_server as ws


def _make_server(tmp: str) -> ws.RemoteAccessServer:
    return ws.RemoteAccessServer(
        host="127.0.0.1",
        use_tunnel=False,
        url_file=Path(tmp) / "remote-url.json",
        uds_path=Path(tmp) / "sorcar.sock",
    )


class TestBroadcastUpdateAvailableOffLoop(unittest.TestCase):
    """_broadcast_update_available must not block the event loop."""

    def test_slow_read_version_does_not_stall_loop(self) -> None:
        """A 0.6s _read_version leaves the loop free to run other tasks."""
        old = ws._read_version

        def slow_read_version() -> str:
            time.sleep(0.6)
            return "0.0.1"

        ws._read_version = slow_read_version
        try:
            with tempfile.TemporaryDirectory() as tmp:
                srv = _make_server(tmp)
                srv._latest_version = "9999.0.0"
                ticks = asyncio.run(self._run(srv))
        finally:
            ws._read_version = old
        self.assertGreaterEqual(
            ticks, 5,
            "event loop made almost no progress while _read_version "
            "ran — the directory scan is blocking the loop",
        )

    @staticmethod
    async def _run(srv: ws.RemoteAccessServer) -> int:
        counter = [0]

        async def heartbeat() -> None:
            while True:
                counter[0] += 1
                await asyncio.sleep(0.05)

        hb = asyncio.create_task(heartbeat())
        try:
            await srv._broadcast_update_available()
        finally:
            hb.cancel()
        return counter[0]


class TestWatchdogIpCheckAcceptsPrefetchedIps(unittest.TestCase):
    """_watchdog_check_ip_change(ips) must use the pre-fetched result."""

    def test_prefetched_ips_skip_inline_discovery(self) -> None:
        """Passing ips avoids the blocking _get_local_ips call entirely."""
        old = ws._get_local_ips

        def must_not_be_called() -> frozenset[str]:
            raise AssertionError(
                "_get_local_ips was called inline despite pre-fetched ips",
            )

        ws._get_local_ips = must_not_be_called
        try:
            with tempfile.TemporaryDirectory() as tmp:
                srv = _make_server(tmp)
                srv._last_ips = frozenset({"192.0.2.1"})
                # Baseline unchanged: no restart, no inline discovery.
                self.assertFalse(
                    srv._watchdog_check_ip_change(frozenset({"192.0.2.1"})),
                )
        finally:
            ws._get_local_ips = old

    def test_no_arg_call_still_discovers(self) -> None:
        """The no-arg form (used by older tests) still self-discovers."""
        old = ws._get_local_ips
        ws._get_local_ips = lambda: frozenset({"192.0.2.7"})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                srv = _make_server(tmp)
                srv._last_ips = frozenset()
                self.assertFalse(srv._watchdog_check_ip_change())
                # First non-empty discovery becomes the baseline.
                self.assertEqual(srv._last_ips, frozenset({"192.0.2.7"}))
        finally:
            ws._get_local_ips = old


if __name__ == "__main__":
    unittest.main()
