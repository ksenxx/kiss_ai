# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A loopback reclaim's endpoint publication cannot outlive ``stop_async``.

Audit 2026-11-01 (V2 F3): after reclaiming ``127.0.0.1:port`` the
watchdog publishes the endpoint file from an executor thread
(``asyncio.to_thread(self._write_local_endpoint_file)``).
``stop_async`` cancelled the watchdog task, but cancelling a
``to_thread`` awaiter does not stop its worker: a worker that had not
reached the write yet resumed after ``stop_async`` had removed the
endpoint file and returned, and wrote the stopped listener's URL and
token back.  Same-machine clients then found an endpoint nobody
listened on, and a successor daemon's own endpoint could be overwritten
by that stale publication.  The reclaim (bind + publish) now runs under
``_lifecycle_lock``, which ``stop_async`` holds for the whole
shutdown, so a shutdown waits for a publication in flight instead of
racing it.

Real ``RemoteAccessServer`` bound to this machine's own non-loopback
address (the ``--host 192.168.1.5`` case that needs the loopback alias
on every platform), a real child process holding ``127.0.0.1:port``
at start-up so the alias is reclaimed later, a real endpoint file.  The
worker is held at the entry of the real ``write_endpoint`` with a
``sys.settrace`` hook (:class:`Pause`); nothing is replaced.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import local_endpoint
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import find_free_port
from kiss.tests.server.pause import Pause
from kiss.tests.server.test_web_server_loopback_alias import _hold_loopback


def _own_non_loopback_ip() -> str:
    """Return the address this machine would use to reach the outside, ``""`` offline."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
        except OSError:
            return ""
        return str(probe.getsockname()[0])


class TestReclaimPublicationVsStop(IsolatedAsyncioTestCase):
    """``stop_async`` leaves no endpoint file behind a reclaim in flight."""

    async def asyncSetUp(self) -> None:
        self.host = _own_non_loopback_ip()
        if not self.host:
            self.skipTest("no non-loopback address on this machine")
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-reclaim-stop-"))
        self.port = find_free_port()
        self.endpoint_file = self.tmpdir / "sorcar-local.json"
        self.server = RemoteAccessServer(
            host=self.host, port=self.port, use_tunnel=False,
            work_dir=str(self.tmpdir), url_file=self.tmpdir / "remote-url.json",
            local_endpoint_file=self.endpoint_file,
        )
        self.holder = _hold_loopback(self.port)

    async def asyncTearDown(self) -> None:
        self.holder.kill()
        self.holder.wait()
        await self.server.stop_async()

    async def test_stop_waits_for_the_reclaim_publication(self) -> None:
        self.assertTrue(self.server._wants_loopback_alias())
        await self.server.start_async()
        self.assertIsNone(self.server._ws_loopback_server)
        self.assertIsNotNone(local_endpoint.read_endpoint(self.endpoint_file))
        # The test drives the reclaim tick itself.
        watchdog = self.server._watchdog_task
        assert watchdog is not None
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog
        self.holder.kill()
        self.holder.wait()

        with Pause(local_endpoint.write_endpoint.__code__, lambda _locals: True) as pause:
            reclaim = asyncio.create_task(self.server._watchdog_reclaim_loopback())
            self.server._watchdog_task = reclaim
            self.assertTrue(await asyncio.to_thread(pause.paused.wait, 10))
            self.assertIsNotNone(self.server._ws_loopback_server)
            stop = asyncio.create_task(self.server.stop_async())
            await asyncio.sleep(0.5)
            stop_returned_early = stop.done()
            pause.release()
            await asyncio.wait_for(stop, 20)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(reclaim, 10)
        self.assertIsNone(self.server._ws_server)
        self.assertIsNone(self.server._ws_loopback_server)
        # Nothing joins the executor worker: give a late publication
        # (it lands within milliseconds of the release) time to show.
        stale = None
        for _ in range(40):
            stale = local_endpoint.read_endpoint(self.endpoint_file)
            if stale is not None:
                break
            await asyncio.sleep(0.05)
        self.assertEqual(
            (stop_returned_early, stale), (False, None),
            "stop_async returned while the reclaim's publication was in flight, "
            "and the reclaim republished the endpoint of the stopped server",
        )


if __name__ == "__main__":
    unittest.main()
