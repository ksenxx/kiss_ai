# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``remote_url`` publications rank by the state they
captured, not by when their coroutine starts.

``_republish_urls`` captures the current URL and queues the broadcast
in a task.  The publication's generation used to be assigned when
that task started, so a queued snapshot of an *old* URL could mint a
newer generation than a publication of the *new* URL made after it
and overtake it on every connected client.  The welcome path had the
same shape: a URL read from the URL file was broadcast as-is even when
a tunnel (re)start published a different URL during the read.

Everything is real: a ``RemoteAccessServer`` on a free port, a WSS
client receiving the ``remote_url`` events, a FIFO to pause the URL
file read.  No mocks.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect

from kiss.tests.conftest import posix_only
from kiss.tests.server.test_local_lan_urls import _LiveServerCase, _no_verify_ssl

OLD = "https://old.example.test"
NEW = "https://new.example.test"


class TestUrlPublishGeneration(_LiveServerCase):
    """A superseded ``remote_url`` never lands after its replacement."""

    async def _remote_urls_within(self, ws: Any, seconds: float) -> list[str]:
        """Collect the ``url`` of every ``remote_url`` event for *seconds*."""
        urls: list[str] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        while (remaining := deadline - loop.time()) > 0:
            try:
                ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            except TimeoutError:
                break
            if ev.get("type") == "remote_url":
                urls.append(ev["url"])
        return urls

    async def test_queued_republish_does_not_overtake_newer_publication(self) -> None:
        """A republish queued with URL A loses to a later publication of B.

        Order: queue the republish while A is active; switch to B and
        publish B; let the queued republish run.  The client must end
        up with B and never see A after it.
        """
        self.server.use_tunnel = True
        self.server._active_url = OLD
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await self._auth_ws(ws)
            self.server._republish_urls()
            assert self.server._republish_task is not None
            self.server._active_url = NEW
            await self.server._broadcast_remote_url(NEW, True)
            await self.server._republish_task
            urls = await self._remote_urls_within(ws, 1.0)
        self.assertEqual(urls, [NEW])

    @posix_only("the URL-file read is paused with a FIFO")
    async def test_welcome_revalidates_url_file_result_after_the_read(self) -> None:
        """A URL published during the welcome's file read wins over the file.

        The URL file is a FIFO, so the executor's read blocks until the
        test writes it; meanwhile the active URL becomes NEW.  The
        welcome broadcast must carry NEW, not the OLD URL the file held.
        """
        fifo = Path(self.server.work_dir) / "remote-url.fifo"
        os.mkfifo(fifo)
        self.server.use_tunnel = False
        self.server._url_file = fifo
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await self._auth_ws(ws)
            self.server._active_url = None
            welcome = asyncio.create_task(self.server._send_welcome_info())
            await asyncio.sleep(0.3)
            self.assertFalse(welcome.done(), "the URL-file read did not block on the FIFO")
            self.server._active_url = NEW
            await asyncio.to_thread(fifo.write_text, json.dumps({"tunnel": OLD}))
            await asyncio.wait_for(welcome, timeout=10)
            urls = await self._remote_urls_within(ws, 1.0)
        self.assertEqual(urls, [NEW])
        self.assertEqual(self.server._active_url, NEW)
