# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the ``remote_url`` event says whether the daemon runs in Docker.

Inside a container none of the Local / LAN / tunnel URLs the settings
panel shows reaches the daemon from outside (the container's ports are
published by whoever ran it), so the event carries ``inDocker`` and the
panel paints no URL bar when it is true.

* :func:`kiss.core.browser_handoff.in_docker` is the presence of the
  Docker marker file (``/.dockerenv``); a test cannot create that file
  without root, so the helper is exercised against a marker path of
  the test's own.
* A live server's ``remote_url`` (the welcome-info reply) carries
  ``inDocker`` as a boolean equal to the helper's verdict on this host.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

from websockets.asyncio.client import connect

from kiss.core.browser_handoff import DOCKER_MARKER, in_docker
from kiss.tests.server.conftest import no_verify_ssl
from kiss.tests.server.test_local_lan_urls import _LiveServerCase


class TestInDocker(TestCase):
    """The marker-file check."""

    def test_marker_decides(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / ".dockerenv"
            self.assertFalse(in_docker(marker))
            marker.touch()
            self.assertTrue(in_docker(marker))

    def test_default_marker_is_dockerenv(self) -> None:
        self.assertEqual(DOCKER_MARKER, Path("/.dockerenv"))
        self.assertEqual(in_docker(), DOCKER_MARKER.exists())


class TestRemoteUrlCarriesInDocker(_LiveServerCase):
    """The live ``remote_url`` event stamps ``inDocker``."""

    async def test_in_docker_is_a_boolean_matching_this_host(self) -> None:
        self.server._active_url = "https://docker-test.trycloudflare.com"
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl()
        ) as ws:
            await self._auth_ws(ws)
            await self._send(ws, {"type": "getWelcomeInfo"})
            ev = await self._recv_remote_url(ws)
        self.assertIs(ev["inDocker"], in_docker())
        self.assertEqual(ev["loopbackUrl"], f"http://127.0.0.1:{self.port}")
