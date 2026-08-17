# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E regression tests for the ``RemoteAccessServer`` HTTP wire protocol.

Pins the HTTP wire protocol of ``RemoteAccessServer`` (GET / and
HEAD / over a real TLS connection).

All tests drive real objects — no mocks, patches, or fakes.
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import ssl
import tempfile
import unittest
from pathlib import Path

from kiss.server.web_server import RemoteAccessServer


def _free_port() -> int:
    """Grab an ephemeral localhost port for the live-server test."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestHttpWireProtocol(unittest.IsolatedAsyncioTestCase):
    """Real TLS GET / and HEAD / round-trips against RemoteAccessServer."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-simp2-http-")
        self.port = _free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _raw_request(self, payload: bytes) -> bytes:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", self.port, ssl=ctx,
        )
        try:
            writer.write(payload)
            await writer.drain()
            return await asyncio.wait_for(reader.read(), timeout=5.0)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_get_root_serves_html(self) -> None:
        raw = await self._raw_request(
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        head, _, body = raw.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.1 200"), head[:60])
        self.assertIn(b"text/html", head.lower())
        self.assertIn(b"<html", body[:4096].lower())

    async def test_head_root_returns_200(self) -> None:
        raw = await self._raw_request(b"HEAD / HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertTrue(raw.startswith(b"HTTP/1.1 200"), raw[:60])


if __name__ == "__main__":
    unittest.main()
