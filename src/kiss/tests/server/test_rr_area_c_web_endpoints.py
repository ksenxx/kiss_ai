# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""C-R1 / C-RC3: endpoint-send delegation and local-endpoint publication.

* C-R1 — ``RemoteAccessServer._endpoint_send`` duplicated
  ``WebPrinter._locked_send`` but silently dropped the dead-peer
  removal ``_timed_send`` performs, so a local peer that died
  mid-session stayed in the broadcast set forever and every direct
  reply to it raised out of the dispatch path.  After the fix it
  delegates to the printer's send, so a failed write removes the peer.

* C-RC3 — two daemons starting concurrently used to race on the
  shared local channel.  In the WSS era the shared resource is the
  local endpoint file: the daemon publishes it only AFTER its listener
  is bound (mode 0600, carrying the bound port and this daemon's
  token), and ``stop_async`` removes it only while it still carries
  this daemon's token, so a successor that already overwrote the file
  keeps its clients reachable.

Real sockets, real event loop, real files — no mocks.
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import local_endpoint
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import open_local_connection


class TestEndpointSendRemovesDeadLocalClient(IsolatedAsyncioTestCase):
    """C-R1: a failed direct local reply must evict the dead peer."""

    async def test_dead_peer_removed_not_raised(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=tmpdir.name,
            local_endpoint_file=f"{tmpdir.name}/sorcar-local.json",
        )
        await server.start_private_async()
        try:
            await self._check_dead_peer_evicted(server)
        finally:
            await server.stop_async()

    async def _check_dead_peer_evicted(self, server: RemoteAccessServer) -> None:
        # A real local connection through the real listener.
        _reader, client_writer = await open_local_connection(server)
        peers: set[Any] = set()
        for _ in range(100):
            with server._printer._ws_lock:
                peers = set(server._printer._local_clients)
            if peers:
                break
            await asyncio.sleep(0.02)
        assert len(peers) == 1, f"expected one local peer, got {peers!r}"
        server_side = next(iter(peers))

        # Kill the server-side transport, then send a direct reply to it.
        server_side.transport.abort()
        client_writer.close()
        await asyncio.sleep(0.05)

        # Must NOT raise (the old inline write/drain propagated the
        # write failure into the dispatch path) ...
        for _ in range(3):
            await server._endpoint_send(server_side, '{"type":"ping"}')
            if server_side not in server._printer._local_clients:
                break
            await asyncio.sleep(0.05)
        # ... and the dead peer must be evicted from the broadcast
        # set exactly as _timed_send's failure handler does.
        assert server_side not in server._printer._local_clients, (
            "BUG C-R1: dead local peer stayed in the broadcast set"
        )


class TestLocalEndpointPublication(IsolatedAsyncioTestCase):
    """C-RC3: the endpoint file is written after bind and removed only when owned."""

    async def test_endpoint_written_after_bind_and_removed_when_owned(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        endpoint_file = Path(tmpdir.name) / "sorcar-local.json"
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=tmpdir.name,
            url_file=Path(tmpdir.name) / "remote-url.json",
            local_endpoint_file=endpoint_file,
        )
        assert not endpoint_file.exists(), "endpoint published before bind"

        await server.start_async()
        try:
            assert endpoint_file.exists()
            if sys.platform != "win32":  # NTFS has no POSIX mode bits
                mode = stat.S_IMODE(endpoint_file.stat().st_mode)
                assert mode == 0o600, f"endpoint file mode {oct(mode)}"
            endpoint = local_endpoint.read_endpoint(endpoint_file)
            assert endpoint is not None
            assert endpoint.url == f"wss://127.0.0.1:{server.port}/ws"
            assert endpoint.token == server.local_token
            assert endpoint.pid == os.getpid()
            assert server.port != 0, "OS-picked port not recorded"
        finally:
            await server.stop_async()
        assert not endpoint_file.exists(), (
            "BUG C-RC3: owned endpoint file survived stop_async"
        )

    async def test_endpoint_kept_when_successor_overwrote_it(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        endpoint_file = Path(tmpdir.name) / "sorcar-local.json"
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=tmpdir.name,
            url_file=Path(tmpdir.name) / "remote-url.json",
            local_endpoint_file=endpoint_file,
        )
        await server.start_async()
        successor = local_endpoint.LocalEndpoint(
            url="wss://127.0.0.1:1/ws", token="successor-token",
            ca=None, pid=os.getpid() + 1,
        )
        try:
            # A successor daemon took over the shared path.
            local_endpoint.write_endpoint(endpoint_file, successor)
        finally:
            await server.stop_async()
        assert local_endpoint.read_endpoint(endpoint_file) == successor, (
            "BUG C-RC3: stop_async deleted a successor's endpoint file"
        )
