# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""C-R1 / C-RC3: endpoint-send delegation and UDS bind serialization.

* C-R1 — ``RemoteAccessServer._endpoint_send`` duplicated
  ``WebPrinter._locked_send`` but silently dropped the dead-writer
  removal ``_uds_send`` performs, so a UDS peer that died mid-session
  stayed in the broadcast set forever and every direct reply to it
  raised out of the dispatch path.  After the fix it delegates to the
  printer's send, so a failed write removes the writer.

* C-RC3 — ``_setup_server`` probed, unlinked, and bound the UDS
  pathname non-atomically; two daemons starting concurrently could
  each pass the probe and then unlink the socket the other had just
  bound.  After the fix the sequence is serialized across processes
  by an exclusive ``fcntl.flock`` on a sidecar lock file next to the
  socket.  The test holds the lock (a second open of the same file
  contends even in-process, because ``flock`` locks the open file
  description) and verifies the bind waits for the release.

Real sockets, real event loop, real file locks — no mocks.
"""

from __future__ import annotations

import asyncio
import fcntl
import socket
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from kiss.server.web_server import RemoteAccessServer


def _free_port() -> int:
    """Reserve and release a localhost TCP port for the WSS listener."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class TestEndpointSendRemovesDeadUdsWriter(IsolatedAsyncioTestCase):
    """C-R1: a failed direct UDS reply must evict the dead writer."""

    async def test_dead_writer_removed_not_raised(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=_free_port(),
            work_dir=tmpdir.name,
            uds_path=f"{tmpdir.name}/sorcar.sock",
        )
        # A real UDS connection pair through a real listener.
        accepted: list[asyncio.StreamWriter] = []
        connected = asyncio.Event()

        async def on_connect(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        ) -> None:
            accepted.append(writer)
            connected.set()

        peer_path = f"{tmpdir.name}/peer.sock"
        listener = await asyncio.start_unix_server(
            on_connect, path=peer_path,
        )
        self.addCleanup(listener.close)
        client_reader, client_writer = await asyncio.open_unix_connection(
            peer_path,
        )
        await asyncio.wait_for(connected.wait(), timeout=10)
        server_side_writer = accepted[0]
        server._printer.add_uds_writer(server_side_writer)
        assert server_side_writer in server._printer._uds_writers

        # Kill the transport, then send a direct reply to it.
        server_side_writer.transport.abort()
        client_writer.close()
        await asyncio.sleep(0.05)

        # Must NOT raise (the old inline write/drain propagated the
        # write failure into the dispatch path) ...
        for _ in range(3):
            await server._endpoint_send(
                server_side_writer, '{"type":"ping"}',
            )
            if server_side_writer not in server._printer._uds_writers:
                break
            await asyncio.sleep(0.05)
        # ... and the dead writer must be evicted from the broadcast
        # set exactly as _uds_send's failure handler does.
        assert server_side_writer not in server._printer._uds_writers, (
            "BUG C-R1: dead UDS writer stayed in the broadcast set"
        )


class TestUdsBindSerializedByLock(IsolatedAsyncioTestCase):
    """C-RC3: probe → unlink → bind waits for the sidecar flock."""

    async def test_bind_waits_for_sidecar_lock(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        uds_path = Path(tmpdir.name) / "sorcar.sock"
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=_free_port(),
            work_dir=tmpdir.name,
            uds_path=str(uds_path),
        )

        lock_path = uds_path.with_name(uds_path.name + ".lock")
        holder = open(lock_path, "w", encoding="utf-8")
        self.addCleanup(holder.close)
        fcntl.flock(holder, fcntl.LOCK_EX)

        setup = asyncio.ensure_future(server._setup_server())
        try:
            # While a sibling holds the lock the socket must not be
            # bound: the fix serializes the whole sequence.
            await asyncio.sleep(0.6)
            assert server._uds_server is None, (
                "BUG C-RC3: UDS bound while another process held the "
                "bind lock — the probe/unlink/bind sequence is not "
                "serialized"
            )
            assert not uds_path.exists()

            fcntl.flock(holder, fcntl.LOCK_UN)
            await asyncio.wait_for(setup, timeout=60)
            assert server._uds_server is not None
            assert uds_path.exists()
        finally:
            if not setup.done():
                setup.cancel()
                try:
                    await setup
                except (asyncio.CancelledError, Exception):
                    pass
            await server.stop_async()
