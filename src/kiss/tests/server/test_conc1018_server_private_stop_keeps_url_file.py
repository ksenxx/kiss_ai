# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: stopping a private (loopback-only) daemon leaves the canonical
daemon's ``remote-url.json`` alone, and builds no MCP manager.

The channel-agent launcher's in-process daemon shares the Sorcar home
with the canonical ``kiss-web`` daemon and never writes the URL file
(``start_private_async``), yet ``stop_async`` unlinked it
unconditionally, so ``kiss-web --url`` and the machine probes lost the
live daemon's URL until its watchdog rewrote it (findings I2-F2).

A real :class:`RemoteAccessServer` is started on its private loopback
endpoint and stopped; the pinned URL file (standing in for the
canonical daemon's) must survive with its content.  A daemon that is
not private still removes the file it owns on stop.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.server.web_server as web_server
from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig
from kiss.core.processes import pid_alive
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import redirect_persistence, restore_persistence

URL_CONTENT = '{"local": "http://localhost:1", "remote": "https://canonical.example"}\n'

# A real stdio MCP server that records its pid, so the test can prove
# the child the daemon shutdown must reap is gone afterwards.
MCP_SERVER_SCRIPT = """
import os, sys
from mcp.server.fastmcp import FastMCP

open(sys.argv[1], "w").write(str(os.getpid()))
mcp = FastMCP("stopsrv")


@mcp.tool()
def ping() -> str:
    return "pong"


mcp.run(transport="stdio")
"""


class PrivateStopKeepsUrlFileTest(IsolatedAsyncioTestCase):
    """``stop_async`` removes only a URL file the server itself owns."""

    async def asyncSetUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-conc1018-urlfile-"))
        self._saved = redirect_persistence(self.tmp)
        self.url_file = self.tmp / "remote-url.json"
        self.url_file.write_text(URL_CONTENT, encoding="utf-8")
        web_server._URL_FILE = self.url_file

    async def asyncTearDown(self) -> None:
        del web_server._URL_FILE
        restore_persistence(self._saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _server(self) -> RemoteAccessServer:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            local_endpoint_file=self.tmp / "sorcar-local.json",
            work_dir=str(self.tmp),
        )
        self.assertEqual(server._url_file, self.url_file)
        return server

    async def test_private_daemon_stop_keeps_the_canonical_url_file(self) -> None:
        server = self._server()
        await server.start_private_async()
        self.assertTrue(server._local_only)
        self.assertEqual(self.url_file.read_text(encoding="utf-8"), URL_CONTENT)
        await server.stop_async()
        self.assertTrue(self.url_file.exists(), "the private daemon unlinked the URL file")
        self.assertEqual(self.url_file.read_text(encoding="utf-8"), URL_CONTENT)

    async def test_public_daemon_stop_removes_its_own_url_file(self) -> None:
        server = self._server()
        self.assertFalse(server._local_only)
        await server.stop_async()
        self.assertFalse(self.url_file.exists())

    async def test_stop_without_mcp_use_builds_no_mcp_manager(self) -> None:
        """D1-F2: the shutdown path disconnects only an EXISTING MCP manager.

        A daemon that never used an MCP server used to build one (a
        loop thread plus an ``atexit`` hook) on shutdown just to
        disconnect an empty pool.
        """
        if MCPManager._instance is not None:
            self.skipTest("an earlier test in this process already built the MCP manager")
        server = self._server()
        await server.start_private_async()
        await server.stop_async()
        self.assertIsNone(MCPManager._instance)
        self.assertNotIn("mcp-manager", [t.name for t in threading.enumerate()])

    async def test_stop_disconnects_an_existing_mcp_manager(self) -> None:
        """An existing manager's live connection is torn down on stop.

        A real stdio MCP server is connected (handshake complete, its
        tool listed) before the daemon stops; afterwards the manager
        survives for an embedder, but its pool is empty and the child
        process is gone.
        """
        script = self.tmp / "mcp_server.py"
        script.write_text(MCP_SERVER_SCRIPT, encoding="utf-8")
        pid_file = self.tmp / "mcp_server.pid"
        config = MCPServerConfig(
            name="stopsrv",
            command=sys.executable,
            args=(str(script), str(pid_file)),
        )
        manager = MCPManager.instance()
        saved_streams = (sys.stdin, sys.stderr)
        # pytest's capture replaces the streams with objects that have no
        # file descriptor; the stdio transport hands real ones to the child.
        sys.stdin = open(os.devnull, encoding="utf-8")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
        try:
            conn = await asyncio.to_thread(manager.connect, config)
            self.assertEqual(conn.error, "", conn.error)
            self.assertIn("ping", [t.name for t in conn.tools])
            self.assertEqual(len(manager._connections), 1)
            pid = int(pid_file.read_text(encoding="utf-8"))
            self.assertTrue(pid_alive(pid))

            server = self._server()
            await server.start_private_async()
            await server.stop_async()

            self.assertIs(MCPManager._instance, manager)
            self.assertEqual(manager._connections, {})
            self.assertTrue(self._wait_dead(pid), f"MCP child {pid} survived the stop")
        finally:
            manager.shutdown()
            sys.stdin.close()
            sys.stderr.close()
            sys.stdin, sys.stderr = saved_streams

    @staticmethod
    def _wait_dead(pid: int) -> bool:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not pid_alive(pid):
                return True
            time.sleep(0.05)
        return False
