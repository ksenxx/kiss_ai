# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-12 (leftovers): a remote tab's send cannot wedge on a deaf daemon.

``RemoteTab.send`` wrote with the synchronous ``websockets`` client's
``send``, which has no deadline and holds the connection's write lock.
A remote daemon that stopped reading (frozen, suspended, half-open NAT)
left the executor thread of ``remote_tabs.route`` blocked for good:

* the client's command loop (``async for message in websocket`` in
  ``web_server``) awaits that ``to_thread`` call, so every later
  command of that surface (any tab: ``stop``, ``submit``, ``closeTab``)
  was never read;
* ``RemoteTab.close`` (a ``closeTab``, a re-bind, ``close_all`` at
  daemon shutdown) needs the same write lock, so the daemon could not
  shut down either.

The send now goes through ``local_endpoint.send`` with the link's
``REQUEST_TIMEOUT``; a write that misses it closes the socket
underneath, which fails the command on the tab and ends the stream as
a lost connection.

The remote daemon stand-in is the sorcar-core fixer's loopback TLS
server: it completes the WebSocket upgrade, answers ``auth`` with
``auth_ok`` and never reads again, so a large ``submit`` fills the
kernel buffers.  Before the fix this test times out waiting for the
tab's ``error`` (and its teardown hangs in ``close_all``).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.agents.sorcar.test_audit1012_sorcar_core_machine_link_send import (
    _auth_ok_then_deaf_server,
)
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection

_TAB = "tab-deaf"


class TestRemoteTabDeafDaemon(IsolatedAsyncioTestCase):
    """One local daemon, one deaf "remote daemon", one client."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-deaf-tab-")
        self.deaf_thread, port, ca_file, self.deaf_stop = _auth_ok_then_deaf_server(
            Path(self.tmpdir) / "deaf",
        )
        deaf = mach.Machine(name="deaf", url=f"https://127.0.0.1:{port}", password="pw")
        deaf.cert_file().parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ca_file, deaf.cert_file())
        mach.save_machines([deaf])
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=find_free_port(),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        self.deaf_stop.set()
        self.deaf_thread.join(5)
        mach.close_links()
        mach.save_machines([])
        deaf_cert = mach.Machine(name="deaf").cert_file()
        deaf_cert.unlink(missing_ok=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _until(self, reply_type: str, timeout: float) -> dict[str, Any]:
        """Return the first *reply_type* message, waiting at most *timeout* seconds in all."""
        deadline = time.monotonic() + timeout
        while True:
            line = await asyncio.wait_for(
                self.reader.readline(),
                timeout=max(deadline - time.monotonic(), 0.1),
            )
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type:
                return msg

    async def test_submit_to_a_deaf_machine_fails_in_time_and_keeps_the_daemon_responsive(
        self,
    ) -> None:
        started = time.monotonic()
        await self._send(
            {
                "type": "submit",
                "tabId": _TAB,
                "machine": "deaf",
                "prompt": "x" * (32 * 1024 * 1024),
                "workDir": self.tmpdir,
            }
        )
        error = await self._until("error", timeout=60.0)
        self.assertEqual(error.get("tabId"), _TAB)
        self.assertIn("deaf", error["text"])
        self.assertLess(time.monotonic() - started, 40.0)
        # The stream is gone and the tab is home again.
        deadline = time.monotonic() + 10.0
        while self.server._server_api.remote_tabs.machine_of(_TAB) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(_TAB), "")
        # The client's command loop is free: the next command is answered at once.
        started = time.monotonic()
        await self._send({"type": "ping"})
        await self._until("pong", timeout=10.0)
        self.assertLess(time.monotonic() - started, 5.0)
        # Daemon shutdown (``close_all``) does not wait behind the stream.
        started = time.monotonic()
        await self.server.stop_async()
        self.assertLess(time.monotonic() - started, 15.0)
