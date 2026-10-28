# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a ``machineAction`` whose registration blows up still answers (F11).

``machines.add_machine`` can raise something other than ``MachineError``
(a malformed IPv6 address makes ``urlsplit`` raise ``ValueError`` while
deriving the default name); ``_reply_machine_action`` ran that in a
background task without the ``try/except`` of its sibling replies, so
the task died silently and the client never got its ``machines`` reply.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection


class TestMachineActionReplyOnUnexpectedError(IsolatedAsyncioTestCase):
    """A ``machineAction`` that raises an unexpected exception is answered with its error."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        root = Path(self.tmpdir)
        certfile = root / "cert.pem"
        keyfile = root / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=root / "remote-url.json",
            local_endpoint_file=root / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _ask(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()
        for _ in range(100):
            line = await asyncio.wait_for(self.reader.readline(), timeout=20.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def test_malformed_ipv6_address_is_answered_with_an_error(self) -> None:
        reply = await self._ask({
            "type": "machineAction", "action": "add", "name": "",
            "address": "[::1", "machinePassword": "pw",
        }, "machines")
        self.assertEqual(reply["action"], "add")
        self.assertIn("Invalid IPv6 URL", reply["error"])
        # Nothing was registered: the report lists this daemon alone.
        self.assertEqual([m for m in reply["machines"] if not m["local"]], [])
        # The connection is still served: a plain poll answers too.
        polled = await self._ask({"type": "getMachines"}, "machines")
        self.assertEqual(polled["error"], "")
