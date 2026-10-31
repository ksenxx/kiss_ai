# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a work-dir request queued behind a machine ``add`` is answered after the add.

``ServerApi`` orders a machine's registry change and its work-dir round
trips with one lock per machine name.  The ``add``'s reply (a
``machines`` list, pinged anew) used to go out after that lock was
released, so the request waiting behind the add could answer first
whenever the reply's pings took longer than the request.  The existing
``test_machine_queues_sweep1030.py`` hit this now and then under load;
here a second registered machine whose ping stalls (a listener that
never answers the TLS handshake) makes the add's reply slow every time.

A real daemon and a client on its local connection.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection


def _silent_listener() -> socket.socket:
    """A loopback listener that accepts nothing: a TLS handshake against it stalls."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    return listener


def _url(listener: socket.socket) -> str:
    """The ``https://`` address of *listener*."""
    return f"https://127.0.0.1:{listener.getsockname()[1]}"


class TestMachineAddReplyOrder(IsolatedAsyncioTestCase):
    """A daemon with a stalled machine ``mute``, an add target, and a local client."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        root = Path(self.tmpdir)
        self.target = _silent_listener()  # the add's address; closed to fail the add
        self.mute = _silent_listener()  # registered; every ping of it stalls
        certfile = root / "cert.pem"
        keyfile = root / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        # Registered before the daemon starts, so its registry watcher
        # sees no change (and pings nothing) during the test.
        mute = mach.Machine(name="mute", url=_url(self.mute), password="pw")
        mute.cert_file().parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(certfile, mute.cert_file())  # pinned: the link really connects
        mach.save_machines([mute])
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=root / "remote-url.json",
            local_endpoint_file=root / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)

    async def asyncTearDown(self) -> None:
        self.writer.close()
        self.target.close()
        self.mute.close()
        await self.server.stop_async()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _next(self) -> dict[str, Any]:
        line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
        assert line, "local connection closed"
        msg: dict[str, Any] = json.loads(line.decode("utf-8"))
        return msg

    async def test_the_add_reply_comes_before_the_queued_work_dir_reply(self) -> None:
        await self._send({
            "type": "machineAction", "action": "add", "name": "",
            "address": _url(self.target), "machinePassword": "pw",
        })
        # Named after the address's host: it waits for the add.
        await self._send({"type": "getMachineWorkDirs", "machine": "127.0.0.1"})
        await self._send({"type": "ping"})
        while (await self._next())["type"] != "pong":
            pass
        self.target.close()  # the add fails now; its reply then pings "mute"
        order: list[str] = []
        while len(order) < 2:
            msg = await self._next()
            if msg["type"] == "machineWorkDirs" or msg.get("action") == "add":
                order.append(msg["type"])
        self.assertEqual(order, ["machines", "machineWorkDirs"])
