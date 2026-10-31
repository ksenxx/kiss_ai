# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: machine round trips queue per machine, and outdated mentions are dropped.

A real daemon and a client on its local connection; a registered
machine whose address is a loopback listener that accepts and never
speaks, so every handshake against it stalls until the listener is
closed (the test closes it once it has seen what it needs, instead of
waiting out ``REQUEST_TIMEOUT``).

* Regression (audit V7): one daemon-wide lock ordered every machine's
  registry change and work-dir round trip, so a stalled machine (or an
  ``add`` of a mistyped address) held up every other machine's WORKDIR
  drop-down.  The order kept is per machine: another machine is
  answered meanwhile.
* Regression (audit V7): the ``@`` mention asks the tab's machine for
  its file list at every keystroke, and each request was forwarded, one
  after another, before the one for the mention on screen.  Of the
  requests that waited together behind one on its way, only the last is
  forwarded now; the page drops every list but the current one anyway.

A ``ping`` answered with ``pong`` marks the point where the daemon has
dispatched every frame sent before it (it reads a connection's frames
in order) and started the background reply of each.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection


def _get_files(prefix: str) -> dict[str, Any]:
    """The ``getFiles`` the page sends for mention *prefix* in tab ``tab-1`` bound to ``mute``."""
    return {"type": "getFiles", "prefix": prefix, "tabId": "tab-1", "machine": "mute"}


def _add_or_work_dirs(msg: dict[str, Any]) -> bool:
    """Whether *msg* answers a ``machineAction add`` or a work-dir request."""
    return msg["type"] == "machineWorkDirs" or msg.get("action") == "add"


class TestMachineQueues(IsolatedAsyncioTestCase):
    """A daemon, a silent machine, and a client on the local connection."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        root = Path(self.tmpdir)
        self.silent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.silent.bind(("127.0.0.1", 0))
        self.silent.listen(8)  # never accepted: the TLS handshake stalls
        self.silent_url = f"https://127.0.0.1:{self.silent.getsockname()[1]}"
        self.certfile = root / "cert.pem"
        keyfile = root / "key.pem"
        _generate_self_signed_cert(self.certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(self.certfile),
            keyfile=str(keyfile), url_file=root / "remote-url.json",
            local_endpoint_file=root / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)

    async def asyncTearDown(self) -> None:
        self.writer.close()
        self.silent.close()
        await self.server.stop_async()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, cmd: dict[str, Any]) -> None:
        """Send *cmd* on the local connection."""
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _next(self) -> dict[str, Any]:
        """Return the next message of the local connection (waiting 60 s at most)."""
        line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
        assert line, "local connection closed"
        msg: dict[str, Any] = json.loads(line.decode("utf-8"))
        return msg

    async def _read(self, reply_type: str) -> dict[str, Any]:
        """Return the next *reply_type* message, skipping the others."""
        while (msg := await self._next())["type"] != reply_type:
            pass
        return msg

    async def _next_answer(self) -> dict[str, Any]:
        """Return the next message answering an ``add`` or a work-dir request."""
        while not _add_or_work_dirs(msg := await self._next()):
            pass
        return msg

    async def _ping(self) -> list[dict[str, Any]]:
        """Send ``ping`` and return the messages that came before its ``pong``."""
        await self._send({"type": "ping"})
        before: list[dict[str, Any]] = []
        while (msg := await self._next())["type"] != "pong":
            before.append(msg)
        return before

    def _register_silent_machine(self, name: str) -> None:
        """Register *name* at the silent listener with a pinned certificate.

        With the pin the link really connects (and stalls in the TLS
        handshake) rather than failing at once on a missing pin.
        """
        machine = mach.Machine(name=name, url=self.silent_url, password="pw")
        machine.cert_file().parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.certfile, machine.cert_file())
        mach.save_machines([*mach.load_machines(), machine])

    async def test_a_stalled_machine_holds_up_no_other_machines_work_dirs(self) -> None:
        self._register_silent_machine("mute")
        started = time.monotonic()
        await self._send({"type": "getMachineWorkDirs", "machine": "mute"})
        # ``nope`` is not registered: its reply needs no network.
        await self._send({"type": "getMachineWorkDirs", "machine": "nope"})
        first = await self._read("machineWorkDirs")
        self.assertEqual(first["machine"], "nope", "nope's reply waited for mute's")
        self.assertIn("no machine named 'nope'", first["error"])
        self.assertLess(time.monotonic() - started, mach.REQUEST_TIMEOUT / 2)
        self.silent.close()  # mute's handshake fails now
        mute = await self._read("machineWorkDirs")
        self.assertEqual(mute["machine"], "mute")
        self.assertNotEqual(mute["error"], "")

    async def test_an_unreachable_add_holds_up_no_other_machine(self) -> None:
        started = time.monotonic()
        await self._send({
            "type": "machineAction", "action": "add", "name": "typo",
            "address": self.silent_url, "machinePassword": "pw",
        })
        await self._send({"type": "getMachineWorkDirs", "machine": "nope"})
        first = await self._next_answer()
        self.assertEqual(first["type"], "machineWorkDirs", "nope's reply waited for the add")
        self.assertEqual(first["machine"], "nope")
        self.assertLess(time.monotonic() - started, mach.REQUEST_TIMEOUT / 2)
        self.silent.close()  # the add's handshakes fail now
        added = await self._next_answer()
        self.assertEqual(added["action"], "add")
        self.assertNotEqual(added["error"], "")
        self.assertEqual([m["name"] for m in added["machines"]][1:], [])

    async def test_an_add_without_a_name_queues_under_the_hosts_name(self) -> None:
        # The machine an add without a name registers is named after the
        # address's host: a work-dir request for that name waits for it.
        await self._send({
            "type": "machineAction", "action": "add", "name": "",
            "address": self.silent_url, "machinePassword": "pw",
        })
        await self._send({"type": "getMachineWorkDirs", "machine": "127.0.0.1"})
        seen = await self._ping()  # both are on their way
        self.silent.close()  # the add's handshakes fail now
        order = [msg["type"] for msg in seen if _add_or_work_dirs(msg)]
        while len(order) < 2:
            order.append((await self._next_answer())["type"])
        self.assertEqual(order, ["machines", "machineWorkDirs"])

    async def test_outdated_file_requests_are_never_forwarded(self) -> None:
        self._register_silent_machine("mute")
        await self._send(_get_files("s"))
        seen = await self._ping()  # "s" is on its way to the machine
        await self._send(_get_files("sr"))
        await self._send(_get_files("src"))
        seen += await self._ping()  # both wait behind "s"
        self.assertEqual([m for m in seen if m["type"] == "files"], [], "'s' did not stall")
        self.silent.close()  # "s" fails now
        # The lists come in the order asked, so a list for "sr" would
        # come before the one for "src".
        prefixes = [(await self._read("files"))["prefix"] for _ in range(2)]
        self.assertEqual(prefixes, ["s", "src"])
