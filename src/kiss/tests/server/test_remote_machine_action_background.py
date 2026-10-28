# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a slow ``machineAction`` does not hold up the connection's other commands.

A real daemon and a client on its local connection.  The client adds a
machine whose address is a listener that accepts and then never
answers (``add`` tries the TLS handshake for ``REQUEST_TIMEOUT``
twice), and sends ``ping`` right behind it.  The ``pong`` must arrive
at once -- the round trip runs in the background like the subpanel's
poll -- and the refused ``add`` is answered with its ``machines``
reply afterwards.
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


class TestMachineActionInBackground(IsolatedAsyncioTestCase):
    """``machineAction``, ``getMachineWorkDirs`` and ``setWorkDir(machine)`` run off the stream."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        root = Path(self.tmpdir)
        # A listener that accepts connections and never speaks: every
        # handshake against it runs to its timeout.
        self.silent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.silent.bind(("127.0.0.1", 0))
        self.silent.listen(8)
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
        await self.server.stop_async()
        self.silent.close()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _assert_pong_is_prompt(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        """Send *cmd* then ``ping``; the pong must not wait for *cmd*'s *reply_type* reply.

        Returns the *reply_type* reply, which must take the silent
        host's handshake timeout at least.
        """
        await self._send(cmd)
        sent = time.monotonic()
        await self._send({"type": "ping"})
        pong_at: float | None = None
        deadline = sent + 60.0
        while True:
            remaining = deadline - time.monotonic()
            self.assertGreater(remaining, 0, f"no {reply_type} reply within 60s")
            line = await asyncio.wait_for(self.reader.readline(), timeout=remaining)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == "pong":
                pong_at = time.monotonic()
            elif msg.get("type") == reply_type:
                break
        self.assertIsNotNone(pong_at, "the pong waited for the machine round trip")
        assert pong_at is not None
        self.assertLess(pong_at - sent, 4.0)
        self.assertGreaterEqual(time.monotonic() - sent, mach.REQUEST_TIMEOUT)
        return msg

    def _register_silent_machine(self, name: str) -> None:
        """Register *name* at the silent listener with a pinned certificate.

        With the pin the link really connects (and stalls in the TLS
        handshake) rather than failing at once on a missing pin.
        """
        machine = mach.Machine(name=name, url=self.silent_url, password="pw")
        machine.cert_file().parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.certfile, machine.cert_file())
        mach.save_machines([*mach.load_machines(), machine])

    async def test_add_of_a_silent_host_does_not_block_ping(self) -> None:
        reply = await self._assert_pong_is_prompt({
            "type": "machineAction", "action": "add", "name": "mute",
            "address": self.silent_url, "machinePassword": "pw",
        }, "machines")
        self.assertEqual(reply["action"], "add")
        self.assertNotEqual(reply["error"], "")
        self.assertEqual([m["name"] for m in reply["machines"]][1:], [])

    async def test_work_dirs_of_a_silent_machine_do_not_block_ping(self) -> None:
        # Each command gets a machine of its own, registered right
        # before it: a link that timed out (on the request, or on the
        # registry watcher's ping of a new machine) backs off and
        # answers the next request at once.
        self._register_silent_machine("mute")
        reply = await self._assert_pong_is_prompt(
            {"type": "getMachineWorkDirs", "machine": "mute"}, "machineWorkDirs",
        )
        self.assertEqual(reply["machine"], "mute")
        self.assertNotEqual(reply["error"], "")
        self.assertNotIn("set", reply)
        self._register_silent_machine("mute2")
        reply = await self._assert_pong_is_prompt(
            {"type": "setWorkDir", "machine": "mute2", "workDir": self.tmpdir}, "machineWorkDirs",
        )
        self.assertEqual(reply["machine"], "mute2")
        self.assertIs(reply["set"], True)
        self.assertNotEqual(reply["error"], "")

    async def test_machine_replies_keep_command_order(self) -> None:
        """A quick machine command sent behind a slow one is answered after it.

        ``setWorkDir`` on the silent machine stalls for the handshake
        timeout; ``getMachineWorkDirs`` of an unregistered machine
        answers without any network.  Run as independent background
        tasks the quick reply would overtake the slow one (and a later
        ``setWorkDir`` could apply before an earlier one): the replies
        must come back in the order the commands were sent.
        """
        self._register_silent_machine("mute")
        await self._send({"type": "setWorkDir", "machine": "mute", "workDir": self.tmpdir})
        await self._send({"type": "getMachineWorkDirs", "machine": "nope"})
        replies: list[str] = []
        deadline = time.monotonic() + 60.0
        while len(replies) < 2:
            remaining = deadline - time.monotonic()
            self.assertGreater(remaining, 0, "no machineWorkDirs replies within 60s")
            line = await asyncio.wait_for(self.reader.readline(), timeout=remaining)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == "machineWorkDirs":
                replies.append(msg["machine"])
        self.assertEqual(replies, ["mute", "nope"])
