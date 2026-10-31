# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a Git or file action forwarded to a slow machine holds up none of the page's commands.

The Explorer's and Source Control view's ``fsAction`` / ``gitAction``
of a tab bound to a machine are forwarded to the machine's daemon and
waited for up to ``MACHINE_COMMIT_TIMEOUT`` (15 minutes: the machine
may wait for the repository's lock or copy a whole folder).  A
connection's commands are dispatched one after another, so a forward
awaited inline held up every later command of the page (a ``stop``, a
``submit``, a tab switch) for as long as the machine took.  The
forwards now run in the background, one after another per connection
and machine, in the order sent.

The "machine" is a loopback listener that accepts the TCP connection
but never answers the TLS client hello, which holds the forward's
connect for the link's handshake bound (``REQUEST_TIMEOUT``): the one
real way to make a machine slow (see
``test_remote_bind_does_not_block_connection.py``).  Real local daemon,
real local connection; no mocks, patches or test doubles.
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

TAB = "tab-on-hole"


class MachineActionDoesNotBlockTest(IsolatedAsyncioTestCase):
    """The page's commands sent while a machine action waits are answered at once."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.hole = socket.socket()
        self.hole.bind(("127.0.0.1", 0))
        self.hole.listen(8)  # never accepted: the TLS handshake waits for good
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        machine = mach.Machine(
            name="hole", url=f"https://127.0.0.1:{self.hole.getsockname()[1]}", password="pw",
        )
        machine.cert_file().parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(certfile, machine.cert_file())
        mach.save_machines([machine])
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        mach.save_machines([])
        self.hole.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _next(self) -> dict[str, Any]:
        line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
        assert line, "local connection closed"
        msg: dict[str, Any] = json.loads(line.decode("utf-8"))
        return msg

    async def _until(self, reply_type: str) -> dict[str, Any]:
        """The first *reply_type* message."""
        for _ in range(300):
            msg = await self._next()
            if msg.get("type") == reply_type:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def _action_tokens(self, count: int) -> list[str]:
        """The tokens of the next *count* ``gitActionResult`` / ``fsResult`` replies, in order."""
        tokens: list[str] = []
        for _ in range(300):
            msg = await self._next()
            if msg.get("type") in ("gitActionResult", "fsResult"):
                self.assertEqual(msg["machine"], "hole")
                self.assertIn("hole (https://127.0.0.1:", msg["error"])
                tokens.append(str(msg["token"]))
                if len(tokens) == count:
                    return tokens
        raise AssertionError(f"only {tokens} answered")

    @staticmethod
    def _git_action(token: str) -> dict[str, Any]:
        return {
            "type": "gitAction", "action": "checkout", "sha": "abc1234", "tabId": TAB,
            "workDir": "/work", "token": token, "machine": "hole",
        }

    @staticmethod
    def _fs_action(token: str) -> dict[str, Any]:
        return {
            "type": "fsAction", "action": "delete", "path": "/work/a.txt", "tabId": TAB,
            "workDir": "/work", "token": token, "machine": "hole",
        }

    async def test_the_pages_next_commands_are_answered_while_an_action_waits(self) -> None:
        started = time.monotonic()
        await self._send(self._git_action("git-1"))
        await self._send({"type": "ping"})
        await self._send({"type": "openTab", "tabId": "tab-other", "title": "other"})
        await self._until("pong")
        await self._until("tabs_state")
        self.assertLess(
            time.monotonic() - started, mach.REQUEST_TIMEOUT / 2,
            "the page's commands waited for the machine's Git action",
        )
        self.assertEqual(await self._action_tokens(1), ["git-1"])
        self.assertGreaterEqual(
            time.monotonic() - started, mach.REQUEST_TIMEOUT / 2,
            "the machine did not hold the action: the test exercised nothing",
        )

    async def test_a_pages_actions_on_a_machine_keep_their_order(self) -> None:
        await self._send(self._fs_action("fs-1"))
        await self._send(self._git_action("git-2"))
        await self._send(self._fs_action("fs-3"))
        self.assertEqual(await self._action_tokens(3), ["fs-1", "git-2", "fs-3"])
