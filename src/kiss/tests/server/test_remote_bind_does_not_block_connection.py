# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a ``submit`` binding a tab to a slow machine does not hold up the connection.

Binding a tab to a machine connects to it first, and a black-holed
host (a listener that never completes the TLS handshake) holds that
connect for the link's whole timeout.  A connection's commands are
dispatched one after another, so a bind routed inline froze every
other tab of the surface for those seconds (V4-5).  The bind now runs
as a background task; the bound tab's own later commands queue behind
it in wire order (a ``stop`` reaches the machine after the run it
refers to, a ``closeTab`` closes what the bind opened), while the other
tabs' commands are answered at once.

The "machine" here is a loopback listener whose backlog accepts the
TCP connection but nobody ever answers the TLS client hello: the one
real way to pause a connect (``test_remote_tabs.py`` names this case
as not coverable with a real daemon).  Real local daemon, real local
connection.

Not covered: the warning branch of ``ServerApi._dispatch_logged`` (a
queued command whose handler raises) needs a handler that fails, which
no real command does without a fault injected.
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


class BindDoesNotBlockConnectionTest(IsolatedAsyncioTestCase):
    """The commands sent while a bind connects are answered at once."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.hole = socket.socket()
        self.hole.bind(("127.0.0.1", 0))
        self.hole.listen(4)  # never accepted: the TLS handshake waits for good
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

    async def _until(self, reply_type: str, tab_id: str = "") -> dict[str, Any]:
        """The first *reply_type* message (of *tab_id* when given)."""
        for _ in range(300):
            line = await asyncio.wait_for(self.reader.readline(), timeout=30.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type and (not tab_id or msg.get("tabId") == tab_id):
                return msg
        raise AssertionError(f"no {reply_type} reply")

    @staticmethod
    def _pending(api: Any, tab: str) -> bool:
        """Whether *tab* has a detached command pending on any connection."""
        return any(pending_tab == tab for _conn, pending_tab in api._pending_routes)

    async def _wait_route_done(self, api: Any, tab: str) -> None:
        """Wait until *tab* has no detached route pending any more."""
        for _ in range(int(mach.REQUEST_TIMEOUT * 4) + 40):
            if not self._pending(api, tab):
                return
            await asyncio.sleep(0.25)
        raise AssertionError(f"the routes of {tab} never finished")

    async def test_other_tabs_are_answered_while_the_bind_connects(self) -> None:
        tab, other = "tab-hole", "tab-other"
        await self._send({"type": "openTab", "tabId": tab, "title": "hole"})
        await self._until("tabs_state")
        started = time.monotonic()
        await self._send({"type": "submit", "tabId": tab, "prompt": "Say hello", "machine": "hole"})
        await self._send({"type": "ping"})
        await self._send({"type": "openTab", "tabId": other, "title": "other"})
        await self._until("pong")
        tabs = await self._until("tabs_state")
        self.assertIn(other, [t["tabId"] for t in tabs["tabs"]])
        self.assertLess(
            time.monotonic() - started, mach.REQUEST_TIMEOUT / 2,
            "the other tab's commands waited for the bind's connect",
        )
        api = self.server._server_api
        self.assertTrue(self._pending(api, tab), "the bind is still connecting")
        # The handshake times out: the run is refused on its tab.
        error = await self._until("error", tab)
        self.assertEqual(error["code"], "prompt_refused")
        self.assertIn("hole (https://127.0.0.1:", error["text"])
        self.assertGreaterEqual(
            time.monotonic() - started, mach.REQUEST_TIMEOUT / 2,
            "the machine did not hold the connect: the test exercised nothing",
        )
        await self._wait_route_done(api, tab)
        self.assertEqual(api.remote_tabs.machine_of(tab), "")

    async def test_the_tabs_own_close_follows_its_bind(self) -> None:
        """A ``closeTab`` sent while the bind connects is applied after it, in wire order.

        The bind's refusal still lands on the tab (it is open until
        the close is processed), then the close takes the tab out of
        the registry, with the other tabs answered meanwhile.
        """
        tab = "tab-hole-close"
        await self._send({"type": "openTab", "tabId": tab, "title": "hole"})
        await self._until("tabs_state")
        await self._send({"type": "submit", "tabId": tab, "prompt": "Say hello", "machine": "hole"})
        await self._send({"type": "closeTab", "tabId": tab})
        await self._send({"type": "ping"})
        await self._until("pong")
        api = self.server._server_api
        self.assertTrue(self._pending(api, tab))
        self.assertIn(tab, [t["tabId"] for t in self.server._agent_server.tab_registry.snapshot()])
        error = await self._until("error", tab)
        self.assertEqual(error["code"], "prompt_refused")
        tabs = await self._until("tabs_state")
        self.assertNotIn(tab, [t["tabId"] for t in tabs["tabs"]], "the close followed the bind")
        await self._wait_route_done(api, tab)
        self.assertEqual(api.remote_tabs.machine_of(tab), "")

    async def test_a_stop_sent_during_the_bind_follows_it(self) -> None:
        """A ``stop`` sent while the bind connects queues behind it.

        When the bind is refused the stop goes to the local handler, as
        it would have inline (nothing to stop here).
        """
        tab = "tab-hole-stop"
        await self._send({"type": "submit", "tabId": tab, "prompt": "Say hello", "machine": "hole"})
        await self._send({"type": "stop", "tabId": tab})
        api = self.server._server_api
        self.assertTrue(self._pending(api, tab))
        await self._send({"type": "ping"})
        await self._until("pong")
        error = await self._until("error", tab)
        self.assertEqual(error["code"], "prompt_refused")
        await self._wait_route_done(api, tab)
        self.assertEqual(api.remote_tabs.machine_of(tab), "")

    async def test_an_open_tab_after_a_queued_close_reopens_the_tab(self) -> None:
        """Every command of the tab queues behind its bind, not only the routed ones.

        ``submit`` (bind), ``closeTab``, ``openTab`` in wire order: the
        tab must end up open.  An ``openTab`` dispatched at once, while
        the close still waited behind the bind, found the tab still
        registered, did nothing, and was undone by the close.
        """
        tab = "tab-hole-reopen"
        await self._send({"type": "openTab", "tabId": tab, "title": "hole"})
        await self._until("tabs_state")
        await self._send({"type": "submit", "tabId": tab, "prompt": "Say hello", "machine": "hole"})
        await self._send({"type": "closeTab", "tabId": tab})
        await self._send({"type": "openTab", "tabId": tab, "title": "hole again"})
        await self._send({"type": "ping"})
        await self._until("pong")
        api = self.server._server_api
        self.assertTrue(self._pending(api, tab))
        error = await self._until("error", tab)
        self.assertEqual(error["code"], "prompt_refused")
        await self._wait_route_done(api, tab)
        tabs = {t["tabId"]: t for t in self.server._agent_server.tab_registry.snapshot()}
        self.assertIn(tab, tabs, "the queued close undid the later openTab")
        self.assertEqual(tabs[tab]["title"], "hole again")
        self.assertEqual(api.remote_tabs.machine_of(tab), "")

    async def test_another_connection_s_close_of_the_tab_is_not_held_behind_the_bind(self) -> None:
        """Only the binding connection's own commands for the tab queue behind its bind.

        A second surface showing the same tab closes it while the bind
        connects: inline, the two connections never waited for each
        other (the registry orders them), so the close is applied at
        once and the bind's late refusal lands on a closed tab.
        """
        tab = "tab-hole-other-surface"
        await self._send({"type": "openTab", "tabId": tab, "title": "hole"})
        await self._until("tabs_state")
        reader, writer = await open_local_connection(self.server)
        self.addCleanup(writer.close)
        started = time.monotonic()
        await self._send({"type": "submit", "tabId": tab, "prompt": "Say hello", "machine": "hole"})
        await self._send({"type": "ping"})
        await self._until("pong")
        api = self.server._server_api
        self.assertTrue(self._pending(api, tab), "the bind is connecting")
        writer.write(json.dumps({"type": "closeTab", "tabId": tab}).encode("utf-8") + b"\n")
        await writer.drain()
        msg: dict[str, Any] = {}
        for _ in range(300):
            line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            msg = json.loads(line.decode("utf-8"))
            if msg.get("type") == "tabs_state":
                break
        self.assertEqual(msg.get("type"), "tabs_state", "no tabs_state within 300 frames")
        self.assertNotIn(tab, [t["tabId"] for t in msg["tabs"]])
        self.assertLess(
            time.monotonic() - started, mach.REQUEST_TIMEOUT / 2,
            "the other connection's close waited for the bind's connect",
        )
        self.assertTrue(self._pending(api, tab), "the bind is still connecting")
        error = await self._until("error", tab)
        self.assertEqual(error["code"], "prompt_refused")
        await self._wait_route_done(api, tab)
        self.assertEqual(api.remote_tabs.machine_of(tab), "")
