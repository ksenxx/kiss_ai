# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: subpanel forwards to a machine stay bounded (audit 2026-10-18).

A real local daemon, a real remote daemon registered as ``box``, and a
"black hole" machine ``hole`` whose port accepts the connection and
never answers (a host that drops packets):

* a subpanel poll repeated while its predecessor is still waiting on
  the machine is dropped -- one reply per in-flight poll, so the
  executor threads the forwards hold stay bounded by the number of
  connections -- and the next poll after the reply is forwarded again;
* a ``cronAction`` (a user's action) is never dropped;
* the daemon's stop does not wait for a forward still in flight.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

from kiss.agents.sorcar import machines as mach
from kiss.tests.server.test_remote_machine_settings import RemoteMachineCase


class TestPanelForwardDedup(RemoteMachineCase):
    """One forward in flight per (connection, poll, machine)."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.listener.settimeout(0.2)
        self.accepted: list[socket.socket] = []
        self.stop_accepting = threading.Event()
        self.acceptor = threading.Thread(target=self._accept_forever, daemon=True)
        self.acceptor.start()
        hole = mach.Machine(
            name="hole", url=f"https://127.0.0.1:{self.listener.getsockname()[1]}",
            password="pw",
        )
        self.hole_cert = hole.cert_file()
        self.hole_cert.parent.mkdir(parents=True, exist_ok=True)
        (box,) = [m for m in mach.load_machines() if m.name == "box"]
        self.hole_cert.write_bytes(box.cert_file().read_bytes())
        mach.save_machines([box, hole])

    def _accept_forever(self) -> None:
        while not self.stop_accepting.is_set():
            try:
                self.accepted.append(self.listener.accept()[0])
            except TimeoutError:
                continue

    async def asyncTearDown(self) -> None:
        self.stop_accepting.set()
        self.acceptor.join(5)
        for conn in self.accepted:
            conn.close()
        self.listener.close()
        self.hole_cert.unlink(missing_ok=True)
        await super().asyncTearDown()

    async def _hole_replies(self, reply_type: str, within: float) -> list[dict[str, object]]:
        """The forwarded *reply_type* replies for ``hole`` arriving within *within* seconds."""
        replies: list[dict[str, object]] = []
        deadline = time.monotonic() + within
        while True:
            try:
                reply = await asyncio.wait_for(
                    self._next(reply_type, forwarded=True), max(0.01, deadline - time.monotonic()),
                )
            except TimeoutError:
                return replies
            if reply["machine"] == "hole":
                replies.append(reply)

    async def test_a_repeated_poll_waits_for_the_one_in_flight(self) -> None:
        for _ in range(3):
            await self._send({"type": "getSpendReport", "machine": "hole"})
        # The first poll's dial gives up after REQUEST_TIMEOUT; its reply
        # is the one answer, the two repeats were dropped (without the
        # de-dup they queue on the link and answer right behind it).
        replies = await self._hole_replies("spendReport", mach.REQUEST_TIMEOUT + 6)
        self.assertEqual(len(replies), 1, replies)
        self.assertIn("unreachable", str(replies[0]["error"]))
        # Answered: the next poll is forwarded again (the machine is backed off).
        started = time.monotonic()
        again = await self._ask({"type": "getSpendReport", "machine": "hole"}, "spendReport")
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual(again["machine"], "hole")
        self.assertIn("unreachable", again["error"])

    async def test_cron_actions_are_never_dropped(self) -> None:
        for name in ("box one", "box two"):
            await self._send({
                "type": "cronAction", "action": "create", "name": name, "machine": "box",
                "schedule": "every 1d", "what": f"echo {name}", "kind": "command",
            })
        first = await self._next("cronJobs", forwarded=True)
        second = await self._next("cronJobs", forwarded=True)
        self.assertEqual((first.get("error", ""), second.get("error", "")), ("", ""))
        names = [j["name"] for j in second["jobs"]]
        self.assertIn("box one", names)
        self.assertIn("box two", names)

    async def test_stop_does_not_wait_for_a_forward_in_flight(self) -> None:
        await self._send({"type": "getSpendReport", "machine": "hole"})
        await asyncio.sleep(0.5)
        api = self.server._server_api
        self.assertEqual(len(api._background_replies), 1)
        started = time.monotonic()
        await self.server.stop_async()
        self.assertLess(time.monotonic() - started, mach.REQUEST_TIMEOUT - 1)
        self.assertEqual(api._background_replies, set())

    async def test_a_poll_received_during_the_stop_spawns_no_reply(self) -> None:
        await self._send({"type": "getSpendReport", "machine": "hole"})
        await asyncio.sleep(0.5)
        api = self.server._server_api
        (pending,) = api._background_replies
        # A frame arriving while the stop awaits the cancelled replies:
        # the done callback of the reply being cancelled sends it, so
        # it reaches the still-open command ingress at exactly that
        # moment.
        sent = asyncio.Event()

        def send_at_cancellation(task: asyncio.Task[None]) -> None:
            if task.cancelled():
                self.writer.write(b'{"type":"getRecommendations","machine":"hole"}\n')
                sent.set()

        pending.add_done_callback(send_at_cancellation)
        await self.server.stop_async()
        self.assertTrue(sent.is_set())
        self.assertEqual(api._background_replies, set())
        self.assertEqual(api._panel_replies, {})
