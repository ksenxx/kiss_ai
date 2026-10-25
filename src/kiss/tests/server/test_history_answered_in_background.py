# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a CHATS page in flight does not hold up the same connection's next command.

A connection's commands are dispatched one after another, and a
``getHistory`` reply merges every registered machine's history: a
second or more over the machines' links, their whole timeout when one
is down.  Serviced inline it held the page's next ``stop`` / ``submit``
/ ``activeTasksQuery`` until the last machine had answered or timed
out, and itself waited behind every slow forward in front of it; the
reply now goes out from a background task (``ServerApi.get_history``),
the connection's pages in the order they were asked for.

Real daemon over its local WSS endpoint, one real client; the
registered machine is a socket that accepts and never answers, so the
page takes the link's whole budget and the ordering is observable.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

from kiss.agents.sorcar import machines as mach
from kiss.tests.server.two_client_harness import TwoClientHarness


class HistoryAnsweredInBackgroundTest(TwoClientHarness):
    """``activeTasksQuery`` sent right after a slow ``getHistory`` is answered first."""

    def setUp(self) -> None:
        super().setUp()
        self.hole = socket.socket()
        self.hole.bind(("127.0.0.1", 0))
        self.hole.listen(5)
        self.addCleanup(self.hole.close)
        machine = mach.Machine("hole", f"https://127.0.0.1:{self.hole.getsockname()[1]}")
        pin = machine.cert_file()
        pin.parent.mkdir(parents=True, exist_ok=True)
        # Any pin lets the link dial; the handshake then hangs on the hole.
        pin.write_text((Path(self.tmpdir) / "cert.pem").read_text(encoding="utf-8"))
        self.addCleanup(pin.unlink, missing_ok=True)
        mach.save_machines([machine])
        self.addCleanup(mach.save_machines, [])

    def test_the_next_command_is_answered_before_the_page(self) -> None:
        client = self.open_client("a")
        client.ready()
        self.wait_event(client, "models")

        started = time.monotonic()
        client.send({"type": "getHistory", "query": "", "offset": 0, "generation": 7})
        client.send({"type": "activeTasksQuery"})
        # A later page of the connection: fast (this daemon's rows only),
        # yet answered after the slow one.
        client.send({
            "type": "getHistory", "query": "", "offset": 0, "generation": 8,
            "machine": mach.LOCAL_MACHINE,
        })
        self.wait_event(client, "activeTasksResponse")
        self.assertLess(
            time.monotonic() - started, mach.REQUEST_TIMEOUT / 2,
            "activeTasksQuery waited for the machine's history",
        )
        self.assertEqual(client.of_type("history"), [])

        slow = self.wait_event(client, "history", timeout=60.0)
        self.assertEqual(slow["generation"], 7)
        self.assertEqual(slow["machines"], ["hole"])
        self.assertTrue(all("machine" not in s for s in slow["sessions"]))
        fast = self.wait_event(client, "history", generation=8, timeout=20.0)
        self.assertEqual(fast["sessions"], slow["sessions"])
        replies = ("activeTasksResponse", "history")
        order = [e["type"] for e in client.events if e["type"] in replies]
        self.assertEqual(order, ["activeTasksResponse", "history", "history"])
