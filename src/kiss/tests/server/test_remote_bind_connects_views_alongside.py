# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A remote chat opens after one round of handshakes, not two.

A history click on a remote machine's chat binds the tab to the
machine: the daemon connects the tab's stream (TLS, WebSocket upgrade,
login), then forwards the ``resumeSession``.  The machine's view stream
(its browser pages and terminals) used to be connected after the tab's
stream, before the resume was forwarded, so the user waited for two
full rounds of handshakes before the first frame of the transcript; on
a link with tens of milliseconds of latency that was most of the delay.
Now the view stream connects alongside the tab's stream
(:meth:`kiss.server.remote_views.RemoteViews.warm`).

The remote daemon is reached through a TCP proxy that delays every
byte by ``DELAY`` seconds in each direction, so each round trip costs
``2 * DELAY``.  The gap between the binding's ``tabMachine`` (sent once
the tab's stream is up) and the replay's first frame (``status``) is
one forwarded round trip when the view stream connected alongside,
and three or more round trips (its own connect) when it did not.

A bind that is refused (the machine is down) leaves no view stream
behind: the one warmed for it is closed, as the registry has nothing
bound to the machine.

A click on another chat of the same machine opens it in a new tab and
closes the tab left behind.  That close found no tab bound to the
machine (the new one was still connecting) and closed the view stream,
which the bind then connected again before it forwarded the resume:
the second round of handshakes was back.  The view stream is kept for
a bind on its way (:meth:`kiss.server.remote_views.RemoteViews.expect`),
so the switch costs the new tab's handshake alone, counted here by the
connections the proxy accepts; and a stream kept for a bind that is
then refused is closed once the refusal is in.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urlsplit

from kiss.agents.sorcar import machines as mach
from kiss.tests.conftest import find_free_port
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness

DELAY = 0.25
"""One-way delay of the proxy, seconds (a round trip is twice that)."""


async def _delayed_pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    """Copy *src* to *dst*, each chunk DELAY seconds late."""
    try:
        while True:
            data = await src.read(65536)
            if not data:
                break
            await asyncio.sleep(DELAY)
            dst.write(data)
            await dst.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        dst.close()


class TestRemoteBindConnectsViewsAlongside(RemoteTabsHarness):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        target = urlsplit(self.remote.url)
        host, port = str(target.hostname), int(target.port or 0)

        self.dials = 0
        self.refusing = False

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            self.dials += 1
            if self.refusing:
                # Held past the close of the tab left behind, then dropped.
                await asyncio.sleep(4 * DELAY)
                writer.close()
                return
            upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
            await asyncio.gather(
                _delayed_pump(reader, upstream_writer),
                _delayed_pump(upstream_reader, writer),
            )

        self.proxy_port = find_free_port()
        self.proxy = await asyncio.start_server(handle, "127.0.0.1", self.proxy_port)
        slow_url = self.remote.url.replace(f":{port}", f":{self.proxy_port}")
        await self._send({
            "type": "machineAction", "action": "add", "name": "slow",
            "address": slow_url, "machinePassword": "pw-remote",
        })
        # The add's own reply (an earlier ``machines`` reply may be in
        # flight): the registry holds ``slow`` from then on.
        reply = await self._until("machines")
        while not any(m.get("name") == "slow" for m in reply["machines"]):
            reply = await self._until("machines")
        self.assertEqual(reply["error"], "")
        self.assertIn("slow", [m.name for m in mach.load_machines()])

    async def asyncTearDown(self) -> None:
        # Not waited for: a link socket the daemon still holds through
        # the proxy would keep ``wait_closed`` waiting.
        self.proxy.close()
        await super().asyncTearDown()

    async def _box_row(self, title: str) -> dict[str, Any]:
        """The history row of box's task *title*, as the CHATS panel gets it."""
        reply = await self._ask({"type": "getHistory", "machine": "box"}, "history")
        return next(s for s in reply["sessions"] if s["title"] == title)

    async def _click_history_row(self, tab: str, row: dict[str, Any], machine: str) -> None:
        """Send what a click on a CHATS-panel row of *machine* sends."""
        await self._send({
            "type": "resumeSession", "id": row["id"], "taskId": row["task_id"],
            "tabId": tab, "machine": machine, "title": row["title"],
        })

    async def _frames_after_click(self, tab: str, row: dict[str, Any]) -> dict[str, float]:
        """Click *row* into *tab* through ``slow``; seconds after the click per first frame type."""
        await self._send({"type": "openTab", "tabId": tab, "title": row["title"]})
        await self._until("tabs_state")
        t0 = time.monotonic()
        await self._click_history_row(tab, row, "slow")
        seen: dict[str, float] = {}
        for _ in range(300):
            msg = await self._next()
            if msg.get("tabId") == tab:
                seen.setdefault(str(msg["type"]), time.monotonic() - t0)
                if msg["type"] == "error":
                    raise AssertionError(f"the click was refused: {msg['text']}")
            if "task_events" in seen:
                return seen
        raise AssertionError(f"no task_events after the click: {seen}")

    async def _switch_from_a_chat_of_slow(self, refusing: bool = False) -> None:
        """Show a chat of ``slow`` in tab-one, then send a page's click on another of its chats.

        The page opens the chat in a new tab (tab-two), resumes it
        there and closes the tab it left.  With *refusing*, the proxy
        drops the connections made from the click on.
        """
        await self._submit("tab-a", "Say hello", "box")
        await self._run_until_idle("tab-a")
        await self._send({"type": "closeTab", "tabId": "tab-a"})
        await self._until("tabs_state")
        hello = await self._box_row("Say hello")
        await self._frames_after_click("tab-one", hello)
        self.switched = (self.server._server_api.remote_views._streams["slow"], self.dials)
        self.refusing = refusing
        await self._send({"type": "openTab", "tabId": "tab-two", "title": hello["title"]})
        await self._click_history_row("tab-two", hello, "slow")
        await self._send({"type": "closeTab", "tabId": "tab-one"})

    async def test_a_switch_to_another_chat_of_the_machine_keeps_its_view_stream(self) -> None:
        """The new tab's stream is the one connection made: the view stream stayed open."""
        await self._switch_from_a_chat_of_slow()
        await self._until("task_events", "tab-two")
        stream, dials = self.switched
        views = self.server._server_api.remote_views
        self.assertIs(views._streams.get("slow"), stream, "the view stream was connected again")
        self.assertEqual(self.dials - dials, 1)

    async def test_a_view_stream_kept_for_a_bind_that_is_refused_is_closed(self) -> None:
        """The tab left is closed while the bind connects; the refusal leaves nothing bound."""
        await self._switch_from_a_chat_of_slow(refusing=True)
        error = await self._until("error", "tab-two")
        self.assertIn("slow (https://127.0.0.1:", error["text"])
        views = self.server._server_api.remote_views
        for _ in range(100):
            if "slow" not in views._streams:
                break
            await asyncio.sleep(0.1)
        self.assertNotIn("slow", views._streams)

    async def test_the_replay_follows_the_binding_within_one_round_trip(self) -> None:
        """The first replay frame comes one forwarded round trip after ``tabMachine``."""
        await self._submit("tab-a", "Say hello", "box")
        await self._run_until_idle("tab-a")
        await self._send({"type": "closeTab", "tabId": "tab-a"})
        await self._until("tabs_state")
        hello = await self._box_row("Say hello")
        seen = await self._frames_after_click("tab-slow", hello)
        gap = seen["status"] - seen["tabMachine"]
        # One round trip (2 * DELAY) plus the daemons' own work; a view
        # stream connected in between costs at least three more.
        self.assertLess(gap, 5 * DELAY, seen)
        views = self.server._server_api.remote_views
        self.assertIn("slow", views._streams)
        # The binding keeps the view stream; closing the tab releases it.
        await self._send({"type": "closeTab", "tabId": "tab-slow"})
        await self._until("tabs_state")
        for _ in range(100):
            if "slow" not in views._streams:
                break
            await asyncio.sleep(0.1)
        self.assertNotIn("slow", views._streams)

    async def test_a_refused_bind_leaves_no_view_stream(self) -> None:
        """A click on a machine that is down is refused, and no view stream stays open for it."""
        await self._submit("tab-a", "Say hello", "box")
        await self._run_until_idle("tab-a")
        await self._send({"type": "closeTab", "tabId": "tab-a"})
        await self._until("tabs_state")
        hello = await self._box_row("Say hello")
        dead = mach.Machine(name="dead", url=f"https://127.0.0.1:{find_free_port()}",
                            password="x")
        mach.save_machines([*mach.load_machines(), dead])
        await self._send({"type": "openTab", "tabId": "tab-dead", "title": hello["title"]})
        await self._until("tabs_state")
        await self._click_history_row("tab-dead", hello, "dead")
        error = await self._until("error", "tab-dead")
        self.assertIn("dead (https://127.0.0.1:", error["text"])
        views = self.server._server_api.remote_views
        self.assertNotIn("dead", views._streams)
