# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``tabs_state`` baseline a ``ready`` establishes.

A client's ``ready`` handshake is answered with the canonical
``tabs_state`` snapshot, whatever the client restored: that is how a
page connecting late, or reconnecting after an outage, learns the tabs
the other clients created meanwhile.  (A separate ``getTabsState``
command once served clients that connected without ``ready``; nothing
sends it any more, V5-1.)

Driven against a real :class:`RemoteAccessServer` over real ``wss://``
connections (no mocks).
"""

from __future__ import annotations

from typing import Any

from kiss.tests.server.test_web_server_tab_mirroring import TabMirroringBase


class TestReadyTabsStateBaseline(TabMirroringBase):
    """``ready`` is answered with the canonical tab snapshot."""

    async def test_ready_gives_a_late_client_the_baseline(self) -> None:
        # Client A (a remote web app) registers a tab — the only
        # mutation the registry ever sees.
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-a1", "title": "remote task",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_a, present={"tab-a1"}),
        )

        # Client B connects later, restoring nothing: its ``ready``
        # brings it the tab A opened.
        ws_b = await self._connect_ok()
        await self._ready(ws_b)
        snap_b = await self._wait_for_snapshot_with(ws_b, present={"tab-a1"})
        self.assertIsNotNone(
            snap_b, "ready must answer with the canonical tabs_state snapshot",
        )
        assert snap_b is not None
        entry = next(
            t for t in snap_b["tabs"] if t.get("tabId") == "tab-a1"
        )
        self.assertEqual(entry.get("title"), "remote task")

        # The snapshot is a plain broadcast: idempotent for client A too.
        snap_a = await self._wait_for_snapshot_with(ws_a, present={"tab-a1"})
        self.assertIsNotNone(snap_a)

    async def test_ready_on_an_empty_registry_gives_an_empty_snapshot(self) -> None:
        # A fresh install: no tabs at all.  The client still gets an
        # (empty) snapshot so its baseline is established.
        ws = await self._connect_ok()
        await self._ready(ws)

        def _empty(ev: dict[str, Any]) -> bool:
            return ev.get("tabs") == []

        snap = await self._wait_for_event(ws, "tabs_state", pred=_empty)
        self.assertIsNotNone(
            snap, "ready on an empty registry must still broadcast an empty tabs_state snapshot",
        )
