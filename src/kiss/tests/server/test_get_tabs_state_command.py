# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``getTabsState`` daemon command.

The VS Code extension host's long-lived controller sends
``getTabsState`` on every daemon (re)connect.  The daemon otherwise
broadcasts ``tabs_state`` only after registry mutations and webview
``ready`` syncs, so a host with no open chat webview (editor-tabs
mode with all panels closed) would have no baseline snapshot: the
next remote-client mutation would be the FIRST snapshot it ever sees,
and a host reconnecting after an outage would never learn of tabs
created while it was away.

Driven against a real :class:`RemoteAccessServer` over real ``wss://``
connections (no mocks).
"""

from __future__ import annotations

from typing import Any

from kiss.tests.server.test_web_server_tab_mirroring import TabMirroringBase


class TestGetTabsState(TabMirroringBase):
    """``getTabsState`` re-broadcasts the canonical tab snapshot."""

    async def test_get_tabs_state_broadcasts_snapshot(self) -> None:
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

        # Client B (the extension host controller) connects WITHOUT a
        # webview `ready` handshake and asks for the baseline.
        ws_b = await self._connect_ok()
        await self._send(ws_b, {"type": "getTabsState"})
        snap_b = await self._wait_for_snapshot_with(ws_b, present={"tab-a1"})
        self.assertIsNotNone(
            snap_b,
            "getTabsState must broadcast the canonical tabs_state "
            "snapshot to a client that never sent `ready`",
        )
        assert snap_b is not None
        entry = next(
            t for t in snap_b["tabs"] if t.get("tabId") == "tab-a1"
        )
        self.assertEqual(entry.get("title"), "remote task")

        # The reply is a plain broadcast: idempotent for client A too.
        snap_a = await self._wait_for_snapshot_with(ws_a, present={"tab-a1"})
        self.assertIsNotNone(snap_a)

    async def test_get_tabs_state_with_empty_registry(self) -> None:
        # A fresh install: no tabs at all.  The requester still gets a
        # (empty) snapshot so its baseline is established.
        ws = await self._connect_ok()
        await self._send(ws, {"type": "getTabsState", "tabId": "ignored"})

        def _empty(ev: dict[str, Any]) -> bool:
            return ev.get("tabs") == []

        snap = await self._wait_for_event(ws, "tabs_state", pred=_empty)
        self.assertIsNotNone(
            snap,
            "getTabsState on an empty registry must still broadcast an "
            "empty tabs_state snapshot",
        )
