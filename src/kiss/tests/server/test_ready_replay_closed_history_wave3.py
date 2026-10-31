# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A connection's own replay announces a finished sub-agent as closed history.

A finished sub-agent's tab closes on the live surface, so a reload, a
second browser or the VS Code webview opening (each sends ``ready``)
must not reopen it.  The daemon marks the ``openSubagentTab`` of a
finished child ``closedHistory`` on every connection-scoped replay
(the tab the ``ready`` names, a deferred tab, the deferred tab's later
fetch), and the page then keeps the tab closed; a history click still
announces it without the mark, so the tab opens.  A real daemon and
real ``wss://`` clients; nothing is mocked.
"""

from __future__ import annotations

from typing import Any

from websockets.asyncio.client import ClientConnection

from kiss.tests.server.test_ready_replay_conn_scoped import _seed_subagent_row
from kiss.tests.server.test_web_server_tab_mirroring import TabMirroringBase


class TestReadyReplayClosedHistory(TabMirroringBase):
    """Only a connection-scoped replay marks a finished child ``closedHistory``."""

    async def _seed_parent_with_finished_child(self) -> tuple[str, str, str]:
        """Record a parent chat with one finished sub-agent; return (chat, child, child tab)."""
        parent_id, chat_id = self._seed_chat("Parent task", [
            {"type": "prompt", "text": "fan out"},
        ])
        sub_id = _seed_subagent_row(parent_id, chat_id, "child transcript")
        return chat_id, sub_id, f"tab-p__sub_{sub_id}"

    async def _announcement(self, ws: ClientConnection, sub_tab: str) -> dict[str, Any]:
        """The next ``openSubagentTab`` *ws* gets for *sub_tab*."""
        ev = await self._wait_for_event(
            ws, "openSubagentTab", pred=lambda e: e.get("tab_id") == sub_tab,
        )
        assert ev is not None, f"no openSubagentTab for {sub_tab}"
        return ev

    async def _open_parent_by_history_click(
        self, ws: ClientConnection, chat_id: str, sub_tab: str,
    ) -> dict[str, Any]:
        """Open ``tab-p`` and resume *chat_id* in it; return the child's announcement."""
        await self._send(ws, {"type": "openTab", "tabId": "tab-p", "title": "new chat"})
        self.assertIsNotNone(await self._wait_for_snapshot_with(ws, present={"tab-p"}))
        await self._send(ws, {"type": "resumeSession", "chatId": chat_id, "tabId": "tab-p"})
        return await self._announcement(ws, sub_tab)

    async def test_history_click_opens_the_finished_child(self) -> None:
        """A history click's announcement carries ``isDone`` and no ``closedHistory``."""
        chat_id, _sub_id, sub_tab = await self._seed_parent_with_finished_child()
        ws_a = await self._connect_ok()
        await self._ready(ws_a)

        ev = await self._open_parent_by_history_click(ws_a, chat_id, sub_tab)

        self.assertTrue(ev["isDone"])
        self.assertNotIn("closedHistory", ev)

    async def test_ready_of_the_tab_on_screen_marks_the_child_closed(self) -> None:
        """The tab a ``ready`` names replays with its finished child marked closed."""
        chat_id, _sub_id, sub_tab = await self._seed_parent_with_finished_child()
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._open_parent_by_history_click(ws_a, chat_id, sub_tab)

        ws_b = await self._connect_ok()
        await self._ready(ws_b, tab_id="tab-p")
        ev = await self._announcement(ws_b, sub_tab)

        self.assertTrue(ev["isDone"])
        self.assertIs(ev["closedHistory"], True)

    async def test_deferred_tab_and_its_fetch_mark_the_child_closed(self) -> None:
        """A deferred parent, and the scoped fetch of it, mark the finished child closed."""
        chat_id, _sub_id, sub_tab = await self._seed_parent_with_finished_child()
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._open_parent_by_history_click(ws_a, chat_id, sub_tab)

        ws_b = await self._connect_ok()
        await self._ready(ws_b, tab_id="elsewhere")
        deferred = await self._announcement(ws_b, sub_tab)
        self.assertIs(deferred["closedHistory"], True)

        await self._send(ws_b, {
            "type": "resumeSession", "chatId": chat_id, "tabId": "tab-p", "scoped": True,
        })
        fetched = await self._announcement(ws_b, sub_tab)
        self.assertIs(fetched["closedHistory"], True)
