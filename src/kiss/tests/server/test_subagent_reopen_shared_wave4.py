# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An explicit reopening of a sub-agent's tab is shared by every surface.

Expanding a fan-out panel asks the daemon for each of its sub-agents
again (``resumeSession`` with ``reopen: true``).  The announcement the
daemon broadcasts for it says ``reopened``, so every surface opens the
tab, even one that mirrored an earlier close of it.  A finished
sub-agent reopened this way stays open for surfaces that connect later:
their replay announces it ``reopened`` instead of ``closedHistory``,
until some surface closes the tab again.  A resume without ``reopen``
(a page fetching a spawned sub-agent's transcript) is not a reopening.
A real daemon and real ``wss://`` clients; nothing is mocked.
"""

from __future__ import annotations

from typing import Any

from websockets.asyncio.client import ClientConnection

from kiss.tests.server.test_ready_replay_conn_scoped import _seed_subagent_row
from kiss.tests.server.test_web_server_tab_mirroring import TabMirroringBase


class TestSubagentReopenIsShared(TabMirroringBase):
    """``reopen`` marks the announcement and survives into later replays."""

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

    async def _open_parent(
        self, ws: ClientConnection, chat_id: str, sub_tab: str,
    ) -> None:
        """Open ``tab-p`` and resume *chat_id* in it, consuming the child's announcement."""
        await self._send(ws, {"type": "openTab", "tabId": "tab-p", "title": "new chat"})
        self.assertIsNotNone(await self._wait_for_snapshot_with(ws, present={"tab-p"}))
        await self._send(ws, {"type": "resumeSession", "chatId": chat_id, "tabId": "tab-p"})
        await self._announcement(ws, sub_tab)

    async def _resume_child(
        self, ws: ClientConnection, sub_id: str, sub_tab: str, reopen: bool,
    ) -> dict[str, Any]:
        """Ask for the child's tab as an expanded panel (``reopen``) or a page fetch does."""
        cmd: dict[str, Any] = {
            "type": "resumeSession", "taskId": sub_id, "tabId": sub_tab,
            "parentTabId": "tab-p",
        }
        if reopen:
            cmd["reopen"] = True
        await self._send(ws, cmd)
        return await self._announcement(ws, sub_tab)

    async def _new_surface(self, sub_tab: str) -> tuple[ClientConnection, dict[str, Any]]:
        """Connect a surface showing ``tab-p``; return it and its replay's child announcement."""
        ws = await self._connect_ok()
        await self._ready(ws, tab_id="tab-p")
        return ws, await self._announcement(ws, sub_tab)

    async def _replayed_on_a_new_surface(self, sub_tab: str) -> dict[str, Any]:
        """Connect a new surface showing ``tab-p``; return its announcement of the child."""
        _ws, ev = await self._new_surface(sub_tab)
        return ev

    async def test_the_reopening_is_announced_reopened_to_every_surface(self) -> None:
        """Another surface gets the reopening's announcement marked ``reopened``."""
        chat_id, sub_id, sub_tab = await self._seed_parent_with_finished_child()
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._open_parent(ws_a, chat_id, sub_tab)
        # B's own replay is over once it announced the child (closed).
        ws_b, replayed = await self._new_surface(sub_tab)
        self.assertIs(replayed["closedHistory"], True)

        mine = await self._resume_child(ws_a, sub_id, sub_tab, reopen=True)
        other = await self._announcement(ws_b, sub_tab)

        for ev in (mine, other):
            self.assertTrue(ev["isDone"])
            self.assertIs(ev.get("reopened"), True, ev)
            self.assertNotIn("closedHistory", ev)

    async def test_a_new_surface_replays_the_reopened_child_open(self) -> None:
        """A ready replay, and a deferred one, announce the reopened child ``reopened``."""
        chat_id, sub_id, sub_tab = await self._seed_parent_with_finished_child()
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._open_parent(ws_a, chat_id, sub_tab)
        await self._resume_child(ws_a, sub_id, sub_tab, reopen=True)

        on_screen = await self._replayed_on_a_new_surface(sub_tab)
        ws_d = await self._connect_ok()
        await self._ready(ws_d, tab_id="elsewhere")
        deferred = await self._announcement(ws_d, sub_tab)

        for ev in (on_screen, deferred):
            self.assertTrue(ev["isDone"])
            self.assertIs(ev.get("reopened"), True, ev)
            self.assertNotIn("closedHistory", ev)

    async def test_closing_the_reopened_tab_makes_it_closed_history_again(self) -> None:
        """Once a surface closes the reopened tab, a new surface keeps it closed."""
        chat_id, sub_id, sub_tab = await self._seed_parent_with_finished_child()
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._open_parent(ws_a, chat_id, sub_tab)
        await self._resume_child(ws_a, sub_id, sub_tab, reopen=True)

        await self._send(ws_a, {"type": "closeTab", "tabId": sub_tab})
        self.assertIsNotNone(await self._wait_for_event(
            ws_a, "closeSubagentTab", pred=lambda e: e.get("tab_id") == sub_tab,
        ))
        ev = await self._replayed_on_a_new_surface(sub_tab)

        self.assertIs(ev["closedHistory"], True)
        self.assertNotIn("reopened", ev)

    async def test_a_resume_without_reopen_is_no_reopening(self) -> None:
        """A page fetching the child's transcript neither marks it nor keeps it open."""
        chat_id, sub_id, sub_tab = await self._seed_parent_with_finished_child()
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._open_parent(ws_a, chat_id, sub_tab)

        fetched = await self._resume_child(ws_a, sub_id, sub_tab, reopen=False)
        ev = await self._replayed_on_a_new_surface(sub_tab)

        self.assertNotIn("reopened", fetched)
        self.assertIs(ev["closedHistory"], True)
        self.assertNotIn("reopened", ev)
