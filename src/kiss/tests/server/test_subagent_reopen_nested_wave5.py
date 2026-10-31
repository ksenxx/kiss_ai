# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A reopened finished grandchild is shared by every surface, a later one too.

A parent chat fanned out to a child, which fanned out to a grandchild;
both are finished.  A user expands the parent's fan-out panel
(reopening the child) and then the child's panel (reopening the
grandchild): that surface shows both tabs.  Every other surface must
show the same, including one that connects afterwards and one that
fetches the child's transcript later.  A surface opens a finished
sub-agent only below an open parent, so the child's announcement must
come before the grandchild's.  A grandchild nobody reopened, or whose
reopened tab was closed again, stays closed.  A real daemon and real
``wss://`` clients; nothing is mocked.  One test also plays the frames
the new surface got into the real chat page (``media/main.js`` under
jsdom) and reads which sub-agent tabs it shows.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from websockets.asyncio.client import ClientConnection

from kiss.tests.server.test_ready_replay_conn_scoped import _seed_subagent_row
from kiss.tests.server.test_web_server_tab_mirroring import TabMirroringBase

_WEBAPP_TEST_DIR = Path(__file__).resolve().parents[2] / "webapp" / "test"

# Plays the frames on stdin into a fresh page showing the tab argv[1];
# prints the sub-agent tabs it shows and the tabs it asked the daemon
# to close (a close is mirrored to every surface).
_PAGE_SCRIPT = r"""
const {makePreloadedPage, send} = require(process.argv[2] + '/simplify2_harness');
const {openSubTabIds} = require(process.argv[2] + '/subagent_tab_helpers');
const page = makePreloadedPage(process.argv[1]);
for (const ev of JSON.parse(require('fs').readFileSync(0, 'utf8'))) send(page.win, ev);
process.stdout.write(JSON.stringify({
  shown: openSubTabIds(page.win),
  closed: page.posted.filter(m => m.type === 'closeTab').map(m => m.tabId),
}));
page.close();
"""


@dataclass(frozen=True)
class Frame:
    """A daemon frame of type *kind* addressed to tab *tab* under the key *key*."""

    kind: str
    tab: str
    key: str = "tabId"

    def matches(self, ev: dict[str, Any]) -> bool:
        """Return whether *ev* is this frame."""
        return ev.get("type") == self.kind and ev.get(self.key) == self.tab


def _tab_frames(frames: list[dict[str, Any]], tabs: set[str]) -> list[tuple[str, str, bool]]:
    """The (type, tab, reopened) of every sub-agent tab frame of *frames* about *tabs*."""
    return [
        (ev["type"], ev["tab_id"], ev.get("reopened") is True)
        for ev in frames
        if ev.get("type") in ("openSubagentTab", "subagentDone") and ev.get("tab_id") in tabs
    ]


class TestNestedSubagentReopenIsShared(TabMirroringBase):
    """Reopened grandchildren replay open, after their parent; closed ones stay closed."""

    def _seed_three_generations(self) -> tuple[str, str, str, str, str]:
        """Record parent -> child -> grandchild, all finished; return chat, ids and tab ids."""
        parent_id, chat_id = self._seed_chat("Parent task", [
            {"type": "prompt", "text": "fan out"},
        ])
        child_id = _seed_subagent_row(parent_id, chat_id, "child transcript")
        grand_id = _seed_subagent_row(child_id, chat_id, "grandchild transcript")
        child_tab = f"tab-p__sub_{child_id}"
        return chat_id, child_id, grand_id, child_tab, f"{child_tab}__sub_{grand_id}"

    async def _frames_until(
        self, ws: ClientConnection, *wanted: Frame, timeout: float = 8.0,
    ) -> list[dict[str, Any]]:
        """Every frame *ws* gets until each of *wanted* has come."""
        frames: list[dict[str, Any]] = []
        missing = list(wanted)
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                break
            frames.append(json.loads(raw))
            missing = [frame for frame in missing if not frame.matches(frames[-1])]
            if not missing:
                return frames
        got = [(ev.get("type"), ev.get("tab_id") or ev.get("tabId")) for ev in frames]
        self.fail(f"{missing} never came; got {got}")

    async def _resume(
        self, ws: ClientConnection, task_id: str, tab_id: str, **extra: Any,
    ) -> None:
        """Send ``resumeSession`` for sub-agent *task_id* shown in *tab_id*."""
        await self._send(ws, {
            "type": "resumeSession", "taskId": task_id, "tabId": tab_id, **extra,
        })

    async def _shown(self, ws: ClientConnection, tab_id: str) -> dict[str, Any]:
        """Wait for sub-agent *tab_id*'s announcement and transcript on *ws*; return the first."""
        frames = await self._frames_until(
            ws, Frame("openSubagentTab", tab_id, "tab_id"), Frame("task_events", tab_id),
        )
        return next(ev for ev in frames if Frame("openSubagentTab", tab_id, "tab_id").matches(ev))

    async def _surface_with_parent_open(self, chat_id: str, child_tab: str) -> ClientConnection:
        """Connect A, open ``tab-p`` and resume *chat_id* in it (a history click)."""
        ws = await self._connect_ok()
        await self._ready(ws)
        await self._send(ws, {"type": "openTab", "tabId": "tab-p", "title": "new chat"})
        self.assertIsNotNone(await self._wait_for_snapshot_with(ws, present={"tab-p"}))
        await self._send(ws, {"type": "resumeSession", "chatId": chat_id, "tabId": "tab-p"})
        await self._shown(ws, child_tab)
        return ws

    async def _reopen(
        self, ws: ClientConnection, task_id: str, tab_id: str, parent_tab: str,
    ) -> None:
        """Expand the panel naming *task_id* on *ws*; check its announcement says ``reopened``."""
        await self._resume(ws, task_id, tab_id, parentTabId=parent_tab, reopen=True)
        announcement = await self._shown(ws, tab_id)
        self.assertIs(announcement.get("reopened"), True, announcement)

    async def _both_reopened(self) -> tuple[ClientConnection, str, str, str]:
        """Surface A reopens the child, then the grandchild; return A, the child and both tabs."""
        chat_id, child_id, grand_id, child_tab, grand_tab = self._seed_three_generations()
        ws_a = await self._surface_with_parent_open(chat_id, child_tab)
        await self._reopen(ws_a, child_id, child_tab, "tab-p")
        await self._reopen(ws_a, grand_id, grand_tab, child_tab)
        return ws_a, child_id, child_tab, grand_tab

    async def _replay_of_new_surface(
        self, shown_tab: str, *wanted: Frame,
    ) -> list[dict[str, Any]]:
        """Connect a surface showing *shown_tab*; return its replay's frames until *wanted*."""
        ws = await self._connect_ok()
        await self._ready(ws, tab_id=shown_tab)
        return await self._frames_until(ws, *wanted)

    async def test_new_surfaces_open_the_reopened_grandchild_after_its_parent(self) -> None:
        """A ready replay, on screen or deferred, opens the child, then the grandchild."""
        _ws_a, _child_id, child_tab, grand_tab = await self._both_reopened()
        for shown_tab in ("tab-p", "elsewhere"):
            frames = await self._replay_of_new_surface(
                shown_tab, Frame("replay_deferred", grand_tab),
            )
            self.assertEqual(
                _tab_frames(frames, {child_tab, grand_tab}),
                [("openSubagentTab", child_tab, True), ("openSubagentTab", grand_tab, True)],
                shown_tab,
            )

    async def test_a_new_page_shows_the_reopened_child_and_grandchild(self) -> None:
        """The real page fed a new surface's replay shows both tabs and closes nothing."""
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        _ws_a, _child_id, child_tab, grand_tab = await self._both_reopened()
        frames = await self._replay_of_new_surface(
            "tab-p", Frame("replay_deferred", grand_tab),
        )
        page = await asyncio.to_thread(
            subprocess.run,
            [
                "node", "--no-maglev", "--no-concurrent-sparkplug",
                "-e", _PAGE_SCRIPT, "tab-p", str(_WEBAPP_TEST_DIR),
            ],
            input=json.dumps(frames),
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_WEBAPP_TEST_DIR.parent),
        )
        self.assertEqual(page.returncode, 0, page.stderr)
        self.assertEqual(
            json.loads(page.stdout), {"shown": [child_tab, grand_tab], "closed": []},
        )

    async def test_fetching_the_child_reannounces_the_reopened_grandchild(self) -> None:
        """A page fetching the child, or a panel expanded again, re-announces the grandchild."""
        ws_a, child_id, child_tab, grand_tab = await self._both_reopened()
        ws_b = await self._connect_ok()
        await self._ready(ws_b, tab_id="tab-p")
        await self._frames_until(ws_b, Frame("replay_deferred", grand_tab))

        # B activates the child's deferred tab: its transcript, then its fan-out.
        await self._resume(ws_b, child_id, child_tab, scoped=True)
        fetched = await self._frames_until(ws_b, Frame("replay_deferred", grand_tab))
        # A expands the parent's panel again (a broadcast reopening of the child).
        await self._resume(ws_a, child_id, child_tab, parentTabId="tab-p", reopen=True)
        expanded = await self._frames_until(ws_a, Frame("task_events", grand_tab))

        for frames in (fetched, expanded):
            self.assertEqual(
                _tab_frames(frames, {grand_tab}), [("openSubagentTab", grand_tab, True)],
            )

    async def test_a_grandchild_not_reopened_or_closed_again_stays_closed(self) -> None:
        """Only the child reopened, or the grandchild's reopened tab closed: it stays closed."""
        chat_id, child_id, grand_id, child_tab, grand_tab = self._seed_three_generations()
        ws_a = await self._surface_with_parent_open(chat_id, child_tab)
        await self._reopen(ws_a, child_id, child_tab, "tab-p")
        replay_over = (
            Frame("openSubagentTab", child_tab, "tab_id"),
            Frame("subagentDone", grand_tab, "tab_id"),
        )
        replays = [await self._replay_of_new_surface("tab-p", *replay_over)]
        await self._reopen(ws_a, grand_id, grand_tab, child_tab)
        await self._send(ws_a, {"type": "closeTab", "tabId": grand_tab})
        await self._frames_until(ws_a, Frame("closeSubagentTab", grand_tab, "tab_id"))
        replays.append(await self._replay_of_new_surface("tab-p", *replay_over))

        for frames in replays:
            self.assertEqual(
                sorted(_tab_frames(frames, {child_tab, grand_tab})),
                [("openSubagentTab", child_tab, True), ("subagentDone", grand_tab, False)],
            )
