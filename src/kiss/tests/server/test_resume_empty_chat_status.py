# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: resuming a chat with nothing to replay reports it idle.

The webview keeps a chat adopted from the tab registry open until the
daemon has told it the chat's state (``statusKnown`` in main.js): the
``status`` or ``task_events`` of the replay every ``ready`` triggers.
A chat with neither a history row nor a live run has nothing to
replay, so ``_replay_session`` must send an explicit
``status running:false`` instead of nothing, scoped to the connection
whose ``ready`` asked for it.
"""

from __future__ import annotations

import uuid

from kiss.tests.server.test_web_server_tab_mirroring import TabMirroringBase


class TestResumeEmptyChatStatus(TabMirroringBase):
    """A resume with nothing to replay still settles the tab's state."""

    async def test_resume_without_history_or_run_reports_idle(self) -> None:
        """An unknown chat yields ``status running:false`` for its tab."""
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)
        chat_id = "chat-" + uuid.uuid4().hex
        # A user-initiated resume (no replayConnId) is broadcast: both
        # windows learn the tab is idle.
        await self._send(ws_a, {
            "type": "resumeSession", "chatId": chat_id, "tabId": "tab-n",
        })
        for ws in (ws_a, ws_b):
            ev = await self._wait_for_event(
                ws, "status", pred=lambda e: e.get("tabId") == "tab-n",
            )
            self.assertIsNotNone(ev, "no status for the empty chat")
            assert ev is not None
            self.assertIs(ev["running"], False)

        # The resume bound tab-n to the chat in the registry: a window
        # connecting now replays it through its own ready, and the
        # idle status reaches that window alone.
        ws_c = await self._connect_ok()
        await self._ready(ws_c)
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_c, present={"tab-n"}),
            "the resume did not register the tab",
        )
        ev_c = await self._wait_for_event(
            ws_c, "status", pred=lambda e: e.get("tabId") == "tab-n",
        )
        self.assertIsNotNone(ev_c, "the connecting window was not told")
        assert ev_c is not None
        self.assertIs(ev_c["running"], False)
        leaked = await self._wait_for_event(
            ws_a, "status", timeout=1.5,
            pred=lambda e: e.get("tabId") == "tab-n",
        )
        self.assertIsNone(leaked, "a conn-scoped status reached another window")
