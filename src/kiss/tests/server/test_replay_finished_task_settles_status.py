# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a finished task's replay settles the tab as idle.

A replay of a finished task (a history row with nothing live attached)
used to carry no ``status`` at all, so the chat page kept whatever
running flag the tab had.  A tab that saw its task's ``running:true``,
lost its connection and missed the task's end got the finished
transcript on reconnect while still marked running: it kept a Stop
button and landed at the end of the transcript instead of on its task
panel (``main.js`` ``landOpenedTranscript`` lands a running tab at the
end, a finished one on its task panel).

``AgentServer._replay_session`` now sends ``status running:false``
BEFORE the ``task_events`` of a finished task
(:meth:`AgentServer._broadcast_replay_status`), scoped like the
transcript; a tab that meanwhile owns a live run of its own gets its
``running:true`` re-asserted instead.

The first two tests drive a real :class:`RemoteAccessServer` over real
``wss://`` connections (the harness of ``test_web_server_tab_mirroring``);
the last drives :class:`AgentServer` in-process with a real registered
:class:`AgentState`, the only way to hold a tab busy without an LLM.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import time
import unittest
from typing import Any

from websockets.asyncio.client import ClientConnection

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import (
    clear_agent_states,
    redirect_persistence,
    restore_persistence,
)
from kiss.tests.server.test_web_server_tab_mirroring import TabMirroringBase

_TAB = "tab-a"


class TestReplayFinishedTaskSettlesStatus(TabMirroringBase):
    """``status running:false`` precedes a finished task's transcript."""

    async def _frames_until_replay(
        self, ws: ClientConnection, timeout: float = 8.0,
    ) -> list[dict[str, Any]]:
        """Return the ``status``/``task_events`` frames of the tab up to its replay.

        Collection stops at the first ``task_events`` of the tab (it is
        included) or at *timeout*, whichever comes first.
        """
        got: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return got
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                return got
            ev = json.loads(raw)
            if ev.get("tabId") != _TAB or ev.get("type") not in ("status", "task_events"):
                continue
            got.append(dict(ev))
            if ev["type"] == "task_events":
                return got

    def _assert_idle_then_transcript(
        self, frames: list[dict[str, Any]], task_id: str, label: str,
    ) -> None:
        """*frames* must be exactly ``status running:false`` then the transcript."""
        self.assertEqual(
            [f["type"] for f in frames], ["status", "task_events"],
            label + ": frames " + str([f["type"] for f in frames]),
        )
        self.assertIs(frames[0]["running"], False, label + ": the status is idle")
        self.assertEqual(str(frames[1]["task_id"]), task_id, label + ": wrong task")

    async def _open_tab(self, ws: ClientConnection) -> None:
        """Open the tab on *ws* and wait for its registry snapshot."""
        await self._send(ws, {"type": "openTab", "tabId": _TAB, "title": "new chat"})
        self.assertIsNotNone(await self._wait_for_snapshot_with(ws, present={_TAB}))

    async def test_history_click_sends_idle_status_before_transcript(self) -> None:
        """A user's resume (broadcast) settles the tab idle in every window."""
        task_id, chat_id = self._seed_chat("Finished task", [
            {"type": "prompt", "text": "the prompt"},
            {"type": "text", "text": "the answer"},
        ])
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)
        await self._open_tab(ws_a)
        self.assertIsNotNone(await self._wait_for_snapshot_with(ws_b, present={_TAB}))
        await self._send(ws_a, {"type": "resumeSession", "chatId": chat_id, "tabId": _TAB})
        for label, ws in (("clicking window", ws_a), ("mirroring window", ws_b)):
            self._assert_idle_then_transcript(
                await self._frames_until_replay(ws), task_id, label,
            )

    async def test_ready_replay_idle_status_is_scoped_to_the_reconnecting_client(
        self,
    ) -> None:
        """A reconnect's replay carries the idle status to that client alone."""
        task_id, chat_id = self._seed_chat("Finished task", [
            {"type": "prompt", "text": "the prompt"},
            {"type": "text", "text": "the answer"},
        ])
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._open_tab(ws_a)
        await self._send(ws_a, {"type": "resumeSession", "chatId": chat_id, "tabId": _TAB})
        self._assert_idle_then_transcript(
            await self._frames_until_replay(ws_a), task_id, "history click",
        )
        # A window that missed the task's end reconnects showing the tab.
        ws_c = await self._connect_ok()
        await self._ready(ws_c, tab_id=_TAB)
        self._assert_idle_then_transcript(
            await self._frames_until_replay(ws_c), task_id, "reconnecting window",
        )
        leaked = await self._frames_until_replay(ws_a, timeout=1.5)
        self.assertEqual(leaked, [], "a conn-scoped replay reached another window")


class TestReplayIdleStatusSuppressedForBusyTab(unittest.TestCase):
    """A tab that owns a live run keeps its spinner through a replay.

    The owner decides the status that precedes a transcript with
    nothing live attached: a running owner is re-asserted
    (``running:true``, ``attached``), never told ``running:false``.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-replay-idle-")
        self.saved = redirect_persistence(self.tmpdir)
        clear_agent_states()

    def tearDown(self) -> None:
        clear_agent_states()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_busy_owner_gets_the_transcript_but_no_idle_status(self) -> None:
        """The replay delivers the transcript; the status re-asserts the live run."""
        task_id, chat_id = th._add_task("Finished task")
        th._append_chat_event({"type": "text", "text": "the answer"}, task_id=task_id)
        th._save_task_result(task_id, "success: done")
        server = AgentServer()
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            with lock:
                events.append(event)

        server.printer.broadcast = capture  # type: ignore[assignment]
        # The tab's own run (of another chat) is live: `_cmd_run`
        # installs exactly such a state before its worker starts.
        live = AgentState("live-task", chat_id="other-chat", tab_id=_TAB, server_owned=True)
        live.is_task_active = True
        agent_state.register(live)

        server._replay_session(chat_id, _TAB)

        with lock:
            frames = [e for e in events if e.get("tabId") == _TAB]
        self.assertEqual(
            [e["type"] for e in frames if e["type"] == "task_events"],
            ["task_events"],
            "the finished transcript is still delivered",
        )
        statuses = [e for e in frames if e["type"] == "status"]
        self.assertEqual(
            [e["running"] for e in statuses],
            [True],
            "a stale running:false would kill the live run's spinner",
        )
        self.assertTrue(statuses[0].get("attached"), "a catch-up, not a run starting")
        self.assertEqual(
            [e["type"] for e in frames][:2],
            ["status", "task_events"],
            "the status precedes the transcript",
        )


if __name__ == "__main__":
    unittest.main()
