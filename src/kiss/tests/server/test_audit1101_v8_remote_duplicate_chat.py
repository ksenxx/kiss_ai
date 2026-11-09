# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 V8 F3: one tab per chat holds for a machine's chat too.

A tab bound to machine ``box`` shows the machine's chat.  A
``resumeSession`` of that same chat (its ``chatId`` and ``machine``,
a CHATS-panel click on another surface that landed before the first
tab's replay told it the chat is on screen) into a second tab used to
publish a second registry entry: two tabs, two streams, the same
transcript twice.  The registry displaces the other tab bound to a
local chat (``TabRegistry.update_tab``); a machine's chat is now
displaced the same way: the earlier tab's stream ends, its binding is
announced gone and its registry entry closes on every surface, as the
machine's own daemon already did on its side.

Round 2: the displaced tab's close runs after its stream's end (a
socket close handshake, outside the lock that retired it).  A local
chat another surface resumes into the tab meanwhile owns it (its
publication re-stamped the registry entry): the late close stands
down instead of removing that chat's tab.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence
from kiss.server.remote_tabs import RemoteTabs
from kiss.server.server import AgentServer
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.remote_tabs_probe import machine_of
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness


class TestOneTabPerRemoteChat(RemoteTabsHarness):
    """The later resume of a machine's chat displaces the tab that showed it."""

    def _registry(self) -> dict[str, dict[str, str]]:
        return {
            e["tabId"]: e for e in self.server._agent_server.tab_registry.snapshot()
        }

    async def _frames_until(self, reply_type: str, tab_id: str) -> list[dict[str, Any]]:
        """Every frame up to and including the first *reply_type* of *tab_id*."""
        frames: list[dict[str, Any]] = []
        for _ in range(300):
            msg = await self._next()
            frames.append(msg)
            if msg.get("type") == reply_type and msg.get("tabId") == tab_id:
                return frames
        raise AssertionError(f"no {reply_type} reply on {tab_id}")

    @staticmethod
    def _bindings_told(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The ``tabMachine`` announcements among *frames*, as ``{tabId, machine}``."""
        return [
            {"tabId": m.get("tabId"), "machine": m.get("machine")}
            for m in frames
            if m.get("type") == "tabMachine"
        ]

    async def test_resuming_a_shown_remote_chat_elsewhere_displaces_the_first_tab(self) -> None:
        first, second = "remote-first", "remote-second"
        await self._submit(first, "Say hello", "box")
        await self._run_until_idle(first)
        tabs = self.server._server_api.remote_tabs
        chat = tabs._tabs[first].chat_id
        self.assertTrue(chat, "the run's frames told the tab its chat")
        self.assertIn(first, self._registry())

        await self._send({
            "type": "resumeSession", "tabId": second, "chatId": chat, "machine": "box",
        })
        frames = await self._frames_until("task_events", second)
        self.assertEqual(frames[-1].get("chat_id"), chat)
        # The displaced tab: binding gone, stream ended, registry entry closed.
        self.assertIn({"tabId": first, "machine": ""}, self._bindings_told(frames))
        self.assertEqual(machine_of(tabs, first), "")
        self.assertEqual(machine_of(tabs, second), "box")
        self.assertEqual({tid: t.chat_id for tid, t in tabs._tabs.items()}, {second: chat})
        registry = self._registry()
        self.assertNotIn(first, registry, "the displaced tab left the registry")
        self.assertIn(second, registry)
        self.assertEqual(
            [m.get("type") for m in frames if m.get("tabId") == first],
            ["tabMachine"],
            "no frame of the chat reached the displaced tab",
        )

    async def test_a_tab_learning_its_chat_from_a_replay_displaces_the_chat_s_other_tab(
        self,
    ) -> None:
        first, second = "remote-learner", "remote-shown"
        await self._submit(first, "Say hello", "box")
        frames = await self._run_until_idle(first)
        task = next(str(m["taskId"]) for m in frames if m.get("type") == "result")
        tabs = self.server._server_api.remote_tabs
        chat = tabs._tabs[first].chat_id
        self.assertTrue(chat)
        await self._send({
            "type": "resumeSession", "tabId": second, "chatId": chat, "machine": "box",
        })
        await self._until("task_events", second)
        self.assertEqual(machine_of(tabs, first), "", "the chat moved to the second tab")

        # A resume by task alone: the tab learns the chat from the
        # replay's first frame, and the chat's other tab goes then.
        await self._send({
            "type": "resumeSession", "tabId": first, "taskId": task, "machine": "box",
        })
        frames = await self._frames_until("task_events", first)
        self.assertEqual(frames[-1].get("chat_id"), chat)
        # Told after the replay frame that carried the chat (its
        # ``status`` or its transcript): maybe among these frames, maybe next.
        if {"tabId": second, "machine": ""} not in self._bindings_told(frames):
            self.assertEqual((await self._until("tabMachine", second))["machine"], "")
        self.assertEqual(machine_of(tabs, second), "")
        self.assertEqual({tid: t.chat_id for tid, t in tabs._tabs.items()}, {first: chat})
        registry = self._registry()
        self.assertNotIn(second, registry)
        self.assertIn(first, registry)

    async def test_two_tabs_of_different_remote_chats_coexist(self) -> None:
        first, second = "remote-a", "remote-b"
        await self._submit(first, "Say hello", "box")
        await self._run_until_idle(first)
        await self._submit(second, "Say hello again", "box")
        await self._run_until_idle(second)
        tabs = self.server._server_api.remote_tabs
        chats = {tid: t.chat_id for tid, t in tabs._tabs.items()}
        self.assertEqual(set(chats), {first, second})
        self.assertNotEqual(chats[first], chats[second])
        self.assertTrue({first, second} <= set(self._registry()))

    async def test_a_local_chat_resumed_into_the_displaced_tab_meanwhile_keeps_it(
        self,
    ) -> None:
        """The late close of a displaced tab stands down when a newer publication owns it."""
        first, second = "remote-displaced", "remote-winner"
        await self._submit(first, "Say hello", "box")
        await self._run_until_idle(first)
        tabs = self.server._server_api.remote_tabs
        chat = tabs._tabs[first].chat_id
        self.assertTrue(chat)
        server = self.server._agent_server
        local_task, local_chat = persistence._add_task("LOCAL REPLACEMENT")
        persistence._save_task_result("<p>Local replacement</p>", task_id=local_task)
        reader2, writer2 = await open_local_connection(self.server)
        try:
            # The displacement is held at the retired stream's end: the
            # tab is already off the stream, its close has not run yet.
            with Pause(
                RemoteTabs._end_stream.__code__,
                lambda loc: getattr(loc.get("tab"), "tab_id", "") == first,
            ) as pause:
                await self._send({
                    "type": "resumeSession", "tabId": second, "chatId": chat,
                    "machine": "box",
                })
                self.assertTrue(
                    await asyncio.to_thread(pause.paused.wait, 20),
                    "the displacement never reached the stream's end",
                )
                self.assertEqual(machine_of(tabs, first), "")
                # Another surface resumes a local chat into the displaced tab.
                writer2.write(json.dumps({
                    "type": "resumeSession", "tabId": first, "chatId": local_chat,
                    "taskId": local_task,
                }).encode() + b"\n")
                await writer2.drain()
                for _ in range(200):
                    msg = json.loads(await asyncio.wait_for(reader2.readline(), 20))
                    if msg.get("type") == "task_events" and msg.get("tabId") == first:
                        self.assertEqual(msg.get("chat_id"), local_chat)
                        break
                else:
                    raise AssertionError("the local chat was never replayed")
                self.assertEqual(server.tab_registry.bindings()[first], local_chat)
                self.assertEqual(server._tab_chat_views[first], local_chat)
                pause.release()
                await self._until("task_events", second)
            # The old displacement's close stood down: the local chat's tab stays.
            self.assertEqual(server.tab_registry.bindings().get(first), local_chat)
            self.assertEqual(server._tab_chat_views.get(first), local_chat)
            self.assertEqual(machine_of(tabs, second), "box")
            self.assertEqual({tid: t.chat_id for tid, t in tabs._tabs.items()}, {second: chat})
        finally:
            writer2.close()

    async def test_a_late_title_restore_of_the_displaced_tab_does_not_keep_it(self) -> None:
        """The displaced stream's own late ``promptOpened`` restore does not defeat the close."""
        first, second = "remote-restored", "remote-taker"
        await self._submit(first, "Say hello", "box")
        await self._run_until_idle(first)
        tabs = self.server._server_api.remote_tabs
        chat = tabs._tabs[first].chat_id
        self.assertTrue(chat)
        target = Path(self.remote.home) / "notes.md"
        target.write_text("first draft\n", encoding="utf-8", newline="")
        # The machine's answer to the path-only prompt is held at the
        # title restore; the displacement at its close, after the
        # capture of the tab's generation.  The end of the stream's
        # relay thread (the displacement closed its socket) tells when
        # the restore has run: the machine's next frame, the file's
        # content, is no signal, as under load it is sent after the
        # close and never arrives.
        restore = Pause(RemoteTabs.retitle.__code__, lambda loc: loc.get("tab_id") == first)
        close = Pause(
            AgentServer._close_tab_if_generation.__code__,
            lambda loc: loc.get("tab_id") == first,
        )
        with Pauses(restore, close):
            await self._submit(first, str(target), "box")
            self.assertTrue(
                await asyncio.to_thread(restore.paused.wait, 20),
                "the machine never answered the path-only prompt",
            )
            relay = tabs._tabs[first]._thread
            await self._send({
                "type": "resumeSession", "tabId": second, "chatId": chat, "machine": "box",
            })
            self.assertTrue(
                await asyncio.to_thread(close.paused.wait, 20),
                "the displacement never reached the close",
            )
            self.assertEqual(machine_of(tabs, first), "")
            restore.release()
            await asyncio.to_thread(relay.join, 20)
            self.assertFalse(relay.is_alive(), "the relay never went past the restore")
            close.release()
            await self._until("task_events", second)
        self.assertNotIn(first, self._registry(), "the displaced tab left the registry")
        self.assertIn(second, self._registry())
        self.assertEqual(machine_of(tabs, second), "box")
        self.assertEqual({tid: t.chat_id for tid, t in tabs._tabs.items()}, {second: chat})


if __name__ == "__main__":
    import unittest

    unittest.main()
