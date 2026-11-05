# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 V8 F1: a local chat resumed into a joined sub-agent tab leaves the stream.

A sub-agent tab of a run on machine ``box`` joined its parent's stream
(``resumeSession {taskId}``).  A ``resumeSession`` naming a LOCAL chat
(``chatId``) into that tab is handled here: the daemon replays the
local chat into it and publishes the tab with that chat.  The tab must
leave the machine's stream first, or the remote sub-agent's later
``result`` is relayed, stamped with the tab, into the local chat's
view, while the header's drop-down still names the machine.

A ``taskId``-only resume of a task the stream never saw still leaves a
sub-agent tab on its stream (``test_remote_tab_binding_sweep1030``):
that is the sub-agent tab's own fetch, not a chat click.

Audit 2026-11-04 V8: the tab that left must be gone for good.  The
remote sub-agent's ``subagentDone`` names the tab (the stream had
kept the task it was derived from), and the webapp closes the tab it
names: the local chat vanished from every surface.  And a frame of the
tab the relay thread had already found eligible (the remote's replay
of the sub-agent's transcript, held between the relay's two lock
holds) was delivered after the tab left and after the local chat's
replay, replacing it on screen.  The same for the sub-agent's
``subagentDone``, a global frame naming the tab in ``tab_id``: found
to be the stream's, then held, it closed the tab the local chat had
taken meanwhile.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from types import FrameType
from typing import Any

from kiss.agents.sorcar import persistence
from kiss.server.remote_tabs import RemoteTab
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness, _FinishingModel


class _HoldBetweenLockHolds:
    """Hold the thread relaying a *kind* frame of *tab_id* between the two lock holds.

    A ``threading.settrace_all_threads`` hook on the real
    :meth:`RemoteTab._relay_frame` (nothing is replaced), as
    :class:`kiss.tests.server.pause.Pause`: the relay thread is held
    at the line of the second ``with self.lock:``, after the frame's
    eligibility was decided and the lock released, so a local resume
    of the tab lands inside that window for sure.  ``held`` is set
    once the thread is held, ``passed`` once it returned.
    """

    def __init__(self, tab_id: str, kind: str = "task_events") -> None:
        self._tab_id = tab_id
        self._kind = kind
        self._code = RemoteTab._relay_frame.__code__
        lines, first = inspect.getsourcelines(RemoteTab._relay_frame)
        self._line = [first + i for i, s in enumerate(lines) if s.strip() == "with self.lock:"][1]
        self._frame: FrameType | None = None
        self.held = threading.Event()
        self.passed = threading.Event()
        self._go = threading.Event()

    def __enter__(self) -> _HoldBetweenLockHolds:
        threading.settrace_all_threads(self._trace)
        return self

    def __exit__(self, *exc: object) -> None:
        threading.settrace_all_threads(None)  # type: ignore[arg-type]
        self.release()

    def release(self) -> None:
        """Let the held thread go on."""
        self._go.set()

    def _trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is not self._code:
            return None
        msg = frame.f_locals.get("msg") or {}
        if msg.get("type") != self._kind:
            return None
        if self._tab_id not in (msg.get("tabId"), msg.get("tab_id")):
            return None
        if event == "line" and frame.f_lineno == self._line and not self.held.is_set():
            self._frame = frame
            self.held.set()
            self._go.wait(60)
        elif event == "return" and frame is self._frame:
            self.passed.set()
        return self._trace


class TestLocalChatResumeLeavesTheStream(RemoteTabsHarness):
    """The remote child's output never reaches the tab that shows a local chat."""

    async def test_local_chat_resume_unbinds_a_joined_sub_agent_tab(self) -> None:
        root = "root-for-local-takeover"
        spawned = await self._fan_out(root)
        remote_task = str(spawned["task_id"])
        child = f"{root}__sub_{remote_task}"
        await self._send({"type": "resumeSession", "taskId": remote_task, "tabId": child})
        await self._until("task_events", child)
        tabs = self.server._server_api.remote_tabs
        self.assertEqual(tabs.machine_of(child), "box", "the sub-agent tab joined the stream")

        local_task, local_chat = persistence._add_task("ONLY LOCAL CHAT CONTENT")
        persistence._save_task_result("<p>Only local result</p>", task_id=local_task)
        await self._send({
            "type": "resumeSession", "taskId": local_task, "chatId": local_chat, "tabId": child,
        })
        frames: list[dict[str, Any]] = []
        for _ in range(300):
            msg = await self._next()
            frames.append(msg)
            if msg.get("type") == "task_events" and msg.get("tabId") == child:
                break
        else:
            raise AssertionError("the local chat was never replayed into the tab")
        self.assertEqual(frames[-1].get("chat_id"), local_chat)
        # The binding is told gone before the transcript, as a bound
        # tab's unbinding is: the drop-down shows this machine.
        self.assertIn(
            {"type": "tabMachine", "tabId": child, "machine": ""},
            [{k: m.get(k) for k in ("type", "tabId", "machine")} for m in frames],
        )
        self.assertEqual(tabs.machine_of(child), "")
        self.assertEqual(tabs.machine_of(root), "box", "the parent stays bound")
        self.assertNotIn(child, tabs._tabs[root].tab_ids)

        # The remote sub-agent finishes: its result is the machine's
        # chat's, not the local chat's, and its ``subagentDone`` (which
        # closes the tab it names on every surface) is not the local
        # chat's tab's either.  The parent's run ends after it.
        _FinishingModel.released.set()
        leaked: list[dict[str, Any]] = []
        for _ in range(400):
            msg = await self._next()
            if msg.get("tabId") == child and msg.get("type") in ("result", "status"):
                leaked.append(msg)
            if msg.get("type") == "subagentDone" and msg.get("tab_id") == child:
                leaked.append(msg)
            if (
                msg.get("tabId") == root
                and msg.get("type") == "status"
                and not msg.get("running")
                and not msg.get("replayId")
            ):
                break
        else:
            raise AssertionError("the parent's run never ended")
        self.assertEqual(
            [m for m in leaked if str(m.get("taskId")) != local_task],
            [],
            "no frame of the remote sub-agent reached the local chat's tab",
        )

    async def test_frame_in_flight_is_dropped_once_the_tab_left_the_stream(self) -> None:
        root = "root-for-late-replay"
        spawned = await self._fan_out(root)
        remote_task = str(spawned["task_id"])
        child = f"{root}__sub_{remote_task}"
        local_task, local_chat = persistence._add_task("ONLY LOCAL CHAT CONTENT")
        tabs = self.server._server_api.remote_tabs
        with _HoldBetweenLockHolds(child) as hold:
            # The join's replay of the remote transcript is found
            # eligible, then held with the stream's lock released.
            await self._send({"type": "resumeSession", "taskId": remote_task, "tabId": child})
            self.assertTrue(await asyncio.to_thread(hold.held.wait, 30), "no remote replay")
            self.assertEqual(tabs.machine_of(child), "box", "the sub-agent tab joined the stream")
            await self._send({
                "type": "resumeSession", "taskId": local_task, "chatId": local_chat,
                "tabId": child,
            })
            local = await self._until("task_events", child)
            self.assertEqual(local.get("chat_id"), local_chat)
            self.assertEqual(tabs.machine_of(child), "", "the tab left the stream")
            hold.release()
            self.assertTrue(await asyncio.to_thread(hold.passed.wait, 30), "relay stuck")
        # The relay returned: what it delivered was sent before the
        # reply to a request made now.
        await self._send({"type": "getHistory"})
        stale: list[dict[str, Any]] = []
        for _ in range(300):
            msg = await self._next()
            if msg.get("type") == "history":
                break
            if msg.get("type") == "task_events" and msg.get("tabId") == child:
                stale.append(msg)
        else:
            raise AssertionError("no history reply")
        self.assertEqual(
            [str(m.get("task_id")) for m in stale],
            [],
            "the remote transcript reached the tab after the local chat's",
        )

    async def test_a_done_in_flight_cannot_close_the_tab_that_left_the_stream(self) -> None:
        root = "root-for-late-done"
        spawned = await self._fan_out(root)
        remote_task = str(spawned["task_id"])
        child = f"{root}__sub_{remote_task}"
        await self._send({"type": "resumeSession", "taskId": remote_task, "tabId": child})
        await self._until("task_events", child)
        local_task, local_chat = persistence._add_task("ONLY LOCAL CHAT CONTENT")
        tabs = self.server._server_api.remote_tabs
        with _HoldBetweenLockHolds(child, "subagentDone") as hold:
            # The sub-agent's end is found to be the stream's to relay,
            # then held with the stream's lock released.
            _FinishingModel.released.set()
            self.assertTrue(await asyncio.to_thread(hold.held.wait, 30), "no subagentDone")
            self.assertEqual(tabs.machine_of(child), "box", "the sub-agent tab joined the stream")
            await self._send({
                "type": "resumeSession", "taskId": local_task, "chatId": local_chat,
                "tabId": child,
            })
            local = await self._until("task_events", child)
            self.assertEqual(local.get("chat_id"), local_chat)
            self.assertEqual(tabs.machine_of(child), "", "the tab left the stream")
            hold.release()
            self.assertTrue(await asyncio.to_thread(hold.passed.wait, 30), "relay stuck")
        await self._send({"type": "getHistory"})
        stale: list[dict[str, Any]] = []
        for _ in range(300):
            msg = await self._next()
            if msg.get("type") == "history":
                break
            if msg.get("type") == "subagentDone" and msg.get("tab_id") == child:
                stale.append(msg)
        else:
            raise AssertionError("no history reply")
        self.assertEqual(stale, [], "the sub-agent's end closed the tab the local chat had taken")


if __name__ == "__main__":
    import unittest

    unittest.main()
