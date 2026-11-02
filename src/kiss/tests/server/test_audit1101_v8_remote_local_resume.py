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
"""

from __future__ import annotations

from typing import Any

from kiss.agents.sorcar import persistence
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness, _FinishingModel


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
        # chat's, not the local chat's.  The parent's run ends after it.
        _FinishingModel.released.set()
        leaked: list[dict[str, Any]] = []
        for _ in range(400):
            msg = await self._next()
            if msg.get("tabId") == child and msg.get("type") in ("result", "status"):
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


if __name__ == "__main__":
    import unittest

    unittest.main()
