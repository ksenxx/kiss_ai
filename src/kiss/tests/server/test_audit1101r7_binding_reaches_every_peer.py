# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 round 7: a tab's machine binding reaches every peer, before the tab's frames.

``test_sub_agents_of_a_chat_shown_from_a_machine_are_fetched`` failed
under a 98-way parallel run with ``KeyError: 'tabMachine'``: the
sub-agent tab's transcript reached the client before the tab's
binding.  Two windows in :mod:`kiss.server.remote_tabs` allowed it:

* ``_resume`` made the sub-agent tab one of its parent's stream
  (``tab_ids.add``) and announced the binding only after releasing the
  locks; a frame of the tab already on the stream (the machine replays
  a finished sub-agent's transcript to every window right after the
  parent's) was relayed in between.  The binding is now delivered
  under the stream's lock, with the join.
* ``tabMachine`` was a tab-stamped copy, withheld from a local peer
  that had addressed only other tabs (``_local_clients_for_tab``): a
  sub-agent tab adopted by the parent's replay announced its binding
  before the client ever addressed it, so the client never got it, and
  its own resume re-announced only after the tab's frames could reach
  it.  A binding is now a registry event, delivered to every client as
  ``new_tab`` and ``subagentDone`` are.

The first window is timing (the probe that widened it failed 4 of 5
runs on the old code, none on the new); the second is deterministic
and pinned here.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness, _FinishingModel


class TestBindingReachesEveryPeer(RemoteTabsHarness):
    """A ``tabMachine`` is not narrowed to the peers that addressed the tab."""

    async def test_a_binding_reaches_a_peer_that_addressed_only_other_tabs(self) -> None:
        """A second local peer, busy with its own tab, still learns a tab bound by this one."""
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(
                json.dumps({"type": "openTab", "tabId": "tab-other", "title": "other"}).encode()
                + b"\n",
            )
            await writer.drain()
            for _ in range(50):
                line = await asyncio.wait_for(reader.readline(), timeout=30.0)
                if json.loads(line.decode("utf-8")).get("type") == "tabs_state":
                    break
            else:
                self.fail("the other peer never got its tabs_state")
            await self._submit("tab-p", "Say hello", "box")
            await self._run_until_idle("tab-p")
            for _ in range(200):
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=10.0)
                except TimeoutError:
                    self.fail("the binding of tab-p never reached the peer busy with tab-other")
                msg: dict[str, Any] = json.loads(line.decode("utf-8"))
                if msg.get("type") == "tabMachine" and msg.get("tabId") == "tab-p":
                    self.assertEqual(msg["machine"], "box")
                    return
            self.fail("200 frames without the binding of tab-p")
        finally:
            writer.close()

    async def test_a_sub_agent_tab_adopted_by_the_parents_replay_is_bound_before_its_frames(
        self,
    ) -> None:
        """A history click's replay announces a finished sub-agent's tab; its binding comes first.

        The clicking peer never addressed the sub-agent tab: the binding
        must reach it all the same, ahead of the ``openSubagentTab``
        that adopts the tab onto the parent's stream.
        """
        _FinishingModel.parallel_on_first_call = True
        await self._submit("tab-p", "Fan out to a sub-agent", "box")
        child_task = ""
        for _ in range(500):
            msg = await self._next()
            if msg.get("type") == "new_tab" and msg.get("parent_tab_id") == "tab-p":
                child_task = str(msg["task_id"])
            if msg.get("type") == "status" and msg.get("tabId") == "tab-p" and not msg["running"]:
                break
        self.assertTrue(child_task, "the run fanned out")
        await self._send({"type": "closeTab", "tabId": "tab-p"})
        await self._until("tabs_state")
        tab = "tab-click"
        await self._send({"type": "openTab", "tabId": tab, "title": "Fan out to a sub-agent"})
        history = await self._ask({"type": "getHistory", "machine": "box"}, "history")
        row = next(s for s in history["sessions"] if s["title"] == "Fan out to a sub-agent")
        await self._send({
            "type": "resumeSession", "id": row["id"], "taskId": row["task_id"],
            "tabId": tab, "machine": "box", "title": row["title"],
        })
        sub = f"{tab}__sub_{child_task}"
        seen: list[tuple[str, str]] = []
        for _ in range(300):
            msg = await self._next()
            if msg.get("type") == "tabMachine" and msg.get("tabId") == sub:
                seen.append(("tabMachine", str(msg.get("machine"))))
            elif msg.get("type") == "openSubagentTab" and msg.get("tab_id") == sub:
                seen.append(("openSubagentTab", ""))
                break
        else:
            self.fail(f"the parent's replay never announced the sub-agent tab: {seen}")
        self.assertEqual(seen[0], ("tabMachine", "box"), seen)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(sub), "box")
