# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Taking a bound tab off its stream ends the sub-agent tabs the stream relayed.

A root tab bound to a machine relays the ``new_tab`` of every sub-agent
its run spawns, and the surfaces open a ``{root}__sub_{task}`` tab for
each.  Once the root tab leaves the stream (``newChat`` on it, a submit
that names another machine, a History-panel click into it, or a
resume that joins it to another stream) no frame of the machine can
reach those tabs any more, so nothing but this daemon can stop their
spinners: it ends them with ``subagentDone`` as it does for a stream
the machine dropped.

Also pinned here: a resume that joins an already bound tab to another
stream retires the first stream instead of leaving it mapped to the
tab (two streams for one tab), the bind stand-down counter is kept
only while a bind is connecting, and a sub-agent tab that another
client bound while its old stream was ending is left to its new stream.

Two real daemons (the remote one with a local fake model), one real
client connection; no mocks, patches or test doubles.

Not tested: ``MachineStream._relay`` closing its socket when a frame's
delivery raised (the stream then ends as lost).  No real remote daemon
emits a frame that makes ``_relay_frame`` raise, and a double would be
needed to inject one; every stream end that a real daemon produces
goes through the same ``finally`` and is exercised by every test here.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from kiss.server.remote_tabs import RemoteTabs
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.pause import Pause
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness, _FinishingModel


class TestUnbindEndsSubAgentTabs(RemoteTabsHarness):
    """``newChat`` / a re-bind / a joining resume on a bound root tab."""

    async def _collect_until_unbound(self, tab: str) -> list[dict[str, Any]]:
        """Collect frames up to the ``tabMachine`` "" that ends *tab*'s binding."""
        seen: list[dict[str, Any]] = []
        for _ in range(400):
            msg = await self._next()
            seen.append(msg)
            if msg.get("type") == "tabMachine" and msg.get("tabId") == tab:
                self.assertEqual(msg["machine"], "")
                return seen
        raise AssertionError("the tab was never unbound")

    async def test_new_chat_on_the_bound_tab_ends_its_sub_agent_tabs(self) -> None:
        """The sub-agent tab a surface opened and resumed ends on ``newChat``."""
        tab = "tab-newchat-parent"
        new_tab = await self._fan_out(tab)
        sub_tab = f"{tab}__sub_{new_tab['task_id']}"  # as the webapp names it
        await self._send({"type": "resumeSession", "taskId": new_tab["task_id"], "tabId": sub_tab})
        await self._until("task_events", sub_tab)
        remote_tabs = self.server._server_api.remote_tabs
        self.assertEqual(remote_tabs.machine_of(sub_tab), "box")

        await self._send({"type": "newChat", "tabId": tab})
        seen = await self._collect_until_unbound(tab)
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertEqual(types.count(("subagentDone", sub_tab)), 1, types)
        self.assertEqual(remote_tabs.machine_of(tab), "")
        self.assertEqual(remote_tabs.machine_of(sub_tab), "")
        self.assertEqual(remote_tabs._unbinds, {}, "no bind in flight: nothing counted")

    async def test_rebinding_to_another_machine_ends_the_sub_agent_tabs(self) -> None:
        """A submit naming another machine displaces the stream and ends its sub-agent tabs."""
        added = await self._ask({
            "type": "machineAction", "action": "add", "name": "box2",
            "address": self.remote.url, "machinePassword": "pw-remote",
        }, "machines")
        self.assertEqual(added["error"], "")
        tab = "tab-rebind-parent"
        new_tab = await self._fan_out(tab)
        announced = f"{tab}__sub_{new_tab['task_id']}"

        await self._submit(tab, "Say hello", "box2")
        seen: list[dict[str, Any]] = []
        for _ in range(400):
            msg = await self._next()
            seen.append(msg)
            if msg.get("type") == "tabMachine" and msg.get("tabId") == tab \
                    and msg.get("machine") == "box2":
                break
        else:
            raise AssertionError("the tab was never re-bound")
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertEqual(types.count(("subagentDone", announced)), 1, types)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "box2")
        _FinishingModel.released.set()
        await self._run_until_idle(tab)

    async def test_joining_resume_retires_the_tabs_first_stream(self) -> None:
        """A resume that joins a bound root tab to another stream retires its own stream."""
        remote_tabs = self.server._server_api.remote_tabs
        first = "tab-joined-first"
        await self._submit(first, "Say hello", "box")
        await self._run_until_idle(first)
        stream_a = remote_tabs._tabs[first]
        other = "tab-joined-other"
        new_tab = await self._fan_out(other)
        announced = f"{other}__sub_{new_tab['task_id']}"
        stream_b = remote_tabs._tabs[other]
        self.assertIsNot(stream_a, stream_b)

        await self._send({"type": "resumeSession", "taskId": new_tab["task_id"], "tabId": first})
        # The sub-agent's transcript reaches the joined tab, which stays
        # shown as the machine's (no ``tabMachine`` "": it still is).
        seen: list[dict[str, Any]] = []
        for _ in range(400):
            msg = await self._next()
            seen.append(msg)
            if msg.get("type") == "task_events" and msg.get("tabId") == first:
                break
        else:
            raise AssertionError("the sub-agent's transcript never reached the joined tab")
        self.assertIs(remote_tabs._tabs[first], stream_b, "the tab joined the other stream")
        self.assertTrue(stream_a.retired, "the first stream was retired")
        self.assertEqual(remote_tabs._views._bound.get(first), None)
        self.assertEqual(remote_tabs.machine_of(first), "box")
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertNotIn(("subagentDone", announced), types, "the other stream is untouched")
        self.assertNotIn(("tabMachine", first), types)
        # Then the sub-agent's end.
        _FinishingModel.released.set()
        for _ in range(400):
            msg = await self._next()
            if msg.get("type") == "subagentDone" and msg.get("tab_id") == first:
                break
        else:
            raise AssertionError("the sub-agent's end never reached the joined tab")
        self.assertEqual(remote_tabs.machine_of(other), "box")

    async def test_stand_down_counter_is_kept_only_while_a_bind_connects(self) -> None:
        """Closing tabs that were never bound leaves no bookkeeping behind."""
        remote_tabs = self.server._server_api.remote_tabs
        for i in range(5):
            await self._send({"type": "closeTab", "tabId": f"never-bound-{i}"})
            await self._send({"type": "newChat", "tabId": f"never-bound-{i}"})
        tab = "tab-counted"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        self.assertEqual(remote_tabs._unbinds, {})
        self.assertEqual(remote_tabs._connecting, {})
        await self._send({"type": "closeTab", "tabId": tab})
        await self._until("tabs_state")
        self.assertEqual(remote_tabs.machine_of(tab), "")
        self.assertEqual(remote_tabs._unbinds, {})

    async def test_retiring_the_root_spares_a_sub_agent_tab_that_rebound_meanwhile(self) -> None:
        """A sub-agent tab bound elsewhere while its old stream ends is not ended with it.

        ``_retire`` unmaps the stream's sub-agent tabs under the lock,
        and ``_end_stream`` ends them outside it (the socket close can
        wait on the peer).  A client that binds one of those tabs in
        between (a History-panel click into the machine's chat, from a
        second connection) owns it by then: the stale stream's
        ``subagentDone`` must not reach it, or every surface closes a
        tab that is bound and live.
        """
        remote_tabs = self.server._server_api.remote_tabs
        root = "tab-retiring-root"
        new_tab = await self._fan_out(root)
        child = f"{root}__sub_{new_tab['task_id']}"
        await self._send({"type": "resumeSession", "taskId": new_tab["task_id"], "tabId": child})
        await self._until("task_events", child)
        old = remote_tabs._tabs[root]
        self.assertIs(remote_tabs._tabs[child], old)
        reader2, writer2 = await open_local_connection(self.server)
        self.addCleanup(writer2.close)
        with Pause(RemoteTabs._end_stream.__code__, lambda args: args.get("tab") is old) as hold:
            await self._send({"type": "newChat", "tabId": root})
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 30), "newChat never retired")
            self.assertNotIn(child, remote_tabs._tabs)
            cmd = {"type": "resumeSession", "tabId": child, "taskId": new_tab["task_id"],
                   "machine": "box"}
            writer2.write(json.dumps(cmd).encode("utf-8") + b"\n")
            await writer2.drain()
            for _ in range(400):
                line = await asyncio.wait_for(reader2.readline(), 60)
                assert line, "second connection closed"
                msg = json.loads(line.decode("utf-8"))
                if msg.get("type") == "task_events" and msg.get("tabId") == child:
                    break
            else:
                raise AssertionError("the child never replayed on its new stream")
            rebound = remote_tabs._tabs[child]
            self.assertIsNot(rebound, old)
            self.assertFalse(rebound.retired)
            hold.release()
            seen = await self._collect_until_unbound(root)
        ended = [m["tab_id"] for m in seen if m.get("type") == "subagentDone"]
        self.assertEqual(ended, [], "the rebound child was ended by its old stream")
        self.assertIs(remote_tabs._tabs[child], rebound)
        self.assertEqual(remote_tabs.machine_of(child), "box")
        _FinishingModel.released.set()

