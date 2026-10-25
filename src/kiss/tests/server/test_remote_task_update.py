# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the Task update subpanel of a tab whose task runs on a machine.

A real local daemon, a real remote daemon ("box") answering a fake
OpenAI endpoint and a real local client (the harness of
:mod:`kiss.tests.server.test_remote_tabs`).  The task-info panel polls
``getTaskUpdate`` for the task running in the visible tab; a tab bound
to a registered machine runs its task there, so the page stamps the
poll ``machine`` and the daemon forwards it to that machine's daemon
(:func:`kiss.server.sidebar_panels.machine_task_update_reply`), which
owns the update.  Checked here:

* a poll stamped with the machine answers ``exists: true`` for the
  task running there (before the fix, only the local lookup ran and the
  subpanel stayed hidden for every remote task); the reply keeps the
  poll's ``tabId`` and ``token`` and is stamped ``machine``;
* the full protocol round-trips: a poll naming the reply's ``sig``
  is answered ``unchanged: true``;
* a poll without the stamp stays local and answers ``exists: false``
  (the local daemon runs no task in that tab), so the client's stamp is
  what shows the subpanel;
* a poll naming an unregistered machine is refused with an
  ``exists: false`` reply carrying ``tabId``, ``token`` and the error;
* a ``refresh: true`` poll (the panel's button) makes the machine's
  daemon run its update agent for the task (``running: true``).

The bound on forwards in flight (a repeated timer poll of one tab is
dropped while its predecessor waits on the machine, a refresh never) is
:mod:`kiss.tests.server.test_audit1018_panel_forward_dedup`.
"""

from __future__ import annotations

import threading

from kiss.tests.server.test_remote_tabs import RemoteTabsHarness, _FinishingModel


class TestRemoteTaskUpdate(RemoteTabsHarness):
    """``getTaskUpdate`` of a tab bound to "box"."""

    async def test_poll_of_a_bound_tab_is_answered_by_its_machine(self) -> None:
        tab = "tab-upd"
        _FinishingModel.released = threading.Event()
        await self._fan_out(tab)  # the root task on box keeps running
        try:
            reply = await self._ask({
                "type": "getTaskUpdate", "tabId": tab, "token": "t1", "machine": "box",
            }, "taskUpdate")
            self.assertEqual(reply["machine"], "box")
            self.assertEqual(reply["tabId"], tab)
            self.assertEqual(reply["token"], "t1")
            self.assertEqual(reply["error"], "")
            self.assertTrue(reply["exists"], reply)
            self.assertNotIn("dueAt", reply)
            self.assertIn("task-update-digest", reply["content"])
            self.assertTrue(reply["taskId"])
            self.assertTrue(reply["sig"])

            again = await self._ask({
                "type": "getTaskUpdate", "tabId": tab, "token": "t2",
                "knownSig": reply["sig"], "machine": "box",
            }, "taskUpdate")
            self.assertEqual(again["machine"], "box")
            self.assertEqual(again["token"], "t2")
            self.assertTrue(again["exists"])
            self.assertTrue(again.get("unchanged"), again)

            local = await self._ask({
                "type": "getTaskUpdate", "tabId": tab, "token": "t3",
            }, "taskUpdate")
            self.assertNotIn("machine", local)
            self.assertEqual(local["token"], "t3")
            self.assertFalse(local["exists"], "this daemon runs no task in the tab")

            unknown = await self._ask({
                "type": "getTaskUpdate", "tabId": tab, "token": "t4", "machine": "nowhere",
            }, "taskUpdate")
            self.assertEqual(unknown["machine"], "nowhere")
            self.assertEqual(unknown["tabId"], tab)
            self.assertEqual(unknown["token"], "t4")
            self.assertFalse(unknown["exists"])
            self.assertEqual(unknown["sig"], "")
            self.assertEqual(unknown["content"], "")
            self.assertIn("nowhere", unknown["error"])

            # The refresh button: the machine's daemon starts its update
            # agent for the task and reports it running.
            forced = await self._ask({
                "type": "getTaskUpdate", "tabId": tab, "token": "t5", "refresh": True,
                "machine": "box",
            }, "taskUpdate")
            self.assertEqual(forced["machine"], "box")
            self.assertEqual(forced["token"], "t5")
            self.assertTrue(forced["exists"])
            self.assertTrue(forced["running"], forced)
        finally:
            _FinishingModel.released.set()
        await self._run_until_idle(tab)
