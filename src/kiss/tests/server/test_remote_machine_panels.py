# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the Schedule, Machines, Apps, Spend and Recommendations subpanels of a machine.

The fixture of :mod:`kiss.tests.server.test_remote_machine_settings`: a
real local daemon, a real remote daemon registered as ``box``, a client
on the local connection.  The subpanel commands stamped ``machine``
(:data:`kiss.server.sidebar_panels.MACHINE_PANEL_REPLIES`) are answered
by the machine's daemon, the reply stamped ``machine`` and ``forwarded``:

* a ``cronAction`` creates, lists and removes a job in the MACHINE's
  cron store, which this daemon's own Schedule never shows, and a
  refused one comes back as the ``cronJobs`` reply carrying ``error``
  and the ``action``;
* ``getMachines`` / ``machineAction`` list, add and remove the
  MACHINE's registered machines, which this daemon's own registry
  never shows, and a refused action comes back carrying ``error`` and
  the ``action``;
* ``getSpendReport``, ``getRecommendations`` and ``getAppsStatus`` are
  the machine's figures, apps and tips;
* an unknown machine answers each with an empty reply of the right
  type carrying the ``error``.
"""

from __future__ import annotations

import asyncio

from kiss.agents.sorcar import machines as mach
from kiss.tests.server.test_remote_machine_settings import RemoteMachineCase


class TestRemoteMachinePanels(RemoteMachineCase):
    """The subpanel commands stamped ``machine`` are the machine's."""

    async def test_cron_jobs_are_the_machines(self) -> None:
        created = await self._ask_box({
            "type": "cronAction", "action": "create", "name": "box nightly",
            "schedule": "every 1d", "what": "echo hi", "kind": "command",
        }, "cronJobs")
        self.assertEqual(created.get("error", ""), "")
        self.assertEqual(created["action"], "create")
        self.assertEqual(created["machine"], "box")
        self.assertIs(created["forwarded"], True)
        (job,) = [j for j in created["jobs"] if j["name"] == "box nightly"]
        self.assertEqual(job["what"], "echo hi")

        # The machine's own default jobs (the recommendations one) are listed with it.
        listed = await self._ask_box({"type": "getCronJobs"}, "cronJobs")
        self.assertIn("box nightly", [j["name"] for j in listed["jobs"]])
        mine = await self._ask({"type": "getCronJobs"}, "cronJobs")
        self.assertNotIn("forwarded", mine)
        self.assertNotIn("box nightly", [j["name"] for j in mine["jobs"]])

        removed = await self._ask_box(
            {"type": "cronAction", "action": "remove", "jobId": job["id"]}, "cronJobs",
        )
        self.assertEqual(removed.get("error", ""), "")
        self.assertEqual(removed["action"], "remove")
        self.assertNotIn("box nightly", [j["name"] for j in removed["jobs"]])

    async def test_a_refused_cron_action_is_the_machines_error(self) -> None:
        refused = await self._ask_box({
            "type": "cronAction", "action": "create", "name": "broken",
            "schedule": "whenever you feel like it", "what": "echo hi", "kind": "command",
        }, "cronJobs")
        self.assertTrue(refused["error"], refused)
        self.assertEqual(refused["action"], "create")
        self.assertNotIn("broken", [j["name"] for j in refused["jobs"]])
        missing = await self._ask_box(
            {"type": "cronAction", "action": "remove", "jobId": "no-such-job"}, "cronJobs",
        )
        self.assertIn("no-such-job", missing["error"])
        self.assertEqual(missing["action"], "remove")

    async def test_registered_machines_are_the_machines(self) -> None:
        listed = await self._ask_box({"type": "getMachines"}, "machines")
        self.assertEqual(listed.get("error", ""), "")
        self.assertEqual(listed["machine"], "box")
        self.assertIs(listed["forwarded"], True)
        # Only the machine's own row: box has registered no machine.
        (own,) = listed["machines"]
        self.assertIs(own["local"], True)

        # box registers a machine of its own (its own daemon, reachable).
        added = await self._ask_box({
            "type": "machineAction", "action": "add", "name": "loop",
            "address": self.remote.url, "machinePassword": "pw-remote",
        }, "machines")
        self.assertEqual(added.get("error", ""), "")
        self.assertEqual(added["action"], "add")
        self.assertIn("loop", [m["name"] for m in added["machines"]])
        mine = await self._ask({"type": "getMachines"}, "machines")
        self.assertNotIn("forwarded", mine)
        self.assertEqual([m["name"] for m in mine["machines"][1:]], ["box"])

        removed = await self._ask_box(
            {"type": "machineAction", "action": "remove", "name": "loop"}, "machines",
        )
        self.assertEqual(removed.get("error", ""), "")
        self.assertEqual(removed["action"], "remove")
        self.assertNotIn("loop", [m["name"] for m in removed["machines"]])

    async def test_a_refused_machine_action_is_the_machines_error(self) -> None:
        missing = await self._ask_box(
            {"type": "machineAction", "action": "remove", "name": "loop"}, "machines",
        )
        self.assertEqual(missing["error"], "no machine named 'loop'")
        self.assertEqual(missing["action"], "remove")
        bad = await self._ask_box({
            "type": "machineAction", "action": "add", "name": "bad name",
            "address": "https://127.0.0.1:1",
        }, "machines")
        self.assertIn("not a valid machine name", bad["error"])
        self.assertEqual(bad["action"], "add")
        # box's own row is still listed with the refusal.
        self.assertEqual([m["local"] for m in bad["machines"]], [True])

    async def test_spend_recommendations_and_apps_are_the_machines(self) -> None:
        spend = await self._ask_box({"type": "getSpendReport"}, "spendReport")
        self.assertEqual(spend.get("error", ""), "")
        self.assertEqual(spend["machine"], "box")
        self.assertIs(spend["forwarded"], True)
        self.assertEqual(spend["total"]["tasks"], 0)
        self.assertEqual(spend["days"], [])

        tips = await self._ask_box({"type": "getRecommendations"}, "recommendations")
        self.assertEqual(tips.get("error", ""), "")
        self.assertIs(tips["forwarded"], True)
        self.assertEqual(tips["tasks"], [])
        self.assertEqual(tips["generatedAt"], 0)

        apps = await self._ask_box({"type": "getAppsStatus"}, "appsStatus")
        self.assertEqual(apps.get("error", ""), "")
        self.assertIs(apps["forwarded"], True)
        self.assertIsInstance(apps["apps"], list)
        self.assertIn("checkedAt", apps)
        # The local reply is this daemon's own: not forwarded.
        mine = await self._ask({"type": "getSpendReport"}, "spendReport")
        self.assertNotIn("forwarded", mine)

    async def test_subpanel_forwards_travel_on_a_link_of_their_own(self) -> None:
        """A slow subpanel exchange (a 45 s app probe) holds no settings request up, nor vice versa.

        The link answers one request at a time: the subpanels' polls
        use the ``panels`` lane, the settings sheet the default one.
        Holding one lane's link (as a request in flight does) leaves
        the other lane answering.
        """
        await self._ask_box({"type": "getCronJobs"}, "cronJobs")
        await self._ask_box({"type": "getConfig"}, "configData")
        (box,) = [m for m in mach.load_machines() if m.name == "box"]
        panels, default = mach.link_for(box, "panels"), mach.link_for(box)
        self.assertIsNot(panels, default)
        with panels._lock:
            settings = await asyncio.wait_for(
                self._ask_box({"type": "getConfig"}, "configData"), 5.0,
            )
        self.assertEqual(settings.get("error", ""), "")
        with default._lock:
            jobs = await asyncio.wait_for(
                self._ask_box({"type": "getCronJobs"}, "cronJobs"), 5.0,
            )
        self.assertEqual(jobs.get("error", ""), "")
        # The registry change of a saved remote password retires both lanes.
        await self._ask_box(
            {"type": "saveConfig", "config": {"remote_password": "pw-new"}}, "configData",
        )
        (box,) = [m for m in mach.load_machines() if m.name == "box"]
        self.assertIsNot(mach.link_for(box, "panels"), panels)
        again = await self._ask_box({"type": "getCronJobs"}, "cronJobs")
        self.assertEqual(again.get("error", ""), "")

    async def test_unknown_machine_answers_each_with_an_empty_reply_and_the_error(self) -> None:
        for cmd, reply_type, empty in (
            ({"type": "getCronJobs"}, "cronJobs", {"jobs": []}),
            ({"type": "cronAction", "action": "pause", "jobId": "x"}, "cronJobs",
             {"jobs": [], "action": "pause"}),
            ({"type": "getMachines"}, "machines", {"machines": [], "checkedAt": 0}),
            ({"type": "machineAction", "action": "remove", "name": "x"}, "machines",
             {"machines": [], "checkedAt": 0, "action": "remove"}),
            ({"type": "getAppsStatus"}, "appsStatus", {"apps": [], "checkedAt": 0}),
            ({"type": "getSpendReport"}, "spendReport", {}),
            ({"type": "getRecommendations"}, "recommendations",
             {"generatedAt": 0, "tasks": []}),
        ):
            await self._send({**cmd, "machine": "nowhere"})
            reply = await self._next(reply_type, forwarded=True)
            self.assertEqual(reply["machine"], "nowhere")
            self.assertIn("no machine named 'nowhere'", reply["error"])
            for key, value in empty.items():
                self.assertEqual(reply[key], value, (reply_type, key))
