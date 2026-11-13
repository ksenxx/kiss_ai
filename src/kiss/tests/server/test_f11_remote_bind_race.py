# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: two concurrent re-binds of one tab to the same machine (F11).

``RemoteTabs._bind`` snapshotted the tab's stand-down counter
(``_unbinds``) and then detached the tab's old stream OUTSIDE the lock,
through ``_unbind``, which bumps that very counter for every bind in
flight.  Two binds of a tab bound to ``box`` that both name ``box2``
each bumped the counter once after both had taken their snapshot, so
each found it moved and stood down: neither bound the tab and both
commands vanished.  Now a bind detaches the old stream inside its first
critical section and snapshots the counter after that, so the loser of
the race finds the winner's stream and joins it.  Nor does the winner's
taking the tab over stand down a bind the daemon expects but whose
thread has not run (``expect_bind``): only a command's unbind does.

Two real daemons (``box`` and ``box2`` name the same remote daemon) and
the local daemon's real ``RemoteTabs``; which thread takes the lock
first is the scheduler's call, so the race is repeated on fresh tabs.
"""

from __future__ import annotations

import asyncio

from kiss.tests.server.remote_tabs_probe import machine_of
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness

_ROUNDS = 12


class TestConcurrentRebindsToOneMachine(RemoteTabsHarness):
    """Both binds end with the tab bound to the new machine; neither stands down."""

    async def test_both_binds_return_the_new_machines_stream(self) -> None:
        added = await self._ask({
            "type": "machineAction", "action": "add", "name": "box2",
            "address": self.remote.url, "machinePassword": "pw-remote",
        }, "machines")
        self.assertEqual(added["error"], "")
        remote_tabs = self.server._server_api.remote_tabs
        cmd = {"workDir": self.tmpdir}
        for i in range(_ROUNDS):
            tab = f"tab-rebind-race-{i}"
            first = await asyncio.to_thread(remote_tabs._bind, tab, "box", cmd, "first")
            assert first is not None
            self.assertEqual(machine_of(remote_tabs, tab), "box")
            # Each bind is expected before either thread runs, as dispatched commands' are.
            racing = [dict(cmd), dict(cmd)]
            for raced in racing:
                remote_tabs.expect_bind(tab, raced)
            results = await asyncio.gather(
                asyncio.to_thread(remote_tabs._bind, tab, "box2", racing[0], "race a"),
                asyncio.to_thread(remote_tabs._bind, tab, "box2", racing[1], "race b"),
            )
            self.assertTrue(
                all(result is not None for result in results),
                f"round {i}: a bind stood down: {results}",
            )
            self.assertIs(results[0], results[1], "the loser joined the winner's stream")
            self.assertEqual(machine_of(remote_tabs, tab), "box2")
            self.assertTrue(first.retired, "the old stream was retired")
            await asyncio.to_thread(remote_tabs._unbind, tab)
            self.assertEqual(machine_of(remote_tabs, tab), "")
        self.assertEqual(remote_tabs._unbinds, {})
        self.assertEqual(remote_tabs._connecting, {})
        self.assertEqual(remote_tabs._expected, {})
