# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a registry change made by another process retires the daemon's links.

``machines.save_machines`` retires the cached links (and ssh tunnels)
of every machine whose entry changed or vanished, but only in the
process that writes the registry.  A ``/machine remove`` run by the
``sorcar`` CLI, in its own process, left the daemon's authenticated
link to the removed machine (and an SSH machine's ``ssh -N -L``
forward) up until the daemon exited.  The daemon's registry watcher now
runs the same sweep when it sees ``machines.json`` change.

A real local daemon, a real remote daemon registered as ``box``, and a
real second Python process that rewrites the registry.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import MACHINES_REGISTRY_POLL_SECONDS
from kiss.tests.server.test_remote_machine_settings import RemoteMachineCase

_REWRITE = (
    "from kiss.agents.sorcar import machines as m; "
    "m.save_machines([x for x in m.load_machines() if x.name != {drop!r}])"
)


def _box_links() -> list[tuple[str, str, str, str, str]]:
    """The keys of this process's cached links to ``box``."""
    with mach._links_lock:
        return [key for key in mach._links if key[0] == "box"]


class TestRegistryWatchRetiresLinks(RemoteMachineCase):
    """The watcher sweeps the link cache like the registry's writer does."""

    async def _rewrite_in_another_process(self, drop: str) -> None:
        """Save the registry minus machine *drop* from a separate Python process."""
        await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", _REWRITE.format(drop=drop)],
            check=True, env=dict(os.environ), timeout=120,
        )

    async def _box_link_cached(self) -> None:
        reply = await self._ask_box({"type": "getConfig"}, "configData")
        self.assertEqual(reply.get("error", ""), "")
        self.assertTrue(_box_links(), "a forwarded request caches the link to box")

    async def test_a_machine_removed_elsewhere_loses_its_link(self) -> None:
        await self._box_link_cached()
        await self._rewrite_in_another_process(drop="box")
        deadline = time.monotonic() + MACHINES_REGISTRY_POLL_SECONDS + 20
        while _box_links() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        self.assertEqual(_box_links(), [])

    async def test_an_unchanged_machine_keeps_its_link(self) -> None:
        await self._box_link_cached()
        kept = _box_links()
        await self._rewrite_in_another_process(drop="nothing-to-drop")
        # The watcher's push follows its sweep.
        pushed = await asyncio.wait_for(
            self._next("machines"), MACHINES_REGISTRY_POLL_SECONDS + 20,
        )
        self.assertIn("box", [row["name"] for row in pushed["machines"]])
        self.assertEqual(_box_links(), kept)
