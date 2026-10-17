# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a finished installer's watcher leaves the barrier to the next installer.

``_watch_update_exit`` polls the detached installer every 0.2 s and, once
it is gone, lowered the run-admission barrier (``AgentServer._update_installing``)
unconditionally.  An Update click or the idle poller landing in the gap
between the installer's exit and the poll that notices it finds
``_update_in_progress()`` false, raises the barrier for ITS installer and
spawns it; the stale watcher then wakes, lowers the barrier, and the
second installer runs with task submits admitted: the daemon restart it
ends with kills them mid-run, which is what the barrier exists to prevent.

The fix: the watcher lowers the barrier only while no installer is
being spawned or running (``_update_in_progress()``), as the idle
poller's cancellation path already did.

The daemon is real (local transport, fake PyPI endpoint, a stub
``install.sh`` that blocks on a release file); the admitted run uses a
model name that is not configured, so ``_run_task`` exits by itself
after registering the run.  Nothing is patched.  The second update is
started by calling the daemon's own ``_handle_run_update`` from the test
coroutine, which shares the event loop: that is the only way to land in
the 0.2 s gap deterministically (a message over the socket would add a
loop hop the stale watcher can win), and it is exactly the call the
``runUpdate`` command and the idle poller make.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import kiss.server.agent_state as agent_state
from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

TAB = "tab-a1017-barrier"
_UNKNOWN_MODEL = "audit1017-barrier-no-such-model"

# Blocks until the release file appears; gives up once the tmpdir is
# gone because it runs detached.
HELD_INSTALL_SH = """#!/bin/bash
while [ ! -e "{release}" ]; do
    [ -d "{tmpdir}" ] || exit 5
    sleep 0.05
done
exit 0
"""


class TestStaleWatcherLeavesBarrierToNextInstaller(_UpdateCheckTestBase):
    """The barrier stays up for installer B although installer A's watcher wakes under it."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.install_root = Path(self.tmpdir) / "kiss_ai"
        self.install_root.mkdir()
        self.server._install_root = self.install_root
        self.server._update_log_path = Path(self.tmpdir) / "update.log"
        self.release = Path(self.tmpdir) / "release"
        script = self.install_root / "install.sh"
        script.write_text(HELD_INSTALL_SH.format(release=self.release, tmpdir=self.tmpdir))
        script.chmod(0o755)
        self._writers: list[LocalWriter] = []
        self._clear_tab()

    async def asyncTearDown(self) -> None:
        self.release.write_text("")
        proc = self.server._update_proc
        if proc is not None and proc.poll() is None:
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait, 5), 6)
            except (TimeoutError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait()
        await self._wait_until(self._tab_idle, "the tab to go idle")
        self._clear_tab()
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await super().asyncTearDown()

    # -- helpers --------------------------------------------------------

    def _clear_tab(self) -> None:
        with agent_state.STATE_LOCK:
            stale = [st.task_id for st in agent_state.snapshot() if st.tab_id == TAB]
        for task_id in stale:
            agent_state.unregister(task_id)

    def _tab_idle(self) -> bool:
        with agent_state.STATE_LOCK:
            st = agent_state.find_by_tab(TAB)
            return st is None or (st.task_thread is None and not st.is_task_active)

    def _barrier_up(self) -> bool:
        with agent_state.STATE_LOCK:
            return bool(self.server._agent_server._update_installing)

    def _installer_running(self) -> bool:
        proc = self.server._update_proc
        return proc is not None and proc.poll() is None and not self.server._update_starting

    async def _send(self, writer: LocalWriter, cmd: dict[str, object]) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _wait_until(self, predicate: Any, what: str, timeout: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(f"timed out waiting for {what}")
            await asyncio.sleep(0.02)

    async def _run_and_expect(
        self, reader: LocalReader, writer: LocalWriter, prompt: str, admitted: bool,
    ) -> None:
        """Submit a ``run`` on TAB and assert the barrier refused or admitted it."""
        await self._send(writer, {
            "type": "run", "tabId": TAB, "prompt": prompt, "workDir": self.tmpdir,
            "model": _UNKNOWN_MODEL, "useWorktree": False, "autoCommit": False,
            "classifyTasks": False,
        })
        if admitted:
            # The admitted run reaches the runner's own model check.
            result = await self._wait_for_event(reader, "result")
            self.assertEqual(result.get("tabId"), TAB)
            self.assertIn("No model available", str(result.get("text")))
            await self._wait_until(self._tab_idle, "the tab to go idle")
            return
        status = await self._wait_for_event(reader, "status")
        self.assertEqual(status.get("tabId"), TAB)
        self.assertFalse(status.get("running"))
        err = await self._wait_for_event(reader, "error")
        self.assertIn("update is being installed", str(err.get("text")))
        with agent_state.STATE_LOCK:
            self.assertIsNone(agent_state.find_by_tab(TAB))

    # -- test -----------------------------------------------------------

    async def test_stale_watcher_keeps_the_barrier_of_the_next_installer(self) -> None:
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        await self._send_ready(writer, TAB)
        await self._wait_for_event(reader, "update_available")

        # Land in the gap: installer A has exited but its watcher, asleep
        # between two polls, has not seen it yet.  The watcher wins the
        # gap only when its 0.2 s timer fires within the few ms the exit
        # takes to reach us, so a handful of attempts always finds one.
        for _attempt in range(20):
            await self._send(writer, {"type": "runUpdate"})
            await self._wait_until(self._installer_running, "installer A to be running")
            self.assertTrue(self._barrier_up())
            proc_a = self.server._update_proc
            watcher_a = self.server._update_watch_task
            assert proc_a is not None and watcher_a is not None
            self.release.write_text("")
            await asyncio.to_thread(proc_a.wait, 5)
            self.release.unlink()  # the next installer blocks again
            if not watcher_a.done():
                break
            await self._wait_until(lambda: not self._barrier_up(), "the barrier to drop")
        else:
            self.fail("installer A's watcher noticed every exit before the test did")

        # No await between the ``not done()`` check and this call: the
        # handler raises the barrier for installer B before the stale
        # watcher can run.  The watcher then wakes during or after B's
        # spawn and must leave the barrier alone.
        await self.server._handle_run_update("")
        await self._wait_until(self._installer_running, "installer B to be running")
        await asyncio.wait_for(watcher_a, 5)
        self.assertTrue(self._barrier_up())
        await self._run_and_expect(reader, writer, "started during update B", admitted=False)

        # B exits: its own watcher lowers the barrier and runs are admitted.
        self.release.write_text("")
        await self._wait_until(lambda: not self._barrier_up(), "the barrier to drop")
        installer_b = self.server._update_proc
        assert installer_b is not None
        self.assertIsNotNone(installer_b.poll())
        await self._run_and_expect(reader, writer, "started after update B", admitted=True)
