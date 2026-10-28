# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a cancelled Update handler leaves its spawn in charge.

``_handle_run_update`` raises the run-admission barrier and the
``_update_starting`` single-flight flag, then starts the installer in
the executor.  Cancelling the handler (``stop_async``) cannot stop a
spawn already running or queued there; ``asyncio.shield`` keeps the
executor job alive, but the cancellation still unwinds the handler at
once, and its ``finally`` used to clear both guards while the spawn
was still pending: ``_update_proc`` not yet published, so
``_update_in_progress()`` false, so the barrier went down.  Once the
spawn published the process, an installer was running with task
submits admitted and no exit watcher to ever report its outcome or
lower a barrier again.

The fix hands the guards and the watcher to the spawn job itself: a
done-callback on the executor future releases what a failed spawn
holds and starts the exit watcher of a successful one, whether or not
the handler is still around.

The daemon is real (local transport, fake PyPI endpoint, a stub
``install.sh`` that blocks on a release file); nothing is patched.
The spawn is parked deterministically by making the loop's default
executor a one-worker pool whose worker is held by a gate job queued
ahead of the spawn.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import kiss.server.agent_state as agent_state
from kiss.server.web_server import _cancel_task
from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

TAB = "tab-f9-spawn-owns-guards"
_UNKNOWN_MODEL = "f9-spawn-owns-guards-no-such-model"

# Blocks until the release file appears; gives up once the tmpdir is
# gone because it runs detached.
HELD_INSTALL_SH = """#!/bin/bash
while [ ! -e "{release}" ]; do
    [ -d "{tmpdir}" ] || exit 5
    sleep 0.05
done
exit 0
"""


class TestCancelledUpdateHandlerLeavesSpawnInCharge(_UpdateCheckTestBase):
    """Guards stay up through a cancelled handler; the spawn's watcher lowers them."""

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
        self.gate = threading.Event()
        self._clear_tab()
        # The periodic PyPI check would otherwise queue its fetch on the
        # single worker ahead of the spawn.
        await _cancel_task(self.server._version_check_task)
        self.server._version_check_task = None
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))

    async def asyncTearDown(self) -> None:
        self.gate.set()
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
        return proc is not None and proc.poll() is None

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

    async def test_cancelled_handler_leaves_guards_and_watcher_to_the_spawn(self) -> None:
        loop = asyncio.get_running_loop()
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        await self._send_ready(writer, TAB)

        # The single worker runs jobs in submission order: the first
        # gate holds it while the handler queues its install.sh lookup;
        # the second gate queues behind that lookup, so the spawn the
        # handler submits once the lookup returns queues behind the
        # second gate.
        first_gate = threading.Event()
        first_blocker = loop.run_in_executor(None, first_gate.wait)
        handler = asyncio.create_task(self.server._handle_run_update(""))
        await asyncio.sleep(0)
        self.assertTrue(self.server._update_starting)
        self.assertTrue(self._barrier_up())
        blocker = loop.run_in_executor(None, self.gate.wait)
        first_gate.set()
        await first_blocker
        notice = await self._wait_for_event(reader, "notice")
        self.assertIn("getting installed", str(notice.get("text")))

        # The notice is sent right before the handler suspends on the
        # spawn: cancelling now lands there, with the spawn queued and
        # ``_update_proc`` not yet published.
        self.assertFalse(handler.done())
        self.assertIsNone(self.server._update_proc)
        handler.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await handler
        self.assertIsNone(self.server._update_proc)
        self.assertTrue(
            self.server._update_starting,
            "the cancelled handler cleared the single-flight guard of its pending spawn",
        )
        self.assertTrue(
            self._barrier_up(),
            "the cancelled handler lowered the barrier of its pending spawn",
        )
        # (A ``run`` sent now would queue behind the gate too: commands
        # are dispatched on the same worker.)

        # Release the worker: the spawn publishes the installer and its
        # exit watcher takes over the barrier.
        self.gate.set()
        await blocker
        await self._wait_until(self._installer_running, "the installer to be running")
        await self._wait_until(lambda: not self.server._update_starting, "the spawn to finish")
        self.assertTrue(self._barrier_up(), "the barrier dropped under a running installer")
        watcher = self.server._update_watch_task
        self.assertIsNotNone(watcher, "no exit watcher for the installer the handler spawned")
        assert watcher is not None
        self.assertFalse(watcher.done())
        await self._run_and_expect(reader, writer, "started during the update", admitted=False)

        # The installer exits: the watcher lowers the barrier and runs
        # are admitted again.
        self.release.write_text("")
        await asyncio.wait_for(watcher, 5)
        self.assertFalse(self._barrier_up())
        self.assertFalse(self.server._update_starting)
        await self._run_and_expect(reader, writer, "started after the update", admitted=True)

    async def test_failed_spawn_submission_releases_the_guards(self) -> None:
        """A submission that raises (executor shut down) owns no guards.

        The executor is shut down after the handler queued its
        install.sh lookup: the queued lookup still runs (``wait=False``
        drains the queue), but the spawn submission that follows it
        raises ``RuntimeError`` before any done-callback exists, so the
        handler itself must release the single-flight flag and the
        barrier on its way out.
        """
        loop = asyncio.get_running_loop()
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        await self._send_ready(writer, TAB)
        # ``ready`` fans out through the executor too: let it finish
        # (focusInput is its last reply; the sentinel drains the
        # worker) so only the handler meets the shutdown executor.
        await self._wait_for_event(reader, "focusInput")
        await loop.run_in_executor(None, threading.Event().is_set)

        executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        blocker = loop.run_in_executor(None, self.gate.wait)
        handler = asyncio.create_task(self.server._handle_run_update(""))
        await asyncio.sleep(0)
        self.assertTrue(self.server._update_starting)
        self.assertTrue(self._barrier_up())
        executor.shutdown(wait=False)
        self.gate.set()
        await blocker
        with self.assertRaises(RuntimeError):
            await handler
        # Commands are dispatched on the default executor too: give the
        # daemon a live one again before anything else runs on it.
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        self.assertIsNone(self.server._update_proc)
        self.assertIsNone(self.server._update_watch_task)
        self.assertFalse(
            self.server._update_starting,
            "the failed spawn submission left the single-flight guard up",
        )
        self.assertFalse(
            self._barrier_up(),
            "the failed spawn submission left the admission barrier up",
        )
        await self._run_and_expect(reader, writer, "started after the failed update", admitted=True)
