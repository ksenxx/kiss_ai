# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: "Update when idle" Cancel leaves a poller that already
owns the run-admission barrier alone.

:meth:`RemoteAccessServer._arm_update_barrier_if_idle` (an executor
thread) raises ``AgentServer._update_installing`` under ``STATE_LOCK``
the moment it finds no active task.  The poller then used to clear
``_update_when_idle_armed`` only on its next event-loop step.  A
Cancel that ran between those two instants read the flag as still
armed and cancelled the poller although it was already launching the
installer: the cancel landed inside ``_handle_run_update`` — at the
spawn ``await`` it left a started installer with the barrier lowered
and no exit watcher (findings I2-F1).

The fix consumes the armed flag inside the same critical section that
raises the barrier, so the later disarm sees ``False`` and stands down,
exactly as :meth:`_cancel_update_when_idle` documents.

Driven through the real daemon (local WSS transport, fake PyPI
endpoint).  The test gates the executor thread at the exact handoff:
the real idle check has raised the barrier and consumed the flag, and
its result has not yet resumed the poller.  The Cancel is sent and
fully processed (its ``update_available`` rebroadcast arrives) inside
that window; it must find nothing to cancel.  Then the gate opens and
the installer (a script that records its launch and sleeps) must run
exactly once with the barrier up until its exit watcher lowers it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import kiss.server.agent_state as agent_state
import kiss.server.web_server as ws
from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

TAB = "tab-conc1018-idle-owner"

INSTALL_SH = """#!/bin/bash
echo launched >> "{marker}"
sleep 1
exit 0
"""


class TestUpdateIdleCancelLeavesBarrierOwnerAlone(_UpdateCheckTestBase):
    """A Cancel that loses to the idle verdict does not cancel the launch."""

    async def asyncSetUp(self) -> None:
        self._orig_poll = ws._IDLE_UPDATE_POLL_S
        ws._IDLE_UPDATE_POLL_S = 0.3
        await super().asyncSetUp()
        self.install_root = Path(self.tmpdir) / "kiss_ai"
        self.install_root.mkdir()
        self.server._install_root = self.install_root
        self.server._update_log_path = Path(self.tmpdir) / "update.log"
        self.marker = Path(self.tmpdir) / "launched"
        script = self.install_root / "install.sh"
        script.write_text(INSTALL_SH.format(marker=self.marker))
        script.chmod(0o755)
        self.agent_server = self.server._agent_server
        self.busy = agent_state.AgentState(
            "conc1018-idle-owner-busy",
            tab_id="tab-conc1018-busy",
            is_task_active=True,
        )
        self._writers: list[LocalWriter] = []
        self.verdict_reached = threading.Event()
        self.release_verdict = threading.Event()
        self._real_arm = self.server._arm_update_barrier_if_idle
        self.server._arm_update_barrier_if_idle = self._gated_arm  # type: ignore[method-assign]

    async def asyncTearDown(self) -> None:
        self.release_verdict.set()
        agent_state.unregister(self.busy.task_id, self.busy)
        proc = self.server._update_proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await super().asyncTearDown()
        ws._IDLE_UPDATE_POLL_S = self._orig_poll

    def _gated_arm(self) -> bool:
        """Executor thread: the real idle check, parked once it owns the launch.

        The park sits after the critical section that raised the
        barrier and consumed the armed flag, and before the result
        resumes the poller: the window the Cancel must land in.
        """
        armed = self._real_arm()
        if armed:
            self.verdict_reached.set()
            self.release_verdict.wait(timeout=30)
        return armed

    async def _send(self, writer: LocalWriter, cmd: dict[str, object]) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _wait_pending_idle(self, reader: LocalReader, expected: bool) -> None:
        for _ in range(200):
            ev = await self._wait_for_event(reader, "update_available")
            if ev.get("pendingIdle") is expected:
                return
        raise AssertionError(f"never saw pendingIdle={expected}")

    async def _wait_until(self, predicate: Any, what: str, timeout: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(f"timed out waiting for {what}")
            await asyncio.sleep(0.01)

    def _barrier_up(self) -> bool:
        with agent_state.STATE_LOCK:
            return bool(self.agent_server._update_installing)

    def _installer_alive(self) -> bool:
        proc = self.server._update_proc
        return proc is not None and proc.poll() is None

    async def test_cancel_after_idle_verdict_does_not_cancel_the_launch(self) -> None:
        agent_state.register(self.busy)
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        await self._send_ready(writer, TAB)
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)
        self.assertFalse(self.verdict_reached.is_set())

        # Retire the busy task: the next idle check raises the barrier,
        # consumes the flag and parks before its verdict reaches the
        # poller.
        self.busy.is_task_active = False
        await self._wait_until(
            self.verdict_reached.is_set,
            "the idle check to take ownership of the launch",
        )
        self.assertTrue(self._barrier_up())
        self.assertFalse(self.server._update_when_idle_armed)
        self.assertFalse(self.marker.exists())
        poller = self.server._update_when_idle_task
        self.assertIsNotNone(poller)

        # The Cancel lands inside the handoff window and is processed to
        # completion (its rebroadcast arrives) while the gate is closed.
        await self._send(writer, {"type": "updateWhenIdle", "cancel": True})
        await self._wait_pending_idle(reader, False)
        self.assertFalse(self.release_verdict.is_set())
        self.assertIs(self.server._update_when_idle_task, poller, "the Cancel cancelled the poller")
        assert poller is not None
        self.assertFalse(poller.cancelled())
        self.assertTrue(self._barrier_up(), "the Cancel lowered the barrier")
        self.assertFalse(self.marker.exists())

        # Let the verdict reach the poller: it launches the installer.
        self.release_verdict.set()
        await self._wait_until(self.marker.exists, "the installer to be launched")
        await self._wait_until(
            lambda: self.server._update_watch_task is not None,
            "the installer's exit watcher",
        )
        self.assertTrue(self._installer_alive())
        self.assertTrue(self._barrier_up(), "the live installer owns the barrier")
        self.assertFalse(self.server._update_when_idle_armed)

        # The installer exits (sleep 1) and its watcher lowers the barrier.
        await self._wait_until(
            lambda: not self._installer_alive(),
            "the installer to exit",
            timeout=10,
        )
        await self._wait_until(
            lambda: not self._barrier_up(),
            "the exit watcher to lower the barrier",
        )
        self.assertEqual(self.marker.read_text().count("launched"), 1)
        self.assertIsNone(self.server._update_when_idle_task)
