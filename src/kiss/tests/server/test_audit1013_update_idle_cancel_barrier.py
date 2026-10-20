# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: cancelling "Update when idle" never leaves the run barrier up.

The idle poller checks for idle in an executor thread
(:meth:`RemoteAccessServer._arm_update_barrier_if_idle`, which raises
``AgentServer._update_installing`` under ``STATE_LOCK`` the moment it
sees no active task).  When the toast's Cancel arrived while that thread
had started but was still waiting for ``STATE_LOCK``, the cancelled
poller lowered the barrier first and the thread then raised it for
nobody: no installer owned it, no exit watcher would lower it, and every
later ``run`` was refused with "An update is being installed" until the
daemon restarted.

The thread's arming is conditional on the armed generation
(``_update_when_idle_armed``) inside the same ``STATE_LOCK`` section, and
Cancel clears it under the lock.  A generation rather than a flag,
because a Cancel followed by a fresh "Update when idle" during that
same wait arms a NEW poller: the parked check of the cancelled one must
not consume that arming (its verdict goes to a cancelled task, so the
barrier would stay up and the new poller would wait forever), which is
the second test here.

Both tests drive the real daemon (local WSS transport, fake PyPI
endpoint) and hold ``STATE_LOCK`` — an ``RLock`` — on the event-loop
thread, so the poller's idle check parks on it in its executor thread
while the Cancel (and re-arm) handlers, on the loop, re-enter the lock at
once.  The admitted ``run`` that proves the barrier is down uses a model
name that is not configured, so the runner exits at its own "No model
available" check and nothing is patched.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import FrameType
from typing import Any

import kiss.server.agent_state as agent_state
import kiss.server.web_server as ws
from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

TAB = "tab-a1013-idle-cancel"
_UNKNOWN_MODEL = "audit1013-idle-cancel-no-such-model"

# Records every launch so a wrong (or a missing) one is visible.
INSTALL_SH = """#!/bin/bash
echo launched >> "{marker}"
exit 0
"""


def _poller_thread_inside_arm() -> bool:
    """Return whether some thread is executing ``_arm_update_barrier_if_idle``."""
    for top in sys._current_frames().values():
        frame: FrameType | None = top
        while frame is not None:
            if frame.f_code.co_name == "_arm_update_barrier_if_idle":
                return True
            frame = frame.f_back
    return False


class TestUpdateIdleCancelBarrier(_UpdateCheckTestBase):
    """Cancel racing the poller's idle check leaves the daemon accepting runs."""

    async def asyncSetUp(self) -> None:
        self._orig_poll = ws._IDLE_UPDATE_POLL_S
        ws._IDLE_UPDATE_POLL_S = 0.5
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
            "a1013-idle-cancel-busy", tab_id="tab-a1013-busy", is_task_active=True,
        )
        self._writers: list[LocalWriter] = []
        self._lock_held = False

    async def asyncTearDown(self) -> None:
        if self._lock_held:
            agent_state.STATE_LOCK.release()
        agent_state.unregister(self.busy.task_id, self.busy)
        proc = self.server._update_proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        await self._wait_tab_idle()
        with agent_state.STATE_LOCK:
            stale = [st.task_id for st in agent_state.snapshot() if st.tab_id == TAB]
        for task_id in stale:
            agent_state.unregister(task_id)
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await super().asyncTearDown()
        ws._IDLE_UPDATE_POLL_S = self._orig_poll

    async def _client(self) -> tuple[LocalReader, LocalWriter]:
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        await self._send_ready(writer, TAB)
        return reader, writer

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

    async def _wait_tab_idle(self) -> None:
        def idle() -> bool:
            with agent_state.STATE_LOCK:
                st = agent_state.find_by_tab(TAB)
                return st is None or (st.task_thread is None and not st.is_task_active)
        await self._wait_until(idle, "the tab to go idle")

    def _barrier_up(self) -> bool:
        with agent_state.STATE_LOCK:
            return bool(self.agent_server._update_installing)

    async def _arm(self) -> tuple[LocalReader, LocalWriter]:
        agent_state.register(self.busy)
        reader, writer = await self._client()
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)
        # The poller's first idle check runs at once (and finds the
        # busy task); give it time to finish before the lock is taken
        # so the next one parks on it.
        await asyncio.sleep(0.1)
        self.assertFalse(_poller_thread_inside_arm())
        return reader, writer

    async def _park_idle_check(self) -> tuple[LocalReader, LocalWriter]:
        """Arm, then hold ``STATE_LOCK`` on the loop with the poller's check parked on it.

        The task the check would find is gone, so without the
        generation check the parked check WOULD arm once it gets the
        lock.
        """
        reader, writer = await self._arm()
        agent_state.STATE_LOCK.acquire()
        self._lock_held = True
        self.busy.is_task_active = False
        await self._wait_until(
            _poller_thread_inside_arm, "the poller's idle check to start",
        )
        self.assertTrue(self.server._update_when_idle_armed)
        self.assertFalse(self._barrier_up())
        return reader, writer

    def _release_lock(self) -> None:
        agent_state.STATE_LOCK.release()
        self._lock_held = False

    async def _run_is_admitted(self, reader: LocalReader, writer: LocalWriter) -> None:
        """The real consumer: a fresh run reaches the runner's own model check."""
        await self._send(writer, {
            "type": "run",
            "tabId": TAB,
            "prompt": "after the cancelled idle update",
            "workDir": self.tmpdir,
            "model": _UNKNOWN_MODEL,
            "useWorktree": False,
            "autoCommit": False,
            "classifyTasks": False,
        })
        result = await self._wait_for_event(reader, "result")
        self.assertEqual(result.get("tabId"), TAB)
        self.assertIn("No model available", str(result.get("text")))

    async def test_cancel_while_idle_check_waits_for_the_lock(self) -> None:
        reader, writer = await self._park_idle_check()

        # Cancel runs on the loop, re-enters the held lock and disarms
        # while the check is still parked.
        await self._send(writer, {"type": "updateWhenIdle", "cancel": True})
        await self._wait_pending_idle(reader, False)
        self.assertFalse(self.server._update_when_idle_armed)
        self.assertIsNone(self.server._update_when_idle_task)
        self.assertTrue(_poller_thread_inside_arm())

        # Release: the check gets the lock and must not raise the
        # barrier for a cancelled update.
        self._release_lock()
        await self._wait_until(
            lambda: not _poller_thread_inside_arm(), "the idle check to finish",
        )
        self.assertFalse(self._barrier_up())
        self.assertFalse(self.marker.exists())
        await self._run_is_admitted(reader, writer)

    async def test_rearm_during_parked_check_belongs_to_the_new_poller(self) -> None:
        reader, writer = await self._park_idle_check()
        first_gen = self.server._update_when_idle_armed

        # Cancel, then arm again, both while the first poller's check
        # is parked: the arming now belongs to a second poller.
        await self._send(writer, {"type": "updateWhenIdle", "cancel": True})
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)
        self.assertNotEqual(self.server._update_when_idle_armed, first_gen)
        second_task = self.server._update_when_idle_task
        self.assertIsNotNone(second_task)

        # Release: the stale check must stand down and the second
        # poller's own check must find idle and launch the installer,
        # exactly once; its exit watcher lowers the barrier again.
        self._release_lock()
        await self._wait_until(
            lambda: self.marker.exists() and not self._barrier_up(),
            "the second poller to launch the installer and the barrier to drop",
            timeout=15.0,
        )
        self.assertEqual(self.marker.read_text().count("launched"), 1)
        await self._wait_until(
            lambda: self.server._update_when_idle_task is None, "the second poller to finish",
        )
        await self._run_is_admitted(reader, writer)
