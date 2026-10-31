# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a Cancel and a re-arm of "Update when idle" launch exactly once.

The idle poller's check (:meth:`RemoteAccessServer._arm_update_barrier_if_idle`)
raises ``AgentServer._update_installing`` under ``STATE_LOCK`` the moment
it sees no active task, and consumes the armed generation
(``_update_when_idle_armed``) in the same section.  A Cancel followed by
a fresh "Update when idle" arms a NEW poller: the cancelled one must not
consume that arming (its verdict would go to a cancelled task, so the
barrier would stay up and the new poller would wait forever).

The check used to run in an executor thread, where it could park on
``STATE_LOCK`` while the Cancel ran; this file then also covered a
Cancel landing during that wait.  The check now runs on the event loop,
in the step that hands over to :meth:`_handle_run_update`, so a Cancel
lands either while the poller sleeps (it is cancelled there and never
checks again; ``test_update_when_idle.py`` covers that) or after the
verdict (``test_conc1018_server_update_idle_cancel_owner.py``).

Driven through the real daemon (local WSS transport, fake PyPI
endpoint).  The admitted ``run`` that proves the barrier is down uses a
model name that is not configured, so the runner exits at its own "No
model available" check and nothing is patched.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
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


class TestUpdateIdleCancelBarrier(_UpdateCheckTestBase):
    """A Cancel and a re-arm leave one launch and a daemon accepting runs."""

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

    async def asyncTearDown(self) -> None:
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
        return reader, writer

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

    async def test_rearm_after_cancel_belongs_to_the_new_poller(self) -> None:
        reader, writer = await self._arm()
        first_gen = self.server._update_when_idle_armed
        self.assertTrue(first_gen)

        # Cancel, then arm again, both while the first poller waits
        # for the busy task: the arming now belongs to a second poller.
        await self._send(writer, {"type": "updateWhenIdle", "cancel": True})
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)
        self.assertNotEqual(self.server._update_when_idle_armed, first_gen)
        second_task = self.server._update_when_idle_task
        self.assertIsNotNone(second_task)

        # Retire the task: the second poller's check must find idle and
        # launch the installer, exactly once; its exit watcher lowers
        # the barrier again.
        self.busy.is_task_active = False
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
