# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the update toast's "Update when idle" action.

Pressing "Update when idle" sends ``updateWhenIdle`` to the daemon, which
arms a poller that runs ``install.sh`` the first time no task is in
flight, and rebroadcasts ``update_available`` with ``pendingIdle`` so
every chat window's toast shows the armed state.  ``cancel: true``
disarms it; a direct ``runUpdate`` supersedes it.

The daemon is real (UDS transport, fake PyPI endpoint, a stub
``install.sh`` that records each launch in a marker file); a running
task is represented by a live ``AgentState`` in the agent registry,
exactly as the task runner registers one.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import kiss.server.agent_state as agent_state
import kiss.server.web_server as ws
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

INSTALL_SH = """#!/bin/bash
echo launched >> "{marker}"
exit 0
"""

# Blocks until the release file appears, so an update can be held "in
# progress" for as long as a test needs; gives up once the tmpdir is
# gone because it runs detached and outlives pytest.
HELD_INSTALL_SH = """#!/bin/bash
while [ ! -e "{release}" ]; do
    [ -d "{tmpdir}" ] || exit 5
    sleep 0.05
done
exit 0
"""


class TestUpdateWhenIdle(_UpdateCheckTestBase):
    """Arm, wait for idle, install; cancel; supersede; idempotence."""

    async def asyncSetUp(self) -> None:
        self._orig_poll = ws._IDLE_UPDATE_POLL_S
        ws._IDLE_UPDATE_POLL_S = 0.05
        await super().asyncSetUp()
        self.install_root = Path(self.tmpdir) / "kiss_ai"
        self.install_root.mkdir()
        self.server._install_root = self.install_root
        self.server._update_log_path = Path(self.tmpdir) / "update.log"
        self.marker = Path(self.tmpdir) / "launched"
        self.release = Path(self.tmpdir) / "release"
        self._install_stub(INSTALL_SH.format(marker=self.marker))
        self.busy = agent_state.AgentState(
            "idle-update-task", tab_id="idle-update-tab", is_task_active=True,
        )
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        agent_state.unregister(self.busy.task_id, self.busy)
        self.release.write_text("")
        proc = self.server._update_proc
        if proc is not None and proc.poll() is None:
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait, 5), 6)
            except (TimeoutError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait()
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await super().asyncTearDown()
        ws._IDLE_UPDATE_POLL_S = self._orig_poll

    def _install_stub(self, body: str) -> None:
        script = self.install_root / "install.sh"
        script.write_text(body)
        script.chmod(0o755)

    def _launches(self) -> int:
        if not self.marker.exists():
            return 0
        return len(self.marker.read_text().splitlines())

    async def _client(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await self._connect_uds()
        self._writers.append(writer)
        await self._send_ready(writer, "tab-idle-update")
        return reader, writer

    async def _send(self, writer: asyncio.StreamWriter, cmd: dict[str, object]) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _wait_pending_idle(
        self, reader: asyncio.StreamReader, expected: bool,
    ) -> dict[str, object]:
        """Read events until an ``update_available`` with the wanted flag."""
        for _ in range(200):
            ev = await self._wait_for_event(reader, "update_available")
            if ev.get("pendingIdle") is expected:
                return ev
        raise AssertionError(f"never saw pendingIdle={expected}")

    async def _wait_launches(self, count: int, timeout: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while self._launches() < count:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(
                    f"installer launched {self._launches()} times, wanted {count}",
                )
            await asyncio.sleep(0.02)

    async def _settle(self) -> None:
        """Sleep through several idle polls so a wrong launch would show."""
        await asyncio.sleep(ws._IDLE_UPDATE_POLL_S * 6)

    @requires_unix_sockets
    async def test_installs_once_the_running_task_finishes(self) -> None:
        agent_state.register(self.busy)
        reader, writer = await self._client()
        ev = await self._wait_pending_idle(reader, False)
        self.assertTrue(ev["available"])

        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)
        await self._settle()
        self.assertEqual(self._launches(), 0)
        self.assertTrue(self.server._update_when_idle_armed)

        # The task finishes: the poller disarms, installs (the "getting
        # installed" notice goes to every window), then rebroadcasts.
        self.busy.is_task_active = False
        notice = await self._wait_for_event(reader, "notice")
        self.assertIn("getting installed", str(notice["text"]))
        self.assertNotIn("connId", notice)
        await self._wait_pending_idle(reader, False)
        await self._wait_launches(1)
        self.assertFalse(self.server._update_when_idle_armed)
        await self._settle()
        self.assertEqual(self._launches(), 1)
        # The finished poller has dropped its own task reference.
        self.assertIsNone(self.server._update_when_idle_task)

    @requires_unix_sockets
    async def test_installs_immediately_when_already_idle(self) -> None:
        reader, writer = await self._client()
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_launches(1)
        self.assertFalse(self.server._update_when_idle_armed)

    @requires_unix_sockets
    async def test_cancel_disarms_and_nothing_is_installed(self) -> None:
        agent_state.register(self.busy)
        reader, writer = await self._client()
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)

        await self._send(writer, {"type": "updateWhenIdle", "cancel": True})
        await self._wait_pending_idle(reader, False)
        self.assertIsNone(self.server._update_when_idle_task)

        self.busy.is_task_active = False
        await self._settle()
        self.assertEqual(self._launches(), 0)

    @requires_unix_sockets
    async def test_cancel_with_nothing_pending_is_harmless(self) -> None:
        reader, writer = await self._client()
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle", "cancel": True})
        await self._wait_pending_idle(reader, False)
        await self._settle()
        self.assertEqual(self._launches(), 0)

    @requires_unix_sockets
    async def test_direct_update_supersedes_pending_idle_update(self) -> None:
        agent_state.register(self.busy)
        reader, writer = await self._client()
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)

        await self._send(writer, {"type": "runUpdate"})
        await self._wait_pending_idle(reader, False)
        await self._wait_launches(1)
        self.assertIsNone(self.server._update_when_idle_task)

        # The poller was disarmed, so the task finishing launches nothing.
        self.busy.is_task_active = False
        await self._settle()
        self.assertEqual(self._launches(), 1)

    @requires_unix_sockets
    async def test_arming_twice_keeps_the_single_poller(self) -> None:
        """A second arm is ignored while a poller task exists.

        The same ``_update_when_idle_task is None`` gate also rejects a
        re-arm during a finished poller's final broadcast (after a very
        fast installer exit); that window cannot be held open
        deterministically without patching the broadcast, so only the
        armed case is exercised here.
        """
        agent_state.register(self.busy)
        reader, writer = await self._client()
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)
        first = self.server._update_when_idle_task

        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)
        self.assertIs(self.server._update_when_idle_task, first)

        self.busy.is_task_active = False
        await self._wait_launches(1)
        await self._settle()
        self.assertEqual(self._launches(), 1)

    @requires_unix_sockets
    async def test_not_armed_while_an_update_is_already_running(self) -> None:
        self._install_stub(
            HELD_INSTALL_SH.format(release=self.release, tmpdir=self.tmpdir),
        )
        reader, writer = await self._client()
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "runUpdate"})
        notice = await self._wait_for_event(reader, "notice")
        self.assertIn("getting installed", str(notice["text"]))
        for _ in range(200):
            if self.server._update_in_progress() and self.server._update_proc:
                break
            await asyncio.sleep(0.02)
        self.assertTrue(self.server._update_in_progress())

        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, False)
        self.assertIsNone(self.server._update_when_idle_task)

    @requires_unix_sockets
    async def test_shutdown_cancels_the_pending_poller(self) -> None:
        agent_state.register(self.busy)
        reader, writer = await self._client()
        await self._wait_pending_idle(reader, False)
        await self._send(writer, {"type": "updateWhenIdle"})
        await self._wait_pending_idle(reader, True)
        task = self.server._update_when_idle_task
        assert task is not None

        await self.server.stop_async()
        self.assertTrue(task.cancelled())
        self.assertIsNone(self.server._update_when_idle_task)
        self.busy.is_task_active = False
        await self._settle()
        self.assertEqual(self._launches(), 0)
