# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the self-update run-admission barrier (sweep 10/30).

* The shutdown sweep (``_stop_active_agent_tasks``) stops every task, so
  the daemon looks idle right before it exits.  Neither the
  deferred-restart loop nor an armed "Update when idle" may take that
  for idle: an installer spawned then would restart the daemon the user
  just stopped.
* A deferred-restart record whose environment ``subprocess.Popen``
  rejects (``ValueError: embedded null byte``) is dropped like any
  other unusable record; the loop keeps finishing later records.
  The loop also logs and survives any other exception of one poll,
  but nothing in a real poll raises any more (``claim`` turns an
  unreadable record into ``None``, ``_spawn_detached`` reports
  ``OSError`` and ``ValueError``), so that branch cannot be reached
  end to end without a test double and is not tested here.
* The idle poller raises the barrier and takes the installer's
  single-flight guard (``_update_starting``) in one event-loop step.
  An installer's exit watcher, a loop callback, lowers the barrier
  whenever no installer is in progress, so a loop step that saw the
  poller's barrier up with nothing in progress would let a watcher drop
  it and admit a run right before the update.  Waking a watcher at that
  exact moment cannot be scheduled from outside the event loop, so the
  test checks the invariant the watcher relies on: it parks the
  poller's idle check on ``STATE_LOCK``, releases it, and then watches
  from the loop thread itself (no other loop callback can run, as a
  watcher's step would) until the barrier goes up.

The daemon is a real ``RemoteAccessServer`` (``_UpdateCheckTestBase``);
the installers are bash scripts that write a mark and wait for the test
to release them.  A running task is an ``AgentState`` registered as
active.  Only the poll interval is shortened.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import kiss.server.web_server as ws
from kiss.server import agent_state
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

# Writes its arguments to {mark}, then holds until {release} exists.
INSTALLER_STUB = """#!/bin/bash
printf '%s\\n' "$*" > "{mark}.tmp" && mv "{mark}.tmp" "{mark}"
while [ ! -e "{release}" ]; do
    [ -d "{tmpdir}" ] || exit 5
    sleep 0.05
done
exit 0
"""


def hold_state_lock(held: threading.Event, go: threading.Event, limit: float) -> None:
    """Hold ``agent_state.STATE_LOCK`` until *go* is set or *limit* seconds pass."""
    with agent_state.STATE_LOCK:
        held.set()
        go.wait(limit)


class UpdateBarrierTest(_UpdateCheckTestBase):
    """The barrier is raised only for an installer that really starts."""

    async def asyncSetUp(self) -> None:
        # Before the server starts: its pollers read the interval per sleep.
        self._orig_poll = ws._IDLE_UPDATE_POLL_S
        ws._IDLE_UPDATE_POLL_S = 0.05
        await super().asyncSetUp()
        tmp = Path(self.tmpdir)
        self.release = tmp / "release"
        self.record = tmp / ".kiss-web.restart-pending"
        self.restart_mark = tmp / "restart-started"
        self.restart_installer = tmp / "checkout" / "install.sh"
        self._write_stub(self.restart_installer, self.restart_mark)
        self.update_mark = tmp / "update-started"
        root = tmp / "kiss_ai"
        self._write_stub(root / "install.sh", self.update_mark)
        self.server._install_root = root
        self.server._update_log_path = tmp / "update.log"
        self.workdir = tmp / "work"
        self.workdir.mkdir()
        self._tasks: list[agent_state.AgentState] = []

    async def asyncTearDown(self) -> None:
        for task in self._tasks:
            agent_state.unregister(task.task_id, task)
        self.release.write_text("")
        proc = self.server._update_proc
        if proc is not None and proc.poll() is None:
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait, 5), 6)
            except (TimeoutError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait()
        await super().asyncTearDown()
        ws._IDLE_UPDATE_POLL_S = self._orig_poll

    # -- helpers --------------------------------------------------------

    def _write_stub(self, path: Path, mark: Path) -> None:
        path.parent.mkdir()
        path.write_text(
            INSTALLER_STUB.format(mark=mark, release=self.release, tmpdir=self.tmpdir),
        )
        path.chmod(0o755)

    def _start_task(self, name: str) -> agent_state.AgentState:
        """Register an in-flight task, as a running agent run does."""
        task = agent_state.AgentState(
            f"barrier-sweep-{name}", tab_id=f"tab-{name}", server_owned=True,
            is_task_active=True,
        )
        agent_state.register(task)
        self._tasks.append(task)
        return task

    def _finish_task(self, task: agent_state.AgentState) -> None:
        agent_state.unregister(task.task_id, task)
        self._tasks.remove(task)

    def _write_record(self, **overrides: Any) -> None:
        fields = {
            "installer": str(self.restart_installer),
            "home": str(Path(self.tmpdir) / "home"),
            "workdir": str(self.workdir),
        }
        fields.update(overrides)
        tmp = self.record.with_name(self.record.name + ".tmp")
        tmp.write_text(json.dumps(fields) + "\n")
        os.replace(tmp, self.record)

    def _barrier_up(self) -> bool:
        with agent_state.STATE_LOCK:
            return bool(self.server._agent_server._update_installing)

    async def _wait_until(self, predicate: Any, what: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError(f"timed out waiting for {what}")
            await asyncio.sleep(0.02)

    async def _polls(self, count: int = 10) -> None:
        """Give the pollers *count* chances to act."""
        await asyncio.sleep(count * ws._IDLE_UPDATE_POLL_S + 0.1)

    async def _shutdown_sweep_then_finish(self, task: agent_state.AgentState) -> None:
        """Run the shutdown sweep, then let the task end as the sweep makes it."""
        await asyncio.to_thread(self.server._stop_active_agent_tasks, 2.0)
        self._finish_task(task)
        await self._polls()

    # -- tests ----------------------------------------------------------

    async def test_shutdown_sweep_does_not_start_a_deferred_restart(self) -> None:
        task = self._start_task("deferred")
        self._write_record()
        await self._polls()
        self.assertFalse(self.restart_mark.exists())

        await self._shutdown_sweep_then_finish(task)

        self.assertFalse(
            self.restart_mark.exists(),
            "the deferred restart started while the daemon was shutting down",
        )
        self.assertFalse(self._barrier_up())
        self.assertFalse(self.server._update_in_progress())

    async def test_shutdown_sweep_does_not_start_an_update_when_idle(self) -> None:
        task = self._start_task("idle-update")
        await self.server._handle_update_when_idle()
        await self._polls()
        self.assertFalse(self.update_mark.exists())

        await self._shutdown_sweep_then_finish(task)

        self.assertFalse(
            self.update_mark.exists(),
            "the idle update started its installer while the daemon was shutting down",
        )
        self.assertIsNone(self.server._update_proc)
        self.assertFalse(self._barrier_up())

    async def test_a_record_popen_rejects_is_dropped_and_the_loop_lives_on(self) -> None:
        # claim() accepts any non-empty home; Popen refuses the NUL in
        # the child's environment with ValueError, not OSError.
        self._write_record(home="bad\x00home")
        await self._wait_until(lambda: not self.record.exists(), "the bad record to be taken")
        await self._wait_until(
            lambda: not self.server._update_in_progress(), "the failed spawn to settle",
        )
        self.assertFalse(self._barrier_up())
        self.assertFalse(self.restart_mark.exists())

        self._write_record()
        await self._wait_until(self.restart_mark.exists, "the next restart to start")
        self.assertEqual(
            self.restart_mark.read_text().split(), ["--restart-only", "--non-interactive"],
        )
        self.assertTrue(self._barrier_up())
        self.assertTrue(self.server._update_in_progress())

    async def test_idle_poller_takes_the_installer_guard_with_the_barrier(self) -> None:
        held, go = threading.Event(), threading.Event()
        holder = threading.Thread(target=hold_state_lock, args=(held, go, 2.0), daemon=True)
        holder.start()
        self.assertTrue(held.wait(5))
        try:
            await self.server._handle_update_when_idle()
            # The poller takes its first step: its idle check waits for the lock.
            await asyncio.sleep(0.05)
        finally:
            go.set()
        # Watch from the loop thread, so no other loop callback runs, until
        # the idle check has raised the barrier.
        deadline = time.monotonic() + 10.0
        while not self._barrier_up():
            self.assertLess(time.monotonic(), deadline, "the barrier never went up")
            time.sleep(0.001)
        self.assertTrue(
            self.server._update_in_progress(),
            "the poller's barrier was up with no installer in progress",
        )
        holder.join(5)
        await self._wait_until(self.update_mark.exists, "the update installer to start")
