# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: no task can start once the self-update is launching.

``_run_update_when_idle`` used to observe "no active tasks", yield to
the event loop several times (executor hops for the install-script
lookup and the spawn) and only then start ``install.sh``, which
restarts the daemon.  A ``run`` submitted in that window started a
task the restart killed mid-flight.

The fix is a run-admission barrier (``VSCodeServer._update_installing``):
the idle poller raises it in the SAME ``STATE_LOCK`` critical section
as its idle verdict (:meth:`RemoteAccessServer._arm_update_barrier_if_idle`),
a direct Update click raises it before spawning, and ``_cmd_run``
refuses a fresh task while it is up.  The barrier is lowered when the
installer exits (any exit code) or the spawn fails, so a daemon the
installer did not restart accepts tasks again.

The daemon is real (UDS transport, fake PyPI endpoint, stub
``install.sh`` scripts); admitted runs use a model name that is not
configured, so ``_run_task`` takes its own "No model available" exit
after registering the run — no model is ever called and nothing is
patched.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import kiss.server.agent_state as agent_state
import kiss.server.web_server as ws
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

TAB = "tab-a0924-barrier"
_UNKNOWN_MODEL = "audit0924-barrier-no-such-model"

# Blocks until the release file appears (see test_update_when_idle.py);
# gives up once the tmpdir is gone because it runs detached.
HELD_INSTALL_SH = """#!/bin/bash
while [ ! -e "{release}" ]; do
    [ -d "{tmpdir}" ] || exit 5
    sleep 0.05
done
exit 0
"""

FAILING_INSTALL_SH = """#!/bin/bash
echo "boom" >&2
exit 1
"""


class TestUpdateRunBarrier(_UpdateCheckTestBase):
    """Fresh runs are refused while the installer is launching or running."""

    async def asyncSetUp(self) -> None:
        self._orig_poll = ws._IDLE_UPDATE_POLL_S
        ws._IDLE_UPDATE_POLL_S = 0.05
        await super().asyncSetUp()
        self.install_root = Path(self.tmpdir) / "kiss_ai"
        self.install_root.mkdir()
        self.server._install_root = self.install_root
        self.server._update_log_path = Path(self.tmpdir) / "update.log"
        self.release = Path(self.tmpdir) / "release"
        self._install_stub(
            HELD_INSTALL_SH.format(release=self.release, tmpdir=self.tmpdir),
        )
        self.vscode = self.server._vscode_server
        self._writers: list[asyncio.StreamWriter] = []
        self._threads: list[threading.Thread] = []
        self._hold = threading.Event()
        self._clear_tab()

    async def asyncTearDown(self) -> None:
        self._hold.set()
        for th in self._threads:
            th.join(timeout=5)
        self.release.write_text("")
        proc = self.server._update_proc
        if proc is not None and proc.poll() is None:
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait, 5), 6)
            except (TimeoutError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait()
        await self._wait_tab_idle()
        self._clear_tab()
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await super().asyncTearDown()
        ws._IDLE_UPDATE_POLL_S = self._orig_poll

    # -- helpers --------------------------------------------------------

    def _install_stub(self, body: str) -> None:
        script = self.install_root / "install.sh"
        script.write_text(body)
        script.chmod(0o755)

    def _clear_tab(self) -> None:
        with agent_state.STATE_LOCK:
            stale = [
                st.task_id for st in agent_state.snapshot() if st.tab_id == TAB
            ]
        for task_id in stale:
            agent_state.unregister(task_id)

    def _run_cmd(self, prompt: str) -> dict[str, Any]:
        return {
            "type": "run",
            "tabId": TAB,
            "prompt": prompt,
            "workDir": self.tmpdir,
            "model": _UNKNOWN_MODEL,
            "useWorktree": False,
            "autoCommit": False,
            "classifyTasks": False,
        }

    def _barrier_up(self) -> bool:
        with agent_state.STATE_LOCK:
            return bool(self.vscode._update_installing)

    async def _client(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await self._connect_uds()
        self._writers.append(writer)
        await self._send_ready(writer, TAB)
        return reader, writer

    async def _send(self, writer: asyncio.StreamWriter, cmd: dict[str, object]) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _wait_until(self, predicate: Any, what: str, timeout: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(f"timed out waiting for {what}")
            await asyncio.sleep(0.02)

    async def _wait_installer_running(self) -> None:
        await self._wait_until(
            lambda: self.server._update_proc is not None
            and self.server._update_proc.poll() is None
            and not self.server._update_starting,
            "the installer to be running",
        )

    async def _wait_tab_idle(self) -> None:
        def idle() -> bool:
            with agent_state.STATE_LOCK:
                st = agent_state.find_by_tab(TAB)
                return st is None or (
                    st.task_thread is None and not st.is_task_active
                )
        await self._wait_until(idle, "the tab to go idle")

    async def _submit_and_expect_refusal(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Send a ``run`` on TAB and assert the update barrier refused it."""
        await self._send(writer, self._run_cmd("started during the update"))
        status = await self._wait_for_event(reader, "status")
        self.assertEqual(status.get("tabId"), TAB)
        self.assertFalse(status.get("running"))
        err = await self._wait_for_event(reader, "error")
        self.assertEqual(err.get("tabId"), TAB)
        self.assertIn("update is being installed", str(err.get("text")))
        # Nothing was registered for the tab: no state, no thread.
        with agent_state.STATE_LOCK:
            self.assertIsNone(agent_state.find_by_tab(TAB))

    async def _submit_and_expect_admission(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Send a ``run`` on TAB and assert ``_run_task`` really ran it."""
        await self._send(writer, self._run_cmd("started after the update"))
        # The admitted run reaches the runner's own model check.
        result = await self._wait_for_event(reader, "result")
        self.assertEqual(result.get("tabId"), TAB)
        self.assertIn("No model available", str(result.get("text")))
        await self._wait_tab_idle()
        st = agent_state.find_by_tab(TAB)
        assert st is not None
        self.assertEqual(st.last_user_prompt, "started after the update")

    # -- tests ----------------------------------------------------------

    @requires_unix_sockets
    async def test_direct_update_refuses_runs_until_the_installer_exits(self) -> None:
        reader, writer = await self._client()
        await self._wait_for_event(reader, "update_available")
        self.assertFalse(self._barrier_up())

        await self._send(writer, {"type": "runUpdate"})
        notice = await self._wait_for_event(reader, "notice")
        self.assertIn("getting installed", str(notice["text"]))
        await self._wait_installer_running()
        self.assertTrue(self._barrier_up())
        await self._submit_and_expect_refusal(reader, writer)

        # The installer exits (a run the daemon survived): the barrier
        # drops and the same submit is admitted.
        self.release.write_text("")
        await self._wait_until(lambda: not self._barrier_up(), "the barrier to drop")
        await self._submit_and_expect_admission(reader, writer)

    @requires_unix_sockets
    async def test_idle_poller_arms_barrier_with_its_idle_verdict(self) -> None:
        busy = agent_state.AgentState(
            "a0924-barrier-busy", tab_id="tab-a0924-busy", is_task_active=True,
        )
        agent_state.register(busy)
        try:
            reader, writer = await self._client()
            await self._wait_for_event(reader, "update_available")
            await self._send(writer, {"type": "updateWhenIdle"})
            await self._wait_until(
                lambda: self.server._update_when_idle_armed, "the poller to arm",
            )
            # A live task keeps both the installer AND the barrier down.
            await asyncio.sleep(ws._IDLE_UPDATE_POLL_S * 6)
            self.assertFalse(self._barrier_up())
            self.assertIsNone(self.server._update_proc)

            # The task ends: from the poller's idle verdict on, no new
            # task may start — before, during and after the spawn.
            busy.is_task_active = False
            await self._wait_until(self._barrier_up, "the barrier to rise")
            await self._wait_installer_running()
            self.assertTrue(self._barrier_up())
            await self._submit_and_expect_refusal(reader, writer)

            self.release.write_text("")
            await self._wait_until(lambda: not self._barrier_up(), "the barrier to drop")
            await self._submit_and_expect_admission(reader, writer)
        finally:
            agent_state.unregister(busy.task_id, busy)

    @requires_unix_sockets
    async def test_failed_installer_lowers_the_barrier(self) -> None:
        self._install_stub(FAILING_INSTALL_SH)
        reader, writer = await self._client()
        await self._wait_for_event(reader, "update_available")
        await self._send(writer, {"type": "runUpdate"})
        err = await self._wait_for_event(reader, "error")
        self.assertIn("update failed", str(err.get("text")))
        self.assertFalse(self._barrier_up())
        await self._submit_and_expect_admission(reader, writer)

    @requires_unix_sockets
    async def test_running_task_still_accepts_steering_during_update(self) -> None:
        # A task that was already running when the update started is
        # the user's explicit choice: typing into its tab still steers
        # it instead of being refused.
        worker = threading.Thread(target=self._hold.wait, daemon=True)
        worker.start()
        self._threads.append(worker)
        running = agent_state.AgentState(
            "a0924-barrier-running", tab_id=TAB, server_owned=True,
            task_thread=worker, is_task_active=True,
        )
        agent_state.register(running)
        try:
            reader, writer = await self._client()
            await self._wait_for_event(reader, "update_available")
            await self._send(writer, {"type": "runUpdate"})
            await self._wait_installer_running()
            self.assertTrue(self._barrier_up())

            await self._send(writer, self._run_cmd("steer the running task"))
            echo = await self._wait_for_event(reader, "prompt")
            self.assertEqual(echo.get("text"), "steer the running task")
            self.assertEqual(echo.get("tabId"), TAB)
            with agent_state.STATE_LOCK:
                self.assertEqual(
                    running.pending_user_messages, ["steer the running task"],
                )
        finally:
            self._hold.set()
            agent_state.unregister(running.task_id, running)

    @requires_unix_sockets
    async def test_barrier_not_armed_while_a_task_thread_is_installed(self) -> None:
        # ``busy()`` counts an installed worker thread even before the
        # run raises ``is_task_active`` (the start window): the idle
        # re-verification must refuse to arm for it too.
        worker = threading.Thread(target=self._hold.wait, daemon=True)
        worker.start()
        self._threads.append(worker)
        starting = agent_state.AgentState(
            "a0924-barrier-starting", tab_id="tab-a0924-starting",
            server_owned=True, task_thread=worker,
        )
        agent_state.register(starting)
        try:
            self.assertFalse(
                await asyncio.to_thread(self.server._arm_update_barrier_if_idle),
            )
            self.assertFalse(self._barrier_up())
        finally:
            agent_state.unregister(starting.task_id, starting)
        self.assertTrue(
            await asyncio.to_thread(self.server._arm_update_barrier_if_idle),
        )
        self.assertTrue(self._barrier_up())
        with agent_state.STATE_LOCK:
            self.vscode._update_installing = False

    @requires_unix_sockets
    async def test_cmd_run_refuses_under_barrier_without_state_or_thread(self) -> None:
        # No installer involved: the barrier set the way the daemon
        # sets it is enough for ``_cmd_run`` to refuse with the
        # status/error pair and register nothing (a worker thread is
        # only ever created together with the tab's state).
        reader, writer = await self._client()
        await self._wait_for_event(reader, "update_available")
        self.server._set_update_barrier(True)
        try:
            await self._submit_and_expect_refusal(reader, writer)
        finally:
            self.server._set_update_barrier(False)
        await self._submit_and_expect_admission(reader, writer)

    @requires_unix_sockets
    async def test_arming_and_admission_never_overlap(self) -> None:
        # Stress the atomicity: an arming attempt races a submit many
        # times.  Whatever the interleaving, a run must never be found
        # alive under a raised barrier — either the submit saw the
        # barrier (refused) or the poller saw the live thread (deferred).
        violations = 0
        for i in range(25):
            with agent_state.STATE_LOCK:
                self.vscode._update_installing = False
            self._clear_tab()
            start = threading.Barrier(2)
            outcome: dict[str, Any] = {}

            def arm() -> None:
                start.wait()
                outcome["armed"] = self.server._arm_update_barrier_if_idle()

            def submit() -> None:
                start.wait()
                self.vscode._cmd_run(self._run_cmd(f"race {i}"))
                with agent_state.STATE_LOCK:
                    st = agent_state.find_by_tab(TAB)
                    outcome["live_under_barrier"] = (
                        st is not None
                        and st.task_thread is not None
                        and bool(self.vscode._update_installing)
                    )

            t1 = threading.Thread(target=arm)
            t2 = threading.Thread(target=submit)
            t1.start()
            t2.start()
            await asyncio.to_thread(t1.join)
            await asyncio.to_thread(t2.join)
            if outcome["live_under_barrier"]:
                violations += 1
            await self._wait_tab_idle()
        with agent_state.STATE_LOCK:
            self.vscode._update_installing = False
        self.assertEqual(violations, 0)
