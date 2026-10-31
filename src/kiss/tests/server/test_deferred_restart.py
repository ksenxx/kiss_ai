# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: kiss-web finishes the restart an installer deferred.

``install.sh`` does not restart a daemon with tasks in flight.  It records
the deferred restart in the file the daemon names in its
``activeTasksResponse`` (``restartPendingFile``), and the daemon's
``_deferred_restart_loop`` runs the recorded installer with
``--restart-only`` once no task is in flight, holding the run-admission
barrier while it runs.

The daemon is a real ``RemoteAccessServer`` whose home is a temp dir
(``url_file=`` there).  The recorded "installer" is a bash script that
writes down how it was started and then waits for the test to release
it, so the barrier can be observed while it runs; it exits on its own
once the temp dir is gone.  The update installer of the Update-button
test is another such script that records a deferred restart the way
``install.sh`` does.  A running task is an ``AgentState`` registered as
active, as in ``test_check_active_tasks_script``.  Nothing is patched;
only the poll interval is shortened.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import kiss.server.web_server as ws
from kiss.server import agent_state
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

# Writes "<args>\n<KISS_HOME>\n<physical cwd>" to {mark}, then holds until
# {release} exists.  Git Bash spells the cwd ``/c/Users/...`` (and the
# temp directory ``/tmp``); its ``pwd -W`` gives the Windows form the
# tests compare with.
RESTART_STUB = """#!/bin/bash
printf '%s\\n' "$*" "$KISS_HOME" "$({pwd})" > "{mark}.tmp" && mv "{mark}.tmp" "{mark}"
while [ ! -e "{release}" ]; do
    [ -d "{tmpdir}" ] || exit 5
    sleep 0.05
done
exit 0
"""

# The Update button's installer deferring its restart: once {go} exists
# it records the restart in {record}, as install.sh's
# record_deferred_restart does, then exits once {done} exists.
UPDATE_STUB = """#!/bin/bash
while [ ! -e "{go}" ]; do
    [ -d "{tmpdir}" ] || exit 5
    sleep 0.05
done
printf '%s\\n' '{record_json}' > "{record}.tmp" && mv "{record}.tmp" "{record}"
while [ ! -e "{done}" ]; do
    [ -d "{tmpdir}" ] || exit 5
    sleep 0.05
done
exit 0
"""


class DeferredRestartTest(_UpdateCheckTestBase):
    """The daemon runs ``install.sh --restart-only`` once its tasks are done."""

    async def asyncSetUp(self) -> None:
        # Before the server starts: its poller reads the interval per sleep.
        self._orig_poll = ws._IDLE_UPDATE_POLL_S
        ws._IDLE_UPDATE_POLL_S = 0.05
        await super().asyncSetUp()
        tmp = Path(self.tmpdir)
        self.record = tmp / ".kiss-web.restart-pending"
        self.mark = tmp / "restart-started"
        self.release = tmp / "release"
        self.installer = tmp / "checkout" / "install.sh"
        self.installer.parent.mkdir()
        self.installer.write_text(
            RESTART_STUB.format(
                mark=self.mark, release=self.release, tmpdir=self.tmpdir,
                pwd="pwd -W" if os.name == "nt" else "pwd -P",
            ),
        )
        self.installer.chmod(0o755)
        self.home = tmp / "brand-home"
        self.workdir = tmp / "work"
        self.workdir.mkdir()
        self.server._update_log_path = tmp / "update.log"
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

    def _start_task(self, name: str) -> agent_state.AgentState:
        """Register an in-flight task, as a running agent run does."""
        task = agent_state.AgentState(
            f"deferred-restart-{name}", tab_id=f"tab-{name}", server_owned=True,
            is_task_active=True,
        )
        agent_state.register(task)
        self._tasks.append(task)
        return task

    def _finish_task(self, task: agent_state.AgentState) -> None:
        agent_state.unregister(task.task_id, task)
        self._tasks.remove(task)

    def _record_json(self, **overrides: Any) -> str:
        fields = {
            "installer": str(self.installer),
            "home": str(self.home),
            "workdir": str(self.workdir),
        }
        fields.update(overrides)
        return json.dumps(fields)

    def _write_record(self, text: str) -> None:
        tmp = self.record.with_name(self.record.name + ".tmp")
        tmp.write_text(text + "\n")
        os.replace(tmp, self.record)

    def _barrier_up(self) -> bool:
        with agent_state.STATE_LOCK:
            return bool(self.server._agent_server._update_installing)

    def _started(self) -> list[str]:
        """The restart installer's mark: its args, KISS_HOME and cwd."""
        return self.mark.read_text().splitlines()

    async def _wait_until(self, predicate: Any, what: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError(f"timed out waiting for {what}")
            await asyncio.sleep(0.02)

    async def _polls(self, count: int = 10) -> None:
        """Give the poller *count* chances to act."""
        await asyncio.sleep(count * ws._IDLE_UPDATE_POLL_S + 0.1)

    async def _assert_dropped(self) -> None:
        """The record goes away without a restart and the barrier stays down."""
        await self._wait_until(lambda: not self.record.exists(), "the record to be dropped")
        await self._polls()
        self.assertFalse(self.mark.exists())
        self.assertFalse(self._barrier_up())
        self.assertFalse(self.server._update_in_progress())
        leftovers = [
            p.name for p in Path(self.tmpdir).iterdir() if p.name.startswith(self.record.name)
        ]
        self.assertEqual(leftovers, [])

    # -- tests ----------------------------------------------------------

    async def test_the_daemon_names_its_record_to_the_installer(self) -> None:
        self.assertEqual(self.server._restart_pending_path(), self.record)

    async def test_restart_waits_for_the_running_task_then_runs_the_installer(self) -> None:
        task = self._start_task("busy")
        self._write_record(self._record_json())
        await self._polls()
        # A task is in flight: nothing happens and runs are still admitted.
        self.assertFalse(self.mark.exists())
        self.assertTrue(self.record.exists())
        self.assertFalse(self._barrier_up())

        self._finish_task(task)
        await self._wait_until(self.mark.exists, "the restart installer to start")
        args, kiss_home, cwd = self._started()
        self.assertEqual([args, kiss_home], ["--restart-only --non-interactive", str(self.home)])
        self.assertEqual(Path(cwd), self.workdir.resolve())
        self.assertFalse(self.record.exists())
        # The restart owns the barrier and the single-flight guard.
        self.assertTrue(self._barrier_up())
        self.assertTrue(self.server._update_in_progress())

        # It exits without restarting the daemon: runs are admitted again
        # and nothing else is started.
        self.release.write_text("")
        await self._wait_until(lambda: not self._barrier_up(), "the barrier to drop")
        self.assertFalse(self.server._update_in_progress())
        self.mark.unlink()
        await self._polls()
        self.assertFalse(self.mark.exists())

    async def test_a_record_left_by_the_update_installer_is_finished_after_it_exits(
        self,
    ) -> None:
        """The Update button's installer defers: the daemon restarts once it is gone."""
        tmp = Path(self.tmpdir)
        go, done = tmp / "go", tmp / "done"
        root = tmp / "kiss_ai"
        root.mkdir()
        self.server._install_root = root
        update = root / "install.sh"
        update.write_text(UPDATE_STUB.format(
            go=go, done=done, tmpdir=self.tmpdir, record=self.record,
            record_json=self._record_json(),
        ))
        update.chmod(0o755)

        await self.server._handle_run_update("")
        await self._wait_until(
            lambda: self.server._update_proc is not None and not self.server._update_starting,
            "the update installer to start",
        )
        update_proc = self.server._update_proc
        go.write_text("")
        await self._wait_until(self.record.exists, "the deferred restart to be recorded")
        await self._polls()
        # The update installer still runs: the record waits for it.
        self.assertFalse(self.mark.exists())
        self.assertTrue(self.record.exists())

        done.write_text("")
        await self._wait_until(self.mark.exists, "the restart installer to start")
        assert update_proc is not None
        self.assertEqual(update_proc.poll(), 0)
        self.assertEqual(self._started()[0], "--restart-only --non-interactive")
        self.assertTrue(self._barrier_up())
        self.assertIsNot(self.server._update_proc, update_proc)

    async def test_a_record_older_than_the_daemon_is_still_finished(self) -> None:
        """A daemon restarted another way (a reboot) still runs from the old unit."""
        self._write_record(self._record_json())
        an_hour_ago = time.time() - 3600
        os.utime(self.record, (an_hour_ago, an_hour_ago))
        await self._wait_until(self.mark.exists, "the restart installer to start")
        self.assertEqual(self._started()[0], "--restart-only --non-interactive")
        self.assertFalse(self.record.exists())

    async def test_a_record_naming_no_installer_is_dropped(self) -> None:
        self._write_record(self._record_json(installer=str(Path(self.tmpdir) / "gone.sh")))
        await self._assert_dropped()

    async def test_a_record_without_a_home_is_dropped(self) -> None:
        self._write_record(self._record_json(home=""))
        await self._assert_dropped()

    async def test_an_unreadable_record_is_dropped(self) -> None:
        self._write_record("{not json")
        await self._assert_dropped()

    async def test_a_record_that_is_not_an_object_is_dropped(self) -> None:
        self._write_record(json.dumps([str(self.installer)]))
        await self._assert_dropped()

    async def test_a_gone_work_dir_falls_back_to_home(self) -> None:
        self._write_record(self._record_json(workdir=str(Path(self.tmpdir) / "gone")))
        await self._wait_until(self.mark.exists, "the restart installer to start")
        self.assertEqual(Path(self._started()[2]), Path.home().resolve())
