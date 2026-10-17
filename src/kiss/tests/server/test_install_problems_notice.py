# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the web app's notice about a failed install.

``install.sh`` records every failure of a run in
``$KISS_HOME/.install-problems`` (:mod:`kiss.server.install_problems`).
The daemon reports the file two ways:

* every page that sends ``ready`` gets a sticky ``error`` notification
  listing the lines while the file is less than a week old (the daemon
  the installer restarted is a new process, so this is the only way
  its failures reach the user);
* when the daemon's own update ends without restarting the daemon
  (``_watch_update_exit``), every window gets the same notification,
  whatever the installer's exit status; an installer that exits non-zero
  without recording a reason still gets the generic ``error`` to the
  window that clicked.

The daemon is real (local transport, fake PyPI endpoint, stub
``install.sh`` scripts that write the file or not).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from kiss.core.brand import PRODUCT_NAME
from kiss.server.install_problems import NOTICE_DAYS, NOTICE_ID, PROBLEMS_FILE
from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

HOOK_LINE = "WARNING: post-install hook /home/u/.s10s/post-install.d/brand exited with status 3"
RESTART_LINE = "WARNING: kiss-web was not restarted; start it with ~/.s10s/s10s/.venv/bin/kiss-web"
SYNC_LINE = "ERROR: uv sync failed in /home/u/.s10s/s10s"


class TestInstallProblemsNotice(_UpdateCheckTestBase):
    """The ``ready`` notice and the update watcher's notice."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.home = Path(self.tmpdir) / "kiss-home"
        self.home.mkdir()
        self._saved_home = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(self.home)
        self._writers: list[LocalWriter] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        if self._saved_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_home
        await super().asyncTearDown()

    @property
    def problems(self) -> Path:
        return self.home / PROBLEMS_FILE

    async def _ready(self, tab: str) -> tuple[LocalReader, list[dict[str, object]]]:
        """Connect, send ``ready`` and collect the events up to the ``tabs_state``
        that ends the daemon's answer; return the reader too for later events."""
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        await self._send_ready(writer, tab)
        events: list[dict[str, object]] = []
        for _ in range(200):
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            self.assertTrue(line, "local connection closed before tabs_state")
            msg = json.loads(line.decode("utf-8"))
            events.append(msg)
            if msg.get("type") == "tabs_state":
                return reader, events
        raise AssertionError("did not see tabs_state")

    def _notices(self, events: list[dict[str, object]]) -> list[dict[str, object]]:
        return [ev for ev in events if ev.get("id") == NOTICE_ID]

    async def test_fresh_file_sends_the_notice_on_ready(self) -> None:
        self.problems.write_text(f"{HOOK_LINE}\n\n{RESTART_LINE}\n")
        _, events = await self._ready("tab-1")
        notices = self._notices(events)
        self.assertEqual(len(notices), 1)
        notice = notices[0]
        self.assertEqual(notice["type"], "notification")
        self.assertEqual(notice["severity"], "error")
        self.assertIs(notice["sticky"], True)
        self.assertEqual(notice["title"], f"{PRODUCT_NAME} update: 2 problems")
        message = str(notice["message"])
        self.assertIn(f"The last {PRODUCT_NAME} install ran into 2 problems:\n", message)
        self.assertIn(f"\u2022 {HOOK_LINE}\n\u2022 {RESTART_LINE}\n", message)
        self.assertIn("~/.kiss/install.log", message)
        # Every (re)connecting page gets it, under the same id so a
        # reconnect replaces the toast instead of stacking one.
        _, events = await self._ready("tab-2")
        self.assertEqual(len(self._notices(events)), 1)

    async def test_one_problem_is_singular(self) -> None:
        self.problems.write_text(f"{SYNC_LINE}\n")
        _, events = await self._ready("tab-1")
        notice = self._notices(events)[0]
        self.assertEqual(notice["title"], f"{PRODUCT_NAME} update: 1 problem")
        self.assertIn("ran into 1 problem:\n", str(notice["message"]))

    async def test_no_file_or_empty_file_sends_no_notice(self) -> None:
        _, events = await self._ready("tab-1")
        self.assertEqual(self._notices(events), [])
        self.assertIn("tricksData", [ev.get("type") for ev in events])
        self.problems.write_text("\n  \n")
        _, events = await self._ready("tab-2")
        self.assertEqual(self._notices(events), [])

    async def test_stale_file_sends_no_notice(self) -> None:
        self.problems.write_text(f"{HOOK_LINE}\n")
        stale = time.time() - (NOTICE_DAYS * 86400 + 3600)
        os.utime(self.problems, (stale, stale))
        _, events = await self._ready("tab-1")
        self.assertEqual(self._notices(events), [])
        # Written again by a later install, it is fresh again.
        self.problems.write_text(f"{HOOK_LINE}\n")
        _, events = await self._ready("tab-2")
        self.assertEqual(len(self._notices(events)), 1)

    async def _run_update(self, script_body: str) -> tuple[LocalReader, LocalReader]:
        """Run the daemon's update with a stub installer from a clicking
        window and a bystander window; return both readers once the
        installer has exited and the watcher has reported."""
        install_root = Path(self.tmpdir) / "kiss_ai"
        install_root.mkdir(exist_ok=True)
        script = install_root / "install.sh"
        script.write_text("#!/bin/bash\n" + script_body)
        script.chmod(0o755)
        self.server._install_root = install_root
        self.server._update_log_path = Path(self.tmpdir) / "update.log"
        bystander, _ = await self._ready("tab-bystander")
        clicker, events = await self._ready("tab-clicker")
        writer = self._writers[-1]
        if not any(ev.get("type") == "update_available" for ev in events):
            await self._wait_for_event(clicker, "update_available")
        writer.write(json.dumps({"type": "runUpdate"}).encode("utf-8") + b"\n")
        await writer.drain()
        deadline = asyncio.get_running_loop().time() + 10
        while self.server._update_proc is None:
            self.assertLess(asyncio.get_running_loop().time(), deadline, "installer never ran")
            await asyncio.sleep(0.05)
        proc = self.server._update_proc
        await asyncio.to_thread(proc.wait, 5)
        # The watcher polls the installer every 0.2 s and reports then.
        await asyncio.sleep(0.5)
        return clicker, bystander

    async def _drain(self, reader: LocalReader) -> list[dict[str, object]]:
        """The events *reader* receives within two seconds (the test
        base's fake PyPI makes the daemon broadcast ``update_available``
        without pause, so the drain is bounded by time, not by silence)."""
        events: list[dict[str, object]] = []
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=0.3)
            except TimeoutError:
                break
            if not line:
                break
            events.append(json.loads(line.decode("utf-8")))
        return events

    async def test_watcher_reports_a_warning_of_a_clean_exit_to_every_window(self) -> None:
        clicker, bystander = await self._run_update(
            f"printf '%s\\n' {HOOK_LINE!r} > {str(self.problems)!r}\nexit 0\n"
        )
        for reader in (clicker, bystander):
            notices = self._notices(await self._drain(reader))
            self.assertEqual(len(notices), 1, "every window is told")
            self.assertEqual(notices[0]["severity"], "error")
            self.assertIn(f"\u2022 {HOOK_LINE}", str(notices[0]["message"]))

    async def test_watcher_reports_the_recorded_reason_of_a_failed_exit(self) -> None:
        clicker, bystander = await self._run_update(
            f"printf '%s\\n' {SYNC_LINE!r} > {str(self.problems)!r}\nexit 1\n"
        )
        events = await self._drain(clicker)
        self.assertEqual(len(self._notices(events)), 1)
        self.assertIn(f"\u2022 {SYNC_LINE}", str(self._notices(events)[0]["message"]))
        self.assertEqual([ev for ev in events if ev.get("type") == "error"], [],
                         "the recorded reason replaces the generic error")
        self.assertEqual(len(self._notices(await self._drain(bystander))), 1)

    async def test_failed_exit_without_a_record_keeps_the_generic_error(self) -> None:
        # An earlier run's file does not pass for this run's reason.
        self.problems.write_text(f"{HOOK_LINE}\n")
        old = time.time() - 3600
        os.utime(self.problems, (old, old))
        clicker, bystander = await self._run_update("exit 1\n")
        events = await self._drain(clicker)
        errors = [ev for ev in events if ev.get("type") == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn(f"{PRODUCT_NAME} update failed (exit 1)", str(errors[0]["text"]))
        self.assertEqual(self._notices(events), [])
        bystander_events = await self._drain(bystander)
        self.assertEqual([ev for ev in bystander_events if ev.get("type") == "error"], [])
        self.assertEqual(self._notices(bystander_events), [])

    async def test_clean_exit_without_a_record_closes_the_notice(self) -> None:
        # The daemon stayed (restart deferred, say) with the toast of an
        # earlier failed run on its pages: a clean run takes it down.
        clicker, bystander = await self._run_update("exit 0\n")
        for reader in (clicker, bystander):
            events = await self._drain(reader)
            notices = self._notices(events)
            self.assertEqual(len(notices), 1)
            self.assertEqual(
                {key: notices[0][key] for key in ("type", "id", "close")},
                {"type": "notification", "id": NOTICE_ID, "close": True},
            )
            self.assertEqual([ev for ev in events if ev.get("type") == "error"], [])
