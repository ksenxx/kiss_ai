# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the retired VS Code extension's aftermath in the
daemon.

``install.sh`` writes ``$KISS_HOME/.vscode-retired`` when it finds a
copy of the retired extension; every page that then sends ``ready`` gets
a sticky ``notification`` saying the extension is gone for good and
naming the web app's loopback URL, until the flag is a week old
(:mod:`kiss.server.vscode_retired`).  And the daemon's own update runs
the installer from the daemon's work dir, not from the bootstrap clone
the installer would otherwise serve as the user's work dir.

The daemon is real (local transport, fake PyPI endpoint, a stub
``install.sh`` that records its working directory).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from kiss.server.install_problems import NOTICE_DAYS
from kiss.server.vscode_retired import (
    NOTICE_ID,
    VSCODE_RETIRED_DISMISSED,
    VSCODE_RETIRED_FLAG,
)
from kiss.tests.local_ws import LocalWriter
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

INSTALL_SH = """#!/bin/bash
: > "./{stamp}"
pwd > "{marker}"
exit 0
"""
"""Stub installer: touches *stamp* in its working directory, then records that directory.

The stamp is what the tests check — Git for Windows' bash prints MSYS
paths (``/c/Users/...``, ``/tmp/...``) from ``pwd``, so the recorded
directory is for diagnostics only.
"""


class TestVscodeRetired(_UpdateCheckTestBase):
    """The ``ready`` notice and the update's working directory."""

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

    async def _ready_events(self) -> list[dict[str, object]]:
        """Connect, send ``ready`` and collect the events up to the
        ``tabs_state`` that ends the daemon's answer to it (the bootstrap
        events are broadcast through the printer and land before it)."""
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        await self._send_ready(writer, "tab-vscode")
        events: list[dict[str, object]] = []
        for _ in range(200):
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            self.assertTrue(line, "local connection closed before tabs_state")
            msg = json.loads(line.decode("utf-8"))
            events.append(msg)
            if msg.get("type") == "tabs_state":
                return events
        raise AssertionError("did not see tabs_state")

    def _notices(self, events: list[dict[str, object]]) -> list[dict[str, object]]:
        return [ev for ev in events if ev.get("id") == NOTICE_ID]

    async def test_fresh_flag_sends_the_notice_on_ready(self) -> None:
        (self.home / VSCODE_RETIRED_FLAG).write_text("2026-10-15T00:00:00Z\n")
        notices = self._notices(await self._ready_events())
        self.assertEqual(len(notices), 1)
        notice = notices[0]
        self.assertEqual(notice["type"], "notification")
        self.assertEqual(notice["severity"], "warning")
        self.assertIs(notice["sticky"], True)
        self.assertEqual(notice["title"], "VS Code extension removed")
        message = str(notice["message"])
        self.assertIn("VS Code extension has been removed and will never be updated", message)
        self.assertIn(f"Use this web app instead: http://127.0.0.1:{self.server.port}", message)
        # Every (re)connecting page gets it, under the same id so a
        # reconnect replaces the toast instead of stacking one.
        self.assertEqual(len(self._notices(await self._ready_events())), 1)

    async def test_no_flag_sends_no_notice(self) -> None:
        events = await self._ready_events()
        self.assertEqual(self._notices(events), [])
        # The other bootstrap events still arrive.
        self.assertIn("tricksData", [ev.get("type") for ev in events])
        self.assertIn("tipsData", [ev.get("type") for ev in events])

    async def test_stale_flag_sends_no_notice(self) -> None:
        flag = self.home / VSCODE_RETIRED_FLAG
        flag.write_text("2026-10-01T00:00:00Z\n")
        stale = time.time() - (NOTICE_DAYS * 86400 + 3600)
        os.utime(flag, (stale, stale))
        self.assertEqual(self._notices(await self._ready_events()), [])
        # Touched again by a later install, it is fresh again.
        flag.write_text("2026-10-15T00:00:00Z\n")
        self.assertEqual(len(self._notices(await self._ready_events())), 1)

    async def test_notice_offers_do_not_show_this(self) -> None:
        (self.home / VSCODE_RETIRED_FLAG).write_text("2026-10-15T00:00:00Z\n")
        notice = self._notices(await self._ready_events())[0]
        self.assertIs(notice["dontShowAgain"], True)

    async def test_dismiss_notice_closes_everywhere_and_for_good(self) -> None:
        (self.home / VSCODE_RETIRED_FLAG).write_text("2026-10-15T00:00:00Z\n")
        # Two windows see the toast.
        reader_a, writer_a = await self._connect_local()
        self._writers.append(writer_a)
        await self._send_ready(writer_a, "tab-a")
        await self._wait_for_event(reader_a, "tabs_state")
        reader_b, writer_b = await self._connect_local()
        self._writers.append(writer_b)
        await self._send_ready(writer_b, "tab-b")
        await self._wait_for_event(reader_b, "tabs_state")
        # Window A ticks "Do not show this".
        writer_a.write(
            json.dumps({"type": "dismissNotice", "id": NOTICE_ID}).encode("utf-8") + b"\n"
        )
        await writer_a.drain()
        # Both windows get the close for that toast...
        for reader in (reader_a, reader_b):
            close = await self._wait_for_event(reader, "notification")
            close.pop("ts", None)
            self.assertEqual(close, {"type": "notification", "id": NOTICE_ID, "close": True})
        # ...the choice is on disk...
        marker = self.home / VSCODE_RETIRED_DISMISSED
        self.assertTrue(marker.is_file())
        self.assertRegex(marker.read_text(), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\n$")
        # ...and no later page gets the notice, even after a later
        # install refreshes the flag.
        self.assertEqual(self._notices(await self._ready_events()), [])
        (self.home / VSCODE_RETIRED_FLAG).write_text("2026-10-16T00:00:00Z\n")
        self.assertEqual(self._notices(await self._ready_events()), [])

    async def test_dismiss_notice_ignores_other_ids(self) -> None:
        (self.home / VSCODE_RETIRED_FLAG).write_text("2026-10-15T00:00:00Z\n")
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        await self._send_ready(writer, "tab-x")
        await self._wait_for_event(reader, "tabs_state")
        writer.write(
            json.dumps({"type": "dismissNotice", "id": "install-problems"}).encode("utf-8")
            + b"\n"
        )
        await writer.drain()
        # No marker, no close: the next ping answer arrives first.
        writer.write(json.dumps({"type": "ping"}).encode("utf-8") + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        self.assertNotEqual(json.loads(line.decode("utf-8")).get("type"), "notification")
        self.assertFalse((self.home / VSCODE_RETIRED_DISMISSED).exists())
        self.assertEqual(len(self._notices(await self._ready_events())), 1)

    async def _run_update(self, marker: Path, stamp: str) -> str:
        """Run the daemon's update with a stub installer; return the cwd it recorded."""
        install_root = Path(self.tmpdir) / "kiss_ai"
        install_root.mkdir(exist_ok=True)
        script = install_root / "install.sh"
        script.write_text(INSTALL_SH.format(marker=marker, stamp=stamp))
        script.chmod(0o755)
        self.server._install_root = install_root
        self.server._update_log_path = Path(self.tmpdir) / "update.log"
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        await self._send_ready(writer, "tab-update")
        await self._wait_for_event(reader, "update_available")
        writer.write(json.dumps({"type": "runUpdate"}).encode("utf-8") + b"\n")
        await writer.drain()
        deadline = asyncio.get_running_loop().time() + 10
        while not marker.exists() or not marker.read_text().strip():
            self.assertLess(asyncio.get_running_loop().time(), deadline, "installer never ran")
            await asyncio.sleep(0.05)
        proc = self.server._update_proc
        if proc is not None:
            await asyncio.to_thread(proc.wait, 5)
        return marker.read_text().strip()

    async def test_update_runs_the_installer_from_the_daemon_work_dir(self) -> None:
        work_dir = Path(self.tmpdir) / "projects"
        work_dir.mkdir()
        self.server.work_dir = str(work_dir)
        await self._assert_update_runs_in(work_dir, Path(self.tmpdir) / "cwd-1")

    async def test_update_without_a_work_dir_runs_from_home(self) -> None:
        self.server.work_dir = str(Path(self.tmpdir) / "gone")
        await self._assert_update_runs_in(Path.home(), Path(self.tmpdir) / "cwd-2")

    async def _assert_update_runs_in(self, expected: Path, marker: Path) -> None:
        """Run the daemon's update and check the stub installer ran inside *expected*."""
        stamp = f".kiss-test-update-cwd-{marker.name}-{os.getpid()}"
        try:
            cwd = await self._run_update(marker, stamp)
            self.assertTrue((expected / stamp).is_file(), f"installer ran in {cwd!r}")
        finally:
            (expected / stamp).unlink(missing_ok=True)

