# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the "Remind me later" update snooze.

The sticky "update available" toast in the sidebar webview and the
remote webapp is driven by the daemon's ``update_available``
broadcast.  A ``snoozeUpdate`` client command records a 24-hour
snooze in ``$KISS_HOME/.update-check.json`` — the SAME cache file the
VS Code extension host's ``UpdateChecker.js`` uses — and the daemon
rebroadcasts with ``snoozed: true`` so every client's toast
disappears and stays away across window reloads until the snooze
expires or a NEWER release ships.

These tests reuse the real-server harness of
``test_update_available_check`` (real UDS connections, a real local
HTTP server impersonating PyPI, no mocks) and redirect the daemon's
KISS home to a temp dir so the real ``~/.kiss/.update-check.json``
is never touched.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path

import kiss.server.web_server as ws
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

_DAY_MS = 24 * 60 * 60 * 1000


class _SnoozeTestBase(_UpdateCheckTestBase):
    """Update-check harness with the KISS home redirected to a temp dir."""

    PYPI_VERSION = "2099.1.1"
    PYPI_PAYLOAD: dict[str, object] | None = {
        "info": {"version": "2099.1.1"},
    }

    async def asyncSetUp(self) -> None:
        self._home_tmp = tempfile.mkdtemp(prefix="kiss-snooze-home-")
        self._saved_kiss_home = ws._KISS_HOME
        ws._KISS_HOME = Path(self._home_tmp)
        await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        try:
            await super().asyncTearDown()
        finally:
            ws._KISS_HOME = self._saved_kiss_home
            shutil.rmtree(self._home_tmp, ignore_errors=True)

    def _cache_path(self) -> Path:
        return Path(self._home_tmp) / ".update-check.json"

    def _write_cache(self, data: dict[str, object]) -> None:
        self._cache_path().write_text(json.dumps(data), encoding="utf-8")

    async def _snooze(self, writer: asyncio.StreamWriter, latest: str) -> None:
        writer.write(
            json.dumps({"type": "snoozeUpdate", "latest": latest})
            .encode("utf-8") + b"\n",
        )
        await writer.drain()

    async def _close(self, writer: asyncio.StreamWriter) -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def _wait_for_snoozed(
        self,
        reader: asyncio.StreamReader,
        want: bool,
        timeout: float = 5.0,
    ) -> dict[str, object]:
        """Read ``update_available`` events until ``snoozed == want``.

        The harness shrinks the periodic check interval to 0.2s, so an
        ``update_available`` broadcast queued BEFORE a just-sent
        ``snoozeUpdate`` command can arrive after it; skipping stale
        events keeps the assertions race-free.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise AssertionError(
                    f"no update_available with snoozed={want}",
                )
            ev = await self._wait_for_event(
                reader, "update_available", timeout=remaining,
            )
            if ev.get("snoozed") == want:
                return ev


class TestSnoozeUpdateCommand(_SnoozeTestBase):
    """The ``snoozeUpdate`` command records and rebroadcasts the snooze."""

    async def test_snooze_records_and_rebroadcasts(self) -> None:
        """Clicking "Remind me later" silences every window at once."""
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-snooze-1")
            ev = await self._wait_for_event(reader, "update_available")
            self.assertEqual(ev.get("available"), True)
            self.assertEqual(ev.get("snoozed"), False)

            before_ms = int(time.time() * 1000)
            await self._snooze(writer, "2099.1.1")
            ev2 = await self._wait_for_snoozed(reader, True)
            after_ms = int(time.time() * 1000)
            self.assertEqual(ev2.get("available"), True)

            cache = json.loads(self._cache_path().read_text(encoding="utf-8"))
            self.assertEqual(cache["snoozedLatest"], "2099.1.1")
            self.assertGreaterEqual(
                cache["snoozeUntilMs"], before_ms + _DAY_MS,
            )
            self.assertLessEqual(cache["snoozeUntilMs"], after_ms + _DAY_MS)
        finally:
            await self._close(writer)

    async def test_reload_after_snooze_stays_snoozed(self) -> None:
        """The reported bug: the toast must NOT reappear on window reload."""
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-snooze-2")
            await self._wait_for_event(reader, "update_available")
            await self._snooze(writer, "2099.1.1")
            await self._wait_for_snoozed(reader, True)
        finally:
            await self._close(writer)

        # A brand-new connection simulates the reloaded window.
        reader2, writer2 = await self._connect_uds()
        try:
            await self._send_ready(writer2, "tab-snooze-2-reloaded")
            ev = await self._wait_for_event(reader2, "update_available")
            self.assertEqual(ev.get("available"), True)
            self.assertEqual(
                ev.get("snoozed"),
                True,
                "a window reload within the 24h snooze must not "
                "resurface the update toast",
            )
        finally:
            await self._close(writer2)

    async def test_snooze_preserves_extension_cooldown_fields(self) -> None:
        """The extension's fetch-cooldown fields survive the snooze write."""
        self._write_cache({"lastCheckMs": 123456, "lastLatest": "2099.1.0"})
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-snooze-3")
            await self._wait_for_event(reader, "update_available")
            await self._snooze(writer, "2099.1.1")
            await self._wait_for_snoozed(reader, True)
        finally:
            await self._close(writer)
        cache = json.loads(self._cache_path().read_text(encoding="utf-8"))
        self.assertEqual(cache["lastCheckMs"], 123456)
        self.assertEqual(cache["lastLatest"], "2099.1.0")
        self.assertEqual(cache["snoozedLatest"], "2099.1.1")

    async def test_snooze_without_latest_falls_back_to_cache(self) -> None:
        """A version-less snooze uses the cache's last known latest."""
        self._write_cache({"lastCheckMs": 1, "lastLatest": "2099.1.1"})
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-snooze-4")
            await self._wait_for_event(reader, "update_available")
            writer.write(b'{"type": "snoozeUpdate"}\n')
            await writer.drain()
            ev = await self._wait_for_snoozed(reader, True)
            self.assertEqual(ev.get("available"), True)
        finally:
            await self._close(writer)
        cache = json.loads(self._cache_path().read_text(encoding="utf-8"))
        self.assertEqual(cache["snoozedLatest"], "2099.1.1")


class TestSnoozedStateFromExtensionFile(_SnoozeTestBase):
    """A snooze written by the extension host silences the daemon toast."""

    async def test_extension_written_snooze_is_honored(self) -> None:
        """Clicking "Remind me later" on the native popup gates the toast."""
        # UpdateChecker.js's snoozeUpdateNotification wrote this file.
        self._write_cache({
            "lastCheckMs": 1,
            "lastLatest": "2099.1.1",
            "snoozeUntilMs": int(time.time() * 1000) + _DAY_MS,
            "snoozedLatest": "2099.1.1",
        })
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-ext-snooze")
            ev = await self._wait_for_event(reader, "update_available")
            self.assertEqual(ev.get("available"), True)
            self.assertEqual(ev.get("snoozed"), True)
        finally:
            await self._close(writer)

    async def test_expired_snooze_resurfaces_toast(self) -> None:
        """After 24 hours the toast comes back."""
        self._write_cache({
            "lastCheckMs": 1,
            "lastLatest": "2099.1.1",
            "snoozeUntilMs": int(time.time() * 1000) - 1,
            "snoozedLatest": "2099.1.1",
        })
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-expired-snooze")
            ev = await self._wait_for_event(reader, "update_available")
            self.assertEqual(ev.get("available"), True)
            self.assertEqual(ev.get("snoozed"), False)
        finally:
            await self._close(writer)

    async def test_newer_release_breaks_through_snooze(self) -> None:
        """A release newer than the snoozed one must notify anyway."""
        self._write_cache({
            "lastCheckMs": 1,
            "lastLatest": "2099.1.0",
            "snoozeUntilMs": int(time.time() * 1000) + _DAY_MS,
            "snoozedLatest": "2099.1.0",  # older than PyPI's 2099.1.1
        })
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-breakthrough")
            ev = await self._wait_for_event(reader, "update_available")
            self.assertEqual(ev.get("available"), True)
            self.assertEqual(ev.get("snoozed"), False)
        finally:
            await self._close(writer)

    async def test_corrupt_cache_is_ignored_and_overwritten(self) -> None:
        """A corrupt cache file neither crashes nor snoozes; snooze heals it."""
        self._cache_path().write_text("not json{", encoding="utf-8")
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-corrupt")
            ev = await self._wait_for_event(reader, "update_available")
            self.assertEqual(ev.get("snoozed"), False)
            await self._snooze(writer, "2099.1.1")
            await self._wait_for_snoozed(reader, True)
        finally:
            await self._close(writer)
        cache = json.loads(self._cache_path().read_text(encoding="utf-8"))
        self.assertEqual(cache["lastCheckMs"], 0)
        self.assertEqual(cache["lastLatest"], "")
        self.assertEqual(cache["snoozedLatest"], "2099.1.1")


class TestSnoozeHelperBranches(_SnoozeTestBase):
    """Direct coverage for helper branches unreachable over the wire."""

    async def test_is_update_snoozed_field_validation(self) -> None:
        # No cache file at all.
        self.assertFalse(ws._is_update_snoozed("2099.1.1"))
        # JSON that is not an object.
        self._cache_path().write_text("[1, 2]", encoding="utf-8")
        self.assertFalse(ws._is_update_snoozed("2099.1.1"))
        # Non-numeric snoozeUntilMs.
        self._write_cache({"snoozeUntilMs": "soon", "snoozedLatest": "x"})
        self.assertFalse(ws._is_update_snoozed("2099.1.1"))
        # Non-string snoozedLatest still snoozes (compares as equal).
        self._write_cache({
            "snoozeUntilMs": int(time.time() * 1000) + _DAY_MS,
            "snoozedLatest": 7,
        })
        self.assertTrue(ws._is_update_snoozed("2099.1.1"))

    async def test_record_snooze_ignores_malformed_prior_fields(self) -> None:
        self._write_cache({"lastCheckMs": "x", "lastLatest": 5})
        ws._record_update_snooze("")
        cache = json.loads(self._cache_path().read_text(encoding="utf-8"))
        self.assertEqual(cache["lastCheckMs"], 0)
        self.assertEqual(cache["lastLatest"], "")
        self.assertEqual(cache["snoozedLatest"], "")
        self.assertTrue(ws._is_update_snoozed("2099.1.1"))
