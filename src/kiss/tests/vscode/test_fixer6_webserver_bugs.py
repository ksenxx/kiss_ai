# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Fixer-6 findings (F1, F2, F7, F11).

Covers, over REAL objects (no mocks, patches, or fakes):

* F1: ``WebPrinter._send_locks`` must not be re-populated for an
  endpoint that ``_remove_endpoint`` already dropped — the re-insert
  race leaked one ``asyncio.Lock`` per lost race for the daemon's
  lifetime.
* F2: ``RemoteAccessServer._auth_failures`` must not grow without
  bound: expired entries are deleted (not written back as empty
  lists) and stale entries of OTHER source IPs are swept when a new
  failure is recorded.
* F7: a ``ready`` command carrying non-str ``tabId`` values (top
  level or inside ``restoredTabs``) must not abort ready handling —
  an unhashable id used to raise ``TypeError`` inside
  ``_cancel_pending_tab_close`` and skip the remaining restored-tab
  resumes and merge replays.
* F11: ``VSCodeServer._teardown_tab_resources`` must survive a
  duck-typed printer without ``cleanup_tab`` (same getattr-guarded
  contract as every sibling cleanup path).

The live-server tests drive a real :class:`RemoteAccessServer` over a
real Unix-domain socket connection, mirroring the harness of
``test_simplify_web_server_regr.py``.
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
import kiss.server.web_server as ws_mod
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer
from kiss.server.web_server import RemoteAccessServer


def _redirect_persistence(tmpdir: str) -> tuple[Any, Any, Any]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_persistence(saved: tuple[Any, Any, Any]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


class TestFixer6LiveServer(unittest.IsolatedAsyncioTestCase):
    """E2E tests over a real running RemoteAccessServer (UDS)."""

    async def asyncSetUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-fixer6-live-")
        self.saved = _redirect_persistence(self.tmpdir)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_uds(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        self.addAsyncCleanup(self._close_writer, writer)
        return reader, writer

    async def _close_writer(self, writer: asyncio.StreamWriter) -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def _send(self, writer: asyncio.StreamWriter, msg: dict) -> None:
        writer.write(json.dumps(msg).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        wanted_type: str,
        max_events: int = 50,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == wanted_type:
                return msg
        raise AssertionError(f"no {wanted_type!r} event observed")

    async def _server_side_writer(self) -> Any:
        """Return the server-side UDS writer of the latest connection."""
        writers: list[Any] = []
        for _ in range(200):
            with self.server._printer._ws_lock:
                writers = list(self.server._printer._uds_writers)
            if writers:
                return writers[-1]
            await asyncio.sleep(0.01)
        raise AssertionError("server never registered the UDS writer")

    # ------------------------------------------------------------------
    # F1: _send_locks re-insert leak
    # ------------------------------------------------------------------

    async def test_send_lock_not_recreated_for_removed_endpoint(self) -> None:
        """A send racing endpoint removal must not leak a _send_locks entry.

        Reproduces the real interleaving deterministically:
        ``_send_to_ws_clients`` snapshots the endpoint list under the
        lock and calls ``_schedule_send`` OUTSIDE it, so removal can
        land in between.  Here removal happens first, then the
        straggler ``_schedule_send`` fires from a worker thread (as a
        broadcast from an agent thread would).
        """
        await self._connect_uds()
        printer = self.server._printer
        writer = await self._server_side_writer()
        with printer._ws_lock:
            self.assertIn(writer, printer._pending_sends)
        printer.remove_uds_writer(writer)
        with printer._ws_lock:
            self.assertNotIn(writer, printer._send_locks)
            self.assertNotIn(writer, printer._pending_sends)
        # The straggler send that lost the race with removal.
        data = json.dumps({"type": "notice", "text": "straggler"})
        await asyncio.to_thread(printer._schedule_send, writer, data)
        # Give the loop time to run (or cancel) the send coroutine.
        await asyncio.sleep(0.3)
        with printer._ws_lock:
            self.assertNotIn(
                writer, printer._send_locks,
                "_send_locks entry re-created for a removed endpoint",
            )
            self.assertNotIn(writer, printer._pending_sends)

    async def test_send_lock_no_leak_under_connect_disconnect_stress(
        self,
    ) -> None:
        """Concurrent sends + removals never leave orphaned send locks."""
        printer = self.server._printer
        data = json.dumps({"type": "notice", "text": "x", "tabId": "t"})
        for _ in range(10):
            await self._connect_uds()
            writer = await self._server_side_writer()

            def hammer(w: Any = writer) -> None:
                for _ in range(5):
                    time.sleep(random.uniform(0.0, 0.01))
                    printer._schedule_send(w, data)

            t = threading.Thread(target=hammer)
            t.start()
            await asyncio.sleep(random.uniform(0.0, 0.01))
            printer.remove_uds_writer(writer)
            await asyncio.to_thread(t.join)
        await asyncio.sleep(0.3)
        with printer._ws_lock:
            orphans = set(printer._send_locks) - set(printer._pending_sends)
        self.assertFalse(
            orphans, f"orphaned _send_locks entries leaked: {orphans}",
        )

    # ------------------------------------------------------------------
    # F7: malformed ready tab ids
    # ------------------------------------------------------------------

    async def test_ready_with_non_str_tab_ids_still_serviced(self) -> None:
        """Non-str tabId/chatId values must not abort ready handling."""
        reader, writer = await self._connect_uds()
        # An unhashable top-level tabId used to raise TypeError inside
        # _cancel_pending_tab_close BEFORE focusInput was sent.
        await self._send(
            writer,
            {
                "type": "ready",
                "tabId": {"bad": 1},
                "restoredTabs": [
                    {"tabId": ["evil"], "chatId": 123},
                    {"tabId": "rt-good", "chatId": ""},
                ],
            },
        )
        focus = await self._drain_until(reader, "focusInput")
        self.assertEqual(focus.get("tabId"), "")
        # Connection is alive and a well-formed ready still round-trips.
        await self._send(
            writer, {"type": "ready", "tabId": "t-after", "restoredTabs": []},
        )
        focus = await self._drain_until(reader, "focusInput")
        self.assertEqual(focus.get("tabId"), "t-after")

    async def test_ready_malformed_restored_entry_does_not_abort_loop(
        self,
    ) -> None:
        """Entries after a malformed restoredTabs entry are still processed."""
        reader, writer = await self._connect_uds()
        # Arm a deferred close for the tab that appears AFTER the
        # malformed entry; _handle_ready must cancel it.
        self.server._schedule_tab_close("rt-later")
        with self.server._pending_tab_closes_lock:
            self.assertIn("rt-later", self.server._pending_tab_closes)
        await self._send(
            writer,
            {
                "type": "ready",
                "tabId": "t-main",
                "restoredTabs": [
                    {"tabId": ["evil"], "chatId": {"also": "bad"}},
                    {"tabId": "rt-later", "chatId": ""},
                ],
            },
        )
        await self._drain_until(reader, "focusInput")
        for _ in range(200):
            with self.server._pending_tab_closes_lock:
                if "rt-later" not in self.server._pending_tab_closes:
                    break
            await asyncio.sleep(0.01)
        with self.server._pending_tab_closes_lock:
            self.assertNotIn(
                "rt-later", self.server._pending_tab_closes,
                "restored tab after malformed entry was not re-claimed",
            )


class TestAuthFailureBookkeeping(unittest.TestCase):
    """F2: _auth_failures pruning and cross-IP sweeping (real server)."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-fixer6-auth-")
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_expired_entry_removed_not_kept_as_empty_list(self) -> None:
        stale = time.monotonic() - ws_mod._AUTH_FAIL_WINDOW - 10.0
        self.server._auth_failures["9.9.9.9"] = [stale]
        self.assertFalse(self.server._is_auth_locked("9.9.9.9"))
        self.assertNotIn(
            "9.9.9.9", self.server._auth_failures,
            "expired IP entry written back as an empty list",
        )

    def test_record_failure_sweeps_stale_entries_of_other_ips(self) -> None:
        stale = time.monotonic() - ws_mod._AUTH_FAIL_WINDOW - 10.0
        for i in range(50):
            self.server._auth_failures[f"10.0.0.{i}"] = [stale]
        fresh_ip = "5.5.5.5"
        self.server._record_auth_failure(fresh_ip)
        self.assertEqual(
            set(self.server._auth_failures), {fresh_ip},
            "stale entries of other IPs were not swept",
        )

    def test_lockout_still_engages_and_fresh_entries_survive_sweep(
        self,
    ) -> None:
        other_ip = "8.8.8.8"
        self.server._record_auth_failure(other_ip)
        ip = "7.7.7.7"
        for _ in range(ws_mod._AUTH_FAIL_MAX):
            self.server._record_auth_failure(ip)
        self.assertTrue(self.server._is_auth_locked(ip))
        # Fresh (in-window) entries of other IPs survive the sweep.
        self.assertIn(other_ip, self.server._auth_failures)
        # Below the threshold, the fresh IP is not locked.
        self.assertFalse(self.server._is_auth_locked(other_ip))


class _BroadcastOnlyPrinter:
    """Duck-typed printer implementing only the broadcast subset.

    Deliberately lacks ``cleanup_tab`` to exercise the getattr-guarded
    printer contract documented on ``_printer_cleanup_tab``.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* instead of writing it anywhere."""
        self.events.append(event)


class TestTeardownWithDuckTypedPrinter(unittest.TestCase):
    """F11: tab teardown must tolerate printers without cleanup_tab."""

    def test_close_tab_survives_printer_without_cleanup_tab(self) -> None:
        server = VSCodeServer(printer=cast(Any, _BroadcastOnlyPrinter()))
        # Unknown tab id: _close_tab pops nothing and reaches
        # _teardown_tab_resources(tab_id, None), which used to call
        # printer.cleanup_tab unguarded -> AttributeError.
        server._close_tab("fixer6-tab-x")
        # Direct teardown of a tab with viewer state behaves the same.
        with server._state_lock:
            server._tab_chat_views["fixer6-tab-y"] = "chat-1"
        server._teardown_tab_resources("fixer6-tab-y", None)
        with server._state_lock:
            self.assertNotIn("fixer6-tab-y", server._tab_chat_views)


if __name__ == "__main__":
    unittest.main()
