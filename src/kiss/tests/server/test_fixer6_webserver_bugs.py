# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Fixer-6 findings (F1, F2, F7, F11).

Covers, over REAL objects (no mocks, patches, or fakes):

* F1: ``WebPrinter._send_locks`` must not be re-populated for an
  endpoint that ``remove_client`` already dropped — the re-insert
  race leaked one ``asyncio.Lock`` per lost race for the daemon's
  lifetime.
* F2: ``RemoteAccessServer._auth_failures`` must not grow without
  bound: expired entries are deleted (not written back as empty
  lists) and stale entries of OTHER source IPs are swept when a new
  failure is recorded.
* F7: a ``ready`` command carrying non-str ``tabId`` values (top
  level or inside ``restoredTabs``) must not abort ready handling —
  an unhashable id used to raise ``TypeError`` while processing the
  restored-tab list and skip the remaining restored-tab resumes.
* F11: ``AgentServer._teardown_tab_resources`` must survive a
  duck-typed printer without ``cleanup_tab`` (same getattr-guarded
  contract as every sibling cleanup path).
* A21-3 (audit 2026-11-07): a ``_schedule_send`` that loses the race
  with the loop's close (``is_running`` seen True, the loop closed
  before ``run_coroutine_threadsafe`` hands the coroutine over) drops
  the send AND closes the unscheduled coroutine, instead of leaking
  it to the collector's "coroutine ... was never awaited" warning.
* A22-1 (audit 2026-11-07): a ``ready`` resumes the tabs bound AFTER
  its wait for the connection's CHATS page, so a tab closed on another
  surface during that wait is not resumed (and reopened) from a stale
  snapshot.

The live-server tests drive a real :class:`RemoteAccessServer` over a
real local WebSocket connection, mirroring the harness of
``test_simplify_web_server_regr.py``.
"""

from __future__ import annotations

import asyncio
import gc
import json
import random
import shutil
import sys
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
import kiss.server.web_server as ws_mod
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.server.conftest import redirect_persistence, restore_persistence
from kiss.tests.server.pause import Pause


class TestFixer6LiveServer(unittest.IsolatedAsyncioTestCase):
    """E2E tests over a real running RemoteAccessServer (local WSS)."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-fixer6-live-")
        self.saved = redirect_persistence(self.tmpdir)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        th._close_thread_db()
        restore_persistence(self.saved)
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_local(
        self,
    ) -> tuple[LocalReader, LocalWriter]:
        reader, writer = await open_local_connection(
            self.server, limit=16 * 1024 * 1024,
        )
        self.addAsyncCleanup(self._close_writer, writer)
        return reader, writer

    async def _close_writer(self, writer: LocalWriter) -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def _send(self, writer: LocalWriter, msg: dict) -> None:
        writer.write(json.dumps(msg).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: LocalReader,
        wanted_type: str,
        max_events: int = 50,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "local connection closed unexpectedly"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == wanted_type:
                return msg
        raise AssertionError(f"no {wanted_type!r} event observed")

    async def _server_side_writer(self) -> Any:
        """Return the server-side connection of the latest local client."""
        writers: list[Any] = []
        for _ in range(200):
            with self.server._printer._ws_lock:
                writers = list(self.server._printer._local_clients)
            if writers:
                return writers[-1]
            await asyncio.sleep(0.01)
        raise AssertionError("server never registered the local client")

    async def test_send_lock_not_recreated_for_removed_endpoint(self) -> None:
        """A send racing endpoint removal must not leak a _send_locks entry.

        Reproduces the real interleaving deterministically:
        ``_send_to_ws_clients`` snapshots the endpoint list under the
        lock and calls ``_schedule_send`` OUTSIDE it, so removal can
        land in between.  Here removal happens first, then the
        straggler ``_schedule_send`` fires from a worker thread (as a
        broadcast from an agent thread would).
        """
        await self._connect_local()
        printer = self.server._printer
        writer = await self._server_side_writer()
        with printer._ws_lock:
            self.assertIn(writer, printer._pending_sends)
        printer.remove_client(writer)
        with printer._ws_lock:
            self.assertNotIn(writer, printer._send_locks)
            self.assertNotIn(writer, printer._pending_sends)
        data = json.dumps({"type": "notice", "text": "straggler"})
        await asyncio.to_thread(printer._schedule_send, writer, data)
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
            await self._connect_local()
            writer = await self._server_side_writer()

            def hammer(w: Any = writer) -> None:
                for _ in range(5):
                    time.sleep(random.uniform(0.0, 0.01))
                    printer._schedule_send(w, data)

            t = threading.Thread(target=hammer)
            t.start()
            await asyncio.sleep(random.uniform(0.0, 0.01))
            printer.remove_client(writer)
            await asyncio.to_thread(t.join)
        await asyncio.sleep(0.3)
        with printer._ws_lock:
            orphans = set(printer._send_locks) - set(printer._pending_sends)
        self.assertFalse(
            orphans, f"orphaned _send_locks entries leaked: {orphans}",
        )

    async def test_ready_with_non_str_tab_ids_still_serviced(self) -> None:
        """Non-str tabId/chatId values must not abort ready handling."""
        reader, writer = await self._connect_local()
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
        await self._send(
            writer, {"type": "ready", "tabId": "t-after", "restoredTabs": []},
        )
        focus = await self._drain_until(reader, "focusInput")
        self.assertEqual(focus.get("tabId"), "t-after")

    async def test_ready_malformed_restored_entry_does_not_abort_loop(
        self,
    ) -> None:
        """Entries after a malformed restoredTabs entry are still processed.

        The session resume of the LATER (valid) restored tab must
        still be dispatched to the backend — a malformed entry before
        it must not abort the restored-tab loop.
        """
        dispatched: list[dict[str, Any]] = []
        orig_run_cmd = self.server._run_cmd

        async def recording_run_cmd(cmd: dict[str, Any]) -> None:
            dispatched.append(dict(cmd))
            await orig_run_cmd(cmd)

        self.server._run_cmd = recording_run_cmd  # type: ignore[method-assign]
        reader, writer = await self._connect_local()
        await self._send(
            writer,
            {
                "type": "ready",
                "tabId": "t-main",
                "restoredTabs": [
                    {"tabId": ["evil"], "chatId": {"also": "bad"}},
                    {"tabId": "rt-later", "chatId": "chat-rt-later"},
                ],
            },
        )
        await self._drain_until(reader, "focusInput")
        resumes: list[dict[str, Any]] = []
        for _ in range(100):
            resumes = [
                c for c in dispatched if c.get("type") == "resumeSession"
            ]
            if resumes:
                break
            await asyncio.sleep(0.05)
        self.assertEqual(
            len(resumes), 1,
            "the resume of the restored tab after the malformed entry "
            "was not dispatched — the restored-tab loop aborted",
        )
        self.assertEqual(resumes[0].get("tabId"), "rt-later")
        self.assertEqual(resumes[0].get("chatId"), "chat-rt-later")

    async def test_tab_closed_during_the_history_wait_is_not_resumed(self) -> None:
        """A22-1 (audit 2026-11-07), the ready handler's part: the resumes a ``ready``
        issues follow the registry as it is AFTER the wait for the connection's
        CHATS page, not the snapshot taken before it.  A tab another surface closed
        during that wait (up to 3 s) was resumed from the stale snapshot, and the
        replay's publication reopened it on every surface.  The page's reply is
        held in its worker thread (:class:`Pause` on the real ``_get_history``)
        while the tab is closed."""
        task_id, chat_id = await asyncio.to_thread(th._add_task, "chat A task")
        await asyncio.to_thread(
            th._append_chat_event, {"type": "text_delta", "text": "A"}, task_id=task_id,
        )
        registry = self.server._agent_server.tab_registry
        registry.update_tab("t-stale", chat_id=chat_id, create=True)
        reader, writer = await self._connect_local()
        pause = Pause(AgentServer._get_history.__code__, lambda _locals: True)
        with pause:
            await self._send(writer, {"type": "getHistory", "offset": 0, "generation": 1})
            self.assertTrue(await asyncio.to_thread(pause.paused.wait, 10))
            await self._send(writer, {"type": "ready", "tabId": "t-stale", "restoredTabs": []})
            # ``focusInput`` is sent right before the registry sync; the
            # ready is then waiting for the held page.
            await self._drain_until(reader, "focusInput")
            await asyncio.sleep(0.3)
            self.server._agent_server._close_tab("t-stale")
            self.assertFalse(registry.has_tab("t-stale"))
        for _ in range(40):
            await asyncio.sleep(0.05)
            self.assertFalse(
                registry.has_tab("t-stale"),
                "the ready resumed the stale snapshot and reopened the closed tab",
            )


class TestAuthFailureBookkeeping(unittest.TestCase):
    """F2: _auth_failures pruning and cross-IP sweeping (real server)."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-fixer6-auth-")
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_expired_entry_removed_not_kept_as_empty_list(self) -> None:
        stale = time.monotonic() - ws_mod._AUTH_FAIL_WINDOW - 10.0
        self.server._auth_failures["9.9.9.9"] = [stale]
        self.assertEqual(self.server._auth_lock_remaining("9.9.9.9"), 0.0)
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
        self.assertGreater(self.server._auth_lock_remaining(ip), 0.0)
        self.assertIn(other_ip, self.server._auth_failures)
        self.assertEqual(self.server._auth_lock_remaining(other_ip), 0.0)


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
        server = AgentServer(printer=cast(Any, _BroadcastOnlyPrinter()))
        server._close_tab("fixer6-tab-x")
        with server._state_lock:
            server._tab_chat_views["fixer6-tab-y"] = "chat-1"
        server._teardown_tab_resources("fixer6-tab-y", None)
        with server._state_lock:
            self.assertNotIn("fixer6-tab-y", server._tab_chat_views)


class TestScheduleSendOnClosingLoop(unittest.TestCase):
    """A send that loses the race with the loop's close leaks no coroutine."""

    def test_coroutine_is_closed_when_the_loop_closes_under_the_handoff(self) -> None:
        """The real loop is stopped and closed at the exact hand-off statement."""
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        printer = ws_mod.WebPrinter()
        printer._loop = loop
        handoff = asyncio.run_coroutine_threadsafe.__code__

        def _close_loop_at_handoff(frame: Any, event: str, arg: Any) -> Any:
            # Held at the entry of ``run_coroutine_threadsafe``: the
            # coroutine exists, ``is_running`` was True a moment ago.
            if event == "call" and frame.f_code is handoff:
                sys.settrace(None)
                loop.call_soon_threadsafe(loop.stop)
                thread.join(10)
                loop.close()
            return None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sys.settrace(_close_loop_at_handoff)
            try:
                printer._schedule_send(cast(Any, object()), json.dumps({"type": "notice"}))
            finally:
                sys.settrace(None)
            gc.collect()
        self.assertTrue(loop.is_closed(), "the hand-off was not held at the closed loop")
        with printer._ws_lock:
            self.assertEqual(printer._pending_sends, {})
        leaked = [str(w.message) for w in caught if "never awaited" in str(w.message)]
        self.assertEqual(leaked, [], "the unscheduled _locked_send coroutine leaked")


if __name__ == "__main__":
    unittest.main()
