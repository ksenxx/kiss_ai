# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests pinning web_server behavior before simplification.

Covers the exact code paths refactored by the web_server.py
simplification pass:

* ``WebPrinter`` endpoint add/remove + broadcast fan-out over a real
  Unix-domain socket connection (shared add/remove helper refactor).
* ``_handle_run_update`` connId-stamped ``error`` / ``notice`` events
  (shared stamped-broadcast helper).
* ``stop_async`` cancelling the watchdog and version-check tasks
  (shared cancel helper).
* Tunnel bookkeeping reset via ``_stop_tunnel`` / ``_detach_tunnel``
  (shared reset helper).
* ``_get_local_ips`` filtering, ``_version_tuple`` /
  ``_compare_versions`` semantics.
* ``_process_request`` routing for ``/``, ``/trajectories`` and 404s.

All tests drive real objects — no mocks, patches, or fakes.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from websockets.datastructures import Headers
from websockets.http11 import Request

from kiss.server import agent_state
from kiss.server.web_server import (
    RemoteAccessServer,
    _compare_versions,
    _get_local_ips,
    _version_tuple,
)


class TestPureHelpers(unittest.TestCase):
    """Pin pure helper semantics."""

    def test_get_local_ips_filters_loopback_and_linklocal(self) -> None:
        ips = _get_local_ips()
        self.assertIsInstance(ips, frozenset)
        for addr in ips:
            self.assertFalse(addr.startswith("127."))
            self.assertFalse(addr.startswith("169.254."))
            self.assertFalse(addr.startswith("::ffff:"))

    def test_version_tuple(self) -> None:
        self.assertEqual(_version_tuple("2026.6.1"), (2026, 6, 1))
        self.assertEqual(_version_tuple(" 2026.6 "), (2026, 6))
        self.assertIsNone(_version_tuple("abc"))
        self.assertIsNone(_version_tuple(""))

    def test_compare_versions(self) -> None:
        self.assertEqual(_compare_versions("2026.7.3", "2026.7.2"), 1)
        self.assertEqual(_compare_versions("2026.6", "2026.6.0"), 0)
        self.assertEqual(_compare_versions("2026.6", "2026.6.1"), -1)
        self.assertEqual(_compare_versions("junk", "2026.6"), 0)


class TestTunnelStateReset(unittest.TestCase):
    """_stop_tunnel / _detach_tunnel reset bookkeeping with no proc."""

    def _make_server(self) -> RemoteAccessServer:
        tmpdir = Path(tempfile.mkdtemp(prefix="kiss-simp-tun-"))
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        return RemoteAccessServer(
            url_file=tmpdir / "remote-url.json",
            uds_path=tmpdir / "sorcar.sock",
        )

    def _seed(self, server: RemoteAccessServer) -> None:
        server._tunnel_metrics_port = 12345
        server._tunnel_started_at = time.monotonic()
        server._tunnel_unhealthy_ticks = 3
        server._tunnel_failure_count = 2
        server._tunnel_next_retry = time.monotonic() + 60
        server._tunnel_rate_limited = True
        server._active_url = "https://x.trycloudflare.com"
        server._tunnel_adopted_pid = None

    def _assert_reset(self, server: RemoteAccessServer) -> None:
        self.assertIsNone(server._tunnel_proc)
        self.assertIsNone(server._tunnel_adopted_pid)
        self.assertIsNone(server._tunnel_metrics_port)
        self.assertIsNone(server._tunnel_started_at)
        self.assertEqual(server._tunnel_unhealthy_ticks, 0)
        self.assertEqual(server._tunnel_failure_count, 0)
        self.assertEqual(server._tunnel_next_retry, 0.0)
        self.assertFalse(server._tunnel_rate_limited)
        self.assertIsNone(server._active_url)

    def test_stop_tunnel_resets_all_state(self) -> None:
        server = self._make_server()
        self._seed(server)
        server._stop_tunnel()
        self._assert_reset(server)

    def test_detach_tunnel_resets_all_state(self) -> None:
        server = self._make_server()
        self._seed(server)
        server._detach_tunnel()
        self._assert_reset(server)


import kiss.agents.sorcar.persistence as th  # noqa: E402


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


class TestLiveServerPaths(unittest.IsolatedAsyncioTestCase):
    """E2E tests over a real running RemoteAccessServer (WSS + UDS)."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-simp-live-")
        self.saved = _redirect_persistence(self.tmpdir)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        self.server._install_root = Path(self.tmpdir) / "kiss_ai"
        self.server._update_log_path = Path(self.tmpdir) / "update.log"
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        agent_state.agent_states.clear()
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
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == wanted_type:
                return msg
        raise AssertionError(f"no {wanted_type!r} event observed")

    async def test_run_update_without_install_script_errors(self) -> None:
        """runUpdate with no install.sh broadcasts the extension-parity error."""
        reader, writer = await self._connect_uds()
        await self._send(writer, {"type": "runUpdate"})
        err = await self._drain_until(reader, "error")
        self.assertIn("install.sh not found", str(err.get("text")))

    async def test_run_update_with_install_script_notices_and_runs(self) -> None:
        """runUpdate with a real install.sh emits notice and spawns it."""
        root = self.server._install_root
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "ran.marker"
        (root / "install.sh").write_text(
            f"#!/bin/bash\necho done > {marker}\n",
        )
        reader, writer = await self._connect_uds()
        await self._send(writer, {"type": "runUpdate"})
        notice = await self._drain_until(reader, "notice")
        self.assertIn("update of KISS Sorcar", str(notice.get("text")))
        for _ in range(200):
            if marker.exists():
                break
            await asyncio.sleep(0.05)
        self.assertTrue(marker.exists(), "install.sh was not executed")

    async def test_stop_async_cancels_background_tasks(self) -> None:
        """stop_async cancels the watchdog and version-check tasks."""
        watchdog = self.server._watchdog_task
        version = self.server._version_check_task
        self.assertIsNotNone(watchdog)
        self.assertIsNotNone(version)
        await self.server.stop_async()
        self.assertIsNone(self.server._watchdog_task)
        self.assertIsNone(self.server._version_check_task)
        assert watchdog is not None and version is not None
        self.assertTrue(watchdog.cancelled() or watchdog.done())
        self.assertTrue(version.cancelled() or version.done())

    async def test_process_request_routing(self) -> None:
        """/, /trajectories, /media and unknown paths route correctly."""

        def req(path: str) -> Request:
            return Request(path, Headers({"Host": "localhost"}))

        conn = cast(Any, None)

        resp = await self.server._process_request(conn, req("/"))
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"<html", resp.body.lower())
        resp = await self.server._process_request(conn, req(""))
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(
            await self.server._process_request(conn, req("/ws")),
        )
        for tpath in ("/trajectories", "/trajectories/"):
            resp = await self.server._process_request(conn, req(tpath))
            assert resp is not None
            self.assertEqual(resp.status_code, 200)
        resp = await self.server._process_request(conn, req("/media/main.js"))
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        resp = await self.server._process_request(conn, req("/nope"))
        assert resp is not None
        self.assertEqual(resp.status_code, 404)
        parsed = urlsplit("/media/main.css?v=abc")
        self.assertEqual(parsed.path, "/media/main.css")
        resp = await self.server._process_request(
            conn, req("/media/main.css?v=abc"),
        )
        assert resp is not None
        self.assertEqual(resp.status_code, 200)

    async def test_broadcast_reaches_uds_and_stops_after_removal(self) -> None:
        """Tab-stamped broadcasts fan out to UDS writers until removed."""
        reader, writer = await self._connect_uds()
        writers: list[Any] = []
        for _ in range(100):
            with self.server._printer._ws_lock:
                writers = list(self.server._printer._uds_writers)
            if writers:
                break
            await asyncio.sleep(0.02)
        self.assertEqual(len(writers), 1)
        self.server._printer.broadcast(
            {"type": "notice", "text": "hello", "tabId": "t1"},
        )
        msg = await self._drain_until(reader, "notice")
        self.assertEqual(msg.get("text"), "hello")
        self.server._printer.remove_uds_writer(writers[0])
        self.server._printer.broadcast(
            {"type": "notice", "text": "gone", "tabId": "t1"},
        )
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(reader.readline(), timeout=0.4)


if __name__ == "__main__":
    unittest.main()
