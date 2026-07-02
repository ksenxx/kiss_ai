# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests pinning web_server behavior before simplification.

Covers the exact code paths refactored by the web_server.py
simplification pass:

* ``_WebMergeState`` navigation / resolution logic (``current()`` guard
  merge).
* ``WebPrinter`` endpoint add/remove + broadcast fan-out over a real
  Unix-domain socket connection (shared add/remove helper refactor).
* ``cliTaskStart`` / ``cliTaskEnd`` dispatch through a real UDS client
  (merged dispatch branch).
* ``_handle_run_update`` connId-stamped ``error`` / ``notice`` events
  (shared stamped-broadcast helper).
* Merge-action completion popping both ``_merge_states`` and
  ``_merge_action_locks`` (shared pop helper).
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

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _compare_versions,
    _get_local_ips,
    _version_tuple,
    _WebMergeState,
)


def _merge_data_two_files(tmpdir: Path) -> dict[str, Any]:
    """Build a real two-file, three-hunk merge_data payload on disk."""
    base_a = tmpdir / "a.base"
    cur_a = tmpdir / "a.txt"
    base_a.write_text("one\ntwo\nthree\nfour\n")
    cur_a.write_text("ONE\ntwo\nTHREE\nfour\n")
    base_b = tmpdir / "b.base"
    cur_b = tmpdir / "b.txt"
    base_b.write_text("alpha\n")
    cur_b.write_text("beta\n")
    return {
        "work_dir": str(tmpdir),
        "files": [
            {
                "name": "a.txt",
                "base": str(base_a),
                "current": str(cur_a),
                "hunks": [
                    {"bs": 0, "bc": 1, "cs": 0, "cc": 1},
                    {"bs": 2, "bc": 1, "cs": 2, "cc": 1},
                ],
            },
            {
                "name": "b.txt",
                "base": str(base_b),
                "current": str(cur_b),
                "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 1}],
            },
        ],
    }


class TestWebMergeState(unittest.TestCase):
    """Pin _WebMergeState navigation / resolution semantics."""

    def test_empty_merge_data_current_is_none(self) -> None:
        state = _WebMergeState({"files": []})
        self.assertEqual(state.total_hunks, 0)
        self.assertEqual(state.remaining, 0)
        self.assertIsNone(state.current())
        # Navigation on an empty review must be a no-op, not a crash.
        state.advance()
        state.go_prev()
        self.assertIsNone(state.current())
        self.assertEqual(state.resolutions(), [])
        self.assertEqual(state.all_unresolved(), [])

    def test_navigation_and_resolution_flow(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="kiss-simp-ms-"))
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        state = _WebMergeState(_merge_data_two_files(tmpdir))
        self.assertEqual(state.total_hunks, 3)
        self.assertEqual(state.remaining, 3)
        self.assertEqual(state.current(), (0, 0))
        state.mark_resolved(0, 0, "accepted")
        state.advance()
        self.assertEqual(state.current(), (0, 1))
        self.assertEqual(state.remaining, 2)
        state.go_prev()
        # (0, 0) is resolved so prev wraps to the previous UNRESOLVED.
        self.assertEqual(state.current(), (1, 0))
        self.assertEqual(state.unresolved_in_file(0), [1])
        self.assertEqual(state.all_unresolved(), [(0, 1), (1, 0)])
        self.assertTrue(state.is_resolved(0, 0))
        self.assertFalse(state.is_resolved(0, 1))
        # Resolve everything: current() must become None, not point at
        # the last (now-resolved) hunk.
        state.mark_resolved(0, 1, "rejected")
        state.mark_resolved(1, 0, "accepted")
        self.assertEqual(state.remaining, 0)
        self.assertIsNone(state.current())
        resolved = {(r["fi"], r["hi"]): r["status"] for r in state.resolutions()}
        self.assertEqual(
            resolved,
            {(0, 0): "accepted", (0, 1): "rejected", (1, 0): "accepted"},
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


class TestMergeActionCompletion(unittest.TestCase):
    """Completing a review must pop state AND the per-tab action lock."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-simp-mac-"))
        self.server = RemoteAccessServer(
            url_file=self.tmpdir / "remote-url.json",
            uds_path=self.tmpdir / "sorcar.sock",
        )

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_accept_and_reject_flow_completes_and_cleans_up(self) -> None:
        tab_id = "simp-merge-tab"
        self.server._register_merge_state(
            tab_id, _merge_data_two_files(self.tmpdir),
        )

        async def drive() -> None:
            self.server._loop = asyncio.get_running_loop()
            self.server._printer._loop = self.server._loop
            for action in ("accept", "reject", "accept"):
                await self.server._handle_web_merge_action({
                    "type": "mergeAction",
                    "action": action,
                    "tabId": tab_id,
                })

        asyncio.run(drive())
        self.assertNotIn(tab_id, self.server._merge_states)
        self.assertNotIn(tab_id, self.server._merge_action_locks)
        # The rejected hunk (file a, hunk 1) restored base content on
        # disk while the accepted hunk (0) kept the agent's content.
        self.assertEqual(
            (self.tmpdir / "a.txt").read_text(),
            "ONE\ntwo\nthree\nfour\n",
        )
        # File b's single hunk was accepted — agent content kept.
        self.assertEqual((self.tmpdir / "b.txt").read_text(), "beta\n")

    def test_unknown_tab_action_is_noop(self) -> None:
        async def drive() -> None:
            self.server._loop = asyncio.get_running_loop()
            self.server._printer._loop = self.server._loop
            await self.server._handle_web_merge_action({
                "type": "mergeAction", "action": "next",
                "tabId": "simp-ghost",
            })

        asyncio.run(drive())
        self.assertNotIn("simp-ghost", self.server._merge_action_locks)


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
        _RunningAgentState.running_agent_states.clear()
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
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == wanted_type:
                return msg
        raise AssertionError(f"no {wanted_type!r} event observed")

    async def test_cli_task_start_end_dispatch(self) -> None:
        """cliTaskStart/cliTaskEnd toggle the CLI-running registry."""
        reader, writer = await self._connect_uds()
        await self._send(
            writer, {"type": "cliTaskStart", "taskId": "task-42"},
        )
        for _ in range(100):
            if self.server._is_cli_task_running("task-42"):
                break
            await asyncio.sleep(0.02)
        self.assertTrue(self.server._is_cli_task_running("task-42"))
        self.assertEqual(
            self.server._snapshot_cli_running_task_ids(), {"task-42"},
        )
        # Bad task ids are ignored without dropping the connection.
        await self._send(writer, {"type": "cliTaskStart", "taskId": 7})
        await self._send(writer, {"type": "cliTaskEnd", "taskId": ""})
        await self._send(
            writer, {"type": "cliTaskEnd", "taskId": "task-42"},
        )
        for _ in range(100):
            if not self.server._is_cli_task_running("task-42"):
                break
            await asyncio.sleep(0.02)
        self.assertFalse(self.server._is_cli_task_running("task-42"))
        # Connection still alive: a ready round-trips a focusInput.
        await self._send(
            writer,
            {"type": "ready", "tabId": "cli-tab", "restoredTabs": []},
        )
        focus = await self._drain_until(reader, "focusInput")
        self.assertEqual(focus.get("tabId"), "cli-tab")

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

        # ``_process_request`` never touches the connection for these
        # routes; pass a typed-away None instead of a real socket.
        conn = cast(Any, None)

        resp = await self.server._process_request(conn, req("/"))
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"<html", resp.body.lower())
        resp = await self.server._process_request(conn, req(""))
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        # /ws lets the WebSocket handshake proceed.
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
        # Query strings are stripped before routing.
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
        # Identify the server-side writer for this connection.
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
        # After removal, broadcasts no longer reach the peer.
        self.server._printer.remove_uds_writer(writers[0])
        self.server._printer.broadcast(
            {"type": "notice", "text": "gone", "tabId": "t1"},
        )
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(reader.readline(), timeout=0.4)


if __name__ == "__main__":
    unittest.main()
