# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: a failing reject write must not kill the client connection.

``_reject_hunk_in_file``'s final ``open(write_to, "w")`` (and
``_restore_base_bytes``'s ``write_bytes``) can raise ``OSError`` — the
canonical real-world trigger is the agent deleting a tracked file and
creating a DIRECTORY at the same path: the merge view lists the file
as deleted (with a ``.deleted`` placeholder as ``current``), and the
user rejecting that deletion makes the server try to write the
restored content to a path that is now a directory
(``IsADirectoryError``).

Before the fix the exception propagated out of
``_apply_web_merge_action`` → ``_handle_web_merge_action`` →
``_dispatch_client_command`` → the ``_ws_handler`` message loop's
``except Exception``, which EXITED the loop: the whole authenticated
WebSocket connection was torn down over one failed hunk rejection, the
``finally`` block armed deferred ``closeTab`` timers for every tab the
connection had touched, and after the grace window the in-flight
review was force-finished (all-done = accept remaining) and the tab
closed — silently accepting changes the user was actively rejecting.

The same unguarded dispatch also let ANY malformed client field that
raises (e.g. an unhashable ``tabId``) kill the connection.

These tests use a real ``RemoteAccessServer`` with real wss://
connections and real files on disk — no mocks.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    """Point the sorcar persistence layer at a per-test directory."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    """Undo :func:`_redirect_persistence`."""
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _find_free_port() -> int:
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _deleted_file_entry(work: Path, name: str) -> dict[str, Any]:
    """Build a merge entry for a tracked file the agent deleted.

    Mirrors ``_prepare_merge_view``: ``current`` is an empty
    ``.deleted`` placeholder, ``target`` is the real workspace path,
    ``base`` holds the pre-task content, and the single hunk is the
    whole-file deletion.  The caller decides what (if anything) sits
    at the target path on disk.
    """
    merge_tmp = work / "merge-temp"
    placeholder = merge_tmp / ".deleted" / name
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text("")
    base = merge_tmp / name
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("alpha\nbeta\n")
    return {
        "name": name,
        "current": str(placeholder),
        "base": str(base),
        "target": str(work / name),
        "hunks": [{"bs": 0, "bc": 2, "cs": 0, "cc": 0}],
    }


def _modified_file_entry(work: Path, name: str) -> dict[str, Any]:
    """Build a merge entry for a normally-modified text file."""
    current = work / name
    base = work / f"{name}.base"
    lines = "".join(f"line{i}\n" for i in range(8))
    current.write_text(lines.replace("line2\n", "edit2\n"))
    base.write_text(lines)
    return {
        "name": name,
        "current": str(current),
        "base": str(base),
        "target": str(current),
        "hunks": [{"bs": 2, "bc": 1, "cs": 2, "cc": 1}],
    }


class TestRejectWriteFailure(IsolatedAsyncioTestCase):
    """A failed reject write must degrade gracefully, never drop the client."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt6-rejfail-")
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)

        self.port = _find_free_port()
        self.url = f"wss://127.0.0.1:{self.port}/ws"
        self.ctx = _no_verify_ssl()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
        )
        await self.server.start_async()
        self._sockets: list[ClientConnection] = []

    async def asyncTearDown(self) -> None:
        for ws in self._sockets:
            try:
                await ws.close()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_ok(self) -> ClientConnection:
        """Open + successfully authenticate one WSS connection."""
        ws = await connect(self.url, ssl=self.ctx)
        self._sockets.append(ws)
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(resp["type"], "auth_ok")
        return ws

    def _open_review(self, tab_id: str, files: list[dict[str, Any]]) -> None:
        """Register an in-flight review through the real broadcast path."""
        merge_data = {"work_dir": self.tmpdir, "files": files}
        hunk_count = sum(len(f["hunks"]) for f in files)
        self.server._printer.broadcast({
            "type": "merge_data",
            "tabId": tab_id,
            "data": merge_data,
            "hunk_count": hunk_count,
        })
        with self.server._merge_states_lock:
            self.assertIn(tab_id, self.server._merge_states)

    async def _collect_events(
        self, ws: ClientConnection, timeout: float,
    ) -> list[dict[str, Any]]:
        """Drain events from *ws* for *timeout* seconds."""
        got: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                break
            got.append(json.loads(raw))
        return got

    async def _assert_connection_alive(self, ws: ClientConnection) -> None:
        """Probe liveness: an activeTasksQuery must get a direct reply."""
        await ws.send(json.dumps({"type": "activeTasksQuery"}))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            ev = json.loads(raw)
            if ev.get("type") == "activeTasksResponse":
                return
        self.fail("no activeTasksResponse received")  # pragma: no cover

    async def test_reject_on_dir_target_keeps_connection_and_review(self) -> None:
        """Rejecting a deletion whose target is now a directory must not
        kill the connection, must surface an error event, and must leave
        the hunk unresolved (the write did not happen)."""
        tab_id = "tab-rej-dir"
        work = Path(self.tmpdir) / "work-rej-dir"
        work.mkdir()
        entry = _deleted_file_entry(work, "cfg")
        # The agent replaced the deleted file with a directory of the
        # same name (e.g. cfg → cfg/ with files inside).
        (work / "cfg").mkdir()
        (work / "cfg" / "inner.txt").write_text("x\n")
        self._open_review(tab_id, [entry])

        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "mergeAction", "action": "reject", "tabId": tab_id,
        }))
        events = await self._collect_events(ws, timeout=2.0)

        # The connection must survive the failed write.
        try:
            await self._assert_connection_alive(ws)
        except Exception as exc:  # noqa: BLE001 — turn closure into a failure
            self.fail(
                "BUG: a failed reject write killed the whole WebSocket "
                f"connection ({type(exc).__name__}: {exc})",
            )

        # The user must be told the rejection failed.
        error_events = [e for e in events if e.get("type") == "error"]
        self.assertTrue(
            error_events,
            f"no error event broadcast for the failed reject: {events}",
        )

        # The hunk must stay unresolved — nothing was written to disk.
        with self.server._merge_states_lock:
            state = self.server._merge_states.get(tab_id)
        self.assertIsNotNone(state, "merge state must survive the failure")
        assert state is not None
        self.assertEqual(state.remaining, 1)
        self.assertEqual(state.resolutions(), [])
        # The directory the agent created is untouched.
        self.assertTrue((work / "cfg").is_dir())
        self.assertEqual((work / "cfg" / "inner.txt").read_text(), "x\n")

    async def test_reject_all_partial_failure_keeps_good_file_and_review(
        self,
    ) -> None:
        """reject-all with one restorable and one unrestorable file must
        restore the good file, report the bad one, and keep its hunk
        unresolved instead of dying mid-way / zombifying the review."""
        tab_id = "tab-rejall-dir"
        work = Path(self.tmpdir) / "work-rejall-dir"
        work.mkdir()
        good = _modified_file_entry(work, "good.txt")
        bad = _deleted_file_entry(work, "cfg")
        (work / "cfg").mkdir()
        self._open_review(tab_id, [good, bad])

        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "mergeAction", "action": "reject-all", "tabId": tab_id,
        }))
        events = await self._collect_events(ws, timeout=2.0)

        try:
            await self._assert_connection_alive(ws)
        except Exception as exc:  # noqa: BLE001
            self.fail(
                "BUG: a failed reject-all write killed the whole "
                f"WebSocket connection ({type(exc).__name__}: {exc})",
            )

        # The good file must be restored to its base content.
        self.assertEqual(
            (work / "good.txt").read_text(),
            "".join(f"line{i}\n" for i in range(8)),
            "restorable file must be reverted even when a sibling fails",
        )
        error_events = [e for e in events if e.get("type") == "error"]
        self.assertTrue(
            error_events,
            f"no error event broadcast for the failed reject-all: {events}",
        )
        # The failed file's hunk stays unresolved; the review survives
        # (NOT zombified with remaining == 0 and no all-done dispatched).
        with self.server._merge_states_lock:
            state = self.server._merge_states.get(tab_id)
        self.assertIsNotNone(state, "merge state must survive the failure")
        assert state is not None
        self.assertEqual(state.remaining, 1)
        statuses = {
            (r["fi"], r["hi"]): r["status"] for r in state.resolutions()
        }
        self.assertEqual(statuses, {(0, 0): "rejected"})

    async def test_unhashable_tab_id_does_not_kill_connection(self) -> None:
        """A malformed client field that raises (unhashable tabId) must be
        contained per-message, not tear down the authenticated session."""
        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "ready", "tabId": {"x": 1}, "restoredTabs": [],
        }))
        await asyncio.sleep(0.3)
        try:
            await self._assert_connection_alive(ws)
        except Exception as exc:  # noqa: BLE001
            self.fail(
                "BUG: a malformed tabId killed the whole WebSocket "
                f"connection ({type(exc).__name__}: {exc})",
            )


if __name__ == "__main__":
    unittest.main()
