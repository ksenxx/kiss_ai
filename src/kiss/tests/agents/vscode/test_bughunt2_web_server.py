# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt round 2 integration tests for ``kiss.server.web_server``.

Bug — WSS connections leak ``cliTaskStart`` registrations forever.

``RemoteAccessServer._dispatch_client_command`` is the single shared
dispatch body for BOTH transports (its docstring: "so the two
transports cannot drift in behaviour"), and it accepts the
``cliTaskStart`` / ``cliTaskEnd`` envelopes from either one.  A task
id announced via ``cliTaskStart`` is recorded in
``_cli_running_tasks`` and stamped into the connection's per-conn
``cli_tasks`` set precisely so the connection handler can clean it up
if the peer disconnects without a matching ``cliTaskEnd`` (Ctrl+C,
crash, abrupt termination).

But only ``_uds_handler``'s ``finally`` block performs that stale-task
sweep — ``_ws_handler``'s ``finally`` does not.  A WSS client that
sends ``cliTaskStart`` and then drops (browser closed, network cut,
crashed helper tunnelling over WSS) therefore leaves the task id in
``_cli_running_tasks`` for the remainder of the daemon's lifetime:

* ``VSCodeServer._replay_session`` (via the ``_is_cli_task_running``
  hook) keeps subscribing fresh webview tabs to the dead task and
  shows the blinking-green-circle "running" indicator forever;
* ``_snapshot_cli_running_task_ids`` keeps unioning the id into the
  History panel's ``is_running`` flags (permanent pulsing green dot);
* the entry is never reclaimed — an unbounded stale-state leak.

The test spins up the real :class:`RemoteAccessServer` on a free port
and talks to it over a real ``wss://`` connection — no mocks.
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


class WssCliTaskLeakTest(IsolatedAsyncioTestCase):
    """A dropped WSS peer must not leak its ``cliTaskStart`` entries."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt2-ws-")
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

    async def _wait_running(self, task_id: str, expected: bool) -> bool:
        """Poll the server's CLI-running lookup until it matches."""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.server._is_cli_task_running(task_id) == expected:
                return True
            await asyncio.sleep(0.05)
        return self.server._is_cli_task_running(task_id) == expected

    async def test_ws_disconnect_sweeps_stale_cli_tasks(self) -> None:
        """A WSS peer dropping without ``cliTaskEnd`` must be swept.

        Mirrors the guarantee ``_uds_handler`` already provides for
        the local CLI: after the connection that announced
        ``cliTaskStart`` goes away, the daemon must not keep
        reporting the task as running (permanent blinking-green
        "running" indicator + unbounded ``_cli_running_tasks``
        growth).
        """
        task_id = "task-bughunt2-wss-leak"
        ws = await self._connect_ok()
        await ws.send(json.dumps({"type": "cliTaskStart", "taskId": task_id}))
        self.assertTrue(
            await self._wait_running(task_id, True),
            "cliTaskStart over WSS must register the running task",
        )
        # Abruptly drop the announcing connection without cliTaskEnd.
        await ws.close()
        self.assertTrue(
            await self._wait_running(task_id, False),
            "WSS disconnect must sweep the connection's stale "
            "cliTaskStart registrations exactly like a UDS "
            "disconnect does",
        )

    async def test_ws_cli_task_end_still_clears(self) -> None:
        """The explicit ``cliTaskEnd`` path keeps working over WSS."""
        task_id = "task-bughunt2-wss-end"
        ws = await self._connect_ok()
        await ws.send(json.dumps({"type": "cliTaskStart", "taskId": task_id}))
        self.assertTrue(await self._wait_running(task_id, True))
        await ws.send(json.dumps({"type": "cliTaskEnd", "taskId": task_id}))
        self.assertTrue(
            await self._wait_running(task_id, False),
            "cliTaskEnd over WSS must clear the running task",
        )


if __name__ == "__main__":
    unittest.main()
