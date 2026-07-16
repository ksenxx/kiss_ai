# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt integration tests for ``kiss.server.web_server``.

Bug 1 — ``ready`` fan-out drops the per-connection ``workDir`` stamp.

``RemoteAccessServer._dispatch_client_command`` stamps every command
from a connection with that connection's announced ``setWorkDir``
folder, and ``_cmd_get_config`` documents that the reported
``work_dir`` is taken from the command's ``workDir`` "stamped per
connection by RemoteAccessServer" so the settings panel shows the
folder *this* instance will actually use.  But ``_handle_ready``
fans the webview's ``ready`` message out into synthesized
``getModels`` / ``getInputHistory`` / ``getConfig`` commands that
carry only the ``connId`` — the ``workDir`` stamp on the incoming
``ready`` command is dropped.  The ``configData`` reply triggered by
every page load / reconnect therefore reports the daemon-global
work_dir instead of the instance's pinned folder, violating the
per-window/per-instance work_dir invariant.

Bug 2 — ``_is_auth_locked`` grows ``_auth_failures`` without bound.

``_authenticate_ws`` calls ``_is_auth_locked(ip)`` for every incoming
connection, and ``_is_auth_locked`` unconditionally writes the pruned
failure list back with ``self._auth_failures[ip] = fails`` — creating
a permanent empty-list entry for every source IP that ever connects,
even ones that always authenticate successfully.  Entries whose
failures have all expired are likewise never removed.  The dict grows
monotonically for the lifetime of the daemon.

Both tests spin up the real :class:`RemoteAccessServer` on a free
port and talk to it over real ``wss://`` connections — no mocks, no
LLM calls.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.server.vscode_config as vc
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


class _ServerTestBase(IsolatedAsyncioTestCase):
    """Shared real-server setup/teardown for the bug-hunt tests."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt-ws-")
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

    async def _connect(self, password: str = "") -> ClientConnection:
        """Open one WSS connection and send an auth attempt."""
        ws = await connect(self.url, ssl=self.ctx)
        self._sockets.append(ws)
        await ws.send(json.dumps({"type": "auth", "password": password}))
        return ws

    async def _connect_ok(self) -> ClientConnection:
        """Open + successfully authenticate one WSS connection."""
        ws = await self._connect("")
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(resp["type"], "auth_ok")
        return ws

    async def _drain_until(
        self,
        ws: ClientConnection,
        predicate: Any,
        max_events: int = 100,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Read events from *ws* until *predicate* matches one."""
        for _ in range(max_events):
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError(
            f"predicate never matched within {max_events} events",
        )


class ReadyFanoutWorkDirTest(_ServerTestBase):
    """Bug 1: ready-triggered ``configData`` must carry the pinned work_dir."""

    async def test_ready_config_reply_uses_connection_work_dir(self) -> None:
        """``ready`` after ``setWorkDir`` must report the pinned folder.

        Mirrors exactly what the webapp does on every page load: the
        WS shim replays ``setWorkDir`` right after ``auth_ok`` and
        then ``main.js`` sends ``ready``.  The ``configData`` reply
        the server fans out for ``ready`` must show the connection's
        own pinned folder — not the daemon-global one persisted by a
        different instance.
        """
        dir_mine = Path(self.tmpdir) / "inst_mine"
        dir_other = Path(self.tmpdir) / "inst_other"
        dir_mine.mkdir()
        dir_other.mkdir()
        # Another instance persisted its folder globally.
        vc.save_config({"work_dir": str(dir_other)})

        ws = await self._connect_ok()
        await ws.send(json.dumps(
            {"type": "setWorkDir", "workDir": str(dir_mine)},
        ))
        await ws.send(json.dumps(
            {"type": "ready", "tabId": "tab-1", "restoredTabs": []},
        ))
        cfg_event = await self._drain_until(
            ws, lambda m: m.get("type") == "configData",
        )
        self.assertEqual(
            cfg_event.get("config", {}).get("work_dir"),
            str(dir_mine),
            "configData fanned out for 'ready' must report the "
            "connection's pinned work_dir, not the global fallback",
        )


class AuthFailureRegistryGrowthTest(_ServerTestBase):
    """Bug 2: successful clients must not leak ``_auth_failures`` entries."""

    async def test_successful_auth_leaves_no_failure_entry(self) -> None:
        """Repeated successful logins must not grow ``_auth_failures``.

        Every connection attempt routes through ``_is_auth_locked``,
        which must not permanently register source IPs that have no
        live failures — otherwise the registry grows monotonically
        for the daemon's lifetime (one entry per distinct client IP,
        never reclaimed).
        """
        for _ in range(3):
            ws = await self._connect_ok()
            await ws.close()
        self.assertEqual(
            len(self.server._auth_failures),
            0,
            "auth-failure registry must stay empty for clients that "
            "never failed authentication",
        )

    async def test_failed_auth_still_recorded_and_lockout_works(self) -> None:
        """Real failures are still tracked and still trigger lockout."""
        vc.save_config({"remote_password": "s3cret"})
        # 5 failures (each connection allows 2 attempts; drive 3
        # connections of 2 bad attempts each = 6 recorded failures).
        for _ in range(3):
            ws = await self._connect("wrong")
            # First failure elicits an auth_required retry prompt.
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_required")
            await ws.send(json.dumps({"type": "auth", "password": "wrong"}))
            # Second failure closes the connection (error event first).
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "error")
            await ws.close()
        self.assertTrue(self.server._is_auth_locked("127.0.0.1"))
        # A locked IP is refused outright: the server closes the
        # socket without ever sending auth_ok (the send or the recv
        # observes the close, depending on timing).
        with self.assertRaises(Exception):
            ws = await self._connect("s3cret")
            json.loads(await asyncio.wait_for(ws.recv(), timeout=5))


if __name__ == "__main__":
    unittest.main()
