# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the daemon's local channel is the same WSS listener on every platform.

The daemon used to bind a second, Unix-domain-socket listener for
same-machine clients and had to skip it cleanly on Windows (CPython
there has no ``socket.AF_UNIX``).  That platform fork is gone: local
clients now connect to the one WSS listener and prove they are local
with the per-start token from the endpoint file.  This test runs the
real server and asserts the platform-independent contract: start logs
nothing at WARNING or above, a remote WSS client and a token-bearing
local client are both served, the endpoint file exists while the daemon
runs and is gone once it stops.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import signal
import socket
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from websockets.asyncio.client import connect

import kiss.core.vscode_config as vc
from kiss.server import agent_state
from kiss.server.web_server import (
    _SHUTDOWN_SIGNALS,
    RemoteAccessServer,
    _generate_self_signed_cert,
)
from kiss.tests.local_ws import open_local_connection


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestLocalChannelOnEveryPlatform(IsolatedAsyncioTestCase):
    """One WSS listener serves remote and local clients; no platform fallback."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-local-platform-"))
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        certfile, keyfile = self.tmpdir / "cert.pem", self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _free_port()
        self.endpoint_file = self.tmpdir / "sorcar-local.json"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            local_endpoint_file=self.endpoint_file,
            work_dir=str(self.tmpdir),
        )
        self.stopped = False

    async def asyncTearDown(self) -> None:
        if not self.stopped:
            await self.server.stop_async()
        agent_state.agent_states.clear()
        vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _start_capturing_logs(self) -> list[logging.LogRecord]:
        """Start the server and return every ``kiss.server.web_server`` record."""
        records: list[logging.LogRecord] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Collect(level=logging.DEBUG)
        log = logging.getLogger("kiss.server.web_server")
        saved_level = log.level
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        try:
            await self.server.start_async()
        finally:
            log.removeHandler(handler)
            log.setLevel(saved_level)
        return records

    async def _wss_auth_round_trip(self) -> None:
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            while True:
                msg: dict[str, Any] = json.loads(
                    await asyncio.wait_for(ws.recv(), 30),
                )
                if msg.get("type") == "auth_ok":
                    return

    async def test_wss_serves_remote_and_local_clients(self) -> None:
        records = await self._start_capturing_logs()
        self.assertEqual(
            [r.getMessage() for r in records if r.levelno >= logging.WARNING],
            [],
            "a clean start must not log warnings about the local channel",
        )
        self.assertIsNotNone(self.server._ws_server)
        await self._wss_auth_round_trip()

        # Local channel: the endpoint file names this listener and a
        # client presenting its token is served over the same protocol.
        self.assertTrue(self.endpoint_file.exists())
        endpoint = json.loads(self.endpoint_file.read_text())
        self.assertEqual(endpoint["url"], f"wss://127.0.0.1:{self.port}/ws")
        self.assertEqual(endpoint["token"], self.server.local_token)
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(
                json.dumps({"type": "getDefaultModel"}).encode() + b"\n",
            )
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), 30)
            self.assertTrue(line, "local client got no reply")
        finally:
            writer.close()
            await writer.wait_closed()

        await self.server.stop_async()
        self.stopped = True
        self.assertFalse(
            self.endpoint_file.exists(),
            "stop_async must remove the endpoint file it wrote",
        )


class TestShutdownSignalsFollowPlatform(TestCase):
    """SIGTERM is always handled; SIGHUP only where the platform defines it."""

    def test_signal_table_and_handler(self) -> None:
        self.assertIn(signal.SIGTERM, _SHUTDOWN_SIGNALS)
        sighup = getattr(signal, "SIGHUP", None)
        if sighup is None:
            self.assertEqual(_SHUTDOWN_SIGNALS, (signal.SIGTERM,))
        else:
            self.assertEqual(set(_SHUTDOWN_SIGNALS), {signal.SIGTERM, sighup})

        tmp = Path(tempfile.mkdtemp(prefix="kiss-shutdown-signals-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        srv = RemoteAccessServer(
            host="127.0.0.1", port=0,
            url_file=tmp / "unused-url.json",
            local_endpoint_file=tmp / "unused-local.json",
        )
        saved = {sig: signal.getsignal(sig) for sig in _SHUTDOWN_SIGNALS}
        try:
            srv._install_signal_handlers()
            for sig in _SHUTDOWN_SIGNALS:
                self.assertEqual(signal.getsignal(sig), srv._handle_shutdown_signal)
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
        # No running loop: SIGTERM must fall back to KeyboardInterrupt
        # and latch the shutdown flag, on every platform.
        with self.assertRaises(KeyboardInterrupt):
            srv._handle_shutdown_signal(int(signal.SIGTERM))
        self.assertTrue(srv._shutdown_initiated)
