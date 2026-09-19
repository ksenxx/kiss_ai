# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the daemon serves WSS whether or not the platform has AF_UNIX.

CPython on Windows has no ``socket.AF_UNIX`` and no
``asyncio.start_unix_server``.  Before the fix, ``_setup_server``
tripped over the missing attribute inside its UDS ``try`` block and
logged a WARNING with a traceback on every daemon start, even though
nothing was wrong with the platform.  The daemon must instead skip
UDS setup cleanly: one INFO line, ``_uds_server is None``, and the
WSS listener still serving.

The same test file runs on both platforms without faking anything:
on a POSIX host it asserts the normal path (UDS bound, socket file
present, a UDS client gets served, no fallback log); on Windows it
asserts the fallback path.  Either way a real WSS client must complete
the auth handshake against the running server.
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
    _unix_sockets_supported,
)

_HAS_AF_UNIX = hasattr(socket, "AF_UNIX")
_FALLBACK_TEXT = "Unix-domain sockets are unavailable"
_BIND_FAILURE_TEXT = "Failed to bind UDS"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestUdsSetupFollowsPlatform(IsolatedAsyncioTestCase):
    """UDS is bound where AF_UNIX exists and skipped cleanly where it does not."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-uds-platform-"))
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        certfile, keyfile = self.tmpdir / "cert.pem", self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _free_port()
        self.uds_path = self.tmpdir / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            uds_path=self.uds_path,
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

    async def test_wss_serves_and_uds_matches_platform(self) -> None:
        records = await self._start_capturing_logs()
        fallback = [r for r in records if _FALLBACK_TEXT in r.getMessage()]
        bind_failures = [
            r for r in records if _BIND_FAILURE_TEXT in r.getMessage()
        ]
        self.assertEqual(bind_failures, [], "UDS bind must never log a failure")
        self.assertIsNotNone(self.server._ws_server)
        await self._wss_auth_round_trip()

        self.assertEqual(_unix_sockets_supported(), _HAS_AF_UNIX)
        if _HAS_AF_UNIX:
            # Normal path: UDS bound, no fallback message, a UDS
            # client is served over the same protocol.
            self.assertEqual(fallback, [])
            self.assertIsNotNone(self.server._uds_server)
            self.assertTrue(self.uds_path.exists())
            self.assertIsNotNone(self.server._uds_inode)
            reader, writer = await asyncio.open_unix_connection(
                str(self.uds_path),
            )
            try:
                writer.write(
                    json.dumps({"type": "getDefaultModel"}).encode() + b"\n",
                )
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), 30)
                self.assertTrue(line, "UDS client got no reply")
            finally:
                writer.close()
                await writer.wait_closed()
            self.assertTrue(await self.server._uds_socket_is_live())
        else:
            # Fallback path (Windows): exactly one INFO line, no
            # listener, no socket file, and the probes are inert.
            self.assertEqual(len(fallback), 1, [r.getMessage() for r in records])
            self.assertEqual(fallback[0].levelno, logging.INFO)
            self.assertIsNone(fallback[0].exc_info)
            self.assertEqual(
                [r for r in records if r.levelno >= logging.WARNING], [],
                "no warning may be logged for the missing UDS channel",
            )
            self.assertIsNone(self.server._uds_server)
            self.assertIsNone(self.server._uds_inode)
            self.assertFalse(self.uds_path.exists())
            self.assertFalse(await self.server._uds_socket_is_live())
            self.server._unlink_own_uds_socket()  # must be a silent no-op

        await self.server.stop_async()
        self.stopped = True
        self.assertIsNone(self.server._uds_server)
        self.assertFalse(self.uds_path.exists())


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
            url_file=tmp / "unused-url.json", uds_path=tmp / "unused.sock",
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
