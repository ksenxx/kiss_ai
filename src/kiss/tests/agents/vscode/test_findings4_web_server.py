# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""FINDINGS-4 regression tests for web_server.py.

End-to-end tests against a real :class:`RemoteAccessServer` with real
UDS connections (no mocks):

- F4-02: ``stop_async()`` must close established UDS client streams,
  and (residual) JOIN the in-flight handler coroutines so none touch
  server state after shutdown returns.
- F4-06: ``_handle_submit`` must refuse new tasks once shutdown began.
- F4-10: concurrent self-signed TLS generation publishes a matched
  cert/key pair.
- F4-13: concurrent ``runUpdate`` requests launch a single installer.
- F4-14: SIGHUP triggers the shutdown path instead of being ignored.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import signal
import tempfile
import threading
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class TestFindings4WebServer(IsolatedAsyncioTestCase):
    """E2E tests for the F4 web_server fixes."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)

        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await asyncio.open_unix_connection(str(self.uds_path))

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, cmd: dict) -> None:
        writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
        await writer.drain()

    async def test_f402_stop_async_closes_established_uds_clients(
        self,
    ) -> None:
        """stop_async must disconnect already-accepted UDS streams."""
        reader, writer = await self._connect()
        await self._send(writer, {"type": "activeTasksQuery"})
        line = await asyncio.wait_for(reader.readline(), timeout=3)
        self.assertTrue(line, "no reply before shutdown")

        await self.server.stop_async()

        # The established stream must reach EOF; a live stream would
        # keep answering queries after shutdown returned.
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=3)
            if not line:
                break
        writer.close()

    async def test_f4_stop_async_drains_uds_handlers(self) -> None:
        """stop_async must JOIN in-flight UDS handler coroutines.

        Closing the client stream merely unblocks the handler's
        readline(); without a drain the handler (and its cleanup
        ``finally``) may still be running — touching server state —
        after stop_async returns.
        """
        _reader, writer = await self._connect()
        await self._send(writer, {"type": "activeTasksQuery"})
        await asyncio.sleep(0.15)
        handlers = set(self.server._uds_handler_tasks)
        self.assertTrue(handlers, "UDS handler task was not tracked")

        await self.server.stop_async()

        for task in handlers:
            self.assertTrue(
                task.done(),
                "a UDS handler coroutine was still running after "
                "stop_async returned",
            )
        self.assertFalse(self.server._uds_handler_tasks)
        writer.close()

    async def test_f406_submit_refused_after_shutdown_started(self) -> None:
        """_handle_submit must not start tasks once shutdown began."""
        from kiss.server import agent_state

        self.server._shutdown_initiated = True
        try:
            await self.server._handle_submit(
                {"tabId": "f406-tab", "prompt": "do work"},
            )
            state = agent_state.find_by_tab("f406-tab")
            self.assertTrue(
                state is None or not state.is_task_active,
                "a task was started after shutdown was initiated; the "
                "worker sweep already ran and this task would be "
                "killed abruptly at process exit",
            )
        finally:
            self.server._shutdown_initiated = False

    async def test_f410_concurrent_tls_generation_is_serialised(
        self,
    ) -> None:
        """Sibling generators must publish a matched cert/key pair."""
        import kiss.server.web_server as ws

        contexts: list[object] = []
        errors: list[BaseException] = []

        def _gen() -> None:
            try:
                contexts.append(ws._create_ssl_context())
            except BaseException as exc:  # noqa: BLE001 — test collector
                errors.append(exc)

        threads = [threading.Thread(target=_gen) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertFalse(
            errors,
            f"concurrent TLS generation failed: {errors!r} — a "
            "mismatched or missing cert/key pair was observed",
        )
        self.assertEqual(len(contexts), 6)

    async def test_f410_mismatched_tls_pair_self_heals(self) -> None:
        """A crash-published mismatched cert/key pair must self-heal.

        A daemon that died between writing the new key and the new
        cert leaves a mismatched pair on disk; without recovery every
        subsequent ``_create_ssl_context()`` raises ``SSLError``
        forever (both files exist and the cert is not expiring).
        """
        import ssl as ssl_mod

        import kiss.server.web_server as ws

        tls_dir = Path(self.tmpdir) / "tls-heal"
        pair_b = Path(self.tmpdir) / "tls-other"
        tls_dir.mkdir()
        pair_b.mkdir()
        _generate_self_signed_cert(tls_dir / "cert.pem", tls_dir / "key.pem")
        _generate_self_signed_cert(pair_b / "cert.pem", pair_b / "key.pem")
        # Simulate the crash window: key from pair B, cert from pair A.
        (tls_dir / "key.pem").write_bytes((pair_b / "key.pem").read_bytes())
        probe = ssl_mod.SSLContext(ssl_mod.PROTOCOL_TLS_SERVER)
        with self.assertRaises(ssl_mod.SSLError):
            probe.load_cert_chain(
                str(tls_dir / "cert.pem"), str(tls_dir / "key.pem"),
            )

        saved_tls_dir = ws._TLS_DIR
        ws._TLS_DIR = tls_dir
        try:
            ctx = ws._create_ssl_context()
        finally:
            ws._TLS_DIR = saved_tls_dir
        self.assertIsInstance(ctx, ssl_mod.SSLContext)
        # The healed on-disk pair must now be self-consistent.
        probe = ssl_mod.SSLContext(ssl_mod.PROTOCOL_TLS_SERVER)
        probe.load_cert_chain(
            str(tls_dir / "cert.pem"), str(tls_dir / "key.pem"),
        )

    async def test_f413_concurrent_run_update_single_flight(self) -> None:
        """Two concurrent runUpdate requests spawn ONE installer."""
        install_root = Path(self.tmpdir) / "kiss_ai"
        install_root.mkdir()
        marker = install_root / "runs.txt"
        (install_root / "install.sh").write_text(
            f"#!/bin/bash\necho run >> {marker}\nsleep 1\n",
        )
        self.server._install_root = install_root
        self.server._update_log_path = Path(self.tmpdir) / "update.log"

        await asyncio.gather(
            self.server._handle_run_update(),
            self.server._handle_run_update(),
        )
        await asyncio.sleep(0.5)
        runs = (
            marker.read_text().strip().splitlines()
            if marker.exists() else []
        )
        self.assertEqual(
            len(runs), 1,
            f"{len(runs)} installers were launched concurrently; the "
            "update must be single-flight",
        )

    async def test_f414_sighup_triggers_shutdown_path(self) -> None:
        """SIGHUP must route through shutdown, not be swallowed."""
        srv = RemoteAccessServer(
            host="127.0.0.1", port=0,
            url_file=Path(self.tmpdir) / "unused-url.json",
            uds_path=Path(self.tmpdir) / "unused.sock",
        )
        # No running loop: the handler must fall back to raising
        # KeyboardInterrupt (previously SIGHUP returned silently,
        # leaving the daemon running with the OS default replaced).
        with self.assertRaises(KeyboardInterrupt):
            srv._handle_shutdown_signal(signal.SIGHUP)
        self.assertTrue(srv._shutdown_initiated)
