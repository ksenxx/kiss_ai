# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the hourly PyPI update-available check.

``RemoteAccessServer`` periodically polls PyPI for the latest
``kiss-agent-framework`` release and broadcasts an
``update_available`` event to every connected client when the
installed version is older.  The VS Code webview (UDS) and the
remote browser webview (WSS) both receive this event and decorate
the "Update" button in the settings panel with a green download
icon so the user notices that an upgrade is waiting.

These tests spin up a local HTTP server that impersonates the PyPI
JSON API (no mocks/patches) and override
``web_server._PYPI_LATEST_URL`` to point at it.  The
``_VERSION_CHECK_INTERVAL`` constant is also overridden to a small
value so the periodic loop runs within the test budget.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
import kiss.agents.vscode.web_server as ws
from kiss.agents.vscode.web_server import RemoteAccessServer


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


class _PypiStub:
    """Local HTTP server impersonating the PyPI JSON endpoint."""

    def __init__(self, payload: dict[str, object] | None,
                 status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.hits = 0
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None
        self.url: str = ""

    def start(self) -> None:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(
                self, format: str, *args: object,  # noqa: A002
            ) -> None:
                # Silence stderr so the test output stays clean.
                del format, args

            def do_GET(self) -> None:
                stub.hits += 1
                if stub.payload is None:
                    self.send_response(stub.status)
                    self.end_headers()
                    return
                body = json.dumps(stub.payload).encode("utf-8")
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{port}/pypi/kiss-agent-framework/json"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


class _UpdateCheckTestBase(IsolatedAsyncioTestCase):
    """Shared setup for update-check integration tests."""

    PYPI_VERSION = "2099.1.1"
    PYPI_PAYLOAD: dict[str, object] | None = {
        "info": {"version": PYPI_VERSION},
    }
    PYPI_STATUS = 200

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_grace = ws._TAB_CLOSE_GRACE
        self._orig_url = ws._PYPI_LATEST_URL
        self._orig_interval = ws._VERSION_CHECK_INTERVAL
        ws._TAB_CLOSE_GRACE = 0.05
        # Tight check interval keeps the test fast.
        ws._VERSION_CHECK_INTERVAL = 0.2

        self.pypi = _PypiStub(self.PYPI_PAYLOAD, status=self.PYPI_STATUS)
        self.pypi.start()
        ws._PYPI_LATEST_URL = self.pypi.url

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.agents.vscode.web_server import _generate_self_signed_cert
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
        ws._TAB_CLOSE_GRACE = self._orig_grace
        ws._PYPI_LATEST_URL = self._orig_url
        ws._VERSION_CHECK_INTERVAL = self._orig_interval
        await self.server.stop_async()
        self.pypi.stop()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_uds(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )

    async def _send_ready(
        self, writer: asyncio.StreamWriter, tab_id: str,
    ) -> None:
        writer.write(
            json.dumps(
                {"type": "ready", "tabId": tab_id, "restoredTabs": []},
            ).encode("utf-8") + b"\n",
        )
        await writer.drain()

    async def _wait_for_event(
        self,
        reader: asyncio.StreamReader,
        wanted_type: str,
        timeout: float = 5.0,
        max_events: int = 200,
    ) -> dict[str, object]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                raise AssertionError("UDS closed before " + wanted_type)
            msg = json.loads(line.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == wanted_type:
                return msg
        raise AssertionError(f"did not see {wanted_type!r}")


class TestUpdateAvailableBroadcast(_UpdateCheckTestBase):
    """End-to-end: newer PyPI version is broadcast over UDS."""

    PYPI_VERSION = "2099.1.1"
    PYPI_PAYLOAD = {"info": {"version": "2099.1.1"}}

    async def test_update_available_broadcast_to_new_client(self) -> None:
        """A client that connects gets an ``update_available`` event."""
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-update-1")
            ev = await self._wait_for_event(reader, "update_available")
            self.assertEqual(ev.get("available"), True)
            self.assertEqual(ev.get("latest"), "2099.1.1")
            current = ev.get("current")
            self.assertIsInstance(current, str)
            self.assertNotEqual(current, "2099.1.1")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_periodic_loop_polls_repeatedly(self) -> None:
        """The hourly loop keeps polling — multiple hits within the test."""
        # The first check fires on startup, then ``_VERSION_CHECK_INTERVAL``
        # (0.2s) controls the cadence.  Wait long enough for at least
        # three polls so the periodic behaviour is exercised.
        deadline = asyncio.get_event_loop().time() + 3.0
        while self.pypi.hits < 3:
            if asyncio.get_event_loop().time() >= deadline:
                break
            await asyncio.sleep(0.05)
        self.assertGreaterEqual(self.pypi.hits, 3)


class TestUpdateAvailableSameVersion(_UpdateCheckTestBase):
    """When PyPI reports the current version, ``available`` is False."""

    PYPI_VERSION = "0.0.0"  # Will be overridden below.

    async def asyncSetUp(self) -> None:
        from kiss._version import __version__
        # Pin the stub to the *current* installed version so the check
        # sees no upgrade is needed.
        type(self).PYPI_PAYLOAD = {"info": {"version": __version__}}
        await super().asyncSetUp()

    async def test_event_marks_not_available_when_current(self) -> None:
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-update-2")
            ev = await self._wait_for_event(reader, "update_available")
            self.assertEqual(ev.get("available"), False)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


class TestUpdateCheckHandlesNetworkErrors(_UpdateCheckTestBase):
    """A failing PyPI endpoint must not crash the server."""

    PYPI_PAYLOAD = None
    PYPI_STATUS = 500

    async def test_failing_pypi_does_not_break_server(self) -> None:
        # Sleep so the periodic loop fires at least twice against
        # the failing endpoint.
        await asyncio.sleep(1.0)
        # Server is still healthy: UDS still answers ``ready``.
        reader, writer = await self._connect_uds()
        try:
            await self._send_ready(writer, "tab-update-3")
            focus = await self._wait_for_event(reader, "focusInput",
                                               timeout=2.0)
            self.assertEqual(focus.get("tabId"), "tab-update-3")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


class TestVersionCompare(IsolatedAsyncioTestCase):
    """Unit-style coverage for the version compare helper."""

    async def test_compare_versions_ordering(self) -> None:
        self.assertEqual(ws._compare_versions("2026.6.10", "2026.6.9"), 1)
        self.assertEqual(ws._compare_versions("2026.6.9", "2026.6.10"), -1)
        self.assertEqual(ws._compare_versions("2026.6.9", "2026.6.9"), 0)
        # Different number of components.
        self.assertEqual(ws._compare_versions("2026.7", "2026.6.9"), 1)
        self.assertEqual(ws._compare_versions("2026.6", "2026.6.0"), 0)
        # Garbage falls back to equality so a malformed PyPI payload
        # never falsely claims an update.
        self.assertEqual(ws._compare_versions("bad", "2026.6.9"), 0)
