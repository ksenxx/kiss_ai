# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the hourly PyPI update-available check.

``RemoteAccessServer`` periodically polls PyPI for the latest
``kiss-agent-framework`` release and broadcasts an
``update_available`` event to every connected client when the
installed version is older.  Every browser tab receives this event
and decorates the "Update" button in the settings panel with a green
download icon so the user notices that an upgrade is waiting.

These tests spin up a local HTTP server that impersonates the PyPI
JSON API (no mocks/patches) and override
``web_server._PYPI_LATEST_URL`` to point at it.  The
``_VERSION_CHECK_INTERVAL`` constant is also overridden to a small
value so the periodic loop runs within the test budget.

The root conftest switches the check off for the whole test session
(``KISS_DISABLE_UPDATE_CHECK=1``, inherited by subprocess daemons);
these are the check's own tests, so :class:`_UpdateCheckTestBase`
lifts the switch while one of them runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
import kiss.server.web_server as ws
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.conftest import redirect_persistence, restore_persistence

_KILL_SWITCH = "KISS_DISABLE_UPDATE_CHECK"


class _PypiStub:
    """Local HTTP server impersonating the PyPI JSON endpoint.

    It also counts the ``CONNECT`` a client sends when the stub is its
    HTTPS proxy, and refuses the tunnel with *status*.
    """

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

            do_CONNECT = do_GET  # noqa: N815 — BaseHTTPRequestHandler API

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
    # False: the daemon under test starts with the session's kill switch.
    LIFTS_KILL_SWITCH = True

    async def asyncSetUp(self) -> None:
        if self.LIFTS_KILL_SWITCH:
            self.addCleanup(
                os.environ.__setitem__, _KILL_SWITCH, os.environ.pop(_KILL_SWITCH, ""),
            )
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        self._orig_url = ws._PYPI_LATEST_URL
        self._orig_interval = ws._VERSION_CHECK_INTERVAL
        ws._VERSION_CHECK_INTERVAL = 0.2

        self.pypi = _PypiStub(self.PYPI_PAYLOAD, status=self.PYPI_STATUS)
        self.pypi.start()
        ws._PYPI_LATEST_URL = self.pypi.url

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.tests.local_ws import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        # The server stops first: its 0.2 s check loop must not see the
        # real PyPI address come back while the kill switch is lifted.
        await self.server.stop_async()
        ws._PYPI_LATEST_URL = self._orig_url
        ws._VERSION_CHECK_INTERVAL = self._orig_interval
        self.pypi.stop()
        th._close_thread_db()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_local(
        self,
    ) -> tuple[LocalReader, LocalWriter]:
        return await open_local_connection(
            self.server, limit=16 * 1024 * 1024,
        )

    async def _send_ready(
        self, writer: LocalWriter, tab_id: str,
    ) -> None:
        writer.write(
            json.dumps(
                {"type": "ready", "tabId": tab_id, "restoredTabs": []},
            ).encode("utf-8") + b"\n",
        )
        await writer.drain()

    async def _wait_for_event(
        self,
        reader: LocalReader,
        wanted_type: str,
        timeout: float = 5.0,
        max_events: int = 200,
    ) -> dict[str, object]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                raise AssertionError("local connection closed before " + wanted_type)
            msg = json.loads(line.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == wanted_type:
                return msg
        raise AssertionError(f"did not see {wanted_type!r}")


class TestUpdateAvailableBroadcast(_UpdateCheckTestBase):
    """End-to-end: newer PyPI version is broadcast over the local channel."""

    PYPI_VERSION = "2099.1.1"
    PYPI_PAYLOAD = {"info": {"version": "2099.1.1"}}

    async def test_update_available_broadcast_to_new_client(self) -> None:
        """A client that connects gets an ``update_available`` event."""
        reader, writer = await self._connect_local()
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
        deadline = asyncio.get_event_loop().time() + 3.0
        while self.pypi.hits < 3:
            if asyncio.get_event_loop().time() >= deadline:
                break
            await asyncio.sleep(0.05)
        self.assertGreaterEqual(self.pypi.hits, 3)


class TestUpdateAvailableSameVersion(_UpdateCheckTestBase):
    """When PyPI reports the current version, ``available`` is False."""

    PYPI_VERSION = "0.0.0"

    async def asyncSetUp(self) -> None:
        from kiss.core._version import __version__
        type(self).PYPI_PAYLOAD = {"info": {"version": __version__}}
        await super().asyncSetUp()

    async def test_event_marks_not_available_when_current(self) -> None:
        reader, writer = await self._connect_local()
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
        await asyncio.sleep(1.0)
        reader, writer = await self._connect_local()
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


class TestUpdateCheckSwitchedOff(_UpdateCheckTestBase):
    """With the session's ``KISS_DISABLE_UPDATE_CHECK=1`` the daemon never asks."""

    LIFTS_KILL_SWITCH = False

    async def test_the_daemon_neither_asks_pypi_nor_learns_a_release(self) -> None:
        await asyncio.sleep(1.0)  # five ticks of the 0.2 s check loop
        self.assertEqual(self.pypi.hits, 0)
        self.assertIsNone(self.server._latest_version)


def test_a_subprocess_test_daemon_does_not_ask_pypi(tmp_path: Path) -> None:
    """The session's kill switch reaches the daemons ``start_remote_daemon`` spawns.

    The child reaches HTTPS hosts through a local proxy, where its
    request to PyPI would arrive as ``CONNECT pypi.org:443``.  The check
    runs right after the daemon wrote ``remote-url.json``.
    """
    proxy = _PypiStub(None, status=502)
    proxy.start()
    address = proxy.url.partition("/pypi/")[0]
    home = tmp_path / "home"
    daemon = start_remote_daemon(home, env={
        "HTTPS_PROXY": address, "https_proxy": address, "NO_PROXY": "", "no_proxy": "",
        "HOME": str(tmp_path),
    })
    try:
        deadline = time.monotonic() + 30
        while not (home / "remote-url.json").exists():
            assert time.monotonic() < deadline, "the daemon never finished starting"
            time.sleep(0.05)
        time.sleep(1.0)
        assert proxy.hits == 0
    finally:
        daemon.stop()
        proxy.stop()


class TestVersionCompare(IsolatedAsyncioTestCase):
    """Unit-style coverage for the version compare helper."""

    async def test_compare_versions_ordering(self) -> None:
        self.assertEqual(ws._compare_versions("2026.6.10", "2026.6.9"), 1)
        self.assertEqual(ws._compare_versions("2026.6.9", "2026.6.10"), -1)
        self.assertEqual(ws._compare_versions("2026.6.9", "2026.6.9"), 0)
        self.assertEqual(ws._compare_versions("2026.7", "2026.6.9"), 1)
        self.assertEqual(ws._compare_versions("2026.6", "2026.6.0"), 0)
        self.assertEqual(ws._compare_versions("bad", "2026.6.9"), 0)
