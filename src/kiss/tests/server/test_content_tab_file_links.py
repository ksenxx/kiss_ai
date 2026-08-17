# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.agents.vscode.test_content_tab_file_links``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert

_PY_SOURCE = 'def greet(name):\n    return "hello " + name\n'


_HTML_SOURCE = (
    "<!DOCTYPE html><html><body>"
    "<h1 id='marker'>KISS-HTML-MARKER</h1>"
    "</body></html>"
)


_MD_SOURCE = "# KISS-MD-TITLE\n\nSome **bold** words.\n"


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


class _ServerHarness:
    """A real RemoteAccessServer running on a background event loop."""

    def __init__(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-content-tab-")
        tmp = Path(self.tmpdir)
        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = tmp / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = tmp / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        self.work_dir = tmp / "repo"
        self.work_dir.mkdir()
        (self.work_dir / "sample.py").write_text(_PY_SOURCE)
        (self.work_dir / "page.html").write_text(_HTML_SOURCE)
        (self.work_dir / "notes.md").write_text(_MD_SOURCE)
        (self.work_dir / "binary.bin").write_bytes(b"\x00\x01\x02\x03")

        certfile = tmp / "cert.pem"
        keyfile = tmp / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _find_free_port()
        self.base_url = f"https://127.0.0.1:{self.port}"
        self.ws_url = f"wss://127.0.0.1:{self.port}/ws"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp / "remote-url.json",
            uds_path=tmp / "sorcar.sock",
            work_dir=str(self.work_dir),
        )
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self._thread.start()
        asyncio.run_coroutine_threadsafe(
            self.server.start_async(), self.loop,
        ).result(60)

    def run(self, coro: Coroutine[Any, Any, dict]) -> dict:
        """Run *coro* on the server loop and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(60)

    def stop(self) -> None:
        """Stop the server, its loop, and restore redirected globals."""
        try:
            asyncio.run_coroutine_threadsafe(
                self.server.stop_async(), self.loop,
            ).result(60)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=30)
            self.loop.close()
            if th._db_conn is not None:
                th._db_conn.close()
            th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence
            vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
            shutil.rmtree(self.tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def harness():
    """One shared real server for every browser test in this module."""
    h = _ServerHarness()
    yield h
    h.stop()


async def _ws_request(harness: _ServerHarness, payload: dict) -> dict:
    """Authenticate over wss:// , send *payload*, return the
    ``fileContent`` reply."""
    async with connect(harness.ws_url, ssl=_no_verify_ssl()) as ws:
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == "auth_ok":
                break
        await ws.send(json.dumps(payload))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == "fileContent":
                reply: dict = msg
                return reply


class TestOpenFileBackend:
    """Protocol-level tests for the ``openFile`` → ``fileContent``
    request/reply over a real ``wss://`` connection (no browser).

    Each request coroutine runs on the harness's own background event
    loop so these tests coexist with the sync-Playwright tests above
    (which forbid a running event loop in the pytest thread).
    """

    def _request(self, harness: _ServerHarness, payload: dict) -> dict:
        return harness.run(_ws_request(harness, payload))

    def test_absolute_path_returns_content(self, harness) -> None:
        path = str(harness.work_dir / "sample.py")
        reply = self._request(
            harness, {"type": "openFile", "path": path, "tabId": "t-1"},
        )
        assert reply["name"] == "sample.py"
        assert reply["content"] == _PY_SOURCE
        assert reply["tabId"] == "t-1"
        assert "error" not in reply

    def test_relative_path_uses_work_dir_field(self, harness) -> None:
        reply = self._request(harness, {
            "type": "openFile",
            "path": "page.html",
            "workDir": str(harness.work_dir),
        })
        assert reply["name"] == "page.html"
        assert reply["content"] == _HTML_SOURCE

    def test_missing_file_returns_error(self, harness) -> None:
        reply = self._request(
            harness, {"type": "openFile", "path": "/no/such/file.py"},
        )
        assert "File not found" in reply["error"]
        assert "content" not in reply

    def test_binary_file_returns_error(self, harness) -> None:
        path = str(harness.work_dir / "binary.bin")
        reply = self._request(harness, {"type": "openFile", "path": path})
        assert "binary" in reply["error"].lower()
        assert "content" not in reply

    def test_oversized_file_returns_error(self, harness) -> None:
        big = harness.work_dir / "big.txt"
        big.write_text("x" * 2_500_000)
        try:
            reply = self._request(
                harness, {"type": "openFile", "path": str(big)},
            )
            assert "too large" in reply["error"]
            assert "content" not in reply
        finally:
            big.unlink()
