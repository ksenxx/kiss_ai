# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.webapp.test_content_tab_file_links``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.server import web_server
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import _generate_self_signed_cert
from kiss.tests.persistence_helpers import close_db
from kiss.tests.server.conftest import find_free_port, no_verify_ssl

_PY_SOURCE = 'def greet(name):\n    return "hello " + name\n'


_HTML_SOURCE = (
    "<!DOCTYPE html><html><body>"
    "<h1 id='marker'>KISS-HTML-MARKER</h1>"
    "</body></html>"
)


_MD_SOURCE = "# KISS-MD-TITLE\n\nSome **bold** words.\n"


class _ServerHarness:
    """A real RemoteAccessServer running on a background event loop."""

    def __init__(self) -> None:
        # Canonical (symlink-free): the daemon reports paths it has
        # resolved (fileContent/fileSaved, git worktree roots), so a
        # symlinked temp dir (macOS /var -> /private/var) would make
        # the paths tests compare differ from the ones they created.
        self.tmpdir = os.path.realpath(tempfile.mkdtemp(prefix="kiss-content-tab-"))
        tmp = Path(self.tmpdir)
        self._saved_persistence = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = tmp / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = tmp / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        self.work_dir = tmp / "repo"
        self.work_dir.mkdir()
        # newline="\n": openFile echoes the bytes on disk, so the fixtures
        # must not let Windows text mode rewrite "\n" as "\r\n".
        (self.work_dir / "sample.py").write_text(_PY_SOURCE, newline="\n")
        (self.work_dir / "page.html").write_text(_HTML_SOURCE, newline="\n")
        (self.work_dir / "notes.md").write_text(_MD_SOURCE, newline="\n")
        (self.work_dir / "binary.bin").write_bytes(b"\x00\x01\x02\x03")

        certfile = tmp / "cert.pem"
        keyfile = tmp / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = find_free_port()
        self.base_url = f"https://127.0.0.1:{self.port}"
        self.ws_url = f"wss://127.0.0.1:{self.port}/ws"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp / "remote-url.json",
            local_endpoint_file=tmp / "sorcar-local.json",
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
        """Stop the server, join its loop's worker threads, stop the loop, restore the globals.

        The worker threads are joined before the loop closes, as
        ``asyncio.run`` joins them for the real daemon: work a
        connection started in the loop's default executor
        (``asyncio.to_thread``, e.g. the Connectors status probe a
        page asks for) would otherwise outlive the harness.
        """
        try:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.server.stop_async(), self.loop,
                ).result(60)
            finally:
                asyncio.run_coroutine_threadsafe(
                    self.loop.shutdown_default_executor(), self.loop,
                ).result(60)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=30)
            self.loop.close()
            close_db()
            th._DB_PATH, th._KISS_DIR = self._saved_persistence
            vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
            shutil.rmtree(self.tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def harness():
    """One shared real server for every browser test in this module."""
    h = _ServerHarness()
    yield h
    h.stop()


async def _refuses_executor_work() -> bool:
    """Whether the running loop takes no more work for its default executor: it is joining it."""
    try:
        await asyncio.get_running_loop().run_in_executor(None, int)
    except RuntimeError:
        return True
    return False


def _work_until_joined(
    loop: asyncio.AbstractEventLoop, working: threading.Event, joined: threading.Event,
) -> None:
    """Executor work of *loop* that lasts until the loop joins its executor, told by *joined*."""
    working.set()
    try:
        while not asyncio.run_coroutine_threadsafe(_refuses_executor_work(), loop).result(30):
            time.sleep(0.01)
    except (RuntimeError, TimeoutError):
        return  # the loop was stopped or closed over this work: nothing joined it
    joined.set()


def test_harness_stop_joins_the_work_its_loop_started_in_threads() -> None:
    """``_ServerHarness.stop`` returns once the work in its loop's executor is over, not before.

    The work here ends when the loop refuses more work for its
    executor, which is how joining the executor begins: a ``stop`` that
    closed the loop without joining returned over work still running
    (a real page's status probe was left to the session's reaper).
    """
    own = _ServerHarness()
    working, joined = threading.Event(), threading.Event()
    own.loop.call_soon_threadsafe(
        own.loop.run_in_executor, None, _work_until_joined, own.loop, working, joined,
    )
    try:
        assert working.wait(30), "the loop never started the work"
    finally:
        own.stop()
    assert joined.is_set(), "the harness closed its loop over work running in its executor"


async def _ws_request(harness: _ServerHarness, payload: dict) -> dict:
    """Authenticate over wss:// , send *payload*, return the
    ``fileContent`` reply."""
    async with connect(harness.ws_url, ssl=no_verify_ssl()) as ws:
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
        assert "unsupported" not in reply

    def test_binary_file_returns_error(self, harness) -> None:
        """A binary file no tab can show: its error, flagged, naming the resolved file."""
        path = str(harness.work_dir / "binary.bin")
        reply = self._request(harness, {
            "type": "openFile", "path": "binary.bin", "workDir": str(harness.work_dir),
        })
        assert reply["error"] == "Cannot display binary file: binary.bin"
        assert (reply["unsupported"], reply["path"], reply["name"]) == (True, path, "binary.bin")
        assert "content" not in reply
        # Past the text cap it keeps the sentence it had, and is flagged too.
        big = harness.work_dir / "big.bin"
        big.write_bytes(b"\0" * (web_server._OPEN_FILE_MAX_BYTES + 1))
        try:
            reply = self._request(harness, {"type": "openFile", "path": str(big)})
        finally:
            big.unlink()
        assert reply["error"] == f"File too large to display: {big}"
        assert (reply["unsupported"], reply["path"], reply["name"]) == (True, str(big), "big.bin")

    def test_download_serves_the_bytes_of_any_file(self, harness) -> None:
        """``download`` answers with the file's bytes, typed by its name, for the client to save.

        A type a browser would run as a page, and a name that tells
        none, are plain bytes.
        """
        docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        plain = "application/octet-stream"
        cases = {
            "report.docx": docx, "clip.mp4": "video/mp4", "blob.kissdata": plain,
            "page.html": plain, "data.xml": plain, "drawing.svg": plain,
        }
        folder = harness.work_dir / "downloads"
        folder.mkdir()
        try:
            for name, mime in cases.items():
                data = b"PK\x03\x04\x00" + name.encode()
                (folder / name).write_bytes(data)
                reply = self._request(harness, {
                    "type": "openFile", "path": name, "workDir": str(folder),
                    "tabId": "t-1", "token": "open:3", "download": True,
                })
                assert reply == {
                    "type": "fileContent", "path": str(folder / name), "name": name,
                    "tabId": "t-1", "token": "open:3", "binary": True, "download": True,
                    "mime": mime, "size": len(data),
                    "base64": base64.b64encode(data).decode("ascii"),
                }, name
            # Only ``true`` asks for the bytes, and a picture a tab
            # shows is not turned into a download by being opened.
            (folder / "dot.png").write_bytes(b"\x89PNG\r\n\x1a\n\0")
            for name, download in (("report.docx", "yes"), ("dot.png", False)):
                reply = self._request(harness, {
                    "type": "openFile", "path": str(folder / name), "download": download,
                })
                assert "download" not in reply, name
                assert reply.get("unsupported", False) is (name == "report.docx"), name
            # The size cap is the one of the files a viewer tab shows.
            big = folder / "big.bin"
            with open(big, "wb") as f:
                f.truncate(web_server._OPEN_BINARY_MAX_BYTES + 1)
            reply = self._request(harness, {
                "type": "openFile", "path": str(big), "download": True,
            })
            assert reply["error"] == f"File too large to display: {big}"
            assert "unsupported" not in reply and "base64" not in reply
        finally:
            shutil.rmtree(folder)

    def test_line_number_echoed_in_reply(self, harness) -> None:
        """A ``path:NN`` link's line arrives as ``line``; the reply
        echoes it so the client can jump the content tab there."""
        path = str(harness.work_dir / "sample.py")
        reply = self._request(
            harness,
            {"type": "openFile", "path": path, "line": 2, "tabId": "t-9"},
        )
        assert reply["line"] == 2
        assert reply["content"] == _PY_SOURCE
        assert "error" not in reply

    def test_invalid_line_values_not_echoed(self, harness) -> None:
        """Non-positive, non-int, and bool ``line`` values are dropped:
        the reply carries no ``line`` field at all."""
        path = str(harness.work_dir / "sample.py")
        for bad in (0, -3, True, False, "7", 2.5, None):
            reply = self._request(
                harness, {"type": "openFile", "path": path, "line": bad},
            )
            assert "line" not in reply, f"line={bad!r} leaked into reply"
            assert reply["content"] == _PY_SOURCE

    def test_line_echoed_even_on_error_reply(self, harness) -> None:
        """The echo does not depend on the read succeeding — a missing
        file's error reply still carries the validated line."""
        reply = self._request(
            harness,
            {"type": "openFile", "path": "/no/such/file.py", "line": 5},
        )
        assert reply["line"] == 5
        assert "File not found" in reply["error"]

    def test_oversized_file_returns_error(self, harness) -> None:
        big = harness.work_dir / "big.txt"
        big.write_text("x" * 2_500_000)
        try:
            reply = self._request(
                harness, {"type": "openFile", "path": str(big)},
            )
            assert "too large" in reply["error"]
            assert "content" not in reply
            assert "unsupported" not in reply
        finally:
            big.unlink()
