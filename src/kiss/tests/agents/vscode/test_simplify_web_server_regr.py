# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Live-server routing tests that assert on the real vscode media assets.

Split from ``kiss.tests.server.test_simplify_web_server_regr``: these
tests assert that the real ``/media/main.js`` and ``/media/main.css``
files (bundled under ``src/kiss/agents/vscode/media``) are served, so
they live in tests/agents/vscode while the server-only majority of the
original file lives in tests/server.
"""


from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlsplit

from websockets.datastructures import Headers
from websockets.http11 import Request

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.test_simplify_web_server_regr import (
    _redirect_persistence,
    _restore_persistence,
)

# ``_process_request`` refuses non-loopback peers while the remote
# password is empty, and it fails closed on a missing peer address, so
# a bare ``None`` connection would be answered 403 instead of routed.
_LOOPBACK_CONN = SimpleNamespace(remote_address=("127.0.0.1", 0))


class TestLiveServerPaths(unittest.IsolatedAsyncioTestCase):
    """E2E tests over a real running RemoteAccessServer (WSS + UDS)."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-simp-live-")
        self.saved = _redirect_persistence(self.tmpdir)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        self.server._install_root = Path(self.tmpdir) / "kiss_ai"
        self.server._update_log_path = Path(self.tmpdir) / "update.log"
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_process_request_routing(self) -> None:
        """/, /trajectories, /media and unknown paths route correctly."""

        def req(path: str) -> Request:
            return Request(path, Headers({"Host": "localhost"}))

        conn = cast(Any, _LOOPBACK_CONN)

        resp = await self.server._process_request(conn, req("/"))
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"<html", resp.body.lower())
        resp = await self.server._process_request(conn, req(""))
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(
            await self.server._process_request(conn, req("/ws")),
        )
        for tpath in ("/trajectories", "/trajectories/"):
            resp = await self.server._process_request(conn, req(tpath))
            assert resp is not None
            self.assertEqual(resp.status_code, 200)
        resp = await self.server._process_request(conn, req("/media/main.js"))
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        resp = await self.server._process_request(conn, req("/nope"))
        assert resp is not None
        self.assertEqual(resp.status_code, 404)
        parsed = urlsplit("/media/main.css?v=abc")
        self.assertEqual(parsed.path, "/media/main.css")
        resp = await self.server._process_request(
            conn, req("/media/main.css?v=abc"),
        )
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
