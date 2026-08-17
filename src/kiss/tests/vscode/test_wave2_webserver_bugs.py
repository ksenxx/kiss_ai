# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Tests that stay behind from ``kiss.tests.vscode.test_wave2_webserver_bugs``
(now ``kiss.tests.server.test_wave2_webserver_bugs``): they assert on assets outside
kiss.core/kiss.agents.sorcar/kiss.server (e.g. the real
chat.html/main.js media content), so they keep their original
location while the server-only majority of the file moved to
tests/server.
"""


from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from websockets.datastructures import Headers
from websockets.http11 import Request

import kiss.server.web_server as ws_mod
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.test_wave2_webserver_bugs import _redirect_persistence, _restore_persistence


class TestF5AndF10LiveServer(unittest.IsolatedAsyncioTestCase):
    """E2E tests over a real running RemoteAccessServer."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-w2f5-live-")
        self.saved = _redirect_persistence(self.tmpdir)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self._stopped = False

    async def _stop_server(self) -> None:
        if not self._stopped:
            self._stopped = True
            await self.server.stop_async()

    async def asyncTearDown(self) -> None:
        await self._stop_server()
        _restore_persistence(self.saved)
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_f5_media_and_voice_model_served_correctly(self) -> None:
        """F5: HTTP branches serve correct bytes (reads off the loop)."""
        req = Request("/media/main.js", Headers())
        resp = await self.server._process_request(cast(Any, None), req)
        assert resp is not None
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.body, (ws_mod.MEDIA_DIR / "main.js").read_bytes())

        missing = Request("/media/../secrets.txt", Headers())
        resp = await self.server._process_request(cast(Any, None), missing)
        assert resp is not None
        self.assertEqual(resp.status_code, 404)

        payload = os.urandom(256 * 1024)
        cache = Path(self.tmpdir) / "models" / "model.tar.gz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
        saved_cache = ws_mod.VOICE_MODEL_CACHE
        ws_mod.VOICE_MODEL_CACHE = cache
        try:
            req = Request("/voice-model.tar.gz", Headers())
            resp = await self.server._process_request(cast(Any, None), req)
            assert resp is not None
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.body, payload)
        finally:
            ws_mod.VOICE_MODEL_CACHE = saved_cache
