# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: ``/media/`` handles symlink cycles and runs checks off-thread.

``Path.resolve()`` raises ``OSError`` (``ELOOP``) on a symlink cycle;
before the fix that exception escaped ``_process_request`` and surfaced
as an internal server error instead of a 404.  The containment check and
stat must also run in the same worker thread as the payload read, not on
the event loop — verified with a real ``Path`` subclass that records the
thread executing ``resolve()``.
"""

from __future__ import annotations

import asyncio
import ssl
import tempfile
import threading
from pathlib import Path
from typing import Self
from unittest import IsolatedAsyncioTestCase

from kiss.server import web_server
from kiss.server.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class _ResolveRecordingPath(Path):
    """A real ``Path`` whose ``resolve`` records the calling thread."""

    recorded: list[int] = []

    def resolve(self, strict: bool = False) -> Self:
        _ResolveRecordingPath.recorded.append(threading.get_ident())
        return super().resolve(strict)


class TestMediaSymlinkLoop(IsolatedAsyncioTestCase):
    """Serve /media/ from a temp dir containing a symlink cycle."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()
        self._loop_thread = threading.get_ident()

        # Swap MEDIA_DIR only AFTER start (the cached HTML page was
        # already built from the real media dir at server init).
        self._media_dir = Path(tempfile.mkdtemp(prefix="kiss_media_loop_"))
        (self._media_dir / "ok.txt").write_text("hello media")
        (self._media_dir / "loop").symlink_to(self._media_dir / "loop")
        self._orig_media_dir = web_server.MEDIA_DIR
        _ResolveRecordingPath.recorded = []
        web_server.MEDIA_DIR = _ResolveRecordingPath(str(self._media_dir))

    async def asyncTearDown(self) -> None:
        web_server.MEDIA_DIR = self._orig_media_dir
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _http_get(self, path: str) -> tuple[int, str]:
        import urllib.error
        import urllib.request

        url = f"https://127.0.0.1:{self.port}{path}"
        ctx = _no_verify_ssl()

        def _fetch() -> tuple[int, str]:
            try:
                resp = urllib.request.urlopen(url, timeout=5, context=ctx)
                return resp.status, resp.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode() if e.fp else ""

        return await asyncio.get_event_loop().run_in_executor(None, _fetch)

    async def test_symlink_loop_returns_404(self) -> None:
        """A symlink cycle under /media/ yields a clean 404, not a 500."""
        status, _ = await self._http_get("/media/loop")
        self.assertEqual(status, 404)

    async def test_regular_media_file_still_served(self) -> None:
        """A regular file inside MEDIA_DIR is still served with 200."""
        status, body = await self._http_get("/media/ok.txt")
        self.assertEqual(status, 200)
        self.assertEqual(body, "hello media")

    async def test_containment_check_runs_off_loop_thread(self) -> None:
        """``resolve()`` for the containment check must not run on the loop."""
        _ResolveRecordingPath.recorded = []
        status, _ = await self._http_get("/media/ok.txt")
        self.assertEqual(status, 200)
        self.assertTrue(_ResolveRecordingPath.recorded)
        for ident in _ResolveRecordingPath.recorded:
            self.assertNotEqual(ident, self._loop_thread)
