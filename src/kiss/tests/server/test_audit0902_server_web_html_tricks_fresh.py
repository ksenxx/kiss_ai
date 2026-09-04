# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the remote webapp page must not serve a trick list frozen at daemon start.

``RemoteAccessServer`` rendered ``media/chat.html`` once in ``__init__``
and served those bytes for every later ``GET /``.  The page embeds
``window.__TRICKS__`` (the Inject-instruction panel's list), so a user
who edits ``~/.kiss/MY_INJECTION.md`` while the daemon runs saw the
OLD list in the remote webapp until the daemon restarted — while the
same daemon's ghost-text ``trick`` completions (``tricks.read_tricks``
is re-read on every keystroke) and the VS Code webview (which rebuilds
the page per tab) already offered the NEW trick.  Two surfaces of one
list disagreeing is exactly the stale-cache inconsistency this pins:
after the file changes, the next page load must carry the new trick,
and the page must agree with the daemon's own completion source.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer

_TRICK_A = "Alpha trick: run the tests first."
_TRICK_B = "Bravo trick: profile before optimising."


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


class _RecordingPrinter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class TestServedTricksFollowFileEdits(IsolatedAsyncioTestCase):
    """Edit MY_INJECTION.md while the daemon runs; reload the page."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": ""})
        self._saved_env = {
            k: os.environ.get(k) for k in ("KISS_HOME", "KISS_INJECTIONS_PATH")
        }
        kiss_dir = Path(tempfile.mkdtemp(prefix="kiss_html_fresh_")) / ".kiss"
        kiss_dir.mkdir(parents=True)
        self.my_injection = kiss_dir / "MY_INJECTION.md"
        self.my_injection.write_text("## Trick\n\n" + _TRICK_A + "\n", encoding="utf-8")
        empty_bundled = kiss_dir / "empty_INJECTIONS.md"
        empty_bundled.write_text("", encoding="utf-8")
        os.environ["KISS_HOME"] = str(kiss_dir)
        os.environ["KISS_INJECTIONS_PATH"] = str(empty_bundled)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)

    async def _served_tricks(self) -> list[str]:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", self.port, ssl=_no_verify_ssl(),
            server_hostname="localhost",
        )
        writer.write(
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=15)
        writer.close()
        body = raw.split(b"\r\n\r\n", 1)[1].decode("utf-8")
        m = re.search(r"window\.__TRICKS__\s*=\s*(\[.*?\]);", body, re.DOTALL)
        assert m is not None, "TRICKS_JSON placeholder was not substituted"
        return list(json.loads(m.group(1).replace("<\\/", "</")))

    def _daemon_trick_completions(self, query: str) -> list[str]:
        printer = _RecordingPrinter()
        backend = self.server._vscode_server
        saved = backend.printer
        backend.printer = printer  # type: ignore[assignment]
        try:
            backend._complete(query)
        finally:
            backend.printer = saved
        comps = next(e for e in printer.events if e["type"] == "completions")
        return [c["text"] for c in comps["completions"] if c["type"] == "trick"]

    async def test_page_reload_serves_edited_tricks(self) -> None:
        self.assertEqual(await self._served_tricks(), [_TRICK_A])

        # The user edits the file while the daemon keeps running.
        self.my_injection.write_text(
            "## Trick\n\n" + _TRICK_B + "\n\n## Trick\n\n" + _TRICK_A + "\n",
            encoding="utf-8",
        )
        # The daemon's own completion source already sees the edit …
        self.assertEqual(self._daemon_trick_completions("Bravo"), [_TRICK_B])
        # … so the next page load must agree with it.
        self.assertEqual(
            await self._served_tricks(), [_TRICK_B, _TRICK_A],
            "BUG: remote webapp serves the trick list frozen at daemon "
            "start; the Inject panel disagrees with ghost completions",
        )
