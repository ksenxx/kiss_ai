"""Integration tests for VS Code agent audit round 7.

These tests confirm concrete bugs and inconsistencies found in
``src/kiss/agents/vscode``:

* ``RemoteAccessServer.start_async`` must publish the same active URL
  state as the blocking server path, otherwise the standalone web UI
  cannot show its own web/mobile URL when started from an existing
  asyncio loop.
* Runtime backend/web events must be represented in ``types.ts`` so the
  extension's message contract matches the events actually emitted by
  Python and consumed by ``main.js``.
* Legacy ``userActionDone`` messages must be routed to the active tab's
  task process, not the shared service process, so they can unblock the
  agent waiting in that tab.
* The VSIX package ignore file must exclude local/generated artifacts
  from the extension payload.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.agents.vscode.vscode_config import CONFIG_PATH, save_config
from kiss.agents.vscode.web_server import _URL_FILE, RemoteAccessServer

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestStartAsyncPublishesRemoteUrl(IsolatedAsyncioTestCase):
    """``start_async`` should expose URL state just like ``start``."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self.work_dir = tempfile.mkdtemp()
        self._config_backup: str | None = None
        if CONFIG_PATH.is_file():
            self._config_backup = CONFIG_PATH.read_text()
        self._url_backup: bytes | None = None
        if _URL_FILE.is_file():
            self._url_backup = _URL_FILE.read_bytes()
        _URL_FILE.unlink(missing_ok=True)
        save_config({"remote_password": ""})
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            use_tunnel=False,
            work_dir=self.work_dir,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        shutil.rmtree(self.work_dir, ignore_errors=True)
        if self._config_backup is not None:
            CONFIG_PATH.write_text(self._config_backup)
        else:
            CONFIG_PATH.unlink(missing_ok=True)
        if self._url_backup is not None:
            _URL_FILE.parent.mkdir(parents=True, exist_ok=True)
            _URL_FILE.write_bytes(self._url_backup)
        else:
            _URL_FILE.unlink(missing_ok=True)

    async def test_start_async_writes_url_file_and_remote_url_event(self) -> None:
        """A real async server publishes local URL metadata and broadcasts it."""
        expected_url = f"https://localhost:{self.port}"
        self.assertTrue(_URL_FILE.is_file())
        self.assertEqual(json.loads(_URL_FILE.read_text())["local"], expected_url)
        self.assertEqual(self.server._active_url, expected_url)

        no_verify = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        no_verify.check_hostname = False
        no_verify.verify_mode = ssl.CERT_NONE
        async with connect(f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            auth = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
            self.assertEqual(auth["type"], "auth_ok")
            await ws.send(json.dumps({"type": "getWelcomeSuggestions"}))

            received: list[dict[str, Any]] = []
            for _ in range(3):
                try:
                    received.append(
                        json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                    )
                except TimeoutError:
                    break

        remote_urls = [ev.get("url") for ev in received if ev.get("type") == "remote_url"]
        self.assertEqual(remote_urls, [expected_url])


class TestMessageProtocolConsistency(unittest.TestCase):
    """The TypeScript protocol should include Python/web runtime events."""

    def test_to_webview_message_declares_runtime_events(self) -> None:
        """Events emitted by Python and handled by main.js are in types.ts."""
        text = (_VSCODE_DIR / "src" / "types.ts").read_text()
        for event_type in ["configData", "showWelcome", "usage_info", "merge_nav"]:
            self.assertIn(
                f"type: '{event_type}'",
                text,
                f"types.ts is missing the {event_type!r} webview event",
            )

    def test_user_action_done_routes_to_active_task_process(self) -> None:
        """Legacy userActionDone must target the active tab task process."""
        text = (_VSCODE_DIR / "src" / "SorcarSidebarView.ts").read_text()
        start = text.index("case 'userActionDone':")
        end = text.index("case 'recordFileUsage':", start)
        block = text[start:end]
        self.assertIn("doneTabId", block)
        self.assertIn("this._taskProcesses.get(doneTabId)", block)
        self.assertIn("tabId: doneTabId", block)

    def test_vscodeignore_excludes_local_generated_artifacts(self) -> None:
        """VSIX packaging should not include local finder/build artifacts."""
        ignore = (_VSCODE_DIR / ".vscodeignore").read_text()
        for pattern in [".DS_Store", "**/.DS_Store", "*.vsix"]:
            self.assertIn(pattern, ignore)
