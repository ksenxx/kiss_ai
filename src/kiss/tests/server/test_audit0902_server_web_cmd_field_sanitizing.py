# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the file handlers sanitise ``workDir`` / ``tabId`` / ``title`` alike.

``openFile``, ``shareChat``, ``shareChatTasks``, ``checkPaths`` and
``ready`` each carried a private copy of the same "blank a non-string
field, fall back to the daemon work dir" block.  The copies now share
``RemoteAccessServer._cmd_str`` / ``_cmd_work_dir``; these tests drive
every handler over a real ``wss://`` connection with malformed field
types and pin the behaviour the copies used to implement one by one,
including every fallback arm of the work-dir resolution.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.core.vscode_config as vc
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestCommandFieldSanitizing(IsolatedAsyncioTestCase):
    """Malformed client fields never reach path/reply construction."""

    async def asyncSetUp(self) -> None:
        # resolve(): the server replies with resolved paths, and on macOS
        # mkdtemp returns /var/... which is a symlink to /private/var/...
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-cmd-sanitize-")).resolve()
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        self.work_dir = self.tmpdir / "repo"
        self.work_dir.mkdir()
        (self.work_dir / "hello.txt").write_text("hi\n", encoding="utf-8")
        certfile, keyfile = self.tmpdir / "cert.pem", self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            uds_path=self.tmpdir / "sorcar.sock",
            work_dir=str(self.work_dir),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _roundtrip(
        self, payloads: list[dict[str, Any]], reply_type: str,
    ) -> dict[str, Any]:
        """Authenticate, send *payloads* in order, return the first
        reply of *reply_type*."""
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if msg.get("type") == "auth_ok":
                    break
            for payload in payloads:
                await ws.send(json.dumps(payload))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if msg.get("type") == reply_type:
                    return dict(msg)

    async def test_open_file_non_string_workdir_and_tabid(self) -> None:
        reply = await self._roundtrip(
            [{"type": "openFile", "path": "hello.txt", "workDir": 42,
              "tabId": 7}],
            "fileContent",
        )
        self.assertEqual(reply["content"], "hi\n")
        self.assertEqual(reply["path"], str(self.work_dir / "hello.txt"))
        self.assertEqual(reply["tabId"], "")

    async def test_open_file_non_string_path_is_ignored(self) -> None:
        # The malformed openFile yields no reply; the follow-up
        # checkPaths reply is the first one the client sees.
        reply = await self._roundtrip(
            [
                {"type": "openFile", "path": ["hello.txt"], "tabId": "t"},
                {"type": "checkPaths", "paths": ["hello.txt"], "tabId": "t"},
            ],
            "pathsExist",
        )
        self.assertEqual(reply["results"], {"hello.txt": True})

    async def test_open_file_falls_back_to_server_work_dir(self) -> None:
        # Backend work dir cleared: the third fallback arm (this
        # server's own work_dir) must still resolve relative paths.
        self.server._vscode_server.work_dir = ""
        reply = await self._roundtrip(
            [{"type": "openFile", "path": "hello.txt", "tabId": "t"}],
            "fileContent",
        )
        self.assertEqual(reply["content"], "hi\n")

    async def test_check_paths_echoes_blank_for_non_string_workdir(self) -> None:
        reply = await self._roundtrip(
            [{"type": "checkPaths", "paths": ["hello.txt", 3, ""],
              "workDir": ["x"], "tabId": None}],
            "pathsExist",
        )
        self.assertEqual(reply["results"], {"hello.txt": True})
        self.assertEqual(reply["workDir"], "")
        self.assertEqual(reply["tabId"], "")

    async def test_check_paths_non_list_paths_yield_empty_results(self) -> None:
        reply = await self._roundtrip(
            [{"type": "checkPaths", "paths": "hello.txt", "tabId": "t"}],
            "pathsExist",
        )
        self.assertEqual(reply["results"], {})
        self.assertEqual(reply["tabId"], "t")

    async def test_check_paths_explicit_workdir_wins(self) -> None:
        other = self.tmpdir / "other"
        other.mkdir()
        (other / "only-here.txt").write_text("x", encoding="utf-8")
        reply = await self._roundtrip(
            [{"type": "checkPaths", "paths": ["only-here.txt"],
              "workDir": str(other), "tabId": "t"}],
            "pathsExist",
        )
        self.assertEqual(reply["results"], {"only-here.txt": True})
        self.assertEqual(reply["workDir"], str(other))

    async def test_share_chat_tasks_non_string_ids(self) -> None:
        reply = await self._roundtrip(
            [{"type": "shareChatTasks", "tabId": 5, "chatId": {"a": 1}}],
            "share_tasks",
        )
        self.assertEqual(reply["tabId"], "")
        self.assertEqual(reply["chatId"], "")
        self.assertEqual(reply["tasks"], [])
        self.assertFalse(reply["truncated"])

    async def test_share_chat_non_string_title_and_workdir(self) -> None:
        reply = await self._roundtrip(
            [{"type": "shareChat", "chatId": "c1", "html": "<p>x</p>",
              "title": 12, "workDir": 0, "tabId": 9}],
            "share_done",
        )
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(reply["tabId"], "")
        out = Path(reply["path"])
        self.assertEqual(out.parent, self.work_dir / "reports")
        self.assertIn("<title>KISS Sorcar chat</title>", out.read_text())
