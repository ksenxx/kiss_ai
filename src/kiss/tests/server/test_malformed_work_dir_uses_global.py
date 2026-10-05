# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a malformed ``workDir`` resolves to the global working directory.

The working directory is one daemon-wide value (the last ``setWorkDir``
from any connection).  A command whose ``workDir`` is missing, empty or
a non-string (``123``, ``["x"]``, ``{"x": 1}``, ``True``) must resolve
against that global directory — never crash, never leak the raw value —
while an explicit string ``workDir`` still wins, and a malformed
``setWorkDir`` must not move the global directory.

Each test opens two real ``wss://`` connections: window A picks
directory A with ``setWorkDir``; window B then picks directory B, which
makes B the global directory for everyone.  Window A's ``openFile`` /
``checkPaths`` / ``ready`` with a malformed ``workDir`` therefore
operate in B.
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


MALFORMED_WORK_DIRS: tuple[Any, ...] = (123, ["x"], {"x": 1}, True)


class TestMalformedWorkDirUsesGlobal(IsolatedAsyncioTestCase):
    """Non-string ``workDir`` resolves to the global dir, never leaks."""

    async def asyncSetUp(self) -> None:
        # resolve(): the server replies with resolved paths, and on macOS
        # mkdtemp returns /var/... which is a symlink to /private/var/...
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-pinned-wd-")).resolve()
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        self.dir_a = self.tmpdir / "window-a"
        self.dir_b = self.tmpdir / "window-b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()
        # newline="\n": the reply must echo the bytes on disk, so the
        # fixture must not let Windows text mode turn "\n" into "\r\n".
        (self.dir_a / "only-in-a.txt").write_text("A\n", encoding="utf-8", newline="\n")
        (self.dir_b / "only-in-b.txt").write_text("B\n", encoding="utf-8", newline="\n")
        certfile, keyfile = self.tmpdir / "cert.pem", self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            local_endpoint_file=self.tmpdir / "sorcar-local.json",
            work_dir=str(self.dir_b),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _auth(self, ws: Any) -> None:
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == "auth_ok":
                return

    @staticmethod
    async def _recv_type(ws: Any, reply_type: str) -> dict[str, Any]:
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == reply_type:
                return dict(msg)

    async def _global_roundtrip(
        self, payloads: list[dict[str, Any]], reply_type: str,
    ) -> dict[str, Any]:
        """Window A picks A, then window B picks B (the global directory
        is now B for both), then send *payloads* from A and return A's
        first reply of *reply_type*."""
        url = f"wss://127.0.0.1:{self.port}/ws"
        async with (
            connect(url, ssl=_no_verify_ssl()) as ws_a,
            connect(url, ssl=_no_verify_ssl()) as ws_b,
        ):
            await self._auth(ws_a)
            await self._auth(ws_b)
            await ws_a.send(json.dumps(
                {"type": "setWorkDir", "workDir": str(self.dir_a)},
            ))
            # Commands are sequential PER CONNECTION only: A's pick must
            # be known complete (an A-side probe resolves against A)
            # before B's pick, or A could move the global back afterwards.
            await ws_a.send(json.dumps(
                {"type": "checkPaths", "paths": ["only-in-a.txt"], "tabId": "a"},
            ))
            probe_a = await self._recv_type(ws_a, "pathsExist")
            self.assertEqual(probe_a["results"], {"only-in-a.txt": True})
            await ws_b.send(json.dumps(
                {"type": "setWorkDir", "workDir": str(self.dir_b)},
            ))
            await ws_b.send(json.dumps(
                {"type": "checkPaths", "paths": ["only-in-b.txt"], "tabId": "b"},
            ))
            probe_b = await self._recv_type(ws_b, "pathsExist")
            self.assertEqual(probe_b["results"], {"only-in-b.txt": True})
            self.assertEqual(
                self.server._vscode_server.work_dir, str(self.dir_b),
            )
            for payload in payloads:
                await ws_a.send(json.dumps(payload))
            return await self._recv_type(ws_a, reply_type)

    async def test_open_file_uses_global_for_every_malformed_work_dir(
        self,
    ) -> None:
        for bad in MALFORMED_WORK_DIRS:
            with self.subTest(work_dir=bad):
                reply = await self._global_roundtrip(
                    [{"type": "openFile", "path": "only-in-b.txt",
                      "workDir": bad, "tabId": "t"}],
                    "fileContent",
                )
                self.assertEqual(reply["content"], "B\n")
                self.assertEqual(
                    reply["path"], str(self.dir_b / "only-in-b.txt"),
                )

    async def test_check_paths_uses_global_for_malformed_work_dir(self) -> None:
        reply = await self._global_roundtrip(
            [{"type": "checkPaths", "paths": ["only-in-a.txt", "only-in-b.txt"],
              "workDir": 123, "tabId": "t"}],
            "pathsExist",
        )
        self.assertEqual(
            reply["results"], {"only-in-a.txt": False, "only-in-b.txt": True},
        )
        # The echo is the client's key (a non-string counts as none), not
        # the directory the paths were resolved against.
        self.assertEqual(reply["workDir"], "")

    async def test_ready_reports_global_for_malformed_work_dir(self) -> None:
        # ``ready`` fans out into ``getConfig`` whose reply names the
        # work dir every window runs tasks in.
        reply = await self._global_roundtrip(
            [{"type": "ready", "workDir": ["x"], "tabId": "t"}],
            "configData",
        )
        self.assertEqual(reply["config"]["work_dir"], str(self.dir_b))

    async def test_missing_and_empty_work_dir_use_global(self) -> None:
        for payload in (
            {"type": "checkPaths", "paths": ["only-in-b.txt"], "tabId": "t"},
            {"type": "checkPaths", "paths": ["only-in-b.txt"], "workDir": "",
             "tabId": "t"},
        ):
            with self.subTest(payload=payload):
                reply = await self._global_roundtrip([payload], "pathsExist")
                self.assertEqual(reply["results"], {"only-in-b.txt": True})
                self.assertEqual(reply["workDir"], "")

    async def test_malformed_set_work_dir_does_not_move_the_global(
        self,
    ) -> None:
        reply = await self._global_roundtrip(
            [
                {"type": "setWorkDir", "workDir": 5},
                {"type": "checkPaths", "paths": ["only-in-b.txt"], "tabId": "t"},
            ],
            "pathsExist",
        )
        self.assertEqual(reply["results"], {"only-in-b.txt": True})
        self.assertEqual(self.server._vscode_server.work_dir, str(self.dir_b))

    async def test_explicit_string_work_dir_wins_over_global(self) -> None:
        reply = await self._global_roundtrip(
            [{"type": "checkPaths", "paths": ["only-in-a.txt"],
              "workDir": str(self.dir_a), "tabId": "t"}],
            "pathsExist",
        )
        self.assertEqual(reply["results"], {"only-in-a.txt": True})
        self.assertEqual(reply["workDir"], str(self.dir_a))
