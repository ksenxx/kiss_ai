# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a malformed ``workDir`` never escapes the connection's pinned folder.

``ServerApi.dispatch`` stamps the connection's pinned ``work_dir``
onto every command "lacking" a ``workDir`` — but the check was
``not cmd.get("workDir")``, so a truthy non-string value (``123``,
``["x"]``, ``{"x": 1}``) survived dispatch, was blanked by the
handler's ``_cmd_str`` and then fell back to the DAEMON-GLOBAL work
dir: another window's folder.  Dispatch must treat a missing, empty
or non-string ``workDir`` alike and stamp the pin.

Each test opens two real ``wss://`` connections: window A pins itself
to directory A with ``setWorkDir``; window B then pins itself to
directory B, which also moves the daemon-global fallback to B (the
last ``setWorkDir`` wins there).  Window A's ``openFile`` /
``checkPaths`` / ``ready`` with a malformed ``workDir`` must still
operate in A.
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


class TestPinnedWorkDirBeatsMalformedField(IsolatedAsyncioTestCase):
    """Non-string ``workDir`` resolves to the pin, never the global dir."""

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
        (self.dir_a / "only-in-a.txt").write_text("A\n", encoding="utf-8")
        (self.dir_b / "only-in-b.txt").write_text("B\n", encoding="utf-8")
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

    async def _pinned_roundtrip(
        self, payloads: list[dict[str, Any]], reply_type: str,
    ) -> dict[str, Any]:
        """Pin window A to A, window B to B (moving the global fallback
        to B), then send *payloads* from A and return A's first reply
        of *reply_type*."""
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
            # Commands are sequential PER CONNECTION only, so a reply on
            # B's socket says nothing about A's progress: A's setWorkDir
            # must be known complete — through an A-side probe reply —
            # before B's setWorkDir may move the global fallback, or A
            # could still move it back to A afterwards.
            await ws_a.send(json.dumps(
                {"type": "checkPaths", "paths": ["only-in-a.txt"], "tabId": "a"},
            ))
            echo_a = await self._recv_type(ws_a, "pathsExist")
            self.assertEqual(echo_a["workDir"], str(self.dir_a))
            # Window B's pin is complete once its own (unstamped)
            # checkPaths echoes B: the global fallback is now B.
            await ws_b.send(json.dumps(
                {"type": "setWorkDir", "workDir": str(self.dir_b)},
            ))
            await ws_b.send(json.dumps(
                {"type": "checkPaths", "paths": ["only-in-b.txt"], "tabId": "b"},
            ))
            echo_b = await self._recv_type(ws_b, "pathsExist")
            self.assertEqual(echo_b["workDir"], str(self.dir_b))
            self.assertEqual(
                self.server._vscode_server.work_dir, str(self.dir_b),
            )
            for payload in payloads:
                await ws_a.send(json.dumps(payload))
            return await self._recv_type(ws_a, reply_type)

    async def test_open_file_uses_pin_for_every_malformed_work_dir(self) -> None:
        for bad in MALFORMED_WORK_DIRS:
            with self.subTest(work_dir=bad):
                reply = await self._pinned_roundtrip(
                    [{"type": "openFile", "path": "only-in-a.txt",
                      "workDir": bad, "tabId": "t"}],
                    "fileContent",
                )
                self.assertEqual(reply["content"], "A\n")
                self.assertEqual(
                    reply["path"], str(self.dir_a / "only-in-a.txt"),
                )

    async def test_check_paths_echoes_pin_for_malformed_work_dir(self) -> None:
        reply = await self._pinned_roundtrip(
            [{"type": "checkPaths", "paths": ["only-in-a.txt", "only-in-b.txt"],
              "workDir": 123, "tabId": "t"}],
            "pathsExist",
        )
        self.assertEqual(
            reply["results"], {"only-in-a.txt": True, "only-in-b.txt": False},
        )
        self.assertEqual(reply["workDir"], str(self.dir_a))

    async def test_ready_reports_pin_for_malformed_work_dir(self) -> None:
        # ``ready`` fans out into ``getConfig`` whose reply names the
        # work dir the window will run tasks in.
        reply = await self._pinned_roundtrip(
            [{"type": "ready", "workDir": ["x"], "tabId": "t"}],
            "configData",
        )
        self.assertEqual(reply["config"]["work_dir"], str(self.dir_a))

    async def test_missing_and_empty_work_dir_still_use_pin(self) -> None:
        # The two cases the old ``not cmd.get("workDir")`` guard already
        # handled must keep working after the guard is generalised.
        for payload in (
            {"type": "checkPaths", "paths": ["only-in-a.txt"], "tabId": "t"},
            {"type": "checkPaths", "paths": ["only-in-a.txt"], "workDir": "",
             "tabId": "t"},
        ):
            with self.subTest(payload=payload):
                reply = await self._pinned_roundtrip([payload], "pathsExist")
                self.assertEqual(reply["workDir"], str(self.dir_a))

    async def test_malformed_set_work_dir_does_not_move_the_pin(self) -> None:
        reply = await self._pinned_roundtrip(
            [
                {"type": "setWorkDir", "workDir": 5},
                {"type": "checkPaths", "paths": ["only-in-a.txt"], "tabId": "t"},
            ],
            "pathsExist",
        )
        self.assertEqual(reply["workDir"], str(self.dir_a))

    async def test_explicit_string_work_dir_wins_over_pin(self) -> None:
        reply = await self._pinned_roundtrip(
            [{"type": "checkPaths", "paths": ["only-in-b.txt"],
              "workDir": str(self.dir_b), "tabId": "t"}],
            "pathsExist",
        )
        self.assertEqual(reply["results"], {"only-in-b.txt": True})
        self.assertEqual(reply["workDir"], str(self.dir_b))
