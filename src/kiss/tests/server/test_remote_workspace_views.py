# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the Explorer and Source Control of a registered machine.

A real local daemon and a real remote daemon (:mod:`kiss.tests.remote_daemon`)
registered as machine ``box`` whose working directory is a git
repository, and a client on the local connection (whose own ``listDir`` /
``gitStatus`` / ``saveFile`` the daemon drops):

* every workspace command stamped ``machine`` (``listDir``,
  ``gitStatus``, ``gitLog``, ``gitShow``, ``gitAction``, ``fsAction``,
  ``openFile``, ``saveFile``) is answered by the machine's daemon --
  the machine's folders, repository and files, never this daemon's --
  with the reply stamped ``machine`` and the ``token`` / ``tabId``
  echoed;
* ``openFile`` answers with the browser's ``fileContent`` (the machine
  cannot open an editor here) and ``saveFile`` writes on the machine;
* an unknown machine answers with an ``error`` reply of the right type;
* a daemon link that presents the machine's local token from loopback
  (what an SSH machine's tunnel does) is not a local client there: its
  workspace commands are answered instead of dropped.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.test_explorer_scm_commands import _git


class TestRemoteWorkspaceViews(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon whose work dir is a repository."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        root = Path(self.tmpdir)
        self.box_work = root / "box-work"
        self.box_work.mkdir()
        _git(self.box_work, "init", "-q", "-b", "box-main")
        (self.box_work / "box-file.txt").write_text("on the box\n", newline="\n")
        (self.box_work / "box-dir").mkdir()
        _git(self.box_work, "add", "-A")
        _git(self.box_work, "commit", "-q", "-m", "box: first commit")
        self.box_sha = _git(self.box_work, "rev-parse", "HEAD").strip()
        (self.box_work / "box-file.txt").write_text("on the box, edited\n", newline="\n")
        # This daemon's own repository, which must never answer.
        self.home_work = root / "home-work"
        self.home_work.mkdir()
        _git(self.home_work, "init", "-q", "-b", "home-main")
        (self.home_work / "home-file.txt").write_text("at home\n", newline="\n")
        _git(self.home_work, "add", "-A")
        _git(self.home_work, "commit", "-q", "-m", "home: first commit")
        self.remote = await asyncio.to_thread(
            start_remote_daemon, root / "remote-home", password="pw-remote",
            config={"work_dir": str(self.box_work)},
        )
        certfile = root / "cert.pem"
        keyfile = root / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=root / "remote-url.json",
            local_endpoint_file=root / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)
        await self._ask({"type": "setWorkDir", "workDir": str(self.home_work)}, "workDirChanged")
        added = await self._ask({
            "type": "machineAction", "action": "add", "name": "box",
            "address": self.remote.url, "machinePassword": "pw-remote",
        }, "machines")
        self.assertEqual(added["error"], "")

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        self.remote.stop()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _ask(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* message, noting the others."""
        await self._send(cmd)
        self.skipped: list[dict[str, Any]] = []
        for _ in range(300):
            line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type:
                return msg
            self.skipped.append(msg)
        raise AssertionError(f"no {reply_type} reply")

    def _on_box(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """*cmd* as the remote webapp sends it for a tab bound to ``box``."""
        return {
            "workDir": str(self.box_work), "tabId": "tab-box", "token": "t-" + cmd["type"],
            "machine": "box", **cmd,
        }

    async def test_git_status_and_log_are_the_machines(self) -> None:
        status = await self._ask(self._on_box({"type": "gitStatus"}), "gitStatus")
        self.assertEqual(status.get("error", ""), "")
        self.assertEqual(status["machine"], "box")
        self.assertEqual(status["token"], "t-gitStatus")
        self.assertEqual(status["tabId"], "tab-box")
        self.assertEqual(status["branch"], "box-main")
        self.assertEqual([c["path"] for c in status["changes"]], ["box-file.txt"])
        log = await self._ask(self._on_box({"type": "gitLog", "limit": 10}), "gitLog")
        self.assertEqual(log.get("error", ""), "")
        self.assertEqual(log["machine"], "box")
        self.assertEqual([c["sha"] for c in log["commits"]], [self.box_sha])
        self.assertEqual(log["commits"][0]["subject"], "box: first commit")

    async def test_list_dir_and_fs_action_touch_the_machines_folder(self) -> None:
        listing = await self._ask(
            self._on_box({"type": "listDir", "path": str(self.box_work)}), "dirListing",
        )
        self.assertEqual(listing.get("error", ""), "")
        self.assertEqual(listing["machine"], "box")
        self.assertEqual(
            sorted(e["name"] for e in listing["entries"] if e["name"] != ".git"),
            ["box-dir", "box-file.txt"],
        )
        created = await self._ask(self._on_box({
            "type": "fsAction", "action": "newFile",
            "path": str(self.box_work / "box-dir"), "name": "made-here.txt",
        }), "fsResult")
        self.assertEqual(created.get("error", ""), "")
        self.assertEqual(created["machine"], "box")
        self.assertEqual(created["token"], "t-fsAction")
        self.assertTrue((self.box_work / "box-dir" / "made-here.txt").is_file())
        self.assertFalse((self.home_work / "box-dir").exists())

    async def test_open_file_and_save_file_read_and_write_on_the_machine(self) -> None:
        opened = await self._ask(self._on_box({
            "type": "openFile", "path": "box-file.txt",
        }), "fileContent")
        self.assertEqual(opened.get("error", ""), "")
        self.assertEqual(opened["machine"], "box")
        self.assertEqual(opened["tabId"], "tab-box")
        self.assertEqual(opened["content"], "on the box, edited\n")
        saved = await self._ask(self._on_box({
            "type": "saveFile", "path": opened["path"], "content": "saved from afar\n",
            "version": opened["version"],
        }), "fileSaved")
        self.assertTrue(saved["ok"], saved)
        self.assertEqual(saved["machine"], "box")
        self.assertEqual(saved["token"], "t-saveFile")
        self.assertEqual((self.box_work / "box-file.txt").read_text(), "saved from afar\n")

    async def test_big_html_page_on_the_machine_arrives_in_pieces(self) -> None:
        """An HTML page of the machine bigger than one ``fileContent``
        frame comes back whole: the machine's daemon sends it in pieces
        (``more`` on every one but the last), the link joins them, and
        this daemon relays the reply to its client in pieces again,
        each stamped ``machine``, ``tabId`` and ``token``."""
        from kiss.server import web_server

        chunk = web_server._FILE_CONTENT_CHUNK_CHARS
        filler = "<p>" + "z" * 1000 + "</p>\n"
        text = (
            "<!DOCTYPE html><html><body>\n"
            + filler * (chunk // len(filler) + 1)
            + "</body></html>\n"
        )
        # ``newline=""``: the daemon serves the file's bytes as they are,
        # so the text must not pick up ``\r\n`` on Windows.
        (self.box_work / "box-page.html").write_text(text, encoding="utf-8", newline="")
        await self._send(self._on_box({"type": "openFile", "path": "box-page.html"}))
        pieces: list[dict[str, Any]] = []
        while True:
            line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") != "fileContent":
                continue
            pieces.append(msg)
            if msg.get("more") is not True:
                break
        self.assertEqual(len(pieces), 2)
        self.assertEqual(pieces[0].get("error", ""), "")
        self.assertEqual(pieces[0]["name"], "box-page.html")
        self.assertNotIn("version", pieces[0])  # past the editor cap: read-only
        for piece in pieces:
            self.assertEqual(piece["machine"], "box")
            self.assertEqual(piece["tabId"], "tab-box")
            self.assertEqual(piece["token"], "t-openFile")
            self.assertEqual(piece["path"], pieces[0]["path"])
        self.assertEqual(len(pieces[0]["content"]), chunk)
        self.assertEqual("".join(p["content"] for p in pieces), text)

    async def test_git_show_and_git_action_run_on_the_machine(self) -> None:
        shown = await self._ask(self._on_box({
            "type": "gitShow", "sha": self.box_sha, "mode": "patch",
        }), "gitShow")
        self.assertEqual(shown.get("error", ""), "")
        self.assertEqual(shown["machine"], "box")
        self.assertIn("box: first commit", shown["text"])
        self.assertIn("box-file.txt", shown["text"])
        acted = await self._ask(self._on_box({
            "type": "gitAction", "action": "createBranch", "sha": self.box_sha,
            "name": "from-afar",
        }), "gitActionResult")
        self.assertEqual(acted.get("error", ""), "")
        self.assertEqual(acted["machine"], "box")
        self.assertTrue(acted["ok"])
        self.assertIn("from-afar", _git(self.box_work, "branch", "--list", "from-afar"))
        self.assertEqual(_git(self.home_work, "branch", "--list", "from-afar"), "")

    async def test_unknown_machine_answers_with_an_error_of_the_right_type(self) -> None:
        for cmd_type, reply_type in (
            ("gitStatus", "gitStatus"), ("fsAction", "fsResult"), ("openFile", "fileContent"),
        ):
            cmd = self._on_box({"type": cmd_type, "action": "newFile", "path": "x", "name": "y"})
            cmd["machine"] = "nowhere"
            reply = await self._ask(cmd, reply_type)
            self.assertEqual(reply["machine"], "nowhere")
            self.assertEqual(reply["token"], "t-" + cmd_type)
            self.assertEqual(reply["tabId"], "tab-box")
            self.assertIn("no machine named 'nowhere'", reply["error"])

    async def test_a_daemon_link_with_the_local_token_is_not_a_local_client(self) -> None:
        # An SSH machine's link reaches the daemon through the tunnel,
        # from loopback, with the daemon's local token: were it a local
        # client, the daemon would drop the Explorer's requests and the
        # views would stay empty.
        endpoint = local_endpoint.read_endpoint(
            self.remote.home / local_endpoint.LOCAL_ENDPOINT_FILE,
        )
        assert endpoint is not None
        ws = await connect(
            endpoint.url, ssl=local_endpoint.client_ssl_context(endpoint),
            open_timeout=10.0, close_timeout=2.0, compression=None,
        )
        try:
            await ws.send(json.dumps({"type": "auth", "token": endpoint.token, "link": True}))
            self.assertEqual(json.loads(await ws.recv())["type"], "auth_ok")
            await ws.send(json.dumps({
                "type": "listDir", "path": str(self.box_work), "token": "link-1",
            }))
            await ws.send(json.dumps({"type": "gitStatus", "token": "link-2"}))
            await ws.send(json.dumps({"type": "openFile", "path": "box-file.txt"}))
            replies: dict[str, dict[str, Any]] = {}
            while len(replies) < 3:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 60.0))
                if msg.get("type") in ("dirListing", "gitStatus", "fileContent"):
                    replies[msg["type"]] = msg
        finally:
            await ws.close()
        self.assertEqual(replies["dirListing"]["token"], "link-1")
        self.assertEqual(replies["gitStatus"]["branch"], "box-main")
        self.assertEqual(replies["fileContent"]["content"], "on the box, edited\n")
