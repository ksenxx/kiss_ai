# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the file links of a tab bound to a registered machine.

A real local daemon and a real remote daemon (:mod:`kiss.tests.remote_daemon`)
registered as machine ``box``, and a client on the local connection:

* a ``checkPaths`` stamped ``machine`` is answered by the machine's
  daemon against the machine's files (a file that exists only there
  is reported present, one that exists only here absent), with the
  ``pathsExist`` reply stamped ``machine`` and the ``tabId`` /
  ``workDir`` echoed;
* an ``openFile`` stamped ``machine`` (a click on such a link) answers
  with the machine's file content;
* a ``checkPaths`` without ``machine`` is answered here, so the same
  paths resolve the other way round;
* an unknown machine answers with a ``pathsExist`` carrying ``error``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon


class TestRemoteCheckPaths(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon with a file of its own."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        root = Path(self.tmpdir)
        self.box_work = root / "box-work"
        self.box_work.mkdir()
        (self.box_work / "box-file.txt").write_text("on the box\n", newline="\n")
        self.home_work = root / "home-work"
        self.home_work.mkdir()
        (self.home_work / "home-file.txt").write_text("at home\n", newline="\n")
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

    async def _ask(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* message."""
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()
        for _ in range(300):
            line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    def _check(self, machine: str | None, work_dir: Path) -> dict[str, Any]:
        cmd: dict[str, Any] = {
            "type": "checkPaths", "paths": ["box-file.txt", "home-file.txt"],
            "workDir": str(work_dir), "tabId": "tab-box",
        }
        if machine is not None:
            cmd["machine"] = machine
        return cmd

    async def test_check_paths_of_a_bound_tab_are_the_machines(self) -> None:
        reply = await self._ask(self._check("box", self.box_work), "pathsExist")
        self.assertEqual(reply.get("error", ""), "")
        self.assertEqual(reply["machine"], "box")
        self.assertEqual(reply["tabId"], "tab-box")
        self.assertEqual(reply["workDir"], str(self.box_work))
        self.assertEqual(
            reply["results"], {"box-file.txt": True, "home-file.txt": False},
        )
        opened = await self._ask({
            "type": "openFile", "path": "box-file.txt", "workDir": str(self.box_work),
            "tabId": "tab-box", "machine": "box",
        }, "fileContent")
        self.assertEqual(opened.get("error", ""), "")
        self.assertEqual(opened["machine"], "box")
        self.assertEqual(opened["content"], "on the box\n")

    async def test_check_paths_without_a_machine_are_answered_here(self) -> None:
        reply = await self._ask(self._check(None, self.home_work), "pathsExist")
        self.assertNotIn("machine", reply)
        self.assertEqual(
            reply["results"], {"box-file.txt": False, "home-file.txt": True},
        )

    async def test_unknown_machine_answers_with_an_error(self) -> None:
        reply = await self._ask(self._check("nowhere", self.box_work), "pathsExist")
        self.assertEqual(reply["machine"], "nowhere")
        self.assertEqual(reply["tabId"], "tab-box")
        self.assertIn("no machine named 'nowhere'", reply["error"])
