# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the working directories of a registered machine.

A real local daemon and a real remote daemon (:mod:`kiss.tests.remote_daemon`)
registered as machine ``box``, and a client on the local connection:

* ``getMachineWorkDirs`` answers with the machine's working directory
  and the directories selected there so far (those that still exist);
* ``setWorkDir`` with ``machine`` makes the directory the machine's
  working directory -- persisted there, listed first there -- and
  leaves this daemon's own working directory alone (no
  ``workDirChanged`` here);
* ``listDir`` with ``machine`` lists the machine's folder, even from a
  local connection, with the reply stamped ``machine``;
* an unknown or unreachable machine answers with ``error`` on every
  one of them;
* ``getMachines`` notices a machine registered behind the daemon's
  back (``/machine add``, the CLI) without a ``refresh``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon


class TestRemoteWorkDirs(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon seeded with work dirs, one client."""

    async def asyncSetUp(self) -> None:
        # realpath: ``listDir`` replies with the resolved directory, and
        # macOS's ``/var/folders`` is a symlink to ``/private/var/...``.
        self.tmpdir = os.path.realpath(tempfile.mkdtemp())
        mach.save_machines([])
        root = Path(self.tmpdir)
        self.box_work = root / "box-work"
        self.box_other = root / "box-other"
        (self.box_work / "sub-a").mkdir(parents=True)
        (self.box_work / "sub-b").mkdir()
        self.box_other.mkdir()
        self.home_work = root / "home-work"
        self.home_work.mkdir()
        now = time.time()
        self.remote = await asyncio.to_thread(
            start_remote_daemon, root / "remote-home", password="pw-remote",
            config={
                "work_dir": str(self.box_work),
                "recent_work_dirs": [
                    {"path": str(self.box_work), "ts": now},
                    {"path": str(self.box_other), "ts": now - 60},
                    {"path": str(root / "box-gone"), "ts": now - 120},
                ],
            },
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

    async def _next(self, timeout: float = 60.0) -> dict[str, Any]:
        line = await asyncio.wait_for(self.reader.readline(), timeout=timeout)
        assert line, "local connection closed"
        msg: dict[str, Any] = json.loads(line.decode("utf-8"))
        return msg

    async def _ask(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* message, noting the others."""
        await self._send(cmd)
        self.skipped: list[dict[str, Any]] = []
        for _ in range(300):
            msg = await self._next()
            if msg.get("type") == reply_type:
                return msg
            self.skipped.append(msg)
        raise AssertionError(f"no {reply_type} reply")

    def _remote_config(self) -> dict[str, Any]:
        text = (Path(self.tmpdir) / "remote-home" / "config.json").read_text(encoding="utf-8")
        config: dict[str, Any] = json.loads(text)
        return config

    async def test_get_machine_work_dirs_reports_the_machines_directories(self) -> None:
        reply = await self._ask({"type": "getMachineWorkDirs", "machine": "box"}, "machineWorkDirs")
        self.assertEqual(reply["error"], "")
        self.assertEqual(reply["machine"], "box")
        self.assertEqual(reply["workDir"], str(self.box_work))
        # Most recent first; the directory that no longer exists is left out.
        self.assertEqual(
            [row["path"] for row in reply["recentWorkDirs"]],
            [str(self.box_work), str(self.box_other)],
        )
        self.assertNotIn("set", reply)

    async def test_set_work_dir_with_machine_adopts_on_the_machine_only(self) -> None:
        reply = await self._ask({
            "type": "setWorkDir", "workDir": str(self.box_other), "machine": "box",
        }, "machineWorkDirs")
        self.assertEqual(reply["error"], "")
        self.assertTrue(reply["set"])
        self.assertEqual(reply["workDir"], str(self.box_other))
        self.assertEqual(
            [row["path"] for row in reply["recentWorkDirs"]],
            [str(self.box_other), str(self.box_work)],
        )
        # The machine persisted it; this daemon's own directory is untouched
        # and no client here was told to re-scope.
        self.assertEqual(self._remote_config()["work_dir"], str(self.box_other))
        self.assertNotIn("workDirChanged", [m.get("type") for m in self.skipped])
        config = await self._ask({"type": "getConfig"}, "configData")
        self.assertEqual(config["config"]["work_dir"], str(self.home_work))
        self.assertEqual(
            [row["path"] for row in config["config"]["recent_work_dirs"]],
            [str(self.home_work)],
        )
        # A later read agrees with the pick.
        again = await self._ask({"type": "getMachineWorkDirs", "machine": "box"}, "machineWorkDirs")
        self.assertEqual(again["workDir"], str(self.box_other))

    async def test_set_work_dir_without_machine_still_adopts_here(self) -> None:
        other = Path(self.tmpdir) / "home-other"
        other.mkdir()
        changed = await self._ask({"type": "setWorkDir", "workDir": str(other)}, "workDirChanged")
        self.assertEqual(changed["workDir"], str(other))
        self.assertEqual(self._remote_config()["work_dir"], str(self.box_work))

    async def test_list_dir_with_machine_lists_the_machines_folder(self) -> None:
        # The connection is a local peer, whose own listDir the
        # daemon drops: a machine's listing is answered all the same.
        reply = await self._ask({
            "type": "listDir", "path": str(self.box_work), "token": "picker:7",
            "machine": "box", "workDir": str(self.home_work), "tabId": "tab-1",
        }, "dirListing")
        self.assertEqual(reply.get("error", ""), "")
        self.assertEqual(reply["machine"], "box")
        self.assertEqual(reply["token"], "picker:7")
        self.assertEqual(reply["path"], str(self.box_work))
        self.assertEqual(
            sorted(e["name"] for e in reply["entries"] if e["isDir"]), ["sub-a", "sub-b"],
        )
        missing = await self._ask({
            "type": "listDir", "path": str(self.box_work / "nope"), "token": "workdir:3",
            "machine": "box",
        }, "dirListing")
        self.assertEqual(missing["machine"], "box")
        self.assertEqual(missing["token"], "workdir:3")
        self.assertTrue(missing["error"])

    async def test_unknown_machine_is_an_error(self) -> None:
        dirs = await self._ask({"type": "getMachineWorkDirs", "machine": "nope"}, "machineWorkDirs")
        self.assertIn("no machine named 'nope'", dirs["error"])
        self.assertEqual(dirs["workDir"], "")
        self.assertEqual(dirs["recentWorkDirs"], [])
        picked = await self._ask({
            "type": "setWorkDir", "workDir": str(self.box_other), "machine": "nope",
        }, "machineWorkDirs")
        self.assertIn("no machine named 'nope'", picked["error"])
        self.assertTrue(picked["set"])
        self.assertEqual(self._remote_config()["work_dir"], str(self.box_work))
        listing = await self._ask({
            "type": "listDir", "path": str(self.box_work), "token": "picker:1", "machine": "nope",
        }, "dirListing")
        self.assertIn("no machine named 'nope'", listing["error"])
        self.assertEqual(listing["machine"], "nope")
        self.assertEqual(listing["token"], "picker:1")
        self.assertEqual(listing["entries"], [])

    async def test_unreachable_machine_is_an_error(self) -> None:
        self.remote.stop()
        dirs = await self._ask({"type": "getMachineWorkDirs", "machine": "box"}, "machineWorkDirs")
        self.assertTrue(dirs["error"])
        self.assertEqual(dirs["machine"], "box")
        listing = await self._ask({
            "type": "listDir", "path": str(self.box_work), "token": "picker:2", "machine": "box",
        }, "dirListing")
        self.assertTrue(listing["error"])
        self.assertEqual(listing["entries"], [])

    async def test_get_machines_notices_a_registry_change_without_refresh(self) -> None:
        before = await self._ask({"type": "getMachines"}, "machines")
        self.assertEqual([m["name"] for m in before["machines"] if not m["local"]], ["box"])
        # Registered behind the daemon's back, as `/machine add` or the
        # CLI does: the cached report is stale and is rebuilt.
        await asyncio.to_thread(mach.add_machine, "box2", self.remote.url, "pw-remote")
        after = await self._machines_listing(["box", "box2"])
        self.assertEqual(after["action"], "")
        await asyncio.to_thread(mach.remove_machine, "box2")
        await self._machines_listing(["box"])

    async def _machines_listing(self, expected: list[str]) -> dict[str, Any]:
        """Ask ``getMachines`` and return the reply naming *expected* remote machines.

        The daemon's registry watcher also pushes an unsolicited
        ``machines`` frame (pinged from the registry as it was when the
        watcher woke) after every registration; such a frame about the
        previous state may land before the direct reply, so the frames
        are read until one lists *expected*.
        """
        await self._send({"type": "getMachines"})
        seen: list[list[str]] = []
        for _ in range(300):
            msg = await self._next()
            if msg.get("type") != "machines":
                continue
            names = [m["name"] for m in msg["machines"] if not m["local"]]
            if names == expected:
                return msg
            seen.append(names)
            if len(seen) > 2:
                break
        raise AssertionError(f"no machines reply listing {expected}; got {seen}")
