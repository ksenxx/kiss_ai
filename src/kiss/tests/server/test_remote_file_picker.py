# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the ``@`` file picker of a tab whose tasks run on another machine.

Regression: a chat tab bound to a registered machine (the user's laptop
page driving a cloud VM) typed ``@`` and saw no file menu at all.  The
page's ``getFiles`` was answered by the laptop's own daemon, which
searched the laptop's disk for the VM's folder and found nothing, and
an empty ``files`` reply hides the picker.

A real local daemon and a real remote daemon
(:mod:`kiss.tests.remote_daemon`) registered as machine ``box``, and a
client on the local connection:

* a ``getFiles`` stamped ``machine`` lists the machine's files, even on
  the machine's first scan of its folder (it answers with an empty
  ``loading`` list at first; the forward waits for the real list);
* a ``recordFileUsage`` stamped ``machine`` is counted on the machine
  (its next list ranks the file ``frequent``) and not here;
* a ``getFiles`` without ``machine`` is answered here;
* an unknown machine answers with an empty ``files`` reply carrying
  ``error`` and echoing ``prefix`` and ``tabId``, so the page can match
  it to the mention being typed;
* a machine that never answers (a loopback listener that never
  completes the TLS handshake, as in
  ``test_remote_bind_does_not_block_connection.py``) holds up only its
  own file list: the connection's next command and another machine's
  list are answered meanwhile.

Not covered, as no real daemon can be made to do it on cue and a test
double is not allowed here:

* a machine whose first scan outlasts the forward's wait
  (:data:`kiss.server.sidebar_panels.FILES_INDEX_WAIT`) gets the empty
  ``loading`` reply, which would need a scan a minute long (the page's
  asking again on it is covered by
  ``src/kiss/webapp/test/atPickerOnPickedMachine.test.js``);
* a forward asks past a list echoing another ``prefix`` or ``tabId``
  (an earlier request's list, sent by the machine's scan after that
  request returned ``loading``), which needs that list to arrive in
  the instant between the link draining stale frames and reading the
  reply.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
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


def _register_silent_machine(name: str, certfile: Path) -> socket.socket:
    """Register machine *name* at a loopback listener that never completes the TLS handshake.

    Returns:
        The listener, for the caller to close.
    """
    hole = socket.socket()
    hole.bind(("127.0.0.1", 0))
    hole.listen(4)  # never accepted: the TLS handshake waits for good
    machine = mach.Machine(
        name=name, url=f"https://127.0.0.1:{hole.getsockname()[1]}", password="pw",
    )
    machine.cert_file().parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(certfile, machine.cert_file())
    mach.save_machines([*mach.load_machines(), machine])
    return hole


class TestRemoteFilePicker(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon, each with a file of its own."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        root = Path(self.tmpdir)
        self.box_work = root / "box-work"
        self.box_work.mkdir()
        (self.box_work / "box_only_module.py").write_text("x = 1\n")
        self.home_work = root / "home-work"
        self.home_work.mkdir()
        (self.home_work / "home_only_module.py").write_text("y = 2\n")
        # The machine's daemon scans its user's home directory as it
        # starts, on the one file-index worker box-work's scan then
        # waits for.  Given the developer's home (880k entries on one
        # Mac, scanned cold in over 15 minutes: the daemon's KISS_HOME
        # is new, so no listings are cached), box's list stayed
        # ``loading`` past FILES_INDEX_WAIT; an empty home of its own
        # keeps the test about the forward, not the machine's disk.
        box_user_home = root / "box-user-home"
        box_user_home.mkdir()
        self.remote = await asyncio.to_thread(
            start_remote_daemon, root / "remote-home", password="pw-remote",
            config={"work_dir": str(self.box_work)},
            env={"HOME": str(box_user_home), "USERPROFILE": str(box_user_home)},
        )
        self.certfile = root / "cert.pem"
        keyfile = root / "key.pem"
        _generate_self_signed_cert(self.certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(self.certfile),
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
        """Send *cmd* on the local connection."""
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _read(self, reply_type: str) -> dict[str, Any]:
        """Return the next *reply_type* message of the local connection."""
        for _ in range(300):
            line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def _ask(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* message."""
        await self._send(cmd)
        return await self._read(reply_type)

    def _get_files(
        self, prefix: str, machine: str | None, work_dir: Path | None = None,
    ) -> dict[str, Any]:
        """A ``getFiles`` for tab ``tab-box``; *work_dir* None sends no folder.

        Both daemons share this test's filesystem, so a request that
        names no folder is what tells them apart: the machine answers
        from its configured ``box-work``, this daemon from ``home-work``.
        """
        cmd: dict[str, Any] = {"type": "getFiles", "prefix": prefix, "tabId": "tab-box"}
        if work_dir is not None:
            cmd["workDir"] = str(work_dir)
        if machine is not None:
            cmd["machine"] = machine
        return cmd

    @staticmethod
    def _texts(reply: dict[str, Any], kind: str) -> list[str]:
        return [item["text"] for item in reply["files"] if item["type"] == kind]

    async def test_files_of_a_bound_tab_are_the_machines(self) -> None:
        # The machine has never scanned box-work: its first answer is
        # the empty ``loading`` list, which must not reach the page.
        reply = await self._ask(self._get_files("only_module", "box"), "files")
        self.assertEqual(reply.get("error", ""), "")
        self.assertNotIn("loading", reply)
        self.assertEqual(reply["machine"], "box")
        self.assertEqual(reply["tabId"], "tab-box")
        self.assertEqual(reply["prefix"], "only_module")
        self.assertEqual(self._texts(reply, "file"), ["./box_only_module.py"])
        # The page names the machine's folder when it knows it.
        named = await self._ask(
            self._get_files("only_module", "box", self.box_work), "files",
        )
        self.assertEqual(named["machine"], "box")
        self.assertEqual(self._texts(named, "file"), ["./box_only_module.py"])

    async def test_files_without_a_machine_are_answered_here(self) -> None:
        reply = await self._ask(self._get_files("only_module", None), "files")
        if reply.get("loading"):
            # This daemon's own first scan: its real list follows.
            reply = await self._ask(self._get_files("only_module", None), "files")
        self.assertNotIn("machine", reply)
        self.assertEqual(self._texts(reply, "file"), ["./home_only_module.py"])

    async def test_a_mention_picked_for_the_machine_is_counted_there(self) -> None:
        await self._ask(self._get_files("", "box"), "files")
        await self._send({
            "type": "recordFileUsage", "path": "./box_only_module.py", "machine": "box",
        })
        frequent: list[str] = []
        for _ in range(50):
            reply = await self._ask(self._get_files("", "box"), "files")
            frequent = self._texts(reply, "frequent")
            if frequent:
                break
            await asyncio.sleep(0.1)
        self.assertEqual(frequent, ["./box_only_module.py"])
        # This daemon, asked about the same folder (the test's daemons
        # share one disk), never counted the mention.
        local = await self._ask(self._get_files("", None, self.box_work), "files")
        if local.get("loading"):
            local = await self._ask(self._get_files("", None, self.box_work), "files")
        self.assertIn("./box_only_module.py", self._texts(local, "file"))
        self.assertEqual(self._texts(local, "frequent"), [])

    async def test_unknown_machine_answers_with_an_error(self) -> None:
        # A mention picked for a machine that is gone is dropped quietly:
        # the daemon keeps answering the connection afterwards.
        await self._send({"type": "recordFileUsage", "path": "./x.py", "machine": "nowhere"})
        reply = await self._ask(self._get_files("only", "nowhere"), "files")
        self.assertEqual(reply["machine"], "nowhere")
        self.assertEqual(reply["tabId"], "tab-box")
        self.assertEqual(reply["prefix"], "only")
        self.assertEqual(reply["files"], [])
        self.assertIn("no machine named 'nowhere'", reply["error"])

    async def test_a_silent_machine_holds_up_only_its_own_list(self) -> None:
        # box scans its folder first (behind its daemon's start-up scan
        # of home, which takes seconds on a busy host): timed below is
        # one round trip.
        await self._ask(self._get_files("only_module", "box"), "files")
        hole = _register_silent_machine("hole", self.certfile)
        self.addCleanup(hole.close)
        started = time.monotonic()
        await self._send(self._get_files("src", "hole"))
        await self._ask({"type": "ping"}, "pong")
        box = await self._ask(self._get_files("only_module", "box"), "files")
        self.assertLess(
            time.monotonic() - started, mach.REQUEST_TIMEOUT / 2,
            "the ping or box's list waited for the silent machine's list",
        )
        self.assertEqual(box["machine"], "box")
        self.assertEqual(self._texts(box, "file"), ["./box_only_module.py"])
        silent = await self._read("files")
        self.assertGreaterEqual(
            time.monotonic() - started, mach.REQUEST_TIMEOUT / 2,
            "the machine did not hold the request: the test exercised nothing",
        )
        self.assertEqual(silent["machine"], "hole")
        self.assertEqual(silent["tabId"], "tab-box")
        self.assertEqual(silent["prefix"], "src")
        self.assertEqual(silent["files"], [])
        self.assertTrue(silent["error"])
