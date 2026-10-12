# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the right sidebar's MACHINES subpanel data and the
Schedule subpanel's action buttons, over the local connection of a real
daemon.

* ``getMachines`` lists the local machine first (always reachable, at
  the daemon's own address and port) and then every registered
  machine, pinged through its daemon link: a real remote daemon
  (:mod:`kiss.tests.remote_daemon`) is reachable, a URL nobody listens
  on and an SSH login nobody serves are unreachable with the reason;
  the pings are cached and ``refresh`` pings again.
* ``machineAction`` adds (after a real handshake: a wrong password is
  refused) and removes machines, answering with the fresh list.
* ``cronAction`` pauses, resumes, edits, runs and removes a stored cron
  job, answering with the fresh ``cronJobs`` list and the error text of
  a refused action.
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

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar import machines as mach
from kiss.server import sidebar_panels
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon


def _forget_machine_pings() -> None:
    """Drop the cached machine pings so the next report pings anew."""
    with sidebar_panels._machines_lock:
        sidebar_panels._machines_checked_at = 0.0


class TestMachinesAndCronActions(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon, one registry, one cron store."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved_jobs = cron_agent.load_jobs()
        cron_agent.save_jobs([])
        mach.save_machines([])
        _forget_machine_pings()
        self.remote = await asyncio.to_thread(
            start_remote_daemon, Path(self.tmpdir) / "remote-home", password="pw-remote",
        )
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        self.remote.stop()
        mach.save_machines([])
        _forget_machine_pings()
        cron_agent.save_jobs(self.saved_jobs)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _ask(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* message."""
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()
        for _ in range(100):
            line = await asyncio.wait_for(self.reader.readline(), timeout=90.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def _machines(self, **fields: Any) -> dict[str, Any]:
        return await self._ask({"type": "getMachines", **fields}, "machines")

    async def test_get_machines_lists_local_then_registered_with_status(self) -> None:
        """The local row leads; a live remote is reachable, dead addresses are not."""
        await asyncio.to_thread(mach.add_machine, "box", self.remote.url, "pw-remote")
        dead_port = find_free_port()
        mach.save_machines([
            *mach.load_machines(),
            mach.Machine(name="dead", url=f"https://127.0.0.1:{dead_port}"),
            mach.Machine(name="sshbox", ssh=f"nobody@127.0.0.1:{dead_port}"),
        ])
        reply = await self._machines()
        self.assertEqual((reply["action"], reply["error"]), ("", ""))
        rows = {row["name"]: row for row in reply["machines"]}
        self.assertEqual(
            [row["name"] for row in reply["machines"]],
            [socket.gethostname().split(".")[0], "box", "dead", "sshbox"],
        )
        local = reply["machines"][0]
        self.assertEqual(local["local"], True)
        self.assertEqual(local["status"], "reachable")
        self.assertEqual(local["ip"], "127.0.0.1")
        self.assertEqual(local["port"], self.server.port)
        self.assertEqual(local["address"], f"https://127.0.0.1:{self.server.port}")
        box = rows["box"]
        self.assertEqual(box["local"], False)
        self.assertEqual(box["status"], "reachable")
        self.assertEqual(box["error"], "")
        self.assertEqual(box["ip"], "127.0.0.1")
        self.assertEqual(box["port"], self.remote.port)
        self.assertEqual(box["address"], self.remote.url)
        dead = rows["dead"]
        self.assertEqual(dead["status"], "unreachable")
        self.assertIn("dead", dead["error"])
        self.assertEqual((dead["ip"], dead["port"]), ("127.0.0.1", dead_port))
        sshbox = rows["sshbox"]
        self.assertEqual(sshbox["status"], "unreachable")
        self.assertEqual(sshbox["address"], f"ssh nobody@127.0.0.1:{dead_port}")
        self.assertEqual(sshbox["ip"], "127.0.0.1")
        self.assertEqual(sshbox["port"], 0, "no tunnel: no mapped port")
        self.assertIn("sshbox", sshbox["error"])
        # The pings are cached: a second request serves the same probe,
        # and `refresh` pings again.
        again = await self._machines()
        self.assertEqual(again["checkedAt"], reply["checkedAt"])
        refreshed = await self._machines(refresh=True)
        self.assertGreaterEqual(refreshed["checkedAt"], reply["checkedAt"])
        self.assertEqual(
            [row["status"] for row in refreshed["machines"]],
            ["reachable", "reachable", "unreachable", "unreachable"],
        )

    async def test_machine_action_adds_after_a_handshake_and_removes(self) -> None:
        """A wrong password is refused and registers nothing; add and remove send the list."""
        refused = await self._ask(
            {
                "type": "machineAction", "action": "add", "name": "box",
                "address": self.remote.url, "password": "wrong",
            },
            "machines",
        )
        self.assertNotEqual(refused["error"], "")
        self.assertEqual([row["name"] for row in refused["machines"]][1:], [])
        self.assertEqual(mach.load_machines(), [])
        added = await self._ask(
            {
                "type": "machineAction", "action": "add", "name": "",
                "address": self.remote.url, "password": "pw-remote",
            },
            "machines",
        )
        self.assertEqual((added["action"], added["error"]), ("add", ""))
        names = [row["name"] for row in added["machines"]]
        self.assertEqual(names[1:], ["127.0.0.1"], "an empty name defaults to the host")
        self.assertEqual(added["machines"][1]["status"], "reachable")
        removed = await self._ask(
            {"type": "machineAction", "action": "remove", "name": "127.0.0.1"}, "machines",
        )
        self.assertEqual(removed["error"], "")
        self.assertEqual(len(removed["machines"]), 1)
        self.assertEqual(mach.load_machines(), [])
        missing = await self._ask(
            {"type": "machineAction", "action": "remove", "name": "127.0.0.1"}, "machines",
        )
        self.assertEqual(missing["error"], "no machine named '127.0.0.1'")
        bogus = await self._ask(
            {"type": "machineAction", "action": "rename", "name": "x"}, "machines",
        )
        self.assertEqual(bogus["error"], "unknown action 'rename'")
        self.assertEqual(bogus["action"], "rename")
        # A malformed address (a port that is no number) is refused with
        # a reply too, not swallowed by the transport.
        malformed = await self._ask(
            {
                "type": "machineAction", "action": "add", "name": "bad",
                "address": "https://127.0.0.1:abc", "password": "x",
            },
            "machines",
        )
        self.assertIn("is not a machine address", malformed["error"])
        self.assertEqual(mach.load_machines(), [])

    async def test_cron_action_edits_the_stored_job_and_reports_errors(self) -> None:
        """Pause, resume, update, run now and remove answer with the fresh list."""
        cron_agent.save_jobs([{
            "id": "j1", "name": "Nightly", "schedule": "every 1d", "command": "echo night",
            "enabled": True, "next_run_at": time.time() + 3600,
        }])

        async def act(**fields: Any) -> dict[str, Any]:
            return await self._ask({"type": "cronAction", "jobId": "j1", **fields}, "cronJobs")

        paused = await act(action="pause")
        self.assertEqual((paused["action"], paused["error"]), ("pause", ""))
        self.assertEqual(paused["jobs"][0]["enabled"], False)
        resumed = await act(action="resume")
        self.assertEqual(resumed["jobs"][0]["enabled"], True)
        updated = await act(
            action="update", name="Morning", schedule="every 2h", what="echo morning",
        )
        self.assertEqual(updated["error"], "")
        row = updated["jobs"][0]
        self.assertEqual(
            (row["name"], row["schedule"], row["what"]), ("Morning", "every 2h", "echo morning"),
        )
        refused = await act(action="update", name="Morning", schedule="whenever", what="x")
        self.assertTrue(refused["error"].startswith("Unsupported schedule 'whenever':"))
        self.assertEqual(refused["jobs"][0]["schedule"], "every 2h")
        ran = await act(action="run_now")
        self.assertEqual(ran["error"], "")
        for _ in range(200):
            if "j1" not in cron_agent.running_job_ids():
                break
            await asyncio.sleep(0.05)
        self.assertEqual(cron_agent.load_jobs()[0]["last_summary"], "morning")
        removed = await act(action="remove")
        self.assertEqual(removed["error"], "")
        self.assertEqual(removed["jobs"], [])
        missing = await act(action="pause")
        self.assertEqual(missing["error"], "no job with id 'j1'")
        self.assertEqual(missing["jobs"], [])
        # The header's + form: `create` needs no jobId and carries the
        # kind; the reply lists the new job (an error leaves the list).
        created = await act(
            action="create", jobId="", name="Night", schedule="every 1d",
            what="echo night", kind="command",
        )
        self.assertEqual(created["error"], "")
        (row,) = created["jobs"]
        self.assertEqual(
            (row["name"], row["schedule"], row["kind"], row["what"]),
            ("Night", "every 1d", "command", "echo night"),
        )
        refused = await act(
            action="create", name="Night", schedule="every 1d", what="echo night",
            kind="command",
        )
        self.assertTrue(refused["error"].startswith("duplicate: job "), refused["error"])
        self.assertEqual(len(refused["jobs"]), 1)
