# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the reply to a Schedule or Machines subpanel action names the row it acted on.

A clicked Delete stays spent (disabled) until the reply to its own
removal comes back: the subpanel re-arms it only on a ``remove`` reply
naming its row, ``jobId`` for a cron job and ``name`` for a machine, so
the reply to another row's removal, sent meanwhile, does not.  The
replies, this daemon's own and those of a registered machine relayed
here (or made up here when the machine cannot be asked), name the row.

Real local daemon, real local connection; the "machine" of the relayed
replies is one that is not registered, so the relay makes up the reply;
no mocks, patches or test doubles.
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


class PanelRemoveNamesRowTest(IsolatedAsyncioTestCase):
    """``cronAction`` / ``machineAction`` replies name their ``jobId`` / ``name``."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        root = Path(self.tmpdir)
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

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _ask(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()
        for _ in range(100):
            line = await asyncio.wait_for(self.reader.readline(), timeout=20.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def test_a_refused_job_removal_names_its_job(self) -> None:
        reply = await self._ask(
            {"type": "cronAction", "action": "remove", "jobId": "no-such-job"}, "cronJobs",
        )
        self.assertEqual((reply["action"], reply["jobId"]), ("remove", "no-such-job"))
        self.assertTrue(reply["error"], "removing a job that does not exist is refused")

    async def test_a_refused_machine_removal_names_its_machine(self) -> None:
        reply = await self._ask(
            {"type": "machineAction", "action": "remove", "name": "ghost"}, "machines",
        )
        self.assertEqual((reply["action"], reply["name"]), ("remove", "ghost"))
        self.assertTrue(reply["error"], "removing a machine never registered is refused")
        polled = await self._ask({"type": "getMachines"}, "machines")
        self.assertEqual((polled["action"], polled["name"]), ("", ""))

    async def test_a_relayed_refusal_names_its_row(self) -> None:
        job = await self._ask({
            "type": "cronAction", "action": "remove", "jobId": "j1", "machine": "nowhere",
        }, "cronJobs")
        self.assertEqual(
            (job["machine"], job["action"], job["jobId"]), ("nowhere", "remove", "j1"),
        )
        self.assertIn("no machine named 'nowhere'", job["error"])
        machine = await self._ask({
            "type": "machineAction", "action": "remove", "name": "box", "machine": "nowhere",
        }, "machines")
        self.assertEqual(
            (machine["machine"], machine["action"], machine["name"]), ("nowhere", "remove", "box"),
        )
        self.assertIn("no machine named 'nowhere'", machine["error"])
        polled = await self._ask({"type": "getCronJobs", "machine": "nowhere"}, "cronJobs")
        self.assertNotIn("jobId", polled, "a poll names no row")
