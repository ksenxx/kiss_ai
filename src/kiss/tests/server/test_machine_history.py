# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the CHATS panel lists the tasks of every registered machine.

A local daemon runs in this process and a remote one in its own process
(:mod:`kiss.tests.remote_daemon`).  With the remote registered through
:func:`kiss.agents.sorcar.machines.add_machine`, the local daemon's
``getHistory`` reply carries the union of both histories, each remote
row stamped with its machine, a task present on both machines listed
once (the local copy), and the registered machine names for the
panel's filter.
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
from kiss.agents.sorcar import persistence as th
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.conftest import redirect_persistence

ID_SHARED = "c" * 32


class TestMachineHistory(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon, one registry."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        # Local rows: one task shared with the remote (same id) and one of its own.
        self.local_task, _chat = th._add_task("local only", chat_id="chat-local")
        shared, _chat = th._add_task("shared task (local copy)", chat_id="chat-shared")
        db = th._get_db()
        db.execute("UPDATE task_history SET id = ? WHERE id = ?", (ID_SHARED, shared))
        db.execute("UPDATE task_history SET tags = 'coding' WHERE id = ?", (self.local_task,))
        # The local copy of the shared task is the newer one, so it is the
        # copy the merged list keeps (and the newest row overall).
        db.execute(
            "UPDATE task_history SET timestamp = ? WHERE id = ?",
            (time.time() + 100.0, ID_SHARED),
        )
        db.commit()

        self.remote = await asyncio.to_thread(
            start_remote_daemon, Path(self.tmpdir) / "remote-home", password="pw-remote",
            seed=[
                {"chat_id": "chat-r", "task": "remote alpha"},
                {"chat_id": "chat-shared", "task": "shared task", "task_id": ID_SHARED},
            ],
        )
        await asyncio.to_thread(mach.add_machine, "box", self.remote.url, "pw-remote")

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
        if th._db_conn is not None:
            th._db_conn.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved  # type: ignore[assignment]
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _history(self, **fields: Any) -> dict[str, Any]:
        cmd = {"type": "getHistory", "query": "", "offset": 0, "generation": 3, **fields}
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()
        for _ in range(100):
            line = await asyncio.wait_for(self.reader.readline(), timeout=20.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == "history" and msg.get("generation") == 3:
                return msg
        raise AssertionError("no history reply")

    async def test_history_merges_both_machines_without_duplicates(self) -> None:
        """Every machine's tasks are listed once, stamped, with the filter names."""
        reply = await self._history()
        self.assertEqual(reply["machines"], ["box"])
        self.assertEqual(reply["localMachine"], socket.gethostname().split(".")[0])
        rows = {s["title"]: s for s in reply["sessions"]}
        self.assertEqual(
            set(rows), {"local only", "shared task (local copy)", "remote alpha"},
        )
        self.assertEqual(len(reply["sessions"]), 3)
        self.assertNotIn("machine", rows["local only"])
        self.assertNotIn("machine", rows["shared task (local copy)"])
        self.assertEqual(rows["shared task (local copy)"]["task_id"], ID_SHARED)
        self.assertEqual(rows["remote alpha"]["machine"], "box")
        self.assertEqual(rows["remote alpha"]["machine_url"], self.remote.url)
        self.assertEqual(rows["remote alpha"]["id"], "chat-r")
        # The remote row carries the chat-header fields its daemon computed.
        self.assertEqual(rows["remote alpha"]["chat_first_task"], "remote alpha")
        # Newest first across machines.
        self.assertEqual(
            [s["title"] for s in reply["sessions"]],
            ["shared task (local copy)", "remote alpha", "local only"],
        )
        # The date range spans both histories.
        timestamps = [s["timestamp"] for s in reply["sessions"]]
        self.assertLessEqual(reply["dateRange"]["min"], min(timestamps))
        self.assertGreaterEqual(reply["dateRange"]["max"], max(timestamps))

    async def test_machine_filter_selects_one_source(self) -> None:
        """``machine`` narrows the page to one machine or to this daemon."""
        reply = await self._history(machine="box")
        self.assertEqual(
            sorted(s["title"] for s in reply["sessions"]), ["remote alpha", "shared task"],
        )
        self.assertTrue(all(s["machine"] == "box" for s in reply["sessions"]))
        self.assertEqual(reply["machines"], ["box"])
        reply = await self._history(machine=mach.LOCAL_MACHINE)
        self.assertEqual(
            sorted(s["title"] for s in reply["sessions"]),
            ["local only", "shared task (local copy)"],
        )
        self.assertTrue(all("machine" not in s for s in reply["sessions"]))
        self.assertEqual(reply["machines"], ["box"])
        reply = await self._history(machine="nobody")
        self.assertEqual(reply["sessions"], [])

    async def test_a_daemon_registered_with_itself_does_not_recurse(self) -> None:
        """A machine pointing back at this daemon adds nothing and never loops."""
        await asyncio.to_thread(
            mach.add_machine, "self", f"https://127.0.0.1:{self.server.port}", "",
        )
        reply = await self._history()
        self.assertEqual(reply["machines"], ["box", "self"])
        self.assertEqual(
            [s["title"] for s in reply["sessions"]],
            ["shared task (local copy)", "remote alpha", "local only"],
        )
        self.assertTrue(all(s.get("machine", "") != "self" for s in reply["sessions"]))

    async def test_query_tag_and_offset_apply_to_the_union(self) -> None:
        """A search and a tag reach the remote too; offsets page the union."""
        reply = await self._history(query="alpha")
        self.assertEqual([s["title"] for s in reply["sessions"]], ["remote alpha"])
        reply = await self._history(tag="coding")
        self.assertEqual([s["title"] for s in reply["sessions"]], ["local only"])
        page = await self._history(offset=2)
        self.assertEqual(len(page["sessions"]), 1)
        self.assertEqual(page["offset"], 2)
        self.assertEqual(
            page["sessions"][0]["title"], (await self._history())["sessions"][2]["title"],
        )

    async def test_without_machines_the_reply_is_local_only(self) -> None:
        """An empty registry: local rows, no stamps, an empty machine list."""
        self.assertTrue(mach.remove_machine("box"))
        reply = await self._history()
        self.assertEqual(reply["machines"], [])
        self.assertEqual(
            sorted(s["title"] for s in reply["sessions"]),
            ["local only", "shared task (local copy)"],
        )
        self.assertTrue(all("machine" not in s for s in reply["sessions"]))

    async def test_the_reply_labels_an_address_named_machine_by_its_host_name(self) -> None:
        """A machine registered by its address is labelled by the host name of its history reply."""
        mach._hosts.clear()
        await asyncio.to_thread(mach.add_machine, "", self.remote.url, "pw-remote")
        host = socket.gethostname().split(".")[0]
        reply = await self._history()
        self.assertEqual(reply["machines"], ["box", "127.0.0.1"])
        self.assertEqual(reply["machineLabels"], {"box": "box", "127.0.0.1": host})
        # The label outlives the machine: the host name of its last reply.
        self.remote.stop()
        reply = await self._history()
        self.assertEqual(reply["machineLabels"], {"box": "box", "127.0.0.1": host})

    async def test_a_dead_machine_hides_nothing_local(self) -> None:
        """Once the remote is down its rows vanish; the local ones stay."""
        self.remote.stop()
        mach._hosts.clear()
        reply = await self._history()
        self.assertEqual(reply["machines"], ["box"])
        self.assertEqual(
            reply["machineLabels"], {"box": "box"}, "a chosen name is the label",
        )
        self.assertEqual(
            sorted(s["title"] for s in reply["sessions"]),
            ["local only", "shared task (local copy)"],
        )
