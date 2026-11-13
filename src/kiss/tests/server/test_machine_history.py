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
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import persistence as th
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import _generate_self_signed_cert, open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.conftest import redirect_persistence, restore_persistence

ID_SHARED = "c" * 32


class TestMachineHistory(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon, one registry."""

    async def asyncSetUp(self) -> None:
        # Cleanups, each registered as its resource starts, not asyncTearDown:
        # unittest skips the tear-down when the set-up fails.
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.saved = redirect_persistence(self.tmpdir)
        self.addCleanup(restore_persistence, self.saved)
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

        self.addCleanup(mach.save_machines, [])
        self.remote = await asyncio.to_thread(
            start_remote_daemon, Path(self.tmpdir) / "remote-home", password="pw-remote",
            seed=[
                {"chat_id": "chat-r", "task": "remote alpha"},
                {"chat_id": "chat-shared", "task": "shared task", "task_id": ID_SHARED},
            ],
        )
        self.addCleanup(self.remote.stop)
        await asyncio.to_thread(mach.add_machine, "box", self.remote.url, "pw-remote")

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        # Registered before the start: one that fails part-way is stopped too.
        self.addAsyncCleanup(self.server.stop_async)
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)
        self.addCleanup(self.writer.close)

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

    async def _redirect_remote_database(self, target: Path | None) -> None:
        """Ask the remote test daemon to use *target*, then wait for its acknowledgement."""
        control = self.remote.home / "history-db-redirect"
        status = self.remote.home / "history-db-redirect.status"
        expected = target or self.remote.home / "history.db"
        if target is None:
            control.unlink(missing_ok=True)
        else:
            control.write_text(str(target))
        for _ in range(200):
            try:
                if status.read_text() == str(expected):
                    return
            except OSError:
                pass
            await asyncio.sleep(0.05)
        self.fail(f"remote database redirect was not acknowledged: {expected}")

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

    async def test_a_synced_row_is_listed_as_the_machine_it_ran_on(self) -> None:
        """A row a history sync copied in from a machine is that machine's, not this daemon's."""
        synced, _chat = th._add_task("synced from box", chat_id="chat-synced")
        stray, _chat = th._add_task("synced from an unregistered host", chat_id="chat-stray")
        db = th._get_db()
        db.execute("UPDATE task_history SET machine = 'box' WHERE id = ?", (synced,))
        db.execute("UPDATE task_history SET machine = 'gone-host' WHERE id = ?", (stray,))
        db.commit()
        reply = await self._history()
        rows = {s["title"]: s for s in reply["sessions"]}
        self.assertEqual(rows["synced from box"]["machine"], "box")
        self.assertEqual(rows["synced from box"]["machine_url"], self.remote.url)
        self.assertNotIn("machine", rows["synced from an unregistered host"])
        self.assertNotIn("machine", rows["local only"])
        self.assertTrue(all("origin" not in s for s in reply["sessions"]))
        # This machine's own rows -- what another daemon's fan-out asks
        # for -- leave out every row that ran elsewhere.
        reply = await self._history(machine=mach.LOCAL_MACHINE)
        self.assertEqual(
            sorted(s["title"] for s in reply["sessions"]),
            ["local only", "shared task (local copy)"],
        )

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

    async def test_a_machine_that_stops_answering_keeps_its_rows(self) -> None:
        """The rows a machine last answered with stay listed while it is down."""
        before = await self._history()
        self.remote.stop()
        reply = await self._history()
        self.assertEqual(
            [s["title"] for s in reply["sessions"]],
            [s["title"] for s in before["sessions"]],
        )
        self.assertEqual(
            [s["title"] for s in reply["sessions"] if s.get("machine") == "box"],
            ["remote alpha"],
        )
        self.assertEqual(reply["dateRange"], before["dateRange"])
        # A search is not remembered: it reaches the machine or shows nothing of it.
        reply = await self._history(query="alpha")
        self.assertEqual(reply["sessions"], [])

    async def test_a_machine_answers_while_a_write_to_its_history_is_stuck(self) -> None:
        """A machine whose daemon waits to write its history still answers CHATS and the spend sync.

        Another connection holds the write lock of the machine's ``history.db`` with a
        change it has not committed (a history sync merging a delta, a CLI run), so a
        write of the machine's daemon waits for the database, as the process's writer.
        What another daemon's link asks for (``getHistory`` for the CHATS rows,
        ``getTaskUsage`` for the spend sync) reads the last committed state and waits
        for neither: each answers within the link's ``REQUEST_TIMEOUT`` (the link
        raises otherwise), and the daemon keeps answering a ``ping``.
        """
        machine = next(m for m in mach.load_machines() if m.name == "box")
        alpha = self.remote.task_ids["chat-r/remote alpha"]
        holder = sqlite3.connect(self.remote.home / "history.db", isolation_level=None)
        self.addCleanup(holder.close)
        holder.execute("UPDATE task_history SET cost = 2.5 WHERE id = ?", (alpha,))
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("UPDATE task_history SET cost = 99 WHERE id = ?", (alpha,))
        # On a lane of its own: a connection's commands are served one at a
        # time, and this one is not done until the database is free.
        favorite = {"type": "setFavorite", "taskId": alpha, "isFavorite": True}
        await asyncio.to_thread(mach.link_for(machine, "writer").request, favorite, "")
        await asyncio.sleep(1.0)  # the daemon's write now waits for the database

        link = mach.link_for(machine)
        page = await asyncio.to_thread(mach.fetch_history, machine, "", "", mach.HISTORY_PAGE)
        usage = await asyncio.to_thread(
            link.request, {"type": "getTaskUsage", "taskIds": [alpha]}, "taskUsage",
        )
        await asyncio.to_thread(link.request, {"type": "ping"}, "pong")
        rows = {s["title"]: s for s in page.rows}
        self.assertEqual(set(rows), {"remote alpha", "shared task"})
        self.assertFalse(rows["remote alpha"]["is_favorite"], "the daemon's write was not stuck")
        self.assertEqual(rows["remote alpha"]["cost"], 2.5)
        self.assertEqual(usage["tasks"][alpha]["cost"], 2.5)

        holder.execute("ROLLBACK")
        # The daemon's write lands once the database is free.
        for _ in range(100):
            page = await asyncio.to_thread(mach.fetch_history, machine, "", "", mach.HISTORY_PAGE)
            if any(s["is_favorite"] for s in page.rows):
                break
            await asyncio.sleep(0.1)
        self.assertEqual([s["title"] for s in page.rows if s["is_favorite"]], ["remote alpha"])

    async def _write(
        self, cmd: dict[str, Any], reply_type: str = "tasks_updated",
    ) -> dict[str, Any]:
        """Send the CHATS-panel write *cmd* and return the first *reply_type* message."""
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()
        for _ in range(100):
            line = await asyncio.wait_for(self.reader.readline(), timeout=20.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type:
                return msg
            self.assertNotEqual(msg.get("type"), "notice", msg)
        raise AssertionError(f"no {reply_type}")

    async def test_writes_to_a_remote_row_reach_its_machine(self) -> None:
        """Favourite, rename and delete stamped ``machine`` change that machine's history only."""
        rows = {s["title"]: s for s in (await self._history(machine="box"))["sessions"]}
        alpha = rows["remote alpha"]["task_id"]
        await self._write(
            {"type": "setFavorite", "taskId": alpha, "isFavorite": True, "machine": "box"},
        )
        rows = {s["title"]: s for s in (await self._history())["sessions"]}
        self.assertTrue(rows["remote alpha"]["is_favorite"])
        self.assertFalse(rows["local only"].get("is_favorite"))

        await self._write(
            {"type": "setChatSummary", "chatId": "chat-r", "summary": "Renamed", "machine": "box"},
        )
        rows = {s["title"]: s for s in (await self._history())["sessions"]}
        self.assertEqual(rows["remote alpha"]["chat_summary"], "Renamed")

        # The shared chat goes from the machine; this daemon's copy stays.
        await self._write(
            {"type": "deleteChat", "chatId": "chat-shared", "tabId": "t", "machine": "box"},
        )
        await self._write(
            {"type": "deleteTask", "taskId": alpha, "tabId": "t", "machine": "box"},
        )
        self.assertEqual((await self._history(machine="box"))["sessions"], [])
        self.assertEqual(
            sorted(s["title"] for s in (await self._history())["sessions"]),
            ["local only", "shared task (local copy)"],
        )

    async def test_a_machine_whose_history_fails_answers_with_the_failure(self) -> None:
        """A machine that cannot open its ``history.db`` still answers the spend sync and a delete.

        The test daemon redirects its database path to a directory, so
        every open raises ``sqlite3.OperationalError``.  Neither command
        was answered: the asking link waited out its timeout and dropped
        its connection, and the delete's failure read as a machine that
        could not be reached.
        """
        machine = next(m for m in mach.load_machines() if m.name == "box")
        alpha = self.remote.task_ids["chat-r/remote alpha"]
        db_file = self.remote.home / "history.db"
        blocked = db_file.with_name("history.db.blocked")
        blocked.mkdir()
        await self._redirect_remote_database(blocked)

        usage = await asyncio.to_thread(
            mach.link_for(machine).request,
            {"type": "getTaskUsage", "taskIds": [alpha]}, "taskUsage",
        )
        self.assertNotIn("tasks", usage, "a failed read must not pass for 'no such task'")
        self.assertIn("unable to open database file", usage["error"])
        delete = {"type": "deleteTask", "taskId": alpha, "tabId": "tab-9", "machine": "box"}
        notice = await self._write(delete, "notice")
        self.assertEqual(notice["tabId"], "tab-9")
        self.assertIn("box: Cannot delete the task: unable to open database file", notice["text"])

        await self._redirect_remote_database(None)
        blocked.rmdir()
        await self._write(delete)
        self.assertEqual(
            [s["title"] for s in (await self._history(machine="box"))["sessions"]],
            ["shared task"],
        )

    async def test_a_write_to_an_unknown_machine_tells_the_tab(self) -> None:
        """A write naming no registered machine answers with a notice for the tab."""
        notice = await self._write(
            {"type": "deleteTask", "taskId": "x", "tabId": "tab-9", "machine": "nobody"},
            "notice",
        )
        self.assertEqual(notice["tabId"], "tab-9")
        self.assertIn("nobody", notice["text"])

    async def test_a_dead_machine_hides_nothing_local(self) -> None:
        """A machine down before it ever answered lists nothing; the local rows stay."""
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
