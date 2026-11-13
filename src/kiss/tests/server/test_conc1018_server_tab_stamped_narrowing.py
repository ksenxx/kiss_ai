# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a ``tabId``-stamped event (``status``, ``askUser``, ...) reaches
only the local peers that can show the tab, like a task-event copy.

Fan-out copies were already narrowed by ``_local_clients_for_tab`` (a
headless ``run`` peer that addressed only its own ``api-…`` tab is
skipped), but ``_deliver_tab_stamped`` sent every stamped event to
every local peer regardless (findings I1-F1).  Real daemon, real local
connections: peer A (tab-a) and the silent peer C get tab-a's
``status``; peer B (tab-b) does not; an empty-stamped global event
still reaches all three.

The unscoped ``task_events`` replay (a ``resumeSession`` with no
connection scope, reserved by ``_reserve_replay_send``) follows the
same narrowing: sending every sub-agent transcript of a big fan-out to
every ``daemon_client`` run on the machine kept the koushik-sorcar
daemon's event loop busy for 50 s (2026-11-12), long enough for every
link and local connection to miss its keepalive.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection


class TestTabStampedNarrowing(unittest.IsolatedAsyncioTestCase):
    """Stamped non-task events follow the local-peer narrowing."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-conc1018-stamped-")
        self.saved = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_private_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        th._close_thread_db()
        th._DB_PATH, th._KISS_DIR = self.saved
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(self) -> tuple[LocalReader, LocalWriter]:
        reader, writer = await open_local_connection(self.server)
        self.addAsyncCleanup(self._close, writer)
        return reader, writer

    async def _close(self, writer: LocalWriter) -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def _send(self, writer: LocalWriter, msg: dict[str, Any]) -> None:
        writer.write(json.dumps(msg).encode() + b"\n")
        await writer.drain()

    async def _wait_for_interest(self, tab_id: str) -> None:
        printer = self.server._printer
        for _ in range(300):
            with printer._ws_lock:
                if any(tab_id in tabs for tabs in printer._local_tab_sets.values()):
                    return
            await asyncio.sleep(0.01)
        raise AssertionError(f"server never recorded interest in {tab_id}")

    async def _collect(self, reader: LocalReader, seconds: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return events
            try:
                line = await asyncio.wait_for(reader.readline(), remaining)
            except TimeoutError:
                return events
            if not line:
                return events
            events.append(json.loads(line))

    async def test_stamped_status_skips_peers_that_addressed_only_other_tabs(self) -> None:
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        reader_c, _writer_c = await self._connect()
        await self._send(writer_a, {"type": "stop", "tabId": "tab-a"})
        await self._send(writer_b, {"type": "stop", "tabId": "tab-b"})
        await self._wait_for_interest("tab-a")
        await self._wait_for_interest("tab-b")
        # Drop the stop replies before the events under test are emitted.
        await asyncio.gather(
            self._collect(reader_a, 0.3),
            self._collect(reader_b, 0.3),
            self._collect(reader_c, 0.3),
        )

        printer = self.server._printer
        await asyncio.to_thread(
            printer.broadcast,
            {"type": "status", "running": True, "tabId": "tab-a", "startTs": 1},
        )
        await asyncio.to_thread(
            printer.broadcast,
            {"type": "tabs_state", "tabs": [], "tabId": ""},
        )

        got_a, got_b, got_c = await asyncio.gather(
            self._collect(reader_a, 1.0),
            self._collect(reader_b, 1.0),
            self._collect(reader_c, 1.0),
        )
        statuses = [
            [e for e in got if e["type"] == "status" and e.get("tabId") == "tab-a"]
            for got in (got_a, got_b, got_c)
        ]
        self.assertEqual([len(s) for s in statuses], [1, 0, 1])
        for got in (got_a, got_b, got_c):
            self.assertIn("tabs_state", [e["type"] for e in got])

    async def test_unscoped_replay_skips_peers_that_addressed_only_other_tabs(self) -> None:
        task_id, chat_id = th._add_task("replayed task")
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        reader_c, _writer_c = await self._connect()
        await self._send(writer_b, {"type": "stop", "tabId": "tab-b"})
        await self._wait_for_interest("tab-b")
        await asyncio.gather(
            self._collect(reader_a, 0.3),
            self._collect(reader_b, 0.3),
            self._collect(reader_c, 0.3),
        )

        # No ``scoped`` and no ``replayConnId``: a history click, whose
        # transcript every surface showing tab-a must get.
        await self._send(writer_a, {
            "type": "resumeSession", "chatId": chat_id, "taskId": task_id,
            "tabId": "tab-a",
        })

        got_a, got_b, got_c = await asyncio.gather(
            self._collect(reader_a, 1.5),
            self._collect(reader_b, 1.5),
            self._collect(reader_c, 1.5),
        )
        replays = [
            [e for e in got if e["type"] == "task_events" and e.get("task_id") == task_id]
            for got in (got_a, got_b, got_c)
        ]
        self.assertEqual([len(r) for r in replays], [1, 0, 1])
        self.assertEqual(replays[0][0]["tabId"], "tab-a")


if __name__ == "__main__":
    unittest.main()
