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

The unscoped ``task_events`` replay slot (``_reserve_replay_send``)
is still sent to every local peer: narrowing it needs
``JsonPrinter.replay_snapshot`` to carry the target tab, a file this
slice does not own.
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
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_private_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
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


if __name__ == "__main__":
    unittest.main()
