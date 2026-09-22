# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Task events reach only the UDS peers that can show them.

Before this change every task event (every streamed token) was copied
to every connected UDS client; with dozens of headless ``run`` clients
the daemon's event loop drowned in per-client send coroutines.  The
tests drive a real :class:`RemoteAccessServer` over real Unix-domain
sockets: a peer that addressed only another tab is skipped, while a
webview peer and a peer that never addressed a tab keep receiving every
copy, and global events still reach everyone.
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
from kiss.tests.conftest import requires_unix_sockets


@requires_unix_sockets
class TestTargetedUdsFanout(unittest.IsolatedAsyncioTestCase):
    """E2E fan-out routing over a live daemon socket."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-uds-fanout-")
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_unix_connection(str(self.uds_path))
        self.addAsyncCleanup(self._close, writer)
        return reader, writer

    async def _close(self, writer: asyncio.StreamWriter) -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def _send(self, writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
        writer.write(json.dumps(msg).encode() + b"\n")
        await writer.drain()

    async def _wait_for_uds_writers(self, count: int) -> None:
        for _ in range(300):
            with self.server._printer._ws_lock:
                if len(self.server._printer._uds_writers) >= count:
                    return
            await asyncio.sleep(0.01)
        raise AssertionError(f"server never registered {count} UDS writers")

    async def _wait_for_interest(self, tab_id: str) -> None:
        printer = self.server._printer
        for _ in range(300):
            with printer._ws_lock:
                if any(tab_id in tabs for tabs in printer._uds_local_tab_sets.values()):
                    return
            await asyncio.sleep(0.01)
        raise AssertionError(f"server never recorded interest in {tab_id}")

    async def _collect(self, reader: asyncio.StreamReader, seconds: float) -> list[dict[str, Any]]:
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

    async def test_task_copies_skip_peers_that_addressed_only_other_tabs(self) -> None:
        """Peer A (tab-a) and the silent peer C get tab-a's events; peer B (tab-b) does not."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        reader_c, _writer_c = await self._connect()
        await self._wait_for_uds_writers(3)
        await self._send(writer_a, {"type": "stop", "tabId": "tab-a"})
        await self._send(writer_b, {"type": "stop", "tabId": "tab-b"})
        await self._wait_for_interest("tab-a")
        await self._wait_for_interest("tab-b")
        # Drop the stop replies before the events under test are emitted.
        await self._collect(reader_a, 0.3)
        await self._collect(reader_b, 0.3)
        await self._collect(reader_c, 0.3)

        printer = self.server._printer
        printer.subscribe_tab("task-1", "tab-a")
        await asyncio.to_thread(
            printer.broadcast,
            {"type": "text_delta", "taskId": "task-1", "text": "hello"},
        )
        await asyncio.to_thread(printer.broadcast, {"type": "tasks_updated"})

        got_a, got_b, got_c = await asyncio.gather(
            self._collect(reader_a, 1.0),
            self._collect(reader_b, 1.0),
            self._collect(reader_c, 1.0),
        )
        types_a = [e["type"] for e in got_a]
        types_b = [e["type"] for e in got_b]
        types_c = [e["type"] for e in got_c]
        self.assertIn("text_delta", types_a)
        self.assertEqual([e["tabId"] for e in got_a if e["type"] == "text_delta"], ["tab-a"])
        self.assertNotIn("text_delta", types_b)
        self.assertIn("text_delta", types_c)
        for types in (types_a, types_b, types_c):
            self.assertIn("tasks_updated", types)

    async def test_webview_peer_receives_copies_for_other_tabs(self) -> None:
        """A connection that announced a webview mirrors every tab, whatever it addressed."""
        reader_w, writer_w = await self._connect()
        await self._wait_for_uds_writers(1)
        await self._send(writer_w, {"type": "stop", "tabId": "tab-w"})
        await self._wait_for_interest("tab-w")
        printer = self.server._printer
        conn_ids: list[str] = []
        for _ in range(300):
            with printer._ws_lock:
                conn_ids = list(printer._uds_local_tab_sets)
            if conn_ids:
                break
            await asyncio.sleep(0.01)
        printer.mark_uds_webview(conn_ids[0])
        await self._collect(reader_w, 0.3)

        printer.subscribe_tab("task-2", "tab-other")
        await asyncio.to_thread(
            printer.broadcast,
            {"type": "text_delta", "taskId": "task-2", "text": "hi"},
        )
        got = await self._collect(reader_w, 1.0)
        self.assertEqual(
            [e["tabId"] for e in got if e["type"] == "text_delta"], ["tab-other"],
        )


if __name__ == "__main__":
    unittest.main()
