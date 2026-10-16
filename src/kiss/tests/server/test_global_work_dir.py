# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: the working directory is ONE global value.

Every browser tab shares the daemon's
single working directory: ``setWorkDir`` from any connection -- the
chat-header work-dir drop-down, the remote Explorer's check mark -- adopts
it for every task of every connection, persists it as ``config.json``
``work_dir`` and broadcasts ``workDirChanged`` to every client.  A
command carrying its own ``workDir`` (the Python API's ``work_dir=``)
still wins.

These tests bind a loopback WSS listener on an ephemeral port with an
endpoint file under a temp dir (not the production
``~/.kiss/sorcar-local.json``) and open two real local client
connections that simulate two clients.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.server.conftest import file_names, redirect_persistence, restore_persistence


class TestGlobalWorkDir(IsolatedAsyncioTestCase):
    """Two local connections (= two clients) sharing one work dir."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)

        self.dir_a = Path(self.tmpdir) / "ws_a"
        self.dir_b = Path(self.tmpdir) / "ws_b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()
        (self.dir_a / "alpha.txt").write_text("alpha")
        (self.dir_b / "beta.txt").write_text("beta")

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self._writers: list[LocalWriter] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[LocalReader, LocalWriter]:
        """Open one local connection (simulates one client)."""
        reader, writer = await open_local_connection(
            self.server, limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        return reader, writer

    async def _send(
        self, writer: LocalWriter, cmd: dict[str, Any],
    ) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: LocalReader,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 100,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Read events until *predicate* matches or the budget expires.

        Broadcasts are fanned out to every connection, so a reader may
        see events triggered by the other window's commands; the
        predicate is responsible for picking out the wanted one.
        """
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "connection closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError(
            f"predicate never matched within {max_events} events",
        )

    async def _pick_work_dir(
        self, writer: LocalWriter, reader: LocalReader, work_dir: Path,
    ) -> None:
        """Send ``setWorkDir`` on *writer* and wait until *reader* sees
        the ``workDirChanged`` broadcast for it.

        Commands from different connections run concurrently in the
        daemon's executor, so a command sent on another connection right
        after ``setWorkDir`` may otherwise run BEFORE the adoption.
        *reader* must be one whose earlier broadcasts were already
        drained, or a stale ``workDirChanged`` for the same directory
        would be taken for the new one.
        """
        await self._send(
            writer, {"type": "setWorkDir", "workDir": str(work_dir)},
        )
        await self._drain_until(
            reader,
            lambda m: (
                m.get("type") == "workDirChanged"
                and m.get("workDir") == str(work_dir)
            ),
        )

    @staticmethod
    def _files_event_with(name: str) -> Callable[[dict[str, Any]], bool]:
        """Predicate: a populated ``files`` event containing *name*."""
        def _pred(msg: dict[str, Any]) -> bool:
            return (
                msg.get("type") == "files"
                and not msg.get("loading")
                and name in file_names(msg)
            )
        return _pred

    async def test_set_work_dir_from_one_window_applies_to_every_window(
        self,
    ) -> None:
        """The core contract: the LAST ``setWorkDir`` from any connection
        is where every connection's work_dir-dependent commands run.

        Window A picks folder A, window B then picks folder B: a
        ``getFiles`` WITHOUT an explicit ``workDir`` from window A scans
        folder B.  Window A picking folder A again moves window B too.
        """
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._pick_work_dir(writer_a, reader_a, self.dir_a)
        await self._pick_work_dir(writer_b, reader_a, self.dir_b)

        await self._send(writer_a, {"type": "getFiles", "prefix": ""})
        ev_a = await self._drain_until(
            reader_a, self._files_event_with("./beta.txt"),
        )
        self.assertNotIn("./alpha.txt", file_names(ev_a))

        await self._pick_work_dir(writer_a, reader_a, self.dir_a)
        await self._send(writer_b, {"type": "getFiles", "prefix": ""})
        ev_b = await self._drain_until(
            reader_b, self._files_event_with("./alpha.txt"),
        )
        self.assertNotIn("./beta.txt", file_names(ev_b))

    async def test_set_work_dir_persists_and_broadcasts(self) -> None:
        """``setWorkDir`` persists ``config.json`` ``work_dir`` (a daemon
        restart keeps it) and every connection -- the picking one and
        the other window -- receives ``workDirChanged``; ``getConfig``
        then reports the same directory to both."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            writer_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )
        for rd in (reader_a, reader_b):
            await self._drain_until(
                rd,
                lambda m: (
                    m.get("type") == "workDirChanged"
                    and m.get("workDir") == str(self.dir_b)
                ),
            )
        self.assertEqual(vc.load_config().get("work_dir"), str(self.dir_b))
        for rd, wr in ((reader_a, writer_a), (reader_b, writer_b)):
            await self._send(wr, {"type": "getConfig"})
            cfg = await self._drain_until(
                rd, lambda m: m.get("type") == "configData",
            )
            self.assertEqual(cfg["config"]["work_dir"], str(self.dir_b))

    async def test_explicit_work_dir_wins_over_global_work_dir(
        self,
    ) -> None:
        """A command carrying its own ``workDir`` must keep it: the
        Python API's ``sorcar.run(work_dir=...)`` and the chat page's
        per-tab file requests take precedence over the global value."""
        reader_a, writer_a = await self._connect()
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            writer_a,
            {"type": "getFiles", "prefix": "", "workDir": str(self.dir_b)},
        )
        ev = await self._drain_until(
            reader_a, self._files_event_with("./beta.txt"),
        )
        self.assertNotIn("./alpha.txt", file_names(ev))

    async def test_empty_set_work_dir_keeps_global_work_dir(self) -> None:
        """An empty ``setWorkDir`` must not clear the global value."""
        reader_a, writer_a = await self._connect()
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(writer_a, {"type": "setWorkDir", "workDir": ""})
        await self._send(writer_a, {"type": "getFiles", "prefix": ""})
        ev = await self._drain_until(
            reader_a, self._files_event_with("./alpha.txt"),
        )
        self.assertNotIn("./beta.txt", file_names(ev))

    async def test_concurrent_adoptions_end_in_one_consistent_state(
        self,
    ) -> None:
        """Commands run on a thread pool, so a pick and a settings save
        can race.  Whichever adoption wins, the live value, the
        persisted ``config.json`` value and the last ``workDirChanged``
        every client saw must all name the same directory."""
        reader, _writer = await self._connect()
        backend = self.server._agent_server
        dirs = [str(self.dir_a), str(self.dir_b)]
        barrier = threading.Barrier(16)

        def adopt(i: int) -> None:
            barrier.wait()
            backend._apply_new_work_dir(dirs[i % 2])

        threads = [threading.Thread(target=adopt, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        live = backend.work_dir
        self.assertIn(live, dirs)
        self.assertEqual(vc.load_config().get("work_dir"), live)
        # The last broadcast the client saw names the live directory.
        last = ""
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=0.3)
            except TimeoutError:
                break
            msg = json.loads(line.decode("utf-8"))
            if msg.get("type") == "workDirChanged":
                last = msg["workDir"]
        self.assertEqual(last, live)
