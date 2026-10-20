# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: the chat header's machine pick is ONE global value.

``setMachine`` from any connection (the chat-header machine drop-down)
is the machine a new chat's tasks run on for every connection: the
daemon persists it as ``config.json`` ``last_machine`` (``configData``
carries it on load) and broadcasts ``machineChanged`` to every client.
A chat that already shows a task keeps that task's machine on the
client side (``kiss.server.remote_tabs`` binds it); the global pick
never moves a bound tab.

These tests bind a loopback WSS listener on an ephemeral port with an
endpoint file under a temp dir and open two real local client
connections that simulate two surfaces.
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
from kiss.tests.server.conftest import redirect_persistence, restore_persistence


class TestGlobalMachinePick(IsolatedAsyncioTestCase):
    """Two local connections (= two surfaces) sharing one machine pick."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
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
        vc.save_config({"last_machine": ""})

    async def _connect(self) -> tuple[LocalReader, LocalWriter]:
        """Open one local connection (simulates one surface)."""
        reader, writer = await open_local_connection(
            self.server, limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        return reader, writer

    async def _send(self, writer: LocalWriter, cmd: dict[str, Any]) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: LocalReader,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 100,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Read events until *predicate* matches or the budget expires."""
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

    @staticmethod
    def _machine_changed(machine: str) -> Callable[[dict[str, Any]], bool]:
        """Predicate: the ``machineChanged`` broadcast for *machine*."""
        def _pred(msg: dict[str, Any]) -> bool:
            return msg.get("type") == "machineChanged" and msg.get("machine") == machine
        return _pred

    async def _config_of(
        self, reader: LocalReader, writer: LocalWriter,
    ) -> dict[str, Any]:
        """This daemon's ``configData`` reply to a ``getConfig`` on *writer*."""
        await self._send(writer, {"type": "getConfig"})
        ev = await self._drain_until(
            reader,
            lambda m: m.get("type") == "configData" and not m.get("forwarded"),
        )
        cfg = ev["config"]
        assert isinstance(cfg, dict)
        return cfg

    async def test_set_machine_persists_and_reaches_every_surface(self) -> None:
        """A pick on one surface is persisted and broadcast to both
        surfaces; ``getConfig`` then reports it to both; a pick of this
        machine (``""``) clears it the same way."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        await self._send(writer_a, {"type": "setMachine", "machine": "gpu-box"})
        for rd in (reader_a, reader_b):
            await self._drain_until(rd, self._machine_changed("gpu-box"))
        self.assertEqual(vc.load_config().get("last_machine"), "gpu-box")
        for rd, wr in ((reader_a, writer_a), (reader_b, writer_b)):
            cfg = await self._config_of(rd, wr)
            self.assertEqual(cfg["last_machine"], "gpu-box")

        await self._send(writer_b, {"type": "setMachine", "machine": ""})
        for rd in (reader_a, reader_b):
            await self._drain_until(rd, self._machine_changed(""))
        self.assertEqual(vc.load_config().get("last_machine"), "")
        cfg = await self._config_of(reader_a, writer_a)
        self.assertEqual(cfg["last_machine"], "")

    async def test_unchanged_or_junk_pick_is_not_broadcast(self) -> None:
        """Picking the machine already picked, or sending a pick that is
        not a string, changes nothing: no broadcast, config untouched.
        The next real pick is still broadcast (nothing is wedged)."""
        reader_a, writer_a = await self._connect()
        await self._send(writer_a, {"type": "setMachine", "machine": "gpu-box"})
        await self._drain_until(reader_a, self._machine_changed("gpu-box"))
        await self._send(writer_a, {"type": "setMachine", "machine": "gpu-box"})
        await self._send(writer_a, {"type": "setMachine", "machine": 42})
        await self._send(writer_a, {"type": "setMachine", "machine": "office"})
        ev = await self._drain_until(
            reader_a, lambda m: m.get("type") == "machineChanged",
        )
        self.assertEqual(ev["machine"], "office", "only the real change is broadcast")
        self.assertEqual(vc.load_config().get("last_machine"), "office")

    async def test_racing_picks_end_with_the_persisted_one_broadcast_last(
        self,
    ) -> None:
        """Commands run on a thread pool, so two surfaces' picks can
        race.  The broadcast goes out under the config lock: the LAST
        ``machineChanged`` every client sees names the persisted pick,
        so no surface is left showing a pick ``config.json`` lost."""
        reader, _writer = await self._connect()
        backend = self.server._agent_server
        names = ["gpu-box", "office", "lab", "rack-7"]
        barrier = threading.Barrier(len(names) * 4)

        def pick(i: int) -> None:
            barrier.wait()
            backend._cmd_set_machine({"type": "setMachine", "machine": names[i % len(names)]})

        threads = [threading.Thread(target=pick, args=(i,)) for i in range(len(names) * 4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        persisted = vc.load_config().get("last_machine")
        self.assertIn(persisted, names)
        seen: list[str] = []
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                ev = await self._drain_until(
                    reader, lambda m: m.get("type") == "machineChanged", timeout=0.5,
                )
            except TimeoutError:
                break
            seen.append(str(ev["machine"]))
        self.assertTrue(seen, "at least one pick was broadcast")
        self.assertEqual(seen[-1], persisted, "the last broadcast names the persisted pick")

    async def test_settings_save_keeps_the_pick(self) -> None:
        """The settings sheet's ``saveConfig`` (which never carries the
        pick) merges into ``config.json``: the pick survives it."""
        reader_a, writer_a = await self._connect()
        await self._send(writer_a, {"type": "setMachine", "machine": "gpu-box"})
        await self._drain_until(reader_a, self._machine_changed("gpu-box"))
        await self._send(
            writer_a, {"type": "saveConfig", "config": {"max_budget": 7}},
        )
        cfg = await self._config_of(reader_a, writer_a)
        self.assertEqual(cfg["last_machine"], "gpu-box")
        self.assertEqual(vc.load_config().get("last_machine"), "gpu-box")
