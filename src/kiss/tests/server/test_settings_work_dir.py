# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The daemon side of the chat header's working directory drop-down.

Tested against a real ``RemoteAccessServer`` over its local endpoint:

* ``getConfig`` reports the daemon's ONE global work_dir to every
  connection (= every browser window and webapp instance): the last
  ``setWorkDir`` from any of them, which also replaces the persisted
  value;
* a ``saveConfig`` carrying ``config.work_dir`` persists it to
  ``config.json`` and moves the global work_dir;
* a ``saveConfig`` that omits ``work_dir`` -- which is what the
  Settings panel sends on every surface, since it has no working
  directory field -- leaves the persisted value alone.

The chat page half (the Settings form never carries ``work_dir``; every
client picks through the chat header's drop-down) is covered by the real DOM tests in
``webapp/test/settingsWorkDirField.test.js`` and
``webapp/test/chatWorkDirSelect.test.js``.
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
from unittest import IsolatedAsyncioTestCase, TestCase

import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.server.commands import _CommandsMixin
from kiss.server.server import AgentServer
from kiss.server.web_server import Payload, RemoteAccessServer, WebPrinter
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.server.conftest import redirect_persistence, restore_persistence
from kiss.tests.server.pause import Pause


class TestWorkDirConfigRoundTrip(IsolatedAsyncioTestCase):
    """getConfig / saveConfig handle ``work_dir`` over a real local connection."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)

        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        self.dir_a = Path(self.tmpdir) / "ws_a"
        self.dir_b = Path(self.tmpdir) / "ws_b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.tests.local_ws import _generate_self_signed_cert
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
        th._close_thread_db()
        restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[LocalReader, LocalWriter]:
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
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "local connection closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError(
            f"predicate never matched within {max_events} events",
        )

    @staticmethod
    def _config_data_with_work_dir(
        work_dir: str,
    ) -> Callable[[dict[str, Any]], bool]:
        def _pred(msg: dict[str, Any]) -> bool:
            return (
                msg.get("type") == "configData"
                and msg.get("config", {}).get("work_dir") == work_dir
            )
        return _pred

    async def test_get_config_reports_the_one_global_work_dir(
        self,
    ) -> None:
        """``getConfig`` reports the daemon's single global work_dir to
        every connection: the last ``setWorkDir`` from ANY window is
        what both windows see (and scope their history and Explorer
        by), because it is where both windows' tasks run."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        # Each window's pick is acknowledged (``reply``) before the other
        # window's goes out: the two connections are served by two
        # handlers on the thread pool, so nothing else orders B's
        # adoption after A's and before A's ``getConfig`` (full-suite
        # run of 2026-11-05: A's reply still named ``dir_a``).
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a), "reply": True},
        )
        await self._drain_until(
            reader_a, self._config_data_with_work_dir(str(self.dir_a)),
        )
        await self._send(
            writer_b, {"type": "setWorkDir", "workDir": str(self.dir_b), "reply": True},
        )
        await self._drain_until(
            reader_b, self._config_data_with_work_dir(str(self.dir_b)),
        )

        await self._send(writer_a, {"type": "getConfig"})
        await self._drain_until(
            reader_a, self._config_data_with_work_dir(str(self.dir_b)),
        )

        await self._send(writer_b, {"type": "getConfig"})
        await self._drain_until(
            reader_b, self._config_data_with_work_dir(str(self.dir_b)),
        )

    async def test_get_config_reports_the_server_machine_name(self) -> None:
        """Every ``configData`` reply carries the server machine's short
        host name — the chat page shows it centered in the status bar so
        the user always sees which machine the agent runs on.  It is the
        name the machine registry gives the local daemon (not
        ``platform.node()``, which carries the domain on cloud hosts), so
        the status bar and the chat header's machine drop-down agree."""
        from kiss.agents.sorcar.machines import local_machine_name

        reader, writer = await self._connect()
        await self._send(writer, {"type": "getConfig"})
        msg = await self._drain_until(
            reader, lambda m: m.get("type") == "configData",
        )
        self.assertEqual(msg.get("machine"), local_machine_name())
        self.assertNotIn(".", msg["machine"])

    async def test_set_work_dir_replaces_persisted_value_for_everyone(
        self,
    ) -> None:
        """A ``setWorkDir`` (a chat-header drop-down pick) replaces
        the persisted work_dir: ``getConfig`` reports the new directory
        both to the picking connection and to a connection opened
        afterwards that announced nothing, and ``config.json`` holds it."""
        vc.save_config({"work_dir": str(self.dir_a)})

        reader_picker, writer_picker = await self._connect()
        await self._send(
            writer_picker,
            {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )
        await self._send(writer_picker, {"type": "getConfig"})
        await self._drain_until(
            reader_picker, self._config_data_with_work_dir(str(self.dir_b)),
        )

        reader_fresh, writer_fresh = await self._connect()
        await self._send(writer_fresh, {"type": "getConfig"})
        await self._drain_until(
            reader_fresh, self._config_data_with_work_dir(str(self.dir_b)),
        )
        self.assertEqual(vc.load_config().get("work_dir"), str(self.dir_b))

    async def test_save_config_work_dir_persists_and_updates_fallback(
        self,
    ) -> None:
        """A ``saveConfig`` carrying ``config.work_dir`` (sent by the
        standalone web client) persists the
        value and moves the daemon-wide fallback work_dir."""
        reader, writer = await self._connect()
        await self._send(
            writer,
            {"type": "saveConfig", "config": {"work_dir": str(self.dir_b)}},
        )
        await self._drain_until(
            reader, self._config_data_with_work_dir(str(self.dir_b)),
        )
        self.assertEqual(
            vc.load_config().get("work_dir"), str(self.dir_b),
        )
        self.assertEqual(
            self.server._agent_server.work_dir, str(self.dir_b),
        )

    async def test_save_config_reply_reports_live_work_dir(self) -> None:
        """The ``configData`` answering a ``saveConfig`` reports the
        daemon's live global work_dir, like ``getConfig`` does -- not
        the raw persisted value, which is empty before anything was
        adopted (the daemon runs in its startup fallback then)."""
        vc.save_config({"work_dir": ""})
        self.server._agent_server.work_dir = str(self.dir_b)
        reader, writer = await self._connect()
        await self._send(
            writer, {"type": "saveConfig", "config": {"max_budget": 42}},
        )
        msg = await self._drain_until(
            reader,
            lambda m: (
                m.get("type") == "configData"
                and m.get("config", {}).get("max_budget") == 42
            ),
        )
        self.assertEqual(msg["config"]["work_dir"], str(self.dir_b))

    async def test_save_config_without_work_dir_keeps_persisted_value(
        self,
    ) -> None:
        """A Settings-panel ``saveConfig`` (which omits ``work_dir``:
        the form has no working directory field) must not clobber a
        previously persisted work_dir."""
        vc.save_config({"work_dir": str(self.dir_a)})
        reader, writer = await self._connect()
        await self._send(
            writer, {"type": "saveConfig", "config": {"max_budget": 42}},
        )
        await self._drain_until(
            reader,
            lambda m: (
                m.get("type") == "configData"
                and m.get("config", {}).get("max_budget") == 42
            ),
        )
        self.assertEqual(
            vc.load_config().get("work_dir"), str(self.dir_a),
        )

    async def test_a_held_config_reply_never_undoes_a_newer_work_dir_on_the_wire(
        self,
    ) -> None:
        """Two live connections: a ``getConfig`` reply held between its
        snapshot and its send (the daemon's own ``_broadcast_to_conn``,
        paused at entry on its handler thread) while the other
        connection's ``setWorkDir`` adopts a new directory.  On the
        viewer's wire the last of ``workDirChanged`` / ``configData``
        must name the adopted directory: an adoption that overtook the
        held reply would leave this one surface on the superseded one
        (its Explorer, its history scope, the directory it advertises
        for the next task) while the daemon and the picker moved on.
        The in-process version of this race is
        :class:`TestConfigReplyOrderedAgainstWorkDirChange`."""
        reader_viewer, writer_viewer = await self._connect()
        reader_picker, writer_picker = await self._connect()
        await self._send(
            writer_viewer, {"type": "setWorkDir", "workDir": str(self.dir_a), "reply": True},
        )
        await self._drain_until(
            reader_viewer, self._config_data_with_work_dir(str(self.dir_a)),
        )
        held = Pause(
            _CommandsMixin._broadcast_to_conn.__code__,
            lambda loc: loc["event"].get("type") == "configData",
        )
        with held:
            await self._send(writer_viewer, {"type": "getConfig"})
            self.assertTrue(
                await asyncio.to_thread(held.paused.wait, 10), "no configData reply",
            )
            await self._send(
                writer_picker,
                {"type": "setWorkDir", "workDir": str(self.dir_b), "reply": True},
            )
            # An adoption ordered against the held reply waits for it; an
            # unordered one completes within this window and its
            # ``workDirChanged`` overtakes the stale reply on the wire.
            await asyncio.sleep(0.5)
            held.release()
        seen: list[tuple[str, str]] = []

        def reply_and_change_seen(msg: dict[str, Any]) -> bool:
            if msg.get("type") == "configData":
                seen.append(("configData", msg["config"]["work_dir"]))
            elif msg.get("type") == "workDirChanged":
                seen.append(("workDirChanged", msg["workDir"]))
            return ("workDirChanged", str(self.dir_b)) in seen and "configData" in [
                kind for kind, _ in seen
            ]

        await self._drain_until(reader_viewer, reply_and_change_seen, timeout=10.0)
        self.assertEqual(
            seen[-1][1], str(self.dir_b),
            f"the viewer is left on the superseded directory: {seen}",
        )
        await self._drain_until(
            reader_picker, self._config_data_with_work_dir(str(self.dir_b)), timeout=10.0,
        )
        self.assertEqual(self.server._agent_server.work_dir, str(self.dir_b))


class _WireCapturingWebPrinter(WebPrinter):
    """A real ``WebPrinter`` whose wire is a list: every payload in send order."""

    def __init__(self) -> None:
        super().__init__()
        self.wire: list[dict[str, Any]] = []
        self._wire_lock = threading.Lock()

    def _send_to_conn(self, conn_id: str, data: Payload, ahead: bool = False) -> None:
        self._capture(data)

    def _send_to_ws_clients(self, data: Payload, tab_id: str = "") -> None:
        self._capture(data)

    def _capture(self, data: Payload) -> None:
        # No replay slot (``replay_snapshot``) here: nothing replays.
        assert isinstance(data, str)
        with self._wire_lock:
            self.wire.append(json.loads(data))


class _ParkingReplyServer(AgentServer):
    """Real server whose first ``configData`` reply parks until released."""

    def __init__(self, printer: WebPrinter) -> None:
        super().__init__(printer=printer)
        self.parked = threading.Event()
        self.release = threading.Event()

    def _broadcast_to_conn(self, event: dict[str, Any], conn_id: str) -> None:
        """Park the first ``configData`` reply, built but unsent, then delegate."""
        if event.get("type") == "configData" and not self.parked.is_set():
            self.parked.set()
            if not self.release.wait(timeout=60):
                raise TimeoutError("parked too long in _broadcast_to_conn")
        super()._broadcast_to_conn(event, conn_id)


class TestConfigReplyOrderedAgainstWorkDirChange(TestCase):
    """A delayed ``configData`` reply never undoes a newer ``workDirChanged``.

    Two surfaces' commands run on two executor threads.  One surface's
    ``getConfig`` reply is built (naming the directory of the moment)
    and then sent; a ``setWorkDir`` from the other surface that adopts
    a new directory and broadcasts ``workDirChanged`` in between would
    reach the first surface BEFORE its stale reply, which then puts
    that surface back on the old directory (its Explorer, its history
    scope and the directory it advertises for the next task) while the
    daemon and every other surface stay on the new one.  Snapshot and
    send are ordered against adoptions instead, so whatever the wire
    carries last names the directory the daemon is on.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        self.dir_a = Path(self.tmpdir) / "ws_a"
        self.dir_b = Path(self.tmpdir) / "ws_b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()
        self.printer = _WireCapturingWebPrinter()
        self.server = _ParkingReplyServer(self.printer)
        self.server.work_dir = str(self.dir_a)

    def tearDown(self) -> None:
        self.server.release.set()
        self.server._file_index.stop()
        restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_delayed_config_reply_does_not_undo_a_newer_work_dir(self) -> None:
        """A ``getConfig`` reply parked between its snapshot and its send
        while another connection's ``setWorkDir`` adopts a new directory:
        on the viewer's wire the last of ``workDirChanged`` / ``configData``
        must name the adopted directory, never the superseded one."""
        getter = threading.Thread(
            target=self.server._cmd_get_config, args=({"connId": "viewer"},),
        )
        getter.start()
        self.assertTrue(self.server.parked.wait(timeout=10), "no configData reply")
        adopter = threading.Thread(
            target=self.server._cmd_set_work_dir,
            args=({"workDir": str(self.dir_b), "connId": "picker"},),
        )
        adopter.start()
        # An adoption ordered against the parked reply waits for it (this
        # join then times out); an unordered one completes here and its
        # ``workDirChanged`` precedes the stale reply on the wire.
        adopter.join(timeout=0.5)
        self.server.release.set()
        getter.join(timeout=10)
        adopter.join(timeout=10)
        self.assertFalse(getter.is_alive() or adopter.is_alive(), "a command hung")
        self.assertEqual(self.server.work_dir, str(self.dir_b))

        seen = [
            (e["type"], e.get("workDir") or e["config"]["work_dir"])
            for e in self.printer.wire
            if e["type"] in ("workDirChanged", "configData")
        ]
        self.assertIn(("workDirChanged", str(self.dir_b)), seen)
        self.assertIn("configData", [kind for kind, _ in seen])
        self.assertEqual(
            seen[-1][1], str(self.dir_b),
            f"the viewer is left on the superseded directory: {seen}",
        )
