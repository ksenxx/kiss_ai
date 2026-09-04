# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the "restart complete" toast fires only for a marker actually consumed.

``_maybe_schedule_server_reset_complete`` promised an at-most-once
"KISS Sorcar web server restart complete" toast, but it scheduled the
broadcast whenever the marker path merely *existed*: an unlinkable
marker (read-only parent) or a directory at the path re-announced a
successful restart on EVERY daemon start, and a marker holding junk
instead of the JSON ``_write_server_reset_flag`` wrote was consumed
and announced too.  The consumer must claim the marker atomically,
validate its JSON and only then schedule the toast.

Each test pre-creates a marker of one kind next to the url file,
starts a real daemon, connects a real UDS client and observes whether
the toast is broadcast (a probe command bounds the wait).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import stat
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, skipIf

import kiss.agents.sorcar.persistence as th
import kiss.server.web_server as web_server_mod
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert

_VALID_MARKER = json.dumps({"requested_at": 0.0, "conn_id": ""})


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestResetMarkerClaim(IsolatedAsyncioTestCase):
    """Only a claimed, valid marker produces the restart-complete toast."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-reset-claim-")
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir()
        self._saved_th = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self._saved_delay = web_server_mod._SERVER_RESET_COMPLETE_DELAY
        web_server_mod._SERVER_RESET_COMPLETE_DELAY = 0.1
        self.url_dir = Path(self.tmpdir) / "url"
        self.url_dir.mkdir()
        self.flag_path = self.url_dir / web_server_mod._SERVER_RESET_FLAG_NAME
        self.certfile = Path(self.tmpdir) / "cert.pem"
        self.keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(self.certfile, self.keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server: RemoteAccessServer | None = None
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        web_server_mod._SERVER_RESET_COMPLETE_DELAY = self._saved_delay
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        if self.server is not None:
            await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_th
        os.chmod(self.url_dir, stat.S_IRWXU)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _start(self) -> RemoteAccessServer:
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            certfile=str(self.certfile),
            keyfile=str(self.keyfile),
            url_file=self.url_dir / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        return self.server

    async def _saw_complete_toast(self) -> bool:
        """Connect, outwait the (shrunk) toast delay, then probe.

        The toast, when scheduled, is broadcast to every connection
        present 0.1 s after startup; a ``activeTasksQuery`` probe sent
        well after that bounds the read loop.
        """
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        await asyncio.sleep(0.6)
        writer.write(
            json.dumps({"type": "activeTasksQuery"}).encode("utf-8") + b"\n",
        )
        await writer.drain()
        for _ in range(200):
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            self.assertTrue(line, "UDS closed unexpectedly")
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if (
                msg.get("type") == "notification"
                and msg.get("id") == "server-reset-complete"
            ):
                return True
            if msg.get("type") == "activeTasksResponse":
                return False
        self.fail("probe reply never arrived")

    def _url_dir_entries(self) -> list[str]:
        return sorted(
            p.name for p in self.url_dir.iterdir() if p.name != "remote-url.json"
        )

    async def test_valid_marker_is_claimed_once_and_announced(self) -> None:
        self.flag_path.write_text(_VALID_MARKER, encoding="utf-8")
        await self._start()
        self.assertTrue(await self._saw_complete_toast())
        # Marker consumed; no claimed-copy residue either.
        self.assertEqual(self._url_dir_entries(), [])

    async def test_directory_at_marker_path_is_not_announced(self) -> None:
        self.flag_path.mkdir()
        (self.flag_path / "occupant").write_text("x", encoding="utf-8")
        await self._start()
        self.assertFalse(
            await self._saw_complete_toast(),
            "BUG: a directory at the marker path (never consumable) "
            "announced 'restart complete'",
        )
        self.assertTrue((self.flag_path / "occupant").is_file(), "not ours to delete")

    async def test_malformed_marker_is_removed_but_not_announced(self) -> None:
        self.flag_path.write_text("{not json", encoding="utf-8")
        await self._start()
        self.assertFalse(
            await self._saw_complete_toast(),
            "BUG: a marker that is not the JSON the daemon writes was "
            "announced as a completed restart",
        )
        self.assertEqual(self._url_dir_entries(), [], "junk marker must be consumed")

    async def test_non_object_json_marker_is_not_announced(self) -> None:
        self.flag_path.write_text("[1, 2]", encoding="utf-8")
        await self._start()
        self.assertFalse(await self._saw_complete_toast())
        self.assertEqual(self._url_dir_entries(), [])

    @skipIf(os.geteuid() == 0, "root bypasses directory write permission")
    async def test_unclaimable_marker_is_not_announced(self) -> None:
        # The url file shares the marker's directory and is written at
        # startup, so the directory is made read-only AFTER the daemon
        # is up and the consumer is re-run directly — the same call
        # ``_setup_server`` makes once the listeners are bound.
        server = await self._start()
        self.flag_path.write_text(_VALID_MARKER, encoding="utf-8")
        os.chmod(self.url_dir, stat.S_IRUSR | stat.S_IXUSR)
        try:
            server._maybe_schedule_server_reset_complete()
            self.assertFalse(
                await self._saw_complete_toast(),
                "BUG: a marker the daemon could not consume announced "
                "'restart complete' (and would again on every start)",
            )
            self.assertTrue(self.flag_path.is_file())
        finally:
            os.chmod(self.url_dir, stat.S_IRWXU)
