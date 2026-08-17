# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the UDS predecessor-release wait (restart churn fix).

During a daemon restart the outgoing instance can hold the UDS at
``~/.kiss/sorcar.sock`` live for tens of seconds while it finishes
tunnel cleanup.  The old behaviour made the incoming daemon give up on
the UDS for its whole lifetime the moment it saw a live owner; the VS
Code extension's health probe then read ``sock-missing`` and answered
with yet another daemon restart on every window activation — visible
to the user as a recurring "KISS Sorcar Server is starting ..."
screen.

``RemoteAccessServer._setup_server`` now waits up to
``uds_owner_wait_s`` seconds for the live owner to release the
pathname before falling back to WSS-only.  These tests drive that
logic end-to-end with a real predecessor UDS listener — no mocks.
"""

from __future__ import annotations

import asyncio
import shutil
import stat
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


async def _noop_uds_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
) -> None:
    """Predecessor connection handler: accept and hold the connection."""
    try:
        await reader.read(1)
    except Exception:
        pass
    finally:
        writer.close()


class TestUdsOwnerWait(IsolatedAsyncioTestCase):
    """The new daemon must wait out a predecessor that is shutting down."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.certfile = str(certfile)
        self.keyfile = str(keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server: RemoteAccessServer | None = None
        self.predecessor: asyncio.Server | None = None

    async def asyncTearDown(self) -> None:
        if self.server is not None:
            await self.server.stop_async()
        if self.predecessor is not None:
            self.predecessor.close()
            await self.predecessor.wait_closed()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_server(self, uds_owner_wait_s: float) -> RemoteAccessServer:
        return RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=self.certfile,
            keyfile=self.keyfile,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
            uds_owner_wait_s=uds_owner_wait_s,
        )

    async def test_binds_uds_after_predecessor_releases_it(self) -> None:
        """A live predecessor that exits mid-wait must not cost the UDS.

        The predecessor closes its listener ~1.2s into the new
        daemon's startup, leaving a stale socket file behind (exactly
        what a dying daemon whose cleanup is interrupted leaves).  The
        new daemon must unlink the stale file and bind its own UDS.
        """
        self.predecessor = await asyncio.start_unix_server(
            _noop_uds_client, path=str(self.uds_path),
        )
        self.server = self._make_server(uds_owner_wait_s=15.0)

        async def release_after_delay() -> None:
            await asyncio.sleep(1.2)
            assert self.predecessor is not None
            self.predecessor.close()
            await self.predecessor.wait_closed()

        start_task = asyncio.ensure_future(self.server.start_async())
        release_task = asyncio.ensure_future(release_after_delay())
        try:
            await asyncio.wait_for(start_task, timeout=30.0)
        finally:
            await release_task

        self.assertIsNotNone(self.server._uds_server)
        self.assertTrue(self.uds_path.exists())
        self.assertTrue(stat.S_ISSOCK(self.uds_path.stat().st_mode))
        # The bound UDS is really ours: a fresh client can connect.
        _reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
        )
        writer.close()
        await writer.wait_closed()

    async def test_gives_up_without_stealing_a_persistent_owner(self) -> None:
        """A genuinely concurrent daemon keeps its UDS (F4-03).

        When the owner stays live past the deadline, the new daemon
        must fall back to WSS-only and leave the owner's listener
        untouched.
        """
        self.predecessor = await asyncio.start_unix_server(
            _noop_uds_client, path=str(self.uds_path),
        )
        self.server = self._make_server(uds_owner_wait_s=1.0)
        await asyncio.wait_for(self.server.start_async(), timeout=30.0)

        self.assertIsNone(self.server._uds_server)
        # The predecessor's listener was not stolen or unlinked: it
        # still accepts fresh connections on the same pathname.
        _reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
        )
        writer.close()
        await writer.wait_closed()
