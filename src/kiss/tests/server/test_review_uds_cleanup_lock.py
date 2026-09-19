# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Old-daemon UDS cleanup must obey the startup sidecar flock (review finding 1).

``_unlink_own_uds_socket``'s ownership test (``stat`` inode witness) and
its pathname ``unlink`` are two separate syscalls.  Startup serializes
its own probe → unlink → bind sequence with an exclusive flock on the
``<socket>.lock`` sidecar (C-RC3), but the cleanup used to run OUTSIDE
that protocol: an old daemon could pass its witness check, a successor
holding the flock could rebind the pathname, and the old daemon's
resumed ``unlink`` would then remove the successor's live listener.

The fix makes cleanup take the same sidecar flock (non-blocking) around
the check-and-unlink pair and SKIP the unlink when the lock is
contended — a contender is inside the startup protocol and removes any
stale pathname itself.

These tests drive a real :class:`RemoteAccessServer` bound to a
temporary UDS path — no mocks.  The pre-fix interleaving was confirmed
by temporarily inserting a <0.1s sleep between the witness ``stat`` and
the ``unlink`` (removed again): the successor's freshly bound socket
was unlinked.  The permanent regression discriminator is
``test_cleanup_skips_unlink_while_sidecar_lock_is_contended``: the
pre-fix cleanup ignored the flock and unlinked the pathname while a
concurrent starter held the lock.

Branch-coverage notes (unreachable-without-doubles exceptions):

* ``unlink`` raising ``FileNotFoundError`` requires the pathname to
  vanish between the witness ``stat`` and the ``unlink`` while THIS
  process holds the exclusive sidecar flock — the very interleaving
  the lock forbids for lock-abiding daemons; only an unrelated rogue
  ``rm`` could trigger it, which no real end-to-end setup can schedule
  deterministically.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import stat as stat_mod
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import is_root

# The sidecar lock is fcntl.flock on the UDS path: POSIX only, like the
# Unix-domain socket it guards.
fcntl = pytest.importorskip("fcntl")


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


class TestUdsCleanupSidecarLock(IsolatedAsyncioTestCase):
    """Cleanup participates in the C-RC3 sidecar-flock protocol."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-uds-cleanup-lock-")
        self.saved = _redirect_persistence(self.tmpdir)

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)

        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.lock_path = Path(self.tmpdir) / "sorcar.sock.lock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        # Restore permissions a test may have narrowed so teardown
        # can remove the tree.
        os.chmod(self.tmpdir, 0o700)
        if self.lock_path.is_dir():
            self.lock_path.rmdir()
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _close_own_listener(self) -> None:
        """Close the daemon's UDS listener, leaving the pathname behind.

        Reproduces the state an old daemon is in when it enters
        ``_unlink_own_uds_socket`` during shutdown/rollback: listener
        closed, socket file still present, inode witness recorded.
        """
        assert self.server._uds_server is not None
        self.server._uds_server.close()
        await self.server._uds_server.wait_closed()
        self.assertTrue(self.uds_path.exists())
        self.assertIsNotNone(self.server._uds_inode)

    async def test_cleanup_skips_unlink_while_sidecar_lock_is_contended(
        self,
    ) -> None:
        """Contended flock ⇒ fail closed, even when the witness matches.

        A holder of the sidecar lock is mid probe → unlink → bind; the
        old daemon must not interleave its own unlink with that
        sequence (pre-fix it did: the witness still matched, so the
        pathname was unlinked while the lock was held elsewhere).
        """
        await self._close_own_listener()
        with open(self.lock_path, "w", encoding="utf-8") as contender:
            fcntl.flock(contender, fcntl.LOCK_EX)
            self.server._unlink_own_uds_socket()
            # Pre-fix: the pathname was gone here.
            self.assertTrue(self.uds_path.exists())
        # Lock released and the pathname still names our own socket:
        # cleanup now completes its job.
        self.server._unlink_own_uds_socket()
        self.assertFalse(self.uds_path.exists())

    async def test_cleanup_never_unlinks_a_successors_rebound_socket(
        self,
    ) -> None:
        """A successor's rebind under the flock survives our cleanup."""
        await self._close_own_listener()
        # Successor B: under the sidecar lock, unlink the stale
        # pathname and bind a fresh listener to it — exactly what
        # ``_setup_server`` does.
        async def _noop(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        ) -> None:
            writer.close()

        with open(self.lock_path, "w", encoding="utf-8") as successor:
            fcntl.flock(successor, fcntl.LOCK_EX)
            self.uds_path.unlink()
            successor_server = await asyncio.start_unix_server(
                _noop, path=str(self.uds_path),
            )
            fcntl.flock(successor, fcntl.LOCK_UN)
        try:
            # Old daemon's cleanup: the lock is free, but the inode
            # witness no longer matches — the pathname must survive.
            self.server._unlink_own_uds_socket()
            self.assertTrue(self.uds_path.exists())
            self.assertTrue(
                stat_mod.S_ISSOCK(os.stat(self.uds_path).st_mode),
            )
            # The successor's listener is still connectable.
            _reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self.uds_path)),
                timeout=2.0,
            )
            writer.close()
            await writer.wait_closed()
        finally:
            successor_server.close()
            await successor_server.wait_closed()

    async def test_cleanup_fails_closed_without_an_inode_witness(self) -> None:
        """No recorded bind inode ⇒ never unlink the shared pathname."""
        await self._close_own_listener()
        self.server._uds_inode = None
        self.server._unlink_own_uds_socket()
        self.assertTrue(self.uds_path.exists())

    async def test_cleanup_tolerates_an_already_missing_pathname(self) -> None:
        """A vanished pathname makes the witness ``stat`` fail: no-op."""
        await self._close_own_listener()
        self.uds_path.unlink()
        self.server._unlink_own_uds_socket()  # must not raise
        self.assertFalse(self.uds_path.exists())

    async def test_cleanup_fails_closed_when_lock_file_is_unopenable(
        self,
    ) -> None:
        """An unopenable sidecar ⇒ cannot join the protocol ⇒ no unlink."""
        await self._close_own_listener()
        self.lock_path.unlink()
        self.lock_path.mkdir()  # open(..., "w") now raises IsADirectoryError
        try:
            self.server._unlink_own_uds_socket()
            self.assertTrue(self.uds_path.exists())
        finally:
            self.lock_path.rmdir()

    @unittest.skipIf(is_root(), "root bypasses directory permissions")
    async def test_cleanup_logs_and_survives_an_unlink_error(self) -> None:
        """EACCES on the unlink itself is swallowed (path left behind)."""
        await self._close_own_listener()
        os.chmod(self.tmpdir, 0o500)  # no write ⇒ unlink fails
        try:
            self.server._unlink_own_uds_socket()  # must not raise
            self.assertTrue(self.uds_path.exists())
        finally:
            os.chmod(self.tmpdir, 0o700)
