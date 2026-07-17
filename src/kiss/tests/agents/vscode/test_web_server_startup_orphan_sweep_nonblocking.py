# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""kiss-web restart after ``install.sh`` must not be delayed by sorcar.db.

Production symptom
==================
After running ``install.sh`` (or the Update button), the freshly
respawned ``kiss-web`` daemon took a long time to come back up: the
"KISS Sorcar Server is starting ..." overlay lingered and install.sh's
bounded wait for ``~/.kiss/sorcar.sock`` timed out.

Root cause
==========
``RemoteAccessServer.__init__`` constructs a ``VSCodeServer``, whose
``__init__`` ran the orphan-task recovery sweep
(:func:`kiss.agents.sorcar.persistence._recover_orphaned_tasks`)
**synchronously** — an UPDATE on ``task_history`` behind the
persistence write lock, against connections configured with
``PRAGMA busy_timeout=30000``.  All of this happened BEFORE
``_setup_server`` bound the UDS / WSS listeners.  During an install
restart the previous daemon is SIGTERM/SIGKILLed mid-write, so the
new daemon regularly found the SQLite lock still held (straggler
process flushing, WAL recovery on a multi-GB database) and blocked
for up to ~30 s inside ``__init__`` — with no listening socket the
whole time.

Fix
===
``VSCodeServer.__init__`` starts the sweep on a background daemon
thread (``orphan-task-sweep``).  Startup binds the sockets promptly;
the sweep completes — and still rewrites the sentinel rows — as soon
as the database lock becomes available.

These are end-to-end tests: a real ``RemoteAccessServer`` is
constructed against a real on-disk SQLite database whose write lock
is genuinely held by a second connection (exactly what a dying
previous daemon does), and readiness is probed by connecting to the
real Unix-domain socket.
"""

from __future__ import annotations

import asyncio
import socket
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as _persistence
from kiss.server.web_server import RemoteAccessServer

_SENTINEL = "Agent Failed Abruptly"
_RECOVERED = "Task terminated unexpectedly (process killed)"

# The freshly-started server must accept UDS connections well within
# this budget even while the database write lock is held.  Before the
# fix, startup blocked for the full 30 s ``busy_timeout`` (or longer),
# so this bound cleanly separates the two behaviours while leaving
# ample headroom for slow CI machines.
_STARTUP_BUDGET_SECS = 12.0


def _find_free_port() -> int:
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _row_result(db_path: Path, task_id: str) -> str:
    """Read ``task_history.result`` for *task_id* via a fresh connection."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT result FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, f"task_history row {task_id} missing"
    return str(row[0])


class StartupNotBlockedByLockedDbTest(IsolatedAsyncioTestCase):
    """Server must bind its sockets while sorcar.db is write-locked."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-orphan-sweep-")
        tmp = Path(self.tmpdir)
        kiss_dir = tmp / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        # Point the sorcar persistence layer at a per-test database.
        self._saved = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None
        self.db_path = kiss_dir / "sorcar.db"
        self.uds_path = tmp / "sorcar-test.sock"
        self.url_file = tmp / "remote-url.json"
        self.server: RemoteAccessServer | None = None
        self.locker: sqlite3.Connection | None = None

    async def asyncTearDown(self) -> None:
        if self.server is not None:
            await self.server.stop_async()
        if self.locker is not None:
            try:
                self.locker.rollback()
            except sqlite3.Error:
                pass
            self.locker.close()
        _persistence._close_db()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved

    async def test_startup_binds_uds_while_db_write_locked(self) -> None:
        """End-to-end reproduction of the slow install-restart.

        1. Seed the database with an orphaned task row (the sentinel a
           killed daemon leaves behind — precisely the state after
           ``install.sh`` kills the old daemon mid-task).
        2. Hold the SQLite write lock from a second connection, playing
           the role of the dying previous daemon's final writes.
        3. Start a real ``RemoteAccessServer`` and require the UDS
           listener to accept a connection within
           ``_STARTUP_BUDGET_SECS`` — before the fix this took the
           full 30 s ``busy_timeout``.
        4. Release the lock and verify the background sweep still
           recovers the orphaned row.
        """
        # 1. Seed an orphaned ("Agent Failed Abruptly") row.
        orphan_id, _chat = _persistence._add_task(
            "orphan left by daemon killed during install.sh",
            chat_id="install-restart-chat",
        )
        assert _row_result(self.db_path, orphan_id) == _SENTINEL
        # Drop the cached thread-local connection so only the explicit
        # locker below holds the file.
        _persistence._close_db()

        # 2. Hold the write lock, as the dying previous daemon would.
        self.locker = sqlite3.connect(
            str(self.db_path), isolation_level=None,
        )
        self.locker.execute("BEGIN IMMEDIATE")

        # 3. Construct + start the real server under the held lock.
        started = time.monotonic()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            use_tunnel=False,
            url_file=self.url_file,
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        reader = writer = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self.uds_path)),
                timeout=_STARTUP_BUDGET_SECS,
            )
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass
        elapsed = time.monotonic() - started
        assert elapsed < _STARTUP_BUDGET_SECS, (
            f"server took {elapsed:.1f}s to accept a UDS connection while "
            f"sorcar.db was write-locked — startup is blocked on the "
            f"orphan-task sweep (pre-fix behaviour: ~30s busy_timeout)"
        )
        # The sweep must NOT have completed yet: the lock is still held,
        # proving the sweep genuinely contends for the database and the
        # fast startup was not a fluke of an idle sweep.
        assert _row_result(self.db_path, orphan_id) == _SENTINEL, (
            "orphan row was rewritten while the write lock was held"
        )

        # 4. Release the lock; the background sweep must finish the
        # recovery on its own.
        self.locker.rollback()
        self.locker.close()
        self.locker = None
        sweep = self.server._vscode_server._orphan_sweep_thread
        assert sweep is not None, "orphan sweep thread was not started"
        await asyncio.to_thread(sweep.join, 60)
        assert not sweep.is_alive(), "orphan sweep never finished"
        assert _row_result(self.db_path, orphan_id) == _RECOVERED, (
            "background sweep failed to recover the orphaned row"
        )

    async def test_sweep_recovers_orphans_without_contention(self) -> None:
        """Happy path: with no lock held, startup is fast AND the
        orphaned row is recovered shortly after boot — the deferred
        sweep must not lose the recovery semantics the synchronous
        sweep provided.
        """
        orphan_id, _chat = _persistence._add_task(
            "orphan recovered on uncontended boot",
            chat_id="install-restart-chat-2",
        )
        _persistence._close_db()

        started = time.monotonic()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            use_tunnel=False,
            url_file=self.url_file,
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        elapsed = time.monotonic() - started
        assert elapsed < _STARTUP_BUDGET_SECS, (
            f"uncontended startup took {elapsed:.1f}s"
        )
        sweep = self.server._vscode_server._orphan_sweep_thread
        assert sweep is not None, "orphan sweep thread was not started"
        await asyncio.to_thread(sweep.join, 60)
        assert not sweep.is_alive(), "orphan sweep never finished"
        assert _row_result(self.db_path, orphan_id) == _RECOVERED
