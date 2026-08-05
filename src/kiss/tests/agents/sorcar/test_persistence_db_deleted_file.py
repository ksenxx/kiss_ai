# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: persistence survives deletion of the DB file.

Before the fix, ``_get_db()``'s fast path returned the cached
per-thread connection without checking that the database file still
existed on disk (or was still the SAME file).  If the file was deleted
while a cached connection was open (log rotation, a test cleaning up
``$KISS_HOME``, a user removing ``~/.kiss/sorcar.db`` while the daemon
runs) — and possibly recreated at the same pathname by an independent
``sqlite3.connect`` — every subsequent write went into the orphaned
inode and silently disappeared, while any NEW reader of ``_DB_PATH``
connected to a fresh empty database and failed with
``sqlite3.OperationalError: no such table: task_history``.

This was the shared root cause behind the order-dependent failures of
the wave-2/wave-3 runner tests (``test_f2_*`` / ``test_b2_*``) when
they ran in a large split after a test that removed the redirected DB
file without invalidating the cached connection.

No mocks/patches/fakes: the real persistence layer, a real sqlite
database file, and a real ``os.unlink``.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import _add_task


class TestDbFileDeletedUnderCachedConnection:
    """Writes after an external DB-file deletion must be durable."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss_dbdel_")
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        th._close_db()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _delete_db_files(self) -> None:
        """Remove the DB file and its WAL/SHM side files, keeping the
        cached per-thread connection open (no ``_close_db()``)."""
        for suffix in ("", "-wal", "-shm"):
            path = th._DB_PATH.with_name(th._DB_PATH.name + suffix)
            path.unlink(missing_ok=True)

    def test_write_after_delete_recreates_schema_and_is_visible(self) -> None:
        _add_task("dbdel first", chat_id="")
        assert th._DB_PATH.exists()

        self._delete_db_files()
        assert not th._DB_PATH.exists()

        _add_task("dbdel second", chat_id="")
        assert th._DB_PATH.exists()

        conn = sqlite3.connect(str(th._DB_PATH))
        try:
            rows = conn.execute(
                "SELECT task FROM task_history ORDER BY timestamp",
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == ["dbdel second"]

    def test_reads_also_heal_after_delete(self) -> None:
        _add_task("dbdel read-heal", chat_id="")
        self._delete_db_files()
        entries = th._load_history()
        assert entries == []

    def test_same_path_recreated_file_is_detected(self) -> None:
        _add_task("dbdel recreate first", chat_id="")
        self._delete_db_files()
        sqlite3.connect(str(th._DB_PATH)).close()
        assert th._DB_PATH.exists()

        _add_task("dbdel recreate second", chat_id="")

        conn = sqlite3.connect(str(th._DB_PATH))
        try:
            rows = conn.execute(
                "SELECT task FROM task_history ORDER BY timestamp",
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == ["dbdel recreate second"]

    def test_reads_heal_after_same_path_recreation(self) -> None:
        _add_task("dbdel recreate read-heal", chat_id="")
        self._delete_db_files()
        sqlite3.connect(str(th._DB_PATH)).close()
        entries = th._load_history()
        assert entries == []
