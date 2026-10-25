# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Chat-context text cache must not survive an on-disk DB replacement.

``_get_db()`` detects that the file at ``_DB_PATH`` was deleted or
replaced on disk via its ``(st_dev, st_ino)`` identity and reconnects,
and ``_maybe_reset_caches`` clears the background event writer's
``_next_seq_cache`` / ``_marked_has_events`` for exactly that reason
(the module's own threat model: "a user removing ``~/.kiss/history.db``
while the daemon runs").

The autocomplete chat-context text cache
(``_chat_context_text_cache``) was NOT part of that invalidation:
after the database file is deleted (or swapped) on disk,
``_load_chat_context_text(chat_id)`` kept serving the deleted
database's task/result text as ghost-text context — and because the
cache is checked BEFORE any database access, no later ``_get_db()``
reconnect could ever heal it for a cached ``chat_id``.

The converse is pinned too: a TRANSIENT ``stat`` failure (identity
``_FILE_ID_UNKNOWN``, "assume unchanged") must not reset the caches
from this path either, exactly as ``_get_db()`` ignores it.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _load_chat_context_text,
    _save_task_result,
)
from kiss.tests.conftest import posix_only


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        th._invalidate_chat_context_cache("")
        shutil.rmtree(self.tmpdir, ignore_errors=True)


@posix_only("unlinking a SQLite file another connection holds open")
class TestChatContextCacheDbReplace(_TempDbTestBase):
    """The stale-cache-after-external-DB-removal bug and its guard rails."""

    def _delete_db_files_externally(self) -> None:
        """Simulate ``rm ~/.kiss/history.db*`` by an external process.

        Deliberately does NOT call ``_close_db()`` or any cache
        invalidator — the daemon has no hook that runs when a user
        deletes the file out from under it.
        """
        for suffix in ("", "-wal", "-shm"):
            Path(str(th._DB_PATH) + suffix).unlink(missing_ok=True)

    def test_no_stale_text_after_db_file_deleted(self) -> None:
        """After the DB file is deleted on disk, cached chat context
        for the old database must not be served."""
        task_id, chat_id = _add_task("ctx swap task")
        _save_task_result("ctx swap result", task_id=task_id)
        text = _load_chat_context_text(chat_id)
        assert "ctx swap task" in text
        assert "ctx swap result" in text

        self._delete_db_files_externally()

        assert _load_chat_context_text(chat_id) == ""

    def test_no_stale_text_after_db_file_replaced(self) -> None:
        """A REPLACED file (same pathname, different inode/content)
        must also drop the cached text, even though the path exists."""
        task_id, chat_id = _add_task("old-db task")
        _save_task_result("old-db result", task_id=task_id)
        assert "old-db task" in _load_chat_context_text(chat_id)

        self._delete_db_files_externally()
        replacement_id, _ = _add_task("new-db task", chat_id)
        _save_task_result("new-db result", task_id=replacement_id)

        fresh = _load_chat_context_text(chat_id)
        assert "new-db task" in fresh
        assert "new-db result" in fresh
        assert "old-db task" not in fresh
        assert "old-db result" not in fresh

    def test_cache_still_serves_hits_for_unchanged_db_file(self) -> None:
        """Regression guard: for an UNCHANGED on-disk file the cache
        must still serve hits (no spurious invalidation per call)."""
        task_id, chat_id = _add_task("hit task")
        _save_task_result("hit result", task_id=task_id)
        first = _load_chat_context_text(chat_id)
        assert "hit task" in first

        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "INSERT INTO task_history (id, timestamp, task, chat_id) "
                "VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, time.time(), "sneaky task", chat_id),
            )
            db.commit()
        assert _load_chat_context_text(chat_id) == first
        th._invalidate_chat_context_cache(chat_id)
        assert "sneaky task" in _load_chat_context_text(chat_id)

    @pytest.mark.skipif(
        os.name != "posix" or os.geteuid() == 0,
        reason="needs POSIX directory permissions that a non-root user cannot bypass",
    )
    def test_transient_stat_failure_keeps_caches(self) -> None:
        """A ``stat`` that fails with EACCES is not a replaced database.

        The ghost-text path checks the file identity on every
        keystroke; an unsearchable parent directory makes that
        ``os.stat`` raise ``PermissionError``, which
        ``_db_file_identity`` reports as ``_FILE_ID_UNKNOWN``.  The seq
        caches and the text cache must survive it.
        """
        task_id, chat_id = _add_task("stat task")
        _save_task_result("stat result", task_id=task_id)
        th._queue_chat_event({"type": "system_output", "text": "hi"}, task_id)
        th._flush_chat_events(task_id)
        first = _load_chat_context_text(chat_id)
        assert "stat task" in first
        key_before = th._caches_db_key
        gen_before = th._chat_context_cache_gen
        assert key_before is not None and th._next_seq_cache.get(task_id) == 1

        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.chmod(0)
        try:
            assert th._db_file_identity(str(th._DB_PATH)) is th._FILE_ID_UNKNOWN
            assert _load_chat_context_text(chat_id) == first
        finally:
            kiss_dir.chmod(0o700)
        assert th._caches_db_key == key_before
        assert th._chat_context_cache_gen == gen_before
        assert th._next_seq_cache.get(task_id) == 1
