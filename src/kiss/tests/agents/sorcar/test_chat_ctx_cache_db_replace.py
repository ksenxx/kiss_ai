# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Chat-context text cache must not survive an on-disk DB replacement.

``_get_db()`` detects that the file at ``_DB_PATH`` was deleted or
replaced on disk via its ``(st_dev, st_ino)`` identity and reconnects,
and ``_maybe_reset_caches`` clears the background event writer's
``_next_seq_cache`` / ``_marked_has_events`` for exactly that reason
(the module's own threat model: "a user removing ``~/.kiss/sorcar.db``
while the daemon runs").

The autocomplete chat-context text cache
(``_chat_context_text_cache``) was NOT part of that invalidation:
after the database file is deleted (or swapped) on disk,
``_load_chat_context_text(chat_id)`` kept serving the deleted
database's task/result text as ghost-text context — and because the
cache is checked BEFORE any database access, no later ``_get_db()``
reconnect could ever heal it for a cached ``chat_id``.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _load_chat_context_text,
    _save_task_result,
)


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        th._invalidate_chat_context_cache("")
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestChatContextCacheDbReplace(_TempDbTestBase):
    """The stale-cache-after-external-DB-removal bug and its guard rails."""

    def _delete_db_files_externally(self) -> None:
        """Simulate ``rm ~/.kiss/sorcar.db*`` by an external process.

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

        # The recreated (empty) database has no rows for this chat, so
        # the context text must be empty — not the deleted DB's text.
        assert _load_chat_context_text(chat_id) == ""

    def test_no_stale_text_after_db_file_replaced(self) -> None:
        """A REPLACED file (same pathname, different inode/content)
        must also drop the cached text, even though the path exists."""
        task_id, chat_id = _add_task("old-db task")
        _save_task_result("old-db result", task_id=task_id)
        assert "old-db task" in _load_chat_context_text(chat_id)

        # Build a replacement database at a scratch path containing a
        # DIFFERENT task for the very same chat_id, then move it over
        # the live file — as an external restore-from-backup would.
        self._delete_db_files_externally()
        # Force a reconnect (new file + schema) and insert the
        # replacement row for the same chat_id via the public API.
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

        # Insert a row behind the cache's back (raw SQL bypasses the
        # _add_task invalidation hook).  A cache HIT must serve the
        # old text; a miss would expose the new row.
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "INSERT INTO task_history (id, timestamp, task, chat_id) "
                "VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, time.time(), "sneaky task", chat_id),
            )
            db.commit()
        assert _load_chat_context_text(chat_id) == first
        # And the normal invalidation path still refreshes.
        th._invalidate_chat_context_cache(chat_id)
        assert "sneaky task" in _load_chat_context_text(chat_id)
