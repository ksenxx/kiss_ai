# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for remaining uncovered branches in sorcar/ and vscode/.

Targets:
  persistence.py: lines 125→129, 380→385, 403→404
  helpers.py: lines 133→134, 157→158
  server.py: lines 239→237, 241→237, 466→467, 670→671, 622 (remove pragma)

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

from kiss.agents.sorcar import persistence as th

_SavedState = tuple[Path, "sqlite3.Connection | None", Path]


def _redirect(tmpdir: str) -> _SavedState:
    """Redirect persistence to temp dir and return saved state."""
    old: _SavedState = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved: _SavedState) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestGetDbWithExistingFile:
    """Cover the else branch of 'if not _DB_PATH.exists()' (line 125→129)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_db_when_file_already_exists(self) -> None:
        """_get_db skips stale WAL cleanup when DB file already exists."""
        th._get_db()
        assert th._DB_PATH.exists()
        th._db_conn.close()  # type: ignore[union-attr]
        th._db_conn = None
        db = th._get_db()
        assert db is not None


class TestAppendChatEventNoTask:
    """Cover 'if resolved_task_id is None: return' (line 403→404)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_append_event_no_matching_task(self) -> None:
        """_append_chat_event returns early when task doesn't exist."""
        th._append_chat_event({"type": "test"}, task="nonexistent-task-xyz")
