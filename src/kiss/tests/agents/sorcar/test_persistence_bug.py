# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Appending a chat event for a task that has no row must not raise."""

import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar import persistence


class TestPersistence(unittest.TestCase):
    """Events for unknown tasks are dropped, quietly and without error."""

    def setUp(self):
        """Point persistence at a throwaway database file.

        A real file rather than ``":memory:"``: the module resolves the
        database path (and the sidecars beside it) relative to the
        process's working directory, so an in-memory name littered the
        checkout with journal artifacts.
        """
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-persistence-bug-"))
        self._orig_db_path = persistence._DB_PATH
        self.addCleanup(self._restore_db_path)
        persistence._DB_PATH = self.tmpdir / "sorcar.db"
        persistence._close_db()

    def _restore_db_path(self):
        """Restore the module's database path and drop the temp dir."""
        persistence._DB_PATH = self._orig_db_path
        persistence._close_db()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_append_chat_event_no_task(self):
        """An event naming a task that was never persisted is a no-op."""
        persistence._append_chat_event(event={}, task_id="999")
        persistence._flush_chat_events("999")
        self.assertFalse(persistence._task_has_events("999"))


if __name__ == "__main__":
    unittest.main()
