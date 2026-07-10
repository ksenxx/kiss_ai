# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

import unittest
from pathlib import Path

from kiss.agents.sorcar import persistence


class TestPersistence(unittest.TestCase):

    def setUp(self):
        self._orig_db_path = persistence._DB_PATH
        # ``addCleanup`` runs even when setUp itself raises, unlike tearDown.
        # Register it before mutating the module-global path so no failure in
        # ``_close_db`` can leak the in-memory redirect into later tests.
        self.addCleanup(self._restore_db_path)
        persistence._DB_PATH = Path(":memory:")
        persistence._close_db()

    def _restore_db_path(self):
        persistence._DB_PATH = self._orig_db_path
        persistence._close_db()

    def test_append_chat_event_no_task(self):
        # This test should not raise an exception
        persistence._append_chat_event(event={}, task_id="999")

if __name__ == "__main__":
    unittest.main()
