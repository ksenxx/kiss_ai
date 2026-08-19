# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests targeting uncovered branches in agents/sorcar/.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import pytest


class TestTaskHistory(TestCase):
    def setUp(self) -> None:
        from kiss.agents.sorcar import persistence as th
        self.th = th
        self.tmpdir = Path(tempfile.mkdtemp())
        kiss_dir = self.tmpdir / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

    def tearDown(self) -> None:
        from kiss.agents.sorcar import persistence as th

        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self._saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_search_history_empty_query(self) -> None:
        th = self.th
        th._add_task("task one")
        results = th._search_history("", limit=10)
        assert len(results) >= 1


class TestUsefulTools(TestCase):
    def test_write_and_read(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            result = tools.Write(path, "hello world")
            assert "Successfully" in result
            assert tools.Read(path) == "hello world"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
