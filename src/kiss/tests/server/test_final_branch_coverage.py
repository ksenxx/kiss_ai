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
import tempfile

from kiss.agents.sorcar import persistence as th
from kiss.server.helpers import rank_file_suggestions
from kiss.server.server import VSCodeServer
from kiss.tests.agents.sorcar.test_final_branch_coverage import (  # noqa: F401
    _redirect,
    _restore,
    _SavedState,
)


class TestRankFileSuggestionsWithUsage:
    """Cover usage.get(path, 0) > 0 True branch and frequent loop."""

    def test_frequent_files_with_query(self) -> None:
        """Frequent files are filtered by query and sorted by end distance."""
        files = ["src/main.py", "src/main_test.py", "lib/main.py"]
        usage = {"src/main.py": 3, "lib/main.py": 1}
        result = rank_file_suggestions(files, "main", usage)
        frequent = [r for r in result if r["type"] == "frequent"]
        assert len(frequent) == 2


class TestGetHistoryBranches:
    """Cover _get_history branches for both empty and populated DB."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_server(self) -> tuple[VSCodeServer, list[dict]]:
        server = VSCodeServer()
        events: list[dict] = []
        orig = server.printer.broadcast

        def cap(ev: dict) -> None:
            events.append(ev)
            orig(ev)

        server.printer.broadcast = cap  # type: ignore[assignment]
        return server, events

    def test_get_history_with_entries(self) -> None:
        """_get_history with populated DB enters the loop (line 466→467)."""
        server, events = self._make_server()
        th._add_task("short task")
        th._add_task("a" * 60)
        server._get_history(None, offset=0, generation=0)
        hist = [e for e in events if e.get("type") == "history"]
        assert len(hist) == 1
        sessions = hist[0]["sessions"]
        assert len(sessions) == 2
        long_session = [s for s in sessions if len(s["preview"]) > 50][0]
        assert long_session["title"] == "a" * 60
        assert not long_session["title"].endswith("...")

    def test_get_history_with_query(self) -> None:
        """_get_history with search query filters entries."""
        server, events = self._make_server()
        th._add_task("fix the bug")
        th._add_task("add feature")
        server._get_history("bug", offset=0, generation=0)
        hist = [e for e in events if e.get("type") == "history"]
        assert len(hist) == 1
        sessions = hist[0]["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["preview"] == "fix the bug"


class TestActiveFileMatchesEqualLength:
    """Equal-length identifier candidates keep a stable order.

    ``_active_file_identifier_matches`` sorts longest-first with an
    alphabetical tie-breaker, so two equal-length candidates must both
    be returned, alphabetically ordered.
    """

    def test_equal_length_candidates(self) -> None:
        """Two equal-length candidates: both returned, alphabetical."""
        server = VSCodeServer()
        content = "method_ab method_cd"
        matches = server._active_file_identifier_matches(
            "x method_", snapshot_content=content
        )
        assert matches == ["method_ab", "method_cd"]
