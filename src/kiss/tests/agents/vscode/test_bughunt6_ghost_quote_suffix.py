# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: ghost suggestions must not quote-strip history suffixes.

``clip_autocomplete_suggestion`` kept a vestigial "strip surrounding
quotes" step from the era when suggestions came from an LLM.  Both of
its call sites now pass a *continuation suffix* of a real string — a
prefix-matched history task minus the typed query, or an identifier
candidate minus the typed partial — so quote characters at the suffix
boundary are REAL characters of the matched task, not LLM decoration:

* history ``run "make test"`` typed as ``run "make`` → the correct
  continuation is `` test"``, but the trailing ``"`` was stripped, so
  accepting the ghost typed ``run "make test`` — an unbalanced-quote
  string the user never submitted;
* history ``echo "hi" done`` typed as ``echo `` → the correct
  continuation is ``"hi" done``, but the leading ``"`` was stripped,
  so accepting produced ``echo hi" done``.

The whole point of history-prefix ghost text is that accepting it
reproduces the matched history task exactly (the iteration-3
echo-strip fix established the same invariant for suffixes that begin
with the query).

These tests run the real pipeline: a real task row in the persistence
DB, ``_prefix_match_task`` via the server's ``_complete``, and the
broadcast ``ghost`` event.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _RunningAgentState


def _redirect_persistence(tmpdir: str) -> tuple[Any, Any, Any]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_persistence(saved: tuple[Any, Any, Any]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


class TestGhostSuffixQuotesSurvive(unittest.TestCase):
    """History-suffix ghost text must reproduce the task byte-exact."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment,method-assign]

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _ghost_suggestion(self, query: str) -> str:
        self.server._complete(query)
        ghosts = [
            e for e in self.events
            if e.get("type") == "ghost" and e.get("query") == query
        ]
        self.assertTrue(ghosts, f"no ghost event for {query!r}: {self.events!r}")
        return str(ghosts[-1].get("suggestion", ""))

    def test_trailing_quote_in_suffix_survives(self) -> None:
        history_task = 'run "make test"'
        th._add_task(history_task)
        query = 'run "make'
        suggestion = self._ghost_suggestion(query)
        self.assertEqual(
            query + suggestion, history_task,
            "accepting the ghost must reproduce the matched history task "
            f"exactly; got {(query + suggestion)!r}",
        )

    def test_leading_quote_in_suffix_survives(self) -> None:
        history_task = 'echo "hi" done'
        th._add_task(history_task)
        query = "echo "
        suggestion = self._ghost_suggestion(query)
        self.assertEqual(
            query + suggestion, history_task,
            "accepting the ghost must reproduce the matched history task "
            f"exactly; got {(query + suggestion)!r}",
        )

    def test_gap_normalisation_still_applies(self) -> None:
        """Regression guard: double-space collapse from iter-3 still works."""
        th._add_task("parse  arguments")
        suggestion = self._ghost_suggestion("parse")
        self.assertEqual(suggestion, " arguments")


if __name__ == "__main__":
    unittest.main()
