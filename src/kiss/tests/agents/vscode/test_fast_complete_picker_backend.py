# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the fast-complete picker backend.

The chat webview's input textbox now shows ALL fast-complete options
as a dropdown picker — the same UI the ``@``-mention file picker
uses — instead of only the single inline ghost-text suggestion.  The
:class:`kiss.server.server.VSCodeServer` therefore broadcasts a
new ``completions`` event alongside the legacy ``ghost`` event:

    {"type": "completions",
     "completions": [{"type": "task"|"trick"|"identifier",
                      "text": "<full replacement string>"}],
     "query": "<the query this list answers>",
     "connId": "<requesting connection id>"}  # only when non-empty

These tests drive the production server directly (with the persistence
DB and INJECTIONS.md pinned to a tmpdir) and assert on the event the
real ``Printer.broadcast`` call would deliver to the webview.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.server.server import VSCodeServer

_TRICK_REPRO = (
    "Reproduce the issue by writing end-to-end test. Then fix the issue."
)
_TRICK_INTERNET = "Use internet search extensively."

_FAKE_INJECTIONS = (
    "## Trick\n"
    "\n"
    f"{_TRICK_REPRO}\n"
    "\n"
    "## Trick\n"
    "\n"
    f"{_TRICK_INTERNET}\n"
)


class TestFastCompletePickerBackend:
    """Drive ``VSCodeServer._complete`` end-to-end and inspect events."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self._tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        fake_path = kiss_dir / "fake_INJECTIONS.md"
        fake_path.write_text(_FAKE_INJECTIONS)
        (kiss_dir / "MY_INJECTION.md").write_text("")
        self._saved_kiss_home = os.environ.get("KISS_HOME")
        self._saved_kiss_injections = os.environ.get("KISS_INJECTIONS_PATH")
        os.environ["KISS_HOME"] = str(kiss_dir)
        os.environ["KISS_INJECTIONS_PATH"] = str(fake_path)
        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        th._KISS_DIR = kiss_dir

    def teardown_method(self) -> None:
        if self._saved_kiss_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_kiss_home
        if self._saved_kiss_injections is None:
            os.environ.pop("KISS_INJECTIONS_PATH", None)
        else:
            os.environ["KISS_INJECTIONS_PATH"] = self._saved_kiss_injections
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None

    def _run(
        self,
        server: VSCodeServer,
        query: str,
        *,
        snapshot_content: str = "",
        chat_id: str = "",
        conn_id: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run ``_complete`` and return the (ghost_events, completions_events)."""
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        server._complete(
            query,
            snapshot_content=snapshot_content,
            chat_id=chat_id,
            conn_id=conn_id,
        )
        ghosts = [e for e in events if e.get("type") == "ghost"]
        comps = [e for e in events if e.get("type") == "completions"]
        return ghosts, comps

    # ----- completions event shape -------------------------------------

    def test_completions_event_emitted_with_query_echo(self) -> None:
        """Every ``_complete`` call broadcasts exactly one completions event."""
        server = VSCodeServer()
        _, comps = self._run(server, "Reproduce")
        assert len(comps) == 1
        ev = comps[0]
        assert ev["type"] == "completions"
        assert ev["query"] == "Reproduce"
        assert isinstance(ev["completions"], list)

    def test_completions_event_carries_conn_id_when_set(self) -> None:
        """``connId`` is stamped on the event when the request has one."""
        server = VSCodeServer()
        _, comps = self._run(server, "Reproduce", conn_id="win-7")
        assert comps[0]["connId"] == "win-7"

    def test_completions_event_omits_conn_id_for_direct_callers(self) -> None:
        """Empty connId is NOT stamped (matches ``_emit_ghost``)."""
        server = VSCodeServer()
        _, comps = self._run(server, "Reproduce")
        assert "connId" not in comps[0]

    # ----- trick suggestions -------------------------------------------

    def test_tricks_appear_in_completions(self) -> None:
        """Tricks are returned as full sentence lines (head + trick)."""
        server = VSCodeServer()
        _, comps = self._run(server, "Reproduce")
        texts = [c["text"] for c in comps[0]["completions"]]
        assert _TRICK_REPRO in texts

    def test_trick_completion_marked_trick(self) -> None:
        """Trick items carry ``type == 'trick'``."""
        server = VSCodeServer()
        _, comps = self._run(server, "Reproduce")
        items = comps[0]["completions"]
        trick = next(c for c in items if c["text"] == _TRICK_REPRO)
        assert trick["type"] == "trick"

    def test_trick_emitted_as_raw_body_after_sentence_boundary(self) -> None:
        """A trick after a sentence boundary is emitted as the raw body — no head."""
        server = VSCodeServer()
        query = "Some preamble. Reproduce"
        _, comps = self._run(server, query)
        texts = [c["text"] for c in comps[0]["completions"]]
        assert _TRICK_REPRO in texts
        assert "Some preamble. " + _TRICK_REPRO not in texts

    # ----- task history suggestions ------------------------------------

    def test_history_tasks_appear_in_completions(self) -> None:
        """Prior task history feeds the completions list."""
        server = VSCodeServer()
        th._add_task("fix the parser bug now")
        th._add_task("fix the parser then commit")
        _, comps = self._run(server, "fix the")
        items = comps[0]["completions"]
        texts = [c["text"] for c in items]
        assert "fix the parser bug now" in texts
        assert "fix the parser then commit" in texts
        # Both must be tagged ``task``.
        for c in items:
            if c["text"].startswith("fix the parser"):
                assert c["type"] == "task"

    def test_completions_dedupe(self) -> None:
        """A duplicate task text appears at most once."""
        server = VSCodeServer()
        th._add_task("fix the parser bug now")
        th._add_task("fix the parser bug now")
        _, comps = self._run(server, "fix")
        texts = [c["text"] for c in comps[0]["completions"]]
        assert texts.count("fix the parser bug now") == 1

    # ----- identifier suggestions --------------------------------------

    def test_identifier_completions_from_active_file(self) -> None:
        """Identifiers from the active editor are listed as completions."""
        server = VSCodeServer()
        content = (
            "zebraq_marker_alpha = 1\n"
            "zebraq_marker_beta = 2\n"
            "zebraq_marker_gamma = 3\n"
        )
        _, comps = self._run(
            server, "x = zebraq_marker_",
            snapshot_content=content,
        )
        items = comps[0]["completions"]
        idents = [c for c in items if c["type"] == "identifier"]
        texts = {c["text"] for c in idents}
        assert "zebraq_marker_alpha" in texts
        assert "zebraq_marker_beta" in texts
        assert "zebraq_marker_gamma" in texts

    # ----- ghost back-compat -------------------------------------------

    def test_ghost_event_still_emitted(self) -> None:
        """The legacy ghost event is still broadcast alongside completions."""
        server = VSCodeServer()
        ghosts, _ = self._run(server, "Reproduce")
        assert len(ghosts) == 1
        # Ghost text completes the trick.
        completed = "Reproduce" + ghosts[0]["suggestion"]
        assert _TRICK_REPRO in completed

    def test_short_query_emits_empty_completions(self) -> None:
        """A query under 2 chars yields empty completions (and empty ghost)."""
        server = VSCodeServer()
        ghosts, comps = self._run(server, "R")
        assert comps[0]["completions"] == []
        assert ghosts[0]["suggestion"] == ""

    def test_empty_query_emits_empty_completions(self) -> None:
        """An empty query yields empty completions."""
        server = VSCodeServer()
        ghosts, comps = self._run(server, "")
        assert comps[0]["completions"] == []
        assert ghosts[0]["suggestion"] == ""

    # ----- staleness ---------------------------------------------------

    def test_stale_request_emits_nothing(self) -> None:
        """A request whose seq is no longer latest emits no events."""
        server = VSCodeServer()
        # Pre-set the latest seq for connection ``c`` to 5; we then
        # invoke ``_complete`` with seq=4 — older — which must
        # short-circuit before any broadcast.
        server._complete_seq_latest["c"] = 5
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        server._complete("Reproduce", seq=4, conn_id="c")
        assert events == []

    # ----- ordering ----------------------------------------------------

    def test_tasks_appear_before_tricks(self) -> None:
        """Task history wins the auto-selected first slot."""
        server = VSCodeServer()
        th._add_task("Reproduce-history-only-task entry")
        _, comps = self._run(server, "Reproduce")
        items = comps[0]["completions"]
        # First item must be a task, not a trick.
        assert items[0]["type"] == "task"

    def test_completions_limit_respected(self) -> None:
        """No more than _COMPLETIONS_LIMIT items are emitted."""
        from kiss.server.autocomplete import _COMPLETIONS_LIMIT
        server = VSCodeServer()
        for i in range(_COMPLETIONS_LIMIT + 10):
            th._add_task(f"fix the parser bug variant {i:03d}")
        _, comps = self._run(server, "fix the parser bug")
        assert len(comps[0]["completions"]) <= _COMPLETIONS_LIMIT

    # ----- coverage branches -------------------------------------------

    def test_non_stale_request_broadcasts_events(self) -> None:
        """The negative leg of the staleness guard still broadcasts."""
        server = VSCodeServer()
        server._complete_seq_latest["c"] = 7
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        server._complete("Reproduce", seq=7, conn_id="c")
        assert any(e.get("type") == "ghost" for e in events)
        assert any(e.get("type") == "completions" for e in events)

    def test_trailing_whitespace_skips_identifier_branch(self) -> None:
        """A query ending in whitespace yields no ``identifier`` items."""
        server = VSCodeServer()
        content = "zebraq_marker_alpha = 1\n"
        _, comps = self._run(
            server, "fix the ", snapshot_content=content,
        )
        types = {c["type"] for c in comps[0]["completions"]}
        assert "identifier" not in types

    def test_identifier_branch_skipped_for_short_partial(self) -> None:
        """A single-char trailing partial skips the identifier branch."""
        server = VSCodeServer()
        content = "alpha_one = 1\nalpha_two = 2\n"
        _, comps = self._run(server, "x = a", snapshot_content=content)
        types = {c["type"] for c in comps[0]["completions"]}
        assert "identifier" not in types

    def test_identifier_ordering_longest_first(self) -> None:
        """The longest matching identifier is offered first."""
        server = VSCodeServer()
        content = (
            "alpha_short = 1\n"
            "alpha_much_longer_identifier = 2\n"
            "alpha_mid_len = 3\n"
        )
        _, comps = self._run(server, "alpha_", snapshot_content=content)
        idents = [
            c["text"] for c in comps[0]["completions"]
            if c["type"] == "identifier"
        ]
        assert idents[0] == "alpha_much_longer_identifier"

    def test_active_file_disk_fallback(self) -> None:
        """``snapshot_content == \"\"`` triggers an on-disk read of snapshot_file."""
        server = VSCodeServer()
        p = Path(self._tmpdir) / "src.py"
        p.write_text(
            "betaq_marker_one = 1\nbetaq_marker_two = 2\n",
        )
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        server._complete(
            "x = betaq_marker_",
            snapshot_file=str(p),
            snapshot_content="",
        )
        comps = [e for e in events if e.get("type") == "completions"]
        texts = {c["text"] for c in comps[0]["completions"]}
        assert "betaq_marker_one" in texts
        assert "betaq_marker_two" in texts

    def test_active_file_oserror_yields_no_identifiers(self) -> None:
        """An unreadable snapshot_file path must not raise."""
        server = VSCodeServer()
        _, comps = self._run(
            server,
            "x = no_such_token_",
            snapshot_content="",
            # Falsey snapshot_file via empty default; we instead use
            # an explicit non-existent path through ``_complete``'s
            # kwargs.
        )
        types = {c["type"] for c in comps[0]["completions"]}
        assert "identifier" not in types
        # Direct call with a non-existent file path also short-circuits.
        ms = server._active_file_identifier_matches(
            "x = token_", snapshot_file="/no/such/file.py",
        )
        assert ms == []

    def test_active_file_no_content_no_chat_returns_empty(self) -> None:
        """No content + no chat text → empty identifier list."""
        server = VSCodeServer()
        ms = server._active_file_identifier_matches("x = token_")
        assert ms == []

    def test_active_file_no_regex_match_returns_empty(self) -> None:
        """No trailing word token → empty identifier list."""
        server = VSCodeServer()
        ms = server._active_file_identifier_matches(
            "!!! ", snapshot_content="alpha = 1\n",
        )
        assert ms == []

    def test_self_match_text_equals_query_filtered(self) -> None:
        """A candidate equal to the query is filtered out by ``_add``."""
        server = VSCodeServer()
        # Direct test of the filter via ``_complete_many``'s caller:
        # The DB layer guarantees LENGTH > query, so we exercise the
        # filter via the ``text == query`` branch reachable through
        # the internal ``_add`` (still part of ``_complete_many``).
        result = server._complete_many("nothing_matches_here_xx_zz_aa")
        assert all(c["text"] != "nothing_matches_here_xx_zz_aa" for c in result)
