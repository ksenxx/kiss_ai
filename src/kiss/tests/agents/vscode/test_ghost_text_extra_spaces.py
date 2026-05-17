"""Integration test: fast-complete ghost text must not insert extra
spaces between the user's cursor and the suggestion.

Bug repro: when the user's query ends in whitespace (e.g. ``"fix "``) and
the matching history task has *more* than one whitespace at that
position (e.g. ``"fix  the bug"`` with two spaces), the raw suffix
returned by ``_prefix_match_task`` retains the extra leading whitespace.
Rendered in the ghost overlay, this produces two (or more) visible
spaces between the cursor and the start of the ghost text — the bug the
user is reporting.

The fix lives in ``clip_autocomplete_suggestion``: when the query ends
in whitespace, the user has already supplied the cursor-to-ghost
separator, so any leading whitespace on the suggestion must be stripped.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from kiss.agents.sorcar import persistence as th
from kiss.agents.vscode.helpers import clip_autocomplete_suggestion
from kiss.agents.vscode.server import VSCodeServer


class TestGhostTextNoExtraSpaces:
    """Ghost text must show exactly one cursor-to-suggestion separator."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self._tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        th._KISS_DIR = kiss_dir

    def teardown_method(self) -> None:
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved

    def test_repro_extra_space_when_query_ends_in_space(self) -> None:
        """User types 'fix ' (one trailing space); history has 'fix  the bug'
        (two consecutive spaces). The raw suffix from prefix-match would be
        ' the bug' (leading space) — visually rendered after the cursor as
        'fix  the bug' (two spaces). clip_autocomplete_suggestion must strip
        the leading whitespace so the ghost reads 'the bug' cleanly.
        """
        result = clip_autocomplete_suggestion("fix ", " the bug")
        assert result == "the bug", (
            f"Expected suggestion 'the bug' (no leading space) when query "
            f"ends in whitespace, but got {result!r}"
        )

    def test_repro_multiple_leading_spaces_stripped(self) -> None:
        """All leading whitespace is collapsed when query ends in whitespace."""
        result = clip_autocomplete_suggestion("fix ", "   the bug")
        assert result == "the bug"

    def test_repro_via_complete_pipeline(self) -> None:
        """End-to-end via _complete: query 'fix ', history 'fix  the bug now'.

        The broadcast ghost suggestion must not start with whitespace.
        """
        server = VSCodeServer()
        # Insert a task with a double space — the exact pattern that
        # triggered the original bug report.
        th._add_task("fix  the bug now")
        events: list[dict] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        server._complete("fix ")
        ghost = [e for e in events if e.get("type") == "ghost"]
        assert len(ghost) == 1
        suggestion = ghost[0]["suggestion"]
        assert suggestion, "Expected a non-empty ghost suggestion"
        assert not suggestion.startswith(" "), (
            f"Ghost suggestion {suggestion!r} starts with whitespace when "
            f"the query already ends with a space — this is the extra-spaces bug."
        )
        # And the suggestion concatenated to the query must recover the
        # full history task verbatim.
        assert "fix " + suggestion == "fix  the bug now" or (
            suggestion == "the bug now"
        ), f"Suggestion {suggestion!r} should continue the history task cleanly"

    def test_no_strip_when_query_does_not_end_in_space(self) -> None:
        """When the query has no trailing whitespace, a leading space in the
        suggestion is the legitimate cursor-to-ghost separator and must be
        preserved. This guards against over-correction.
        """
        result = clip_autocomplete_suggestion("fix", " the bug")
        assert result == " the bug"

    def test_empty_query_strips_leading_whitespace(self) -> None:
        """An empty query (no cursor context) should not produce a ghost
        prefixed with whitespace either — there's nothing for the user's
        cursor to anchor against, so the ghost must start at the very
        beginning of the input.
        """
        result = clip_autocomplete_suggestion("", "  hello")
        assert result == "hello"

    def test_existing_echo_prefix_behaviour_preserved(self) -> None:
        """The pre-existing behaviour — stripping a fully echoed query
        prefix — must still work after the leading-whitespace fix.
        """
        result = clip_autocomplete_suggestion("hello", "hello world")
        assert result == " world"
