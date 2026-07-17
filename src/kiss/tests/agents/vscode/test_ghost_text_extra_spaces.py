# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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
from kiss.server.helpers import clip_autocomplete_suggestion
from kiss.server.server import VSCodeServer


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

    def test_suffix_starting_with_query_not_restripped(self) -> None:
        """A continuation suffix that itself begins with the query text
        must survive intact: suggestions are always suffixes (the call
        sites strip the query), so re-stripping here would corrupt
        completions like ``hellohello world`` typed as ``hello``.
        """
        result = clip_autocomplete_suggestion("hello", "hello world")
        assert result == "hello world"

    def test_identifier_query_with_multi_space_history_collapses_to_one(
        self,
    ) -> None:
        """User types an identifier (no trailing whitespace); history match
        has TWO consecutive spaces after the prefix end.

        Reproduces the user-reported "sometimes I see some spaces before
        the next token in the ghost text" while fast-completing an
        identifier.  Concretely: user types ``"parse"`` (identifier-style,
        no trailing space) and a history task is ``"parse  arguments"``
        (note the two spaces).  ``_prefix_match_task`` returns the full
        task, ``_complete`` slices ``match[len(query):]`` =
        ``"  arguments"``, and before the fix
        ``clip_autocomplete_suggestion`` left those two leading spaces
        intact (its lstrip branch fires only when the query *ends* in
        whitespace).  The first space is the legitimate cursor-to-ghost
        separator; every additional space renders as visible padding
        between the cursor and the next token.

        After the fix: leading whitespace is collapsed to a single
        separator space when the query ends in a non-whitespace
        character.
        """
        result = clip_autocomplete_suggestion("parse", "  arguments")
        assert result == " arguments", (
            f"Expected one separator space (collapsed from two) but got "
            f"{result!r} — extra spaces leak into the ghost text"
        )

    def test_identifier_query_with_many_space_history_collapses_to_one(
        self,
    ) -> None:
        """More than two leading spaces still collapse to exactly one."""
        result = clip_autocomplete_suggestion("foo", "     bar baz")
        assert result == " bar baz"

    def test_identifier_query_with_tab_history_collapses_to_one(self) -> None:
        """Tabs / mixed whitespace also collapse — pre-wrap renders tabs
        as visible gaps in the overlay.
        """
        result = clip_autocomplete_suggestion("foo", "\t bar")
        assert result == " bar"

    def test_identifier_query_repro_via_complete_pipeline(self) -> None:
        """End-to-end via _complete: query 'parse', history 'parse  arguments'.

        The broadcast ghost suggestion must not start with two or more
        whitespace characters — exactly one separator space is allowed.
        """
        server = VSCodeServer()
        # Insert a task with a double space after the identifier — the
        # exact pattern that triggers the user-reported bug.
        th._add_task("parse  arguments now")
        events: list[dict] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        server._complete("parse")
        ghost = [e for e in events if e.get("type") == "ghost"]
        assert len(ghost) == 1
        suggestion = ghost[0]["suggestion"]
        assert suggestion, "Expected a non-empty ghost suggestion"
        # At most one leading space — the legitimate separator.
        leading = len(suggestion) - len(suggestion.lstrip())
        assert leading <= 1, (
            f"Ghost suggestion {suggestion!r} has {leading} leading "
            f"whitespace chars; exactly one cursor-to-ghost separator is "
            f"allowed when the user's query ends in a non-whitespace "
            f"character — anything more renders as visible padding."
        )
        # And the suggestion must continue the history task readably.
        assert suggestion == " arguments now"
