# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: fast-complete ghost text must suggest INJECTIONS.md
"trick" strings at the beginning of each sentence.

The user's "Inject instruction" tricks come from two files merged by
:func:`kiss.server.tricks.read_tricks`:

* ``~/.kiss/MY_INJECTION.md`` — user-curated tricks, auto-seeded.
* ``src/kiss/INJECTIONS.md`` — bundled tricks, read directly from the
  package (no copy is ever written into ``~/.kiss/``).

A sidebar panel copies them into the textarea on click; this test
verifies they are *also* offered as ghost-text fast-complete
suggestions while typing — but ONLY at a sentence boundary (start of
input or after ``.``/``!``/``?`` + whitespace).

The bundled tricks file is pinned via the ``KISS_INJECTIONS_PATH``
environment variable so tests are independent of whatever the real
``src/kiss/INJECTIONS.md`` happens to contain.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.server.server import VSCodeServer

_LONG_TRICK = (
    "Use claude-opus-4-7 model for all tasks including coding, "
    "bug fixing, and test creation. Use gpt-5.5 model (not codex) "
    "for thorough review of the work done by the other model. "
    "Check if the other model has missed some code."
)

_FAKE_INJECTIONS = (
    "## Trick\n"
    "\n"
    "Reproduce the issue by writing end-to-end test. Then fix the issue.\n"
    "\n"
    "## Trick\n"
    "\n"
    f"{_LONG_TRICK}\n"
    "\n"
    "## Trick\n"
    "\n"
    "Use internet search extensively.\n"
)


class TestGhostTextInjectionSuggestions:
    """Tricks from INJECTIONS.md are offered as ghost text at sentence starts."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self._tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        # Pin the bundled tricks file via the env override so the
        # daemon reads our fake INJECTIONS.md instead of the package
        # copy.  ~/.kiss/MY_INJECTION.md is left empty so the user
        # tricks list is empty and only the bundled fakes appear.
        fake_path = kiss_dir / "fake_INJECTIONS.md"
        fake_path.write_text(_FAKE_INJECTIONS)
        # Empty MY_INJECTION.md so the auto-seed does not contribute
        # extra tricks to the prefix-match dictionary.
        (kiss_dir / "MY_INJECTION.md").write_text("")
        self._saved_kiss_home = os.environ.get("KISS_HOME")
        self._saved_kiss_injections = os.environ.get("KISS_INJECTIONS_PATH")
        os.environ["KISS_HOME"] = str(kiss_dir)
        os.environ["KISS_INJECTIONS_PATH"] = str(fake_path)
        # Isolate the persistence DB so _prefix_match_task can't shadow
        # the trick lookup with a stray history match.
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

    def _ghost_for(self, server: VSCodeServer, query: str) -> str:
        """Run ``_complete(query)`` and return the broadcast ghost suggestion."""
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        server._complete(query)
        ghost = [e for e in events if e.get("type") == "ghost"]
        assert len(ghost) == 1, (
            f"Expected exactly one ghost event, got {len(ghost)}: {ghost!r}"
        )
        suggestion = ghost[0]["suggestion"]
        assert isinstance(suggestion, str)
        return suggestion

    def test_trick_suggested_at_start_of_input(self) -> None:
        """Typing the start of a trick at position 0 produces the rest."""
        server = VSCodeServer()
        suggestion = self._ghost_for(server, "Reproduce")
        assert suggestion, (
            "Expected a ghost suggestion when typing the start of an "
            "INJECTIONS.md trick — got empty string."
        )
        # Concatenating with the typed text should reproduce the trick
        # (modulo the one cursor-to-ghost separator space that
        # clip_autocomplete_suggestion guarantees for identifier-style
        # queries).
        completed = "Reproduce" + suggestion
        assert completed.strip() == (
            "Reproduce the issue by writing end-to-end test. "
            "Then fix the issue."
        ), (
            f"Expected the suggestion to complete the trick verbatim, "
            f"but got {completed!r}"
        )

    def test_trick_suggested_after_period_space(self) -> None:
        """A trick is offered after a sentence-ending period + space."""
        server = VSCodeServer()
        query = "Some preamble text. Reproduce"
        suggestion = self._ghost_for(server, query)
        assert suggestion, (
            "Expected a ghost suggestion when typing the start of a "
            "trick at the beginning of a new sentence (after '. ')."
        )
        completed = query + suggestion
        assert completed.endswith(
            "Reproduce the issue by writing end-to-end test. "
            "Then fix the issue."
        ), f"Suggestion did not complete the trick — got {completed!r}"

    def test_trick_suggested_after_question_mark_space(self) -> None:
        """A trick is offered after a question mark + space."""
        server = VSCodeServer()
        query = "What is it? Use internet"
        suggestion = self._ghost_for(server, query)
        assert suggestion, (
            "Expected a ghost suggestion at the start of a sentence "
            "introduced by '? '."
        )
        completed = query + suggestion
        assert completed.endswith("Use internet search extensively."), (
            f"Suggestion did not complete the trick — got {completed!r}"
        )

    def test_no_trick_suggested_mid_sentence(self) -> None:
        """No trick is offered when the partial sits mid-sentence."""
        server = VSCodeServer()
        # "please Reproduce" — "Reproduce" is not at a sentence start;
        # there is no period, question mark, or exclamation before it.
        suggestion = self._ghost_for(server, "please Reproduce")
        assert suggestion == "", (
            f"Expected NO trick suggestion when the matching partial "
            f"is mid-sentence, but got {suggestion!r}"
        )

    def test_no_trick_suggested_when_no_partial_prefix_match(self) -> None:
        """No suggestion when the current sentence doesn't prefix any trick."""
        server = VSCodeServer()
        suggestion = self._ghost_for(server, "Hello world this")
        assert suggestion == "", (
            f"Expected no suggestion for unrelated text, got {suggestion!r}"
        )

    def test_multi_word_partial_at_sentence_start(self) -> None:
        """A multi-word partial at a sentence start still resolves."""
        server = VSCodeServer()
        suggestion = self._ghost_for(server, "Reproduce the issue by")
        assert suggestion, "Expected suggestion for multi-word partial"
        completed = "Reproduce the issue by" + suggestion
        assert completed.startswith(
            "Reproduce the issue by writing end-to-end test."
        ), f"Got {completed!r}"

    def test_trick_suggested_at_start_after_leading_whitespace(self) -> None:
        """Leading whitespace before the partial doesn't block matching."""
        server = VSCodeServer()
        suggestion = self._ghost_for(server, "  Reproduce")
        assert suggestion, (
            "Leading whitespace in the input shouldn't suppress the "
            "sentence-start trick match."
        )

    def test_trick_suggested_after_newline(self) -> None:
        """A newline counts as a sentence boundary."""
        server = VSCodeServer()
        # Period + newline is still a sentence boundary because the
        # newline is whitespace.
        suggestion = self._ghost_for(server, "Done.\nReproduce")
        assert suggestion, (
            "Expected suggestion after '.\\n' — newline is whitespace."
        )

    def test_no_trick_when_partial_too_short(self) -> None:
        """A single-character partial is below the 2-char ghost threshold."""
        server = VSCodeServer()
        # Only one character typed at the start of a sentence — should
        # not flood the user with a suggestion.
        suggestion = self._ghost_for(server, "R")
        assert suggestion == "", (
            "A single character partial should not trigger a trick "
            "suggestion (matches the existing 2-char ghost minimum)."
        )

    def test_case_sensitive_match(self) -> None:
        """Matching is case-sensitive (mirrors _prefix_match_task)."""
        server = VSCodeServer()
        # Lowercase "reproduce" does not match the capitalised trick.
        suggestion = self._ghost_for(server, "reproduce")
        assert suggestion == "", (
            f"Expected case-sensitive matching to reject lowercase "
            f"prefix, got {suggestion!r}"
        )

    def test_trick_not_offered_when_already_complete(self) -> None:
        """When the user has typed a full trick, no further suggestion."""
        server = VSCodeServer()
        full = (
            "Reproduce the issue by writing end-to-end test. "
            "Then fix the issue."
        )
        suggestion = self._ghost_for(server, full)
        # The completed trick has no further continuation — the
        # partial == full trick means trick[len(partial):] == "".
        assert suggestion == "", (
            f"Expected no suggestion when query already equals trick, "
            f"got {suggestion!r}"
        )
