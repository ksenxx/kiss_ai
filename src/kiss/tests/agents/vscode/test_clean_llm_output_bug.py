"""Integration tests reproducing the ``clean_llm_output`` quote/whitespace bug.

Bug repro: ``clean_llm_output`` is documented to "Strip whitespace and
surrounding quotes from LLM output", and is used by
``_run_oneshot_llm`` to clean the raw text returned by the commit-message
and follow-up-suggestion LLM helpers.  Real LLM responses routinely carry
a trailing newline (and frequently wrap the answer in quotes), e.g.::

    "feat: add widget"\n

The original implementation ran ``text.strip('"').strip("'")`` with **no**
whitespace stripping.  Because the trailing ``\n`` sits *outside* the
closing quote, ``strip('"')`` cannot reach the opening/closing quote pair,
so the result keeps a stray quote and newline (``feat: add widget"\n``).
That stray quote then leaks into committed git messages.

These tests assert the documented contract: surrounding whitespace and
quotes are both removed regardless of order.
"""

from __future__ import annotations

from kiss.agents.vscode.helpers import (
    clean_llm_output,
    clip_autocomplete_suggestion,
)


class TestCleanLlmOutput:
    """``clean_llm_output`` must honour its documented strip contract."""

    def test_quoted_with_trailing_newline(self) -> None:
        """Quotes must be removed even when whitespace surrounds them."""
        assert clean_llm_output('"fix the bug"\n') == "fix the bug"

    def test_quoted_with_surrounding_spaces(self) -> None:
        assert clean_llm_output('  "hello"  ') == "hello"

    def test_single_quoted_with_trailing_newline(self) -> None:
        assert clean_llm_output("'msg'\n") == "msg"

    def test_plain_trailing_newline_stripped(self) -> None:
        assert clean_llm_output("plain\n") == "plain"

    def test_plain_surrounding_whitespace_stripped(self) -> None:
        assert clean_llm_output("  spaced  ") == "spaced"

    def test_unquoted_unchanged(self) -> None:
        assert clean_llm_output("feat: add widget") == "feat: add widget"

    def test_inner_quotes_preserved(self) -> None:
        """Only *surrounding* quotes are stripped, not interior ones."""
        assert clean_llm_output('say "hi" now') == 'say "hi" now'

    def test_empty_and_whitespace_only(self) -> None:
        assert clean_llm_output("") == ""
        assert clean_llm_output("   \n  ") == ""


class TestClipAutocompleteUnaffected:
    """The autocomplete clipper must keep its leading-separator behaviour.

    ``clip_autocomplete_suggestion`` deliberately preserves a single
    leading space as the cursor-to-ghost separator; the
    ``clean_llm_output`` whitespace fix must not regress this.
    """

    def test_separator_space_preserved(self) -> None:
        assert clip_autocomplete_suggestion("fix", " the bug") == " the bug"

    def test_trailing_space_query_strips_leading(self) -> None:
        assert clip_autocomplete_suggestion("fix ", " the bug") == "the bug"

    def test_echo_prefix_removed(self) -> None:
        assert clip_autocomplete_suggestion("hello", "hello world") == " world"

    def test_multi_space_collapsed_to_one(self) -> None:
        assert clip_autocomplete_suggestion("parse", "  arguments") == " arguments"
