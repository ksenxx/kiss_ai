# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Unit tests for the universal backslash line-continuation helper.

Covers every branch of
:func:`kiss.agents.sorcar.cli_line_continuation.ends_with_line_continuation`:
empty buffer, no backslash, single trailing backslash, backslash
followed by whitespace, escaped ``\\\\`` (even count → submit),
odd counts of trailing backslashes (continuation with the even
literal count retained), and whitespace-only + backslash.
"""

from __future__ import annotations

from kiss.agents.sorcar.cli_line_continuation import (
    ends_with_line_continuation,
)


class TestEndsWithLineContinuation:
    def test_empty_buffer_does_not_continue(self) -> None:
        assert ends_with_line_continuation("") == (False, 0)

    def test_plain_text_without_backslash_submits(self) -> None:
        cont, keep = ends_with_line_continuation("hello world")
        assert cont is False
        assert keep == len("hello world")

    def test_single_trailing_backslash_continues(self) -> None:
        cont, keep = ends_with_line_continuation("hello \\")
        assert cont is True
        # Prefix kept = everything except the outermost backslash.
        assert keep == len("hello ")
        assert "hello \\"[:keep] == "hello "

    def test_backslash_followed_by_trailing_spaces_continues(self) -> None:
        buf = "hello \\   "
        cont, keep = ends_with_line_continuation(buf)
        assert cont is True
        # Both the trailing whitespace AND the backslash are removed.
        assert buf[:keep] == "hello "

    def test_backslash_followed_by_trailing_tabs_continues(self) -> None:
        buf = "hello \\\t\t"
        cont, keep = ends_with_line_continuation(buf)
        assert cont is True
        assert buf[:keep] == "hello "

    def test_escaped_backslash_submits_literal(self) -> None:
        # ``\\\\`` (two backslashes) is an escape for a literal ``\`` —
        # Enter submits, both backslashes remain in the buffer.
        buf = "hello \\\\"
        cont, keep = ends_with_line_continuation(buf)
        assert cont is False
        assert keep == len(buf)
        assert buf[:keep] == "hello \\\\"

    def test_odd_count_of_three_backslashes_continues_retaining_even(
        self,
    ) -> None:
        # ``\\\\\\`` (three backslashes): the outermost consumed as the
        # continuation marker, leaving an EVEN count (2) of literal
        # backslashes in the buffer.
        buf = "hello \\\\\\"
        cont, keep = ends_with_line_continuation(buf)
        assert cont is True
        assert buf[:keep] == "hello \\\\"

    def test_odd_count_five_continues_retaining_four(self) -> None:
        buf = "x\\\\\\\\\\"  # five backslashes
        cont, keep = ends_with_line_continuation(buf)
        assert cont is True
        assert buf[:keep] == "x\\\\\\\\"  # four backslashes

    def test_even_count_four_submits_literal(self) -> None:
        buf = "x\\\\\\\\"  # four backslashes
        cont, keep = ends_with_line_continuation(buf)
        assert cont is False
        assert keep == len(buf)

    def test_whitespace_only_plus_backslash_continues(self) -> None:
        cont, keep = ends_with_line_continuation("   \\")
        assert cont is True
        assert "   \\"[:keep] == "   "

    def test_bare_backslash_continues(self) -> None:
        cont, keep = ends_with_line_continuation("\\")
        assert cont is True
        assert keep == 0

    def test_only_whitespace_submits(self) -> None:
        # Whitespace-only buffer with no backslash: not a continuation
        # (there is nothing to continue), keep the whole buffer.
        buf = "   "
        cont, keep = ends_with_line_continuation(buf)
        assert cont is False
        assert keep == len(buf)

    def test_backslash_followed_by_newline_is_not_a_continuation(
        self,
    ) -> None:
        # The helper never strips embedded newlines — a buffer ending
        # in ``\\\n`` has an embedded newline AFTER the backslash, so
        # the tail-strip loop stops on the ``\n`` and the last char
        # ``\n`` (not ``\``) means there is no continuation marker to
        # detect.
        buf = "hello \\\n"
        cont, keep = ends_with_line_continuation(buf)
        assert cont is False
        assert keep == len(buf)

    def test_multi_line_buffer_with_final_line_continuation(self) -> None:
        # An already-multi-line buffer whose last line ends in ``\\``
        # continues on the next Enter.
        buf = "first line\nsecond line \\"
        cont, keep = ends_with_line_continuation(buf)
        assert cont is True
        # The embedded ``\n`` in the middle of the buffer is preserved;
        # only the trailing continuation marker is stripped.
        assert buf[:keep] == "first line\nsecond line "

    def test_backslash_in_middle_of_buffer_does_not_continue(self) -> None:
        # A backslash somewhere in the middle followed by more text is
        # just a literal character — no continuation.
        buf = "before \\ after"
        cont, keep = ends_with_line_continuation(buf)
        assert cont is False
        assert keep == len(buf)
