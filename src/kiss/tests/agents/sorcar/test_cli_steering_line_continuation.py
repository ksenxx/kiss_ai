# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: backslash line-continuation in the steering box.

Feeds real bytes through :meth:`_InputBox.feed` — the same byte
stream the raw-mode terminal reader delivers — and asserts that
ending a line with an unescaped ``\\`` before Enter inserts a newline
into the buffer instead of submitting.  These tests exercise the
production code path used by mid-task steering AND the idle steering
box (both share :class:`_InputBox`).
"""

from __future__ import annotations

import io
import threading

from kiss.agents.sorcar.cli_steering import _InputBox


def _make_box() -> _InputBox:
    return _InputBox(threading.RLock(), io.StringIO())


class TestSteeringLineContinuation:
    def test_trailing_backslash_plus_enter_inserts_newline_not_submit(
        self,
    ) -> None:
        """``line one \\`` + Enter must NOT submit, but insert ``\\n``."""
        box = _make_box()
        submitted: list[str] = []
        # Type ``line one \`` (space, backslash) — the space before
        # the backslash is what a user would naturally type; the
        # helper allows it either way.
        box.feed(b"line one \\", submitted.append, lambda: None)
        assert box.buf == "line one \\"
        # Enter (bare CR — indistinguishable from Shift+Enter on
        # macOS Terminal.app) must NOT submit because the buffer
        # ends with the continuation marker.
        box.feed(b"\r", submitted.append, lambda: None)
        assert submitted == []
        assert box.buf == "line one \n"

    def test_two_lines_joined_by_backslash_continuation_submit_as_one(
        self,
    ) -> None:
        """Full multi-line input via ``\\`` + Enter, then Enter to submit."""
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"line one \\", submitted.append, lambda: None)
        box.feed(b"\r", submitted.append, lambda: None)
        box.feed(b"line two", submitted.append, lambda: None)
        box.feed(b"\r", submitted.append, lambda: None)
        assert submitted == ["line one \nline two"]
        assert box.buf == ""

    def test_three_lines_joined_by_backslash_continuation(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"one\\\rtwo\\\rthree\r", submitted.append, lambda: None)
        assert submitted == ["one\ntwo\nthree"]

    def test_backslash_with_trailing_spaces_still_continues(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        # Trailing whitespace after the ``\`` (a common typo) is
        # stripped together with the continuation marker.
        box.feed(b"first \\   ", submitted.append, lambda: None)
        box.feed(b"\r", submitted.append, lambda: None)
        box.feed(b"second\r", submitted.append, lambda: None)
        assert submitted == ["first \nsecond"]

    def test_escaped_double_backslash_submits_literal(self) -> None:
        # ``\\\\`` (two backslashes) is a literal — Enter submits.
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"literal \\\\", submitted.append, lambda: None)
        box.feed(b"\r", submitted.append, lambda: None)
        assert submitted == ["literal \\\\"]
        assert box.buf == ""

    def test_odd_backslash_count_continues_retaining_even(self) -> None:
        # Three trailing ``\`` = literal ``\\`` + continuation.
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"foo \\\\\\\r", submitted.append, lambda: None)
        assert submitted == []
        assert box.buf == "foo \\\\\n"
        box.feed(b"bar\r", submitted.append, lambda: None)
        assert submitted == ["foo \\\\\nbar"]

    def test_no_continuation_plain_enter_submits_immediately(self) -> None:
        # Regression: the plain-Enter → submit path must be
        # unchanged for buffers without a trailing ``\``.
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"just one line\r", submitted.append, lambda: None)
        assert submitted == ["just one line"]

    def test_empty_buffer_enter_submits_empty(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"\r", submitted.append, lambda: None)
        # Empty buffer submits an empty string — not a continuation.
        assert submitted == [""]

    def test_backslash_across_multiple_feed_chunks(self) -> None:
        # The bytes ``line one \`` + ``\r`` arriving in two separate
        # ``feed`` calls (as they would from ``os.read`` chunking)
        # must still be recognised as a continuation.
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"line one \\", submitted.append, lambda: None)
        box.feed(b"\r", submitted.append, lambda: None)
        box.feed(b"line two", submitted.append, lambda: None)
        box.feed(b"\r", submitted.append, lambda: None)
        assert submitted == ["line one \nline two"]

    def test_paste_does_not_trigger_continuation_by_accident(self) -> None:
        # A bracketed-paste block whose content ends with ``\`` and
        # a CR inside the paste must NOT be interpreted as a
        # continuation submit — the paste branch stores content
        # verbatim.  Only the CR AFTER the paste terminator can
        # trigger submit / continuation.
        box = _make_box()
        submitted: list[str] = []
        # ESC[200~ … ESC[201~ = bracketed paste envelope.
        box.feed(
            b"\x1b[200~pasted \\\r\x1b[201~",
            submitted.append,
            lambda: None,
        )
        assert submitted == []
        # The paste stored ``pasted \\`` (with the CR normalised to
        # ``\n`` by the paste handler).
        assert "pasted \\" in box.buf
        # Now a bare CR AFTER the paste envelope closed sees a
        # buffer whose LAST char (after tail whitespace strip) is
        # ``\n`` (from the CR-inside-paste normalised to ``\n``),
        # NOT ``\`` — so it submits.
        box.feed(b"\r", submitted.append, lambda: None)
        assert len(submitted) == 1
