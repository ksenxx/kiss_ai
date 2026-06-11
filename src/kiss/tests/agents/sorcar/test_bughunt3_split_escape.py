# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt test: escape sequences split across reads must not type text.

Terminals deliver key escape sequences atomically *most* of the time,
but ``os.read`` gives no such guarantee — under load, over SSH, or with
a small read buffer a CSI sequence can arrive split across two chunks.
:meth:`kiss.agents.sorcar.cli_steering._InputBox.feed` used to parse
each chunk independently, so a split arrow key (``ESC`` then ``[A``)
dropped the lone ``ESC`` and typed the literal ``[A`` into the steering
buffer, and a split Shift+Enter (``ESC [13;2`` then ``u``) typed ``u``
instead of inserting a newline.
"""

from __future__ import annotations

import io
import threading

import pytest

from kiss.agents.sorcar.cli_steering import _InputBox


def _box() -> _InputBox:
    """Return a fresh, inactive input box writing to a throwaway buffer."""
    return _InputBox(threading.RLock(), io.StringIO())


def test_arrow_split_after_esc_is_swallowed() -> None:
    """``ESC`` then ``[A`` in the next chunk must not type ``[A``."""
    box = _box()
    box.feed(b"\x1b", lambda _line: None, lambda: None)
    box.feed(b"[A", lambda _line: None, lambda: None)
    assert box.buf == "", (
        f"split CSI arrow typed literal text into the buffer: {box.buf!r}"
    )


def test_arrow_split_after_csi_introducer_is_swallowed() -> None:
    """``ESC [`` then ``A`` in the next chunk must not type ``A``."""
    box = _box()
    box.feed(b"\x1b[", lambda _line: None, lambda: None)
    box.feed(b"A", lambda _line: None, lambda: None)
    assert box.buf == "", (
        f"split CSI arrow typed literal text into the buffer: {box.buf!r}"
    )


def test_split_shift_enter_inserts_newline() -> None:
    """Shift+Enter split mid-sequence must still insert a newline."""
    box = _box()
    box.feed(b"\x1b[13;2", lambda _line: None, lambda: None)
    box.feed(b"u", lambda _line: None, lambda: None)
    assert box.buf == "\n", (
        f"split Shift+Enter mis-parsed; buffer is {box.buf!r}"
    )


def test_split_ctrl_arrow_is_swallowed() -> None:
    """``ESC [1;5`` then ``C`` (Ctrl+Right) must not type ``C``."""
    box = _box()
    box.feed(b"\x1b[1;5", lambda _line: None, lambda: None)
    box.feed(b"C", lambda _line: None, lambda: None)
    assert box.buf == "", (
        f"split Ctrl+arrow typed literal text into the buffer: {box.buf!r}"
    )


def test_split_ss3_arrow_is_swallowed() -> None:
    """``ESC O`` then ``A`` in the next chunk must not type ``A``."""
    box = _box()
    box.feed(b"x\x1bO", lambda _line: None, lambda: None)
    box.feed(b"A", lambda _line: None, lambda: None)
    assert box.buf == "x", (
        f"split SS3 arrow typed literal text into the buffer: {box.buf!r}"
    )


def test_unsplit_sequences_still_swallowed() -> None:
    """Sanity: whole sequences in one chunk keep their behavior."""
    box = _box()
    box.feed(b"hi\x1b[A\x1b[1;5C\x1bOB", lambda _line: None, lambda: None)
    assert box.buf == "hi"
    box.feed(b"\x1b[13;2u", lambda _line: None, lambda: None)
    assert box.buf == "hi\n"


def test_text_after_completed_split_sequence_types() -> None:
    """Bytes after the completing chunk of a split sequence still type."""
    box = _box()
    box.feed(b"\x1b[", lambda _line: None, lambda: None)
    box.feed(b"Aok", lambda _line: None, lambda: None)
    assert box.buf == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
