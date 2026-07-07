# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt test: SS3 key sequences must not type into the steering box.

:meth:`kiss.agents.sorcar.cli_steering._InputBox.feed` swallows CSI
escape sequences (``ESC [ ...``) so arrow keys do not corrupt the edit
buffer.  Terminals in application cursor mode (DECCKM — enabled by many
full-screen programs and left on by some terminals) send arrows as SS3
sequences instead (``ESC O A`` … ``ESC O D``), and F1–F4 are sent as
``ESC O P`` … ``ESC O S`` by xterm-family terminals in *all* modes.

Bug: ``feed`` only recognised the ``[`` introducer, so for an SS3
sequence the ``ESC`` was dropped and the remaining printable bytes were
inserted into the buffer — pressing Up typed ``OA`` and pressing F1
typed ``OP`` into the steering input.
"""

from __future__ import annotations

import io
import threading

import pytest

from kiss.agents.sorcar.cli_steering import _InputBox


def _box() -> _InputBox:
    """Return a fresh, inactive input box writing to a throwaway buffer."""
    return _InputBox(threading.RLock(), io.StringIO())


def test_ss3_arrow_keys_do_not_corrupt_buffer() -> None:
    """Application-mode arrows (ESC O A..D) must be swallowed, not typed."""
    box = _box()
    submitted: list[str] = []
    box.feed(b"hello", submitted.append, lambda: None)
    box.feed(b"\x1bOA\x1bOB\x1bOC\x1bOD", submitted.append, lambda: None)
    assert box.buf == "hello", (
        "SS3 arrow keys typed literal text into the steering buffer: "
        f"{box.buf!r}"
    )
    assert submitted == []


def test_ss3_function_keys_do_not_corrupt_buffer() -> None:
    """F1–F4 (ESC O P..S) must be swallowed, not typed."""
    box = _box()
    box.feed(b"\x1bOP\x1bOQ\x1bOR\x1bOS", lambda _line: None, lambda: None)
    assert box.buf == "", (
        f"SS3 function keys typed literal text into the buffer: {box.buf!r}"
    )


def test_ss3_swallow_keeps_following_text() -> None:
    """Only the 3-byte SS3 sequence is consumed; later keys still type.

    SS3 Up on a non-empty single-line buffer now moves the caret to the
    start of the text (webview-textbox parity), so the following ``cd``
    is inserted at the front — but never the literal ``OA`` bytes.
    """
    box = _box()
    box.feed(b"ab\x1bOAcd", lambda _line: None, lambda: None)
    assert box.buf == "cdab"


def test_csi_arrow_keys_still_swallowed() -> None:
    """Regression guard: normal-mode CSI arrows never type literal text."""
    box = _box()
    box.feed(b"hi\x1b[A\x1b[D", lambda _line: None, lambda: None)
    assert box.buf == "hi"


def test_truncated_ss3_at_chunk_end_is_dropped() -> None:
    """A chunk ending mid-SS3 (``ESC O``) must not insert the ``O``."""
    box = _box()
    box.feed(b"x\x1bO", lambda _line: None, lambda: None)
    assert box.buf == "x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
