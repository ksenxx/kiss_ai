# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 5: C1 control characters must not enter the steering buffer.

Bug: :meth:`kiss.agents.sorcar.cli_steering._InputBox._append_paste`
documents that "other control characters are dropped so they cannot act
as editing keys", and the typed-key path in :meth:`_InputBox.feed`
guards with ``ch >= " "`` — but both checks only exclude the C0 range
(``< " "``) and DEL.  The C1 control range U+0080–U+009F (Unicode
category ``Cc``) sails through both filters: ``"\\x9b" >= " "`` is
``True``.

U+009B *is* the one-character CSI introducer and U+0085/U+008D are
NEL/RI — once in the buffer they are written verbatim to the terminal
by ``cli_panel.clip_buf`` (which only rewrites ``\\n`` and ``\\t``), so
a paste containing C1 bytes corrupts the box row and can start a stray
escape sequence on the terminal.
"""

from __future__ import annotations

import io
import threading
import unicodedata

import pytest

from kiss.agents.sorcar.cli_panel import clip_buf
from kiss.agents.sorcar.cli_steering import _InputBox


def _make_box() -> _InputBox:
    return _InputBox(threading.RLock(), io.StringIO())


def _has_control(text: str) -> bool:
    return any(
        unicodedata.category(ch) == "Cc" and ch not in ("\n", "\t")
        for ch in text
    )


class TestC1Controls:
    def test_pasted_c1_controls_are_dropped(self) -> None:
        """A bracketed paste containing C1 controls must drop them."""
        box = _make_box()
        data = "\x1b[200~a\u009bX\u0085b\u008dc\x1b[201~".encode()
        box.feed(data, lambda _s: None, lambda: None)
        assert not _has_control(box.buf), (
            "C1 control characters from a paste reached the edit buffer "
            f"(they are emitted raw to the terminal): {box.buf!r}"
        )
        assert box.buf == "aXbc"

    def test_typed_c1_controls_are_dropped(self) -> None:
        """C1 controls decoded from typed input must not be inserted."""
        box = _make_box()
        box.feed("y\u009bz".encode(), lambda _s: None, lambda: None)
        assert not _has_control(box.buf), (
            "a typed/forwarded C1 control character reached the edit "
            f"buffer: {box.buf!r}"
        )
        assert box.buf == "yz"

    def test_submitted_line_contains_no_c1_controls(self) -> None:
        """The queued instruction must not carry raw C1 bytes either."""
        box = _make_box()
        submitted: list[str] = []
        box.feed(
            "\x1b[200~do it\u009b now\x1b[201~\r".encode(),
            submitted.append,
            lambda: None,
        )
        assert submitted, "paste + Enter did not submit"
        assert not _has_control(submitted[0]), (
            f"submitted instruction carries C1 controls: {submitted[0]!r}"
        )

    def test_body_row_never_emits_c1_controls(self) -> None:
        """End to end: the rendered body slice must be control-free."""
        box = _make_box()
        box.feed(
            "\x1b[200~hello\u009bworld\x1b[201~".encode(),
            lambda _s: None,
            lambda: None,
        )
        shown = clip_buf(box.buf, 80)
        assert not _has_control(shown), (
            "clip_buf emitted a raw C1 control char (U+009B is a "
            f"one-byte CSI — it corrupts the box row): {shown!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
