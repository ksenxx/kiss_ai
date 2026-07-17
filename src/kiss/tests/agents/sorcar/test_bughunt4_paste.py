# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: the steering box must handle bracketed paste as one block.

Bug: :class:`kiss.ui.cli.cli_steering._InputBox` never enabled
bracketed-paste mode (``ESC[?2004h``) and treated the paste markers
``ESC[200~`` / ``ESC[201~`` as ordinary CSI sequences to swallow.  As a
result pasting a multi-line instruction into the box submitted every
embedded newline as a *separate* queued instruction (each sent to the
model on its own), and ANSI escapes inside pasted text were interpreted
as key presses instead of being stripped.
"""

from __future__ import annotations

import io
import os
import pty
import sys
import threading

import pytest

from kiss.ui.cli.cli_steering import _InputBox


def _make_box() -> _InputBox:
    return _InputBox(threading.RLock(), io.StringIO())


class TestPasteFeed:
    def test_multiline_paste_is_one_block_not_many_submits(self) -> None:
        """A bracketed paste with newlines must not submit partial lines."""
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"\x1b[200~line1\nline2\x1b[201~", submitted.append, lambda: None)
        assert submitted == [], (
            "pasting a multi-line block queued partial instructions: "
            f"{submitted!r}"
        )
        assert box.buf == "line1\nline2"

    def test_paste_then_enter_submits_whole_block(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"\x1b[200~do a\nthen b\x1b[201~\r", submitted.append, lambda: None)
        assert submitted == ["do a\nthen b"]
        assert box.buf == ""

    def test_paste_crlf_normalized_to_newline(self) -> None:
        box = _make_box()
        box.feed(b"\x1b[200~a\r\nb\rc\x1b[201~", lambda _s: None, lambda: None)
        assert box.buf == "a\nb\nc"

    def test_paste_split_across_reads(self) -> None:
        """Paste content and its terminator may be split across reads."""
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"\x1b[200~abc", submitted.append, lambda: None)
        box.feed(b"def\x1b[201", submitted.append, lambda: None)
        box.feed(b"~xyz", submitted.append, lambda: None)
        assert submitted == []
        assert box.buf == "abcdefxyz"

    def test_paste_start_marker_split_across_reads(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"\x1b[20", submitted.append, lambda: None)
        box.feed(b"0~one\ntwo\x1b[201~", submitted.append, lambda: None)
        assert submitted == []
        assert box.buf == "one\ntwo"

    def test_paste_with_embedded_ansi_is_stripped(self) -> None:
        """ANSI sequences inside pasted text must not pollute the buffer."""
        box = _make_box()
        box.feed(b"\x1b[200~a\x1b[31mb\x1b[0mc\x1b[201~", lambda _s: None, lambda: None)
        assert box.buf == "abc", (
            "pasted ANSI escapes leaked into the instruction buffer: "
            f"{box.buf!r}"
        )

    def test_paste_ctrl_chars_do_not_act_as_keys(self) -> None:
        """Control chars in a paste must not clear the line or abort."""
        box = _make_box()
        aborted: list[bool] = []
        box.feed(b"pre", lambda _s: None, lambda: None)
        box.feed(
            b"\x1b[200~a\x15b\x03c\x7fd\x1b[201~",
            lambda _s: None,
            lambda: aborted.append(True),
        )
        assert aborted == [], "Ctrl+C inside a paste aborted the task"
        assert box.buf.startswith("pre"), (
            "Ctrl+U inside a paste cleared the typed text"
        )
        assert "abcd" in box.buf

    def test_typing_still_works_after_paste(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"\x1b[200~block\x1b[201~ tail\r", submitted.append, lambda: None)
        assert submitted == ["block tail"]


class TestPasteModeToggle:
    def test_start_and_stop_toggle_bracketed_paste(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """start() must enable (and stop() disable) bracketed-paste mode."""
        monkeypatch.setenv("COLUMNS", "80")
        monkeypatch.setenv("LINES", "24")
        master, slave = pty.openpty()
        stdin_file = os.fdopen(slave, "r", closefd=False)
        monkeypatch.setattr(sys, "stdin", stdin_file)
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        try:
            box.start()
            assert "\x1b[?2004h" in out.getvalue(), (
                "start() did not enable bracketed-paste mode; terminals "
                "never send paste markers, so multi-line pastes submit "
                "partial instructions"
            )
            box.stop()
            assert "\x1b[?2004l" in out.getvalue(), (
                "stop() did not disable bracketed-paste mode; the mode "
                "leaks into the idle prompt / parent shell"
            )
        finally:
            if box._active:  # pragma: no cover - only on assertion failure
                box.stop()
            stdin_file.close()
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
