# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Multi-line input handling in the sorcar steering box.

The bottom-anchored steering box must distinguish a *bare Enter*
(``\\r``) — which submits the current buffer — from every variant of
modifier+Enter that terminals emit for "insert a newline into the same
prompt" (Shift+Enter, Alt+Enter, Ctrl+J, Cmd+Enter, plus every Shift /
Alt / Ctrl / Cmd combination via xterm modifyOtherKeys and CSI-u).

These tests exercise the real key-parsing path in
:meth:`kiss.ui.cli.cli_steering._InputBox.feed` plus the raw-mode
termios setup in :meth:`_InputBox.start` (the kernel must not rewrite
incoming CR to LF, or the two would arrive indistinguishably).
"""

from __future__ import annotations

import io
import os
import pty
import sys
import termios
import threading

import pytest

from kiss.ui.cli.cli_steering import _NEWLINE_AFTER_ESC, _InputBox


def _make_box() -> _InputBox:
    return _InputBox(threading.RLock(), io.StringIO())


# ---------------------------------------------------------------------------
# Bare CR vs bare LF
# ---------------------------------------------------------------------------


class TestBareEnterVsBareLf:
    def test_bare_cr_submits_single_line(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"hello world\r", submitted.append, lambda: None)
        assert submitted == ["hello world"]
        assert box.buf == ""

    def test_bare_lf_inserts_newline_does_not_submit(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"line one\nline two", submitted.append, lambda: None)
        assert submitted == []
        assert box.buf == "line one\nline two"

    def test_bare_lf_then_cr_submits_full_multiline_buffer(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(
            b"first\nsecond\nthird\r", submitted.append, lambda: None
        )
        assert submitted == ["first\nsecond\nthird"]
        assert box.buf == ""


# ---------------------------------------------------------------------------
# ESC-prefixed Alt+Enter variants
# ---------------------------------------------------------------------------


class TestEscPrefixedAltEnter:
    def test_esc_cr_portable_alt_enter_inserts_newline(self) -> None:
        # xterm-style Alt+Enter: ESC followed by CR.
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"a\x1b\rb\r", submitted.append, lambda: None)
        assert submitted == ["a\nb"]
        assert box.buf == ""

    def test_esc_lf_tmux_m_enter_inserts_newline(self) -> None:
        # tmux M-Enter / some terminals' Alt+Enter: ESC + LF.
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"x\x1b\ny\r", submitted.append, lambda: None)
        assert submitted == ["x\ny"]
        assert box.buf == ""

    def test_esc_cr_alone_does_not_submit(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"alpha\x1b\rbeta", submitted.append, lambda: None)
        assert submitted == []
        assert box.buf == "alpha\nbeta"


# ---------------------------------------------------------------------------
# Modifier matrix for the two CSI encodings
# ---------------------------------------------------------------------------


# All modifier codes 2..16 must insert a newline. The encodings are:
#   modifyOtherKeys=2 form: ESC [ 27 ; <m> ; 13 ~
#   CSI-u (kitty) form:     ESC [ 13 ; <m> u
_ALL_MODIFIERS = list(range(2, 17))


class TestModifyOtherKeysCsi:
    @pytest.mark.parametrize("mod", _ALL_MODIFIERS)
    def test_modify_other_keys_form_inserts_newline(self, mod: int) -> None:
        box = _make_box()
        submitted: list[str] = []
        seq = f"\x1b[27;{mod};13~".encode()
        box.feed(b"a" + seq + b"b\r", submitted.append, lambda: None)
        assert submitted == ["a\nb"]
        assert box.buf == ""

    @pytest.mark.parametrize("mod", _ALL_MODIFIERS)
    def test_modify_other_keys_form_alone_does_not_submit(
        self, mod: int
    ) -> None:
        box = _make_box()
        submitted: list[str] = []
        seq = f"\x1b[27;{mod};13~".encode()
        box.feed(seq, submitted.append, lambda: None)
        assert submitted == []
        assert box.buf == "\n"


class TestCsiUForm:
    @pytest.mark.parametrize("mod", _ALL_MODIFIERS)
    def test_csi_u_form_inserts_newline(self, mod: int) -> None:
        box = _make_box()
        submitted: list[str] = []
        seq = f"\x1b[13;{mod}u".encode()
        box.feed(b"x" + seq + b"y\r", submitted.append, lambda: None)
        assert submitted == ["x\ny"]
        assert box.buf == ""

    @pytest.mark.parametrize("mod", _ALL_MODIFIERS)
    def test_csi_u_form_alone_does_not_submit(self, mod: int) -> None:
        box = _make_box()
        submitted: list[str] = []
        seq = f"\x1b[13;{mod}u".encode()
        box.feed(seq, submitted.append, lambda: None)
        assert submitted == []
        assert box.buf == "\n"


# ---------------------------------------------------------------------------
# Multi-line build + submit
# ---------------------------------------------------------------------------


class TestMultiLineBuildSubmit:
    def test_three_line_build_via_alt_enter_then_submit(self) -> None:
        # Mirror what a real user types in iTerm2: Alt+Enter (ESC\r)
        # between each of three lines, then a plain Enter to submit.
        box = _make_box()
        submitted: list[str] = []
        box.feed(
            b"create line_one.txt"
            b"\x1b\r"
            b"create line_two.txt"
            b"\x1b\r"
            b"create line_three.txt"
            b"\r",
            submitted.append,
            lambda: None,
        )
        assert submitted == [
            "create line_one.txt\n"
            "create line_two.txt\n"
            "create line_three.txt"
        ]
        assert box.buf == ""

    def test_mixed_modifier_variants_all_insert_newline(self) -> None:
        # Alt+Enter, Shift+Enter (modifyOtherKeys=2 with mod=2),
        # Ctrl+Enter (CSI-u with mod=5), then submit.
        box = _make_box()
        submitted: list[str] = []
        box.feed(
            b"a"
            b"\x1b\r"  # Alt+Enter
            b"b"
            b"\x1b[27;2;13~"  # Shift+Enter modifyOtherKeys
            b"c"
            b"\x1b[13;5u"  # Ctrl+Enter CSI-u
            b"d"
            b"\r",
            submitted.append,
            lambda: None,
        )
        assert submitted == ["a\nb\nc\nd"]


# ---------------------------------------------------------------------------
# Split-mid-sequence: bytes arrive across read() boundaries
# ---------------------------------------------------------------------------


class TestSplitMidSequence:
    def test_esc_then_cr_split_across_chunks_inserts_newline(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"a\x1b", submitted.append, lambda: None)
        # The pending ESC must be buffered, not interpreted as bare
        # ESC + CR submit on the next chunk.
        assert submitted == []
        box.feed(b"\rb\r", submitted.append, lambda: None)
        assert submitted == ["a\nb"]

    def test_esc_then_lf_split_across_chunks_inserts_newline(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"x\x1b", submitted.append, lambda: None)
        assert submitted == []
        box.feed(b"\ny\r", submitted.append, lambda: None)
        assert submitted == ["x\ny"]

    def test_esc_then_csi_split_mid_sequence_inserts_newline(self) -> None:
        # The whole modifyOtherKeys sequence arrives in fragments.
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"p\x1b[27;", submitted.append, lambda: None)
        assert submitted == []
        box.feed(b"2;13~q\r", submitted.append, lambda: None)
        assert submitted == ["p\nq"]

    def test_esc_then_csi_u_split_mid_sequence_inserts_newline(self) -> None:
        box = _make_box()
        submitted: list[str] = []
        box.feed(b"u\x1b[13;", submitted.append, lambda: None)
        assert submitted == []
        box.feed(b"2uv\r", submitted.append, lambda: None)
        assert submitted == ["u\nv"]


# ---------------------------------------------------------------------------
# Module-scope table sanity
# ---------------------------------------------------------------------------


class TestNewlineAfterEscTable:
    def test_table_contains_every_modifier_for_both_forms(self) -> None:
        for mod in range(2, 17):
            assert f"[27;{mod};13~" in _NEWLINE_AFTER_ESC
            assert f"[13;{mod}u" in _NEWLINE_AFTER_ESC

    def test_table_contains_bare_cr_and_lf_at_end(self) -> None:
        # The bare-byte variants must come AFTER the multi-byte CSI
        # forms so a startswith() match on a CSI prefix isn't
        # short-circuited by a bare CR / LF check.
        assert "\r" in _NEWLINE_AFTER_ESC
        assert "\n" in _NEWLINE_AFTER_ESC
        cr_idx = _NEWLINE_AFTER_ESC.index("\r")
        lf_idx = _NEWLINE_AFTER_ESC.index("\n")
        # Every multi-byte CSI entry precedes both single-byte entries.
        for entry in _NEWLINE_AFTER_ESC:
            if len(entry) > 1:
                assert _NEWLINE_AFTER_ESC.index(entry) < cr_idx
                assert _NEWLINE_AFTER_ESC.index(entry) < lf_idx


# ---------------------------------------------------------------------------
# termios raw-mode setup must clear ICRNL and INLCR
# ---------------------------------------------------------------------------


class TestStartDisablesIcrnlAndInlcr:
    def test_start_clears_icrnl_and_inlcr_iflags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without clearing ICRNL the kernel rewrites Enter (CR) to LF
        on input, making bare Enter indistinguishable from Ctrl+J / a
        tmux M-Enter — and the steering box would have no way to know
        whether to submit or insert a newline."""
        monkeypatch.setenv("COLUMNS", "80")
        monkeypatch.setenv("LINES", "24")
        master, slave = pty.openpty()
        stdin_file = os.fdopen(slave, "r", closefd=False)
        monkeypatch.setattr(sys, "stdin", stdin_file)
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        try:
            # Confirm the PTY starts with ICRNL set (the default) — if
            # not, this test is vacuous and we want to know.
            pre = termios.tcgetattr(slave)
            assert pre[0] & termios.ICRNL, (
                "PTY default did not have ICRNL set; the test cannot "
                "prove start() clears it"
            )
            box.start()
            post = termios.tcgetattr(slave)
            assert not (post[0] & termios.ICRNL), (
                "start() did not clear ICRNL; Enter would arrive as "
                "LF and collide with Ctrl+J / Alt+Enter"
            )
            assert not (post[0] & termios.INLCR), (
                "start() did not clear INLCR; LF would be rewritten "
                "to CR and submit instead of inserting a newline"
            )
        finally:
            if box._active:
                box.stop()
            stdin_file.close()
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
