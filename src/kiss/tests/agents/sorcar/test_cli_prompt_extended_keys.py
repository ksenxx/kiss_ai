# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: initial task prompt opts into BOTH extended-keyboard
protocols so Shift+Enter reliably inserts a newline on every terminal.

Historical gap closed by this test file:
    Commit ``0e35b6e3`` fixed the Shift+Enter multi-line bug in the
    mid-task steering box (:class:`~kiss.ui.cli.cli_steering._InputBox`)
    by pushing BOTH ``ESC[>4;2m`` (xterm modifyOtherKeys=2) AND
    ``ESC[>1u`` (Kitty keyboard protocol push flag 1) in
    :meth:`_InputBox.start`.  The initial task prompt reader
    (:class:`~kiss.ui.cli.cli_prompt.PtkLineReader`) still
    wrote only ``ESC[>4;2m``, so on Kitty / foot / ghostty — terminals
    that ignore modifyOtherKeys=2 in favour of the Kitty keyboard
    protocol — the very first task prompt would still submit
    Shift+Enter instead of inserting a newline.  The fix writes both
    enable sequences (and both matching disable sequences on exit),
    mirroring :meth:`_InputBox.start` exactly.

These tests capture the raw bytes emitted by
:meth:`PtkLineReader.read` on the prompt_toolkit output and assert
that ALL FOUR sequences are written in the right phases.  A parser-
side test additionally proves that once the terminal is emitting the
Kitty encoding, ``ESC[13;2u`` (Shift+Enter under Kitty) really is
turned into a newline (not a submit) by the tuple key-bindings.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from kiss.ui.cli.cli_prompt import PtkLineReader
from kiss.ui.cli.cli_repl import CliCompleter


class _CapturingOutput(DummyOutput):
    """A :class:`DummyOutput` that records every ``write_raw`` payload.

    prompt_toolkit's REPL writes its own control sequences via
    :meth:`write` (buffered / cooked); the enable / disable CSI
    sequences under test go through :meth:`write_raw`, so recording
    only that channel is a tight test that will not flap on
    unrelated prompt_toolkit rendering churn.
    """

    def __init__(self) -> None:
        super().__init__()
        self.raw_writes: list[str] = []

    def write_raw(self, data: str) -> None:
        self.raw_writes.append(data)


def _drive_read(
    tmp_path: Path,
    keystrokes: str,
    *,
    delay: float = 0.3,
) -> tuple[str, _CapturingOutput]:
    """Drive one :meth:`PtkLineReader.read` and return (line, output).

    Follows the same pipe-input pattern used by
    :mod:`test_cli_multiline_input`: initial keystrokes are buffered
    before the reader starts, any remainder is delivered via a
    background timer once prompt_toolkit's event loop is running.
    """
    completer = CliCompleter(str(tmp_path))
    hist = tmp_path / "hist"
    first, _, rest = keystrokes.partition("\x00")
    captured = _CapturingOutput()
    with create_pipe_input() as pipe:
        def _send_rest() -> None:
            if rest:
                pipe.send_text(rest)

        timer = threading.Timer(delay, _send_rest)
        pipe.send_text(first)
        timer.start()
        try:
            with create_app_session(input=pipe, output=captured):
                reader = PtkLineReader(completer, hist)
                line = reader.read("> ")
        finally:
            timer.cancel()
    return line, captured


class TestReadEmitsEnableAndDisableSequences:
    """Both extended-keyboard protocols are pushed on entry and popped
    on exit — matching the steering box's ``start`` / ``stop`` pair.
    """

    def test_entry_writes_modify_other_keys_level_2(
        self, tmp_path: Path,
    ) -> None:
        """``ESC[>4;2m`` must appear in the raw output at least once.

        Without this xterm modifyOtherKeys enable, iTerm2 / macOS
        Terminal.app / the VS Code integrated terminal all deliver
        Shift+Enter as a bare ``\\r`` and the multi-line UX breaks.
        """
        _, captured = _drive_read(tmp_path, "\r")
        joined = "".join(captured.raw_writes)
        assert "\x1b[>4;2m" in joined, (
            "read() must opt into xterm modifyOtherKeys=2 on entry; "
            f"raw writes were {captured.raw_writes!r}"
        )

    def test_entry_writes_kitty_keyboard_push_flag_1(
        self, tmp_path: Path,
    ) -> None:
        """``ESC[>1u`` must appear in the raw output at least once.

        Kitty / foot / ghostty ignore modifyOtherKeys=2 entirely; on
        those terminals Shift+Enter only becomes a distinct
        ``ESC[13;2u`` sequence after the application pushes Kitty
        keyboard flag 1.  This is the exact gap the fix closes.
        """
        _, captured = _drive_read(tmp_path, "\r")
        joined = "".join(captured.raw_writes)
        assert "\x1b[>1u" in joined, (
            "read() must push the Kitty keyboard protocol flag 1 on "
            "entry so Shift+Enter emits ESC[13;<m>u under kitty / "
            "WezTerm / ghostty / foot; "
            f"raw writes were {captured.raw_writes!r}"
        )

    def test_exit_writes_modify_other_keys_level_0(
        self, tmp_path: Path,
    ) -> None:
        """``ESC[>4;0m`` must be written on exit (restore level 0)."""
        _, captured = _drive_read(tmp_path, "\r")
        joined = "".join(captured.raw_writes)
        assert "\x1b[>4;0m" in joined, (
            "read() must restore modifyOtherKeys to level 0 on exit "
            "so downstream / child processes do not inherit our mode; "
            f"raw writes were {captured.raw_writes!r}"
        )

    def test_exit_writes_kitty_keyboard_pop(
        self, tmp_path: Path,
    ) -> None:
        """``ESC[<u`` must be written on exit (pop the Kitty stack)."""
        _, captured = _drive_read(tmp_path, "\r")
        joined = "".join(captured.raw_writes)
        assert "\x1b[<u" in joined, (
            "read() must pop the Kitty keyboard flag entry we pushed "
            "so we do not leak a stack entry into the shell; "
            f"raw writes were {captured.raw_writes!r}"
        )

    def test_enter_pair_precedes_exit_pair(self, tmp_path: Path) -> None:
        """Both enable sequences fire before either disable sequence.

        Guards against a refactor that accidentally moves the pop /
        restore call out of the ``finally`` block or writes it too
        early (e.g. before the prompt runs).
        """
        _, captured = _drive_read(tmp_path, "\r")
        joined = "".join(captured.raw_writes)
        modify_on = joined.find("\x1b[>4;2m")
        kitty_on = joined.find("\x1b[>1u")
        modify_off = joined.find("\x1b[>4;0m")
        kitty_off = joined.find("\x1b[<u")
        assert modify_on >= 0 < kitty_on
        assert modify_off > modify_on, (
            "modifyOtherKeys disable must come after enable, "
            f"positions on={modify_on} off={modify_off}"
        )
        assert kitty_off > kitty_on, (
            "Kitty pop must come after push, "
            f"positions on={kitty_on} off={kitty_off}"
        )

    def test_exit_pair_still_written_when_prompt_raises(
        self, tmp_path: Path,
    ) -> None:
        """Disable sequences are inside the ``finally`` block.

        If ``session.prompt`` raises (Ctrl+C / Ctrl+D / any other
        exception) the terminal must still be restored to its default
        keyboard modes.  Force :meth:`session.prompt` to raise and
        assert both disable sequences were nevertheless written.
        """
        completer = CliCompleter(str(tmp_path))
        hist = tmp_path / "hist"
        captured = _CapturingOutput()
        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=captured):
                reader = PtkLineReader(completer, hist)

                def _boom(*_args: Any, **_kwargs: Any) -> str:
                    raise KeyboardInterrupt

                # Swap in a raising prompt after the session was
                # constructed so ``read()`` still writes its enable
                # sequences on entry but then unwinds through the
                # ``finally`` block.
                reader.session.prompt = _boom  # type: ignore[method-assign]
                try:
                    reader.read("> ")
                except KeyboardInterrupt:
                    pass
        joined = "".join(captured.raw_writes)
        assert "\x1b[>4;2m" in joined, (
            "entry must still write modifyOtherKeys enable even when "
            "prompt() will raise; got " f"{captured.raw_writes!r}"
        )
        assert "\x1b[>1u" in joined, (
            "entry must still push the Kitty keyboard flag even when "
            "prompt() will raise; got " f"{captured.raw_writes!r}"
        )
        assert "\x1b[>4;0m" in joined, (
            "finally must still disable modifyOtherKeys on exception; "
            f"got {captured.raw_writes!r}"
        )
        assert "\x1b[<u" in joined, (
            "finally must still pop the Kitty keyboard flag on "
            "exception; got " f"{captured.raw_writes!r}"
        )


class TestKittyShiftEnterInsertsNewline:
    """Once the terminal is emitting the Kitty encoding
    (``ESC[13;2u`` for Shift+Enter), the parser inserts a newline
    instead of submitting the line.

    This exercises the actual key-binding dispatch — the module-level
    ``for _mod in (...): _bind_newline_sequence("escape", "[", "1",
    "3", ";", _mod, "u")`` — through a real :class:`PromptSession`.
    """

    def test_kitty_shift_enter_inserts_newline_not_submit(
        self, tmp_path: Path,
    ) -> None:
        """Shift+Enter under the Kitty keyboard protocol
        (``ESC[13;2u``) must insert ``\\n``; the trailing bare ``\\r``
        submits the whole two-line buffer.
        """
        line, _ = _drive_read(tmp_path, "alpha\x00\x1b[13;2uomega\r")
        assert line == "alpha\nomega", (
            f"Kitty Shift+Enter must insert a newline; got {line!r}"
        )

    def test_kitty_shift_enter_three_lines(self, tmp_path: Path) -> None:
        """Three lines glued by two ``ESC[13;2u`` presses submit as
        one string with two embedded newlines.
        """
        line, _ = _drive_read(
            tmp_path, "one\x00\x1b[13;2utwo\x1b[13;2uthree\r",
        )
        assert line == "one\ntwo\nthree", (
            f"three-line Kitty Shift+Enter buffer must submit as one "
            f"string; got {line!r}"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
