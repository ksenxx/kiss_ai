# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for multi-line input + word-wrap in the sorcar CLI.

The interactive sorcar prompt (:mod:`kiss.agents.sorcar.cli_prompt`)
must let the user enter multi-line tasks directly in the input box and
must word-wrap the visible text instead of letting it scroll
horizontally off the panel.

These tests drive a real :class:`PromptSession` through a pipe input
(the same pattern as :mod:`test_at_mention_picker`) and assert that:

* Alt+Enter (Esc+Enter), Ctrl+J, and the Shift+Enter CSI-u /
  modifyOtherKeys escape sequences all insert a real ``\\n`` into the
  buffer instead of submitting the line.
* A bare ``Enter`` still submits.
* The :class:`PromptSession` is configured with ``multiline=True`` and
  ``wrap_lines=True`` so long lines wrap inside the framed panel.
* A ``prompt_continuation`` is wired so wrapped / continuation visual
  rows still carry the cyan ``│`` left border of the input panel.
"""

from __future__ import annotations

import threading
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from kiss.agents.sorcar.cli_prompt import PtkLineReader, _prompt_continuation
from kiss.agents.sorcar.cli_repl import CliCompleter


def _drive(
    tmp_path: Path, keystrokes: str, *, delay: float = 0.5,
) -> str:
    """Run ``PtkLineReader.read`` and feed *keystrokes* through a pipe.

    The early text is sent before the reader starts so the buffer
    already contains it; the rest is delivered by a background timer
    once prompt_toolkit's event loop is running, mirroring the helper
    used in :mod:`test_at_mention_picker`.
    """
    completer = CliCompleter(str(tmp_path))
    hist = tmp_path / "hist"
    first, _, rest = keystrokes.partition("\x00")
    with create_pipe_input() as pipe:
        def _send_rest() -> None:
            if rest:
                pipe.send_text(rest)

        timer = threading.Timer(delay, _send_rest)
        pipe.send_text(first)
        timer.start()
        try:
            with create_app_session(input=pipe, output=DummyOutput()):
                reader = PtkLineReader(completer, hist)
                return reader.read("> ")
        finally:
            timer.cancel()


def test_alt_enter_inserts_newline_then_enter_submits(tmp_path: Path) -> None:
    """Alt+Enter (Esc+Enter) inserts ``\\n``; Enter submits a multi-line line.

    Reproduces the missing multi-line input feature: typing
    ``hello`` + Alt+Enter + ``world`` + Enter must return
    ``"hello\\nworld"``.  Before the fix the very first ``\\r`` after
    Esc was interpreted as a plain Enter (the session was not
    multi-line) and the call returned just ``"hello"``.
    """
    # ``\x1b\r`` is the Esc+Enter (a.k.a. Alt+Enter / Meta+Enter)
    # sequence delivered by every terminal that does not also route
    # Shift+Enter through a CSI-u / modifyOtherKeys binding.
    line = _drive(tmp_path, "hello\x00\x1b\rworld\r")
    assert "\n" in line, f"expected a newline in {line!r}"
    assert line == "hello\nworld"


def test_ctrl_j_inserts_newline(tmp_path: Path) -> None:
    """Ctrl+J (Linefeed, ``\\n``) inserts a real newline.

    Some terminals (notably macOS Terminal.app with the default
    keyboard layout) deliver Ctrl+J as a way to send a literal LF — it
    must therefore insert a newline rather than submit.
    """
    line = _drive(tmp_path, "alpha\x00\nbeta\r")
    assert line == "alpha\nbeta"


def test_shift_enter_csi_u_inserts_newline(tmp_path: Path) -> None:
    """Kitty / CSI-u Shift+Enter (``ESC[13;2u``) inserts a newline."""
    line = _drive(tmp_path, "first\x00\x1b[13;2usecond\r")
    assert line == "first\nsecond"


def test_shift_enter_modify_other_keys_is_treated_as_plain_enter(
    tmp_path: Path,
) -> None:
    """xterm modifyOtherKeys Shift+Enter (``ESC[27;2;13~``) submits.

    Documents the prompt_toolkit limitation: that sequence is
    pre-mapped to :data:`Keys.ControlM` inside
    :data:`prompt_toolkit.input.ansi_escape_sequences.ANSI_SEQUENCES`
    *before* any key bindings run, so on xterm modifyOtherKeys
    terminals Shift+Enter is indistinguishable from plain Enter and
    therefore submits.  Users on those terminals must use Alt+Enter
    or Ctrl+J to insert a newline.
    """
    line = _drive(tmp_path, "top\x00\x1b[27;2;13~")
    assert line == "top"


def test_plain_enter_still_submits_single_line_input(tmp_path: Path) -> None:
    """A plain ``Enter`` on a single line still submits."""
    line = _drive(tmp_path, "just one line\r")
    assert line == "just one line"


def test_three_line_input_via_alt_enter(tmp_path: Path) -> None:
    """Three lines joined by Alt+Enter are returned as one ``\\n``-joined string."""
    line = _drive(
        tmp_path,
        "one\x00\x1b\rtwo\x1b\rthree\r",
    )
    assert line == "one\ntwo\nthree"


def test_prompt_session_is_multiline(tmp_path: Path) -> None:
    """The PromptSession is constructed with ``multiline=True``.

    Without ``multiline=True`` prompt_toolkit treats Enter as an
    immediate submit and never feeds an inserted ``\\n`` back to the
    buffer, so this flag is the structural pre-condition for the
    multi-line user-flow validated by the pipe-driven tests above.
    """
    hist = tmp_path / "hist"
    reader = PtkLineReader(CliCompleter(str(tmp_path)), hist)
    assert bool(reader.session.multiline) is True


def test_prompt_session_wraps_long_lines(tmp_path: Path) -> None:
    """The PromptSession is configured with ``wrap_lines=True``.

    Long input must wrap inside the framed panel rather than scroll
    horizontally off-screen — ``wrap_lines=True`` is what makes
    prompt_toolkit render the overflow on the next visual row.
    """
    hist = tmp_path / "hist"
    reader = PtkLineReader(CliCompleter(str(tmp_path)), hist)
    assert bool(reader.session.wrap_lines) is True


def test_prompt_continuation_keeps_panel_left_border(tmp_path: Path) -> None:
    """Wrapped / continuation visual rows still carry the framed ``│``.

    When the input wraps onto a new visual row (or the user enters a
    multi-line task), prompt_toolkit calls the session's
    ``prompt_continuation`` to render the left margin.  The framed
    input panel paints a cyan ``│`` on the first row, so the
    continuation must paint the same glyph — otherwise the box appears
    broken on every line after the first.
    """
    hist = tmp_path / "hist"
    reader = PtkLineReader(CliCompleter(str(tmp_path)), hist)
    # The PromptSession stores the continuation callable directly; it
    # is the very same module-level function the production code wires
    # in, so calling it here exercises the same code path prompt_toolkit
    # uses for every wrapped / multi-line visual row.
    assert reader.session.prompt_continuation is _prompt_continuation
    rendered = _prompt_continuation(80, 1, 0)
    assert isinstance(rendered, ANSI)
    text = "".join(t[1] for t in to_formatted_text(rendered))
    assert "│" in text, (
        f"expected the framed │ glyph in the continuation, got {text!r}"
    )
