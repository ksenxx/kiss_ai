# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the sorcar idle input textbox grows with line count.

The interactive ``sorcar`` prompt (:func:`kiss.agents.sorcar.cli_repl
._read_line_ptk`, backed by :class:`kiss.agents.sorcar.cli_prompt
.PtkLineReader`) must render a multi-line user prompt as one visible
terminal row per buffer line — exactly like the steering box's
:func:`kiss.agents.sorcar.cli_panel.panel_body` (one row per ``\\n``)
and like the chat-webview textarea (which grows with the content).

These tests drive the real :class:`PromptSession` through a pipe input
the same way :mod:`test_cli_multiline_input` does, but capture the
**rendered** output via :class:`prompt_toolkit.output.plain_text
.PlainTextOutput` and assert that:

* a five-line buffer (entered via Alt+Enter or a bracketed paste)
  produces five distinct visible rows in the rendered output,
* every continuation visual row starts with the cyan ``│ `` painted by
  :func:`kiss.agents.sorcar.cli_prompt._prompt_continuation`,
* a single-line buffer renders no continuation rows (the panel
  shrinks back when there's only one line — chat-webview parity).

If the dynamic height ever regresses (e.g. a future change adds
``Window(height=1, …)``, drops ``multiline=True``, or stops calling
``prompt_continuation``) the rendered output will collapse to a single
row and these tests will fail with an actionable diff.
"""

from __future__ import annotations

import io
import re
import threading
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.plain_text import PlainTextOutput

from kiss.agents.sorcar.cli_prompt import PtkLineReader, _prompt_continuation
from kiss.agents.sorcar.cli_repl import CliCompleter

# Fixed terminal geometry for the capturing output — wide enough that
# the typed lines are never wrap-clipped (so each ``\\n`` becomes a
# distinct visible row, not a wrap), and tall enough that the
# ``reserve_space_for_menu=16`` setting still leaves room for the
# multi-line input.
_ROWS = 40
_COLS = 80
# ANSI SGR escape ``ESC[…m`` — stripped so the per-row assertions match
# against the plain text the user actually sees.
_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")


def _captured_drive(
    tmp_path: Path, keystrokes: str, *, delay: float = 0.5,
) -> tuple[str, str]:
    """Drive ``PtkLineReader`` and return ``(submitted_line, rendered)``.

    Captures the real terminal frames prompt_toolkit paints by wiring
    a :class:`PlainTextOutput` to an in-memory :class:`io.StringIO`,
    then strips SGR colour codes so the row contents can be compared
    against plain literals.
    """
    completer = CliCompleter(str(tmp_path))
    hist = tmp_path / "hist"
    first, _, rest = keystrokes.partition("\x00")
    buf = io.StringIO()
    output = PlainTextOutput(buf)
    # ``PlainTextOutput.get_size`` defaults to a tiny placeholder; pin a
    # realistic terminal geometry so the rendered frame is large enough
    # to accommodate a multi-line buffer plus the menu reservation.
    output.get_size = lambda: Size(rows=_ROWS, columns=_COLS)  # type: ignore[method-assign]
    with create_pipe_input() as pipe:
        def _send_rest() -> None:
            if rest:
                pipe.send_text(rest)

        timer = threading.Timer(delay, _send_rest)
        pipe.send_text(first)
        timer.start()
        try:
            with create_app_session(input=pipe, output=output):
                reader = PtkLineReader(completer, hist)
                line = reader.read("> ")
        finally:
            timer.cancel()
    return line, buf.getvalue()


def _final_input_rows(rendered: str) -> list[str]:
    """Return the visible rows of the *last* rendered frame only.

    prompt_toolkit's :class:`PlainTextOutput` appends each repainted
    frame to its underlying stream rather than overwriting the
    previous one, so the captured buffer is a concatenation of every
    intermediate frame (one per keystroke).  The final frame (the one
    painted right before the buffer is submitted) reflects the user's
    full multi-line input.  Frames are separated by a stretch of
    blank rows — the cursor-positioning + erase-line writes
    prompt_toolkit emits to reuse the screen — so taking the rows
    after the last empty-row stretch yields the final visible block.
    """
    stripped = _SGR_RE.sub("", rendered)
    rows = stripped.splitlines()
    while rows and rows[-1].strip() == "":
        rows.pop()
    # Walk backwards through the trimmed rows until we hit an empty
    # row: everything after that empty row is the final rendered
    # frame the user sees on screen.
    last_block: list[str] = []
    for row in reversed(rows):
        if row.strip() == "":
            break
        last_block.append(row)
    last_block.reverse()
    return last_block


def _rows_starting_with_continuation(rows: list[str]) -> list[str]:
    """Return rendered rows that begin with the ``│ `` continuation prefix.

    :func:`kiss.agents.sorcar.cli_prompt._prompt_continuation` paints
    the cyan ``│`` + space as the left margin of every continuation
    visual row; counting those rows gives the number of visible input
    rows past the first.
    """
    return [r for r in rows if r.startswith("│ ")]


def test_alt_enter_grows_input_textbox_height(tmp_path: Path) -> None:
    """Five Alt+Enter lines render as five visible rows in the input area.

    Reproduces and pins the dynamic-height contract: typing ``one``,
    Alt+Enter, ``two``, Alt+Enter, …, Alt+Enter, ``five``, Enter must
    return ``"one\\ntwo\\nthree\\nfour\\nfive"`` *and* paint five
    distinct visible rows in the rendered terminal frame — one for the
    first ``one`` line (with the ``> `` prompt) and four continuation
    rows opening with the cyan ``│ `` painted by
    :func:`_prompt_continuation`.
    """
    line, rendered = _captured_drive(
        tmp_path,
        "one\x00\x1b\rtwo\x1b\rthree\x1b\rfour\x1b\rfive\r",
    )
    assert line == "one\ntwo\nthree\nfour\nfive"
    rows = _final_input_rows(rendered)
    cont_rows = _rows_starting_with_continuation(rows)
    # Four continuation rows after the first ``> one`` row ⇒ five visible
    # input rows total, matching the five buffer lines.
    assert len(cont_rows) >= 4, (
        f"expected ≥4 continuation rows for 5-line input, got "
        f"{len(cont_rows)} from {rows!r}"
    )
    # Every typed line surfaces as plain text in the final frame.
    last_frame = "\n".join(rows)
    for chunk in ("one", "two", "three", "four", "five"):
        assert chunk in last_frame, (
            f"expected {chunk!r} in rendered frame, got {last_frame!r}"
        )


def test_bracketed_paste_grows_input_textbox_height(tmp_path: Path) -> None:
    """A three-line bracketed paste renders as three visible rows.

    Bracketed paste (``ESC[200~…ESC[201~``) is the channel terminals
    use when the user pastes a multi-line snippet.  Each embedded
    ``\\n`` must surface as a distinct visible row in the input area
    so the user can see (and edit) every pasted line — the same
    dynamic-height contract Alt+Enter satisfies.
    """
    line, rendered = _captured_drive(
        tmp_path,
        "\x1b[200~hello\nworld\nthird\x1b[201~\r",
    )
    assert line == "hello\nworld\nthird"
    rows = _final_input_rows(rendered)
    cont_rows = _rows_starting_with_continuation(rows)
    assert len(cont_rows) >= 2, (
        f"expected ≥2 continuation rows for 3-line paste, got "
        f"{len(cont_rows)} from {rows!r}"
    )
    last_frame = "\n".join(rows)
    for chunk in ("hello", "world", "third"):
        assert chunk in last_frame, (
            f"expected {chunk!r} in pasted rendered frame, got "
            f"{last_frame!r}"
        )


def test_single_line_input_renders_single_input_row(tmp_path: Path) -> None:
    """A single-line input renders no continuation rows.

    The dynamic-height contract is symmetric: a one-line buffer paints
    only the first row (``> hello``) with zero ``│ `` continuation
    rows.  This guards against an accidental "always paint N
    continuation rows" regression (e.g. a fixed-height input window)
    that would push the menu off-screen even for short prompts.
    """
    line, rendered = _captured_drive(tmp_path, "hello\r")
    assert line == "hello"
    rows = _final_input_rows(rendered)
    cont_rows = _rows_starting_with_continuation(rows)
    assert cont_rows == [], (
        f"expected zero continuation rows for single-line input, got "
        f"{cont_rows!r}"
    )


def test_continuation_rows_carry_cyan_border_glyph(tmp_path: Path) -> None:
    """Each continuation visual row begins with the framed ``│`` glyph.

    The framed input panel's left border is the cyan ``│`` painted by
    :func:`_prompt_continuation` on every visual row past the first.
    Without it the multi-line input would visually break out of the
    framed box on every line after the first, contradicting the
    "input textbox" framing the user sees.

    :class:`prompt_toolkit.output.plain_text.PlainTextOutput` filters
    SGR colour escapes from its captured stream, so we drive a real
    three-line input to confirm a non-empty rendered frame *and*
    assert against the very :func:`_prompt_continuation` callable
    prompt_toolkit invokes for each continuation row to confirm the
    cyan SGR attribute is wired through to the framed glyph.
    """
    line, rendered = _captured_drive(
        tmp_path, "a\x00\x1b\rb\x1b\rc\r",
    )
    assert line == "a\nb\nc"
    rows = _final_input_rows(rendered)
    cont_rows = _rows_starting_with_continuation(rows)
    assert len(cont_rows) >= 2, (
        f"expected ≥2 continuation rows for 3-line input, got "
        f"{len(cont_rows)} from {rows!r}"
    )
    # Walk the continuation callable directly — the rendered output
    # strips SGR escapes, but the callable returns the ANSI-styled
    # string prompt_toolkit hands to the renderer.  Every ``│`` glyph
    # must carry the ``ansicyan`` style attribute so the framed left
    # border stays cyan on every continuation row.
    rendered_ansi = _prompt_continuation(_COLS, 1, 0)
    assert isinstance(rendered_ansi, ANSI)
    fragments = list(to_formatted_text(rendered_ansi))
    cyan_glyphs = [
        text for style, text, *_ in fragments
        if "│" in text and "ansicyan" in style
    ]
    assert cyan_glyphs, (
        f"expected at least one cyan │ glyph in the continuation, "
        f"got fragments {fragments!r}"
    )


def test_input_textbox_height_matches_buffer_line_count(tmp_path: Path) -> None:
    """Visible input rows == buffer line count (the chat-webview parity).

    The chat-webview textarea (:mod:`kiss.server`) grows by one
    visible row per ``\\n`` in the model input — no more, no less.
    The CLI idle prompt must match that behaviour: a buffer with
    ``N`` lines renders exactly ``N`` visible input rows in the
    final frame (one chevron row + ``N-1`` continuation rows).
    """
    line, rendered = _captured_drive(
        tmp_path,
        "x\x00\x1b\ry\x1b\rz\x1b\rw\r",
    )
    assert line == "x\ny\nz\nw"
    rows = _final_input_rows(rendered)
    cont_rows = _rows_starting_with_continuation(rows)
    # The first input row is the prompt chevron row (``> x``); the
    # remaining lines (``y``, ``z``, ``w``) each get their own
    # continuation row ⇒ 3 continuation rows for a 4-line buffer.
    assert len(cont_rows) == 3, (
        f"expected exactly 3 continuation rows for 4-line buffer, got "
        f"{len(cont_rows)} from {rows!r}"
    )
