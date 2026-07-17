# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression: the sorcar idle input panel must show ≥3 body rows.

The interactive ``sorcar`` REPL pins its rounded input panel at the
bottom of the terminal via :class:`kiss.ui.cli.cli_steering
.AnchoredRepl` (which wraps :class:`kiss.ui.cli.cli_steering
._InputBox`).  Previously the empty panel showed **one** body row
between the top and bottom borders (the placeholder hint sat alone on
that single row), so the user had no visual signal that they could type
multi-line input and the framed box looked cramped (see the user's
screenshot reproducing the bug).

The contract this module pins:

* the empty panel renders **at least three body rows** (chevron +
  placeholder on row 0, two blank body rows below it) — matching the
  3-line minimum the user expects;
* typing more than three lines grows the panel one body row per
  buffer ``\\n`` (dynamic adjustment upward);
* shrinking the buffer back below three lines collapses the panel
  back to the three-row minimum (dynamic adjustment downward);
* the same minimum applies to the prompt_toolkit-based fallback
  :class:`~kiss.ui.cli.cli_prompt.PtkLineReader` so the idle
  prompt looks identical on both code paths.

Tests run entirely in-memory (no real terminal): the panel renders
to a :class:`io.StringIO` and we count the number of rows that begin
with the cyan ``│`` left border between the rounded top (``╭``) and
bottom (``╰``) borders.
"""

from __future__ import annotations

import io
import re
import threading

from prompt_toolkit.layout.dimension import Dimension

from kiss.ui.cli.cli_panel import PLACEHOLDER, panel_body
from kiss.ui.cli.cli_prompt import PtkLineReader
from kiss.ui.cli.cli_repl import CliCompleter
from kiss.ui.cli.cli_steering import (
    _BOX_H,
    _box_body_h,
    _box_h_for,
    _InputBox,
)

# Stripping ANSI SGR (``ESC[…m``) escapes makes the rendered frame
# legible for the row-counting assertions below.
_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")
# CSI cursor-positioning escapes (``ESC[<row>;<col>H``) carve the
# rendered stream into per-row writes that we walk to count visible
# body rows.
_CUP_RE = re.compile(r"\x1b\[(\d+);(\d+)H")

# Number of body rows the idle / steering input panel must always show
# (one chevron+placeholder/text row plus two padding rows).
_MIN_BODY_ROWS = 3


def _render_box(buf: str) -> str:
    """Render an :class:`_InputBox` with *buf* and return the raw stream.

    Mirrors the rendering performed during a live ``sorcar`` session:
    the box is marked ``_active`` (the start/stop terminal-mode dance
    needs a real fd, which the tests do not have) and ``redraw`` is
    invoked synchronously so every cursor-positioned row write lands
    in the :class:`io.StringIO` we return.
    """
    out = io.StringIO()
    box = _InputBox(threading.RLock(), out)
    box._active = True
    box.buf = buf
    box.redraw()
    return out.getvalue()


def _count_body_rows(rendered: str) -> int:
    """Return the number of body rows in *rendered* (between ╭ and ╰).

    Every body row is written by :meth:`_InputBox._draw_locked` as
    ``ESC[<row>;1H ESC[2K │  <body>  │`` — exactly one row carries the
    rounded-top glyph ``╭`` and exactly one row carries the rounded-
    bottom glyph ``╰``.  The body rows in between are the visible
    input-area rows we want to count.
    """
    stripped = _SGR_RE.sub("", rendered)
    # The rendered stream interleaves cursor-positioning escapes with
    # the row contents; splitting on the CUP escape yields one segment
    # per absolute row write.
    segments = _CUP_RE.split(stripped)
    # segments come out as [pre, row1, col1, content1, row2, col2,
    # content2, …]; we only want the content slices.
    contents = segments[3::3]
    in_box = False
    body = 0
    for chunk in contents:
        if "╭" in chunk:
            in_box = True
            continue
        if "╰" in chunk:
            break
        if in_box and "│" in chunk:
            body += 1
    return body


def test_empty_idle_box_renders_at_least_three_body_rows() -> None:
    """An empty input panel must show ≥3 body rows (chevron + 2 blanks).

    Reproduces the user-reported bug: the idle ``sorcar`` REPL panel
    was rendering with **one** body row — the placeholder hint
    ``Add an instruction for the agent while it works…`` sat alone
    between the rounded top and bottom borders.  The fix pads the
    panel to a 3-row body minimum so the framed input dialog matches
    the user-visible contract of "3 lines, dynamically adjustable".
    """
    rendered = _render_box("")
    assert PLACEHOLDER in _SGR_RE.sub("", rendered)
    body_rows = _count_body_rows(rendered)
    assert body_rows >= _MIN_BODY_ROWS, (
        f"expected ≥{_MIN_BODY_ROWS} body rows for the empty input "
        f"panel (the framed box must show 3 lines of space), got "
        f"{body_rows}.  Rendered frame:\n{rendered!r}"
    )


def test_single_line_buffer_still_renders_three_body_rows() -> None:
    """A one-line typed buffer still occupies the 3-row minimum.

    The dynamic-height contract is asymmetric: the panel grows above
    three rows when the buffer has more than three lines, but it never
    shrinks below three rows even for a single typed character.  This
    is what the user sees once the placeholder is replaced by typed
    input — the framed dialog stays the same size.
    """
    rendered = _render_box("hello")
    assert "hello" in _SGR_RE.sub("", rendered)
    body_rows = _count_body_rows(rendered)
    assert body_rows >= _MIN_BODY_ROWS, (
        f"expected ≥{_MIN_BODY_ROWS} body rows for a one-line typed "
        f"buffer, got {body_rows}.  Rendered frame:\n{rendered!r}"
    )


def test_four_line_buffer_grows_panel_to_four_body_rows() -> None:
    """A four-line buffer dynamically grows the panel to 4 body rows.

    Above the three-row minimum the panel is fully dynamic: one body
    row per buffer ``\\n`` so the user can see every typed line at
    once.  This pins the upward-growth half of the dynamic-height
    contract.
    """
    rendered = _render_box("a\nb\nc\nd")
    body_rows = _count_body_rows(rendered)
    assert body_rows == 4, (
        f"expected exactly 4 body rows for a 4-line buffer, got "
        f"{body_rows}.  Rendered frame:\n{rendered!r}"
    )


def test_box_collapses_back_to_three_rows_after_deletion() -> None:
    """Deleting embedded newlines collapses the panel back to 3 rows.

    Together with the four-line test above this pins both directions
    of the dynamic-height contract: the panel grows on Alt+Enter and
    shrinks on Backspace, never below the three-row minimum.
    """
    grown = _count_body_rows(_render_box("a\nb\nc\nd"))
    assert grown == 4
    shrunk = _count_body_rows(_render_box("a"))
    assert shrunk == _MIN_BODY_ROWS, (
        f"expected the panel to collapse back to {_MIN_BODY_ROWS} body "
        f"rows after deletion, got {shrunk}"
    )


def test_box_body_height_helpers_enforce_three_row_minimum() -> None:
    """``_box_body_h`` / ``_box_h_for`` agree with the rendered minimum.

    The steering box's scroll-region math (and the menu-room math in
    :meth:`_InputBox._menu_h`) consult :func:`_box_body_h` /
    :func:`_box_h_for` to decide how many rows to reserve at the
    bottom of the screen.  Those helpers must report the same minimum
    the rendered frame paints, otherwise the reserved area and the
    visible box would disagree (the agent-output scroll region would
    leak into the box's lower rows).
    """
    assert _box_body_h("") == _MIN_BODY_ROWS
    assert _box_body_h("only one line") == _MIN_BODY_ROWS
    # Two ``\n`` → three buffer lines: still exactly the minimum.
    assert _box_body_h("one\ntwo\nthree") == _MIN_BODY_ROWS
    # Three ``\n`` → four buffer lines: dynamic growth past the floor.
    assert _box_body_h("one\ntwo\nthree\nfour") == 4
    # ``_box_h_for`` = 2 borders + body rows.
    assert _box_h_for("") == 2 + _MIN_BODY_ROWS
    assert _BOX_H == 2 + _MIN_BODY_ROWS, (
        f"the steering box's minimum height constant must equal "
        f"two borders + {_MIN_BODY_ROWS} body rows; got _BOX_H="
        f"{_BOX_H}"
    )


def test_panel_body_returns_minimum_three_rows() -> None:
    """``panel_body`` itself pads to the three-row minimum.

    The :class:`_InputBox` draws every row :func:`panel_body` returns,
    so the row-count contract has to live in :func:`panel_body`.
    Passing an explicit non-default ``min_rows`` value still works
    (the caller can override the floor for non-steering callers).
    """
    rows, is_placeholder = panel_body("", 60)
    assert is_placeholder
    assert len(rows) == _MIN_BODY_ROWS
    # At cols=60 the inner width is 60 - 4 = 56 columns; the chevron
    # prefix "› " (2 cols) plus the 57-char placeholder exceeds it, so
    # panel_body clips row 0 to the widest fitting prefix.
    inner_w = 60 - 4
    assert rows[0] == "› " + PLACEHOLDER[: inner_w - 2]
    assert rows[0] == "› Add an instruction for the agent while it works to ste"
    # At a width where chevron + placeholder fits (default 80-col
    # terminals), the full placeholder is shown untruncated.
    wide_rows, wide_is_placeholder = panel_body("", 80)
    assert wide_is_placeholder
    assert PLACEHOLDER in wide_rows[0]
    # Padding rows are blank (no chevron, no placeholder leak).
    for blank in rows[1:]:
        assert blank.strip() == ""
    # Override: a caller can drop the floor to 1 (legacy single-row
    # behaviour) without recompiling the module.
    legacy, _ = panel_body("", 60, min_rows=1)
    assert len(legacy) == 1


def test_ptk_line_reader_enforces_three_row_minimum(tmp_path) -> None:
    """Fallback :class:`PtkLineReader` reserves ≥3 rows for the input area.

    The cli_repl fallback path renders the idle prompt through
    prompt_toolkit (the rounded panel is drawn around it by
    :func:`_read_line_ptk`).  To keep both code paths' visible height
    identical the prompt_toolkit input window must declare a
    ``Dimension(min=3)`` height so the framed panel still shows three
    body rows even when the buffer is a single line.
    """
    completer = CliCompleter(str(tmp_path))
    reader = PtkLineReader(completer, tmp_path / "hist")
    layout = reader.session.layout
    # Walk every Window in the layout: the input Buffer is the one
    # bound to the default buffer; its enclosing Window must declare a
    # minimum height of 3 so prompt_toolkit reserves three terminal
    # rows for the input area even when the buffer holds one line.
    from prompt_toolkit.layout.containers import Window

    input_window = None
    for container in layout.walk():
        if not isinstance(container, Window):
            continue
        control = container.content
        if getattr(control, "buffer", None) is reader.session.default_buffer:
            input_window = container
            break
    assert input_window is not None, "input window not found in layout"
    height = input_window.height
    assert isinstance(height, Dimension)
    assert height.min == _MIN_BODY_ROWS, (
        f"prompt_toolkit input window must declare height.min="
        f"{_MIN_BODY_ROWS}; got {height.min}"
    )
