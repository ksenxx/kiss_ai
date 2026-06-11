# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt test: the steering box must re-anchor after a terminal resize.

:meth:`kiss.agents.sorcar.cli_steering._InputBox.start` sets the DECSTBM
scroll region (``ESC[1;{rows - BOX_H}r``) once for the rows measured at
start-up.  When the terminal is resized the box rows are recomputed on
every redraw, but the scroll region was never re-emitted — so after a
resize agent output kept scrolling inside the *old* region, overwriting
the box (or leaving it detached mid-screen).
"""

from __future__ import annotations

import io
import threading

import pytest

from kiss.agents.sorcar.cli_steering import _BOX_H, _InputBox


def _active_box() -> _InputBox:
    """Return an input box marked active, writing to a capture buffer."""
    box = _InputBox(threading.RLock(), io.StringIO())
    box._active = True
    return box


def test_redraw_reemits_scroll_region_after_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Growing the terminal must re-emit the DECSTBM scroll region."""
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    box = _active_box()
    box.redraw()
    out = box._out
    out.seek(0)
    out.truncate(0)

    monkeypatch.setenv("LINES", "30")
    box.redraw()
    written = out.getvalue()
    assert f"\x1b[1;{30 - _BOX_H}r" in written, (
        "redraw after a resize did not re-emit the scroll region; agent "
        "output would keep scrolling inside the stale region"
    )


def test_redraw_without_resize_does_not_thrash_scroll_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redraws at a stable size must not re-emit the scroll region."""
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    box = _active_box()
    box.redraw()
    out = box._out
    out.seek(0)
    out.truncate(0)

    box.redraw()
    assert f"\x1b[1;{24 - _BOX_H}r" not in out.getvalue(), (
        "scroll region re-emitted with no size change (cursor-jump thrash)"
    )


def test_resized_box_draws_at_new_bottom_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After shrinking, the box rows must be positioned for the new size."""
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "30")
    box = _active_box()
    box.redraw()
    out = box._out
    out.seek(0)
    out.truncate(0)

    monkeypatch.setenv("LINES", "24")
    box.redraw()
    written = out.getvalue()
    top_row = 24 - _BOX_H + 1
    assert f"\x1b[{top_row};1H" in written
    assert f"\x1b[1;{24 - _BOX_H}r" in written


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
