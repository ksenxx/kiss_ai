# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: shrinking the terminal below the box height must stay valid.

Bug: when the terminal is resized to fewer rows than the steering box
needs (``rows <= _BOX_H``) while the box is active, ``_draw_locked``
computed ``rows - _BOX_H <= 0`` and emitted *invalid* control sequences
— a zero/negative scroll region (``ESC[1;0r`` / ``ESC[1;-1r``) and
zero/negative cursor rows (``ESC[0;1H``, ``ESC[-1;1H``).  Terminals
ignore or misinterpret these, leaving a corrupted scroll region; the
same math also broke ``stop()``'s row erasing.
"""

from __future__ import annotations

import io
import re
import threading

import pytest

from kiss.ui.cli.cli_steering import _InputBox

# Any CSI sequence whose row parameter is zero or negative is invalid.
_BAD_ROW = re.compile(r"\x1b\[(?:1;)?(?:0|-\d+)[rH;]|\x1b\[-\d+")


def _active_box() -> _InputBox:
    box = _InputBox(threading.RLock(), io.StringIO())
    box._active = True
    return box


def test_redraw_after_shrink_to_three_rows_emits_valid_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    box = _active_box()
    box.redraw()
    out = box._out
    out.seek(0)
    out.truncate(0)

    monkeypatch.setenv("LINES", "3")
    box.redraw()
    written = out.getvalue()
    bad = _BAD_ROW.findall(written)
    assert not bad, (
        f"redraw at 3 rows emitted invalid zero/negative-row sequences "
        f"{bad!r} in {written!r}"
    )


def test_redraw_after_shrink_to_two_rows_emits_valid_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    box = _active_box()
    box.redraw()
    out = box._out
    out.seek(0)
    out.truncate(0)

    monkeypatch.setenv("LINES", "2")
    box.redraw()
    written = out.getvalue()
    bad = _BAD_ROW.findall(written)
    assert not bad, (
        f"redraw at 2 rows emitted invalid zero/negative-row sequences "
        f"{bad!r} in {written!r}"
    )


def test_stop_at_tiny_size_emits_valid_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "2")
    box = _active_box()
    out = box._out
    out.seek(0)
    out.truncate(0)
    box.stop()
    written = out.getvalue()
    bad = _BAD_ROW.findall(written)
    assert not bad, (
        f"stop() at 2 rows emitted invalid zero/negative-row sequences "
        f"{bad!r} in {written!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
