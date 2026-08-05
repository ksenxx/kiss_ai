# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the shared sorcar CLI input panel.

These verify that the idle REPL prompt and the anchored steering box are
rendered by the *same* panel helpers, so the two input dialogs are a
single, visually consistent panel.
"""

from __future__ import annotations

import io
import threading

from kiss.ui.cli.cli_panel import (
    BOLD,
    CYAN,
    DIM,
    IDLE_TITLE,
    ORANGE,
    PLACEHOLDER,
    PROMPT_MARKER,
    RESET,
    STEER_TITLE,
    menu_row,
    panel_body,
    panel_bottom,
    panel_top,
)
from kiss.ui.cli.cli_steering import _InputBox


class TestPanelBorders:
    def test_top_border_is_rounded_and_carries_title(self) -> None:
        top = panel_top(IDLE_TITLE, 80)
        assert top.startswith("╭")
        assert top.endswith("╮")
        assert "Sorcar · type a task" in top
        assert len(top) == 80

    def test_bottom_border_is_rounded_and_carries_status(self) -> None:
        bottom = panel_bottom(" queued: 2 ", 80)
        assert bottom.startswith("╰")
        assert bottom.endswith("╯")
        assert "queued: 2" in bottom
        assert len(bottom) == 80

    def test_borders_clip_to_width(self) -> None:
        assert len(panel_top(IDLE_TITLE, 12)) == 12
        assert len(panel_bottom("", 12)) == 12


class TestPanelBody:
    def test_buffer_renders_marker_and_text(self) -> None:
        rows, is_placeholder = panel_body("do something", 80)
        assert is_placeholder is False
        assert len(rows) == 3
        assert f"{PROMPT_MARKER}do something" in rows[0]
        assert len(rows[0]) == 76

    def test_empty_buffer_shows_placeholder(self) -> None:
        rows, is_placeholder = panel_body("", 80)
        assert is_placeholder is True
        assert len(rows) == 3
        assert rows[0].startswith(PROMPT_MARKER)
        assert rows[0].startswith(f"{PROMPT_MARKER}{PLACEHOLDER}")
        for blank in rows[1:]:
            assert blank.strip() == ""

    def test_long_buffer_is_tail_clipped(self) -> None:
        rows, _ = panel_body("x" * 200, 40)
        assert len(rows) == 3
        assert len(rows[0]) == 36
        assert rows[0].startswith(PROMPT_MARKER)

    def test_legacy_single_row_minimum_via_override(self) -> None:
        rows, _ = panel_body("hello", 80, min_rows=1)
        assert len(rows) == 1
        empty, _ = panel_body("", 80, min_rows=1)
        assert len(empty) == 1


class TestSharedPanelAcrossDialogs:
    def test_steering_box_uses_shared_title_and_panel(self) -> None:
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        assert box.title == STEER_TITLE
        box._active = True
        box.buf = "tweak it"
        box.redraw()
        text = out.getvalue()
        assert "╭" in text and "╮" in text and "╰" in text and "╯" in text
        assert PROMPT_MARKER in text
        assert "tweak it" in text

    def test_steering_box_shows_chevron_when_empty(self) -> None:
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.buf = ""
        box.redraw()
        text = out.getvalue()
        assert PROMPT_MARKER in text
        assert PLACEHOLDER in text

    def test_idle_and_steer_share_one_border_renderer(self) -> None:
        idle_top = panel_top(IDLE_TITLE, 80)
        steer_top = panel_top(STEER_TITLE, 80)
        assert idle_top[0] == steer_top[0] == "╭"
        assert idle_top[-1] == steer_top[-1] == "╮"
        assert len(idle_top) == len(steer_top) == 80


class TestMenuRowContrastingOrange:
    """The completion menu mirrors Claude Code's high-contrast palette.

    The highlighted candidate is drawn in bold coral-orange (xterm-256
    index 208) while the other rows stay dim, so the selected entry
    pops the way Claude Code's ``/color orange`` prompt bar does.
    """

    def test_selected_row_uses_bold_orange_with_arrow(self) -> None:
        row = menu_row("install pkg", True, 40)
        assert ORANGE in row
        assert BOLD in row
        assert "❯ install pkg" in row
        assert f"{CYAN}│{RESET}" in row
        assert row.endswith(f"{CYAN}│{RESET}")

    def test_unselected_row_is_dim_and_not_orange(self) -> None:
        row = menu_row("uninstall pkg", False, 40)
        assert DIM in row
        assert ORANGE not in row
        assert BOLD not in row
        assert "❯" not in row
        assert f"{CYAN}│{RESET}" in row

    def test_selected_and_unselected_rows_share_panel_width(self) -> None:
        sel = menu_row("foo", True, 40)
        unsel = menu_row("foo", False, 40)
        import re

        ansi = re.compile(r"\x1b\[[0-9;]*m")
        assert len(ansi.sub("", sel)) == 40
        assert len(ansi.sub("", unsel)) == 40

    def test_menu_row_strips_injected_ansi(self) -> None:
        evil = "ok\x1b[31mRED"
        row = menu_row(evil, True, 40)
        assert "\x1b[31m" not in row
        assert "ok[31mRED" in row


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
