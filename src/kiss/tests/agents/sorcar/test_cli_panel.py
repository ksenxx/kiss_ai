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

from kiss.agents.sorcar.cli_panel import (
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
from kiss.agents.sorcar.cli_steering import _InputBox


class TestPanelBorders:
    def test_top_border_is_rounded_and_carries_title(self) -> None:
        top = panel_top(IDLE_TITLE, 80)
        assert top.startswith("╭")
        assert top.endswith("╮")
        assert "sorcar · type a task" in top
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
        body, is_placeholder = panel_body("do something", 80)
        assert is_placeholder is False
        assert f"{PROMPT_MARKER}do something" in body
        assert len(body) == 76  # cols - 4

    def test_empty_buffer_shows_placeholder(self) -> None:
        body, is_placeholder = panel_body("", 80)
        assert is_placeholder is True
        # The chevron is always shown on the left, then the placeholder.
        assert body.startswith(PROMPT_MARKER)
        assert body.startswith(f"{PROMPT_MARKER}{PLACEHOLDER}")

    def test_long_buffer_is_tail_clipped(self) -> None:
        body, _ = panel_body("x" * 200, 40)
        assert len(body) == 36
        assert body.startswith(PROMPT_MARKER)


class TestSharedPanelAcrossDialogs:
    def test_steering_box_uses_shared_title_and_panel(self) -> None:
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        assert box.title == STEER_TITLE
        box._active = True
        box.buf = "tweak it"
        box.redraw()
        text = out.getvalue()
        # The same rounded glyphs the idle prompt prints appear here.
        assert "╭" in text and "╮" in text and "╰" in text and "╯" in text
        # The chevron is drawn cyan (separated by ANSI codes) before the
        # typed text, so assert each part is present.
        assert PROMPT_MARKER in text
        assert "tweak it" in text

    def test_steering_box_shows_chevron_when_empty(self) -> None:
        # The empty (placeholder) steering box still draws the ``› ``
        # chevron on the left, exactly like the idle ``sorcar`` prompt.
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        box._active = True
        box.buf = ""
        box.redraw()
        text = out.getvalue()
        assert PROMPT_MARKER in text
        assert PLACEHOLDER in text

    def test_idle_and_steer_share_one_border_renderer(self) -> None:
        # Both dialogs build their top border from the same helper, so an
        # equal-width render only differs by the embedded title text.
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
        # Border glyphs stay cyan; row terminates with a RESET.
        assert f"{CYAN}│{RESET}" in row
        assert row.endswith(f"{CYAN}│{RESET}")

    def test_unselected_row_is_dim_and_not_orange(self) -> None:
        row = menu_row("uninstall pkg", False, 40)
        assert DIM in row
        # No orange / bold leakage onto unselected rows — that is what
        # makes the selected row visually contrast.
        assert ORANGE not in row
        assert BOLD not in row
        # No chevron on unselected rows (two-space indent in its place).
        assert "❯" not in row
        assert f"{CYAN}│{RESET}" in row

    def test_selected_and_unselected_rows_share_panel_width(self) -> None:
        # Both rows must take the same number of *display* columns as
        # the input panel below them so the menu stacks cleanly.
        sel = menu_row("foo", True, 40)
        unsel = menu_row("foo", False, 40)
        # Strip ANSI escapes to count display width only.
        import re

        ansi = re.compile(r"\x1b\[[0-9;]*m")
        assert len(ansi.sub("", sel)) == 40
        assert len(ansi.sub("", unsel)) == 40

    def test_menu_row_strips_injected_ansi(self) -> None:
        # A candidate carrying a raw ESC must not be able to escape the
        # row's own orange/dim styling and leak into the next line.
        evil = "ok\x1b[31mRED"
        row = menu_row(evil, True, 40)
        # The ESC byte is stripped, so no foreign SGR can fire; only the
        # row's own orange/bold/reset escapes remain.
        assert "\x1b[31m" not in row
        # The printable "[31m" residue still appears but is now harmless
        # text inside the row's orange-bold span.
        assert "ok[31mRED" in row


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
