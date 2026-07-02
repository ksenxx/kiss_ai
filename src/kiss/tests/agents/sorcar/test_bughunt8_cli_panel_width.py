# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8: panel border lines must occupy exactly *cols* columns.

:func:`kiss.agents.sorcar.cli_panel.panel_top` and
:func:`~kiss.agents.sorcar.cli_panel.panel_bottom` document that the
returned border line is "clipped to *cols*" — i.e. the line occupies
exactly ``cols`` terminal columns, matching the width-aware body rows
built by :func:`~kiss.agents.sorcar.cli_panel._clip_pad` /
:func:`~kiss.agents.sorcar.cli_panel.panel_body`.  The pre-fix code
measured the embedded title / status with ``len()`` (codepoints) and
clipped with a codepoint slice, so a title or status containing
East-Asian wide characters (CJK, emoji) produced a border line wider
than the terminal, wrapping onto the next row and corrupting the
steering box every redraw.
"""

from __future__ import annotations

from kiss.agents.sorcar.cli_panel import (
    display_width,
    panel_bottom,
    panel_top,
)


class TestPanelBorderDisplayWidth:
    def test_panel_top_wide_title_occupies_exactly_cols_columns(self) -> None:
        top = panel_top(" 日本語のタイトル ", 30)
        assert display_width(top) == 30

    def test_panel_bottom_wide_status_occupies_exactly_cols_columns(self) -> None:
        bottom = panel_bottom(" キュー: 2 ", 30)
        assert display_width(bottom) == 30

    def test_panel_top_wide_title_longer_than_panel_is_clipped(self) -> None:
        top = panel_top("漢" * 50, 20)
        assert display_width(top) == 20

    def test_panel_bottom_wide_status_longer_than_panel_is_clipped(self) -> None:
        bottom = panel_bottom("漢" * 50, 20)
        assert display_width(bottom) == 20

    def test_ascii_behaviour_unchanged(self) -> None:
        title = " sorcar "
        top = panel_top(title, 40)
        assert len(top) == 40
        assert display_width(top) == 40
        assert top.startswith("╭─" + title)
        assert top.endswith("╮")
        status = " queued: 2 "
        bottom = panel_bottom(status, 40)
        assert len(bottom) == 40
        assert display_width(bottom) == 40
        assert bottom.startswith("╰")
        assert bottom.endswith(status + "─╯")
