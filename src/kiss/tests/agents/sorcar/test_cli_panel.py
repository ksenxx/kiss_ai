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
    IDLE_TITLE,
    PLACEHOLDER,
    PROMPT_MARKER,
    STEER_TITLE,
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
        assert body.startswith(PLACEHOLDER)

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
        assert f"{PROMPT_MARKER}tweak it" in text

    def test_idle_and_steer_share_one_border_renderer(self) -> None:
        # Both dialogs build their top border from the same helper, so an
        # equal-width render only differs by the embedded title text.
        idle_top = panel_top(IDLE_TITLE, 80)
        steer_top = panel_top(STEER_TITLE, 80)
        assert idle_top[0] == steer_top[0] == "╭"
        assert idle_top[-1] == steer_top[-1] == "╮"
        assert len(idle_top) == len(steer_top) == 80


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
