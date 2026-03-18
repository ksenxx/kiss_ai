"""Tests for chatbot UI HTML generation and auto-resize behavior."""

import unittest

import pytest

from kiss.agents.sorcar.chatbot_ui import CHATBOT_CSS, CHATBOT_JS

# ---------------------------------------------------------------------------
# kiss/agents/sorcar/chatbot_ui.py — CHATBOT_CSS, CHATBOT_JS
# ---------------------------------------------------------------------------

class TestTextareaAutoResize(unittest.TestCase):
    def test_css_max_height_uses_viewport_units(self) -> None:
        idx = CHATBOT_CSS.index("#task-input{")
        block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
        assert "max-height:50vh" in block
        assert "max-height:200px" not in block

    def test_css_overflow_y_hidden_by_default(self) -> None:
        idx = CHATBOT_CSS.index("#task-input{")
        block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
        assert "overflow-y:hidden" in block

    def test_js_auto_resize_no_200px_cap(self) -> None:
        assert "Math.min(this.scrollHeight,200)" not in CHATBOT_JS

    def test_js_sets_height_to_scrollheight(self) -> None:
        assert "inp.style.height=inp.scrollHeight+'px'" in CHATBOT_JS

    def test_js_toggles_overflow_on_input(self) -> None:
        expected = "inp.style.overflowY=inp.scrollHeight>inp.clientHeight?'auto':'hidden'"
        assert expected in CHATBOT_JS

    def test_js_resets_overflow_on_submit(self) -> None:
        assert "inp.style.overflowY='hidden'" in CHATBOT_JS


def test_model_picker_shrinks_on_zoom():
    """#model-picker must shrink to prevent send button overflow on zoom."""
    idx = CHATBOT_CSS.index("#model-picker{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "min-width:0" in block
    assert "overflow:visible" in block


def test_input_actions_no_shrink():
    """#input-actions needs flex-shrink:0 so send button stays visible."""
    idx = CHATBOT_CSS.index("#input-actions{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "flex-shrink:0" in block


class TestGhostCursorPosition(unittest.TestCase):
    """Ghost completion must work at cursor position, not just end of text."""

    def test_fetch_ghost_skips_when_cursor_in_middle(self) -> None:
        """fetchGhost should not suggest when cursor is not at the end."""
        fn = CHATBOT_JS.split("function fetchGhost")[1].split("\nfunction ")[0]
        assert "if(pos<inp.value.length){clearGhost();return}" in fn

    def test_update_ghost_renders_at_cursor(self) -> None:
        """updateGhost should split text at cursor and insert ghost between."""
        fn = CHATBOT_JS.split("function updateGhost")[1].split("\nfunction ")[0]
        assert "inp.value.substring(0,pos)" in fn
        assert "inp.value.substring(pos)" in fn

    def test_ghost_mask_text_uses_transparent_color_not_hidden(self) -> None:
        """Ghost overlay text should preserve layout so trailing text sits after suggestion."""
        idx = CHATBOT_CSS.index(".gm{")
        block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
        assert "color:transparent" in block
        assert "visibility:hidden" not in block

    def test_ghost_overlay_shares_text_metrics_with_textarea(self) -> None:
        """Ghost overlay and textarea must share text metrics so trailing text aligns."""
        idx = CHATBOT_CSS.index("#task-input,")
        block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
        assert "#ghost-overlay" in block
        assert "white-space:pre-wrap" in block
        assert "word-break:break-word" in block
        assert "box-sizing:border-box" in block
        assert "tab-size:8" in block

    def test_ghost_overlay_syncs_runtime_size_scroll_and_padding(self) -> None:
        """Ghost overlay should sync its measured box with the textarea at runtime."""
        fn = CHATBOT_JS.split("function syncGhostOverlay")[1].split("\nfunction ")[0]
        assert "ghostEl.style.width=inp.clientWidth+'px'" in fn
        assert "ghostEl.style.height=inp.clientHeight+'px'" in fn
        assert "ghostEl.style.paddingTop=inp.style.paddingTop" in fn
        assert "ghostEl.style.paddingLeft=inp.style.paddingLeft" in fn
        assert "ghostEl.scrollTop=inp.scrollTop" in fn

    def test_resize_input_keeps_ghost_overlay_in_sync(self) -> None:
        """Resizing the textarea should also resync the ghost overlay."""
        fn = CHATBOT_JS.split("function resizeInput")[1].split("\nfunction ")[0]
        assert "syncGhostOverlay();" in fn

    def test_ghost_suggestion_uses_plain_span_color(self) -> None:
        """Ghost suggestion styling should only color the suggestion text."""
        idx = CHATBOT_CSS.index(".gs{")
        block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
        assert "color:rgba(255,255,255,0.35)" in block
        assert "white-space" not in block

    def test_accept_ghost_inserts_at_cursor(self) -> None:
        """acceptGhost should insert suggestion at cursor, not append to end."""
        fn = CHATBOT_JS.split("function acceptGhost")[1].split("\nfunction ")[0]
        assert "inp.value.substring(0,pos)" in fn
        assert "before+ghostSuggest+after" in fn
        assert "inp.setSelectionRange(newPos,newPos)" in fn

    def test_clear_ghost_resets_cursor_pos(self) -> None:
        """clearGhost must reset ghostCursorPos."""
        assert "ghostCursorPos=-1" in CHATBOT_JS

    def test_ghost_cursor_pos_variable_declared(self) -> None:
        """ghostCursorPos variable must be declared."""
        assert "var ghostCursorPos=-1" in CHATBOT_JS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
