"""Tests for chatbot UI HTML generation and auto-resize behavior."""

import unittest

import pytest

from kiss.agents.sorcar.chatbot_ui import CHATBOT_CSS, CHATBOT_JS, _build_html


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
        assert "this.style.height=this.scrollHeight+'px'" in CHATBOT_JS

    def test_js_toggles_overflow_on_input(self) -> None:
        expected = "this.style.overflowY=this.scrollHeight>this.clientHeight?'auto':'hidden'"
        assert expected in CHATBOT_JS

    def test_js_resets_overflow_on_submit(self) -> None:
        assert "inp.style.overflowY='hidden'" in CHATBOT_JS


def test_model_picker_shrinks_on_zoom():
    """#model-picker must shrink to prevent send button overflow on zoom."""
    idx = CHATBOT_CSS.index("#model-picker{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "min-width:0" in block
    assert "overflow:hidden" in block


def test_input_actions_no_shrink():
    """#input-actions needs flex-shrink:0 so send button stays visible."""
    idx = CHATBOT_CSS.index("#input-actions{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "flex-shrink:0" in block


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
