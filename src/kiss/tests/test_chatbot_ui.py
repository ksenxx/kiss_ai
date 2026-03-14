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


def test_auth_button_and_panel_exist():
    html = _build_html("Test", "", "/tmp")
    assert 'id="auth-btn"' in html
    assert 'id="auth-panel"' in html
    assert 'id="auth-refresh-btn"' in html
    assert 'id="auth-login-btn"' in html
    assert 'id="auth-logout-btn"' in html


def test_auth_panel_js_uses_auth_endpoint():
    assert "function toggleAuthPanel()" in CHATBOT_JS
    assert "fetch('/auth?model='" in CHATBOT_JS
    assert "fetch('/auth',{" in CHATBOT_JS
    assert "loadAuthStatus('login')" in CHATBOT_JS
    assert "loadAuthStatus('logout')" in CHATBOT_JS
    assert "window.open(d.login_url" in CHATBOT_JS


def test_model_vendor_groups_chatgpt_under_openai():
    assert "^(chatgpt|gpt|o[134]|codex|computer-use)" in CHATBOT_JS


def test_model_provider_selector_present_in_html():
    html = _build_html("Test", "", "/tmp")
    assert 'id="model-provider"' in html
    assert '<option value="codex">Codex</option>' in html
    assert '<option value="openai">OpenAI API</option>' in html


def test_model_provider_js_has_codex_catalog_and_filtering():
    assert "function modelProviderKey(m)" in CHATBOT_JS
    assert "gpt-5.4" in CHATBOT_JS
    assert "gpt-5.3-codex-spark" in CHATBOT_JS
    assert "if(!modelMatchesProvider(m))return;" in CHATBOT_JS
    assert "providerPinned=true;" in CHATBOT_JS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
