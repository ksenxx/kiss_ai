"""Tests for the stop button spinner behavior in chatbot_ui."""

from kiss.agents.sorcar.chatbot_ui import CHATBOT_CSS, CHATBOT_JS, CHATBOT_THEME_CSS, _build_html


def test_stop_btn_waiting_css_exists():
    assert "#stop-btn.waiting svg{display:none}" in CHATBOT_CSS
    assert "#stop-btn.waiting::after{" in CHATBOT_CSS
    assert "animation:spin" in CHATBOT_CSS


def test_stop_btn_waiting_css_has_border_spinner():
    idx = CHATBOT_CSS.index("#stop-btn.waiting::after{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "border-radius:50%" in block
    assert "border-top-color" in block


def test_stop_btn_waiting_style_override():
    assert "#stop-btn.waiting{" in CHATBOT_CSS
    block = CHATBOT_CSS.split("#stop-btn.waiting{")[1].split("}")[0]
    assert "border-color:rgba(88,166,255" in block


def test_assistant_panel_stop_btn_waiting_after_size():
    assert "#assistant-panel #stop-btn.waiting::after{width:11px;height:11px}" in CHATBOT_CSS


def test_themed_stop_btn_waiting_css():
    assert "#assistant-panel #stop-btn.waiting{" in CHATBOT_THEME_CSS
    assert "#assistant-panel #stop-btn.waiting::after{" in CHATBOT_THEME_CSS


def test_js_show_spinner_toggles_class():
    assert "stopBtn.classList.add('waiting')" in CHATBOT_JS


def test_js_remove_spinner_toggles_class():
    assert "stopBtn.classList.remove('waiting')" in CHATBOT_JS


def test_js_no_dom_spinner_element():
    assert "wait-spinner" not in CHATBOT_JS
    assert "Waiting ..." not in CHATBOT_JS


def test_js_show_spinner_has_delay():
    assert "setTimeout" in CHATBOT_JS.split("function showSpinner")[1].split("function ")[0]


def test_build_html_contains_stop_btn_waiting_css():
    html = _build_html("Test", "", "/tmp")
    assert "#stop-btn.waiting" in html
    assert "stopBtn.classList.add('waiting')" in html
    assert "stopBtn.classList.remove('waiting')" in html


def test_build_html_no_output_spinner():
    html = _build_html("Test", "", "/tmp")
    assert "wait-spinner" not in html
