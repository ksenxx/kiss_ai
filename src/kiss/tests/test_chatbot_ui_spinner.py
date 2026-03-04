"""Tests for the wait spinner behavior in chatbot_ui."""

from kiss.agents.sorcar.chatbot_ui import CHATBOT_CSS, CHATBOT_JS, CHATBOT_THEME_CSS, _build_html


def test_wait_spinner_css_exists():
    assert "#wait-spinner{" in CHATBOT_CSS


def test_wait_spinner_css_has_border_spinner():
    idx = CHATBOT_CSS.index("#wait-spinner{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "border-radius:50%" in block
    assert "border-top-color" in block


def test_wait_spinner_always_visible():
    idx = CHATBOT_CSS.index("#wait-spinner{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "display:none" not in block
    assert "display:block" not in block


def test_wait_spinner_greyed_out_by_default():
    idx = CHATBOT_CSS.index("#wait-spinner{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "opacity:0.4" in block


def test_wait_spinner_no_animation_by_default():
    idx = CHATBOT_CSS.index("#wait-spinner{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "animation:" not in block


def test_wait_spinner_active_has_animation():
    idx = CHATBOT_CSS.index("#wait-spinner.active{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "animation:spin" in block
    assert "opacity:1" in block


def test_wait_spinner_active_has_accent_color():
    idx = CHATBOT_CSS.index("#wait-spinner.active{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "border-top-color:rgba(88,166,255,0.7)" in block


def test_assistant_panel_wait_spinner_size():
    assert "#assistant-panel #wait-spinner{" in CHATBOT_CSS


def test_themed_wait_spinner_css():
    assert "#assistant-panel #wait-spinner{" in CHATBOT_THEME_CSS


def test_themed_wait_spinner_active_css():
    assert "#assistant-panel #wait-spinner.active{" in CHATBOT_THEME_CSS


def test_js_show_spinner_toggles_class():
    assert "waitSpinner.classList.add('active')" in CHATBOT_JS


def test_js_remove_spinner_toggles_class():
    assert "waitSpinner.classList.remove('active')" in CHATBOT_JS


def test_js_wait_spinner_element_ref():
    assert "var waitSpinner=document.getElementById('wait-spinner')" in CHATBOT_JS


def test_js_show_spinner_has_delay():
    assert "setTimeout" in CHATBOT_JS.split("function showSpinner")[1].split("function ")[0]


def test_build_html_contains_wait_spinner():
    html = _build_html("Test", "", "/tmp")
    assert 'id="wait-spinner"' in html
    assert "#wait-spinner" in html
    assert "waitSpinner.classList.add('active')" in html
    assert "waitSpinner.classList.remove('active')" in html


def test_build_html_spinner_before_send_btn():
    html = _build_html("Test", "", "/tmp")
    spinner_pos = html.index('id="wait-spinner"')
    send_pos = html.index('id="send-btn"')
    stop_pos = html.index('id="stop-btn"')
    assert spinner_pos < send_pos
    assert spinner_pos < stop_pos


def test_no_stop_btn_waiting_css():
    assert "#stop-btn.waiting" not in CHATBOT_CSS
    assert "stopBtn.classList" not in CHATBOT_JS
