"""Tests for chatbot UI HTML generation, auto-resize, spinner, clear button, and text wrapping."""

from __future__ import annotations

import queue
import re

import pytest

from kiss.agents.sorcar.browser_ui import OUTPUT_CSS, BaseBrowserPrinter
from kiss.agents.sorcar.chatbot_ui import (
    CHATBOT_CSS,
    CHATBOT_JS,
    CHATBOT_THEME_CSS,
    _build_html,
)


def _css_block_chatbot(selector: str) -> str:
    idx = CHATBOT_CSS.index(f"{selector}{{")
    return CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]


def _css_block_output(css: str, selector: str) -> str:
    pattern = re.escape(selector) + r"\s*\{([^}]*)\}"
    match = re.search(pattern, css)
    return match.group(1) if match else ""


def _get_clear_handler() -> str:
    marker = "clearBtn.addEventListener('click',function(){"
    start = CHATBOT_JS.index(marker) + len(marker)
    depth = 1
    i = start
    while depth > 0:
        if CHATBOT_JS[i] == "{":
            depth += 1
        elif CHATBOT_JS[i] == "}":
            depth -= 1
        i += 1
    return CHATBOT_JS[start : i - 1]


class TestTextareaAutoResize:
    def test_css_task_input_defaults(self) -> None:
        idx = CHATBOT_CSS.index("#task-input{")
        block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
        assert "max-height:50vh" in block
        assert "max-height:200px" not in block
        assert "overflow-y:hidden" in block

    def test_js_auto_resize_no_200px_cap(self) -> None:
        assert "Math.min(this.scrollHeight,200)" not in CHATBOT_JS

    def test_js_resize_and_overflow_behavior(self) -> None:
        assert "inp.style.height=inp.scrollHeight+'px'" in CHATBOT_JS
        assert "inp.style.overflowY=inp.scrollHeight>inp.clientHeight?'auto':'hidden'" in CHATBOT_JS
        assert "inp.style.overflowY='hidden'" in CHATBOT_JS


def test_model_picker_shrinks_on_zoom():
    idx = CHATBOT_CSS.index("#model-picker{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "min-width:0" in block
    assert "overflow:visible" in block


def test_input_actions_no_shrink():
    idx = CHATBOT_CSS.index("#input-actions{")
    block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
    assert "flex-shrink:0" in block


class TestGhostCursorPosition:
    def test_fetch_ghost_skips_when_cursor_in_middle(self) -> None:
        fn = CHATBOT_JS.split("function fetchGhost")[1].split("\nfunction ")[0]
        assert "if(pos<inp.value.length){clearGhost();return}" in fn

    def test_update_ghost_renders_at_cursor(self) -> None:
        fn = CHATBOT_JS.split("function updateGhost")[1].split("\nfunction ")[0]
        assert "inp.value.substring(0,pos)" in fn
        assert "inp.value.substring(pos)" in fn

    def test_ghost_mask_text_uses_transparent_color_not_hidden(self) -> None:
        idx = CHATBOT_CSS.index(".gm{")
        block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
        assert "color:transparent" in block
        assert "visibility:hidden" not in block

    def test_ghost_overlay_shares_text_metrics_with_textarea(self) -> None:
        idx = CHATBOT_CSS.index("#task-input,")
        block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
        assert "#ghost-overlay" in block
        assert "white-space:pre-wrap" in block
        assert "word-break:break-word" in block
        assert "box-sizing:border-box" in block
        assert "tab-size:8" in block

    def test_ghost_overlay_syncs_runtime_size_scroll_and_padding(self) -> None:
        fn = CHATBOT_JS.split("function syncGhostOverlay")[1].split("\nfunction ")[0]
        assert "ghostEl.style.width=inp.clientWidth+'px'" in fn
        assert "ghostEl.style.height=inp.clientHeight+'px'" in fn
        assert "ghostEl.style.paddingTop=inp.style.paddingTop" in fn
        assert "ghostEl.style.paddingLeft=inp.style.paddingLeft" in fn
        assert "ghostEl.scrollTop=inp.scrollTop" in fn

    def test_resize_input_keeps_ghost_overlay_in_sync(self) -> None:
        fn = CHATBOT_JS.split("function resizeInput")[1].split("\nfunction ")[0]
        assert "syncGhostOverlay();" in fn

    def test_ghost_suggestion_uses_plain_span_color(self) -> None:
        idx = CHATBOT_CSS.index(".gs{")
        block = CHATBOT_CSS[idx : CHATBOT_CSS.index("}", idx) + 1]
        assert "color:rgba(255,255,255,0.35)" in block
        assert "white-space" not in block

    def test_accept_ghost_inserts_at_cursor(self) -> None:
        fn = CHATBOT_JS.split("function acceptGhost")[1].split("\nfunction ")[0]
        assert "inp.value.substring(0,pos)" in fn
        assert "before+ghostSuggest+after" in fn
        assert "inp.setSelectionRange(newPos,newPos)" in fn

    def test_ghost_cursor_pos_variable(self) -> None:
        assert "var ghostCursorPos=-1" in CHATBOT_JS
        assert "ghostCursorPos=-1" in CHATBOT_JS


def test_wait_spinner_default_state():
    block = _css_block_chatbot("#wait-spinner")
    assert "border-radius:50%" in block
    assert "border-top-color" in block
    assert "display:none" not in block
    assert "display:block" not in block
    assert "opacity:0.4" in block
    assert "animation:" not in block


def test_wait_spinner_active_state():
    block = _css_block_chatbot("#wait-spinner.active")
    assert "animation:spin" in block
    assert "opacity:1" in block
    assert "border-top-color:rgba(88,166,255,0.7)" in block


def test_assistant_panel_wait_spinner_size():
    assert "#assistant-panel #wait-spinner{" in CHATBOT_CSS


def test_themed_wait_spinner_css():
    assert "#assistant-panel #wait-spinner{" in CHATBOT_THEME_CSS
    assert "#assistant-panel #wait-spinner.active{" in CHATBOT_THEME_CSS


def test_js_spinner_controls():
    assert "waitSpinner.classList.add('active')" in CHATBOT_JS
    assert "waitSpinner.classList.remove('active')" in CHATBOT_JS
    assert "var waitSpinner=document.getElementById('wait-spinner')" in CHATBOT_JS
    assert "setTimeout" in CHATBOT_JS.split("function showSpinner")[1].split("function ")[0]


def test_no_stop_btn_waiting_css():
    assert "#stop-btn.waiting" not in CHATBOT_CSS
    assert "stopBtn.classList" not in CHATBOT_JS


class TestClearButtonWelcome:
    def test_handler_exists(self) -> None:
        assert "clearBtn.addEventListener('click'" in CHATBOT_JS

    def test_handler_guards_and_replaces_output(self) -> None:
        handler = _get_clear_handler()
        assert "if(running)return;" in handler
        assert 'id="welcome"' in handler
        assert "What can I help you with?" in handler
        assert 'id="suggestions"' in handler

    def test_handler_rebinds_and_resets_state(self) -> None:
        handler = _get_clear_handler()
        assert "suggestionsEl=document.getElementById('suggestions')" in handler
        assert "resetOutputState()" in handler
        assert "pendingFiles=[]" in handler
        assert "renderFileChips()" in handler

    def test_reset_output_state_function(self) -> None:
        assert "function resetOutputState(){" in CHATBOT_JS
        fn_start = CHATBOT_JS.index("function resetOutputState(){")
        fn_body = CHATBOT_JS[fn_start : CHATBOT_JS.index("}", fn_start) + 1]
        for expected in ("state=mkS()", "llmPanel=null", "llmPanelState=mkS()",
                         "lastToolName=''", "pendingPanel=false", "_scrollLock=false"):
            assert expected in fn_body

    def test_handler_clears_input_loads_welcome_and_focuses(self) -> None:
        handler = _get_clear_handler()
        assert "loadWelcome()" in handler
        assert "inp.value=''" in handler
        assert "inp.focus()" in handler
        welcome_pos = handler.index('id="welcome"')
        load_welcome_pos = handler.index("loadWelcome()")
        clear_input_pos = handler.index("inp.value=''")
        assert welcome_pos < load_welcome_pos < clear_input_pos


class TestToolCallHeaderWrapping:
    def test_tc_h_flex_wrap(self) -> None:
        block = _css_block_output(OUTPUT_CSS, ".tc-h")
        assert "display:flex" in block
        assert "flex-wrap:wrap" in block

    def test_tp_word_break_and_min_width(self) -> None:
        block = _css_block_output(OUTPUT_CSS, ".tp")
        assert "word-break:break-all" in block
        assert "min-width:0" in block

    def test_td_word_break_and_min_width(self) -> None:
        block = _css_block_output(OUTPUT_CSS, ".td")
        assert "word-break:break-word" in block
        assert "min-width:0" in block


class TestUsageInfoWrapping:
    def test_usage_wrapping_properties(self) -> None:
        block = _css_block_output(OUTPUT_CSS, ".usage")
        assert "white-space:pre-wrap" in block
        assert "word-break:break-word" in block
        assert "nowrap" not in block
        assert "overflow-x" not in block
        assert "overflow-wrap:break-word" in block


class TestBuildHtmlContainsWrappingCSS:
    def setup_method(self) -> None:
        self.html = _build_html("Test", "", "/tmp")


class TestToolCallBroadcastLongContent:
    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()
        self.cq = self.printer.add_client()

    def teardown_method(self) -> None:
        self.printer.remove_client(self.cq)

    def _drain(self) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                events.append(self.cq.get_nowait())
            except queue.Empty:
                break
        return events


class TestUsageInfoBroadcastLongContent:
    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()
        self.cq = self.printer.add_client()

    def teardown_method(self) -> None:
        self.printer.remove_client(self.cq)


class TestChatbotCSSWrapping:
    def test_no_nowrap_in_usage_or_tc_h(self) -> None:
        for selector in (".usage", "#assistant-panel .usage", "#assistant-panel .tc-h"):
            block = _css_block_output(CHATBOT_CSS, selector)
            assert "nowrap" not in block, f"nowrap found in {selector}"


class TestEventHandlerJSToolCallRendering:
    def test_js_contains_tool_call_classes(self) -> None:
        from kiss.agents.sorcar.browser_ui import EVENT_HANDLER_JS

        for cls in ("tp", "td", "tn"):
            assert f'class="{cls}"' in EVENT_HANDLER_JS or f"class=\"{cls}\"" in EVENT_HANDLER_JS

    def test_js_usage_handler_creates_usage_div(self) -> None:
        from kiss.agents.sorcar.browser_ui import EVENT_HANDLER_JS

        assert "usage_info" in EVENT_HANDLER_JS
        assert "usage" in EVENT_HANDLER_JS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
