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

    def test_build_html_contains_textarea(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert '<textarea id="task-input"' in html
        assert "max-height:50vh" in html


def test_chatbox_has_three_lines():
    """Test that the textarea has rows=3 attribute."""
    html = _build_html("Test", "", "/tmp")
    assert 'rows="3"' in html, "Chatbox textarea should have rows=3"
    assert 'rows="1"' not in html, "Chatbox should not have rows=1"


def test_chatbox_min_height_css():
    """Test that the CSS has min-height of 68px (3 lines)."""
    html = _build_html("Test", "", "/tmp")
    assert "min-height:68px" in html, "Chatbox should have min-height:68px for 3 lines"


def test_autocomplete_resize_in_js():
    """Test that the JS includes resize logic after autocomplete selection."""
    html = _build_html("Test", "", "/tmp")
    assert "inp.style.height='auto'" in html, "selectAC should reset height"
    assert "inp.style.height=inp.scrollHeight+'px'" in html, (
        "selectAC should set height to scrollHeight"
    )


def test_ghost_accept_resize_in_js():
    """Test that the JS includes resize logic after ghost text acceptance."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    resize_count = js.count("inp.style.height='auto'")
    assert resize_count >= 2, (
        f"Should have at least 2 resize calls (input handler + autocomplete + ghost), "
        f"found {resize_count}"
    )


def test_input_resize_handler():
    """Test that input handler has resize logic."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "inp.addEventListener('input'" in js, "Should have input event listener"
    assert "this.style.height='auto'" in js, "Input handler should reset height"


def test_autocomplete_items_have_data_text():
    """Test that autocomplete items store text in dataset.text attribute."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "d.dataset.text=item.text" in js, "AC items should store text in dataset.text"


def test_autocomplete_ghost_preview_on_selection():
    """Test that navigating autocomplete items shows ghost preview text."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "items[acIdx].dataset.text" in js, "updateACSel should read dataset.text"
    assert "ghostSuggest=fullPath.substring(query.length)" in js, (
        "updateACSel should set ghost suggest from selected item"
    )


def test_hide_ac_clears_ghost():
    """Test that hiding autocomplete clears ghost text."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "function hideAC(){ac.style.display='none';acIdx=-1;clearGhost()}" in js, (
        "hideAC should clear ghost text"
    )


def test_cmd_k_keybinding_in_js():
    """Test that Cmd+K / Ctrl+K toggle focus keybinding is present in JS."""
    html = _build_html("Test", "http://127.0.0.1:13338", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "e.key==='k'" in js, "Should listen for 'k' key"
    assert "e.metaKey" in js, "Should check metaKey (Cmd on Mac)"
    assert "e.ctrlKey" in js, "Should check ctrlKey (Ctrl on Windows/Linux)"
    assert "/focus-editor" in js, "Should call focus-editor endpoint"
    assert "inp.focus()" in js, "Should focus the chatbox input"


def test_cmd_k_focuses_code_server_frame():
    """Test that the keybinding references the code-server-frame element."""
    html = _build_html("Test", "http://127.0.0.1:13338", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "getElementById('code-server-frame')" in js, (
        "Keybinding should look up the code-server iframe"
    )


def test_focus_chatbox_sse_handler():
    """Test that the SSE handler for focus_chatbox event is present."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "case'focus_chatbox'" in js, "Should handle focus_chatbox SSE event"


def test_cmd_k_not_present_without_code_server():
    """Test that Cmd+K keybinding is present even without code-server (graceful no-op)."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "e.key==='k'" in js


def test_autocomplete_select_removes_at_symbol():
    """Test that selecting a file via autocomplete does NOT insert '@' prefix."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "inp.value=before+item.text+sep+after" in js, (
        "selectAC should insert path WITHOUT '@' prefix"
    )
    assert "inp.value=before+'@'+item.text" not in js, (
        "selectAC should NOT insert '@' before the path"
    )


def test_autocomplete_select_cursor_position():
    """Test that cursor position after autocomplete does not include '@' offset."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "np=before.length+item.text.length+sep.length" in js, (
        "Cursor position should not add +1 for '@'"
    )
    assert "np=before.length+1+item.text.length" not in js, (
        "Cursor position should not include +1 offset for '@'"
    )


def test_clear_event_appends_active_file_to_user_msg():
    """Test that the clear SSE event augments the user message with active_file info."""
    html = _build_html("Test", "", "/tmp")
    js = html.split("<script>")[1].split("</script>")[0]
    assert "ev.active_file" in js, "clear event handler should check ev.active_file"
    assert "Currently open file in editor" in js, (
        "clear event handler should append editor file info to user message"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
