"""Tests for chatbot UI changes - chatbox height and autocomplete expansion."""

import pytest
from kiss.agents.assistant.chatbot_ui import _build_html


def test_chatbox_has_three_lines():
    """Test that the textarea has rows=3 attribute."""
    html = _build_html("Test", "", "/tmp")
    
    # Check that the textarea has rows="3"
    assert 'rows="3"' in html, "Chatbox textarea should have rows=3"
    
    # Verify the old rows="1" is not present
    assert 'rows="1"' not in html, "Chatbox should not have rows=1"


def test_chatbox_min_height_css():
    """Test that the CSS has min-height of 68px (3 lines)."""
    html = _build_html("Test", "", "/tmp")
    
    # Check that min-height:68px is in the CSS
    assert "min-height:68px" in html, "Chatbox should have min-height:68px for 3 lines"


def test_autocomplete_resize_in_js():
    """Test that the JS includes resize logic after autocomplete selection."""
    html = _build_html("Test", "", "/tmp")
    
    # Check that selectAC function includes height adjustment
    assert "inp.style.height='auto'" in html, "selectAC should reset height"
    assert "inp.style.height=inp.scrollHeight+'px'" in html, "selectAC should set height to scrollHeight"


def test_ghost_accept_resize_in_js():
    """Test that the JS includes resize logic after ghost text acceptance."""
    html = _build_html("Test", "", "/tmp")
    
    # Check that acceptGhost function includes height adjustment
    # There should be at least 2 occurrences - one in selectAC and one in acceptGhost
    js = html.split("<script>")[1].split("</script>")[0]
    
    # Count occurrences of the resize pattern
    resize_count = js.count("inp.style.height='auto'")
    assert resize_count >= 2, f"Should have at least 2 resize calls (input handler + autocomplete + ghost), found {resize_count}"


def test_input_resize_handler():
    """Test that input handler has resize logic."""
    html = _build_html("Test", "", "/tmp")
    
    # Check that input event handler has resize logic
    js = html.split("<script>")[1].split("</script>")[0]
    
    # The input handler should resize on input
    assert "inp.addEventListener('input'" in js, "Should have input event listener"
    
    # Check for the resize pattern in the input handler
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
