"""Tests for chatbot UI changes - chatbox height and autocomplete expansion."""

import pytest
from kiss.agents.assistant.chatbot_ui import _build_html


def test_chatbox_has_four_lines():
    """Test that the textarea has rows=4 attribute."""
    html = _build_html("Test", "", "/tmp")
    
    # Check that the textarea has rows="4"
    assert 'rows="4"' in html, "Chatbox textarea should have rows=4"
    
    # Verify the old rows="1" is not present
    assert 'rows="1"' not in html, "Chatbox should not have rows=1"


def test_chatbox_min_height_css():
    """Test that the CSS has min-height of 96px (4 lines * 24px)."""
    html = _build_html("Test", "", "/tmp")
    
    # Check that min-height:96px is in the CSS
    assert "min-height:96px" in html, "Chatbox should have min-height:96px for 4 lines"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
