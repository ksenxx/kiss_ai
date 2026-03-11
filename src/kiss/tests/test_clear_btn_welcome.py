"""Test that the clear (X) button clears the chatbox and shows the welcome window.

Verifies the clearBtn click handler in CHATBOT_JS:
1. Clears the output area and replaces it with the welcome screen
2. Resets all state (llmPanel, pendingFiles, etc.)
3. Calls loadWelcome() to populate recent tasks
4. Clears the input textbox (inp.value='')
"""

from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS


def _get_clear_handler() -> str:
    """Extract the clearBtn click handler body from CHATBOT_JS."""
    marker = "clearBtn.addEventListener('click',function(){"
    start = CHATBOT_JS.index(marker) + len(marker)
    # Find matching closing brace by counting braces
    depth = 1
    i = start
    while depth > 0:
        if CHATBOT_JS[i] == "{":
            depth += 1
        elif CHATBOT_JS[i] == "}":
            depth -= 1
        i += 1
    return CHATBOT_JS[start : i - 1]


class TestClearButtonWelcome:
    """Verify the X button clears chatbox and launches welcome."""

    def test_handler_exists(self) -> None:
        """The clearBtn click handler must exist in CHATBOT_JS."""
        assert "clearBtn.addEventListener('click'" in CHATBOT_JS

    def test_guards_while_running(self) -> None:
        """Clear should be a no-op while the agent is running."""
        handler = _get_clear_handler()
        assert "if(running)return;" in handler

    def test_replaces_output_with_welcome(self) -> None:
        """The output area must be replaced with the welcome div."""
        handler = _get_clear_handler()
        assert 'id="welcome"' in handler
        assert "What can I help you with?" in handler
        assert 'id="suggestions"' in handler

    def test_rebinds_suggestions_element(self) -> None:
        """suggestionsEl must be re-bound after replacing innerHTML."""
        handler = _get_clear_handler()
        assert "suggestionsEl=document.getElementById('suggestions')" in handler

    def test_resets_state(self) -> None:
        """Internal state variables must be reset."""
        handler = _get_clear_handler()
        assert "state=mkS()" in handler
        assert "llmPanel=null" in handler
        assert "llmPanelState=mkS()" in handler
        assert "lastToolName=''" in handler
        assert "pendingPanel=false" in handler
        assert "_scrollLock=false" in handler

    def test_clears_pending_files(self) -> None:
        """Pending file attachments must be cleared."""
        handler = _get_clear_handler()
        assert "pendingFiles=[]" in handler
        assert "renderFileChips()" in handler

    def test_calls_load_welcome(self) -> None:
        """loadWelcome() must be called to populate recent tasks."""
        handler = _get_clear_handler()
        assert "loadWelcome()" in handler

    def test_clears_input_value(self) -> None:
        """The textarea input must be emptied."""
        handler = _get_clear_handler()
        assert "inp.value=''" in handler

    def test_focuses_input(self) -> None:
        """Input should be focused after clearing."""
        handler = _get_clear_handler()
        assert "inp.focus()" in handler

    def test_clear_then_welcome_order(self) -> None:
        """Welcome HTML must be set before loadWelcome() is called,
        and input must be cleared after."""
        handler = _get_clear_handler()
        welcome_pos = handler.index('id="welcome"')
        load_welcome_pos = handler.index("loadWelcome()")
        clear_input_pos = handler.index("inp.value=''")
        assert welcome_pos < load_welcome_pos < clear_input_pos
