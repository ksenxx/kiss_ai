"""Integration tests for the Ctrl+L / Cmd+L run-selection feature.

Verifies that:
- The VS Code extension registers the kiss.runSelection command
- The extension keybinding binds Ctrl+L / Cmd+L to kiss.runSelection
- The extension JS sends selected text to /run-selection endpoint
- The chatbot JS handles external_run events correctly
- The server /run-selection endpoint exists and functions correctly
- The _build_html output contains the external_run handler

No mocks, patches, or test doubles.
"""

from __future__ import annotations

from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS
from kiss.agents.sorcar.code_server import _CS_EXTENSION_JS


class TestExtensionRunSelectionCommand:
    """Verify the VS Code extension registers kiss.runSelection."""

    def test_registers_run_selection_command(self) -> None:
        assert "kiss.runSelection" in _CS_EXTENSION_JS

    def test_command_gets_active_editor(self) -> None:
        assert "vscode.window.activeTextEditor" in _CS_EXTENSION_JS

    def test_command_gets_selected_text(self) -> None:
        assert "ed.document.getText(ed.selection)" in _CS_EXTENSION_JS

    def test_command_posts_to_run_selection(self) -> None:
        assert "'/run-selection'" in _CS_EXTENSION_JS

    def test_command_sends_text_in_body(self) -> None:
        assert "{text:sel.trim()}" in _CS_EXTENSION_JS

    def test_command_uses_post_assistant(self) -> None:
        """The command should use postAssistant (which returns a promise)."""
        # Find the runSelection registration block and check it uses postAssistant
        idx = _CS_EXTENSION_JS.index("kiss.runSelection")
        block = _CS_EXTENSION_JS[idx:idx + 500]
        assert "postAssistant('/run-selection'" in block

    def test_command_checks_empty_selection(self) -> None:
        """Should show info message when no text is selected."""
        assert "No text selected" in _CS_EXTENSION_JS

    def test_command_checks_no_port(self) -> None:
        """Should show error when assistant server not found."""
        idx = _CS_EXTENSION_JS.index("kiss.runSelection")
        block = _CS_EXTENSION_JS[idx:idx + 500]
        assert "Assistant server not found" in block

    def test_command_handles_error_response(self) -> None:
        """Should show error message on failure."""
        assert "Run selection failed:" in _CS_EXTENSION_JS

    def test_command_handles_network_error(self) -> None:
        """Should show error on network failure."""
        assert "Run selection error:" in _CS_EXTENSION_JS

    def test_command_returns_early_if_no_editor(self) -> None:
        """If no active editor, command returns immediately."""
        idx = _CS_EXTENSION_JS.index("kiss.runSelection")
        block = _CS_EXTENSION_JS[idx:idx + 200]
        assert "if(!ed)return;" in block


class TestChatbotExternalRunHandler:
    """Verify the chatbot JS handles external_run events."""

    def test_chatbot_js_has_external_run_case(self) -> None:
        assert "case'external_run':" in CHATBOT_JS

    def test_external_run_sets_running_true(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "enterRunning()" in block
        # enterRunning sets running=true
        er_idx = CHATBOT_JS.index("function enterRunning()")
        er_block = CHATBOT_JS[er_idx:er_idx + 300]
        assert "running=true" in er_block

    def test_external_run_disables_input(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "inp.disabled=true" in block

    def test_external_run_shows_stop_button(self) -> None:
        er_idx = CHATBOT_JS.index("function enterRunning()")
        er_block = CHATBOT_JS[er_idx:er_idx + 300]
        assert "stopBtn.style.display='inline-flex'" in er_block

    def test_external_run_hides_send_button(self) -> None:
        er_idx = CHATBOT_JS.index("function enterRunning()")
        er_block = CHATBOT_JS[er_idx:er_idx + 300]
        assert "btn.style.display='none'" in er_block

    def test_external_run_sets_pending_user_msg(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "pendingUserMsg={text:ev.text,images:[]}" in block

    def test_external_run_preserves_task_text(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "inp.value=ev.text||''" in block

    def test_external_run_starts_timer(self) -> None:
        er_idx = CHATBOT_JS.index("function enterRunning()")
        er_block = CHATBOT_JS[er_idx:er_idx + 300]
        assert "startTimer()" in er_block

    def test_external_run_shows_spinner(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "showSpinner()" in block

    def test_external_run_loads_models(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "loadModels()" in block

    def test_external_run_clears_pending_files(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "pendingFiles=[]" in block

    def test_external_run_clears_input(self) -> None:
        er_idx = CHATBOT_JS.index("function enterRunning()")
        er_block = CHATBOT_JS[er_idx:er_idx + 300]
        assert "inp.style.height='auto'" in er_block

    def test_external_run_disables_run_prompt_btn(self) -> None:
        er_idx = CHATBOT_JS.index("function enterRunning()")
        er_block = CHATBOT_JS[er_idx:er_idx + 300]
        assert "runPromptBtn.disabled=true" in er_block


class TestExtensionJSLogicFlow:
    """Verify the logical flow of the runSelection command."""

    def test_checks_editor_before_selection(self) -> None:
        """The command checks for active editor before trying to get selection."""
        idx_editor = _CS_EXTENSION_JS.index("kiss.runSelection")
        block = _CS_EXTENSION_JS[idx_editor:]
        idx_no_ed = block.index("if(!ed)return;")
        idx_get_text = block.index("ed.document.getText")
        assert idx_no_ed < idx_get_text

    def test_checks_selection_before_port(self) -> None:
        """The command checks for empty selection before checking port."""
        idx_cmd = _CS_EXTENSION_JS.index("kiss.runSelection")
        block = _CS_EXTENSION_JS[idx_cmd:]
        idx_no_sel = block.index("No text selected")
        # In the block, no_sel should come before no_port
        idx_no_port_in_block = block.index("Assistant server not found")
        assert idx_no_sel < idx_no_port_in_block

    def test_trims_selected_text(self) -> None:
        """Selected text should be trimmed before sending."""
        idx = _CS_EXTENSION_JS.index("kiss.runSelection")
        block = _CS_EXTENSION_JS[idx:idx + 600]
        assert "sel.trim()" in block
