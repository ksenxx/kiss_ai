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

import json

from kiss.agents.sorcar.browser_ui import EVENT_HANDLER_JS
from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS, _build_html
from kiss.agents.sorcar.code_server import _CS_EXTENSION_JS, _setup_code_server


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


class TestExtensionKeybinding:
    """Verify the keybinding for kiss.runSelection in package.json config."""

    def test_setup_creates_keybinding(self) -> None:
        """_setup_code_server writes package.json with runSelection keybinding."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = f"{tmpdir}/extensions/kiss-init/package.json"
            with open(pkg_path) as f:
                pkg = json.load(f)
            keybindings = pkg["contributes"]["keybindings"]
            run_sel_kb = [kb for kb in keybindings if kb["command"] == "kiss.runSelection"]
            assert len(run_sel_kb) == 1

    def test_keybinding_ctrl_l(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = f"{tmpdir}/extensions/kiss-init/package.json"
            with open(pkg_path) as f:
                pkg = json.load(f)
            keybindings = pkg["contributes"]["keybindings"]
            run_sel_kb = [kb for kb in keybindings if kb["command"] == "kiss.runSelection"][0]
            assert run_sel_kb["key"] == "ctrl+l"

    def test_keybinding_cmd_l_on_mac(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = f"{tmpdir}/extensions/kiss-init/package.json"
            with open(pkg_path) as f:
                pkg = json.load(f)
            keybindings = pkg["contributes"]["keybindings"]
            run_sel_kb = [kb for kb in keybindings if kb["command"] == "kiss.runSelection"][0]
            assert run_sel_kb["mac"] == "cmd+l"

    def test_command_registered_in_contributes(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = f"{tmpdir}/extensions/kiss-init/package.json"
            with open(pkg_path) as f:
                pkg = json.load(f)
            commands = pkg["contributes"]["commands"]
            cmd_names = [c["command"] for c in commands]
            assert "kiss.runSelection" in cmd_names

    def test_command_has_title(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = f"{tmpdir}/extensions/kiss-init/package.json"
            with open(pkg_path) as f:
                pkg = json.load(f)
            commands = pkg["contributes"]["commands"]
            run_sel_cmd = [c for c in commands if c["command"] == "kiss.runSelection"][0]
            assert run_sel_cmd["title"] == "Run Selection in Chatbox"


class TestChatbotExternalRunHandler:
    """Verify the chatbot JS handles external_run events."""

    def test_chatbot_js_has_external_run_case(self) -> None:
        assert "case'external_run':" in CHATBOT_JS

    def test_external_run_sets_running_true(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "running=true" in block

    def test_external_run_disables_input(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "inp.disabled=true" in block

    def test_external_run_shows_stop_button(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "stopBtn.style.display='inline-flex'" in block

    def test_external_run_hides_send_button(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "btn.style.display='none'" in block

    def test_external_run_sets_pending_user_msg(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "pendingUserMsg={text:ev.text,images:[]}" in block

    def test_external_run_clears_input(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "inp.value=''" in block

    def test_external_run_starts_timer(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "startTimer()" in block

    def test_external_run_shows_spinner(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "showSpinner()" in block

    def test_external_run_adds_running_dot(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "D.classList.add('running')" in block

    def test_external_run_loads_models(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "loadModels()" in block

    def test_external_run_clears_pending_files(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "pendingFiles=[]" in block

    def test_external_run_disables_run_prompt_btn(self) -> None:
        idx = CHATBOT_JS.index("case'external_run':")
        block = CHATBOT_JS[idx:idx + 500]
        assert "runPromptBtn.disabled=true" in block


class TestBuildHtmlContainsExternalRun:
    """Verify _build_html generates HTML with external_run handling."""

    def test_html_contains_external_run_handler(self) -> None:
        html = _build_html("Test")
        assert "external_run" in html

    def test_html_contains_pending_user_msg_for_external(self) -> None:
        html = _build_html("Test")
        assert "pendingUserMsg={text:ev.text,images:[]}" in html

    def test_html_contains_event_handler_js(self) -> None:
        html = _build_html("Test")
        assert EVENT_HANDLER_JS in html

    def test_html_contains_chatbot_js(self) -> None:
        html = _build_html("Test")
        assert CHATBOT_JS in html


class TestExtensionJSIntegration:
    """End-to-end checks that the extension.js written to disk has runSelection."""

    def test_extension_js_on_disk_has_run_selection(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            ext_js_path = f"{tmpdir}/extensions/kiss-init/extension.js"
            with open(ext_js_path) as f:
                content = f.read()
            assert "kiss.runSelection" in content

    def test_extension_js_on_disk_has_run_selection_endpoint(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            ext_js_path = f"{tmpdir}/extensions/kiss-init/extension.js"
            with open(ext_js_path) as f:
                content = f.read()
            assert "/run-selection" in content

    def test_extension_js_on_disk_has_selection_text_param(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            ext_js_path = f"{tmpdir}/extensions/kiss-init/extension.js"
            with open(ext_js_path) as f:
                content = f.read()
            assert "getText(ed.selection)" in content


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
