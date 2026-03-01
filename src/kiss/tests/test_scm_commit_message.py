"""Tests for the SCM commit message generation feature."""

import json
import os
import tempfile
import unittest

from kiss.agents.assistant.code_server import _CS_EXTENSION_JS, _CS_SETTINGS, _setup_code_server


class TestScmMessageExtensionJS(unittest.TestCase):
    def test_extension_polls_for_pending_scm_message(self) -> None:
        assert "pending-scm-message.json" in _CS_EXTENSION_JS

    def test_extension_reads_scm_message_file(self) -> None:
        assert "fs.existsSync(sp)" in _CS_EXTENSION_JS

    def test_extension_uses_git_extension_api(self) -> None:
        assert "vscode.extensions.getExtension('vscode.git')" in _CS_EXTENSION_JS

    def test_extension_sets_inputbox_value(self) -> None:
        assert "git.repositories[0].inputBox.value=sd.message" in _CS_EXTENSION_JS

    def test_extension_opens_scm_view(self) -> None:
        assert "workbench.view.scm" in _CS_EXTENSION_JS

    def test_extension_unlinks_scm_message_file(self) -> None:
        assert "fs.unlinkSync(sp)" in _CS_EXTENSION_JS

    def test_sp_variable_declared_with_correct_path(self) -> None:
        expected = "var sp=path.join(home,'.kiss','code-server-data','pending-scm-message.json')"
        assert expected in _CS_EXTENSION_JS


class TestGitSettingsEnabled(unittest.TestCase):
    def test_git_auto_repository_detection_enabled(self) -> None:
        assert _CS_SETTINGS["git.autoRepositoryDetection"] is True

    def test_git_scan_max_depth_nonzero(self) -> None:
        assert _CS_SETTINGS["git.repositoryScanMaxDepth"] >= 1

    def test_git_open_repository_in_parent_folders(self) -> None:
        assert _CS_SETTINGS["git.openRepositoryInParentFolders"] == "always"


class TestSetupCodeServerGitDependency(unittest.TestCase):
    def test_extension_package_has_git_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = os.path.join(tmpdir, "extensions", "kiss-init", "package.json")
            with open(pkg_path) as f:
                pkg = json.load(f)
            assert "vscode.git" in pkg.get("extensionDependencies", [])


class TestMagicButtonFillsScm(unittest.TestCase):
    def test_magic_btn_js_does_not_fill_task_input(self) -> None:
        from kiss.agents.assistant.chatbot_ui import CHATBOT_JS
        start = CHATBOT_JS.index("magicBtn.addEventListener")
        end = CHATBOT_JS.index("});", start) + 3
        handler = CHATBOT_JS[start:end]
        assert "inp.value" not in handler

    def test_magic_btn_js_calls_generate_commit_message(self) -> None:
        from kiss.agents.assistant.chatbot_ui import CHATBOT_JS
        assert "generate-commit-message" in CHATBOT_JS

    def test_magic_btn_js_shows_error_on_failure(self) -> None:
        from kiss.agents.assistant.chatbot_ui import CHATBOT_JS
        start = CHATBOT_JS.index("magicBtn.addEventListener")
        end = CHATBOT_JS.index("});", start) + 3
        handler = CHATBOT_JS[start:end]
        assert "d.error" in handler
        assert "alert" in handler


class TestExtensionJSScmBlockOrder(unittest.TestCase):
    def test_scm_handling_before_merge_check(self) -> None:
        scm_pos = _CS_EXTENSION_JS.index("fs.existsSync(sp)")
        merge_pos = _CS_EXTENSION_JS.index("fs.existsSync(mp)")
        assert scm_pos < merge_pos


if __name__ == "__main__":
    unittest.main()
