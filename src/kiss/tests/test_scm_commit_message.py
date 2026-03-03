"""Tests for the SCM commit message generation feature in VS Code extension."""

import json
import os
import tempfile
import unittest

from kiss.agents.sorcar.code_server import _CS_EXTENSION_JS, _CS_SETTINGS, _setup_code_server


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
        depth = _CS_SETTINGS["git.repositoryScanMaxDepth"]
        assert isinstance(depth, int) and depth >= 1

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


class TestGenerateCommitMessageCommand(unittest.TestCase):
    def test_extension_has_generate_commit_message_command(self) -> None:
        assert "kiss.generateCommitMessage" in _CS_EXTENSION_JS

    def test_extension_reads_assistant_port(self) -> None:
        assert "assistant-port" in _CS_EXTENSION_JS

    def test_extension_calls_generate_endpoint(self) -> None:
        assert "/generate-commit-message" in _CS_EXTENSION_JS

    def test_extension_shows_generating_placeholder(self) -> None:
        assert "Generating commit message" in _CS_EXTENSION_JS

    def test_extension_handles_error_response(self) -> None:
        assert "body.error" in _CS_EXTENSION_JS
        assert "showErrorMessage" in _CS_EXTENSION_JS

    def test_extension_sets_scm_input_on_success(self) -> None:
        assert "body.message" in _CS_EXTENSION_JS

    def test_chatbot_js_does_not_contain_magic_btn(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        assert "magicBtn" not in CHATBOT_JS
        assert "magic-btn" not in CHATBOT_JS


class TestExtensionPackageContributes(unittest.TestCase):
    def test_package_has_generate_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = os.path.join(tmpdir, "extensions", "kiss-init", "package.json")
            with open(pkg_path) as f:
                pkg = json.load(f)
            commands = pkg["contributes"]["commands"]
            cmd_names = [c["command"] for c in commands]
            assert "kiss.generateCommitMessage" in cmd_names

    def test_package_has_sparkle_icon(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = os.path.join(tmpdir, "extensions", "kiss-init", "package.json")
            with open(pkg_path) as f:
                pkg = json.load(f)
            commands = pkg["contributes"]["commands"]
            gen_cmd = [c for c in commands if c["command"] == "kiss.generateCommitMessage"][0]
            assert gen_cmd["icon"] == "$(sparkle)"

    def test_package_has_scm_input_menu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = os.path.join(tmpdir, "extensions", "kiss-init", "package.json")
            with open(pkg_path) as f:
                pkg = json.load(f)
            menus = pkg["contributes"]["menus"]
            assert "scm/inputBox" in menus
            scm_menu = menus["scm/inputBox"]
            assert any(m["command"] == "kiss.generateCommitMessage" for m in scm_menu)

    def test_scm_menu_has_git_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _setup_code_server(tmpdir)
            pkg_path = os.path.join(tmpdir, "extensions", "kiss-init", "package.json")
            with open(pkg_path) as f:
                pkg = json.load(f)
            scm_menu = pkg["contributes"]["menus"]["scm/inputBox"]
            gen_item = [m for m in scm_menu if m["command"] == "kiss.generateCommitMessage"][0]
            assert gen_item["when"] == "scmProvider == git"


class TestExtensionJSScmBlockOrder(unittest.TestCase):
    def test_scm_handling_before_merge_check(self) -> None:
        scm_pos = _CS_EXTENSION_JS.index("fs.existsSync(sp)")
        merge_pos = _CS_EXTENSION_JS.index("fs.existsSync(mp)")
        assert scm_pos < merge_pos


if __name__ == "__main__":
    unittest.main()
