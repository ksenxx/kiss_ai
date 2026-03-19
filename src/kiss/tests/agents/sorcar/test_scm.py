"""Tests for SCM: commit attribution and commit message generation via VS Code extension."""

import os
import subprocess
import tempfile

import pytest

from kiss.agents.sorcar.code_server import _CS_EXTENSION_JS, _CS_SETTINGS


def test_git_commit_with_kiss_sorcar_attribution():
    with tempfile.TemporaryDirectory() as repo:
        subprocess.run(["git", "init"], cwd=repo, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo,
            capture_output=True,
        )
        with open(os.path.join(repo, "file.txt"), "w") as f:
            f.write("hello")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        commit_env = {
            **os.environ,
            "GIT_COMMITTER_NAME": "KISS Sorcar",
            "GIT_COMMITTER_EMAIL": "kiss-sorcar@users.noreply.github.com",
        }
        subprocess.run(
            [
                "git", "commit", "-m", "test commit",
                "--author=KISS Sorcar <kiss-sorcar@users.noreply.github.com>",
            ],
            cwd=repo,
            capture_output=True,
            env=commit_env,
        )
        author = subprocess.run(
            ["git", "log", "-1", "--format=%an <%ae>"],
            cwd=repo, capture_output=True, text=True,
        ).stdout.strip()
        committer = subprocess.run(
            ["git", "log", "-1", "--format=%cn <%ce>"],
            cwd=repo, capture_output=True, text=True,
        ).stdout.strip()
        assert author == "KISS Sorcar <kiss-sorcar@users.noreply.github.com>"
        assert committer == "KISS Sorcar <kiss-sorcar@users.noreply.github.com>"


_SCM_EXTENSION_STRINGS = [
    "pending-scm-message.json",
    "fs.existsSync(sp)",
    "vscode.extensions.getExtension('vscode.git')",
    "git.repositories[0].inputBox.value=sd.message",
    "workbench.view.scm",
    "fs.unlinkSync(sp)",
    "var sp=path.join(dataDir,'pending-scm-message.json')",
]

_COMMIT_MSG_EXTENSION_STRINGS = [
    "kiss.generateCommitMessage",
    "assistant-port",
    "/generate-commit-message",
    "Generating commit message",
    "body.error",
    "showErrorMessage",
    "body.message",
]


class TestScmMessageExtensionJS:
    @pytest.mark.parametrize("expected", _SCM_EXTENSION_STRINGS)
    def test_extension_has_scm_string(self, expected: str) -> None:
        assert expected in _CS_EXTENSION_JS


class TestGitSettingsEnabled:
    def test_git_settings(self) -> None:
        assert _CS_SETTINGS["git.autoRepositoryDetection"] is True
        depth = _CS_SETTINGS["git.repositoryScanMaxDepth"]
        assert isinstance(depth, int) and depth >= 1
        assert _CS_SETTINGS["git.openRepositoryInParentFolders"] == "always"


class TestGenerateCommitMessageCommand:
    @pytest.mark.parametrize("expected", _COMMIT_MSG_EXTENSION_STRINGS)
    def test_extension_has_commit_msg_string(self, expected: str) -> None:
        assert expected in _CS_EXTENSION_JS

    def test_chatbot_js_does_not_contain_magic_btn(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        assert "magicBtn" not in CHATBOT_JS
        assert "magic-btn" not in CHATBOT_JS


class TestExtensionJSScmBlockOrder:
    def test_scm_handling_before_merge_check(self) -> None:
        scm_pos = _CS_EXTENSION_JS.index("fs.existsSync(sp)")
        merge_pos = _CS_EXTENSION_JS.index("fs.existsSync(mp)")
        assert scm_pos < merge_pos
