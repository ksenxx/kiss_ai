"""Tests for _disable_copilot_scm_button and its integration with _install_copilot_extension."""

import json
from pathlib import Path

from kiss.agents.sorcar.code_server import (
    _disable_copilot_scm_button,
    _install_copilot_extension,
)


def _make_copilot_chat_pkg(ext_dir: Path, when_clause: str = "scmProvider == git") -> Path:
    """Create a fake github.copilot-chat extension with an scm/inputBox entry."""
    chat_dir = ext_dir / "github.copilot-chat-0.36.2"
    chat_dir.mkdir(parents=True)
    pkg = {
        "name": "copilot-chat",
        "contributes": {
            "menus": {
                "scm/inputBox": [
                    {
                        "command": "github.copilot.git.generateCommitMessage",
                        "when": when_clause,
                    }
                ]
            }
        },
    }
    pkg_path = chat_dir / "package.json"
    pkg_path.write_text(json.dumps(pkg))
    return pkg_path


class TestDisableCopilotScmButton:
    def test_sets_when_to_false(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        ext_dir = data_dir / "extensions"
        pkg_path = _make_copilot_chat_pkg(ext_dir)

        _disable_copilot_scm_button(str(data_dir))

        pkg = json.loads(pkg_path.read_text())
        entry = pkg["contributes"]["menus"]["scm/inputBox"][0]
        assert entry["when"] == "false"

    def test_already_false_no_rewrite(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        ext_dir = data_dir / "extensions"
        pkg_path = _make_copilot_chat_pkg(ext_dir, when_clause="false")
        original_text = pkg_path.read_text()

        _disable_copilot_scm_button(str(data_dir))

        # File should not be rewritten when already "false"
        assert pkg_path.read_text() == original_text

    def test_no_extensions_dir(self, tmp_path: Path) -> None:
        """No crash when extensions dir doesn't exist."""
        _disable_copilot_scm_button(str(tmp_path / "nonexistent"))

    def test_no_copilot_chat_extension(self, tmp_path: Path) -> None:
        """No crash when copilot-chat is not installed."""
        ext_dir = tmp_path / "data" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "some-other-ext").mkdir()
        _disable_copilot_scm_button(str(tmp_path / "data"))

    def test_ignores_non_chat_copilot(self, tmp_path: Path) -> None:
        """Only modifies github.copilot-chat-, not github.copilot-."""
        data_dir = tmp_path / "data"
        ext_dir = data_dir / "extensions"
        copilot_dir = ext_dir / "github.copilot-1.388.0"
        copilot_dir.mkdir(parents=True)
        pkg = {"name": "copilot", "contributes": {"menus": {"view/title": []}}}
        pkg_path = copilot_dir / "package.json"
        pkg_path.write_text(json.dumps(pkg))
        original = pkg_path.read_text()

        _disable_copilot_scm_button(str(data_dir))

        assert pkg_path.read_text() == original

    def test_multiple_scm_entries(self, tmp_path: Path) -> None:
        """Handles multiple entries in scm/inputBox, only disables copilot's."""
        data_dir = tmp_path / "data"
        chat_dir = data_dir / "extensions" / "github.copilot-chat-0.36.2"
        chat_dir.mkdir(parents=True)
        pkg = {
            "contributes": {
                "menus": {
                    "scm/inputBox": [
                        {
                            "command": "github.copilot.git.generateCommitMessage",
                            "when": "scmProvider == git",
                        },
                        {
                            "command": "some.other.command",
                            "when": "scmProvider == git",
                        },
                    ]
                }
            }
        }
        pkg_path = chat_dir / "package.json"
        pkg_path.write_text(json.dumps(pkg))

        _disable_copilot_scm_button(str(data_dir))

        result = json.loads(pkg_path.read_text())
        entries = result["contributes"]["menus"]["scm/inputBox"]
        assert entries[0]["when"] == "false"
        assert entries[1]["when"] == "scmProvider == git"

    def test_no_scm_inputbox_key(self, tmp_path: Path) -> None:
        """No crash when copilot-chat has no scm/inputBox menu."""
        data_dir = tmp_path / "data"
        chat_dir = data_dir / "extensions" / "github.copilot-chat-0.36.2"
        chat_dir.mkdir(parents=True)
        pkg: dict[str, object] = {"contributes": {"menus": {}}}
        (chat_dir / "package.json").write_text(json.dumps(pkg))

        _disable_copilot_scm_button(str(data_dir))  # should not crash

    def test_malformed_json(self, tmp_path: Path) -> None:
        """No crash on malformed package.json."""
        data_dir = tmp_path / "data"
        chat_dir = data_dir / "extensions" / "github.copilot-chat-0.36.2"
        chat_dir.mkdir(parents=True)
        (chat_dir / "package.json").write_text("not valid json{{{")

        _disable_copilot_scm_button(str(data_dir))  # should not crash


class TestInstallCopilotCallsDisable:
    def test_source_code_calls_disable_after_subprocess(self) -> None:
        """Verify _install_copilot_extension calls _disable_copilot_scm_button
        after the subprocess.run (installation), not before."""
        import inspect

        source = inspect.getsource(_install_copilot_extension)
        assert "_disable_copilot_scm_button" in source
        # The call should be after subprocess.run, not before
        idx_subprocess = source.index("subprocess.run")
        idx_disable = source.index("_disable_copilot_scm_button")
        assert idx_disable > idx_subprocess
