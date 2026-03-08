"""Tests for _disable_copilot_scm_button and its integration with _install_copilot_extension."""

import json
from pathlib import Path

from kiss.agents.sorcar.code_server import (
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
