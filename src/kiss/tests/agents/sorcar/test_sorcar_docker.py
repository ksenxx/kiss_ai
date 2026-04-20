"""Tests for the sorcar-docker integration.

Verifies that when KISS_PROJECT_PATH is set (as in the Docker container),
the VS Code extension's findKissProject() returns that path instead of
the embedded kiss_project/ directory inside the extension bundle.

The embedded kiss_project/ is a standalone copy of the source tree without
a .venv — using it in Docker (where a fully-set-up /home/coder/kiss
exists) causes the agent process to fail because uv would need to
recreate the entire environment from scratch.
"""

import re
import unittest
from pathlib import Path

VSCODE_SRC = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src"


class TestFindKissProjectSearchOrder(unittest.TestCase):
    """The env-var and config checks must come before the embedded path check.

    In Docker, KISS_PROJECT_PATH=/home/coder/kiss points to a fully
    set-up project with a .venv.  The embedded kiss_project/ inside the
    extension directory has no .venv and will cause uv to hang or fail.
    """

    def test_env_var_checked_before_embedded_path(self) -> None:
        """KISS_PROJECT_PATH must be checked before the embedded kiss_project/."""
        source = (VSCODE_SRC / "AgentProcess.ts").read_text()

        # Find the position of the KISS_PROJECT_PATH env-var check
        env_match = re.search(r"process\.env\.KISS_PROJECT_PATH", source)
        assert env_match is not None, "KISS_PROJECT_PATH check not found in AgentProcess.ts"

        # Find the embedded path check (the actual path.join call, not comments)
        embedded_match = re.search(r"path\.join\(__dirname.*kiss_project", source)
        assert embedded_match is not None, "embedded kiss_project check not found"

        assert env_match.start() < embedded_match.start(), (
            "KISS_PROJECT_PATH env-var check must appear before the embedded "
            "kiss_project/ check in findKissProject() so that Docker containers "
            "with KISS_PROJECT_PATH set use the fully-configured project path"
        )

    def test_config_setting_checked_before_embedded_path(self) -> None:
        """kissSorcar.kissProjectPath config must be checked before embedded."""
        source = (VSCODE_SRC / "AgentProcess.ts").read_text()

        config_match = re.search(r"kissProjectPath", source)
        assert config_match is not None

        # Find the embedded path check (the actual path.join call, not the comment)
        embedded_match = re.search(r"path\.join\(__dirname.*kiss_project", source)
        assert embedded_match is not None

        assert config_match.start() < embedded_match.start(), (
            "Configuration setting must be checked before embedded kiss_project/"
        )


if __name__ == "__main__":
    unittest.main()
