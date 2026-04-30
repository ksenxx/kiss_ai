"""Tests for the sorcar-docker integration.

Verifies that findKissProject() only uses the env-var and config-setting
search paths (no workspace upward search, embedded path, or common
location fallbacks).
"""

import re
import unittest
from pathlib import Path

VSCODE_SRC = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src"


class TestFindKissProjectSearchOrder(unittest.TestCase):
    """findKissProject() must only check env var and config setting."""

    def test_env_var_check_exists(self) -> None:
        """KISS_PROJECT_PATH env var must be checked."""
        source = (VSCODE_SRC / "AgentProcess.ts").read_text()
        assert re.search(
            r"process\.env\.KISS_PROJECT_PATH", source
        ), "KISS_PROJECT_PATH check not found in AgentProcess.ts"

    def test_config_setting_check_exists(self) -> None:
        """kissSorcar.kissProjectPath config setting must be checked."""
        source = (VSCODE_SRC / "AgentProcess.ts").read_text()
        assert re.search(
            r"kissProjectPath", source
        ), "kissProjectPath config check not found"

    def test_no_workspace_folder_search(self) -> None:
        """No upward search from workspace folders."""
        source = (VSCODE_SRC / "AgentProcess.ts").read_text()
        # Extract the findKissProject function body
        fn_match = re.search(
            r"export function findKissProject\(\)[^{]*\{(.+?)^}",
            source,
            re.DOTALL | re.MULTILINE,
        )
        assert fn_match is not None
        fn_body = fn_match.group(1)
        assert "workspaceFolders" not in fn_body, (
            "findKissProject() should not search workspace folders"
        )

    def test_no_embedded_path_search(self) -> None:
        """No embedded kiss_project/ fallback."""
        source = (VSCODE_SRC / "AgentProcess.ts").read_text()
        fn_match = re.search(
            r"export function findKissProject\(\)[^{]*\{(.+?)^}",
            source,
            re.DOTALL | re.MULTILINE,
        )
        assert fn_match is not None
        fn_body = fn_match.group(1)
        assert "kiss_project" not in fn_body, (
            "findKissProject() should not check embedded kiss_project/"
        )

    def test_no_common_locations_search(self) -> None:
        """No common home-directory location fallbacks."""
        source = (VSCODE_SRC / "AgentProcess.ts").read_text()
        fn_match = re.search(
            r"export function findKissProject\(\)[^{]*\{(.+?)^}",
            source,
            re.DOTALL | re.MULTILINE,
        )
        assert fn_match is not None
        fn_body = fn_match.group(1)
        for loc in ["work", "projects", "dev"]:
            assert f"'{loc}'" not in fn_body, (
                f"findKissProject() should not check common location '{loc}'"
            )

    def test_no_search_upward_function(self) -> None:
        """searchUpward function should not exist (dead code removed)."""
        source = (VSCODE_SRC / "AgentProcess.ts").read_text()
        assert "function searchUpward" not in source, (
            "searchUpward function should be removed"
        )


if __name__ == "__main__":
    unittest.main()
