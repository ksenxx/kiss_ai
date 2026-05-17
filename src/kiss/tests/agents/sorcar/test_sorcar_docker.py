"""Tests for findKissProject() search order.

Verifies that findKissProject() — now in kissPaths.ts after the
single-daemon architecture lifted these helpers out of the deleted
AgentProcess.ts — uses the env-var, config-setting, and embedded
kiss_project search paths (no workspace upward search or common
location fallbacks).
"""

import re
import unittest
from pathlib import Path

VSCODE_SRC = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src"
KISS_PATHS = VSCODE_SRC / "kissPaths.ts"


def _find_kiss_project_body() -> str:
    source = KISS_PATHS.read_text()
    fn_match = re.search(
        r"export function findKissProject\(\)[^{]*\{(.+?)^}",
        source,
        re.DOTALL | re.MULTILINE,
    )
    assert fn_match is not None, "findKissProject() not found in kissPaths.ts"
    return fn_match.group(1)


class TestFindKissProjectSearchOrder(unittest.TestCase):
    """findKissProject() must only check env var, config setting, and embedded path."""

    def test_env_var_check_exists(self) -> None:
        """KISS_PROJECT_PATH env var must be checked."""
        source = KISS_PATHS.read_text()
        assert re.search(
            r"process\.env\.KISS_PROJECT_PATH", source
        ), "KISS_PROJECT_PATH check not found in kissPaths.ts"

    def test_config_setting_check_exists(self) -> None:
        """kissSorcar.kissProjectPath config setting must be checked."""
        source = KISS_PATHS.read_text()
        assert re.search(
            r"kissProjectPath", source
        ), "kissProjectPath config check not found"

    def test_no_workspace_folder_search(self) -> None:
        """No upward search from workspace folders."""
        body = _find_kiss_project_body()
        assert "workspaceFolders" not in body, (
            "findKissProject() should not search workspace folders"
        )

    def test_embedded_path_search_exists(self) -> None:
        """Embedded kiss_project/ bundled with the extension is checked."""
        body = _find_kiss_project_body()
        assert "kiss_project" in body, (
            "findKissProject() should check embedded kiss_project/"
        )

    def test_no_common_locations_search(self) -> None:
        """No common home-directory location fallbacks."""
        body = _find_kiss_project_body()
        for loc in ["work", "projects", "dev"]:
            assert f"'{loc}'" not in body, (
                f"findKissProject() should not check common location '{loc}'"
            )

    def test_no_search_upward_function(self) -> None:
        """searchUpward function should not exist (dead code removed)."""
        source = KISS_PATHS.read_text()
        assert "function searchUpward" not in source, (
            "searchUpward function should be removed"
        )


if __name__ == "__main__":
    unittest.main()
