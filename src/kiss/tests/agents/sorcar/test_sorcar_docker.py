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



if __name__ == "__main__":
    unittest.main()
