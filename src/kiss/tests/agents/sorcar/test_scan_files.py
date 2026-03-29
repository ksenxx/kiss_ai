"""Tests for _load_gitignore_dirs and _scan_files in diff_merge."""

import tempfile
import unittest
from pathlib import Path

from kiss.agents.vscode.diff_merge import _load_gitignore_dirs


class TestLoadGitignoreDirs(unittest.TestCase):
    """Test .gitignore parsing for directory skip list."""

    def test_negation_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("!keep\nfoo\n")
            result = _load_gitignore_dirs(d)
            assert "keep" not in result
            assert "foo" in result


if __name__ == "__main__":
    unittest.main()
