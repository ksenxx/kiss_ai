"""Tests for _load_gitignore_dirs and _scan_files in diff_merge."""

import tempfile
import unittest
from pathlib import Path

from kiss.agents.vscode.diff_merge import _load_gitignore_dirs, _scan_files


class TestLoadGitignoreDirs(unittest.TestCase):
    """Test .gitignore parsing for directory skip list."""

    def test_always_includes_git(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            assert ".git" in _load_gitignore_dirs(d)

    def test_no_gitignore_returns_only_git(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            assert _load_gitignore_dirs(d) == {".git"}

    def test_simple_names(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("node_modules\n__pycache__\n.venv\n")
            result = _load_gitignore_dirs(d)
            assert "node_modules" in result
            assert "__pycache__" in result
            assert ".venv" in result

    def test_trailing_slash_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("build/\ndist/\n")
            result = _load_gitignore_dirs(d)
            assert "build" in result
            assert "dist" in result

    def test_comments_blank_lines_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("# comment\n\n  \nfoo\n")
            result = _load_gitignore_dirs(d)
            assert result == {".git", "foo"}

    def test_negation_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("!keep\nfoo\n")
            result = _load_gitignore_dirs(d)
            assert "keep" not in result
            assert "foo" in result

    def test_globs_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("*.pyc\n?.log\nfoo\n")
            result = _load_gitignore_dirs(d)
            assert result == {".git", "foo"}

    def test_paths_included(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("src/foo\na/b/\nbar\n")
            result = _load_gitignore_dirs(d)
            assert result == {".git", "src/foo", "a/b", "bar"}

    def test_whitespace_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("  venv  \n")
            result = _load_gitignore_dirs(d)
            assert "venv" in result


class TestScanFilesGitignore(unittest.TestCase):
    """Test that _scan_files respects .gitignore for skipping directories."""

    def test_skips_gitignored_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("skipme\n")
            (Path(d) / "skipme").mkdir()
            (Path(d) / "skipme" / "hidden.txt").write_text("x")
            (Path(d) / "keepme").mkdir()
            (Path(d) / "keepme" / "visible.txt").write_text("x")
            (Path(d) / "root.txt").write_text("x")
            result = _scan_files(d)
            assert "keepme/" in result
            assert "keepme/visible.txt" in result
            assert "root.txt" in result
            assert not any("skipme" in p for p in result)

    def test_skips_gitignored_paths(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".gitignore").write_text("src/gen\n")
            (Path(d) / "src" / "gen").mkdir(parents=True)
            (Path(d) / "src" / "gen" / "out.txt").write_text("x")
            (Path(d) / "src" / "keep").mkdir(parents=True)
            (Path(d) / "src" / "keep" / "ok.txt").write_text("x")
            result = _scan_files(d)
            assert "src/keep/ok.txt" in result
            assert not any("gen" in p for p in result)

    def test_no_gitignore_still_skips_dotgit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".git").mkdir()
            (Path(d) / ".git" / "config").write_text("x")
            (Path(d) / "file.txt").write_text("x")
            result = _scan_files(d)
            assert "file.txt" in result
            assert not any(".git" in p for p in result)


if __name__ == "__main__":
    unittest.main()
