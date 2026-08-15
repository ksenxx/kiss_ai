# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests verifying fixes for bugs found in worktree audit round 7.

BUG-30 FIX: _try_setup_worktree now reads current_branch(repo) inside
        repo_lock when released_branch is None, preventing races with
        concurrent tab checkouts.

BUG-31 FIX: Both merge() and _release_worktree now distinguish
        MERGE_FAILED from CONFLICT, giving correct diagnostic messages
        for commit failures (e.g. pre-commit hook rejection).

BUG-32 (obsolete): covered the per-tab merge review data directories
        (``_merge_data_dir``/``_finish_merge``), removed together with
        the interactive diff/merge review workflow.

BUG-33 FIX: copy_dirty_state now unquotes C-style quoted filenames
        from git status --porcelain, correctly handling files with
        non-ASCII characters, control chars, or special characters.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _git,
    _unquote_git_path,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _patch_super_run(
    return_value: str = "success: true\nsummary: test done\n",
) -> Any:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


class TestBug30Fix:
    """BUG-30 FIX: current_branch is now read inside repo_lock when
    released_branch is None, preventing races with concurrent checkouts.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)


    def test_correct_original_branch_recorded(self) -> None:
        """BUG-30 FIX: original_branch is correctly recorded for new worktree."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = WorktreeSorcarAgent("agent_a")
        agent._chat_id = "tab_a"

        assert GitWorktreeOps.current_branch(repo) == "main"

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None
        assert wt.original_branch == "main"

        GitWorktreeOps.remove(repo, wt.wt_dir)
        GitWorktreeOps.prune(repo)
        GitWorktreeOps.delete_branch(repo, wt.branch)

    def test_release_worktree_returns_none_for_no_prior(self) -> None:
        """BUG-30 FIX: _release_worktree returns None when no prior worktree."""
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "no_prior"
        result = agent._release_worktree()
        assert result is None


class TestBug31Fix:
    """BUG-31 FIX: MERGE_FAILED is now handled distinctly from CONFLICT
    in both merge() and _release_worktree, with correct messages.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)



    def test_merge_commit_failure_not_reported_as_conflict(self) -> None:
        """BUG-31 FIX: A commit failure is NOT called a 'conflict'."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test31"

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None

        (wt.wt_dir / "agent_file.txt").write_text("agent work\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes")

        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        result = agent.merge()

        assert "conflict" not in result.lower(), (
            f"BUG-31 NOT fixed: merge result says 'conflict': {result}"
        )
        assert "commit failed" in result.lower(), (
            f"BUG-31 NOT fixed: merge result doesn't mention commit failure: {result}"
        )
        assert "--no-verify" in result, (
            f"BUG-31 NOT fixed: no --no-verify suggestion: {result}"
        )

        hook.unlink()
        GitWorktreeOps.remove(repo, wt.wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, wt.branch):
            GitWorktreeOps.delete_branch(repo, wt.branch)

    def test_release_commit_failure_not_conflict_message(self) -> None:
        """BUG-31 FIX: _release_worktree gives correct message for MERGE_FAILED."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test31r"

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None

        (wt.wt_dir / "agent_file.txt").write_text("agent work\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes")

        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        result = agent._release_worktree()
        assert result is None, "Release should fail"

        warning = agent._merge_conflict_warning
        assert warning is not None, "Warning should be set"

        assert "had conflicts" not in warning, (
            f"BUG-31 NOT fixed: warning says 'had conflicts': {warning}"
        )
        assert "commit failed" in warning.lower(), (
            f"BUG-31 NOT fixed: warning doesn't mention commit failure: {warning}"
        )
        assert "--no-verify" in warning, (
            f"BUG-31 NOT fixed: no --no-verify suggestion: {warning}"
        )

        hook.unlink()
        if GitWorktreeOps.branch_exists(repo, wt.branch):
            GitWorktreeOps.delete_branch(repo, wt.branch)


class TestBug33Fix:
    """BUG-33 FIX: copy_dirty_state now uses _unquote_git_path to
    properly handle C-style quoted filenames from git status.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_unquote_plain_path(self) -> None:
        """Plain paths are returned unchanged."""
        assert _unquote_git_path("simple.txt") == "simple.txt"
        assert _unquote_git_path("dir/file.py") == "dir/file.py"

    def test_unquote_quoted_non_ascii(self) -> None:
        """Quoted non-ASCII path is correctly decoded."""
        quoted = '"caf\\303\\251.txt"'
        assert _unquote_git_path(quoted) == "café.txt"

    def test_unquote_escaped_chars(self) -> None:
        """Standard C escape sequences are handled."""
        assert _unquote_git_path('"hello\\nworld"') == "hello\nworld"
        assert _unquote_git_path('"tab\\there"') == "tab\there"
        assert _unquote_git_path('"back\\\\slash"') == "back\\slash"
        assert _unquote_git_path('"with\\"quote"') == 'with"quote'

    def test_unquote_multiple_octals(self) -> None:
        """Multiple octal sequences are decoded."""
        quoted = '"\\346\\227\\245.txt"'
        assert _unquote_git_path(quoted) == "日.txt"


    def test_non_ascii_filename_copied(self) -> None:
        """BUG-33 FIX: File with non-ASCII name IS copied to worktree."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        _git("config", "core.quotePath", "true", cwd=repo)

        non_ascii_name = "café.txt"
        non_ascii_file = repo / non_ascii_name
        non_ascii_file.write_text("content of café\n")

        status = _git("status", "--porcelain", "-uall", cwd=repo)
        assert status.stdout.strip(), "sanity: git status shows the new file"

        wt_dir = Path(self._tmpdir) / "wt"
        wt_dir.mkdir()
        _git("worktree", "add", "-b", "test-wt", str(wt_dir), cwd=repo)

        result = GitWorktreeOps.copy_dirty_state(repo, wt_dir)
        assert result is True, "Should return True (dirty state was copied)"

        wt_file = wt_dir / non_ascii_name
        assert wt_file.exists(), (
            "BUG-33 NOT fixed: non-ASCII file was NOT copied to worktree"
        )
        assert wt_file.read_text() == "content of café\n"

        _git("worktree", "remove", str(wt_dir), "--force", cwd=repo)
        _git("worktree", "prune", cwd=repo)
        _git("branch", "-D", "test-wt", cwd=repo)

    def test_quoted_filename_parse_correct(self) -> None:
        """BUG-33 FIX: The parsing logic now correctly handles quoted names."""
        quoted_line = '?? "caf\\303\\251.txt"'
        raw_fname = quoted_line[3:]

        fname = _unquote_git_path(raw_fname)
        assert fname == "café.txt", (
            f"BUG-33 NOT fixed: unquoted name is {fname!r}, expected 'café.txt'"
        )
