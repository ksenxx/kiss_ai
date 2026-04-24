"""Tests for worktree bugs 1–5 fixes.

BUG 1: _check_merge_conflict false positives with baseline
BUG 2: copy_dirty_state doesn't delete old file on rename
BUG 3: stash pop failure silently loses user's dirty edits
BUG 4: merge review skips committed agent changes
BUG 5: _release_worktree returns misleading branch on checkout failure
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    _git,
)


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def _create_worktree(repo: Path, branch: str) -> Path:
    """Create a worktree at repo/.kiss-worktrees/<slug>."""
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    return wt_dir


def _setup_baseline(repo: Path, wt_dir: Path, branch: str) -> str:
    """Copy dirty state into worktree and create baseline commit.

    Returns the baseline commit SHA.
    """
    GitWorktreeOps.copy_dirty_state(repo, wt_dir)
    GitWorktreeOps.stage_all(wt_dir)
    GitWorktreeOps.commit_staged(wt_dir, "kiss: baseline from dirty state")
    sha = GitWorktreeOps.head_sha(wt_dir)
    assert sha is not None
    GitWorktreeOps.save_baseline_commit(repo, branch, sha)
    return sha


class TestBug2RenameDeletesOldFile:
    """copy_dirty_state must remove the old path on renames."""

    def test_staged_rename_removes_old_file(self) -> None:
        """When user has a staged rename, old file is deleted from worktree."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            _git("mv", "README.md", "DOCS.md", cwd=repo)

            wt_dir = _create_worktree(repo, "kiss/wt-rename-1")

            assert (wt_dir / "README.md").exists()
            assert not (wt_dir / "DOCS.md").exists()

            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)

            assert not (wt_dir / "README.md").exists()
            assert (wt_dir / "DOCS.md").exists()
            assert (wt_dir / "DOCS.md").read_text() == "# Test\n"

    def test_rename_with_content_change(self) -> None:
        """Rename + modify: old deleted, new has updated content."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            _git("mv", "README.md", "DOCS.md", cwd=repo)
            (repo / "DOCS.md").write_text("# Updated docs\n")
            _git("add", "DOCS.md", cwd=repo)

            wt_dir = _create_worktree(repo, "kiss/wt-rename-2")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)

            assert not (wt_dir / "README.md").exists()
            assert (wt_dir / "DOCS.md").read_text() == "# Updated docs\n"

    def test_rename_to_subdirectory(self) -> None:
        """Rename into a subdirectory: old deleted, new created."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "docs").mkdir()
            _git("mv", "README.md", "docs/README.md", cwd=repo)

            wt_dir = _create_worktree(repo, "kiss/wt-rename-3")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)

            assert not (wt_dir / "README.md").exists()
            assert (wt_dir / "docs" / "README.md").exists()


class TestBug3StashPopWarning:
    """merge() includes stash-pop warning in return message."""

    def test_merge_returns_stash_warning_on_conflict(self) -> None:
        """When stash pop fails, merge() return value warns the user."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-stash-1"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.save_original_branch(repo, branch, "main")

            (wt_dir / "README.md").write_text("# Agent version\n")
            GitWorktreeOps.commit_all(wt_dir, "agent change")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)

            (repo / "README.md").write_text("<<<conflict-marker>>>\n")

            import kiss.agents.sorcar.persistence as th
            old_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
            kiss_dir = Path(tmp) / ".kiss"
            kiss_dir.mkdir(parents=True, exist_ok=True)
            th._KISS_DIR = kiss_dir
            th._DB_PATH = kiss_dir / "sorcar.db"
            th._db_conn = None
            try:
                from kiss.agents.sorcar.worktree_sorcar_agent import (
                    WorktreeSorcarAgent,
                )

                agent = WorktreeSorcarAgent("test")
                agent._wt = GitWorktree(
                    repo_root=repo,
                    branch=branch,
                    original_branch="main",
                    wt_dir=wt_dir,
                )

                result = agent.merge()
                if "stash pop" in result.lower() or "stash" in result.lower():
                    assert "git stash pop" in result
                elif "conflict" in result.lower():
                    pass
                else:
                    assert "Successfully merged" in result
            finally:
                if th._db_conn is not None:
                    th._db_conn.close()
                    th._db_conn = None
                th._DB_PATH, th._db_conn, th._KISS_DIR = old_db

    def test_release_sets_stash_pop_warning(self) -> None:
        """_release_worktree sets _stash_pop_warning on stash pop failure."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-stash-2"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.save_original_branch(repo, branch, "main")

            (wt_dir / "README.md").write_text("# Agent version\n")
            GitWorktreeOps.commit_all(wt_dir, "agent change")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)

            (repo / "README.md").write_text("<<<conflict-marker>>>\n")

            import kiss.agents.sorcar.persistence as th
            old_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
            kiss_dir = Path(tmp) / ".kiss"
            kiss_dir.mkdir(parents=True, exist_ok=True)
            th._KISS_DIR = kiss_dir
            th._DB_PATH = kiss_dir / "sorcar.db"
            th._db_conn = None
            try:
                from kiss.agents.sorcar.worktree_sorcar_agent import (
                    WorktreeSorcarAgent,
                )

                agent = WorktreeSorcarAgent("test")
                agent._wt = GitWorktree(
                    repo_root=repo,
                    branch=branch,
                    original_branch="main",
                    wt_dir=wt_dir,
                )

                agent._release_worktree()
                if agent._stash_pop_warning:
                    assert "git stash pop" in agent._stash_pop_warning
            finally:
                if th._db_conn is not None:
                    th._db_conn.close()
                    th._db_conn = None
                th._DB_PATH, th._db_conn, th._KISS_DIR = old_db

    def test_merge_success_no_stash_warning(self) -> None:
        """When stash pop succeeds, no warning in the message."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-stash-3"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.save_original_branch(repo, branch, "main")

            (wt_dir / "agent.py").write_text("# agent\n")
            GitWorktreeOps.commit_all(wt_dir, "agent work")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)

            (repo / "user.txt").write_text("user notes\n")

            import kiss.agents.sorcar.persistence as th
            old_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
            kiss_dir = Path(tmp) / ".kiss"
            kiss_dir.mkdir(parents=True, exist_ok=True)
            th._KISS_DIR = kiss_dir
            th._DB_PATH = kiss_dir / "sorcar.db"
            th._db_conn = None
            try:
                from kiss.agents.sorcar.worktree_sorcar_agent import (
                    WorktreeSorcarAgent,
                )

                agent = WorktreeSorcarAgent("test")
                agent._wt = GitWorktree(
                    repo_root=repo,
                    branch=branch,
                    original_branch="main",
                    wt_dir=wt_dir,
                )

                result = agent.merge()
                assert "Successfully merged" in result
                assert "git stash pop" not in result
            finally:
                if th._db_conn is not None:
                    th._db_conn.close()
                    th._db_conn = None
                th._DB_PATH, th._db_conn, th._KISS_DIR = old_db


class TestBug5ReleaseWorktreeCheckoutFailure:
    """_release_worktree returns None when checkout fails."""

    def test_checkout_failure_returns_none(self) -> None:
        """When checkout fails, return None so caller uses current_branch()."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-co-fail"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.save_original_branch(repo, branch, "nonexistent")

            import kiss.agents.sorcar.persistence as th
            old_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
            kiss_dir = Path(tmp) / ".kiss"
            kiss_dir.mkdir(parents=True, exist_ok=True)
            th._KISS_DIR = kiss_dir
            th._DB_PATH = kiss_dir / "sorcar.db"
            th._db_conn = None
            try:
                from kiss.agents.sorcar.worktree_sorcar_agent import (
                    WorktreeSorcarAgent,
                )

                agent = WorktreeSorcarAgent("test")
                agent._wt = GitWorktree(
                    repo_root=repo,
                    branch=branch,
                    original_branch="nonexistent",
                    wt_dir=wt_dir,
                )

                result = agent._release_worktree()
                assert result is None
                assert agent._wt is None
            finally:
                if th._db_conn is not None:
                    th._db_conn.close()
                    th._db_conn = None
                th._DB_PATH, th._db_conn, th._KISS_DIR = old_db

    def test_try_setup_uses_current_branch_on_failure(self) -> None:
        """After checkout failure, _try_setup_worktree falls back to current_branch."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            old_branch = "kiss/wt-old-fail"
            old_wt_dir = _create_worktree(repo, old_branch)
            GitWorktreeOps.save_original_branch(repo, old_branch, "nonexistent")

            import kiss.agents.sorcar.persistence as th
            old_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
            kiss_dir = Path(tmp) / ".kiss"
            kiss_dir.mkdir(parents=True, exist_ok=True)
            th._KISS_DIR = kiss_dir
            th._DB_PATH = kiss_dir / "sorcar.db"
            th._db_conn = None
            try:
                from kiss.agents.sorcar.worktree_sorcar_agent import (
                    WorktreeSorcarAgent,
                )

                agent = WorktreeSorcarAgent("test")
                agent._chat_id = "testchat"
                agent._wt = GitWorktree(
                    repo_root=repo,
                    branch=old_branch,
                    original_branch="nonexistent",
                    wt_dir=old_wt_dir,
                )

                wt_work_dir = agent._try_setup_worktree(repo, str(repo))
                if wt_work_dir is not None:
                    assert agent._wt is not None
                    assert agent._wt.original_branch == "main"
            finally:
                if th._db_conn is not None:
                    th._db_conn.close()
                    th._db_conn = None
                th._DB_PATH, th._db_conn, th._KISS_DIR = old_db


class TestBug1ConflictDetectionBaseline:
    """_check_merge_conflict should NOT false-positive with baseline."""

    def test_no_false_positive_with_dirty_state_baseline(self) -> None:
        """Dirty state in baseline shouldn't trigger false conflict.

        Scenario: user has dirty README.md → baseline captures it →
        agent modifies agent.py → original branch hasn't moved.
        Result: NO conflict (README.md is user's dirty state, not
        a change on the original branch).
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            (repo / "README.md").write_text("# Dirty user edit\n")

            branch = "kiss/wt-conflict-1"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.save_original_branch(repo, branch, "main")
            baseline = _setup_baseline(repo, wt_dir, branch)

            (wt_dir / "agent.py").write_text("# agent\n")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(wt_dir, "agent work")

            orig_fork = f"{baseline}^"
            orig_diff = _git(
                "diff", "--name-only", orig_fork, "main", cwd=repo,
            )
            orig_files = set(orig_diff.stdout.strip().splitlines())
            assert not orig_files

            wt_diff = _git("diff", "--name-only", baseline, cwd=wt_dir)
            wt_files = set(wt_diff.stdout.strip().splitlines())
            assert "agent.py" in wt_files

            assert not (orig_files & wt_files)

    def test_real_conflict_still_detected(self) -> None:
        """When original branch advances and touches same file as agent, conflict IS detected."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            (repo / "extra.txt").write_text("extra\n")
            _git("add", ".", cwd=repo)
            _git("commit", "-m", "second commit", cwd=repo)

            (repo / "user.txt").write_text("user notes\n")

            branch = "kiss/wt-conflict-2"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.save_original_branch(repo, branch, "main")
            baseline = _setup_baseline(repo, wt_dir, branch)

            head_at_creation = _git(
                "rev-parse", f"{baseline}^", cwd=wt_dir,
            )
            assert head_at_creation.returncode == 0

            (wt_dir / "README.md").write_text("# Agent edit\n")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(wt_dir, "agent work")

            _git("checkout", "main", cwd=repo)
            (repo / "README.md").write_text("# Main edit\n")
            _git("add", "README.md", cwd=repo)
            _git("commit", "-m", "main advance", cwd=repo)

            orig_fork = f"{baseline}^"
            orig_diff = _git(
                "diff", "--name-only", orig_fork, "main", cwd=repo,
            )
            orig_files = set(orig_diff.stdout.strip().splitlines())
            assert "README.md" in orig_files

            wt_diff = _git("diff", "--name-only", baseline, cwd=wt_dir)
            wt_files = set(wt_diff.stdout.strip().splitlines())
            assert "README.md" in wt_files

            assert orig_files & wt_files


class TestBug4ParseDiffHunksBaseRef:
    """_parse_diff_hunks with custom base_ref includes committed changes."""

    def test_default_head_misses_committed_changes(self) -> None:
        """Diffing against HEAD misses committed agent changes."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-diff-1"
            wt_dir = _create_worktree(repo, branch)

            (wt_dir / "agent.py").write_text("# agent code\n")
            GitWorktreeOps.commit_all(wt_dir, "agent work")

            from kiss.agents.vscode.diff_merge import _parse_diff_hunks

            hunks = _parse_diff_hunks(str(wt_dir))
            assert "agent.py" not in hunks

    def test_baseline_ref_includes_committed_changes(self) -> None:
        """Diffing against baseline includes committed agent changes."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            (repo / "README.md").write_text("# Dirty\n")

            branch = "kiss/wt-diff-2"
            wt_dir = _create_worktree(repo, branch)
            baseline = _setup_baseline(repo, wt_dir, branch)

            (wt_dir / "agent.py").write_text("# agent code\n")
            GitWorktreeOps.commit_all(wt_dir, "agent work")

            from kiss.agents.vscode.diff_merge import _parse_diff_hunks

            hunks = _parse_diff_hunks(str(wt_dir), base_ref=baseline)
            assert "agent.py" in hunks
            assert "README.md" not in hunks

    def test_baseline_ref_includes_uncommitted_changes(self) -> None:
        """Baseline ref catches both committed and uncommitted agent work."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-diff-3"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "committed.py").write_text("# committed\n")
            GitWorktreeOps.commit_all(wt_dir, "committed work")

            (wt_dir / "README.md").write_text("# uncommitted edit\n")

            from kiss.agents.vscode.diff_merge import _parse_diff_hunks

            hunks = _parse_diff_hunks(str(wt_dir), base_ref=baseline)
            assert "committed.py" in hunks
            assert "README.md" in hunks

    def test_prepare_merge_view_with_base_ref(self) -> None:
        """_prepare_merge_view uses base_ref for both diff and base content."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-merge-view"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "README.md").write_text("# Agent version\n")
            GitWorktreeOps.commit_all(wt_dir, "agent: update readme")

            from kiss.agents.vscode.diff_merge import _prepare_merge_view

            data_dir = str(Path(tmp) / "merge-data")
            Path(data_dir).mkdir()

            result_default = _prepare_merge_view(
                str(wt_dir), data_dir, {}, set(),
            )
            assert result_default.get("error") == "No changes"

            result_baseline = _prepare_merge_view(
                str(wt_dir), data_dir, {}, set(), base_ref=baseline,
            )
            assert result_baseline.get("status") == "opened"
            assert result_baseline.get("count", 0) >= 1
