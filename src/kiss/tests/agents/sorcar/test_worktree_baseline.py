"""Tests for baseline commit functionality in worktree mode.

When the user has uncommitted/staged/untracked files, the worktree
captures them in a "baseline commit" so that merge, merge-review,
and changed-file detection only see agent-produced changes.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
    _git,
)


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
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


def _create_worktree(repo: Path, branch: str) -> Path:
    """Create a worktree at repo/.kiss-worktrees/<slug>."""
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    return wt_dir


class TestGitWorktreeBaseline:
    """GitWorktree dataclass includes baseline_commit field."""

    def test_default_none(self) -> None:
        wt = GitWorktree(
            repo_root=Path("/tmp"),
            branch="b",
            original_branch="main",
            wt_dir=Path("/tmp/wt"),
        )
        assert wt.baseline_commit is None

    def test_with_baseline(self) -> None:
        wt = GitWorktree(
            repo_root=Path("/tmp"),
            branch="b",
            original_branch="main",
            wt_dir=Path("/tmp/wt"),
            baseline_commit="abc123",
        )
        assert wt.baseline_commit == "abc123"


class TestBaselineCommitConfig:
    """Save and load baseline commit SHA via git config."""

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-test-123"
            _git("branch", branch, cwd=repo)

            assert GitWorktreeOps.save_baseline_commit(repo, branch, "deadbeef")
            loaded = GitWorktreeOps.load_baseline_commit(repo, branch)
            assert loaded == "deadbeef"

    def test_load_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            result = GitWorktreeOps.load_baseline_commit(repo, "no/such-branch")
            assert result is None


class TestCopyDirtyState:
    """copy_dirty_state mirrors dirty files from main worktree."""

    def test_clean_repo_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            wt_dir = _create_worktree(repo, "kiss/wt-test-1")
            assert not GitWorktreeOps.copy_dirty_state(repo, wt_dir)

    def test_unstaged_modification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Modified\n")
            wt_dir = _create_worktree(repo, "kiss/wt-test-2")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "README.md").read_text() == "# Modified\n"

    def test_staged_modification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Staged\n")
            _git("add", "README.md", cwd=repo)
            wt_dir = _create_worktree(repo, "kiss/wt-test-3")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "README.md").read_text() == "# Staged\n"

    def test_untracked_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "new_file.txt").write_text("new content\n")
            wt_dir = _create_worktree(repo, "kiss/wt-test-4")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "new_file.txt").read_text() == "new content\n"

    def test_deleted_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").unlink()
            wt_dir = _create_worktree(repo, "kiss/wt-test-5")
            assert (wt_dir / "README.md").exists()
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert not (wt_dir / "README.md").exists()

    def test_new_file_in_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "src").mkdir()
            (repo / "src" / "app.py").write_text("print('hi')\n")
            wt_dir = _create_worktree(repo, "kiss/wt-test-6")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "src" / "app.py").read_text() == "print('hi')\n"

    def test_multiple_dirty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Changed\n")
            (repo / "new.txt").write_text("new\n")
            _git("add", "README.md", cwd=repo)
            wt_dir = _create_worktree(repo, "kiss/wt-test-7")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "README.md").read_text() == "# Changed\n"
            assert (wt_dir / "new.txt").read_text() == "new\n"


class TestHeadSha:
    """head_sha returns the current HEAD commit SHA."""

    def test_returns_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            sha = GitWorktreeOps.head_sha(repo)
            assert sha is not None
            assert len(sha) == 40

    def test_bad_dir_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sha = GitWorktreeOps.head_sha(Path(tmp))
            assert sha is None


class TestSquashMergeFromBaseline:
    """squash_merge_from_baseline merges only agent changes."""

    def test_no_agent_changes(self) -> None:
        """When baseline == branch tip, nothing to merge → SUCCESS."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-test-8"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            _git("checkout", "main", cwd=repo)
            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.SUCCESS

    def test_only_agent_changes_merged(self) -> None:
        """Baseline (dirty state) is excluded; only agent work merged."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            (repo / "README.md").write_text("# User edit\n")
            (repo / "user_file.txt").write_text("user content\n")

            branch = "kiss/wt-test-9"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "agent_file.py").write_text("# agent code\n")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(wt_dir, "agent: add file")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.SUCCESS

            assert (repo / "agent_file.py").read_text() == "# agent code\n"
            committed_readme = _git(
                "show", "HEAD:README.md", cwd=repo,
            )
            assert committed_readme.stdout == "# Test\n"
            committed_user = _git(
                "show", "HEAD:user_file.txt", cwd=repo,
            )
            assert committed_user.returncode != 0
            assert (repo / "README.md").read_text() == "# User edit\n"
            assert (repo / "user_file.txt").read_text() == "user content\n"

    def test_conflict_resets_cleanly(self) -> None:
        """On conflict, resets to clean state and returns CONFLICT."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-test-10"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "README.md").write_text("# Agent version\n")
            GitWorktreeOps.commit_all(wt_dir, "agent change")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)
            (repo / "README.md").write_text("# Conflicting main version\n")
            _git("add", "README.md", cwd=repo)
            _git("commit", "-m", "main change", cwd=repo)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.CONFLICT

            status = _git("status", "--porcelain", cwd=repo)
            assert not status.stdout.strip()


class TestEndToEndBaselineWorkflow:
    """Full workflow: dirty main tree → worktree → agent edits → merge."""

    def test_full_cycle_dirty_state_excluded_from_merge(self) -> None:
        """User has dirty files, agent works in worktree, merge only gets agent changes."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            original = "line1\nline2\nline3\nline4\nline5\nline6\n"
            (repo / "code.py").write_text(original)
            _git("add", ".", cwd=repo)
            _git("commit", "-m", "add code.py", cwd=repo)

            dirty = "line1-dirty\nline2\nline3\nline4\nline5\nline6\n"
            (repo / "code.py").write_text(dirty)
            (repo / "notes.txt").write_text("user notes\n")

            branch = "kiss/wt-full-cycle"
            wt_dir = _create_worktree(repo, branch)
            original_branch = "main"
            GitWorktreeOps.save_original_branch(repo, branch, original_branch)

            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            GitWorktreeOps.save_baseline_commit(repo, branch, baseline)

            assert (wt_dir / "code.py").read_text() == dirty
            assert (wt_dir / "notes.txt").read_text() == "user notes\n"

            agent_version = dirty + "agent-added\n"
            (wt_dir / "code.py").write_text(agent_version)
            (wt_dir / "fix.py").write_text("def fix(): pass\n")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(wt_dir, "agent: implement fix")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)

            did_stash = GitWorktreeOps.stash_if_dirty(repo)
            assert did_stash

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.SUCCESS

            assert GitWorktreeOps.stash_pop(repo)

            assert (repo / "fix.py").read_text() == "def fix(): pass\n"

            committed = _git("show", "HEAD:code.py", cwd=repo)
            assert "agent-added" in committed.stdout
            assert "line1-dirty" not in committed.stdout
            assert committed.stdout.startswith("line1\n")

            committed_notes = _git("show", "HEAD:notes.txt", cwd=repo)
            assert committed_notes.returncode != 0

            assert (repo / "notes.txt").read_text() == "user notes\n"
            assert "line1-dirty" in (repo / "code.py").read_text()

    def test_backward_compat_no_baseline(self) -> None:
        """Legacy worktrees (no baseline) still use squash_merge_branch."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-legacy"
            wt_dir = _create_worktree(repo, branch)

            (wt_dir / "agent.py").write_text("# agent\n")
            GitWorktreeOps.commit_all(wt_dir, "agent work")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)

            result = GitWorktreeOps.squash_merge_branch(repo, branch)
            assert result == MergeResult.SUCCESS
            assert (repo / "agent.py").read_text() == "# agent\n"

    def test_changed_files_against_baseline(self) -> None:
        """git diff --name-only against baseline shows only agent files."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Dirty\n")
            (repo / "user.txt").write_text("user\n")

            branch = "kiss/wt-changed"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "agent.py").write_text("# a\n")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(wt_dir, "agent work")

            diff = _git(
                "diff", "--name-only", baseline, "HEAD", cwd=wt_dir,
            )
            changed = diff.stdout.strip().splitlines()
            assert changed == ["agent.py"]


    def test_uncommitted_agent_changes_visible(self) -> None:
        """Uncommitted agent changes visible via git diff against baseline."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Dirty\n")

            branch = "kiss/wt-uncommitted"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "fix.py").write_text("# fix\n")

            diff = _git(
                "diff", "--name-only", baseline, cwd=wt_dir,
            )
            changed = diff.stdout.strip().splitlines()
            untracked = _git(
                "ls-files", "--others", "--exclude-standard", cwd=wt_dir,
            )
            all_changed = set(changed) | set(
                untracked.stdout.strip().splitlines()
            )
            assert "fix.py" in all_changed
            assert "README.md" not in all_changed


class TestRestoreFromGitWithBaseline:
    """_restore_from_git loads baseline_commit from git config."""

    def test_restore_with_baseline(self) -> None:
        import kiss.agents.sorcar.persistence as th

        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

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
                agent._chat_id = "testchat123"

                branch = "kiss/wt-testchat123-999"
                _create_worktree(repo, branch)
                GitWorktreeOps.save_original_branch(repo, branch, "main")
                GitWorktreeOps.save_baseline_commit(
                    repo, branch, "abc123def456",
                )

                agent._restore_from_git(repo)

                assert agent._wt is not None
                assert agent._wt.branch == branch
                assert agent._wt.original_branch == "main"
                assert agent._wt.baseline_commit == "abc123def456"
            finally:
                if th._db_conn is not None:
                    th._db_conn.close()
                    th._db_conn = None
                th._DB_PATH, th._db_conn, th._KISS_DIR = old_db

    def test_restore_without_baseline(self) -> None:
        """Legacy worktrees without baseline still restore correctly."""
        import kiss.agents.sorcar.persistence as th

        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

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
                agent._chat_id = "legacychat"

                branch = "kiss/wt-legacychat-999"
                _create_worktree(repo, branch)
                GitWorktreeOps.save_original_branch(repo, branch, "main")

                agent._restore_from_git(repo)

                assert agent._wt is not None
                assert agent._wt.branch == branch
                assert agent._wt.baseline_commit is None
            finally:
                if th._db_conn is not None:
                    th._db_conn.close()
                    th._db_conn = None
                th._DB_PATH, th._db_conn, th._KISS_DIR = old_db
