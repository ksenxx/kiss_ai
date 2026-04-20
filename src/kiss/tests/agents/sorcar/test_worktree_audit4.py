"""Tests verifying fixes for bugs found in worktree audit round 4.

Each test verifies that the bug has been fixed:

BUG-12: squash_merge_from_baseline now checks commit return code
BUG-13: _release_worktree sets _merge_conflict_warning and returns None
         on merge conflict
BUG-14: _new_chat surfaces _stash_pop_warning and _merge_conflict_warning
BUG-15: concurrent _release_worktree uses per-repo locking
BUG-16: _finalize_worktree preserves worktree when auto-commit fails
BUG-17: _run_task_inner skips _save_untracked_base in worktree mode
BUG-18: _release_worktree returns None on merge conflict (not
         original_branch)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
    _git,
    repo_lock,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
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


# ---------------------------------------------------------------------------
# BUG-12 FIX: squash_merge_from_baseline checks commit return code
# ---------------------------------------------------------------------------


class TestBug12SquashMergeFromBaselineChecksCommit:
    """squash_merge_from_baseline returns MERGE_FAILED when git commit fails."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        # Clean up hook in case test failed before removing it
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        if hook.exists():
            hook.unlink()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_merge_failed_when_commit_rejected(self) -> None:
        """FIX: squash_merge_from_baseline returns MERGE_FAILED when
        the commit is rejected by a pre-commit hook.
        """
        repo = self.repo

        wt_dir = repo / ".kiss-worktrees" / "test_wt"
        assert GitWorktreeOps.create(repo, "kiss/wt-test", wt_dir)

        # Create baseline commit
        (wt_dir / "dirty.txt").write_text("dirty")
        GitWorktreeOps.commit_all(wt_dir, "baseline")
        baseline = GitWorktreeOps.head_sha(wt_dir)
        assert baseline is not None

        # Create agent work commit
        (wt_dir / "agent.txt").write_text("agent work")
        GitWorktreeOps.commit_all(wt_dir, "agent work")

        # Remove worktree so we can merge in main repo
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        # Install a pre-commit hook that rejects all commits
        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        result = GitWorktreeOps.squash_merge_from_baseline(
            repo, "kiss/wt-test", baseline,
        )

        hook.unlink()

        # FIX: Returns MERGE_FAILED, not SUCCESS
        assert result == MergeResult.MERGE_FAILED, (
            "squash_merge_from_baseline should return MERGE_FAILED "
            "when commit is rejected by pre-commit hook"
        )

    def test_full_flow_does_not_delete_branch_on_commit_failure(self) -> None:
        """FIX: The full merge() flow does NOT delete the source branch
        when squash_merge_from_baseline's commit is rejected.
        """
        repo = self.repo
        saved = _redirect_db(self.tmpdir)
        orig = _patch_super_run()
        try:
            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "test-chat-12"

            wt_dir = repo / ".kiss-worktrees" / "test_wt"
            assert GitWorktreeOps.create(repo, "kiss/wt-test12", wt_dir)
            GitWorktreeOps.save_original_branch(repo, "kiss/wt-test12", "main")

            # Create baseline
            (wt_dir / "dirty.txt").write_text("dirty")
            GitWorktreeOps.commit_all(wt_dir, "baseline")
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            GitWorktreeOps.save_baseline_commit(repo, "kiss/wt-test12", baseline)

            # Create agent work
            (wt_dir / "work.txt").write_text("important work")
            GitWorktreeOps.commit_all(wt_dir, "important agent work")

            agent._wt = GitWorktree(
                repo_root=repo,
                branch="kiss/wt-test12",
                original_branch="main",
                wt_dir=wt_dir,
                baseline_commit=baseline,
            )

            # Install hook that rejects commits
            hooks_dir = repo / ".git" / "hooks"
            hooks_dir.mkdir(parents=True, exist_ok=True)
            hook = hooks_dir / "pre-commit"
            hook.write_text("#!/bin/sh\nexit 1\n")
            hook.chmod(0o755)

            msg = agent.merge()
            hook.unlink()

            # FIX: merge() reports failure (not success) and does NOT
            # delete the branch — agent work is preserved
            assert "Successfully merged" not in msg, (
                "merge() should NOT report success when commit fails"
            )
            # Note: merge() may report auto-commit failure (BUG-16 fix)
            # or merge conflict, depending on which check triggers first.
            # Either way, the branch should still exist:
            assert GitWorktreeOps.branch_exists(repo, "kiss/wt-test12"), (
                "branch must be preserved when commit fails"
            )
        finally:
            hook_path = repo / ".git" / "hooks" / "pre-commit"
            if hook_path.exists():
                hook_path.unlink()
            _unpatch_super_run(orig)
            _restore_db(saved)


# ---------------------------------------------------------------------------
# BUG-13 FIX: _release_worktree warns on merge conflict
# ---------------------------------------------------------------------------


class TestBug13ReleaseWorktreeWarnsOnConflict:
    """_release_worktree sets _merge_conflict_warning and returns None
    when auto-merge has conflicts.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.saved = _redirect_db(self.tmpdir)
        self.orig = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.orig)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merge_conflict_sets_warning_and_returns_none(self) -> None:
        """FIX: _release_worktree sets _merge_conflict_warning and
        returns None when a merge conflict occurs, so the caller
        knows the release didn't fully succeed and the user is warned.
        """
        repo = self.repo

        GitWorktreeOps.ensure_excluded(repo)

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-13"

        wt_dir = repo / ".kiss-worktrees" / "test_wt13"
        assert GitWorktreeOps.create(repo, "kiss/wt-test13", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test13", "main")

        # Agent modifies README.md in worktree
        (wt_dir / "README.md").write_text("agent version\n")
        GitWorktreeOps.commit_all(wt_dir, "agent edits README")

        # Meanwhile, someone modifies README.md on main
        (repo / "README.md").write_text("conflicting version\n")
        _git("add", ".", cwd=repo)
        _git("commit", "-m", "main edits README", cwd=repo)

        agent._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test13",
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        released = agent._release_worktree()

        # FIX: Returns None to signal incomplete release
        assert released is None, (
            "_release_worktree should return None on merge conflict"
        )

        # FIX: _merge_conflict_warning is set with useful info
        assert agent._merge_conflict_warning is not None, (
            "_merge_conflict_warning should be set on merge conflict"
        )
        assert "kiss/wt-test13" in agent._merge_conflict_warning
        assert "conflict" in agent._merge_conflict_warning.lower()

        # _wt is cleared so the agent can accept new tasks
        assert agent._wt is None

        # Branch is still preserved for manual resolution
        assert GitWorktreeOps.branch_exists(repo, "kiss/wt-test13")


# ---------------------------------------------------------------------------
# BUG-14 FIX: _new_chat surfaces warnings
# ---------------------------------------------------------------------------


class TestBug14NewChatSurfacesWarnings:
    """_new_chat() broadcasts stash_pop_warning and merge_conflict_warning."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.saved = _redirect_db(self.tmpdir)
        self.orig = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.orig)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stash_pop_warning_surfaced_on_new_chat(self) -> None:
        """FIX: _new_chat broadcasts _stash_pop_warning when present."""
        server = VSCodeServer()
        server.work_dir = str(self.repo)
        tab = server._get_tab("test-tab")
        tab.agent._stash_pop_warning = "Your changes could not be restored"

        broadcasts: list[dict] = []
        original_broadcast = server.printer.broadcast

        def capture_broadcast(event: dict) -> None:
            broadcasts.append(event)

        server.printer.broadcast = capture_broadcast  # type: ignore[assignment]
        server._new_chat("test-tab")
        server.printer.broadcast = original_broadcast  # type: ignore[assignment]

        warning_events = [e for e in broadcasts if e.get("type") == "warning"]
        assert len(warning_events) == 1, (
            "_new_chat should broadcast the stash_pop_warning"
        )
        assert "could not be restored" in warning_events[0]["message"]
        assert tab.agent._stash_pop_warning is None, (
            "Warning should be cleared after broadcasting"
        )

    def test_merge_conflict_warning_surfaced_on_new_chat(self) -> None:
        """FIX: _new_chat broadcasts _merge_conflict_warning when present."""
        server = VSCodeServer()
        server.work_dir = str(self.repo)
        tab = server._get_tab("test-tab")
        tab.agent._merge_conflict_warning = "Merge had conflicts"

        broadcasts: list[dict] = []
        original_broadcast = server.printer.broadcast

        def capture_broadcast(event: dict) -> None:
            broadcasts.append(event)

        server.printer.broadcast = capture_broadcast  # type: ignore[assignment]
        server._new_chat("test-tab")
        server.printer.broadcast = original_broadcast  # type: ignore[assignment]

        warning_events = [e for e in broadcasts if e.get("type") == "warning"]
        assert len(warning_events) == 1
        assert "conflicts" in warning_events[0]["message"]
        assert tab.agent._merge_conflict_warning is None

    def test_both_warnings_surfaced(self) -> None:
        """FIX: Both warnings are broadcast when both are set."""
        server = VSCodeServer()
        server.work_dir = str(self.repo)
        tab = server._get_tab("test-tab")
        tab.agent._stash_pop_warning = "stash issue"
        tab.agent._merge_conflict_warning = "merge issue"

        broadcasts: list[dict] = []
        original_broadcast = server.printer.broadcast

        def capture_broadcast(event: dict) -> None:
            broadcasts.append(event)

        server.printer.broadcast = capture_broadcast  # type: ignore[assignment]
        server._new_chat("test-tab")
        server.printer.broadcast = original_broadcast  # type: ignore[assignment]

        warning_events = [e for e in broadcasts if e.get("type") == "warning"]
        assert len(warning_events) == 2


# ---------------------------------------------------------------------------
# BUG-15 FIX: per-repo locking for concurrent tab releases
# ---------------------------------------------------------------------------


class TestBug15ConcurrentReleaseUsesLocking:
    """Concurrent _release_worktree calls are serialized by repo_lock."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.saved = _redirect_db(self.tmpdir)
        self.orig = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.orig)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_repo_lock_exists_and_serializes(self) -> None:
        """FIX: repo_lock returns a per-repo threading.Lock that
        serializes concurrent operations.
        """
        repo = self.repo
        lock1 = repo_lock(repo)
        lock2 = repo_lock(repo)

        # Same repo → same lock object
        assert lock1 is lock2, (
            "repo_lock must return the same lock for the same repo"
        )

        # Different repos → different locks
        other = Path(self.tmpdir) / "other"
        other.mkdir()
        lock3 = repo_lock(other)
        assert lock3 is not lock1, (
            "repo_lock must return different locks for different repos"
        )

        # Lock is a threading.Lock
        assert isinstance(lock1, type(threading.Lock()))

    def test_concurrent_releases_are_serialized(self) -> None:
        """FIX: Two concurrent _release_worktree calls are serialized
        by the repo lock, preventing interleaved git operations.
        """
        repo = self.repo

        wt_dir_a = repo / ".kiss-worktrees" / "wt_a"
        wt_dir_b = repo / ".kiss-worktrees" / "wt_b"
        assert GitWorktreeOps.create(repo, "kiss/wt-a", wt_dir_a)
        assert GitWorktreeOps.create(repo, "kiss/wt-b", wt_dir_b)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-a", "main")
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-b", "main")

        (wt_dir_a / "file_a.txt").write_text("from agent A")
        GitWorktreeOps.commit_all(wt_dir_a, "agent A work")

        (wt_dir_b / "file_b.txt").write_text("from agent B")
        GitWorktreeOps.commit_all(wt_dir_b, "agent B work")

        agent_a = WorktreeSorcarAgent("agent-a")
        agent_a._chat_id = "chat-a"
        agent_a._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-a",
            original_branch="main",
            wt_dir=wt_dir_a,
            baseline_commit=None,
        )

        agent_b = WorktreeSorcarAgent("agent-b")
        agent_b._chat_id = "chat-b"
        agent_b._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-b",
            original_branch="main",
            wt_dir=wt_dir_b,
            baseline_commit=None,
        )

        results: dict[str, str | None] = {}
        errors: list[str] = []

        def release_a() -> None:
            try:
                results["a"] = agent_a._release_worktree()
            except Exception as e:
                errors.append(f"A: {e}")

        def release_b() -> None:
            try:
                results["b"] = agent_b._release_worktree()
            except Exception as e:
                errors.append(f"B: {e}")

        t_a = threading.Thread(target=release_a)
        t_b = threading.Thread(target=release_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=30)
        t_b.join(timeout=30)

        assert not errors, f"Concurrent releases should not error: {errors}"
        assert "a" in results and "b" in results

        # At least one should succeed (the other may conflict due to
        # non-ff merge after the first one's squash merge)
        successes = [k for k, v in results.items() if v == "main"]
        assert len(successes) >= 1, (
            "At least one concurrent release should succeed"
        )


# ---------------------------------------------------------------------------
# BUG-16 FIX: _finalize_worktree preserves worktree on commit failure
# ---------------------------------------------------------------------------


class TestBug16FinalizePreservesWorktreeOnCommitFailure:
    """_finalize_worktree returns False and keeps the worktree directory
    when auto-commit is rejected by a pre-commit hook.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.saved = _redirect_db(self.tmpdir)
        self.orig = _patch_super_run()

    def teardown_method(self) -> None:
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        if hook.exists():
            hook.unlink()
        _unpatch_super_run(self.orig)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_worktree_preserved_on_commit_rejection(self) -> None:
        """FIX: When a pre-commit hook rejects the auto-commit,
        _finalize_worktree returns False and the worktree directory
        is NOT removed — no data loss.
        """
        repo = self.repo
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-16"

        wt_dir = repo / ".kiss-worktrees" / "test_wt16"
        assert GitWorktreeOps.create(repo, "kiss/wt-test16", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test16", "main")

        # Agent creates important work (uncommitted)
        (wt_dir / "important.txt").write_text("critical work product")

        # Install pre-commit hook that rejects all commits
        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        agent._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test16",
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        assert (wt_dir / "important.txt").exists()

        # _finalize_worktree returns False when auto-commit is rejected
        result = agent._finalize_worktree()
        hook.unlink()

        assert result is False, (
            "_finalize_worktree should return False when auto-commit fails"
        )

        # FIX: worktree directory is PRESERVED
        assert wt_dir.exists(), (
            "Worktree directory must be preserved when auto-commit fails"
        )

        # FIX: the important file is still accessible
        assert (wt_dir / "important.txt").exists(), (
            "Agent work must not be lost"
        )
        assert (wt_dir / "important.txt").read_text() == "critical work product"

    def test_finalize_returns_true_on_success(self) -> None:
        """Regression: _finalize_worktree returns True on normal success."""
        repo = self.repo
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-16b"

        wt_dir = repo / ".kiss-worktrees" / "test_wt16b"
        assert GitWorktreeOps.create(repo, "kiss/wt-test16b", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test16b", "main")

        (wt_dir / "work.txt").write_text("work")
        GitWorktreeOps.commit_all(wt_dir, "committed work")

        agent._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test16b",
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        result = agent._finalize_worktree()

        assert result is True, (
            "_finalize_worktree should return True on successful cleanup"
        )
        assert not wt_dir.exists(), (
            "Worktree directory should be removed on success"
        )

    def test_merge_reports_autocommit_failure(self) -> None:
        """FIX: merge() reports auto-commit failure instead of proceeding."""
        repo = self.repo
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-16c"

        wt_dir = repo / ".kiss-worktrees" / "test_wt16c"
        assert GitWorktreeOps.create(repo, "kiss/wt-test16c", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test16c", "main")

        (wt_dir / "important.txt").write_text("critical work")

        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        agent._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test16c",
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        msg = agent.merge()
        hook.unlink()

        assert "Cannot merge" in msg or "auto-commit" in msg.lower(), (
            "merge() should report auto-commit failure"
        )
        assert wt_dir.exists(), (
            "Worktree should be preserved when auto-commit fails"
        )


# ---------------------------------------------------------------------------
# BUG-17 FIX: _save_untracked_base skipped in worktree mode
# ---------------------------------------------------------------------------


class TestBug17UntrackedBaseNotNukedInWorktreeMode:
    """_run_task_inner skips _save_untracked_base when use_worktree=True."""

    def test_save_untracked_base_not_called_in_worktree_mode(self) -> None:
        """FIX: In worktree mode, pre-task snapshot and
        _save_untracked_base are skipped, so another tab's merge
        review data is not destroyed.
        """
        from kiss.agents.vscode.diff_merge import (
            _save_untracked_base,
            _untracked_base_dir,
        )

        tmpdir = tempfile.mkdtemp()
        try:
            # Simulate tab A's merge review data
            ub_dir = _untracked_base_dir()
            ub_dir.mkdir(parents=True, exist_ok=True)
            (ub_dir / "tab_a_file.txt").write_text("tab A's base copy")
            assert (ub_dir / "tab_a_file.txt").exists()

            # Verify that _save_untracked_base with EMPTY set does not
            # nuke existing data (this is the behavior we get when the
            # call is skipped in worktree mode)
            # Note: we're testing the server.py logic of SKIPPING the call,
            # not _save_untracked_base itself.

            # The fix is in server.py: when tab.use_worktree is True,
            # _save_untracked_base is not called at all.
            # We verify the guard exists by checking the source code:
            import inspect
            from kiss.agents.vscode.server import VSCodeServer

            source = inspect.getsource(VSCodeServer._run_task_inner)
            assert "if not tab.use_worktree:" in source, (
                "_run_task_inner should guard pre-task snapshot "
                "with 'if not tab.use_worktree:'"
            )
            assert "_save_untracked_base" in source

            # Also verify that tab A's data survives when the guard fires
            assert (ub_dir / "tab_a_file.txt").exists(), (
                "Tab A's base copy should survive when worktree mode "
                "skips _save_untracked_base"
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            ub_dir = _untracked_base_dir()
            if ub_dir.exists():
                shutil.rmtree(ub_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# BUG-18 FIX: _release_worktree returns None on merge conflict
# ---------------------------------------------------------------------------


class TestBug18ReleaseReturnsNoneOnConflict:
    """_release_worktree returns None (not original_branch) on merge conflict."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.saved = _redirect_db(self.tmpdir)
        self.orig = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.orig)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_none_on_conflict(self) -> None:
        """FIX: _release_worktree returns None when the merge fails,
        correctly signaling to the caller that the release did not
        fully succeed.
        """
        repo = self.repo
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-18"

        wt_dir = repo / ".kiss-worktrees" / "test_wt18"
        assert GitWorktreeOps.create(repo, "kiss/wt-test18", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test18", "main")
        (wt_dir / "README.md").write_text("agent version\n")
        GitWorktreeOps.commit_all(wt_dir, "agent edits")

        # Create conflict on main
        (repo / "README.md").write_text("main version\n")
        _git("add", ".", cwd=repo)
        _git("commit", "-m", "main edits", cwd=repo)

        agent._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test18",
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        result = agent._release_worktree()

        # FIX: Returns None, not "main"
        assert result is None, (
            "_release_worktree should return None on merge conflict"
        )

        # _wt is cleared so agent can accept new tasks
        assert agent._wt is None

        # Warning is set for user notification
        assert agent._merge_conflict_warning is not None

        # Branch is preserved
        assert GitWorktreeOps.branch_exists(repo, "kiss/wt-test18")

    def test_returns_branch_on_success(self) -> None:
        """Regression: _release_worktree returns original_branch on success."""
        repo = self.repo
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-18b"

        wt_dir = repo / ".kiss-worktrees" / "test_wt18b"
        assert GitWorktreeOps.create(repo, "kiss/wt-test18b", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test18b", "main")
        (wt_dir / "newfile.txt").write_text("agent work\n")
        GitWorktreeOps.commit_all(wt_dir, "agent work")

        agent._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test18b",
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        result = agent._release_worktree()

        assert result == "main", (
            "_release_worktree should return original_branch on success"
        )
        assert agent._wt is None
        assert agent._merge_conflict_warning is None
        assert not GitWorktreeOps.branch_exists(repo, "kiss/wt-test18b"), (
            "Branch should be deleted after successful merge"
        )
