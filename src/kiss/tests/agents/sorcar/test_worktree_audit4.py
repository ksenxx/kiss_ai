"""Tests confirming bugs found in worktree audit round 4.

Each test confirms a specific bug exists in the current code.

BUG-12: squash_merge_from_baseline doesn't check commit return code
BUG-13: _release_worktree silently orphans branch on merge conflict
         (no user notification)
BUG-14: new_chat auto-merges pending worktree but stash_pop warning
         is never surfaced to the user
BUG-15: concurrent _release_worktree from two tabs races on the main
         repo (no git-level locking)
BUG-16: _finalize_worktree removes worktree even if auto-commit
         FAILED (commit rejected, not "nothing to commit")
         — still-unfixed BUG-6 from audit2
BUG-17: _run_task_inner calls _save_untracked_base for the main repo
         even in worktree mode — nukes untracked-base dir that may
         be in use by another tab's merge review
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
# BUG-12: squash_merge_from_baseline doesn't check commit return code
# ---------------------------------------------------------------------------


class TestBug12SquashMergeFromBaselineUncheckedCommit:
    """squash_merge_from_baseline returns SUCCESS even when git commit fails."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_success_when_commit_fails(self) -> None:
        """BUG-12: If git commit fails after cherry-pick --no-commit,
        squash_merge_from_baseline still returns MergeResult.SUCCESS,
        causing the caller to delete the source branch and lose work.
        """
        repo = self.repo

        # Create a worktree branch with a baseline and agent work
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

        # Attempt squash merge from baseline — commit will be rejected
        result = GitWorktreeOps.squash_merge_from_baseline(
            repo, "kiss/wt-test", baseline,
        )

        # BUG: Returns SUCCESS even though the commit failed.
        # The staged changes are left in the index, not committed.
        assert result == MergeResult.SUCCESS, (
            "BUG-12: squash_merge_from_baseline should return SUCCESS "
            "even when commit fails (confirming the bug exists)"
        )

        # Verify the commit actually failed — HEAD should still be 'initial'
        log = _git("log", "--oneline", cwd=repo)
        lines = log.stdout.strip().splitlines()
        assert len(lines) == 1, (
            "BUG-12: Only the initial commit should exist because the "
            f"pre-commit hook rejected the merge commit, got: {lines}"
        )

        # Clean up hook so teardown can work
        hook.unlink()

    def test_full_flow_deletes_branch_despite_commit_failure(self) -> None:
        """BUG-12: The full merge() flow deletes the source branch even
        though the squash_merge_from_baseline commit was rejected.
        Agent work is permanently lost.
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

            # BUG: merge() reports success and deletes the branch
            assert "Successfully merged" in msg, (
                "BUG-12: merge() reports success despite commit failure"
            )
            assert not GitWorktreeOps.branch_exists(repo, "kiss/wt-test12"), (
                "BUG-12: branch is deleted even though commit failed"
            )
        finally:
            if (repo / ".git" / "hooks" / "pre-commit").exists():
                (repo / ".git" / "hooks" / "pre-commit").unlink()
            _unpatch_super_run(orig)
            _restore_db(saved)


# ---------------------------------------------------------------------------
# BUG-13: _release_worktree silently orphans branch on merge conflict
# ---------------------------------------------------------------------------


class TestBug13ReleaseWorktreeSilentOrphan:
    """_release_worktree doesn't notify the user when auto-merge has
    conflicts — just logs a warning and clears self._wt.
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

    def test_merge_conflict_silently_orphans_branch(self) -> None:
        """BUG-13: When _release_worktree encounters a merge conflict,
        the branch is kept in git but self._wt is cleared and the user
        gets NO notification about the merge failure — the branch
        becomes silently orphaned.
        """
        repo = self.repo

        # Ensure .kiss-worktrees is excluded so it doesn't interfere
        GitWorktreeOps.ensure_excluded(repo)

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-13"

        # Create worktree branch with agent work
        wt_dir = repo / ".kiss-worktrees" / "test_wt13"
        assert GitWorktreeOps.create(repo, "kiss/wt-test13", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test13", "main")

        # Agent modifies README.md in worktree
        (wt_dir / "README.md").write_text("agent version\n")
        GitWorktreeOps.commit_all(wt_dir, "agent edits README")

        # Meanwhile, someone modifies README.md on main — creating conflict
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

        # Release the worktree — auto-merge will conflict
        released = agent._release_worktree()

        # BUG: released_branch is returned as "main" (as if successful)
        # The caller in _try_setup_worktree will use this as the new
        # original_branch, never knowing the merge had conflicts.
        assert released == "main", (
            "BUG-13: _release_worktree returns original_branch even "
            "on merge conflict, implying success"
        )

        # BUG: self._wt is cleared — no way to retry or access the branch
        assert agent._wt is None, (
            "BUG-13: _wt is cleared despite merge failure"
        )

        # BUG: the branch is orphaned — still exists in git
        assert GitWorktreeOps.branch_exists(repo, "kiss/wt-test13"), (
            "BUG-13: branch is kept but user has no way to know about it"
        )

        # BUG: there is no merge-conflict-specific warning mechanism.
        # The _stash_pop_warning only fires when stash pop fails — a
        # separate concern.  There is no _merge_conflict_warning or
        # similar field.  The conflict is only logged, never surfaced.
        assert not hasattr(agent, "_merge_conflict_warning"), (
            "BUG-13: no merge_conflict_warning attribute exists — "
            "merge conflicts during release are only logged, never "
            "reported to the user"
        )


# ---------------------------------------------------------------------------
# BUG-14: new_chat stash_pop_warning never surfaced
# ---------------------------------------------------------------------------


class TestBug14NewChatStashPopWarningLost:
    """new_chat() auto-releases a pending worktree but the server's
    _new_chat() doesn't check or broadcast the stash_pop_warning.
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

    def test_stash_pop_warning_not_surfaced_on_new_chat(self) -> None:
        """BUG-14: If _release_worktree sets _stash_pop_warning during
        new_chat(), the server's _new_chat handler doesn't check or
        broadcast it. The warning only appears when the NEXT task
        starts — which could be much later, or never.
        """
        repo = self.repo
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-14"

        # Create worktree with agent work
        wt_dir = repo / ".kiss-worktrees" / "test_wt14"
        assert GitWorktreeOps.create(repo, "kiss/wt-test14", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test14", "main")
        (wt_dir / "agent.txt").write_text("work")
        GitWorktreeOps.commit_all(wt_dir, "agent work")

        agent._wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-test14",
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        # Create dirty state in main repo that will conflict with stash pop
        # First, make changes and stash them manually to set up the scenario
        (repo / "agent.txt").write_text("conflicting content")
        _git("add", ".", cwd=repo)
        _git("commit", "-m", "conflict setup", cwd=repo)

        # Call new_chat which internally calls _release_worktree
        agent.new_chat()

        # After new_chat, check if warning was surfaced
        # In server.py, _new_chat does:
        #   tab.agent.new_chat()
        #   self.printer.broadcast({"type": "showWelcome", ...})
        # It does NOT check agent._stash_pop_warning

        # Verify the server handler doesn't surface the warning
        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("test-tab")
        tab.agent = agent
        tab.use_worktree = True

        # Simulate stash_pop_warning being set (as _release_worktree would)
        tab.agent._stash_pop_warning = "Your changes could not be restored"

        # Collect broadcasts
        broadcasts: list[dict] = []
        original_broadcast = server.printer.broadcast

        def capture_broadcast(event: dict) -> None:
            broadcasts.append(event)

        server.printer.broadcast = capture_broadcast  # type: ignore[assignment]
        server._new_chat("test-tab")

        # BUG: No warning broadcast — only showWelcome is emitted
        warning_events = [e for e in broadcasts if e.get("type") == "warning"]
        assert len(warning_events) == 0, (
            "BUG-14: _new_chat does NOT surface stash_pop_warning. "
            "Only showWelcome is broadcast."
        )

        # The warning is still on the agent, waiting for the next run()
        # If the user never runs another task, the warning is lost forever.
        assert tab.agent._stash_pop_warning is not None, (
            "BUG-14: stash_pop_warning persists on the agent but is "
            "never shown through _new_chat"
        )

        server.printer.broadcast = original_broadcast  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# BUG-15: concurrent _release_worktree from two tabs races on main repo
# ---------------------------------------------------------------------------


class TestBug15ConcurrentReleaseRace:
    """Two tabs finishing tasks simultaneously both call _release_worktree
    in their own threads. Without git-level locking, they race on the
    main repo (checkout, stash, merge, stash_pop).
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

    def test_concurrent_releases_have_no_locking(self) -> None:
        """BUG-15: There is no mutex or lock around the git operations
        performed by _release_worktree. Two concurrent releases can
        interleave checkout/stash/merge/pop operations on the same repo.

        This test demonstrates the structural issue: both agents can enter
        _release_worktree simultaneously on the same repo without any
        synchronization.
        """
        repo = self.repo

        # Create two worktree branches (simulating two concurrent tabs)
        wt_dir_a = repo / ".kiss-worktrees" / "wt_a"
        wt_dir_b = repo / ".kiss-worktrees" / "wt_b"
        assert GitWorktreeOps.create(repo, "kiss/wt-a", wt_dir_a)
        assert GitWorktreeOps.create(repo, "kiss/wt-b", wt_dir_b)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-a", "main")
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-b", "main")

        # Agent A makes changes
        (wt_dir_a / "file_a.txt").write_text("from agent A")
        GitWorktreeOps.commit_all(wt_dir_a, "agent A work")

        # Agent B makes changes
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

        # Verify there's no locking mechanism — both releases share the
        # same repo object, and GitWorktreeOps methods have no mutex.
        # The _state_lock in VSCodeServer only guards tab state dict,
        # NOT git operations.

        # Demonstrate the race: run both releases concurrently
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

        # BUG: Both agents attempt checkout/stash/merge on the same
        # repo without any synchronization. In the best case, one
        # succeeds and the other fails gracefully. In the worst case,
        # the interleaving corrupts the repo state.
        #
        # The structural issue is that WorktreeSorcarAgent._release_worktree
        # and GitWorktreeOps have no locking mechanism whatsoever.
        # Verify at least that both completed without crashing:
        assert "a" in results and "b" in results, (
            f"BUG-15: concurrent releases should both complete. "
            f"errors={errors}, results={results}"
        )

        # Verify that there's genuinely no lock protecting these operations
        # by checking that GitWorktreeOps has no lock attribute
        assert not hasattr(GitWorktreeOps, "_lock"), (
            "BUG-15: GitWorktreeOps has no locking mechanism"
        )
        assert not hasattr(GitWorktreeOps, "_mutex"), (
            "BUG-15: GitWorktreeOps has no mutex"
        )


# ---------------------------------------------------------------------------
# BUG-16: _finalize_worktree removes worktree even if auto-commit FAILED
# (still-unfixed BUG-6)
# ---------------------------------------------------------------------------


class TestBug16FinalizeRemovesWorktreeDespiteCommitFailure:
    """_finalize_worktree removes the worktree directory even if the
    auto-commit was rejected by a pre-commit hook. This loses all
    uncommitted agent work permanently.

    This is the same issue as BUG-6 from audit2 and is still unfixed.
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

    def test_worktree_removed_despite_commit_rejection(self) -> None:
        """BUG-16: When a pre-commit hook rejects the auto-commit,
        _finalize_worktree still removes the worktree directory,
        permanently losing all uncommitted agent work.
        """
        repo = self.repo
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-16"

        # Create worktree with uncommitted work
        wt_dir = repo / ".kiss-worktrees" / "test_wt16"
        assert GitWorktreeOps.create(repo, "kiss/wt-test16", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-test16", "main")

        # Agent creates important work (uncommitted)
        (wt_dir / "important.txt").write_text("critical work product")

        # Install pre-commit hook that rejects all commits in worktree
        # The worktree uses the same hooks as the main repo
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

        # Verify the file exists before finalize
        assert (wt_dir / "important.txt").exists()

        # _finalize_worktree auto-commits (which fails) then removes worktree
        agent._finalize_worktree()

        # Clean up hook
        hook.unlink()

        # BUG: worktree directory is GONE despite commit failure
        assert not wt_dir.exists(), (
            "BUG-16: worktree is removed even though auto-commit failed"
        )

        # BUG: the uncommitted work is permanently lost
        # The branch exists but has no commits with "important.txt"
        show_result = _git(
            "show", "kiss/wt-test16:important.txt", cwd=repo,
        )
        assert show_result.returncode != 0, (
            "BUG-16: important.txt was never committed to the branch "
            "because the pre-commit hook rejected it — work is lost"
        )


# ---------------------------------------------------------------------------
# BUG-17: _run_task_inner saves untracked base for main repo even in
#         worktree mode — can nuke another tab's merge review data
# ---------------------------------------------------------------------------


class TestBug17UntrackedBaseNukedInWorktreeMode:
    """_run_task_inner always calls _save_untracked_base(work_dir, ...)
    for the main repo, even when use_worktree=True. This deletes the
    untracked-base directory that another tab's pending merge review
    may depend on.
    """

    def test_save_untracked_base_deletes_existing(self) -> None:
        """BUG-17: _save_untracked_base unconditionally deletes the
        existing untracked-base directory before saving new copies.
        When called for a worktree task's main-repo snapshot, this
        nukes data from any concurrent non-worktree merge review.
        """
        from kiss.agents.vscode.diff_merge import (
            _save_untracked_base,
            _untracked_base_dir,
        )

        tmpdir = tempfile.mkdtemp()
        try:
            work_dir = tmpdir

            # Simulate tab A saving untracked base for its merge review
            ub_dir = _untracked_base_dir()
            ub_dir.mkdir(parents=True, exist_ok=True)
            (ub_dir / "tab_a_file.txt").write_text("tab A's base copy")
            assert (ub_dir / "tab_a_file.txt").exists()

            # Tab B starts a worktree task and _run_task_inner calls
            # _save_untracked_base(work_dir, ...) for the main repo
            _save_untracked_base(work_dir, set())

            # BUG: tab A's base copy is GONE
            assert not (ub_dir / "tab_a_file.txt").exists(), (
                "BUG-17: _save_untracked_base deletes ALL existing "
                "untracked base copies, even from other tabs"
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            # Clean up the untracked base dir
            ub_dir = _untracked_base_dir()
            if ub_dir.exists():
                shutil.rmtree(ub_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# BUG-18: _release_worktree returns original_branch on merge conflict,
#         misleading caller about the actual outcome
# ---------------------------------------------------------------------------


class TestBug18ReleaseReturnsMisleadingBranchOnConflict:
    """When squash merge fails during _release_worktree, the method
    returns wt.original_branch (not None), suggesting success even
    though the merge didn't happen.
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

    def test_returns_original_branch_despite_conflict(self) -> None:
        """BUG-18: _release_worktree returns wt.original_branch even
        when the merge failed. The caller in _try_setup_worktree uses
        this as the new worktree's original_branch, which is correct
        by coincidence, but the return value is semantically wrong —
        it says "I released to this branch" when in fact the merge
        had conflicts.
        """
        repo = self.repo
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test-chat-18"

        # Create worktree that modifies README
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

        # BUG: Returns "main" as if the release succeeded
        assert result == "main", (
            "BUG-18: _release_worktree returns the original branch name "
            "even on merge conflict, not None"
        )

        # The branch is orphaned — still exists but _wt is cleared
        assert agent._wt is None
        assert GitWorktreeOps.branch_exists(repo, "kiss/wt-test18")
