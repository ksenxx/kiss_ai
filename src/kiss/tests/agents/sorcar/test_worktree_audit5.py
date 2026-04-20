"""Tests confirming bugs found in worktree audit round 5.

Each test CONFIRMS the bug exists (assertions pass when buggy behaviour
is present).

BUG-19: discard() doesn't acquire repo_lock — can race with
        merge/release from another tab on the same repo.
BUG-20: _release_worktree checkout failure silently orphans the branch
        (no _merge_conflict_warning set, user gets no notification).
BUG-21: checkout_error() re-executes `git checkout` — a side-effecting
        diagnostic that can mutate repo state while the caller reports
        failure.
BUG-22: _check_merge_conflict misses staged (but uncommitted) files
        that overlap with worktree changes — reports no conflict when
        `git merge` would actually refuse.
BUG-23: _try_setup_worktree doesn't check commit_staged return value
        for the baseline commit — if a pre-commit hook rejects the
        dirty-state commit, baseline_commit is set to the creation-time
        HEAD (wrong SHA), so downstream squash_merge_from_baseline
        treats the user's dirty state as agent changes.
BUG-24: _get_worktree_changed_files returns [] on transient git diff
        failure; caller in _run_task_inner then silently calls
        discard(), potentially losing agent work.
"""

from __future__ import annotations

import inspect
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
    _git,
    repo_lock,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers (same as prior audit test files)
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


# ===================================================================
# BUG-19: discard() doesn't acquire repo_lock
# ===================================================================


class TestBug19DiscardNoRepoLock:
    """BUG-19: discard() calls GitWorktreeOps.checkout without repo_lock.

    merge() and _release_worktree() both acquire repo_lock before
    checkout + stash + merge + pop.  discard() does not, so it can
    interleave with those operations on the same repo when two tabs
    are active concurrently.
    """

    def test_discard_does_not_acquire_repo_lock(self) -> None:
        """BUG-19: Confirm discard() source code has no repo_lock usage."""
        source = inspect.getsource(WorktreeSorcarAgent.discard)
        # BUG-19: discard() does not use repo_lock — this SHOULD fail
        # once the bug is fixed (discard should acquire the lock).
        assert "repo_lock" not in source, (
            "BUG-19 appears fixed: discard() now uses repo_lock"
        )

    def test_discard_checkout_can_race_with_release(self) -> None:
        """BUG-19: Demonstrate that discard's checkout is unprotected."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            # Create two agents on the same repo (simulating two tabs)
            agent_a = WorktreeSorcarAgent("tab-a")
            agent_a._chat_id = "a"
            agent_b = WorktreeSorcarAgent("tab-b")
            agent_b._chat_id = "b"

            # Set up worktrees for both
            wt_a = agent_a._try_setup_worktree(repo, str(repo))
            assert wt_a is not None
            (agent_a._wt.wt_dir / "a.txt").write_text("a\n")
            GitWorktreeOps.commit_all(agent_a._wt.wt_dir, "agent-a work")

            wt_b = agent_b._try_setup_worktree(repo, str(repo))
            assert wt_b is not None
            (agent_b._wt.wt_dir / "b.txt").write_text("b\n")
            GitWorktreeOps.commit_all(agent_b._wt.wt_dir, "agent-b work")

            lock = repo_lock(repo)

            # Simulate: tab-a holds the lock (mid-merge), tab-b tries discard
            lock.acquire()
            try:
                # discard() should block on repo_lock if it acquired it,
                # but since it doesn't, it proceeds immediately.
                # We verify by checking discard() returns without blocking.
                completed = threading.Event()

                def try_discard() -> None:
                    agent_b.discard()
                    completed.set()

                t = threading.Thread(target=try_discard)
                t.start()
                # BUG-19: discard completes immediately even though the
                # repo lock is held by another operation.
                t.join(timeout=2.0)
                assert completed.is_set(), (
                    "BUG-19 appears fixed: discard() blocked on repo_lock"
                )
            finally:
                lock.release()
        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===================================================================
# BUG-20: _release_worktree checkout failure sets no warning
# ===================================================================


class TestBug20ReleaseCheckoutNoWarning:
    """BUG-20: When _release_worktree fails to checkout the original
    branch, it returns None but does NOT set _merge_conflict_warning.
    The branch is orphaned with no user notification.

    Contrast: the merge-conflict path correctly sets the warning.
    """

    def test_checkout_failure_orphans_branch_silently(self) -> None:
        """BUG-20: _release_worktree sets no warning on checkout failure."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            # Create a second branch "feature" with different content
            _git("checkout", "-b", "feature", cwd=repo)
            (repo / "README.md").write_text("# Feature\n")
            _git("add", ".", cwd=repo)
            _git("commit", "-m", "feature change", cwd=repo)
            _git("checkout", "main", cwd=repo)

            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "test20"

            # Worktree starts from "main" — original_branch = "main"
            wt_work = agent._try_setup_worktree(repo, str(repo))
            assert wt_work is not None

            wt = agent._wt
            assert wt is not None
            branch = wt.branch

            # Create a commit so the branch has content
            (wt.wt_dir / "file.txt").write_text("work\n")
            GitWorktreeOps.commit_all(wt.wt_dir, "agent work")

            # Switch main repo to "feature" and create a dirty conflicting
            # state that prevents checkout back to "main"
            _git("checkout", "feature", cwd=repo)
            (repo / "README.md").write_text("dirty local change\n")

            # Modify original_branch to "main" so release tries to
            # checkout "main", which will fail because of dirty README.md
            agent._wt = GitWorktree(
                repo_root=wt.repo_root,
                branch=wt.branch,
                original_branch="main",
                wt_dir=wt.wt_dir,
                baseline_commit=wt.baseline_commit,
            )

            result = agent._release_worktree()

            # BUG-20: returns None (correct) but no warning is set (wrong)
            assert result is None, "Expected None on checkout failure"

            # The branch should still exist (orphaned)
            assert GitWorktreeOps.branch_exists(repo, branch), (
                "Branch should be kept when checkout fails"
            )

            # BUG-20: no warning was set — user has no idea the branch
            # was orphaned. This SHOULD be set once the bug is fixed.
            assert agent._merge_conflict_warning is None, (
                "BUG-20 appears fixed: warning is now set on checkout failure"
            )

        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===================================================================
# BUG-21: checkout_error() is side-effecting
# ===================================================================


class TestBug21CheckoutErrorSideEffect:
    """BUG-21: GitWorktreeOps.checkout_error() re-executes `git checkout`
    to capture the error message.  This means it actually attempts the
    checkout again, which could succeed on the second try (e.g. if a
    transient lock was held) while the caller reports failure.
    """

    def test_checkout_error_runs_git_checkout(self) -> None:
        """BUG-21: Confirm checkout_error's source contains 'git checkout'."""
        source = inspect.getsource(GitWorktreeOps.checkout_error)
        # checkout_error should NOT re-run the checkout.  It should
        # capture the error from the original failed attempt.
        assert '"checkout"' in source or "'checkout'" in source, (
            "checkout_error should reference 'checkout' (sanity check)"
        )
        # BUG-21: The function body contains _git("checkout", ...)
        # which means it re-runs the checkout command.
        assert "_git(" in source, (
            "BUG-21 appears fixed: checkout_error no longer calls _git"
        )

    def test_checkout_error_can_succeed_on_second_attempt(self) -> None:
        """BUG-21: If a transient condition caused the first checkout to
        fail, checkout_error's re-attempt could succeed, leaving the repo
        on a different branch while the caller reports failure.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            # Create a second branch
            _git("branch", "feature", cwd=repo)

            # Verify that checkout_error actually runs checkout
            # (by checking it on a branch that exists — it will succeed)
            GitWorktreeOps.checkout_error(repo, "feature")

            # If checkout_error ran checkout, we're now on 'feature'
            current = GitWorktreeOps.current_branch(repo)

            # BUG-21: checkout_error actually switched branches!
            assert current == "feature", (
                "BUG-21 appears fixed: checkout_error no longer switches branches"
            )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===================================================================
# BUG-22: _check_merge_conflict misses staged files
# ===================================================================


class TestBug22ConflictMissesStaged:
    """BUG-22: _check_merge_conflict uses GitWorktreeOps.unstaged_files()
    which only detects unstaged modifications.  Staged-but-uncommitted
    files that overlap with worktree changes would cause `git merge` to
    refuse, but _check_merge_conflict reports no conflict.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_staged_overlap_not_detected(self) -> None:
        """BUG-22: Staged changes overlapping with worktree are missed."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        # Create worktree and make a change to shared.txt
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "test22"
        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt = agent._wt
        assert wt is not None

        (wt.wt_dir / "shared.txt").write_text("agent version\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "agent changes shared.txt")

        # In the main repo, stage a change to the same file
        (repo / "shared.txt").write_text("user version\n")
        _git("add", "shared.txt", cwd=repo)

        # Verify the file IS staged
        staged = _git("diff", "--cached", "--name-only", cwd=repo)
        assert "shared.txt" in staged.stdout, "shared.txt should be staged"

        # Verify it's NOT in unstaged_files (the function used by _check_merge_conflict)
        unstaged = GitWorktreeOps.unstaged_files(repo)
        assert "shared.txt" not in unstaged, (
            "staged files should NOT appear in unstaged_files"
        )

        # Set up a server to call _check_merge_conflict
        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("t22")
        tab.agent = agent
        tab.use_worktree = True

        has_conflict = server._check_merge_conflict("t22")

        # BUG-22: conflict is NOT detected even though git merge would
        # refuse due to the staged change.  Once fixed, this should
        # return True.
        assert has_conflict is False, (
            "BUG-22 appears fixed: staged overlap is now detected"
        )

    def test_unstaged_files_only_returns_unstaged(self) -> None:
        """Confirm unstaged_files() uses git diff --name-only (no --cached)."""
        source = inspect.getsource(GitWorktreeOps.unstaged_files)
        # Should NOT contain --cached (that's the bug)
        assert "--cached" not in source, (
            "unstaged_files correctly omits --cached (expected behavior)"
        )


# ===================================================================
# BUG-23: Wrong baseline when pre-commit hook rejects dirty-state commit
# ===================================================================


class TestBug23WrongBaselineOnHookRejection:
    """BUG-23: _try_setup_worktree doesn't check whether commit_staged
    succeeded for the baseline commit.  If a pre-commit hook rejects the
    dirty-state commit, baseline_commit is set to the HEAD at creation
    time (same as the original branch tip), not the dirty-state snapshot.

    Downstream squash_merge_from_baseline then cherry-picks from this
    wrong baseline, treating the user's dirty state as agent changes.
    """

    def test_baseline_set_even_when_commit_fails(self) -> None:
        """BUG-23: baseline_commit is set even when dirty-state commit fails."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            # Create a dirty file in the repo
            (repo / "dirty.txt").write_text("user dirty state\n")

            # Install a pre-commit hook that always rejects
            hooks_dir = repo / ".git" / "hooks"
            hooks_dir.mkdir(parents=True, exist_ok=True)
            hook = hooks_dir / "pre-commit"
            hook.write_text("#!/bin/sh\nexit 1\n")
            hook.chmod(0o755)

            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "test23"

            wt_work = agent._try_setup_worktree(repo, str(repo))
            assert wt_work is not None

            wt = agent._wt
            assert wt is not None

            # The dirty file was copied to the worktree
            assert (wt.wt_dir / "dirty.txt").exists(), (
                "dirty.txt should be copied to worktree"
            )

            # BUG-23: baseline_commit is set even though the commit failed.
            # It's set to the HEAD SHA (creation-time HEAD), not to a
            # commit that contains the dirty state.
            assert wt.baseline_commit is not None, (
                "baseline_commit is set (this is the bug — it's the wrong SHA)"
            )

            # The baseline should be the original HEAD (no dirty-state commit)
            original_head = _git(
                "rev-parse", "HEAD", cwd=repo,
            ).stdout.strip()

            # BUG-23: baseline_commit == original HEAD, not a commit with
            # the dirty state.  This means squash_merge_from_baseline will
            # cherry-pick from the wrong point.
            assert wt.baseline_commit == original_head, (
                "BUG-23 appears fixed: baseline is no longer the creation HEAD"
            )

        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_try_setup_does_not_check_commit_return(self) -> None:
        """BUG-23: Confirm _try_setup_worktree ignores commit_staged return."""
        source = inspect.getsource(WorktreeSorcarAgent._try_setup_worktree)
        # Find the baseline commit block
        # The code does:
        #   GitWorktreeOps.commit_staged(wt_dir, "kiss: baseline from dirty state")
        #   baseline_commit = GitWorktreeOps.head_sha(wt_dir)
        # without checking if commit_staged returned True.
        assert "commit_staged(" in source, "sanity: commit_staged is called"

        # Count how many times commit_staged's return is used in a condition
        lines = source.splitlines()
        for i, line in enumerate(lines):
            region = source[source.index(line):source.index(line) + 200]
            if "commit_staged(" in line and "baseline" in region:
                # Check if the return value is captured/checked
                stripped = line.strip()
                # BUG-23: the return value is NOT captured (no "if" or "=")
                assert not stripped.startswith("if "), (
                    "BUG-23 appears fixed: commit_staged return is now checked"
                )
                assert "= GitWorktreeOps.commit_staged" not in stripped, (
                    "BUG-23 appears fixed: commit_staged return is now captured"
                )
                break


# ===================================================================
# BUG-24: silent discard on transient git-diff failure
# ===================================================================


class TestBug24SilentDiscardOnGitFailure:
    """BUG-24: _get_worktree_changed_files returns [] when git diff fails
    (returncode != 0).  The caller in _run_task_inner then calls
    tab.agent.discard(), silently throwing away agent work.

    A transient git failure (lock file, disk issue) should not cause
    data loss.
    """

    def test_get_changed_files_returns_empty_on_diff_failure(self) -> None:
        """BUG-24: Confirm _get_worktree_changed_files returns [] on failure."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            agent = WorktreeSorcarAgent("test")
            agent._chat_id = "test24"
            wt_work = agent._try_setup_worktree(repo, str(repo))
            assert wt_work is not None

            wt = agent._wt
            assert wt is not None

            # Agent makes real changes
            (wt.wt_dir / "important.txt").write_text("agent work\n")
            GitWorktreeOps.commit_all(wt.wt_dir, "important agent changes")

            # Verify changes exist before sabotage
            server = VSCodeServer()
            server.work_dir = str(repo)
            tab = server._get_tab("t24")
            tab.agent = agent
            tab.use_worktree = True

            changed_before = server._get_worktree_changed_files("t24")
            assert "important.txt" in changed_before, (
                "important.txt should be in changed files"
            )

            # Now make git diff fail by corrupting the baseline ref.
            # Set a non-existent baseline to force git diff to fail.
            agent._wt = GitWorktree(
                repo_root=wt.repo_root,
                branch=wt.branch,
                original_branch=wt.original_branch,
                wt_dir=wt.wt_dir,
                baseline_commit="0000000000000000000000000000000000000000",
            )

            changed_after = server._get_worktree_changed_files("t24")

            # BUG-24: returns [] because git diff failed, even though
            # the agent has committed real changes.
            # Once fixed, this should either return the real changed
            # files or raise an error instead of silently returning [].
            assert changed_after == [], (
                "BUG-24 appears fixed: no longer returns [] on diff failure"
            )

        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_caller_discards_on_empty_changed_files(self) -> None:
        """BUG-24: Confirm _run_task_inner calls discard() when changed=[]."""
        # Verify by examining the source code of _run_task_inner
        source = inspect.getsource(VSCodeServer._run_task_inner)
        # The pattern is:
        #   changed = self._get_worktree_changed_files(tab_id)
        #   if changed:
        #       ...
        #   else:
        #       tab.agent.discard()
        assert "discard()" in source, (
            "sanity: _run_task_inner calls discard()"
        )
        # Confirm the discard is in the else branch of changed check
        lines = source.splitlines()
        found_pattern = False
        for i, line in enumerate(lines):
            if "tab.agent.discard()" in line:
                # Look backwards for the else clause
                for j in range(i - 1, max(i - 5, 0), -1):
                    if "else:" in lines[j]:
                        found_pattern = True
                        break
        assert found_pattern, (
            "discard() is called in else branch of changed-files check"
        )
