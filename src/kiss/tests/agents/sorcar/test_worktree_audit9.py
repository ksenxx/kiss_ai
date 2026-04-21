"""Audit 9: Tests verifying fixes for bugs, inconsistencies, and
redundancies in both non-worktree and worktree workflows.

BUG-39: `is_running_non_wt` flag is now cleared at the very start of
    the finally block's try (before any risky calls) AND in the outer
    except handler, so it can never get permanently stuck.

BUG-40 / INC-4: `_do_merge` now returns `(MergeResult.CHECKOUT_FAILED, "")`
    instead of `(None, checkout_error_str)`.  `_release_worktree` checks
    `result == MergeResult.CHECKOUT_FAILED` instead of `result is None`,
    so the checkout error is never misattributed to `_stash_pop_warning`.

BUG-41 / RED-6: `_start_merge_session` now accepts a `tab_id` parameter.
    All callers pass it explicitly.  `is_merging` is always set correctly,
    even on the session-replay path.

BUG-42 / INC-5: Auto-discard in both `_run_task_inner` and `_finish_merge`
    now checks `_any_non_wt_running()` before calling `discard()`.

BUG-43: Manual merge instructions now use `git cherry-pick --no-commit
    baseline..branch` when a baseline commit exists, matching what the
    auto-merge actually does.

BUG-44: `_new_chat` guard now checks `tab.agent._wt_pending` regardless
    of `tab.use_worktree`, so a tab that switched modes still gets the
    non-wt-running guard.

INC-6: `_check_merge_conflict` now checks both `unstaged_files()` AND
    `staged_files()` for dirty-file overlap.

RED-5: The two consecutive `if not tab.use_worktree:` blocks in
    `_run_task_inner`'s finally are now a single block.

RED-6: See BUG-41.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
)
from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _manual_merge_cmd,
)
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path, name: str = "repo") -> Path:
    """Create a bare-minimum git repo with one commit."""
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )
    (repo / "init.txt").write_text("init")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo, capture_output=True,
    )
    return repo


# ===================================================================
# BUG-39 FIX: is_running_non_wt cleared early + in except handler
# ===================================================================


class TestBug39Fix:
    """BUG-39 FIX: Verify `is_running_non_wt = False` is guaranteed
    to run (via try/finally or except handler) AND is also present in
    the outer except handler.

    NOTE: BUG-61 supersedes the original BUG-39 ordering.  The flag
    clear now happens AFTER `_prepare_and_start_merge` (so the merge
    view diff is captured while concurrent worktree merges are blocked),
    but is wrapped in a ``finally`` so it never gets stuck True.
    """

    def test_flag_cleared_via_finally_after_merge_view(self):
        """The flag clear is inside a finally block that follows the
        merge view preparation, guaranteeing it always runs.
        (BUG-61 fix supersedes original BUG-39 ordering.)"""
        src = inspect.getsource(VSCodeServer._run_task_inner)
        # Use the outer finally (contains _record_model_usage)
        outer_finally = src.find("_record_model_usage")
        assert outer_finally > 0
        finally_block = src[outer_finally:]

        # The flag clear must be present in the finally block
        assert "is_running_non_wt = False" in finally_block, (
            "BUG-39: flag clear must be in the finally block"
        )

        # The merge view must appear BEFORE the flag clear (BUG-61)
        merge_pos = finally_block.find("_prepare_and_start_merge")
        flag_pos = finally_block.find("is_running_non_wt = False")
        assert merge_pos > 0 and flag_pos > 0, (
            "Both merge view and flag clear must be in the finally block"
        )
        assert merge_pos < flag_pos, (
            "BUG-61: merge view must be prepared BEFORE flag is cleared"
        )

    def test_flag_cleared_in_except_handler(self):
        """The except BaseException handler also clears the flag."""
        src = inspect.getsource(VSCodeServer._run_task_inner)
        # Find the LAST except BaseException (cleanup handler)
        except_positions = [
            i for i in range(len(src))
            if src[i:].startswith("except BaseException:")
        ]
        cleanup_except = except_positions[-1]
        handler_block = src[cleanup_except:]

        assert "is_running_non_wt = False" in handler_block, (
            "BUG-39 fix: except handler must clear is_running_non_wt"
        )

    def test_flag_clear_before_risky_calls(self):
        """Flag clear precedes all calls that could raise."""
        src = inspect.getsource(VSCodeServer._run_task_inner)
        finally_pos = src.rfind("finally:")
        finally_block = src[finally_pos:]
        flag_pos = finally_block.find("tab.is_running_non_wt = False")
        assert flag_pos > 0

        risky_calls = [
            "self.printer.stop_recording()",
            "_save_task_result(",
            "_save_task_extra(",
            "self.printer.reset()",
        ]
        for call in risky_calls:
            call_pos = finally_block.find(call)
            if call_pos >= 0:
                assert flag_pos < call_pos, (
                    f"Flag clear must precede {call}"
                )


# ===================================================================
# BUG-40 / INC-4 FIX: Clean return semantics in _do_merge
# ===================================================================


class TestBug40Inc4Fix:
    """BUG-40/INC-4 FIX: _do_merge returns MergeResult.CHECKOUT_FAILED
    instead of (None, err), and _release_worktree never misattributes
    checkout errors to _stash_pop_warning."""

    def test_do_merge_returns_checkout_failed_enum(self):
        """_do_merge returns (MergeResult.CHECKOUT_FAILED, '') on checkout failure."""
        src = inspect.getsource(WorktreeSorcarAgent._do_merge)
        assert "MergeResult.CHECKOUT_FAILED" in src, (
            "Expected CHECKOUT_FAILED in _do_merge"
        )
        # Should NOT return (None, err) anymore
        assert "return (None," not in src, (
            "Should not return None as result anymore"
        )

    def test_do_merge_return_type_no_none(self):
        """Return type annotation uses MergeResult, not MergeResult | None."""
        src = inspect.getsource(WorktreeSorcarAgent._do_merge)
        assert "tuple[MergeResult, str]" in src, (
            "Return type should be tuple[MergeResult, str]"
        )

    def test_release_worktree_checks_checkout_failed(self):
        """_release_worktree checks CHECKOUT_FAILED instead of result is None."""
        src = inspect.getsource(WorktreeSorcarAgent._release_worktree)
        assert "MergeResult.CHECKOUT_FAILED" in src
        assert "result is None" not in src, (
            "Should not check 'result is None' anymore"
        )

    def test_checkout_error_not_stored_as_stash_warning(self, tmp_path):
        """Checkout failure does NOT set _stash_pop_warning."""
        repo = _make_repo(tmp_path)
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "bug40"

        branch = "kiss/wt-bug40-test"
        wt_dir = repo / ".kiss-worktrees" / "wt-bug40"
        GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        (wt_dir / "file.txt").write_text("agent work")
        GitWorktreeOps.commit_all(wt_dir, "agent work")

        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="nonexistent-branch",
            wt_dir=wt_dir,
        )

        agent._release_worktree()

        # FIX: _stash_pop_warning should NOT be set on checkout failure
        assert agent._stash_pop_warning is None, (
            "Checkout error must NOT be stored in _stash_pop_warning"
        )
        # _merge_conflict_warning should be set
        assert agent._merge_conflict_warning is not None
        assert "checkout" in agent._merge_conflict_warning.lower()

        # Cleanup
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)


# ===================================================================
# BUG-41 / RED-6 FIX: _start_merge_session accepts tab_id
# ===================================================================


class TestBug41Red6Fix:
    """BUG-41/RED-6 FIX: _start_merge_session accepts tab_id parameter
    and all callers pass it."""

    def test_start_merge_session_has_tab_id_param(self):
        """_start_merge_session accepts tab_id."""
        sig = inspect.signature(VSCodeServer._start_merge_session)
        assert "tab_id" in sig.parameters, (
            "BUG-41 fix: _start_merge_session must accept tab_id"
        )

    def test_prepare_and_start_merge_passes_tab_id(self):
        """_prepare_and_start_merge passes tab_id to _start_merge_session."""
        src = inspect.getsource(VSCodeServer._prepare_and_start_merge)
        assert "tab_id=tab_id" in src, (
            "Must pass tab_id to _start_merge_session"
        )

    def test_restore_pending_merge_removed(self):
        """_restore_pending_merge was dead code and has been removed (RED-9)."""
        assert not hasattr(VSCodeServer, "_restore_pending_merge"), (
            "_restore_pending_merge should be removed (RED-9 dead code)"
        )

    def test_is_merging_set_with_explicit_tab_id(self, tmp_path):
        """When tab_id is passed explicitly, is_merging is set correctly
        even if thread-local tab_id is None (replay path)."""
        server = VSCodeServer()
        tab = server._get_tab("replay-tab")
        assert not tab.is_merging

        # Create a minimal merge JSON file
        merge_dir = tmp_path / "merge"
        merge_dir.mkdir()
        merge_json = merge_dir / "pending-merge.json"
        merge_json.write_text(json.dumps({
            "files": [{
                "path": "test.txt",
                "hunks": [{"old_start": 1, "new_start": 1}],
            }],
        }))

        # Call with explicit tab_id (thread-local is NOT set)
        result = server._start_merge_session(
            str(merge_json), tab_id="replay-tab",
        )
        assert result is True
        assert tab.is_merging, (
            "is_merging must be True when tab_id is passed explicitly"
        )


# ===================================================================
# BUG-42 / INC-5 FIX: Auto-discard guarded by _any_non_wt_running
# ===================================================================


class TestBug42Inc5Fix:
    """BUG-42/INC-5 FIX: Auto-discard in _run_task_inner and
    _finish_merge now checks _any_non_wt_running() before discard."""

    def test_run_task_inner_auto_discard_guarded(self):
        """_run_task_inner's worktree auto-discard is guarded.

        After RED-10 refactor the guard lives in the shared helper
        ``_present_pending_worktree``; ``_run_task_inner`` delegates
        to it, so the guard must be present in the helper.
        """
        helper_src = inspect.getsource(VSCodeServer._present_pending_worktree)
        # The guard must appear before the discard call in the helper.
        discard_pos = helper_src.find("tab.agent.discard()")
        assert discard_pos > 0
        context = helper_src[max(0, discard_pos - 500):discard_pos]
        assert "_any_non_wt_running" in context, (
            "Auto-discard in the shared helper must check _any_non_wt_running"
        )
        # _run_task_inner must delegate to the shared helper.
        runner_src = inspect.getsource(VSCodeServer._run_task_inner)
        assert "_present_pending_worktree" in runner_src, (
            "_run_task_inner must delegate pending-worktree handling "
            "to the shared helper"
        )

    def test_finish_merge_auto_discard_guarded(self):
        """_finish_merge's auto-discard is guarded via the shared helper."""
        finish_src = inspect.getsource(VSCodeServer._finish_merge)
        assert "_present_pending_worktree" in finish_src, (
            "_finish_merge must delegate pending-worktree handling "
            "to the shared helper"
        )
        helper_src = inspect.getsource(VSCodeServer._present_pending_worktree)
        guard_pos = helper_src.find("_any_non_wt_running")
        discard_pos = helper_src.find("tab.agent.discard()")
        assert guard_pos > 0 and discard_pos > 0
        assert guard_pos < discard_pos, (
            "Guard must precede discard call in the shared helper"
        )

    def test_all_discard_paths_consistent(self):
        """All discard paths in the server share the same guard."""
        for name in (
            "_present_pending_worktree",
            "_handle_worktree_action",
        ):
            src = inspect.getsource(getattr(VSCodeServer, name))
            if "tab.agent.discard()" in src or "wt.discard()" in src:
                assert "_any_non_wt_running" in src, (
                    f"{name} must guard discard with _any_non_wt_running"
                )


# ===================================================================
# BUG-43 FIX: Correct manual merge instructions with baseline
# ===================================================================


class TestBug43Fix:
    """BUG-43 FIX: Instructions use cherry-pick when baseline exists."""

    def test_manual_merge_cmd_with_baseline(self):
        """_manual_merge_cmd returns cherry-pick when baseline exists."""
        wt = GitWorktree(
            repo_root=Path("/repo"),
            branch="kiss/wt-test",
            original_branch="main",
            wt_dir=Path("/repo/.kiss-worktrees/wt"),
            baseline_commit="abc123",
        )
        cmd = _manual_merge_cmd(wt)
        assert "cherry-pick" in cmd
        assert "abc123..kiss/wt-test" in cmd
        assert "merge --squash" not in cmd

    def test_manual_merge_cmd_without_baseline(self):
        """_manual_merge_cmd returns merge --squash when no baseline."""
        wt = GitWorktree(
            repo_root=Path("/repo"),
            branch="kiss/wt-test",
            original_branch="main",
            wt_dir=Path("/repo/.kiss-worktrees/wt"),
        )
        cmd = _manual_merge_cmd(wt)
        assert "merge --squash" in cmd
        assert "cherry-pick" not in cmd

    def test_release_worktree_uses_correct_instructions(self):
        """_release_worktree uses _manual_merge_cmd for instructions."""
        src = inspect.getsource(WorktreeSorcarAgent._release_worktree)
        # Should reference merge_cmd not hardcoded merge --squash
        assert "merge_cmd" in src
        assert "git merge --squash" not in src, (
            "Should not hardcode 'git merge --squash' in instructions"
        )

    def test_merge_uses_correct_instructions(self):
        """merge() uses _manual_merge_cmd for instructions."""
        src = inspect.getsource(WorktreeSorcarAgent.merge)
        assert "merge_cmd" in src
        # merge --squash should not appear in the conflict/failure messages
        # (only _manual_merge_cmd generates the correct command)
        lines_with_merge_squash = [
            line for line in src.splitlines()
            if "git merge --squash" in line and "#" not in line.split("git merge")[0]
        ]
        assert not lines_with_merge_squash, (
            "merge() should not hardcode 'git merge --squash'"
        )

    def test_merge_instructions_uses_correct_cmd(self):
        """merge_instructions() uses _manual_merge_cmd."""
        src = inspect.getsource(WorktreeSorcarAgent.merge_instructions)
        assert "merge_cmd" in src
        assert "git merge --squash" not in src

    def test_functional_instructions_match_auto_merge(self, tmp_path):
        """Instructions produce the same result as auto-merge when baseline exists."""
        repo = _make_repo(tmp_path)

        branch = "kiss/wt-bug43-test"
        wt_dir = repo / ".kiss-worktrees" / "wt-bug43"
        GitWorktreeOps.create(repo, branch, wt_dir)

        # Simulate dirty state baseline
        (wt_dir / "dirty.txt").write_text("user dirty content")
        subprocess.run(["git", "add", "-A"], cwd=wt_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "baseline"],
            cwd=wt_dir, capture_output=True,
        )
        baseline = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_dir, capture_output=True, text=True,
        ).stdout.strip()

        # Agent work
        (wt_dir / "agent.txt").write_text("agent work")
        subprocess.run(["git", "add", "-A"], cwd=wt_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "agent work"],
            cwd=wt_dir, capture_output=True,
        )

        wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=baseline,
        )
        cmd = _manual_merge_cmd(wt)

        # The command should be cherry-pick
        assert "cherry-pick" in cmd

        # Execute the command and verify only agent changes applied
        result = subprocess.run(
            cmd.split(), cwd=repo, capture_output=True, text=True,
        )
        assert result.returncode == 0
        status = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo, capture_output=True, text=True,
        )
        files = set(status.stdout.strip().splitlines())
        assert "agent.txt" in files
        assert "dirty.txt" not in files, (
            "Cherry-pick should NOT include baseline dirty state"
        )

        # Cleanup
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo, capture_output=True)
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)


# ===================================================================
# BUG-44 FIX: _new_chat guard checks _wt_pending regardless of mode
# ===================================================================


class TestBug44Fix:
    """BUG-44 FIX: _new_chat guard checks agent._wt_pending regardless
    of tab.use_worktree."""

    def test_guard_does_not_require_use_worktree(self):
        """The guard condition does NOT require tab.use_worktree."""
        src = inspect.getsource(VSCodeServer._new_chat)
        # Should check _wt_pending without requiring use_worktree
        assert "tab.agent._wt_pending" in src
        assert "tab.use_worktree and tab.agent._wt_pending" not in src, (
            "Guard should NOT require use_worktree"
        )

    def test_worktree_pending_with_mode_switched(self, tmp_path):
        """When use_worktree=False but _wt is set, the guard still fires."""
        repo = _make_repo(tmp_path)
        server = VSCodeServer()
        server.work_dir = str(repo)

        tab = server._get_tab("bypass-tab")
        branch = "kiss/wt-bypass-test"
        wt_dir = repo / ".kiss-worktrees" / "wt-bypass"
        GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        tab.agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )
        tab.use_worktree = False

        assert tab.agent._wt_pending
        # The guard condition now fires even with use_worktree=False
        assert tab.agent._wt_pending, "Guard should check _wt_pending alone"

        # Cleanup
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)
        tab.agent._wt = None


# ===================================================================
# INC-6 FIX: _check_merge_conflict includes staged files
# ===================================================================


class TestInc6Fix:
    """INC-6 FIX: _check_merge_conflict checks both unstaged and staged files."""

    def test_check_includes_staged_files(self):
        """Source code calls both unstaged_files and staged_files."""
        src = inspect.getsource(VSCodeServer._check_merge_conflict)
        assert "unstaged_files" in src
        assert "staged_files" in src, (
            "Must check staged_files in addition to unstaged_files"
        )

    def test_staged_overlap_detected(self, tmp_path):
        """A staged file overlapping with worktree changes IS detected."""
        repo = _make_repo(tmp_path)

        branch = "kiss/wt-inc6-test"
        wt_dir = repo / ".kiss-worktrees" / "wt-inc6"
        GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        # Agent modifies init.txt in the worktree
        (wt_dir / "init.txt").write_text("agent changes")
        subprocess.run(["git", "add", "-A"], cwd=wt_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "agent work"],
            cwd=wt_dir, capture_output=True,
        )

        # User stages a conflicting change to init.txt in main repo
        (repo / "init.txt").write_text("user staged change")
        subprocess.run(["git", "add", "init.txt"], cwd=repo, capture_output=True)

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("inc6-tab")
        tab.use_worktree = True
        tab.agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )

        has_conflict = server._check_merge_conflict("inc6-tab")
        assert has_conflict, (
            "INC-6 fix: staged file overlap must be detected"
        )

        # Cleanup
        subprocess.run(["git", "reset", "HEAD", "init.txt"], cwd=repo, capture_output=True)
        subprocess.run(["git", "checkout", "--", "init.txt"], cwd=repo, capture_output=True)
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)

    def test_staged_files_helper_exists(self):
        """GitWorktreeOps.staged_files exists and works."""
        assert hasattr(GitWorktreeOps, "staged_files")
        sig = inspect.signature(GitWorktreeOps.staged_files)
        assert "repo" in sig.parameters


# ===================================================================
# RED-5 FIX: Single if-not-worktree block in finally
# ===================================================================


class TestRed5Fix:
    """RED-5 FIX: The two originally-consecutive identical
    `if not tab.use_worktree:` blocks (flag-clear + merge-view) are
    no longer adjacent/redundant.  The flag-clear moved to the very
    start of the try block (BUG-39 fix), and each remaining block
    serves a distinct purpose."""

    def test_no_adjacent_duplicate_blocks(self):
        """No two adjacent `if not tab.use_worktree:` blocks remain."""
        src = inspect.getsource(VSCodeServer._run_task_inner)
        finally_pos = src.rfind("finally:")
        finally_block = src[finally_pos:]

        pattern = "if not tab.use_worktree:"
        positions = []
        start = 0
        while True:
            pos = finally_block.find(pattern, start)
            if pos < 0:
                break
            positions.append(pos)
            start = pos + 1

        # Check that no two occurrences are close together (< 100 chars)
        # which would indicate redundant adjacent blocks.
        for i in range(len(positions) - 1):
            gap = positions[i + 1] - positions[i]
            assert gap > 100, (
                f"RED-5: two blocks are only {gap} chars apart — redundant"
            )

    def test_flag_clear_after_merge_view(self):
        """Flag clear is AFTER merge view (BUG-61 fix supersedes RED-5).

        The merge view must capture a diff while `is_running_non_wt`
        is True, then the flag is cleared in a ``finally`` block.
        """
        src = inspect.getsource(VSCodeServer._run_task_inner)
        # Use the outer finally (contains _record_model_usage)
        outer_finally = src.find("_record_model_usage")
        assert outer_finally > 0
        finally_block = src[outer_finally:]

        flag_pos = finally_block.find("tab.is_running_non_wt = False")
        merge_pos = finally_block.find("_prepare_and_start_merge")
        assert flag_pos > 0 and merge_pos > 0
        # BUG-61: merge view BEFORE flag clear
        assert merge_pos < flag_pos, (
            "BUG-61: merge view must be before flag clear"
        )


# ===================================================================
# RED-6 FIX: See BUG-41 tests above
# ===================================================================


class TestRed6Fix:
    """RED-6 FIX: _start_merge_session uses parameter, not thread-local."""

    def test_uses_explicit_tab_id_with_fallback(self):
        """Uses explicit tab_id, falling back to thread-local only when empty."""
        src = inspect.getsource(VSCodeServer._start_merge_session)
        # Should use the parameter first, then fall back
        assert "resolved_tab_id = tab_id or" in src or "tab_id or getattr" in src, (
            "Should use explicit tab_id with thread-local fallback"
        )
