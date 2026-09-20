# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a worktree merge refused because another tab's
non-worktree task occupies the main working tree is merged
AUTOMATICALLY once that task's changes are committed.

Before, the refusal message told the user to "open a new chat to auto
merge" and nothing ever retried.  Now the refusal records the pending
branch on the tab (``AgentState.wt_merge_deferred_branch``) and
``_merge_deferred_worktrees`` retries the merge when the main tree is
committed again:

* at the end of a non-worktree task (its post-task auto-commit
  included);
* after a successful manual Git Commit;
* after the main-tree bar's Discard cleaned the tree.

A dirty main tree (the task ran with auto-commit off and the user has
not decided yet) keeps the worktree waiting.

Every test drives the real ``VSCodeServer`` post-task flow against a
real temp git repo; only the LLM agent body and the LLM commit-message
generator are replaced with deterministic functions.

Not covered here: the ``stranded_repo`` backstop in ``_run_task``'s
outer ``finally`` (task_runner.py).  It fires only when
``_run_task_inner`` crashes between recording the main-tree occupancy
and its own mandatory cleanup — a window with no failure injection
point short of patching the runner itself, which these tests do not
do.  On every normal path the inner cleanup has already released the
occupancy, so the backstop is a no-op (``stranded_repo is None``).
"""

from __future__ import annotations

import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as _persistence
from kiss.server import agent_state
from kiss.tests.server.test_worktree_no_autocommit_branch import (
    _list_kiss_wt_branches,
    _patch_parent_run_create_file,
    _run_git,
)
from kiss.tests.server.test_worktree_repo_aware_busy_guard import (
    _OTHER_TAB,
    _WT_TAB,
    _RepoAwareGuardBase,
)

#: Tab that runs the real non-worktree task whose commit frees the tree.
_DIRECT_TAB = "direct-task-tab"

_AUTO_SENTENCE = (
    "The worktree will be merged automatically once the parent branch "
    "has been committed."
)


class _DeferredMergeBase(_RepoAwareGuardBase):
    """Harness: strand a worktree behind a busy main tree, then free it."""

    def _strand_worktree(self) -> None:
        """Run a worktree task while a task occupies the main tree.

        Leaves ``_WT_TAB`` holding a pending, unmerged worktree whose
        auto-merge the guard refused, and the occupant tab released.
        """
        self._mark_non_wt_task(Path(self.repo))
        self._run_worktree_task_with_changes()
        results = self._worktree_results()
        assert results and not results[-1]["success"], results
        self._unmark_non_wt_task()
        self.events.clear()

    def _unmark_non_wt_task(self) -> None:
        with self.server._state_lock:
            state = agent_state.find_by_tab(_OTHER_TAB)
            assert state is not None
            state.is_task_active = False
            state.is_running_non_wt = False
            state.non_wt_repo_root = None

    def _wt_state(self) -> agent_state.AgentState:
        state = agent_state.find_by_tab(_WT_TAB)
        assert state is not None and state.agent is not None
        return state

    def _run_direct_task(self, *, auto_commit: bool) -> None:
        """Run a real non-worktree task in the main tree of ``self.repo``."""
        # Re-patching: the FIRST patch (in ``_strand_worktree``) saved
        # the real ``run`` for tearDown; keep that, not this stub.
        _patch_parent_run_create_file("direct_out.txt")
        self.server._run_task_inner({
            "prompt": "direct task on the main tree",
            "workDir": self.repo,
            "tabId": _DIRECT_TAB,
            "useWorktree": False,
            "autoCommit": auto_commit,
            "model": "",
        })
        with self.server._state_lock:
            state = agent_state.find_by_tab(_DIRECT_TAB)
            if state is not None and (
                state.task_thread is threading.current_thread()
            ):
                state.task_thread = None

    def _assert_merged(self) -> None:
        results = [
            e for e in self._worktree_results() if e.get("tabId") == _WT_TAB
        ]
        assert results and results[-1]["success"], (
            f"the deferred merge must succeed and report to the tab: {results}"
        )
        assert (Path(self.repo) / "agent_out.txt").exists(), (
            "the merged file must land in the main working tree"
        )
        assert _list_kiss_wt_branches(self.repo) == [], (
            "the merged task branch must be deleted"
        )
        state = self._wt_state()
        assert state.agent is not None and not state.agent._wt_pending
        assert state.wt_merge_deferred_branch is None

    def _assert_still_waiting(self) -> None:
        assert not self._worktree_results(), (
            "no merge may run or be reported while the tree is dirty"
        )
        assert not (Path(self.repo) / "agent_out.txt").exists()
        state = self._wt_state()
        assert state.agent is not None and state.agent._wt_pending
        assert state.wt_merge_deferred_branch == state.agent._wt_branch


class TestRefusalRecordsDeferral(_DeferredMergeBase):
    """Both refusal paths announce and record the automatic merge."""

    def test_post_task_refusal_message_and_marker(self) -> None:
        self._mark_non_wt_task(Path(self.repo))
        self._run_worktree_task_with_changes()

        results = self._worktree_results()
        assert results and not results[-1]["success"], results
        message = results[-1]["message"]
        assert "main working tree" in message, message
        assert _AUTO_SENTENCE in message, message
        assert "open a new chat" not in message, message
        state = self._wt_state()
        assert state.agent is not None
        assert state.wt_merge_deferred_branch == state.agent._wt_branch

    def test_manual_merge_click_refusal_message_and_marker(self) -> None:
        self._strand_worktree()
        # Forget the deferral so the manual click has to record it.
        with self.server._state_lock:
            self._wt_state().wt_merge_deferred_branch = None
        self._mark_non_wt_task(Path(self.repo))

        result = self.server._handle_worktree_action("merge", _WT_TAB)

        assert not result["success"], result
        assert _AUTO_SENTENCE in result["message"], result
        state = self._wt_state()
        assert state.agent is not None
        assert state.wt_merge_deferred_branch == state.agent._wt_branch

    def test_manual_discard_refusal_does_not_defer(self) -> None:
        """Only a MERGE is retried later; the occupant does not block a
        discard, so the internal path cannot even refuse it, and the
        manual refusal of a discard promises nothing."""
        self._strand_worktree()
        with self.server._state_lock:
            self._wt_state().wt_merge_deferred_branch = None
        self._mark_non_wt_task(Path(self.repo))

        result = self.server._handle_worktree_action("discard", _WT_TAB)

        assert not result["success"], result
        assert _AUTO_SENTENCE not in result["message"], result
        assert self._wt_state().wt_merge_deferred_branch is None


class TestDirectTaskEndTriggersMerge(_DeferredMergeBase):
    """The scenario from the bug report, end to end."""

    def test_committed_direct_task_merges_the_waiting_worktree(self) -> None:
        self._strand_worktree()

        self._run_direct_task(auto_commit=True)

        # The direct task's own auto-commit landed first...
        log = _run_git(self.repo, "log", "--format=%s").stdout
        assert "test: deterministic commit message" in log, log
        assert (Path(self.repo) / "direct_out.txt").exists()
        # ...and the waiting worktree was merged right after it.
        self._assert_merged()
        types = self._types()
        assert types.index("worktree_result") > types.index("task_done"), (
            "the deferred merge must be reported after the direct "
            f"task's own end event: {types}"
        )

    def test_uncommitted_direct_task_keeps_the_worktree_waiting(self) -> None:
        """Auto-commit off: the main tree stays dirty, so the merge
        must wait for the user's commit/discard decision."""
        self._strand_worktree()

        self._run_direct_task(auto_commit=False)

        assert (Path(self.repo) / "direct_out.txt").exists()
        assert self.server._main_dirty_files(self.repo), "precondition"
        self._assert_still_waiting()

    def test_merge_runs_even_when_post_commit_cleanup_crashes(self) -> None:
        """The trigger lives in the run's mandatory-cleanup ``finally``:
        a persistence crash AFTER the direct task's commit (the DB path
        is turned into a directory, so every persistence call raises a
        real ``sqlite3.OperationalError`` and the run's normal
        post-task path is abandoned) must not strand the promised
        merge."""
        self._strand_worktree()
        bad = Path(self.tmpdir) / "db-as-directory"
        bad.mkdir()

        def run_then_break_db(self_agent: object, **kwargs: object) -> str:
            # The sabotage happens INSIDE the run: the task row already
            # exists, so the run reaches its post-task commit and only
            # the persistence that follows the commit crashes.
            work_dir = kwargs.get("work_dir")
            assert isinstance(work_dir, str)
            (Path(work_dir) / "direct_out.txt").write_text("agent output\n")
            if _persistence._db_conn is not None:
                _persistence._db_conn.close()
                _persistence._db_conn = None
            _persistence._DB_PATH = bad
            return "success: true\nsummary: stub\n"

        self._parent_class.run = run_then_break_db
        self.server._run_task_inner({
            "prompt": "direct task whose cleanup persistence crashes",
            "workDir": self.repo,
            "tabId": _DIRECT_TAB,
            "useWorktree": False,
            "autoCommit": True,
            "model": "",
        })

        log = _run_git(self.repo, "log", "--format=%s").stdout
        assert "test: deterministic commit message" in log, log
        self._assert_merged()

    def test_direct_task_in_other_repo_does_not_merge(self) -> None:
        self._strand_worktree()
        # Re-patching: the FIRST patch (in ``_strand_worktree``) saved
        # the real ``run`` for tearDown; keep that, not this stub.
        _patch_parent_run_create_file("direct_out.txt")
        self.server._run_task_inner({
            "prompt": "direct task elsewhere",
            "workDir": self.other_repo,
            "tabId": _DIRECT_TAB,
            "useWorktree": False,
            "autoCommit": True,
            "model": "",
        })

        assert (Path(self.other_repo) / "direct_out.txt").exists()
        self._assert_still_waiting()


class TestManualCommitAndDiscardTriggerMerge(_DeferredMergeBase):
    """The main-tree bar's decisions after an uncommitted direct task."""

    def test_manual_git_commit_merges_the_waiting_worktree(self) -> None:
        self._strand_worktree()
        self._run_direct_task(auto_commit=False)
        self._assert_still_waiting()
        self.events.clear()

        self.server._run_autocommit_job(_DIRECT_TAB, self.repo, Path(self.repo))

        assert not self.server._main_dirty_files(self.repo)
        self._assert_merged()
        assert (Path(self.repo) / "direct_out.txt").exists()

    def test_main_tree_discard_merges_the_waiting_worktree(self) -> None:
        self._strand_worktree()
        self._run_direct_task(auto_commit=False)
        self._assert_still_waiting()
        self.events.clear()

        self.server._cmd_main_tree_action({
            "action": "discard", "tabId": _DIRECT_TAB, "workDir": self.repo,
        })

        main_results = [e for e in self.events if e["type"] == "main_tree_result"]
        assert main_results and main_results[-1]["success"], main_results
        assert not (Path(self.repo) / "direct_out.txt").exists()
        self._assert_merged()

    def test_main_tree_do_nothing_keeps_the_worktree_waiting(self) -> None:
        self._strand_worktree()
        self._run_direct_task(auto_commit=False)
        self.events.clear()

        self.server._cmd_main_tree_action({
            "action": "nothing", "tabId": _DIRECT_TAB, "workDir": self.repo,
        })

        main_results = [e for e in self.events if e["type"] == "main_tree_result"]
        assert main_results and main_results[-1]["success"], main_results
        self._assert_still_waiting()

    def test_failed_discard_does_not_trigger(self) -> None:
        """A discard that fails (here: the tab's folder is not a git
        repository) committed nothing and must not trigger."""
        self._strand_worktree()

        self.server._cmd_main_tree_action({
            "action": "discard", "tabId": _DIRECT_TAB, "workDir": self.tmpdir,
        })

        main_results = [e for e in self.events if e["type"] == "main_tree_result"]
        assert main_results and not main_results[-1]["success"], main_results
        self._assert_still_waiting()


class TestDeferredMergeGuards(_DeferredMergeBase):
    """``_merge_deferred_worktrees`` itself: candidates and early refusals."""

    def test_none_repo_is_a_noop(self) -> None:
        self._strand_worktree()
        self.server._merge_deferred_worktrees(None)
        self._assert_still_waiting()

    def test_unreadable_status_fails_closed(self) -> None:
        """When ``git status`` itself fails (corrupt index) the tree is
        not KNOWN to be committed: nothing runs, the deferral stands,
        and the next trigger after recovery merges."""
        self._strand_worktree()
        index = Path(self.repo) / ".git" / "index"
        good_index = index.read_bytes()
        index.write_bytes(b"garbage")
        assert _run_git(self.repo, "status", "--porcelain").returncode != 0

        self.server._merge_deferred_worktrees(Path(self.repo))
        self._assert_still_waiting()

        index.write_bytes(good_index)
        self.server._merge_deferred_worktrees(Path(self.repo))
        self._assert_merged()

    def test_guard_refusal_keeps_deferral_and_reports_nothing(self) -> None:
        """The tab started a new task before the trigger fired: the
        merge is refused by the busy guard, nothing is broadcast, and
        the deferral stands until the next trigger."""
        self._strand_worktree()
        state = self._wt_state()
        with self.server._state_lock:
            state.is_task_active = True

        self.server._merge_deferred_worktrees(Path(self.repo))
        self._assert_still_waiting()

        with self.server._state_lock:
            state.is_task_active = False
        self.server._merge_deferred_worktrees(Path(self.repo))
        self._assert_merged()

    def test_stale_marker_never_targets_a_later_worktree(self) -> None:
        """The marker is keyed by branch: a worktree released outside
        ``_handle_worktree_action`` (a new chat's retirement) leaves it
        behind, and the tab's NEXT pending worktree must not be merged
        on the strength of it."""
        self._strand_worktree()
        state = self._wt_state()
        assert state.agent is not None
        stale_branch = state.wt_merge_deferred_branch
        state.agent.discard()  # what ``_retire_previous_worktree`` does
        assert not state.agent._wt_pending
        assert state.wt_merge_deferred_branch == stale_branch

        # A new worktree task with auto-commit OFF leaves its worktree
        # pending for the user's decision.
        _patch_parent_run_create_file("second_out.txt")
        self.server._run_task_inner({
            "prompt": "second worktree task",
            "workDir": self.repo,
            "tabId": _WT_TAB,
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })
        with self.server._state_lock:
            if state.task_thread is threading.current_thread():
                state.task_thread = None
        assert state.agent._wt_pending
        assert state.agent._wt_branch != stale_branch
        self.events.clear()

        self._run_direct_task(auto_commit=True)

        assert state.agent._wt_pending, (
            "a stale marker must not merge the user's undecided worktree"
        )
        assert not (Path(self.repo) / "second_out.txt").exists()
        assert not [
            e for e in self._worktree_results() if e.get("tabId") == _WT_TAB
        ]

    def test_worktree_action_clears_marker(self) -> None:
        """Any action that owns the worktree ends the deferral."""
        self._strand_worktree()
        result = self.server._handle_worktree_action("discard", _WT_TAB)
        assert result["success"], result
        assert self._wt_state().wt_merge_deferred_branch is None
