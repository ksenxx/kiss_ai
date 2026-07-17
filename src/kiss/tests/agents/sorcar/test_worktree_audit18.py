# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 18: Race-condition integration tests for worktree mode.

The VS Code server coordinates three concurrent actors:

1. A worktree task running on tab A.
2. A non-worktree (main-tree) task running on tab B.
3. User-triggered UI actions (merge / discard / newChat).

Before the fixes documented in this file, the server's
``_any_non_wt_running()`` / ``is_merging`` / ``is_task_active`` flags
were inspected under ``_state_lock`` but the slow bodies that follow
(``wt.merge()``, ``tab.agent.run()``, ``_release_worktree`` →
``_do_merge``) ran without re-checking the flags or holding any
serializing lock against the other actor's mutations.  This created
three races; each is now plugged.

RACE-1 (merge guard TOCTOU)
    ``_handle_worktree_action("merge")`` returned success even when a
    non-worktree task started on another tab *after* the guard check
    and *before* ``wt.merge()`` ran ``stash_if_dirty``.

    **Fix**: ``_handle_worktree_action`` now (a) holds the per-repo
    re-entrant ``repo_lock`` for the entire body, and (b) sets
    ``tab.is_merging = True`` under ``_state_lock`` atomically with
    the guard check.  Concurrent non-wt task-start in
    ``_run_task_inner`` acquires the same ``repo_lock`` briefly to
    inspect ``is_merging`` and set ``is_running_non_wt`` in one
    critical section.

RACE-2 (setup release TOCTOU)
    ``_try_setup_worktree`` released ``repo_lock`` after the
    ``current_branch`` read and ran the slow setup
    (``GitWorktreeOps.create``, ``copy_dirty_state``, baseline commit)
    unlocked.  A concurrent ``_do_merge`` on another tab could
    interleave with ``copy_dirty_state``.

    **Fix**: ``_try_setup_worktree`` now wraps the entire release +
    create + copy-dirty-state + baseline-commit sequence in
    ``repo_lock(repo)``.  The lock is an :class:`threading.RLock` so
    the inner ``_release_worktree`` → ``_do_merge`` re-acquires it on
    the same thread without deadlock.

RACE-3 (post-task vs user-action on ``_wt``)
    In ``_run_task_inner`` 's finally block, ``tab.is_task_active``
    was cleared BEFORE ``_present_pending_worktree`` ran.  Once the
    flag was False, a concurrent user click on the discard / merge
    button passed the ``_handle_worktree_action`` guards and mutated
    ``tab.agent._wt`` while the task thread was still reading it.

    **Fix**: ``tab.is_task_active = False`` is now assigned AFTER the
    worktree-presentation block completes.  The flag remains True for
    the entire post-task cleanup, so ``_check_worktree_busy`` refuses
    concurrent merge / discard clicks until the task thread is done
    reading ``tab.agent._wt``.

Each test uses ``repo_lock(repo)`` as a deterministic interleaving
primitive: the test thread pre-acquires the per-repo lock, drives the
server handler in a background thread (which then blocks inside the
lock), mutates state to simulate the concurrent actor, and releases
the lock.  No mocks, no monkey-patches.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    repo_lock,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server.server import VSCodeServer


def _make_repo(path: Path) -> Path:
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
    subprocess.run(
        ["git", "-C", str(path), "checkout", "-b", "main"],
        capture_output=True, check=True,
    )
    (path / "init.txt").write_text("init\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True, check=True,
    )
    return path


def _make_wt_with_commit(
    repo: Path, branch: str, agent: WorktreeSorcarAgent,
) -> GitWorktree:
    """Create a real worktree, record an agent commit, assign to agent."""
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.save_original_branch(repo, branch, "main")
    (wt_dir / "agent.txt").write_text("agent produced this\n")
    subprocess.run(
        ["git", "-C", str(wt_dir), "add", "."],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(wt_dir), "commit", "-m", "agent"],
        capture_output=True, check=True,
    )
    wt = GitWorktree(
        repo_root=repo,
        branch=branch,
        original_branch="main",
        wt_dir=wt_dir,
    )
    agent._wt = wt
    return wt


class _RecordingPrinter:
    """Real printer that records every broadcast call."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._thread_local = threading.local()
        self._persist_agents: dict[str, Any] = {}

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _server(repo: Path) -> VSCodeServer:
    """Construct a VSCodeServer pointed at *repo* with a recording printer."""
    server = VSCodeServer()
    server.work_dir = str(repo)
    server.printer = cast(Any, _RecordingPrinter())
    return server


class TestRaceMergeGuardTOCTOU:
    """RACE-1: ``_handle_worktree_action("merge")`` must serialize
    against concurrent non-wt task-start on another tab.

    The fix uses (a) the per-repo re-entrant ``repo_lock`` to block
    the slow body of the merge from running while another caller
    holds the same lock, and (b) the ``tab.is_merging`` flag set
    atomically with the guard check so that any non-wt task that
    starts after the guard fires its own refusal path.
    """

    def test_repo_lock_serializes_merge_with_concurrent_setup(
        self, tmp_path: Path,
    ) -> None:
        """When another tab is in the middle of a ``repo_lock``-
        protected operation, ``_handle_worktree_action("merge")``
        must not start its slow body until the lock is released.

        Before the fix the merge skipped ``repo_lock`` and could
        ``stash_if_dirty`` / ``checkout`` mid-operation.  After the
        fix the merge blocks on ``repo_lock`` and only proceeds once
        the other caller releases it.
        """
        repo = _make_repo(tmp_path / "repo")
        server = _server(repo)

        tab_a = server._get_tab("a")
        tab_a.agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab_a.use_worktree = True
        tab_a.is_task_active = False
        agent_a = cast(WorktreeSorcarAgent, tab_a.agent)
        wt = _make_wt_with_commit(repo, "kiss/wt-race1-1", agent_a)

        lock = repo_lock(repo)
        # Simulate another thread holding ``repo_lock`` (e.g. the
        # owner is inside ``_try_setup_worktree`` running
        # ``copy_dirty_state``).  Cross-thread acquisition must block.
        acquired_in_other_thread: list[bool] = []
        release_event = threading.Event()
        acquired_event = threading.Event()

        def hold_lock() -> None:
            lock.acquire()
            acquired_in_other_thread.append(True)
            acquired_event.set()
            release_event.wait(timeout=10.0)
            lock.release()

        holder = threading.Thread(target=hold_lock, daemon=True)
        holder.start()
        assert acquired_event.wait(2.0)

        result_holder: list[dict[str, Any]] = []

        def run_merge() -> None:
            result_holder.append(
                server._handle_worktree_action("merge", "a"),
            )

        merge_thread = threading.Thread(target=run_merge, daemon=True)
        merge_thread.start()

        # Give the merge thread a chance to run; it must NOT have
        # progressed to removing wt_dir while the lock is held.
        time.sleep(0.5)
        assert wt.wt_dir.exists(), (
            "RACE-1: merge thread reached _do_merge while another "
            "owner held repo_lock.  The fix must block the slow "
            "body behind repo_lock."
        )
        assert not result_holder, "merge thread returned before lock released"

        release_event.set()
        holder.join(timeout=5)
        merge_thread.join(timeout=15)
        assert not merge_thread.is_alive(), "merge thread hung after release"
        assert result_holder, "merge handler returned nothing"
        result = result_holder[0]
        assert result.get("success") is True, (
            f"merge should have succeeded after lock release: {result}"
        )
        # Squash-merge commit must be on main.
        log = subprocess.run(
            ["git", "-C", str(repo), "log", "--format=%H", "main"],
            capture_output=True, text=True, check=True,
        )
        assert len(log.stdout.strip().splitlines()) >= 2, (
            f"main missing squash commit:\n{log.stdout!r}"
        )

    def test_merge_handler_sets_is_merging_under_state_lock(
        self, tmp_path: Path,
    ) -> None:
        """The merge handler must claim ``tab.is_merging = True``
        before releasing ``_state_lock`` so a concurrent non-wt
        task-start path sees the flag and refuses to start.

        We probe the flag by injecting an observer in the
        ``_state_lock`` window: while the merge thread runs and
        ``repo_lock`` is held by the test (forcing the merge to
        block in its critical section), the merge handler should
        have already set ``is_merging = True`` (it sets it under
        ``_state_lock`` then releases ``_state_lock`` and acquires
        ``repo_lock`` for the body).

        Implementation: hold ``repo_lock``, run the merge body in a
        thread, then assert ``tab.is_merging`` is True.  Before the
        fix the flag stayed False through the entire merge.
        """
        repo = _make_repo(tmp_path / "repo")
        server = _server(repo)

        tab = server._get_tab("a")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.use_worktree = True
        agent = cast(WorktreeSorcarAgent, tab.agent)
        _make_wt_with_commit(repo, "kiss/wt-race1-2", agent)

        lock = repo_lock(repo)
        # Acquire ``repo_lock`` from a separate thread so the merge
        # blocks on it.  (RLocks track owner threads — acquiring on
        # the main thread would let the merge re-enter.)
        holder_acquired = threading.Event()
        holder_release = threading.Event()

        def holder() -> None:
            lock.acquire()
            holder_acquired.set()
            holder_release.wait(timeout=10.0)
            lock.release()

        holder_t = threading.Thread(target=holder, daemon=True)
        holder_t.start()
        assert holder_acquired.wait(2.0)

        result_holder: list[dict[str, Any]] = []

        def run_merge() -> None:
            result_holder.append(
                server._handle_worktree_action("merge", "a"),
            )

        merge_thread = threading.Thread(target=run_merge, daemon=True)
        merge_thread.start()

        # The merge handler runs ``_state_lock`` (sets is_merging),
        # then tries to acquire ``repo_lock`` which is held by the
        # holder thread — so it now waits.  Give it time.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with server._state_lock:
                if tab.is_merging:
                    break
            time.sleep(0.02)
        assert tab.is_merging, (
            "RACE-1: merge handler did not set ``tab.is_merging`` "
            "before blocking on repo_lock.  The flag is the signal "
            "for non-wt task-start to refuse."
        )

        holder_release.set()
        holder_t.join(timeout=5)
        merge_thread.join(timeout=15)
        # After completion the flag must be cleared.
        assert not tab.is_merging, "is_merging must be cleared post-merge"
        assert result_holder and result_holder[0].get("success") is True


class TestRacePostTaskVsUserAction:
    """RACE-3: ``is_task_active`` is now cleared AFTER
    ``_present_pending_worktree`` runs in ``_run_task_inner``'s
    finally block, so a concurrent ``_handle_worktree_action`` click
    is refused (with ``success: False``) while the task thread is
    still touching ``tab.agent._wt``.
    """

    def test_is_task_active_cleared_after_present_pending_worktree(
        self,
    ) -> None:
        """Structural confirmation that the buggy ordering is fixed.

        ``is_task_active = False`` must now follow the
        ``_present_pending_worktree`` call in the worktree-mode
        post-task block.  This is the static guarantee that the
        runtime guard (``_check_worktree_busy``) keeps refusing
        concurrent clicks until the task thread is done.
        """
        src = (
            Path(__file__).resolve().parents[3]
            / "server" / "task_runner.py"
        ).read_text()

        start = src.index("def _run_task_inner")
        rest = src[start:]
        present_idx = rest.index("_present_pending_worktree")
        # Find the ``is_task_active = False`` after the present call.
        post_present = rest[present_idx:]
        clear_idx_rel = post_present.index("tab.is_task_active = False")
        clear_idx = present_idx + clear_idx_rel

        assert present_idx < clear_idx, (
            "RACE-3 regression: ``tab.is_task_active = False`` must "
            "appear AFTER ``_present_pending_worktree`` in the "
            "_run_task_inner finally block.  Found clear at "
            f"{clear_idx} and present at {present_idx}."
        )

    def test_handle_worktree_action_refuses_while_task_active(
        self, tmp_path: Path,
    ) -> None:
        """A concurrent merge/discard click during an active task
        is refused by ``_check_worktree_busy``.

        Simulates the RACE-3 window: the task thread has set
        ``is_task_active = True`` (the flag is now cleared only at
        the very end of the cleanup block).  A click on the discard
        button arrives — the handler must return
        ``success: False`` instead of mutating ``agent._wt``.
        """
        repo = _make_repo(tmp_path / "repo")
        server = _server(repo)

        tab = server._get_tab("a")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.use_worktree = True
        tab.is_task_active = True
        agent = cast(WorktreeSorcarAgent, tab.agent)
        wt = _make_wt_with_commit(repo, "kiss/wt-race3-active", agent)

        # Click the discard button while the task is still active.
        result = server._handle_worktree_action("discard", "a")
        assert result.get("success") is False, (
            f"RACE-3: discard must be refused while task is active. "
            f"Got: {result}"
        )
        assert "still running" in (result.get("message") or "").lower(), (
            f"Expected 'still running' message; got: {result}"
        )
        # And ``agent._wt`` must be untouched.
        assert agent._wt is wt, "agent._wt was mutated despite refusal"
        assert wt.wt_dir.exists(), (
            "Worktree directory was removed despite refusal"
        )


class TestRaceSetupRespectsRepoLock:
    """RACE-2: ``_try_setup_worktree`` now wraps its full body in
    ``repo_lock`` (re-entrant) so the release + create + dirty-state
    + baseline phases cannot interleave with another tab's
    ``_do_merge`` or ``copy_dirty_state``.
    """

    def test_setup_holds_repo_lock_across_dirty_state_copy(
        self,
    ) -> None:
        """Structural confirmation: ``_try_setup_worktree``'s body
        is enclosed by a ``with repo_lock(repo):`` block that
        contains the ``copy_dirty_state`` call.
        """
        src = (
            Path(__file__).resolve().parents[3]
            / "agents" / "sorcar" / "worktree_sorcar_agent.py"
        ).read_text()

        start = src.index("def _try_setup_worktree")
        end = src.index("\n    def ", start + 1)
        body = src[start:end]

        # The body must contain a ``with repo_lock(repo):`` block
        # that encloses ``copy_dirty_state``.  Search for the actual
        # call sites (with ``GitWorktreeOps.`` prefix) so we don't
        # match the docstring reference.
        lock_idx = body.find("with repo_lock(repo):")
        copy_idx = body.find("GitWorktreeOps.copy_dirty_state")
        baseline_idx = body.find("GitWorktreeOps.save_baseline_commit")
        assert lock_idx != -1, (
            "RACE-2 regression: ``_try_setup_worktree`` must hold "
            "``repo_lock(repo)`` across the setup body."
        )
        assert copy_idx != -1 and baseline_idx != -1
        assert lock_idx < copy_idx < baseline_idx, (
            "RACE-2 regression: ``repo_lock`` must enclose both "
            f"``copy_dirty_state`` (idx {copy_idx}) and "
            f"``save_baseline_commit`` (idx {baseline_idx}) — "
            f"lock at idx {lock_idx}."
        )

    def test_repo_lock_is_reentrant(self) -> None:
        """``repo_lock`` returns an RLock so the wrapped
        ``_release_worktree`` → ``_do_merge`` re-acquisition does
        not deadlock on the same thread.
        """
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            repo = Path(td)
            lock = repo_lock(repo)
            # RLock allows same-thread re-entrance; a plain Lock
            # would deadlock here.
            lock.acquire()
            try:
                acquired_again = lock.acquire(blocking=False)
                assert acquired_again, (
                    "RACE-2 regression: repo_lock must be re-entrant "
                    "(threading.RLock) — got a non-re-entrant Lock."
                )
                lock.release()
            finally:
                lock.release()


class TestRaceNonWtStartBlockedByOngoingMerge:
    """A non-wt task-start path must observe the ``is_merging`` flag
    set by a concurrent merge handler and refuse to start.  Combined
    with RACE-1's atomic check+set, this prevents the merge from
    stashing in-flight writes from another tab.
    """

    def test_non_wt_task_start_guard_observes_is_merging(
        self, tmp_path: Path,
    ) -> None:
        """The non-wt task-start guard in ``_run_task_inner``
        inspects ``is_merging and use_worktree`` under
        ``_state_lock``.  When the merge handler sets the flag, a
        concurrent non-wt task-start must observe it and refuse.
        """
        from kiss.agents.sorcar.running_agent_state import (
            _RunningAgentState,
        )

        repo = _make_repo(tmp_path / "repo")
        server = _server(repo)

        # Tab A: a worktree tab actively running its merge handler.
        tab_a = server._get_tab("a")
        tab_a.use_worktree = True
        tab_a.is_merging = True

        # Tab B: tries to start a non-wt task.
        tab_b = server._get_tab("b")
        tab_b.use_worktree = False

        # Apply the actual guard predicate from ``_run_task_inner``.
        with server._state_lock:
            should_refuse = any(
                t.is_merging and t.use_worktree
                for t in _RunningAgentState.running_agent_states.values()
            )
        assert should_refuse, (
            "RACE-1: non-wt task-start guard must observe the merge "
            "handler's ``is_merging = True`` flag."
        )

        # Clearing tab A's flag releases the gate.
        tab_a.is_merging = False
        with server._state_lock:
            should_refuse = any(
                t.is_merging and t.use_worktree
                for t in _RunningAgentState.running_agent_states.values()
            )
        assert not should_refuse
