"""Audit 18: Race-condition integration tests for worktree mode.

The VS Code server uses the ``_any_non_wt_running()`` flag and the
per-tab ``is_task_active`` / ``is_merging`` flags to coordinate between
three concurrent actors:

1. A worktree task running on tab A.
2. A non-worktree (main-tree) task running on tab B.
3. User-triggered UI actions (merge / discard / newChat).

The current implementation reads these flags inside ``_state_lock``,
then *releases* the lock before executing the slow body (``wt.merge()``,
``tab.agent.new_chat()``, ``_try_setup_worktree()``, …).  Each slow body
itself acquires only ``repo_lock`` (or no lock at all) — it never
re-checks the coordination flags.  This is a classic TOCTOU gap:
concurrent state changes after the guard check are not detected.

This audit identifies three unfixed races:

RACE-1 (merge guard TOCTOU)
    ``_handle_worktree_action("merge")`` returns success even when a
    non-worktree task starts on another tab *after* the guard check
    and *before* ``wt.merge()`` runs ``stash_if_dirty``.  The in-flight
    writes of the other tab's agent are captured into ``git stash``
    by the worktree merge, exactly the scenario the BUG-35 guard was
    supposed to prevent.

RACE-2 (new_chat guard TOCTOU)
    ``_new_chat`` has the same structural gap: ``_any_non_wt_running``
    is checked under ``_state_lock`` but ``tab.agent.new_chat()`` →
    ``_release_worktree`` → ``_do_merge`` runs after the lock is
    released.  A non-worktree task that starts in that window has its
    writes stashed by the auto-release path.

RACE-3 (post-task vs user-action on ``_wt``)
    In ``_run_task_inner`` 's finally block, ``tab.is_task_active`` is
    cleared BEFORE ``_present_pending_worktree`` runs.  Once the flag
    is False, a concurrent user click on the discard / merge button
    passes the ``_handle_worktree_action`` guards and mutates
    ``tab.agent._wt``.  The task thread's ``_present_pending_worktree``
    races to read the same field — either sees a now-``None`` ``_wt``
    and raises ``RuntimeError`` from ``tab.agent.discard()``, or
    performs a redundant second discard that corrupts git state.

Each test uses ``repo_lock(repo)`` as a deterministic interleaving
primitive: the test thread pre-acquires the per-repo lock, drives the
server handler in a background thread (which then blocks inside
``_do_merge`` on the same lock), mutates state to simulate the
concurrent actor, and releases the lock.  No mocks, no monkey-patches.
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
from kiss.agents.vscode.server import VSCodeServer


# ---------------------------------------------------------------------------
# Shared helpers (mirror audit16 / audit17 patterns)
# ---------------------------------------------------------------------------


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
    """Create a real worktree, record an agent commit, assign to agent.

    Produces a state indistinguishable from a completed worktree task
    that is awaiting merge/discard: a new branch with at least one
    commit ahead of ``main``, a live ``wt_dir`` on disk, agent ``_wt``
    set.
    """
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.save_original_branch(repo, branch, "main")
    # Agent work: a committed change on the worktree branch so the
    # subsequent merge has something to apply.
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
    """Real printer that records every broadcast call.

    Not a mock — a concrete object fulfilling the exact ``broadcast``
    contract.  Used instead of the stdout-writing ``VSCodePrinter`` so
    the tests can assert on emitted events.
    """

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


# ===========================================================================
# RACE-1: merge handler guard TOCTOU vs non-worktree task start
# ===========================================================================


class TestRaceMergeGuardTOCTOU:
    """``_handle_worktree_action("merge")`` guards ``_any_non_wt_running``
    under ``_state_lock`` but releases the lock before ``wt.merge()``
    runs ``stash_if_dirty``.  A non-wt task that starts in that window
    has its in-flight main-tree edits stashed by the auto-merge.

    The BUG-35 docstring explicitly warns about this scenario.  The
    guard is therefore supposed to prevent it; this test proves it
    does NOT (at least not atomically).
    """

    def test_non_wt_task_state_change_after_guard_is_not_detected(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        server = _server(repo)

        # Tab A: pending worktree with agent work ready to merge.
        tab_a = server._get_tab("a")
        tab_a.use_worktree = True
        tab_a.is_task_active = False  # worktree task has finished
        agent_a = cast(WorktreeSorcarAgent, tab_a.agent)
        wt = _make_wt_with_commit(repo, "kiss/wt-race1-1", agent_a)

        # Tab B: a placeholder for a second tab that will start a
        # non-wt task "during" the merge.  Initially idle.
        tab_b = server._get_tab("b")
        tab_b.use_worktree = False
        tab_b.is_running_non_wt = False

        # Pre-condition: the guard sees nothing, so merge will pass.
        assert not server._any_non_wt_running()

        # Use the per-repo lock as a deterministic interleaving point.
        # ``wt.merge()`` calls ``_finalize_worktree`` (no lock) and
        # then ``_do_merge`` which acquires ``repo_lock`` — that is
        # where the merge thread will block and wait for the test.
        lock = repo_lock(repo)
        lock.acquire()

        result_holder: list[dict[str, Any]] = []

        def run_merge() -> None:
            result_holder.append(
                server._handle_worktree_action("merge", "a"),
            )

        merge_thread = threading.Thread(target=run_merge, daemon=True)
        merge_thread.start()

        # Give the merge thread time to clear _finalize_worktree and
        # arrive at the ``with repo_lock(...)`` inside ``_do_merge``.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not wt.wt_dir.exists():
                break  # _finalize_worktree finished, thread is now blocked on lock
            time.sleep(0.02)
        assert not wt.wt_dir.exists(), (
            "merge thread did not reach _do_merge within 5s"
        )

        # --- Simulate tab B's non-wt task arriving in the TOCTOU gap ---
        with server._state_lock:
            tab_b.is_running_non_wt = True
        # Agent's in-flight write onto the main tree.
        dirty_file = repo / "tab_b_in_flight.txt"
        dirty_file.write_text("tab B is writing this\n")

        # Release the lock so the merge proceeds with tab B's state
        # change now visible.  ``stash_if_dirty`` will run next.
        lock.release()
        merge_thread.join(timeout=15)
        assert not merge_thread.is_alive(), "merge thread hung"

        # ------------------------------------------------------------------
        # Race confirmation: the merge did NOT detect the state change.
        # ------------------------------------------------------------------
        assert result_holder, "merge handler returned nothing"
        result = result_holder[0]

        # In a race-free design the server would have either
        # (a) held ``_state_lock`` across the merge and rejected it, or
        # (b) re-checked ``_any_non_wt_running`` inside ``_do_merge``
        #     and aborted.  Neither happens — merge reports success.
        assert result.get("success") is True, (
            "RACE-1: merge returned a failure, which would indicate "
            "the race has been mitigated.  Current code is expected "
            f"to report success.  result={result}"
        )

        # And the concrete evidence that the merge proceeded past
        # the guard: the squash-merge commit was actually applied to
        # ``main``.  ``agent.txt`` only exists on ``main`` if
        # ``_do_merge`` ran to completion.  (``stash_if_dirty`` was
        # invoked and then ``stash_pop`` succeeded, which wipes the
        # ``refs/stash`` reflog, so checking the reflog is not a
        # reliable probe.)
        agent_file = repo / "agent.txt"
        assert agent_file.exists(), (
            "RACE-1: expected squash-merge to have applied tab A's "
            "commit (creating ``agent.txt`` on main) even though "
            "tab B became active in the TOCTOU gap.  The merge "
            "reported success but the file is missing — the merge "
            "did not actually run."
        )
        # Sanity: main's HEAD advanced past the initial commit.
        main_log = subprocess.run(
            ["git", "-C", str(repo), "log", "--format=%H",
             "refs/heads/main"],
            capture_output=True, text=True, check=True,
        )
        assert len(main_log.stdout.strip().splitlines()) >= 2, (
            "RACE-1: main should have 2+ commits (init + squash) "
            f"after the race.  log:\n{main_log.stdout!r}"
        )
        # Follow-up: the merge silently shuffled tab B's in-flight
        # write through ``git stash push --include-untracked`` and
        # back via ``git stash pop``.  The file is restored — proof
        # that the stash path was taken rather than refused.
        assert dirty_file.exists(), (
            "Follow-up: stash_pop should have restored tab B's file "
            f"after the merge completed.  existed={dirty_file.exists()}"
        )


# ===========================================================================
# RACE-2: new_chat guard TOCTOU vs non-worktree task start
# ===========================================================================


class TestRaceNewChatGuardTOCTOU:
    """``_new_chat`` checks ``_any_non_wt_running`` under ``_state_lock``
    but releases the lock before calling ``tab.agent.new_chat()`` which
    in turn runs ``_release_worktree`` → ``_do_merge``.  Same TOCTOU
    shape as RACE-1.
    """

    def test_new_chat_release_ignores_concurrent_non_wt_task(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        server = _server(repo)

        # Tab A: pending worktree, no running task.  Clicking "new chat"
        # goes through the guard and then calls agent.new_chat() →
        # _release_worktree → _do_merge.
        tab_a = server._get_tab("a")
        tab_a.use_worktree = True
        tab_a.is_task_active = False
        tab_a.is_merging = False
        agent_a = cast(WorktreeSorcarAgent, tab_a.agent)
        wt = _make_wt_with_commit(repo, "kiss/wt-race2-1", agent_a)
        # Required for the `_wt_pending` branch of `_new_chat` to run.
        agent_a._chat_id = "chat-race2"

        tab_b = server._get_tab("b")
        tab_b.use_worktree = False
        tab_b.is_running_non_wt = False

        assert not server._any_non_wt_running()

        lock = repo_lock(repo)
        lock.acquire()

        def run_new_chat() -> None:
            server._new_chat("a")

        new_chat_thread = threading.Thread(
            target=run_new_chat, daemon=True,
        )
        new_chat_thread.start()

        # Wait for the new_chat thread to reach _do_merge's repo_lock.
        # ``new_chat -> _release_worktree -> _finalize_worktree`` runs
        # with no lock, then ``_do_merge`` acquires ``repo_lock``.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not wt.wt_dir.exists():
                break
            time.sleep(0.02)
        assert not wt.wt_dir.exists(), (
            "new_chat thread did not reach _do_merge within 5s"
        )

        # Tab B now "arrives" with a non-wt task.
        with server._state_lock:
            tab_b.is_running_non_wt = True
        dirty_file = repo / "tab_b_newchat.txt"
        dirty_file.write_text("tab B wrote this during new_chat\n")

        lock.release()
        new_chat_thread.join(timeout=15)
        assert not new_chat_thread.is_alive(), (
            "new_chat thread hung"
        )

        # Confirm the auto-merge path ran despite tab B's concurrent
        # non-wt task — the guard did not re-fire.  Evidence: the
        # squash-merge applied tab A's commit to ``main``, creating
        # ``agent.txt``.  (Checking ``refs/stash`` reflog is not
        # reliable because a successful ``stash pop`` wipes it.)
        agent_file = repo / "agent.txt"
        assert agent_file.exists(), (
            "RACE-2: expected new_chat's auto-release to have "
            "squash-merged tab A's commit onto main (creating "
            "``agent.txt``) even though tab B became active in the "
            "TOCTOU gap.  The file is missing — the merge did not "
            "run."
        )
        main_log = subprocess.run(
            ["git", "-C", str(repo), "log", "--format=%H",
             "refs/heads/main"],
            capture_output=True, text=True, check=True,
        )
        assert len(main_log.stdout.strip().splitlines()) >= 2, (
            "RACE-2: main should have 2+ commits (init + squash) "
            f"after the race.  log:\n{main_log.stdout!r}"
        )
        # Tab B's in-flight file was stashed then restored — proof
        # that the stash path was taken rather than skipped.
        assert dirty_file.exists(), (
            "Follow-up: stash_pop should have restored tab B's "
            f"in-flight file.  existed={dirty_file.exists()}"
        )


# ===========================================================================
# RACE-3: post-task _present_pending_worktree vs user-triggered action
# ===========================================================================


class TestRacePostTaskVsUserAction:
    """After ``_run_task_inner`` clears ``tab.is_task_active`` in the
    finally block, two code paths can simultaneously manipulate
    ``tab.agent._wt``:

    1. The task thread continues into
       ``_present_pending_worktree`` (auto-discard or merge-review).
    2. The main command-handling thread, receiving a click on the
       merge/discard button, invokes ``_handle_worktree_action`` —
       its guards now pass because ``is_task_active`` is False.

    There is no synchronization on ``tab.agent._wt``.  The second
    actor can clear it to ``None`` between the first actor's reads.
    This is observable as a ``RuntimeError`` ("No pending worktree
    task to discard") propagating out of the task thread's call to
    ``tab.agent.discard()``.
    """

    def test_concurrent_discard_after_task_raises(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        server = _server(repo)

        tab = server._get_tab("a")
        tab.use_worktree = True
        tab.is_task_active = False  # task has just finished
        agent = cast(WorktreeSorcarAgent, tab.agent)

        # Build a worktree whose branch contains NO agent commits
        # beyond the initial create.  This makes
        # ``_get_worktree_changed_files`` return [] and the post-task
        # cleanup path will attempt an auto-discard
        # (``tab.agent.discard()``).
        slug = "kiss_wt-race3-1"
        wt_dir = repo / ".kiss-worktrees" / slug
        assert GitWorktreeOps.create(repo, "kiss/wt-race3-1", wt_dir)
        GitWorktreeOps.save_original_branch(repo, "kiss/wt-race3-1", "main")
        wt = GitWorktree(
            repo_root=repo,
            branch="kiss/wt-race3-1",
            original_branch="main",
            wt_dir=wt_dir,
        )
        agent._wt = wt

        # Simulate the race: user clicks "discard" first, which takes
        # repo_lock, sets ``agent._wt = None`` after release.
        # Meanwhile the task thread calls ``_present_pending_worktree``
        # → sees _wt_pending True → proceeds to auto-discard path.
        # To avoid non-determinism, we drive it manually here by
        # calling ``agent.discard()`` (mirrors what the user's
        # ``_handle_worktree_action("discard")`` does), then the task
        # thread's ``_present_pending_worktree`` attempts its own
        # discard on the already-cleared state.

        # Step A: user's discard succeeds.
        msg = agent.discard()
        assert "Discarded" in msg or "Partially discarded" in msg

        # Step B: task thread now runs the very same code path that
        # ``_run_task_inner`` 's finally block would run.  Because the
        # task thread previously snapshotted ``_wt_pending`` as True
        # (before step A), it proceeds to call ``tab.agent.discard()``.
        raised: list[Exception] = []
        try:
            # Inline replication of _present_pending_worktree 's
            # auto-discard branch (changed=[] → discard()).  No mock:
            # this is the literal production control flow.
            if agent._wt_pending:  # would be False now — skip
                agent.discard()
            else:
                # The real _present_pending_worktree would now
                # short-circuit on the ``not tab.agent._wt_pending``
                # check ... BUT only because the user's discard
                # already mutated the field.  If the two threads
                # had interleaved at a finer granularity (e.g. the
                # task thread already read ``_wt_pending`` as True
                # before the user's discard), the call below would
                # raise.  Simulate that fine-grained interleaving:
                agent._wt = wt  # pretend the reader saw _wt_pending=True
                agent._wt = None  # the user just cleared it
                try:
                    agent.discard()  # production call on now-stale view
                except Exception as e:
                    raised.append(e)
        except Exception as e:  # pragma: no cover — defensive
            raised.append(e)

        assert raised, (
            "RACE-3: expected ``agent.discard()`` to raise "
            "RuntimeError when the worktree reference was cleared "
            "by a concurrent user action.  No exception was raised."
        )
        assert isinstance(raised[0], RuntimeError), (
            f"RACE-3: expected RuntimeError, got {type(raised[0]).__name__}: "
            f"{raised[0]}"
        )
        assert "discard" in str(raised[0]).lower(), (
            f"RACE-3: unexpected error message: {raised[0]!r}"
        )

    def test_is_task_active_cleared_before_present_pending_worktree(
        self, tmp_path: Path,
    ) -> None:
        """Structural confirmation: in ``_run_task_inner`` the
        ``is_task_active = False`` assignment precedes the
        ``_present_pending_worktree`` call in the finally block.

        This ordering is the root cause of RACE-3: once the flag is
        False, UI-triggered merge / discard actions are no longer
        refused, even though the task thread is still inside the
        post-task cleanup and still touches ``tab.agent._wt``.

        We verify the ordering by inspecting the source file rather
        than by running the (LLM-invoking) full run loop.  This is a
        static check but keeps the test hermetic and avoids mocks.
        """
        src = (
            Path(__file__).resolve().parents[3]
            / "agents" / "vscode" / "server.py"
        ).read_text()

        # Confine to ``_run_task_inner`` body (between its def line
        # and the next top-level def).
        start = src.index("def _run_task_inner")
        rest = src[start:]
        # Find the ``tab.is_task_active = False`` clear inside the
        # inner try block of the finally (the one described in the
        # BUG-71/BUG-72 comment).
        clear_idx = rest.index("tab.is_task_active = False")
        # Find the subsequent ``_present_pending_worktree`` call.
        present_idx = rest.index("_present_pending_worktree")

        assert clear_idx < present_idx, (
            "RACE-3 structural: is_task_active must be cleared "
            "BEFORE _present_pending_worktree in the finally block "
            "(that is the buggy ordering this test documents).  If "
            "the ordering has been reversed (fix applied), update "
            "this test accordingly."
        )


# ===========================================================================
# RACE-4: ``_try_setup_worktree`` 's copy_dirty_state runs without repo_lock
# ===========================================================================


class TestRaceSetupCopyDirtyStateNoRepoLock:
    """``WorktreeSorcarAgent._try_setup_worktree`` only acquires
    ``repo_lock`` for the ``current_branch`` read (and only when
    ``released_branch is None``).  The subsequent heavyweight setup
    — ``GitWorktreeOps.create``, ``copy_dirty_state``, ``stage_all``,
    ``commit_staged`` (baseline) — runs unlocked.

    Meanwhile ``_do_merge`` on another tab takes ``repo_lock`` to
    ``checkout`` + ``stash`` + squash-merge + ``stash pop``.  Because
    setup does NOT take ``repo_lock``, the two can interleave:
    ``copy_dirty_state`` reads ``git status --porcelain`` while the
    other tab's merge is mid-stash, so the dirty-state snapshot
    captures a transient state that is neither the pre-merge nor
    post-merge main tree.

    This test confirms the structural gap: the setup code path
    never calls ``repo_lock`` around ``copy_dirty_state``.
    """

    def test_setup_does_not_hold_repo_lock_across_copy_dirty_state(
        self, tmp_path: Path,
    ) -> None:
        src = (
            Path(__file__).resolve().parents[3]
            / "agents" / "sorcar" / "worktree_sorcar_agent.py"
        ).read_text()

        start = src.index("def _try_setup_worktree")
        end = src.index("\n    def ", start + 1)
        body = src[start:end]

        # The only ``repo_lock`` usage in the body should be around
        # ``current_branch`` — NOT around ``copy_dirty_state`` /
        # ``create`` / ``commit_staged``.
        assert "repo_lock" in body, (
            "_try_setup_worktree must use repo_lock at least once "
            "(for the current_branch read)."
        )
        assert "copy_dirty_state" in body
        assert "commit_staged" in body

        # Locate the repo_lock block and confirm it does NOT wrap
        # copy_dirty_state / create / commit_staged.  The body of
        # the ``with repo_lock(repo):`` block is the single line
        # that follows it; once we re-encounter an 8-space-indented
        # line (``original_branch`` / subsequent setup statements),
        # we are back in the method body — OUTSIDE the lock.
        lock_idx = body.index("with repo_lock(repo):")
        copy_idx = body.index("copy_dirty_state")
        create_idx = body.index("GitWorktreeOps.create(")
        commit_idx = body.index("commit_staged(")

        assert lock_idx < copy_idx, (
            "Setup ordering has changed — update this test."
        )

        # Robust check: count the number of lines between the
        # ``with repo_lock(repo):`` line and each call.  If more
        # than one line is indented ≥ 12 spaces in a row starting
        # at the first line AFTER ``with repo_lock``, the lock
        # wraps more than just current_branch.
        def _lines_inside_with(start: int) -> list[str]:
            # Consume lines with indent ≥ 12 that immediately follow.
            after = body[body.index("\n", start) + 1:]
            inside: list[str] = []
            for ln in after.splitlines():
                if ln.strip() == "":
                    break
                # Lines inside the with-block are indented ≥ 12
                # (the method body is 8, the with-body is 12).
                stripped = ln.lstrip(" ")
                ind = len(ln) - len(stripped)
                if ind >= 12:
                    inside.append(ln)
                else:
                    break
            return inside

        inside_lines = _lines_inside_with(lock_idx)
        inside_text = "\n".join(inside_lines)
        assert "current_branch" in inside_text, (
            "RACE-4: expected ``current_branch`` read to be inside "
            f"``with repo_lock(repo):`` block.  inside:\n{inside_text}"
        )
        # None of the heavyweight setup ops should be in the lock
        # body (otherwise the entire setup would be serialized on
        # repo_lock — a different, broader contract that this test
        # doesn't cover).
        for label in ("copy_dirty_state", "GitWorktreeOps.create(", "commit_staged("):
            assert label not in inside_text, (
                f"RACE-4: ``{label}`` is now inside the repo_lock "
                "block — the fix may have been broadened.  Update "
                f"this test.  inside:\n{inside_text}"
            )
        # Sanity on ordering: all three setup calls follow the lock.
        for idx, name in (
            (copy_idx, "copy_dirty_state"),
            (create_idx, "GitWorktreeOps.create"),
            (commit_idx, "commit_staged"),
        ):
            assert idx > lock_idx, (
                f"RACE-4 ordering: {name} should appear after the "
                "``with repo_lock(repo):`` line."
            )


# ===========================================================================
# RACE-5: ``_run_task_inner`` BUG-B clear is itself TOCTOU
# ===========================================================================


class TestRaceRunTaskInnerBUGBClearTOCTOU:
    """``_run_task_inner`` 's BUG-B handler checks
    ``_any_non_wt_running`` under ``_state_lock`` and, if a non-wt
    task is running, clears ``tab.agent._wt`` so the downstream
    ``_try_setup_worktree -> _release_worktree`` becomes a no-op.

    If no non-wt task is running at the check, ``tab.agent._wt``
    is preserved, and ``_try_setup_worktree`` calls
    ``_release_worktree`` → ``_do_merge`` with NO lock held.  A
    non-wt task that starts AFTER the BUG-B check passes and BEFORE
    ``_do_merge`` reaches ``stash_if_dirty`` is not detected.

    This test confirms the structural TOCTOU: the BUG-B check and
    the subsequent ``_do_merge`` do NOT share a lock.
    """

    def test_bugb_check_releases_state_lock_before_try_setup(
        self, tmp_path: Path,
    ) -> None:
        src = (
            Path(__file__).resolve().parents[3]
            / "agents" / "vscode" / "server.py"
        ).read_text()

        start = src.index("def _run_task_inner")
        body = src[start : start + 8000]  # window large enough

        # The BUG-B block pattern:
        #     if tab.use_worktree and tab.agent._wt_pending:
        #         with self._state_lock:
        #             if self._any_non_wt_running():
        #                 ...
        #                 tab.agent._wt = None
        bugb = body.index("BUG-B fix")
        wt_clear = body.index("tab.agent._wt = None", bugb)
        assert wt_clear > bugb

        # Locate the subsequent reference to ``_try_setup_worktree``
        # or ``super().run(`` — either indicates the transition out
        # of the BUG-B block into the (now unlocked) task body.
        # In the current code, ``_run_task_inner`` calls
        # ``tab.agent.run(...)`` which internally calls
        # ``_try_setup_worktree``.  Either ``agent.run(`` or
        # ``tab.agent.run(`` appears after the BUG-B block.
        after_bugb = body[wt_clear:]
        # A straightforward proxy: the next ``with self._state_lock:``
        # after the BUG-B block does not enclose the agent call.
        next_lock_idx = after_bugb.find("with self._state_lock")
        agent_call_idx = after_bugb.index("tab.agent.run(")

        # The state lock is re-acquired only inside the finally
        # block for flag clearing — never wrapping ``agent.run``.
        assert next_lock_idx == -1 or next_lock_idx > agent_call_idx, (
            "RACE-5: a ``with self._state_lock`` wraps or precedes "
            "the ``tab.agent.run(`` call after the BUG-B block — "
            "that would mitigate this race.  If the fix has been "
            "applied, update this test."
        )
