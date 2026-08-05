# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""With auto-commit ON, resuming a chat must never pop the diff/merge UI.

The user-observed defect
------------------------

    "why did the diff/merge UI show up on the last task when the
    auto-commit mode is on?"

The task itself had ``useWorktree=True`` and ``autoCommit=True`` and it
SUCCEEDED — its persisted event stream ends with ``result``
(``success: true``) followed by ``task_done`` and carries no merge
event at all.  The merge view therefore did not come from the task; it
came from the *session-resume* path that runs when the user clicks a
row in the task-history panel.

Mechanism
---------

1. :meth:`_TaskRunnerMixin._run_task_inner` auto-merges on success, but
   :meth:`WorktreeSorcarAgent.merge` clears its pending worktree ONLY on
   ``MergeResult.SUCCESS``.  A merge that cannot complete (conflict,
   ``git stash`` failure, rejected pre-commit hook, ...) returns an
   error string and leaves ``_wt_pending`` raised.  Nothing retries.

2. :meth:`_ServerCommandsMixin._replay_session` then calls
   :meth:`_MergeFlowMixin._emit_pending_worktree` unconditionally, which
   walks into ``_present_pending_worktree(try_merge_review=True)`` and
   broadcasts ``merge_data`` + ``merge_started`` — the diff/merge UI —
   even though the user has auto-commit switched on.  Worse,
   ``_replay_session`` had already reset ``tab.auto_commit_mode`` to
   ``False`` a few lines earlier, so the tab could not even report the
   user's real preference.

Expected behaviour
------------------

Auto-commit means "do not ask me".  On resume the pending worktree of an
auto-commit tab must be finalized silently (merged, or discarded when it
holds nothing) rather than presented as a hunk-by-hunk review.  With
auto-commit OFF the review must still appear — that is the whole point
of the setting — so both directions are asserted here.

One deliberate exception is asserted too: work from a task that failed
or was stopped (``_pending_review``) is still shown, because
auto-commit means "do not interrupt me", not "publish work that never
finished".

Every test drives the real :class:`VSCodeServer` command handlers
against a real git repository.  Only the LLM call is replaced by a
deterministic stub, exactly as the sibling auto-commit tests do.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _release_worktree_without_merging

#: Events that make the hunk-by-hunk diff/merge UI appear.
MERGE_UI_EVENTS = ("merge_data", "merge_started")

#: Event that makes the worktree merge/discard prompt appear.
WORKTREE_PROMPT_EVENT = "worktree_done"


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


def _kiss_wt_branches(repo: str) -> list[str]:
    """Names of the ``kiss/wt-*`` branches present in *repo*.

    ``git branch`` prefixes the current branch with ``*`` and a branch
    checked out in a linked worktree with ``+``; both markers are
    stripped so callers compare plain names.
    """
    out = _run_git(repo, "branch", "--list", "kiss/wt-*").stdout
    return [
        line.strip().lstrip("*+ ").strip()
        for line in out.splitlines()
        if line.strip()
    ]


class _Base(unittest.TestCase):
    """A real git repo, an isolated persistence DB and a real server."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-autocommit-ui-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

        # A hooks dir whose pre-commit always fails, so a test can make
        # `git commit` be refused exactly as a real rejecting hook does.
        self.hooks = Path(self.tmpdir) / "hooks"
        self.hooks.mkdir(parents=True, exist_ok=True)
        hook = self.hooks / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict[str, Any]] = []
        printer = self.server.printer

        def capture(event: dict[str, Any]) -> None:
            ev = printer._inject_task_id(event)
            with printer._lock:
                printer._record_event(ev)
            printer._persist_event(ev)
            self.events.append(ev)

        printer.broadcast = capture  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        for tab in list(_RunningAgentState.running_agent_states.values()):
            if tab.agent is not None and tab.agent._wt_pending:
                try:
                    tab.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        _RunningAgentState.running_agent_states.clear()
        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- helpers ---------------------------------------------------

    def _patch_run(self, filename: str | None = "agent_out.txt") -> None:
        """Make the agent write *filename* into its work dir and succeed."""

        def stub_run(_agent: object, **kwargs: object) -> str:
            work_dir = kwargs.get("work_dir")
            if filename is not None and isinstance(work_dir, str) and work_dir:
                (Path(work_dir) / filename).write_text("agent output\n")
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def _run_task(self, tab_id: str, *, auto_commit: bool) -> None:
        self.server._run_task_inner({
            "prompt": "make a change",
            "workDir": self.repo,
            "tabId": tab_id,
            "useWorktree": True,
            "autoCommit": auto_commit,
            "model": "",
        })

    def _types(self) -> list[str]:
        return [str(e.get("type", "")) for e in self.events]

    def _latest_row(self) -> dict[str, Any]:
        rows = _persistence._load_history(limit=10)
        assert rows, "the task should have been persisted"
        return cast(dict[str, Any], rows[0])

    def _resume(self, tab_id: str, chat_id: str, task_id: str) -> None:
        self.server._handle_command({
            "type": "resumeSession",
            "chatId": chat_id,
            "taskId": task_id,
            "tabId": tab_id,
        })

    def _finish_open_review(self, tab_id: str) -> None:
        """Close a hunk review the task itself opened.

        A task run with auto-commit OFF ends inside the diff/merge view,
        leaving ``tab.is_merging`` raised.  Resuming a chat in that
        state is deliberately a no-op (regenerating the view would
        discard the user's accepted/rejected hunks), so the tests that
        exercise a *later* history click must first dismiss the review
        the way the frontend does — with an ``all-done`` mergeAction.
        """
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": tab_id,
            "workDir": self.repo,
        })

    def _assert_resume_reaches_the_decision(self, tab_id: str) -> None:
        """Fail loudly unless the next resume exercises the fixed code.

        ``_emit_pending_worktree`` returns early for a tab that is
        merging, running a task, not in worktree mode or holding no
        pending worktree.  A fixture that trips any of those makes the
        "no merge UI appeared" assertions pass for the wrong reason —
        they would hold with the fix reverted too.  Asserting the
        preconditions keeps the regression honest.
        """
        tab = self.server._get_tab(tab_id)
        assert tab.use_worktree, "the tab must be in worktree mode"
        assert not tab.is_merging, (
            "a review already owns the worktree; resume is a deliberate "
            "no-op in that state and would prove nothing"
        )
        assert not tab.is_task_active, "no task may be running on the tab"
        agent = tab.agent
        assert agent is not None and agent._wt_pending, (
            "the resume must find a pending worktree to act on"
        )
        assert not agent._pending_review, (
            "the task succeeded, so its work is complete and eligible "
            "for the silent finalize"
        )


class TestAutoCommitSuppressesMergeUIOnResume(_Base):
    """Resuming an auto-commit chat must not open a hunk review."""

    def test_resume_with_autocommit_on_does_not_open_merge_ui(self) -> None:
        """The reported defect, end to end.

        The fixture reproduces the *state* the defect needs, not the
        route the user took to it: a worktree task runs, and the tab is
        then left holding a pending branch with the auto-commit toggle
        ON — exactly what a post-task auto-merge that could not complete
        leaves behind.  Driving a real failing merge would need a git
        conflict the stub agent cannot produce, and the resume path
        cannot tell the two apart: it reads ``_wt_pending`` and
        ``auto_commit_mode``, nothing else.  Clicking the task in the
        history panel must then finalize the branch silently instead of
        popping the diff/merge UI.

        "No merge UI appeared" alone is too weak to be a regression:
        the buggy code reached the same silence by a different route,
        because the reset it performed cleared ``use_worktree`` and
        :meth:`_present_pending_worktree` returns immediately for a tab
        that is not in worktree mode.  The branch must therefore also
        be shown to have been *finalized* — silence plus a merged,
        cleaned-up branch is a state only the fixed code produces.
        """
        tab_id = "tab-ac-on"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        row = self._latest_row()
        chat_id, task_id = str(row["chat_id"]), str(row["id"])

        # The state a failing post-task auto-merge leaves behind: the
        # branch is still pending and the toolbar toggle is on.
        tab = self.server._get_tab(tab_id)
        tab.auto_commit_mode = True
        self._assert_resume_reaches_the_decision(tab_id)

        self.events.clear()
        self._resume(tab_id, chat_id, task_id)

        types = self._types()
        for ui_event in MERGE_UI_EVENTS:
            assert ui_event not in types, (
                f"auto-commit is ON, yet resuming the chat broadcast "
                f"{ui_event!r}: {types}"
            )
        assert WORKTREE_PROMPT_EVENT not in types, (
            f"auto-commit is ON, yet resuming the chat asked the user to "
            f"merge or discard: {types}"
        )
        assert _kiss_wt_branches(self.repo) == [], (
            "the pending branch was neither reviewed nor finalized; the "
            f"work is stranded: {_kiss_wt_branches(self.repo)}"
        )
        assert Path(self.repo, "agent_out.txt").exists(), (
            "the branch's work never reached the main working tree"
        )

    def test_resume_preserves_the_tabs_autocommit_preference(self) -> None:
        """A history click must not silently switch auto-commit off.

        ``_replay_session`` used to reset ``tab.auto_commit_mode`` to
        ``False`` for every idle tab, so the very next task submitted
        from that tab ran without auto-commit even though the toolbar
        toggle was still on.
        """
        tab_id = "tab-pref"
        self._patch_run()
        self._run_task(tab_id, auto_commit=True)

        row = self._latest_row()
        self._resume(tab_id, str(row["chat_id"]), str(row["id"]))

        tab = self.server._get_tab(tab_id)
        assert tab.auto_commit_mode is True, (
            "resuming a chat must not turn the tab's auto-commit off"
        )

    def test_resume_with_autocommit_off_still_opens_the_review(self) -> None:
        """Regression guard for the opposite direction.

        With auto-commit OFF the pending worktree must still be
        presented to the user — suppressing it would destroy the manual
        review workflow.
        """
        tab_id = "tab-ac-off"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending, (
            "auto-commit OFF must leave the worktree pending for review"
        )

        row = self._latest_row()
        self.events.clear()
        self._resume(tab_id, str(row["chat_id"]), str(row["id"]))

        types = self._types()
        assert any(t in types for t in (*MERGE_UI_EVENTS, WORKTREE_PROMPT_EVENT)), (
            f"auto-commit is OFF, so resuming must still present the "
            f"pending worktree; got: {types}"
        )


class TestAutoCommitFinalizesPendingWorktreeOnResume(_Base):
    """The silent finalization must actually finalize."""

    def test_resume_merges_the_pending_branch(self) -> None:
        """Auto-commit + a pending worktree with changes → merged.

        Silence alone is not enough: the branch must land on the
        original branch and be cleaned up, otherwise the work is lost
        and the worktree leaks.
        """
        tab_id = "tab-finalize"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        assert _kiss_wt_branches(self.repo), "a task branch should exist"

        row = self._latest_row()
        tab.auto_commit_mode = True
        tab.use_worktree = True
        self.events.clear()
        self._resume(tab_id, str(row["chat_id"]), str(row["id"]))

        assert _kiss_wt_branches(self.repo) == [], (
            "auto-commit resume must merge and clean up the task branch; "
            f"left over: {_kiss_wt_branches(self.repo)}"
        )
        assert Path(self.repo, "agent_out.txt").exists(), (
            "the agent's work must be merged into the main working tree"
        )


class TestSilentFinalizeRefusesUnsafeState(_Base):
    """Auto-commit must never merge work that is not finished."""

    def test_failed_task_work_is_reviewed_not_silently_merged(self) -> None:
        """A stopped or failed task keeps its branch, auto-commit or not.

        ``_run_task_inner`` raises ``agent._pending_review`` when the
        task failed or was stopped, and
        :meth:`WorktreeSorcarAgent._preserve_pending_worktree_for_review`
        states the contract: incomplete, unverified work stays on its
        ``kiss/wt-*`` branch and is never merged into the user's branch
        behind their back.  Auto-commit means "do not interrupt me", not
        "publish half-written work" — so the history click must present
        the branch rather than finalize it.
        """
        tab_id = "tab-failed"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        # Exactly what a failed/stopped task leaves behind.
        agent._pending_review = True
        tab.auto_commit_mode = True

        row = self._latest_row()
        self.events.clear()
        self._resume(tab_id, str(row["chat_id"]), str(row["id"]))

        assert _kiss_wt_branches(self.repo), (
            "the unfinished task's branch was destroyed; its work is "
            "unrecoverable"
        )
        assert not Path(self.repo, "agent_out.txt").exists(), (
            "unfinished work was silently merged into the main working "
            "tree — the exact data-loss this guard prevents"
        )
        types = self._types()
        assert any(t in types for t in (*MERGE_UI_EVENTS, WORKTREE_PROMPT_EVENT)), (
            f"the branch was neither merged nor offered to the user, so "
            f"it is now stranded with no way to act on it: {types}"
        )

    def test_running_task_worktree_is_left_alone(self) -> None:
        """A history click must not finalize a live task's worktree.

        The post-task finalize may bypass the busy guard because it runs
        on the very thread that owns ``is_task_active``.  A resume runs
        on an unrelated thread, so the same bypass would let it merge —
        or, with no changes yet, *delete* — the worktree the agent is
        still writing into.

        Presenting the review instead is just as wrong, so the resume
        must be a complete no-op: no branch change, no merge UI, no
        merge/discard prompt, and no claim on ``is_merging`` that would
        make the task's own post-run finalize refuse.
        """
        tab_id = "tab-live"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        branches_before = _kiss_wt_branches(self.repo)
        assert branches_before, "a task branch should exist"

        tab.auto_commit_mode = True
        tab.is_task_active = True
        row = self._latest_row()
        self.events.clear()
        try:
            self._resume(tab_id, str(row["chat_id"]), str(row["id"]))
        finally:
            tab.is_task_active = False

        assert _kiss_wt_branches(self.repo) == branches_before, (
            "a resume finalized the worktree of a task that is still "
            f"running: {branches_before} -> {_kiss_wt_branches(self.repo)}"
        )
        assert not Path(self.repo, "agent_out.txt").exists(), (
            "a running task's in-progress work was merged into the main "
            "working tree"
        )
        types = self._types()
        for ui_event in (*MERGE_UI_EVENTS, WORKTREE_PROMPT_EVENT):
            assert ui_event not in types, (
                f"a live task's worktree was presented to the user "
                f"({ui_event!r}); the review snapshots a half-written "
                f"tree: {types}"
            )
        assert not tab.is_merging, (
            "the resume left an ownership claim on a live tab; the "
            "task's own post-run finalize would now be refused"
        )
        assert agent._wt_pending, (
            "the running task lost its worktree registration"
        )

    def test_empty_worktree_of_a_running_task_is_not_discarded(self) -> None:
        """The live-task no-op must also cover the discard path.

        A task that has not written anything yet has an EMPTY worktree,
        and every fallback for an empty branch destroys it: the silent
        finalize would discard it, and ``_present_pending_worktree``
        auto-discards it too (``discard_if_empty`` defaults to True).
        Whichever route a resume takes, the agent would find its branch
        deleted underneath it mid-task.  Only a complete no-op is safe.
        """
        tab_id = "tab-live-empty"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        # An agent that has not written anything yet.  The file cannot
        # simply be left unwritten: dismissing the review of an EMPTY
        # worktree legitimately discards it, so the branch would be
        # gone before the live-task scenario even starts.  Removing the
        # file afterwards reaches the same state by a route the fixture
        # controls.
        wt_dir = agent._wt_dir
        assert wt_dir is not None
        Path(wt_dir, "agent_out.txt").unlink()
        assert self.server._get_worktree_changed_files(tab_id) == [], (
            "this test needs an empty worktree to exercise the discard "
            "path"
        )
        branches_before = _kiss_wt_branches(self.repo)
        assert branches_before, "a task branch should exist"

        tab.auto_commit_mode = True
        tab.is_task_active = True
        row = self._latest_row()
        self.events.clear()
        try:
            self._resume(tab_id, str(row["chat_id"]), str(row["id"]))
        finally:
            tab.is_task_active = False

        assert _kiss_wt_branches(self.repo) == branches_before, (
            "a resume discarded the empty worktree of a running task: "
            f"{branches_before} -> {_kiss_wt_branches(self.repo)}"
        )
        assert agent._wt_pending, (
            "the running task's worktree registration was cleared"
        )
        types = self._types()
        for ui_event in (*MERGE_UI_EVENTS, WORKTREE_PROMPT_EVENT):
            assert ui_event not in types, (
                f"a live task's empty worktree produced {ui_event!r}: "
                f"{types}"
            )


    def test_a_just_submitted_task_is_not_treated_as_idle(self) -> None:
        """A resume must not steal the tab from a task that is starting.

        ``_cmd_run`` installs ``tab.task_thread`` under the state lock
        and starts it only after releasing that lock, and the worker
        raises ``is_task_active`` later still.  In the window between,
        both flags read False while the user's task is very much real.

        A resume landing there used to claim ``is_merging``; the worker
        then reached its own guard, saw the claim and answered "Cannot
        run a task while merge review is in progress" — silently
        throwing away the task the user had just submitted.  That is
        why ownership is decided by the one shared ``_tab_busy``
        predicate, which counts an installed-but-unstarted thread as
        busy, instead of by reading the two flags directly.
        """
        tab_id = "tab-submitting"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        branches_before = _kiss_wt_branches(self.repo)
        assert branches_before, "a task branch should exist"

        # Exactly the state `_cmd_run` leaves behind between installing
        # the worker thread and starting it.
        tab.auto_commit_mode = True
        tab.task_thread = threading.Thread(target=lambda: None)
        assert not tab.is_task_active and not tab.is_merging, (
            "this test must reproduce the window in which BOTH flags "
            "still read False, or it proves nothing"
        )
        assert tab.task_thread.ident is None, (
            "the thread must not have been started yet"
        )
        row = self._latest_row()
        self.events.clear()
        try:
            self._resume(tab_id, str(row["chat_id"]), str(row["id"]))
        finally:
            tab.task_thread = None

        assert not tab.is_merging, (
            "a resume claimed the tab while a task was starting; the "
            "worker will now refuse the run the user just submitted"
        )
        assert _kiss_wt_branches(self.repo) == branches_before, (
            "a resume finalized the worktree of a task that was still "
            f"starting: {branches_before} -> {_kiss_wt_branches(self.repo)}"
        )
        assert agent._wt_pending, (
            "the starting task lost its worktree registration"
        )
        types = self._types()
        for ui_event in (*MERGE_UI_EVENTS, WORKTREE_PROMPT_EVENT):
            assert ui_event not in types, (
                f"a starting task's worktree produced {ui_event!r}: "
                f"{types}"
            )


class TestNextTaskDoesNotPublishParkedWork(_Base):
    """Submitting a new task must not merge the previous one's branch."""

    def test_new_task_preserves_a_failed_tasks_branch(self) -> None:
        """A stopped task's work stays on its branch across a new task.

        ``_try_setup_worktree`` has to retire the previous worktree
        before minting a new one, and it used to do that by calling
        ``_release_worktree`` unconditionally — which squash-merges.
        So the same unverified work that
        :meth:`WorktreeSorcarAgent._preserve_pending_worktree_for_review`
        refuses to publish when a tab closes got published anyway the
        moment the user typed a second prompt.  Typing a new prompt is
        not a decision about the old task's work.
        """
        tab_id = "tab-next-task"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        # Exactly what a failed or stopped task leaves behind.
        agent._pending_review = True
        parked = list(_kiss_wt_branches(self.repo))
        assert parked, "the stopped task should have left a branch"

        self._patch_run(filename="second_out.txt")
        self._run_task(tab_id, auto_commit=True)

        assert not Path(self.repo, "agent_out.txt").exists(), (
            "starting a new task published the previous task's "
            "unverified work into the main working tree"
        )
        surviving = _kiss_wt_branches(self.repo)
        for branch in parked:
            assert branch in surviving, (
                f"the parked branch {branch!r} was destroyed by the next "
                f"task; its work is unrecoverable: {surviving}"
            )

    def test_new_chat_preserves_a_failed_tasks_branch(self) -> None:
        """The same policy governs the other retire entry point.

        :meth:`WorktreeSorcarAgent.new_chat` also has to retire the
        previous worktree, and it used to squash-merge unconditionally
        too — so a user who abandoned a failed task by starting a fresh
        chat published its half-written work just the same.
        """
        tab_id = "tab-new-chat"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        agent._pending_review = True
        parked = list(_kiss_wt_branches(self.repo))
        assert parked, "the stopped task should have left a branch"

        agent.new_chat()

        assert not Path(self.repo, "agent_out.txt").exists(), (
            "starting a new chat published the previous task's "
            "unverified work into the main working tree"
        )
        surviving = _kiss_wt_branches(self.repo)
        for branch in parked:
            assert branch in surviving, (
                f"the parked branch {branch!r} was destroyed by the new "
                f"chat; its work is unrecoverable: {surviving}"
            )

    def test_a_stranded_worktree_directory_is_reported_to_the_user(
        self,
    ) -> None:
        """Preserved work the agent could not commit must be announced.

        When the commit inside a preserved worktree is refused — a
        pre-commit hook is the usual culprit — the directory is kept so
        nothing is lost, but the user is the only one who can act on
        it.  Writing that to the log alone strands the work invisibly:
        the release path already surfaces its failures through the
        ``warning`` event, and the preserve path must do the same.
        """
        tab_id = "tab-stranded"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        agent._pending_review = True

        # A change the worktree cannot commit: `commit_all` is refused
        # exactly as a rejecting pre-commit hook would refuse it.
        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "uncommittable.txt").write_text("rejected by the hook\n")
        _run_git(str(wt_dir), "config", "core.hooksPath", str(self.hooks))

        self.events.clear()
        agent.printer = self.server.printer  # type: ignore[attr-defined]
        assert agent._preserve_pending_worktree_for_review()
        agent._flush_warnings(self.server.printer)

        warnings = [
            str(e.get("message", ""))
            for e in self.events
            if e.get("type") == "warning"
        ]
        assert warnings, (
            f"the stranded worktree was never reported to the user; "
            f"only these events were sent: {self._types()}"
        )
        assert any(str(wt_dir) in w for w in warnings), (
            f"the warning must name the directory holding the work so "
            f"the user can recover it: {warnings}"
        )

    def _strand_the_worktree(self, tab_id: str) -> Path:
        """Park a pending worktree holding work that cannot be committed.

        Reproduces the real shape of the problem: the task was stopped
        or failed (``_pending_review``), it left a change behind, and a
        pre-commit hook refuses to commit it.  Whatever retires the
        worktree from here can only keep the directory and tell the
        user where it is.

        Args:
            tab_id: The tab to strand.

        Returns:
            The worktree directory that must survive and be named.
        """
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)
        agent = self.server._get_tab(tab_id).agent
        assert agent is not None and agent._wt_pending
        agent._pending_review = True
        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "uncommittable.txt").write_text("rejected by the hook\n")
        _run_git(str(wt_dir), "config", "core.hooksPath", str(self.hooks))
        return wt_dir

    def _warnings(self) -> list[str]:
        return [
            str(e.get("message", ""))
            for e in self.events
            if e.get("type") == "warning"
        ]

    def test_closing_a_tab_reports_work_it_could_not_commit(self) -> None:
        """Closing the tab is the last chance to say where the work is.

        Tab teardown retires the pending worktree and then destroys the
        tab's printer.  A warning recorded but never flushed dies with
        that printer, so the user closes a tab and is never told their
        stopped task's changes are sitting in a directory only the log
        knows about.  The flush must happen while there is still
        something to say it on.
        """
        tab_id = "tab-closed"
        wt_dir = self._strand_the_worktree(tab_id)

        self.events.clear()
        self.server._handle_command({"type": "closeTab", "tabId": tab_id})

        assert wt_dir.exists(), (
            "the stranded work must be kept on disk; there is nowhere "
            "else it survives"
        )
        warnings = self._warnings()
        assert warnings, (
            f"closing the tab stranded the work silently; the user saw "
            f"only: {self._types()}"
        )
        assert any(str(wt_dir) in w for w in warnings), (
            f"the warning must name the directory holding the work: "
            f"{warnings}"
        )

    def test_uncommittable_work_is_not_reported_as_committed(self) -> None:
        """Recovery instructions must match what actually happened.

        When another task holds the main tree, the pending worktree is
        preserved instead of merged and the user is told to recover it
        with ``git checkout <branch>``.  That is only true when the
        commit succeeded.  With a hook refusing it the work is in the
        worktree directory and *not* on the branch, so the stock
        message sends the user to an empty branch and the note saying
        where the work really is gets overwritten.
        """
        tab_id = "tab-blocked-main"
        wt_dir = self._strand_the_worktree(tab_id)
        agent = self.server._get_tab(tab_id).agent
        assert agent is not None
        branch = agent._wt_branch

        self.events.clear()
        _release_worktree_without_merging(agent, True)
        agent._flush_warnings(self.server.printer)

        warnings = self._warnings()
        assert warnings, "the blocked auto-merge was never reported"
        assert not any("committed on that branch" in w for w in warnings), (
            f"the user was told the work is committed on {branch!r}, but "
            f"the commit was refused and the branch does not carry it: "
            f"{warnings}"
        )
        assert any(str(wt_dir) in w for w in warnings), (
            f"the warning must still name the directory holding the "
            f"work: {warnings}"
        )
        assert any("another task is running" in w for w in warnings), (
            f"the user must also learn why no merge was attempted: "
            f"{warnings}"
        )

    def test_a_stale_warning_does_not_hide_the_recovery_command(
        self,
    ) -> None:
        """The other direction: a commit that worked must say so.

        The warning slot is not a record of what the current preserve
        did.  :meth:`WorktreeSorcarAgent._flush_warnings` deliberately
        puts a warning *back* when its broadcast raises, so text from
        an earlier worktree can still be sitting there.  Deciding "the
        commit must have failed" from a non-empty slot therefore
        swallows the ``git checkout`` recovery command for a branch
        that really does carry the work, and replaces it with a
        directory path that no longer exists.
        """
        tab_id = "tab-stale-warning"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)
        agent = self.server._get_tab(tab_id).agent
        assert agent is not None and agent._wt_pending
        agent._pending_review = True

        # A previous worktree's warning that a failing broadcast put
        # back into the slot — exactly what `_flush_warnings` does.
        stale_dir = Path(self.tmpdir) / "an-older-worktree"
        agent._set_warnings(merge=(
            f"Branch 'kiss/wt-older' still has uncommitted changes. Its "
            f"worktree directory {stale_dir} was kept."
        ))

        class _BrokenPrinter:
            def broadcast(self, event: dict[str, Any]) -> None:
                raise RuntimeError("websocket is gone")

        agent._flush_warnings(_BrokenPrinter())
        with agent._warning_lock:
            assert agent._merge_conflict_warning is not None, (
                "the fixture must leave a restored stale warning behind"
            )

        # This worktree's own commit succeeds: no hook is installed.
        branch = agent._wt_branch
        self.events.clear()
        _release_worktree_without_merging(agent, True)
        agent._flush_warnings(self.server.printer)

        warnings = self._warnings()
        assert warnings, "the blocked auto-merge was never reported"
        assert any(f"git checkout {branch}" in w for w in warnings), (
            f"the commit onto {branch!r} succeeded, but the stale "
            f"warning suppressed the recovery command the user needs: "
            f"{warnings}"
        )
        assert not any(str(stale_dir) in w for w in warnings), (
            f"a directory from an older worktree was reported as the "
            f"home of this task's work: {warnings}"
        )

    def test_nothing_to_preserve_reports_nothing(self) -> None:
        """With no worktree there is no branch to talk about.

        The preserve step returns False when it finds nothing pending.
        Carrying on regardless produced a message about a branch named
        ``None``, and — because there was no outcome to read — pulled
        whatever text happened to be sitting in the warning slot in as
        the description of where "the work" is.  Both halves of that
        sentence would be fiction.
        """
        tab_id = "tab-nothing-pending"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)
        agent = self.server._get_tab(tab_id).agent
        assert agent is not None

        stale_dir = Path(self.tmpdir) / "an-older-worktree"
        stale = (
            f"Branch 'kiss/wt-older' still has uncommitted changes. Its "
            f"worktree directory {stale_dir} was kept."
        )
        agent._set_warnings(merge=stale)
        agent.discard()
        assert not agent._wt_pending, "the fixture must leave nothing pending"

        self.events.clear()
        _release_worktree_without_merging(agent, True)
        agent._flush_warnings(self.server.printer)

        warnings = self._warnings()
        assert not any("Could not auto-merge" in w for w in warnings), (
            f"a worktree that was never there was reported as blocked: "
            f"{warnings}"
        )
        # The stale warning is still owed to the user, so flushing it is
        # right.  What must not happen is this call adopting it: it
        # would then read as the fate of a worktree it never had.
        assert warnings == [stale], (
            f"the pending warning was rewritten by a call that had "
            f"nothing to report: {warnings}"
        )

    def test_new_task_still_merges_a_finished_tasks_branch(self) -> None:
        """The opposite direction: completed work is still published.

        Only ``_pending_review`` work is held back.  A task that ran to
        completion must keep being squash-merged when the next task
        starts, or auto-commit mode would silently stop delivering
        anything.
        """
        tab_id = "tab-next-task-ok"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        assert not agent._pending_review, (
            "the task succeeded, so its work is complete"
        )

        self._patch_run(filename="second_out.txt")
        self._run_task(tab_id, auto_commit=True)

        assert Path(self.repo, "agent_out.txt").exists(), (
            "the previous task's finished work was never merged"
        )


class TestConcurrentResumesClaimTheWorktreeOnce(_Base):
    """One pending worktree, two history clicks, exactly one owner.

    Remote commands are dispatched on a thread pool, so two clicks on
    the same history row — or one from the VS Code sidebar and one from
    the webapp — reach :meth:`_emit_pending_worktree` concurrently.  The
    decision that a worktree is free and the claim on it must therefore
    happen in one atomic step; deciding first and claiming later lets
    both callers believe they own the branch.
    """

    def _resume_twice(self, tab_id: str, chat_id: str, task_id: str) -> None:
        """Fire two ``resumeSession`` commands at once and join them.

        Both threads wait on a barrier immediately before dispatching so
        they enter the server as close to simultaneously as the GIL
        allows.  Exceptions are re-raised in the main thread rather than
        printed to stderr and lost.
        """
        start = threading.Barrier(2)
        failures: list[BaseException] = []

        def click() -> None:
            start.wait(timeout=10)
            try:
                self._resume(tab_id, chat_id, task_id)
            except BaseException as exc:  # pragma: no cover — surfaced below
                failures.append(exc)

        threads = [threading.Thread(target=click) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), "a resume deadlocked"
        if failures:
            raise failures[0]

    def test_two_manual_resumes_open_only_one_review(self) -> None:
        """Auto-commit OFF: the merge view must be broadcast once.

        ``_present_pending_worktree`` starts a review without checking
        who owns the worktree, so when the decision to present is taken
        outside the claim both callers run
        ``_prepare_and_start_merge`` and the frontend receives
        ``merge_data``/``merge_started`` twice for a single branch.  The
        second view replaces the first one's registered merge state,
        throwing away hunk resolutions the user may already have made
        (F4-20).
        """
        tab_id = "tab-race-review"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        assert not tab.is_merging, "the fixture must start with a free worktree"

        row = self._latest_row()
        self.events.clear()
        self._resume_twice(tab_id, str(row["chat_id"]), str(row["id"]))

        types = self._types()
        for ui_event in MERGE_UI_EVENTS:
            assert types.count(ui_event) <= 1, (
                f"two concurrent resumes broadcast {ui_event!r} "
                f"{types.count(ui_event)} times for one worktree; the "
                f"second review discards the first one's hunk "
                f"resolutions: {types}"
            )
        assert any(t in types for t in (*MERGE_UI_EVENTS, WORKTREE_PROMPT_EVENT)), (
            f"auto-commit is OFF, so one of the resumes must still "
            f"present the pending worktree; got: {types}"
        )

    def test_a_review_that_cannot_start_leaves_the_tab_free(self) -> None:
        """The claim must not outlive the attempt that took it.

        Deciding to present the worktree claims ``is_merging`` so a
        second resume cannot open a duplicate review.  That claim is
        speculative: when the review cannot start — no hunks to show,
        the worktree directory gone, ``_prepare_and_start_merge``
        raising — the code falls back to a ``worktree_done`` event and
        no owner exists any more.  Leaving the flag raised would wedge
        the tab: every later task, merge and discard on it is refused,
        and nothing would ever clear it because only
        :meth:`_finish_merge` does, and no merge was ever started.
        """
        tab_id = "tab-review-fallback"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        assert not tab.is_merging

        # A review that cannot start: the fallback path the claim must
        # survive.  `_present_pending_worktree` catches the error and
        # broadcasts `worktree_done` instead.
        real_prepare = self.server._prepare_and_start_merge

        def failing_prepare(*args: Any, **kwargs: Any) -> bool:
            raise RuntimeError("merge review unavailable")

        self.server._prepare_and_start_merge = failing_prepare  # type: ignore[assignment,method-assign]
        try:
            row = self._latest_row()
            self.events.clear()
            self._resume(tab_id, str(row["chat_id"]), str(row["id"]))
        finally:
            self.server._prepare_and_start_merge = real_prepare  # type: ignore[method-assign]

        assert WORKTREE_PROMPT_EVENT in self._types(), (
            f"the fallback event must still reach the user: {self._types()}"
        )
        assert not tab.is_merging, (
            "the speculative claim outlived the review attempt that "
            "took it; the tab is now permanently busy and no later "
            "task, merge or discard on it can ever run"
        )

    def test_two_autocommit_resumes_finalize_the_branch_once(self) -> None:
        """Auto-commit ON: one probe, one action, claim never dropped.

        The claim used to be released as soon as the changed-files probe
        returned and re-taken inside
        :meth:`_handle_worktree_action`.  A second resume slipping into
        that window probes a worktree the first one is about to merge
        and then acts on the stale answer — merging twice, or discarding
        a branch whose files were already taken.

        A sleep cannot demonstrate that gap reliably: it is microseconds
        wide and the loser has to hit it exactly.  The invariant is
        observed directly instead.  The probe and the action are the two
        ends of the window, so both are instrumented to record whether
        the tab was claimed at that moment.  ``is_merging`` must read
        ``True`` at *both* ends, which is only possible if the claim was
        never dropped in between — the released-and-reacquired shape
        enters :meth:`_handle_worktree_action` with the flag clear.

        The instrumentation only observes; both wrappers forward to the
        real implementation and return its result unchanged.
        """
        tab_id = "tab-race-finalize"
        self._patch_run()
        self._run_task(tab_id, auto_commit=False)
        self._finish_open_review(tab_id)

        tab = self.server._get_tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        tab.auto_commit_mode = True

        real_probe = self.server._get_worktree_changed_files
        real_action = self.server._handle_worktree_action
        probes: list[bool] = []
        actions: list[bool] = []

        def instrumented_probe(probe_tab_id: str = "") -> list[str]:
            """Record the claim at the opening edge of the window."""
            probes.append(self.server._get_tab(tab_id).is_merging)
            return real_probe(probe_tab_id)

        def instrumented_action(
            action: str, action_tab_id: str = "", **kwargs: Any,
        ) -> dict[str, Any]:
            """Record the claim at the closing edge of the window."""
            actions.append(self.server._get_tab(tab_id).is_merging)
            return real_action(action, action_tab_id, **kwargs)

        self.server._get_worktree_changed_files = instrumented_probe  # type: ignore[assignment,method-assign]
        self.server._handle_worktree_action = instrumented_action  # type: ignore[assignment,method-assign]
        try:
            row = self._latest_row()
            self.events.clear()
            self._resume_twice(tab_id, str(row["chat_id"]), str(row["id"]))
        finally:
            self.server._get_worktree_changed_files = real_probe  # type: ignore[method-assign]
            self.server._handle_worktree_action = real_action  # type: ignore[method-assign]

        assert len(probes) == 1, (
            f"the worktree was probed {len(probes)} times; the second "
            f"resume entered the claim window and acted on an answer "
            f"the first one was already invalidating"
        )
        assert len(actions) == 1, (
            f"the worktree was acted on {len(actions)} times: {actions}"
        )
        assert probes == [True], (
            "the changed-files probe ran without the tab claimed as "
            "merging, so any concurrent caller was free to walk in"
        )
        assert actions == [True], (
            "the claim was released between the changed-files probe and "
            "the merge it selected; a resume arriving in that window "
            "acts on an answer that is already stale"
        )
        types = self._types()
        for ui_event in MERGE_UI_EVENTS:
            assert ui_event not in types, (
                f"auto-commit is ON, yet a concurrent resume broadcast "
                f"{ui_event!r}: {types}"
            )
        assert types.count("worktree_result") == 1, (
            f"the branch must be finalized exactly once; got "
            f"{types.count('worktree_result')} results: {types}"
        )
        assert _kiss_wt_branches(self.repo) == [], (
            "the task branch must be merged and cleaned up"
        )
        assert Path(self.repo, "agent_out.txt").exists(), (
            "the agent's work must be merged into the main working tree"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
