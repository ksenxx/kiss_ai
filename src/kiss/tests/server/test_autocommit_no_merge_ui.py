# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Resuming a chat or starting the next task never publishes work.

A task isolates its edits with the ``create_worktree`` tool and
publishes them with ``commit_and_merge``.  A task that finished without
the latter — or that failed or was stopped — leaves a pending
``kiss/wt-*`` worktree.  Two daemon paths then touch that worktree
without the user having decided anything about it:

* the session-resume path (:meth:`_ServerCommandsMixin._replay_session`
  → :meth:`_MergeFlowMixin._emit_pending_worktree`), which presents it
  as the ``worktree_done`` Merge / Discard buttons;
* the retire path (:meth:`ChatSorcarAgent._retire_previous_worktree`,
  run before a new task, a new chat or a tab close), which commits the
  work to its own branch and keeps it for review.

Neither may merge the work into the user's branch behind their back,
destroy it, present a live task's half-written tree, or leave a stale
ownership claim on the tab.  Every test drives the real
:class:`AgentServer` command handlers against a real git repository;
only the LLM call is replaced by a deterministic stub that creates the
worktree exactly as the real agent's tool call would.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import init_repo, kiss_wt_branches, run_git

#: Events of the removed hunk-by-hunk diff/merge UI; must never reappear.
MERGE_UI_EVENTS = ("merge_data", "merge_started")

#: Event that makes the worktree merge/discard prompt appear.
WORKTREE_PROMPT_EVENT = "worktree_done"


class _Base(unittest.TestCase):
    """A real git repo, an isolated persistence DB and a real server."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-resume-retire-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        init_repo(self.repo)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "history.db"
        _persistence._db_conn = None

        # A hooks dir whose pre-commit always fails, so a test can make
        # `git commit` be refused exactly as a real rejecting hook does.
        self.hooks = Path(self.tmpdir) / "hooks"
        self.hooks.mkdir(parents=True, exist_ok=True)
        hook = self.hooks / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        self.server = AgentServer()
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
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        agent_state.agent_states.clear()
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
        """Make the agent isolate itself in a worktree, write *filename*
        there and succeed without ``commit_and_merge``."""

        def stub_run(_agent: object, **kwargs: object) -> str:
            agent = cast(ChatSorcarAgent, _agent)
            agent._enter_work_dir(str(kwargs.get("work_dir") or ""))
            message = agent.create_worktree()
            assert agent._wt_pending, message
            if filename is not None:
                (Path(agent.work_dir) / filename).write_text("agent output\n")
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def _tab(self, tab_id: str) -> agent_state.AgentState:
        """Return the registered agent state for *tab_id* (must exist)."""
        state = agent_state.find_by_tab(tab_id)
        assert state is not None, f"no agent state registered for {tab_id!r}"
        return state

    def _run_task(self, tab_id: str) -> None:
        self.server._run_task({
            "prompt": "make a change",
            "workDir": self.repo,
            "tabId": tab_id,
            "model": "",
        })
        # In production the run's worker thread dies when `_run_task`
        # returns, so `AgentState.thread_alive()` reads False.  A direct
        # synchronous call leaves the *test* thread installed as
        # `task_thread` (the run's `finally` looks the state up by the
        # pre-rekey `_state_key` and misses it), which would keep the
        # state `busy()` forever.  Reproduce the post-run idle state.
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
            if state is not None and state.task_thread is threading.current_thread():
                state.task_thread = None
                state.is_task_active = False

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


class TestResumePresentsThePendingWorktree(_Base):
    """A history click offers an unmerged worktree to the user."""

    def test_resume_presents_the_merge_discard_prompt(self) -> None:
        """A task that never merged leaves the decision to the user.

        The pending worktree must be presented on resume — suppressing
        it would strand the work with no way to act on it.
        """
        tab_id = "tab-unmerged"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending, (
            "a task that never called commit_and_merge must leave its "
            "worktree pending for review"
        )

        row = self._latest_row()
        self.events.clear()
        self._resume(tab_id, str(row["chat_id"]), str(row["id"]))

        types = self._types()
        assert WORKTREE_PROMPT_EVENT in types, (
            f"resuming must present the pending worktree; got: {types}"
        )
        for ui_event in MERGE_UI_EVENTS:
            assert ui_event not in types, (
                f"the removed diff/merge review reappeared ({ui_event!r}): "
                f"{types}"
            )


class TestResumeRefusesUnsafeState(_Base):
    """A resume must never act on work that is not finished."""

    def test_failed_task_work_is_presented_not_merged(self) -> None:
        """A stopped or failed task keeps its branch.

        A task that failed or was stopped leaves its worktree pending, and
        :meth:`ChatSorcarAgent._preserve_pending_worktree_for_review`
        states the contract: incomplete, unverified work stays on its
        ``kiss/wt-*`` branch and is never merged into the user's branch
        behind their back — so the history click must present the
        branch rather than finalize it.
        """
        tab_id = "tab-failed"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        # Exactly what a failed/stopped task leaves behind.

        row = self._latest_row()
        self.events.clear()
        self._resume(tab_id, str(row["chat_id"]), str(row["id"]))

        assert kiss_wt_branches(self.repo), (
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
        """A history click must not touch a live task's worktree.

        The post-task presentation may bypass the busy guard because it
        runs on the very thread that owns ``is_task_active``.  A resume
        runs on an unrelated thread, so the same bypass would let it
        present — or, with no changes yet, *delete* — the worktree the
        agent is still writing into.

        The resume must be a complete no-op: no branch change, no merge
        UI, no merge/discard prompt, and no claim on ``is_merging`` that
        would make the task's own post-run presentation refuse.
        """
        tab_id = "tab-live"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        branches_before = kiss_wt_branches(self.repo)
        assert branches_before, "a task branch should exist"

        tab.is_task_active = True
        row = self._latest_row()
        self.events.clear()
        try:
            self._resume(tab_id, str(row["chat_id"]), str(row["id"]))
        finally:
            tab.is_task_active = False

        assert kiss_wt_branches(self.repo) == branches_before, (
            "a resume finalized the worktree of a task that is still "
            f"running: {branches_before} -> {kiss_wt_branches(self.repo)}"
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
        and ``_present_pending_worktree`` auto-discards an empty branch
        (``discard_if_empty`` defaults to True), so a resume that got
        that far would delete the branch underneath the agent mid-task.
        Only a complete no-op is safe.
        """
        tab_id = "tab-live-empty"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
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
        branches_before = kiss_wt_branches(self.repo)
        assert branches_before, "a task branch should exist"

        tab.is_task_active = True
        row = self._latest_row()
        self.events.clear()
        try:
            self._resume(tab_id, str(row["chat_id"]), str(row["id"]))
        finally:
            tab.is_task_active = False

        assert kiss_wt_branches(self.repo) == branches_before, (
            "a resume discarded the empty worktree of a running task: "
            f"{branches_before} -> {kiss_wt_branches(self.repo)}"
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
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        branches_before = kiss_wt_branches(self.repo)
        assert branches_before, "a task branch should exist"

        # Exactly the state `_cmd_run` leaves behind between installing
        # the worker thread and starting it.
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
        assert kiss_wt_branches(self.repo) == branches_before, (
            "a resume finalized the worktree of a task that was still "
            f"starting: {branches_before} -> {kiss_wt_branches(self.repo)}"
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

        ``run`` has to retire the previous worktree before the new
        task can mint its own, and it used to do that by squash-merging
        unconditionally.  So the same unverified work that
        :meth:`ChatSorcarAgent._preserve_pending_worktree_for_review`
        refuses to publish when a tab closes got published anyway the
        moment the user typed a second prompt.  Typing a new prompt is
        not a decision about the old task's work.
        """
        tab_id = "tab-next-task"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        # Exactly what a failed or stopped task leaves behind.
        parked = list(kiss_wt_branches(self.repo))
        assert parked, "the stopped task should have left a branch"

        self._patch_run(filename="second_out.txt")
        self._run_task(tab_id)

        assert not Path(self.repo, "agent_out.txt").exists(), (
            "starting a new task published the previous task's "
            "unverified work into the main working tree"
        )
        surviving = kiss_wt_branches(self.repo)
        for branch in parked:
            assert branch in surviving, (
                f"the parked branch {branch!r} was destroyed by the next "
                f"task; its work is unrecoverable: {surviving}"
            )

    def test_new_chat_preserves_a_failed_tasks_branch(self) -> None:
        """The same policy governs the other retire entry point.

        :meth:`ChatSorcarAgent.new_chat` also has to retire the
        previous worktree, and it used to squash-merge unconditionally
        too — so a user who abandoned a failed task by starting a fresh
        chat published its half-written work just the same.
        """
        tab_id = "tab-new-chat"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        parked = list(kiss_wt_branches(self.repo))
        assert parked, "the stopped task should have left a branch"

        agent.new_chat()

        assert not Path(self.repo, "agent_out.txt").exists(), (
            "starting a new chat published the previous task's "
            "unverified work into the main working tree"
        )
        surviving = kiss_wt_branches(self.repo)
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
        it.  Writing that to the log alone strands the work invisibly;
        the preserve path must surface it through the ``warning`` event.
        """
        tab_id = "tab-stranded"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending

        # A change the worktree cannot commit: `commit_all` is refused
        # exactly as a rejecting pre-commit hook would refuse it.
        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "uncommittable.txt").write_text("rejected by the hook\n")
        run_git(str(wt_dir), "config", "core.hooksPath", str(self.hooks))

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
        or failed, it left a change behind, and a
        pre-commit hook refuses to commit it.  Whatever retires the
        worktree from here can only keep the directory and tell the
        user where it is.

        Args:
            tab_id: The tab to strand.

        Returns:
            The worktree directory that must survive and be named.
        """
        self._patch_run()
        self._run_task(tab_id)
        agent = self._tab(tab_id).agent
        assert agent is not None and agent._wt_pending
        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "uncommittable.txt").write_text("rejected by the hook\n")
        run_git(str(wt_dir), "config", "core.hooksPath", str(self.hooks))
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

        When the retire preserves a pending worktree, the user is told
        the work was committed to its branch.  That is only true when
        the commit succeeded.  With a hook refusing it the work is in
        the worktree directory and *not* on the branch, so the stock
        message sends the user to an empty branch and the note saying
        where the work really is gets overwritten.
        """
        tab_id = "tab-hook-refused"
        wt_dir = self._strand_the_worktree(tab_id)
        agent = self._tab(tab_id).agent
        assert agent is not None
        branch = agent._wt_branch

        self.events.clear()
        agent._retire_previous_worktree()
        agent._flush_warnings(self.server.printer)

        warnings = self._warnings()
        assert warnings, "the refused commit was never reported"
        assert not any("was committed to branch" in w for w in warnings), (
            f"the user was told the work is committed on {branch!r}, but "
            f"the commit was refused and the branch does not carry it: "
            f"{warnings}"
        )
        assert any(str(wt_dir) in w for w in warnings), (
            f"the warning must still name the directory holding the "
            f"work: {warnings}"
        )
        assert wt_dir.exists(), (
            "the directory named in the warning must still hold the work"
        )

    def test_a_stale_warning_does_not_hide_the_recovery_command(
        self,
    ) -> None:
        """The other direction: a commit that worked must say so.

        The warning slot is not a record of what the current preserve
        did.  :meth:`ChatSorcarAgent._flush_warnings` deliberately
        puts a warning *back* when its broadcast raises, so text from
        an earlier worktree can still be sitting there.  Deciding "the
        commit must have failed" from a non-empty slot therefore
        swallows the branch the work really is on and replaces it with
        a directory path that no longer exists.
        """
        tab_id = "tab-stale-warning"
        self._patch_run()
        self._run_task(tab_id)
        agent = self._tab(tab_id).agent
        assert agent is not None and agent._wt_pending

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
        agent._retire_previous_worktree()
        agent._flush_warnings(self.server.printer)

        warnings = self._warnings()
        assert warnings, "the preserved work was never reported"
        assert any(f"committed to branch '{branch}'" in w for w in warnings), (
            f"the commit onto {branch!r} succeeded, but the stale "
            f"warning suppressed the branch the user needs: {warnings}"
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
        self._run_task(tab_id)
        agent = self._tab(tab_id).agent
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
        agent._retire_previous_worktree()
        agent._flush_warnings(self.server.printer)

        warnings = self._warnings()
        assert not any("previous task" in w for w in warnings), (
            f"a worktree that was never there was reported as preserved: "
            f"{warnings}"
        )
        # The stale warning is still owed to the user, so flushing it is
        # right.  What must not happen is this call adopting it: it
        # would then read as the fate of a worktree it never had.
        assert warnings == [stale], (
            f"the pending warning was rewritten by a call that had "
            f"nothing to report: {warnings}"
        )

    def test_new_task_preserves_a_finished_tasks_branch(self) -> None:
        """Finished-but-unmerged work is held back just the same.

        Only ``commit_and_merge`` or the user's Merge button publishes a
        worktree.  A task that ran to completion without merging left
        the decision to the user, so the next task must park its branch
        rather than squash-merge it.
        """
        tab_id = "tab-next-task-ok"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        assert agent._wt_branch is not None
        assert GitWorktreeOps.load_preserve_marker(Path(self.repo), agent._wt_branch), (
            "the post-run path must mark an unmerged worktree as preserved "
            "even though the task succeeded"
        )
        parked = list(kiss_wt_branches(self.repo))
        assert parked, "the finished task should have left a branch"

        self._patch_run(filename="second_out.txt")
        self._run_task(tab_id)

        assert not Path(self.repo, "agent_out.txt").exists(), (
            "starting a new task published the previous task's unmerged "
            "work into the main working tree"
        )
        surviving = kiss_wt_branches(self.repo)
        for branch in parked:
            assert branch in surviving, (
                f"the parked branch {branch!r} was destroyed by the next "
                f"task: {surviving}"
            )


class TestConcurrentResumesClaimTheWorktreeOnce(_Base):
    """One pending worktree, two history clicks, exactly one owner.

    Remote commands are dispatched on a thread pool, so two clicks on
    the same history row — or one from each of two browser windows —
    reach :meth:`_emit_pending_worktree` concurrently.  The
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

    def test_two_manual_resumes_keep_the_branch_and_release_the_tab(
        self,
    ) -> None:
        """Concurrent resumes present, never finalize.

        The presentation claim (``is_merging``, taken atomically inside
        :meth:`_emit_pending_worktree`) exists so two simultaneous
        resumes cannot race each other into the presentation's
        empty-branch auto-discard (F4-20).  Whatever interleaving the
        scheduler picks, the outcome must be: the ``worktree_done``
        prompt reaches the user, the branch survives untouched for the
        explicit Merge / Discard decision, and no resume leaves a stale
        ownership claim behind.
        """
        tab_id = "tab-race-review"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        assert not tab.is_merging, "the fixture must start with a free worktree"
        branches_before = kiss_wt_branches(self.repo)
        assert branches_before, "a task branch should exist"

        row = self._latest_row()
        self.events.clear()
        self._resume_twice(tab_id, str(row["chat_id"]), str(row["id"]))

        types = self._types()
        assert WORKTREE_PROMPT_EVENT in types, (
            f"one of the resumes must present the pending worktree; "
            f"got: {types}"
        )
        assert "worktree_result" not in types, (
            f"a resume finalized a worktree the user never decided on: "
            f"{types}"
        )
        assert kiss_wt_branches(self.repo) == branches_before, (
            "concurrent resumes destroyed the pending branch: "
            f"{branches_before} -> {kiss_wt_branches(self.repo)}"
        )
        assert agent._wt_pending, (
            "a resume cleared the worktree registration without the "
            "user's Merge / Discard decision"
        )
        assert not tab.is_merging, (
            "a resume left a stale ownership claim on the tab; every "
            "later task, merge and discard on it is now refused"
        )

    def test_a_presentation_that_fails_leaves_the_tab_free(self) -> None:
        """The claim must not outlive the attempt that took it.

        Deciding to present the worktree claims ``is_merging`` so a
        second resume cannot race the presentation.  That claim is
        speculative: when the presentation itself blows up — here the
        changed-files probe raising — no owner exists any more.
        Leaving the flag raised would wedge the tab: every later task,
        merge and discard on it is refused, and nothing would ever
        clear it.
        """
        tab_id = "tab-review-fallback"
        self._patch_run()
        self._run_task(tab_id)

        tab = self._tab(tab_id)
        agent = tab.agent
        assert agent is not None and agent._wt_pending
        assert not tab.is_merging

        # A presentation that cannot run: the probe it starts with
        # raises.  `_emit_pending_worktree` must still release the
        # claim it took, via its `finally`.
        real_probe = self.server._get_worktree_changed_files

        def failing_probe(*args: Any, **kwargs: Any) -> list[str]:
            raise RuntimeError("worktree probe unavailable")

        self.server._get_worktree_changed_files = failing_probe  # type: ignore[assignment,method-assign]
        try:
            row = self._latest_row()
            self.events.clear()
            with self.assertRaises(RuntimeError):
                self._resume(tab_id, str(row["chat_id"]), str(row["id"]))
        finally:
            self.server._get_worktree_changed_files = real_probe  # type: ignore[method-assign]

        assert not tab.is_merging, (
            "the speculative claim outlived the presentation attempt "
            "that took it; the tab is now permanently busy and no later "
            "task, merge or discard on it can ever run"
        )
        assert agent._wt_pending, (
            "the failed presentation must leave the worktree pending"
        )

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
