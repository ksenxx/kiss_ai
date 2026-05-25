"""Task-runner mixin for the VS Code server.

Implements the background-thread task lifecycle: ``_run_task`` (status
broadcasts) and ``_run_task_inner`` (pre/post snapshots, agent
invocation, merge-view preparation, persistence).  Also hosts the
cooperative-stop machinery and the ``ask_user_question`` callback.

Split out of ``server.py`` for organisation.
"""

from __future__ import annotations

import base64
import ctypes
import logging
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kiss.agents.sorcar.running_agent_state import _RunningAgentState
    from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, repo_lock
from kiss.agents.sorcar.persistence import (
    _append_chat_event,
    _save_task_extra,
    _save_task_result,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState, parse_task_tags
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _parse_diff_hunks,
    _save_untracked_base,
    _snapshot_files,
)
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import get_available_models

logger = logging.getLogger(__name__)


class _TaskRunnerMixin:
    """Task-lifecycle methods (run, stop, user-question callback)."""

    if TYPE_CHECKING:
        printer: BaseBrowserPrinter
        work_dir: str
        _state_lock: threading.RLock

        def _get_tab(self, tab_id: str) -> _RunningAgentState: ...
        def _any_non_wt_running(self) -> bool: ...
        def _dispose_if_closed(self, tab_id: str) -> None: ...
        def _prepare_and_start_merge(
            self,
            work_dir: str,
            pre_hunks: dict[str, list[tuple[int, int, int, int]]] | None = None,
            pre_untracked: set[str] | None = None,
            pre_file_hashes: dict[str, str] | None = None,
            base_ref: str = "HEAD",
            tab_id: str = "",
        ) -> bool: ...
        def _main_dirty_files(self) -> list[str]: ...
        def _broadcast_autocommit_prompt(self, tab_id: str) -> None: ...
        def _handle_autocommit_action(
            self, action: str, tab_id: str = "",
        ) -> None: ...
        def _handle_worktree_action(
            self, action: str, tab_id: str = "", *, internal: bool = False,
        ) -> dict[str, Any]: ...
        def _present_pending_worktree(
            self, tab_id: str, *, try_merge_review: bool,
            discard_if_empty: bool = True,
        ) -> None: ...
        def _get_worktree_changed_files(self, tab_id: str = "") -> list[str]: ...
        def _extract_result_summary(self) -> str: ...
        def _generate_followup_async(
            self, task: str, result: str, task_id: int | None,
        ) -> None: ...

    def _run_task(self, cmd: dict[str, Any]) -> None:
        """Run the agent with the given task.

        An outer try/finally guarantees that ``status: running: False``
        is **always** broadcast when this method exits, regardless of
        which code-path is taken.
        """
        tab_id = cmd.get("tabId", "")
        try:
            self.printer.broadcast(
                {"type": "status", "running": True, "tabId": tab_id},
            )
            self._run_task_inner(cmd)
        finally:
            with self._state_lock:
                tab = _RunningAgentState.running_agent_states.get(tab_id)
                if tab is not None:
                    tab.task_thread = None
                    tab.stop_event = None
                    tab.user_answer_queue = None
                    tab.is_task_active = False
                    tab.is_running_non_wt = False
                    # Dispose the transient agent — a fresh one is
                    # built per task in ``_CommandsMixin._cmd_run``.
                    # Each worktree task creates a brand-new worktree
                    # and branch (independent of chat id), so there is
                    # no cross-task restoration of worktree state.
                    if tab.agent is not None:
                        tab.last_task_id = (
                            tab.agent._last_task_id or tab.last_task_id
                        )
                        tab.agent = None
                self.printer.broadcast(
                    {"type": "status", "running": False, "tabId": tab_id},
                )
            # If the user clicked closeTab while this task was still
            # running, dispose the now-idle _RunningAgentState.  No-op otherwise.
            self._dispose_if_closed(tab_id)

    @staticmethod
    def _capture_pre_snapshot(
        work_dir: str, repo: Path | None, tab_id: str,
    ) -> tuple[
        str | None,
        dict[str, list[tuple[int, int, int, int]]],
        set[str],
        dict[str, str] | None,
    ]:
        """Capture pre-task git snapshot for non-worktree merge view.

        When *repo* is not None, acquires ``repo_lock`` for atomicity.

        Args:
            work_dir: Repository root directory.
            repo: Repo root Path (None when not in a git repo).
            tab_id: Frontend tab identifier for per-tab isolation.

        Returns:
            ``(head_sha, hunks, untracked, file_hashes)`` tuple.
        """
        def _do_snapshot() -> tuple[
            str | None,
            dict[str, list[tuple[int, int, int, int]]],
            set[str],
            dict[str, str] | None,
        ]:
            head = GitWorktreeOps.head_sha(repo) if repo else None
            hunks = _parse_diff_hunks(work_dir)
            untracked = _capture_untracked(work_dir)
            hashes = _snapshot_files(
                work_dir, set(hunks.keys()) | untracked,
            )
            _save_untracked_base(
                work_dir, untracked | set(hunks.keys()), tab_id=tab_id,
            )
            return head, hunks, untracked, hashes

        if repo:
            with repo_lock(repo):
                return _do_snapshot()
        return _do_snapshot()

    def _run_task_inner(self, cmd: dict[str, Any]) -> None:
        """Inner implementation of _run_task (without the status guarantee)."""
        prompt = cmd.get("prompt", "")
        work_dir = cmd.get("workDir") or self.work_dir
        active_file = cmd.get("activeFile")
        raw_attachments = cmd.get("attachments", [])

        attachments: list[Attachment] | None = None
        if raw_attachments:
            attachments = []
            for att in raw_attachments:
                data_b64 = att.get("data", "")
                mime = att.get("mimeType", "application/octet-stream")
                data = base64.b64decode(data_b64)
                attachments.append(Attachment(data=data, mime_type=mime))

        tab_id = cmd.get("tabId", "")
        tab = self._get_tab(tab_id)
        model = cmd.get("model") or tab.selected_model

        # Build the per-task agent now (the previous agent was
        # disposed at the end of the prior ``_run_task`` so there is
        # no long-lived per-tab agent across distinct task
        # executions).  Tests that need to inject a stub agent (e.g.
        # patch ``tab.agent.run``) can pre-populate ``tab.agent``
        # before calling ``_run_task`` — we honour any agent the
        # caller has already attached.
        if tab.agent is None:
            agent = WorktreeSorcarAgent("Sorcar VS Code")
            tab.agent = agent
        # Sync the agent's chat id to the tab's chat id BEFORE the run
        # starts.  ``_RunningAgentState._get_tab`` eagerly populates
        # ``tab.agent`` (for merge / discard / worktree state callers
        # that read it out-of-task) — at the moment of that eager
        # creation the tab may have had an empty ``chat_id``, e.g.
        # because ``_replay_session`` hasn't yet associated the
        # resumed history row.  Without this sync the agent's stale
        # ``_chat_id == ""`` would survive into
        # :meth:`ChatSorcarAgent.run`, which would mint a fresh uuid
        # and ``build_chat_prompt`` would query history for that
        # never-seen uuid — finding nothing and sending the LLM no
        # prior context.  ``_cmd_run`` guarantees ``tab.chat_id`` is
        # populated (either preserved from ``_replay_session`` or
        # freshly minted) before this thread runs.
        if tab.chat_id:
            tab.agent._chat_id = tab.chat_id
        tab.chat_id = tab.agent.chat_id or tab.chat_id

        available = get_available_models()
        if not available or (model and model not in available):
            no_model_msg = (
                "No model available.  Set at least one API key in the environment."
            )
            self.printer.broadcast({
                "type": "result",
                "text": no_model_msg,
                "success": False,
                "total_tokens": 0,
                "cost": "$0.0000",
                "step_count": 0,
            })
            return

        with self._state_lock:
            if tab.is_merging:
                self.printer.broadcast(
                    {
                        "type": "error",
                        "text": "Cannot run a task while merge review is in progress."
                        " Accept or reject all changes first.",
                        "tabId": tab_id,
                    }
                )
                return
            tab.use_worktree = bool(cmd.get("useWorktree", False))
            tab.use_parallel = bool(cmd.get("useParallel", False))
            # Mirror the "Auto commit" menu toggle for this submit.
            # When ON the post-task block skips the interactive
            # merge/diff workflow and auto-commits agent changes
            # (and auto-merges the worktree in worktree mode).
            tab.auto_commit_mode = bool(cmd.get("autoCommit", False))
            tab.is_task_active = True
            stop_event = tab.stop_event
            use_worktree = tab.use_worktree
        self.printer._thread_local.stop_event = stop_event

        # ``chat_id`` initialisation and the ``clear`` broadcast now
        # happen synchronously in ``_cmd_run`` (before the worker
        # thread starts) so the extension layer's chat_id → tab_id
        # index is populated before any subsequent command races the
        # worker thread.  See ``_CommandsMixin._cmd_run``.

        pre_hunks: dict[str, list[tuple[int, int, int, int]]] = {}
        pre_untracked: set[str] = set()
        pre_file_hashes: dict[str, str] | None = None
        pre_head_sha: str | None = None
        if not use_worktree:
            # RACE-1 fix: atomically check the worktree-merge guard
            # AND mark this tab as non-wt active.  Splitting the
            # check and the ``is_running_non_wt = True`` set into two
            # separate ``_state_lock`` acquisitions left a TOCTOU
            # window in which a worktree merge handler could set
            # ``is_merging = True`` between the check and the set,
            # then proceed to merge while this tab's agent began
            # writing to the main tree.  Holding the lock across
            # both eliminates that window.  Holding ``repo_lock``
            # briefly here additionally blocks the task-start path
            # behind any in-flight ``_handle_worktree_action`` /
            # ``_try_setup_worktree`` body on the same repo.
            repo_for_guard = GitWorktreeOps.discover_repo(Path(work_dir))
            guard_lock: Any = (
                repo_lock(repo_for_guard) if repo_for_guard else None
            )
            if guard_lock is not None:
                guard_lock.acquire()
            try:
                with self._state_lock:
                    if any(
                        t.is_merging and t.use_worktree
                        for t in _RunningAgentState.running_agent_states.values()
                    ):
                        tab.is_task_active = False
                        self.printer.broadcast({
                            "type": "error",
                            "text": "A worktree merge is in progress. "
                            "Wait for it to finish before starting a task.",
                            "tabId": tab_id,
                        })
                        return
                    tab.is_running_non_wt = True
                    deferred = tab.deferred_snapshot
            finally:
                if guard_lock is not None:
                    guard_lock.release()
            if deferred is not None:
                pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
                    deferred
                )
            else:
                try:
                    repo = GitWorktreeOps.discover_repo(Path(work_dir))
                    pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
                        self._capture_pre_snapshot(work_dir, repo, tab_id)
                    )
                except BaseException:
                    with self._state_lock:
                        tab.is_running_non_wt = False
                    raise

        if use_worktree and tab.agent._wt_pending:
            with self._state_lock:
                if self._any_non_wt_running():
                    tab.agent._merge_conflict_warning = (
                        f"Could not auto-merge branch "
                        f"'{tab.agent._wt_branch}' because another "
                        "task is running on the main working tree. "
                        "The branch is preserved for manual resolution."
                    )
                    tab.agent._wt = None

        result_summary = "Agent Failed Abruptly"
        task_end_event: dict[str, Any] | None = None
        # ``agent_returned`` captures the YAML string returned by
        # ``tab.agent.run``.  ``WorktreeSorcarAgent.run`` catches
        # ``Exception`` raised by the inner agent and converts it into
        # a ``success: false`` YAML rather than re-raising — so a
        # caught-and-suppressed failure leaves us with
        # ``task_end_event = "task_done"`` even though the user-visible
        # outcome is a failure.  The finally block re-parses this
        # string to recover the true failure signal and route the
        # post-task workflow through the interactive diff/merge view.
        agent_returned: str = ""
        try:
            # ``start_recording`` / ``_persist_agents`` registration
            # / subscriber wiring is now owned by
            # :meth:`ChatSorcarAgent.run` (it has the ``task_id`` at
            # allocation time, which the task_runner does not).  We
            # only need to pass the initial tab id via
            # ``_subscribe_tab_id`` so the agent subscribes that tab
            # to the task once the id exists.
            tab.task_history_id = None
            subtasks = parse_task_tags(prompt)
            from kiss.agents.vscode.vscode_config import load_config as _load_cfg

            _vcfg = _load_cfg()
            _cfg_budget = float(_vcfg.get("max_budget", 100))
            _cfg_web = _vcfg.get("use_web_browser", True)

            from kiss.agents.vscode.vscode_config import (
                build_model_config,
            )

            _model_config = build_model_config(_vcfg)

            for task_prompt in subtasks:
                # Record the raw user prompt so the post-task
                # auto-commit hooks can incorporate the user's intent
                # in the generated commit message (see
                # :meth:`_MergeFlowMixin._handle_autocommit_action`).
                tab.last_user_prompt = task_prompt
                try:
                    agent_returned = tab.agent.run(
                        prompt_template=task_prompt,
                        model_name=model,
                        work_dir=work_dir,
                        printer=self.printer,
                        current_editor_file=active_file,
                        attachments=attachments,
                        ask_user_question_callback=self._ask_user_question,
                        is_parallel=tab.use_parallel,
                        use_worktree=use_worktree,
                        max_budget=_cfg_budget,
                        web_tools=_cfg_web,
                        model_config=_model_config,
                        _skip_persistence=True,
                        _subscribe_tab_id=tab_id,
                    )
                    # Prefer the summary embedded in the YAML the agent
                    # returned: ``ChatSorcarAgent.run``'s ``finally``
                    # block has already called ``stop_recording`` by
                    # the time control returns here, which pops the
                    # recording buffer and clears the thread-local
                    # ``task_id``.  ``_extract_result_summary`` would
                    # then peek into an empty recording and return ""
                    # for every task — causing "No summary available"
                    # to be persisted into ``task_history.result`` and
                    # later surfaced as the prior-task result inside
                    # the next task's ``build_chat_prompt`` preamble.
                    from kiss.core.printer import (
                        parse_result_yaml as _parse_run_yaml,
                    )
                    _run_parsed = (
                        _parse_run_yaml(agent_returned)
                        if agent_returned else None
                    )
                    if _run_parsed and _run_parsed.get("summary"):
                        result_summary = str(_run_parsed["summary"])
                    else:
                        result_summary = (
                            self._extract_result_summary()
                            or "No summary available"
                        )
                    task_end_event = {"type": "task_done"}
                except KeyboardInterrupt:
                    result_summary = "Task stopped by user"
                    task_end_event = {"type": "task_stopped"}
                except Exception as e:
                    result_summary = f"Task failed: {e}"
                    task_end_event = {"type": "task_error", "text": str(e)}
                else:
                    continue
                finally:
                    tab.task_history_id = tab.agent._last_task_id
                self.printer.broadcast({
                    "type": "result",
                    "text": result_summary,
                    "success": False,
                    "total_tokens": tab.agent.total_tokens_used,
                    "cost": f"${tab.agent.budget_used:.4f}",
                    "step_count": tab.agent.step_count,
                    "tabId": tab_id,
                })
                break
        except BaseException as _outer_exc:
            # Catches every flow that the inner per-subtask handlers
            # do NOT cover:
            #
            # * ``BaseException`` subclasses that are not ``Exception``
            #   and not ``KeyboardInterrupt`` (``SystemExit``,
            #   ``GeneratorExit``, ``asyncio.CancelledError`` on
            #   3.11+, etc.) propagated out of ``tab.agent.run``.  The
            #   inner ``try`` does not match them and Python unwinds
            #   straight through to here, leaving ``result_summary``
            #   at its initial ``"Agent Failed Abruptly"`` sentinel.
            # * Any interrupt that arrives BETWEEN ``try:`` and the
            #   for-loop (e.g. ``_force_stop_thread`` raising
            #   ``KeyboardInterrupt`` via
            #   ``PyThreadState_SetAsyncExc``).
            #
            # Without this rewrite the cleanup finally would persist
            # the sentinel string into ``task_history.result``, which
            # the history sidebar surfaces verbatim as "Agent Failed
            # Abruptly" — the bug we are fixing.  When the inner
            # handlers already produced a meaningful
            # ``result_summary`` (e.g. broadcast inside the loop
            # raised) we preserve theirs.
            if result_summary == "Agent Failed Abruptly":
                if isinstance(_outer_exc, KeyboardInterrupt):
                    result_summary = "Task stopped by user"
                    task_end_event = task_end_event or {"type": "task_stopped"}
                else:
                    _exc_name = type(_outer_exc).__name__
                    result_summary = f"Task failed: {_exc_name}: {_outer_exc}"
                    task_end_event = task_end_event or {
                        "type": "task_error",
                        "text": f"{_exc_name}: {_outer_exc}",
                    }
            else:
                task_end_event = task_end_event or {"type": "task_stopped"}
        finally:
            try:
                # RACE-3 fix: keep ``is_task_active = True`` through
                # the post-task cleanup so a concurrent
                # ``_handle_worktree_action`` (driven by a misclick
                # on the merge/discard button) cannot pass its
                # ``_check_worktree_busy`` guard and mutate
                # ``tab.agent._wt`` while this thread is still
                # inside ``_present_pending_worktree``.  The flag is
                # cleared by ``_finalize_task_active`` at the very
                # end of the cleanup block (and again in the BaseException
                # branch below to keep failure recovery monotonic).
                # When the task ended in failure or user-stop, behave
                # as if the "Auto commit" toggle were OFF for the
                # post-task merge workflow.  Rationale: auto-commit is
                # a fire-and-forget convenience that assumes the agent
                # produced a clean, intentional set of changes.  On
                # ``task_error`` or ``task_stopped`` the changes are
                # by definition incomplete or unverified, so the user
                # must be given the explicit diff/merge (non-worktree)
                # or worktree merge-review workflow to inspect, edit,
                # or discard the partial work.  Mirrors the existing
                # USER_PREFS rule that the worktree branch must be
                # preserved on stop even when auto-commit is ON.
                # ``task_end_event.type`` covers the cases where the
                # exception propagated out of ``tab.agent.run``.
                # :meth:`WorktreeSorcarAgent.run` additionally catches
                # any ``Exception`` raised by the inner run and surfaces
                # it as a ``success: false`` YAML return value — without
                # this YAML re-parse the post-task workflow would treat
                # such caught-and-suppressed failures as a success and
                # auto-commit / auto-merge partial work.
                from kiss.core.printer import parse_result_yaml as _parse_yaml
                _agent_parsed = (
                    _parse_yaml(agent_returned) if agent_returned else None
                )
                _agent_reported_failure = bool(
                    _agent_parsed and _agent_parsed.get("success") is False
                )
                task_failed = bool(
                    (
                        task_end_event
                        and task_end_event.get("type")
                        in ("task_error", "task_stopped")
                    )
                    or _agent_reported_failure
                )
                effective_auto_commit = (
                    tab.auto_commit_mode and not task_failed
                )
                # Per-task printer cleanup (``stop_recording`` etc.)
                # is owned by :meth:`ChatSorcarAgent.run`'s finally.
                # The remaining per-task state (recording buffer,
                # ``_persist_agents``, usage offsets) is dropped at
                # the very end of this block — AFTER all post-task
                # broadcasts have happened — by
                # :meth:`BaseBrowserPrinter.cleanup_task`.
                if not use_worktree:
                    try:
                        if tab.skip_merge:
                            with self._state_lock:
                                tab.deferred_snapshot = (
                                    pre_head_sha,
                                    pre_hunks,
                                    pre_untracked,
                                    pre_file_hashes,
                                )
                        elif effective_auto_commit:
                            # "Auto commit" toggle is ON — skip the
                            # interactive merge/diff workflow entirely
                            # and commit the agent's pending changes
                            # directly.  Mirrors the user clicking
                            # "Auto commit" on the autocommit prompt
                            # without ever opening the merge view.
                            with self._state_lock:
                                tab.deferred_snapshot = None
                            self._handle_autocommit_action("commit", tab_id)
                        else:
                            with self._state_lock:
                                tab.deferred_snapshot = None
                            merge_started = self._prepare_and_start_merge(
                                work_dir, pre_hunks, pre_untracked, pre_file_hashes,
                                base_ref=pre_head_sha or "HEAD",
                                tab_id=tab_id,
                            )
                            if not merge_started:
                                self._broadcast_autocommit_prompt(tab_id)
                    except BaseException:  # pragma: no cover — merge view error handler
                        logger.debug("Merge view error", exc_info=True)
                    finally:
                        with self._state_lock:
                            tab.is_running_non_wt = False
                if task_end_event:  # pragma: no branch — always set
                    _append_chat_event(
                        task_end_event,
                        task_id=tab.task_history_id,
                        task=prompt,
                    )
                _save_task_result(
                    result=result_summary,
                    task_id=tab.task_history_id,
                    task=prompt,
                )
                from kiss._version import __version__

                _save_task_extra(
                    {
                        "model": model,
                        "work_dir": work_dir,
                        "version": __version__,
                        "tokens": tab.agent.total_tokens_used,
                        "cost": round(tab.agent.budget_used, 6),
                        "steps": int(getattr(tab.agent, "total_steps", 0) or 0),
                        "is_parallel": tab.use_parallel,
                        "is_worktree": use_worktree,
                        "auto_commit_mode": tab.auto_commit_mode,
                    },
                    task_id=tab.task_history_id,
                )
                self.printer.broadcast({"type": "tasks_updated"})
                self.printer.reset()
                if use_worktree and tab.agent._wt_pending and not tab.skip_merge:
                    try:
                        if effective_auto_commit:
                            # "Auto commit" toggle is ON in worktree
                            # mode — skip the worktree merge review
                            # entirely.  When the agent actually
                            # modified files in the worktree, auto-
                            # commit + auto-merge the worktree branch
                            # into the original branch.  When the
                            # worktree is empty (no file modifications)
                            # there is nothing to merge — auto-discard
                            # the empty branch so the repo isn't
                            # polluted with a no-op merge commit and
                            # the user isn't left with a leftover
                            # ``kiss/wt-*`` branch to clean up by hand.
                            # ``_handle_worktree_action`` runs the
                            # same generate-message → commit →
                            # squash-merge → cleanup sequence as the
                            # interactive "Merge" button (or the
                            # worktree-remove → branch-delete →
                            # checkout-original sequence for discard).
                            if self._get_worktree_changed_files(tab_id):
                                action = "merge"
                            else:
                                action = "discard"
                            # ``internal=True`` bypasses the
                            # ``is_task_active`` guard — the auto-
                            # merge runs on the same task thread
                            # that owns the flag (RACE-3 fix kept
                            # ``is_task_active = True`` through the
                            # post-task cleanup; clearing it before
                            # the auto-merge would reopen the
                            # window for a concurrent user click).
                            result = self._handle_worktree_action(
                                action, tab_id, internal=True,
                            )
                            self.printer.broadcast({
                                "type": "worktree_result",
                                "tabId": tab_id,
                                **result,
                            })
                        else:
                            # ``discard_if_empty=False``: the user
                            # opted into the worktree workflow with
                            # auto-commit OFF and has not yet chosen
                            # to merge or discard.  Preserve the
                            # branch in git even when no file changes
                            # were made so it is visible in
                            # ``git branch`` for manual action
                            # (fixes the "worktree branch is not
                            # getting created" bug).
                            self._present_pending_worktree(
                                tab_id,
                                try_merge_review=True,
                                discard_if_empty=False,
                            )
                    except BaseException:
                        logger.debug("Worktree merge review error", exc_info=True)
                # RACE-3 fix: clear ``is_task_active`` only AFTER the
                # post-task worktree presentation (auto-discard /
                # merge-review start / ``worktree_done`` broadcast)
                # has completed.  Before this point the flag is the
                # only signal preventing
                # ``_handle_worktree_action("discard"|"merge")`` from
                # racing the task thread to clear ``agent._wt`` —
                # see ``_check_worktree_busy``.
                with self._state_lock:
                    tab.is_task_active = False
                if task_end_event:  # pragma: no branch — always set
                    # Stamp the event with the owning tab so it reaches
                    # only this tab's subscribers (not every connected
                    # client).  ``ChatSorcarAgent.run()``'s finally
                    # clears the thread-local ``task_id`` before we get
                    # here, so without an explicit ``tabId`` the
                    # printer would treat this as a global system event
                    # and the frontend would apply it to whatever tab
                    # happens to be active — e.g. a sub-agent tab
                    # opened by ``run_parallel``'s ``new_tab`` broadcast
                    # — instead of the parent tab that actually owns
                    # the task.
                    self.printer.broadcast({**task_end_event, "tabId": tab_id})
                if tab.task_history_id is not None:
                    self._generate_followup_async(
                        prompt,
                        result_summary,
                        tab.task_history_id,
                    )
                # Free per-task printer state (recording buffer,
                # persist-agent entry, usage offsets, bash buffer).
                # Subscribers are preserved by ``cleanup_task`` so
                # the async followup can still fan out to the
                # originating tab; they are dropped when the tab
                # itself closes via ``cleanup_tab``.
                if tab.task_history_id is not None:
                    self.printer.cleanup_task(tab.task_history_id)
                # Clear the thread-local task id so a subsequent
                # task on the same worker thread (rare; tests) does
                # not see a stale id.
                tl = getattr(self.printer, "_thread_local", None)
                if tl is not None and getattr(tl, "task_id", None) == str(
                    tab.task_history_id,
                ):
                    tl.task_id = None
                tab.task_history_id = None
            except BaseException:  # pragma: no cover — cleanup interrupted
                with self._state_lock:
                    tab.is_task_active = False
                    if not use_worktree:
                        tab.is_running_non_wt = False
                logger.debug("Cleanup interrupted", exc_info=True)
                if task_end_event:
                    self.printer.broadcast({**task_end_event, "tabId": tab_id})

    def _stop_task(self, tab_id: str = "") -> None:
        """Signal the agent to stop.

        Sets the cooperative stop event and, if the task thread doesn't
        exit promptly, forces a ``KeyboardInterrupt`` in the task thread
        using ``ctypes.pythonapi.PyThreadState_SetAsyncExc``.  This
        handles the case where the agent is blocked in an LLM API call
        or other I/O and never reaches a cooperative ``_check_stop()``
        call.

        When *tab_id* is a subscriber (multi-viewer) tab that has no
        ``stop_event`` of its own, the method resolves through the
        printer's subscriber mapping to locate the source tab that owns
        the running task and stops that instead.  This lets a second
        browser client viewing a running task via history-click stop it.

        Args:
            tab_id: The tab to stop.  When falsy (empty string), the
                call is a no-op — a missing ``tabId`` at this layer
                indicates a frontend bug that should not silently
                stop every tab's task.
        """
        if not tab_id:
            logger.debug("_stop_task called without tab_id; ignoring")
            return
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            stop_event = tab.stop_event if tab is not None else None
            task_thread = tab.task_thread if tab is not None else None

        # When the tab has no stop_event (e.g. a subscriber/viewer tab
        # created by _replay_session → subscribe_tab), look up which
        # source tab this viewer is subscribed to and stop that instead.
        if stop_event is None:
            source_tab_id = self._find_source_tab_for_viewer(tab_id)
            if source_tab_id:
                with self._state_lock:
                    source = _RunningAgentState.running_agent_states.get(
                        source_tab_id,
                    )
                    if source is not None:
                        stop_event = source.stop_event
                        task_thread = source.task_thread

        if stop_event:
            stop_event.set()
        if task_thread is not None and task_thread.is_alive():
            threading.Thread(
                target=self._force_stop_thread,
                args=(task_thread,),
                daemon=True,
            ).start()

    def _find_source_tab_for_viewer(self, viewer_tab_id: str) -> str | None:
        """Find a peer tab id sharing the same task as *viewer_tab_id*.

        Scans the printer's ``_subscribers`` mapping
        (``task_id -> {tab_ids}``) to locate the task that
        *viewer_tab_id* is subscribed to, then returns another tab id
        subscribed to the same task whose :class:`_RunningAgentState`
        carries a live ``stop_event`` (i.e. the tab that actually
        started the task).  Returns ``None`` when no such tab exists.

        Args:
            viewer_tab_id: The subscriber/viewer tab id to look up.

        Returns:
            A peer tab id that owns the cooperative stop event for the
            running task, or ``None`` if not found.
        """
        with self.printer._lock:
            task_key: str | None = None
            for task_id, viewers in self.printer._subscribers.items():
                if viewer_tab_id in viewers:
                    task_key = task_id
                    break
            if task_key is None:
                return None
            peers = list(self.printer._subscribers[task_key])
        with self._state_lock:
            for peer in peers:
                if peer == viewer_tab_id:
                    continue
                state = _RunningAgentState.running_agent_states.get(peer)
                if state is not None and state.stop_event is not None:
                    return peer
        return None

    @staticmethod
    def _force_stop_thread(task_thread: threading.Thread) -> None:
        """Watchdog that forces ``KeyboardInterrupt`` in *task_thread*.

        Waits 1 second for the cooperative stop-event mechanism to work.
        If the thread is still alive, raises ``KeyboardInterrupt``
        asynchronously in it.  Retries once after 5 seconds in case the
        first exception was swallowed or the thread was in C code.
        """
        task_thread.join(timeout=1)
        for _ in range(2):  # pragma: no branch — thread always dies within 2 attempts
            if not task_thread.is_alive():
                return
            tid = task_thread.ident
            if tid is not None:  # pragma: no branch — running thread always has ident
                rc = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(tid),
                    ctypes.py_object(KeyboardInterrupt),
                )
                if rc == 0:
                    return
                if rc > 1:  # pragma: no cover — rare: exception set in multiple states
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(tid), None
                    )
            task_thread.join(timeout=5)

    def _await_user_response(self) -> str:
        """Block until the user sends a response, checking stop_event periodically.

        Returns:
            The user's answer string.

        Raises:
            KeyboardInterrupt: If the stop event is set before an answer arrives.
        """
        stop = getattr(self.printer._thread_local, "stop_event", None)
        if stop is None:
            raise KeyboardInterrupt("No stop event set")
        # Resolve the answer queue via the printer's task-id ->
        # subscriber-tabs mapping: pick the first subscribed tab that
        # has a live ``user_answer_queue``.  Any tab subscribed to
        # this task can answer.
        task_key = getattr(self.printer._thread_local, "task_id", None)
        q = None
        if task_key:
            with self.printer._lock:
                tab_ids = list(self.printer._subscribers.get(task_key, ()))
            with self._state_lock:
                for tab_id in tab_ids:
                    tab = _RunningAgentState.running_agent_states.get(tab_id)
                    if tab is not None and tab.user_answer_queue is not None:
                        q = tab.user_answer_queue
                        break
        # M4 — when the tab has no answer queue (e.g. the tab was closed
        # mid-question) there is no path that can ever return a response.
        # Refuse to busy-loop forever; raise immediately so the agent
        # thread can unwind and the user-facing task can finish.
        if q is None:
            raise KeyboardInterrupt(
                "User answer queue is missing (tab closed?); aborting wait",
            )
        while True:
            try:
                return q.get(timeout=0.5)
            except queue.Empty:
                pass
            if stop.is_set():
                raise KeyboardInterrupt("Stopped while waiting for user")

    def _ask_user_question(self, question: str) -> str:
        """Callback for agent questions."""
        self.printer.broadcast({
            "type": "askUser",
            "question": question,
        })
        return self._await_user_response()
