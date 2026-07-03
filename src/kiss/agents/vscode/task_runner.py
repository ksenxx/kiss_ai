# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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
import re
import threading
import time
from contextlib import nullcontext, suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from kiss.agents.sorcar.running_agent_state import _RunningAgentState

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    repo_lock,
    strip_worktree_suffix,
)
from kiss.agents.sorcar.persistence import (
    _append_chat_event,
    _save_task_extra,
    _save_task_result,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _parse_diff_hunks,
    _save_untracked_base,
    _snapshot_files,
)

# Imported at runtime (not just TYPE_CHECKING) so task-id keys can be
# normalised via the ``JsonPrinter._coerce_task_id`` STATIC method
# called on the class.  Calling it on ``self.printer`` instead would
# couple this mixin to a private method of the concrete printer
# instance — the mixin's printer contract is duck-typed (thread-local
# ``task_id`` / ``stop_event``, ``_lock``, ``_subscribers``), and a
# minimal printer without ``_coerce_task_id`` used to crash
# ``_resolve_task_answer_queue`` with ``AttributeError`` (see
# ``TestM4AwaitUserResponseEmptyQueue``).
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import get_available_models
from kiss.core.printer import parse_result_yaml

logger = logging.getLogger(__name__)

# Declare the C signature once so ``PyThreadState_SetAsyncExc`` calls in
# :meth:`_TaskRunnerMixin._interrupt_thread` marshal arguments correctly.
ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
    ctypes.c_ulong,
    ctypes.py_object,
]


def build_task_extra_payload(
    *,
    model: str,
    work_dir: str,
    version: str,
    tokens: int,
    cost: float,
    steps: int,
    is_parallel: bool,
    is_worktree: bool,
    auto_commit_mode: bool,
    start_ms: int,
    end_ms: int,
) -> dict[str, object]:
    """Build the persisted ``task_history.extra`` payload for a completed task.

    Strips the ``.kiss-worktrees/kiss_wt-<slug>`` suffix from *work_dir*
    so the persisted path is the user-visible workspace folder rather
    than an ephemeral worktree directory that would vanish on merge or
    discard.

    Args:
        model: Model name used for the task.
        work_dir: Working directory the task ran from.  Worktree paths
            are stripped to their parent repo.
        version: KISS version string.
        tokens: Total tokens consumed by the agent.
        cost: Total budget consumed by the agent (USD).
        steps: Total agent steps taken.
        is_parallel: Whether parallel sub-agents were enabled.
        is_worktree: Whether the task ran inside a worktree.
        auto_commit_mode: Auto-commit toggle state at completion.
        start_ms: Agent start timestamp in milliseconds since epoch.
        end_ms: Agent end timestamp in milliseconds since epoch.

    Returns:
        Dict ready to pass to ``_save_task_extra``.
    """
    return {
        "model": model,
        "work_dir": strip_worktree_suffix(work_dir),
        "version": version,
        "tokens": tokens,
        "cost": cost,
        "steps": steps,
        "is_parallel": is_parallel,
        "is_worktree": is_worktree,
        "auto_commit_mode": auto_commit_mode,
        "startTs": start_ms,
        "endTs": end_ms,
    }


def parse_task_tags(text: str) -> list[str]:
    """Parse ``<task>...</task>`` tags from *text* and return individual tasks.

    When the input contains one or more ``<task>`` blocks with non-empty
    content, each block's content is returned as a separate list element.
    If no valid ``<task>`` blocks are found (or all are empty/whitespace),
    the original *text* is returned as a single-element list so that
    callers can always iterate without special-casing.

    Args:
        text: Input text potentially containing ``<task>...</task>`` tags.

    Returns:
        List of task strings.  Always contains at least one element.
    """
    tasks = [m.strip() for m in re.findall(r"<task>(.*?)</task>", text, re.DOTALL)]
    tasks = [t for t in tasks if t]
    return tasks if tasks else [text]

# Sentinel pushed onto the user-answer queue by the stop-event watcher
# in :meth:`_TaskRunnerMixin._await_user_response` so the blocking
# ``q.get`` wakes immediately when the task is cancelled mid-question.
# Identity comparison disambiguates it from any string answer.
_STOP_SENTINEL: object = object()


class _TaskRunnerMixin:
    """Task-lifecycle methods (run, stop, user-question callback)."""

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock
        _tab_chat_views: dict[str, str]
        _pending_user_answer_tasks: dict[int, str]

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
        def _main_dirty_files(self, work_dir: str = "") -> list[str]: ...
        def _broadcast_autocommit_prompt(
            self, tab_id: str, work_dir: str = "",
        ) -> None: ...
        def _handle_autocommit_action(
            self, action: str, tab_id: str = "", *, work_dir: str = "",
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
            self, task: str, result: str, task_id: str | None,
        ) -> None: ...
        def _refresh_files_after_task(self, work_dir: str = "") -> None: ...

    def _run_task(self, cmd: dict[str, Any]) -> None:
        """Run the agent with the given task.

        An outer try/finally guarantees that ``status: running: False``
        is **always** broadcast when this method exits, regardless of
        which code-path is taken.
        """
        tab_id = cmd.get("tabId", "")
        # Capture the agent's true start timestamp (ms since epoch)
        # ONCE here and propagate it through the status broadcast,
        # cmd dict (so ``_run_task_inner`` can echo it on task_end
        # broadcasts), and the persisted ``extra`` JSON.  The
        # frontend uses this as the anchor for the "Running …" /
        # "Done (Xm Ys)" label at the top of the chat webview —
        # anchoring on the client's ``Date.now()`` would mis-report
        # elapsed time for any tab that joined the chat late (e.g.
        # opened from history while the agent was already running).
        start_ms = int(time.time() * 1000)
        cmd["_start_ms"] = start_ms
        # The CLI client stamps every ``run`` with a per-submission
        # ``taskId`` (a UUID it minted just before sending) so its
        # dispatcher can filter stale ``status:running=false`` events
        # from a prior task that finished after the new one was sent
        # (review #3 / #4 round 2 — A2 critical).  Echo the id verbatim
        # on every ``status`` broadcast for this run so the client's
        # ``current_task_id`` filter actually works in production.
        # r3-vscode-H1: reject non-string ``taskId`` payloads (list,
        # dict, bool, int) so they do not silently flow into the
        # ``status`` envelope echo where they would later be compared
        # by ``str ==`` against UUID strings on the client.
        _raw_ctid = cmd.get("taskId", "")
        client_task_id = _raw_ctid if isinstance(_raw_ctid, str) and _raw_ctid else ""
        try:
            status_start: dict[str, Any] = {
                "type": "status",
                "running": True,
                "tabId": tab_id,
                "startTs": start_ms,
            }
            if client_task_id:
                status_start["taskId"] = client_task_id
            self.printer.broadcast(status_start)
            self._run_task_inner(cmd)
        except Exception as exc:
            # ``_run_task_inner`` handles failures of the agent run
            # itself, but an exception raised BEFORE its big ``try``
            # block (malformed command fields, a git failure re-raised
            # by the pre-snapshot guard) used to unwind straight
            # through this worker thread: the spinner stopped (the
            # ``finally`` below broadcasts ``running=False``) but no
            # ``result``/``task_error`` event was ever emitted — the
            # task silently vanished from the user's point of view.
            logger.warning(
                "Task setup failed: tab_id=%s error=%s",
                tab_id,
                exc,
                exc_info=True,
            )
            self.printer.broadcast({
                "type": "result",
                "text": f"Task failed: {type(exc).__name__}: {exc}",
                "success": False,
                "total_tokens": 0,
                "cost": "$0.0000",
                "step_count": 0,
                "tabId": tab_id,
            })
        finally:
            with self._state_lock:
                tab = _RunningAgentState.running_agent_states.get(tab_id)
                # Ownership guard: only reset the per-tab task slots
                # when THIS worker still owns them.  The worker can
                # spend a long time between the agent returning and
                # this ``finally`` (autocommit git scans, merge-view
                # preparation, persistence), during which the tab's
                # state can be disposed (``closeTab`` while the task
                # is wedged in cleanup) and RE-CREATED under the same
                # tab id by a reopened frontend tab that immediately
                # starts a NEW task.  Without the guard this stale
                # cleanup would null the FRESH state's ``agent`` (the
                # new worker crashes with ``'NoneType' object has no
                # attribute 'run'``), drop its answer queue and stop
                # event (unanswerable, unstoppable) and discard its
                # queued follow-ups.  ``task_thread`` is the ownership
                # token: ``_cmd_run`` stamps it with the worker thread
                # it starts, so a different (alive) thread here means
                # a newer task owns the tab — its own ``finally`` will
                # clean up.  ``None`` is treated as owned for direct
                # ``_run_task`` invocations that were never armed by
                # ``_cmd_run`` (tests, embedding callers).
                owns_tab = tab is None or (
                    tab.task_thread is None
                    or tab.task_thread is threading.current_thread()
                )
                if tab is not None and owns_tab:
                    tab.task_thread = None
                    tab.stop_event = None
                    tab.user_answer_queue = None
                    # Drop any queued follow-up prompts that the user
                    # typed during this task — the next task gets a
                    # fresh, empty queue.  Keeping them would leak
                    # stale context into the next unrelated task.
                    tab.pending_user_messages.clear()
                    tab.is_task_active = False
                    tab.is_running_non_wt = False
                    # Clear the shutdown-cancellation marker so a reused
                    # tab's next task starts from a clean slate (it is
                    # only meaningful while a task is in flight).
                    tab.interrupted_by_shutdown = False
                    # Dispose the transient agent — a fresh one is
                    # lazily allocated by ``_get_tab`` when the next
                    # task's ``_run_task_inner`` starts.
                    # Preserve the agent when a worktree branch with
                    # changes is still pending user action (merge or
                    # discard).  Disposing it would destroy the
                    # in-memory worktree state that
                    # ``_handle_worktree_action`` needs, causing a
                    # misleading "Not a git repository" error.
                    if tab.agent is not None:
                        tab.last_task_id = (
                            tab.agent._last_task_id or tab.last_task_id
                        )
                        if tab.use_worktree and tab.agent._wt_pending:
                            pass  # keep agent alive for merge/discard
                        else:
                            tab.agent = None
                            tab.use_worktree = False
                # Snapshot the finished task's id so we can broadcast
                # ``status running=False`` to every viewer tab that
                # was subscribed to it (in addition to the launcher
                # tab).  ``status`` events carry an explicit
                # ``tabId`` and are routed verbatim by the printer
                # — i.e. they do NOT pass through the per-task
                # subscriber fan-out — so without an explicit per-
                # subscriber broadcast the viewer tabs would never
                # observe the running→idle transition, leaving their
                # tab-title spinner pulsing forever and their input
                # box stuck in "queue follow-up" mode after the task
                # had ended.  Mirrors the start-time per-viewer
                # broadcast in :meth:`_subscribe_chat_viewers`.
                task_id_for_end = (
                    tab.last_task_id
                    if tab is not None and owns_tab
                    else None
                )
                # When a NEWER task owns the tab (see the ownership
                # guard above), suppress the launcher-tab
                # ``running=False`` broadcast too: the tab's spinner
                # now belongs to the new task, and stopping it here
                # would flip the frontend out of running mode while
                # the new agent is still working.
                if owns_tab:
                    status_end: dict[str, Any] = {
                        "type": "status", "running": False, "tabId": tab_id,
                    }
                    if client_task_id:
                        status_end["taskId"] = client_task_id
                    self.printer.broadcast(status_end)
            self._broadcast_status_end_to_viewers(
                task_id_for_end, tab_id, client_task_id=client_task_id,
            )
            # If the user clicked closeTab while this task was still
            # running, dispose the now-idle _RunningAgentState.  No-op otherwise.
            self._dispose_if_closed(tab_id)

    def _broadcast_status_end_to_viewers(
        self, task_id: str | None, launcher_tab_id: str,
        *, client_task_id: str = "",
    ) -> None:
        """Broadcast ``status running=False`` to every viewer subscribed
        to *task_id*, excluding the launcher tab.

        ``status`` events carry an explicit ``tabId`` and are routed
        verbatim by the printer's transport (no per-task subscriber
        fan-out).  Without an explicit per-viewer broadcast, a tab
        that joined the running task via ``_replay_session`` /
        ``_reattach_running_chat`` (history-resume click) or via
        ``_subscribe_chat_viewers`` (idle viewer of the chat) would
        never receive a ``running=False`` event stamped with its own
        tab id — its frontend would keep ``isRunning=true`` forever,
        the pulsing tab-title indicator would not stop, and follow-up
        user input would keep being routed as ``appendUserMessage``
        against a now-finished task (and dropped).

        Args:
            task_id: The finished task's ``task_history`` row id.
                ``None`` when the worker thread unwound before a task
                id was allocated (very early failure path); in that
                case there is nothing to fan out.
            launcher_tab_id: The tab the task was launched in — its
                ``running=False`` broadcast is emitted directly by
                the caller, so skip it here to avoid duplication.
        """
        if task_id is None:
            return
        task_key = JsonPrinter._coerce_task_id(task_id)
        for viewer_tab_id in self.printer._fanout_targets(task_id):
            if viewer_tab_id == launcher_tab_id:
                continue
            # Skip viewers that are actively running their OWN task:
            # a tab subscribed to this (now finished) task may have
            # since started an unrelated task of its own — ``status``
            # events are stamped with the viewer's tab id, so sending
            # ``running=False`` would stop that tab's spinner and
            # flip its input box out of "queue follow-up" mode while
            # its own agent is still working.  Its own task's end
            # path broadcasts the ``running=False`` it actually needs.
            with self._state_lock:
                viewer_state = _RunningAgentState.running_agent_states.get(
                    viewer_tab_id,
                )
                if viewer_state is not None and viewer_state.is_task_active:
                    viewer_task = (
                        JsonPrinter._coerce_task_id(
                            getattr(viewer_state.agent, "_last_task_id", None),
                        )
                        if viewer_state.agent is not None
                        else ""
                    )
                    if viewer_task != task_key:
                        continue
            payload: dict[str, Any] = {
                "type": "status",
                "running": False,
                "tabId": viewer_tab_id,
            }
            if client_task_id:
                payload["taskId"] = client_task_id
            self.printer.broadcast(payload)

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
        with repo_lock(repo) if repo else nullcontext():
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

    def _run_task_inner(self, cmd: dict[str, Any]) -> None:
        """Inner implementation of _run_task (without the status guarantee)."""
        prompt = cmd.get("prompt", "")
        work_dir = cmd.get("workDir") or self.work_dir
        active_file = cmd.get("activeFile")
        raw_attachments = cmd.get("attachments", [])
        if not isinstance(raw_attachments, list):
            # Iteration 3 made malformed attachment ENTRIES skippable;
            # a non-iterable attachments FIELD (e.g. an int) still
            # raised TypeError at the ``for`` below and killed the
            # task thread before even the model check.
            logger.warning(
                "Ignoring malformed attachments field of type %s",
                type(raw_attachments).__name__,
            )
            raw_attachments = []
        # Agent start timestamp (ms since epoch), stamped on the cmd
        # dict by ``_run_task``.  0 when absent (direct test calls).
        start_ms = int(cmd.get("_start_ms") or 0)

        attachments: list[Attachment] | None = None
        if raw_attachments:
            attachments = []
            for att in raw_attachments:
                try:
                    data_b64 = att.get("data", "")
                    mime = att.get("mimeType", "application/octet-stream")
                    data = base64.b64decode(data_b64)
                except Exception:
                    # A malformed attachment (non-dict entry, invalid
                    # base64) must not kill the task thread:
                    # ``_run_task`` has no except clause, so an
                    # escaping ``binascii.Error`` would silently end
                    # the task — spinner stops with no result/error
                    # event broadcast or persisted.  Skip the bad
                    # attachment and run the task with the rest.
                    logger.warning(
                        "Skipping malformed attachment", exc_info=True,
                    )
                    continue
                attachments.append(Attachment(data=data, mime_type=mime))

        tab_id = cmd.get("tabId", "")
        tab = self._get_tab(tab_id)
        model = cmd.get("model") or tab.selected_model

        # ``_get_tab`` above guarantees the agent slot is populated:
        # it lazily allocates a fresh ``WorktreeSorcarAgent`` whenever
        # ``tab.agent`` is ``None`` (the previous agent was disposed
        # at the end of the prior ``_run_task``, so there is no
        # long-lived per-tab agent across distinct task executions).
        # Tests that need to inject a stub agent (e.g. patch
        # ``tab.agent.run``) can pre-populate ``tab.agent`` before
        # calling ``_run_task`` — we honour any agent the caller has
        # already attached.
        assert tab.agent is not None
        # Stash the frontend tab id on the agent so its
        # ``pre_step_hook`` (see ``SorcarAgent.run``) can resolve back
        # to the owning ``_RunningAgentState`` and drain
        # ``pending_user_messages`` before each model call.
        tab.agent._tab_id = tab_id
        # Stamp the agent's true start timestamp (ms since epoch) on
        # the live agent so ``VSCodeServer._replay_session`` can anchor
        # the "Running …" header of a tab that opens this chat WHILE
        # the task is still running.  The persisted ``extra`` JSON only
        # gains ``startTs`` at task END (``_save_task_extra`` in the
        # cleanup finally below), so during the run the live agent is
        # the only source of truth for the start time.
        tab.agent._task_start_ms = start_ms
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
            # Stamp the owning tab id explicitly: this branch runs on
            # the task thread BEFORE ``ChatSorcarAgent.run`` sets the
            # printer's thread-local ``task_id``, so without ``tabId``
            # the event has no task context and ``WebPrinter.broadcast``
            # would fall into its GLOBAL branch — sending the result
            # verbatim (no ``tabId``) to every connected client, where
            # the frontend renders it into whichever tab is active.
            # That pollutes a *different* tab running an unrelated task.
            # An explicit ``tabId`` scopes the result to this tab only.
            self.printer.broadcast({
                "type": "result",
                "text": no_model_msg,
                "success": False,
                "total_tokens": 0,
                "cost": "$0.0000",
                "step_count": 0,
                "tabId": tab_id,
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
            repo = GitWorktreeOps.discover_repo(Path(work_dir))
            with repo_lock(repo) if repo else nullcontext(), self._state_lock:
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
            try:
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

        logger.info(
            "Task started: tab_id=%s model=%s use_worktree=%s "
            "auto_commit=%s prompt=%r",
            tab_id,
            model,
            use_worktree,
            tab.auto_commit_mode,
            prompt[:200],
        )
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
            from kiss.agents.vscode.vscode_config import (
                build_model_config,
                load_config,
            )

            _vcfg = load_config()
            _cfg_budget = float(_vcfg.get("max_budget", 100))
            _cfg_web = _vcfg.get("use_web_browser", True)
            _model_config = build_model_config(_vcfg)

            # Invoked by ``ChatSorcarAgent.run`` the moment the run's
            # ``task_history`` row id is allocated (before any agent
            # event is broadcast): subscribes every OTHER tab that has
            # this chat open — in any VS Code window or browser
            # window — to the new task's event stream so all viewers
            # of the chat see the live events, not only this tab.
            on_task_id_allocated = partial(
                self._subscribe_chat_viewers,
                source_tab_id=tab_id,
                start_ms=start_ms,
                client_task_id=(
                    cmd["taskId"]
                    if isinstance(cmd.get("taskId"), str) and cmd["taskId"]
                    else ""
                ),
            )

            for task_prompt in subtasks:
                # Record the raw user prompt so the post-task
                # auto-commit hooks can incorporate the user's intent
                # in the generated commit message (see
                # :meth:`_MergeFlowMixin._handle_autocommit_action`).
                tab.last_user_prompt = task_prompt
                subtask_failed = False
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
                        _on_task_id_allocated=on_task_id_allocated,
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
                    _run_parsed = (
                        parse_result_yaml(agent_returned)
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
                    logger.info(
                        "Agent returned: tab_id=%s task_id=%s "
                        "summary=%r",
                        tab_id,
                        tab.task_history_id,
                        result_summary[:200],
                    )
                except KeyboardInterrupt:
                    result_summary, task_end_event = self._cancel_outcome(tab)
                    subtask_failed = True
                    logger.info(
                        "%s: tab_id=%s task_id=%s",
                        result_summary,
                        tab_id,
                        tab.task_history_id,
                    )
                except Exception as e:
                    result_summary = f"Task failed: {e}"
                    task_end_event = {"type": "task_error", "text": str(e)}
                    subtask_failed = True
                    logger.warning(
                        "Task failed: tab_id=%s task_id=%s error=%s",
                        tab_id,
                        tab.task_history_id,
                        e,
                        exc_info=True,
                    )
                finally:
                    tab.task_history_id = tab.agent._last_task_id
                if subtask_failed:
                    self.printer.broadcast({
                        "type": "result",
                        "text": result_summary,
                        "success": False,
                        "total_tokens": tab.agent.total_tokens_used,
                        "cost": f"${tab.agent.budget_used:.4f}",
                        # RelentlessAgent-derived agents accumulate
                        # completed steps into ``total_steps`` and leave
                        # ``step_count`` at 0 — mirror the persisted
                        # ``extra`` metrics below.  Fall back to
                        # ``step_count`` for plain agents whose
                        # ``total_steps`` is 0.
                        "step_count": (
                            int(getattr(tab.agent, "total_steps", 0) or 0)
                            or int(getattr(tab.agent, "step_count", 0) or 0)
                        ),
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
                    result_summary, _cancel_event = self._cancel_outcome(tab)
                    task_end_event = task_end_event or _cancel_event
                else:
                    _exc_name = type(_outer_exc).__name__
                    result_summary = f"Task failed: {_exc_name}: {_outer_exc}"
                    task_end_event = task_end_event or {
                        "type": "task_error",
                        "text": f"{_exc_name}: {_outer_exc}",
                    }
            else:
                task_end_event = task_end_event or {"type": "task_stopped"}
            # Mirror the per-subtask failure broadcast in the for-loop:
            # the cleanup ``finally`` only broadcasts ``task_end_event``
            # (``task_stopped`` / ``task_error`` / ``task_interrupted``)
            # — NOT a ``type: result`` event.  Without this broadcast
            # the VS Code webview never receives a result event for
            # tasks that died in this outer-BaseException path (e.g.
            # ``SystemExit`` / ``asyncio.CancelledError`` from the
            # streaming context manager, or a ``KeyboardInterrupt``
            # arriving BETWEEN the outer ``try:`` and the for-loop
            # body), and ``main.js`` falls back to rendering
            # ``"(no result)"`` even when ``result_summary`` carries a
            # meaningful recovered summary.  ``tab.agent`` may be
            # ``None`` when the outer exception fired before agent
            # assignment, so all metric reads are defensive.
            _agent_for_metrics = getattr(tab, "agent", None)
            self.printer.broadcast({
                "type": "result",
                "text": result_summary,
                "success": False,
                "total_tokens": int(
                    getattr(_agent_for_metrics, "total_tokens_used", 0) or 0
                ),
                "cost": (
                    "$"
                    f"{float(getattr(_agent_for_metrics, 'budget_used', 0.0) or 0.0):.4f}"
                ),
                "step_count": (
                    int(getattr(_agent_for_metrics, "total_steps", 0) or 0)
                    or int(getattr(_agent_for_metrics, "step_count", 0) or 0)
                ),
                "tabId": tab_id,
            })
        finally:
            try:
                # RACE-3 fix: keep ``is_task_active = True`` through
                # the post-task cleanup so a concurrent
                # ``_handle_worktree_action`` (driven by a misclick
                # on the merge/discard button) cannot pass its
                # ``_check_worktree_busy`` guard and mutate
                # ``tab.agent._wt`` while this thread is still
                # inside ``_present_pending_worktree``.  The flag is
                # cleared (``tab.is_task_active = False``) at the very
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
                # invariant that the worktree branch must be
                # preserved on stop even when auto-commit is ON.
                # ``task_end_event.type`` covers the cases where the
                # exception propagated out of ``tab.agent.run``.
                # :meth:`WorktreeSorcarAgent.run` additionally catches
                # any ``Exception`` raised by the inner run and surfaces
                # it as a ``success: false`` YAML return value — without
                # this YAML re-parse the post-task workflow would treat
                # such caught-and-suppressed failures as a success and
                # auto-commit / auto-merge partial work.
                _agent_parsed = (
                    parse_result_yaml(agent_returned) if agent_returned else None
                )
                _agent_reported_failure = bool(
                    _agent_parsed and _agent_parsed.get("success") is False
                )
                task_failed = bool(
                    (
                        task_end_event
                        and task_end_event.get("type")
                        in ("task_error", "task_stopped", "task_interrupted")
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
                # :meth:`JsonPrinter.cleanup_task`.
                if not use_worktree:
                    try:
                        if effective_auto_commit:
                            # "Auto commit" toggle is ON — skip the
                            # interactive merge/diff workflow entirely
                            # and commit the agent's pending changes
                            # directly.  Mirrors the user clicking
                            # "Auto commit" on the autocommit prompt
                            # without ever opening the merge view.
                            self._handle_autocommit_action(
                                "commit", tab_id, work_dir=work_dir,
                            )
                        else:
                            merge_started = self._prepare_and_start_merge(
                                work_dir, pre_hunks, pre_untracked, pre_file_hashes,
                                base_ref=pre_head_sha or "HEAD",
                                tab_id=tab_id,
                            )
                            if not merge_started:
                                self._broadcast_autocommit_prompt(
                                    tab_id, work_dir,
                                )
                    except BaseException:  # pragma: no cover — merge view error handler
                        logger.debug("Merge view error", exc_info=True)
                    finally:
                        with self._state_lock:
                            tab.is_running_non_wt = False
                # ``task_end_event`` is provably always set on entry to
                # this cleanup: ``parse_task_tags`` returns at least one
                # subtask, every per-subtask handler assigns the event,
                # and the outer ``except BaseException`` rewrite covers
                # every other unwind path.
                assert task_end_event is not None
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
                logger.info(
                    "Task result persisted: task_id=%s result=%r",
                    tab.task_history_id,
                    result_summary[:200],
                )
                from kiss._version import __version__

                # Persist agent start / end timestamps (ms since epoch)
                # so a later history load can flip the "Running …"
                # label at the top of the chat webview to "Done (Xm Ys)"
                # as soon as ``Date.now() >= endTs`` — without waiting
                # for a live ``task_done`` event that may have
                # already fired (and been missed) before the tab was
                # opened.
                end_ms = int(time.time() * 1000)
                _save_task_extra(
                    build_task_extra_payload(
                        model=model,
                        work_dir=work_dir,
                        version=__version__,
                        tokens=tab.agent.total_tokens_used,
                        cost=round(tab.agent.budget_used, 6),
                        steps=int(getattr(tab.agent, "total_steps", 0) or 0),
                        is_parallel=tab.use_parallel,
                        is_worktree=use_worktree,
                        auto_commit_mode=tab.auto_commit_mode,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    ),
                    task_id=tab.task_history_id,
                )
                self.printer.broadcast({"type": "tasks_updated"})
                self.printer.reset()
                if use_worktree and tab.agent._wt_pending:
                    # "Lost slides" bug fix: a task that ended in
                    # failure / user-Stop leaves partial, unverified
                    # work in the worktree.  Mark the agent as
                    # pending review BEFORE any post-task presentation
                    # so a subsequent tab teardown (which can fire
                    # asynchronously the moment ``is_task_active``
                    # drops below) preserves the branch instead of
                    # silently squash-merging it into the user's
                    # original branch.  When the user later
                    # explicitly clicks Merge or Discard the flag
                    # is cleared by ``_handle_worktree_action``.
                    if task_failed:
                        tab.agent._pending_review = True
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
                # the task.  ``startTs`` / ``endTs`` echo the
                # agent's true wall-clock so the frontend's
                # "Running …" / "Done (…)" label uses agent time
                # rather than the client's ``Date.now()``.
                self.printer.broadcast({
                    **task_end_event,
                    "tabId": tab_id,
                    "startTs": start_ms,
                    "endTs": end_ms,
                })
                # Refresh the ``@``-mention file cache for this
                # work_dir: the agent may have created or deleted
                # files during its turn, but the cache is only
                # otherwise updated on cold start, daemon-wide
                # ``setWorkDir``, or an explicit refresh.  Without
                # this call the next ``@`` mention serves stale
                # suggestions — new files invisible, deleted files
                # still listed.  The hook is a background rescan
                # that no-ops when the file set is unchanged (only
                # modifications) and broadcasts an updated ``files``
                # event when files were actually added or removed
                # so any open picker UI refreshes immediately.
                self._refresh_files_after_task(work_dir)
                logger.info(
                    "Task lifecycle complete: tab_id=%s task_id=%s "
                    "elapsed_ms=%d event_type=%s",
                    tab_id,
                    tab.task_history_id,
                    end_ms - start_ms,
                    (task_end_event or {}).get("type", "none"),
                )
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
                    self.printer.broadcast({
                        **task_end_event,
                        "tabId": tab_id,
                        "startTs": start_ms,
                        "endTs": int(time.time() * 1000),
                    })

    @staticmethod
    def _cancel_outcome(
        tab: _RunningAgentState,
    ) -> tuple[str, dict[str, Any]]:
        """Resolve the result label + end event for a cancelled task.

        A task is cancelled by injecting a ``KeyboardInterrupt`` into
        the worker thread.  Two unrelated paths do this and are
        otherwise indistinguishable at the ``except KeyboardInterrupt``
        site:

        * the user clicking "Stop" (:meth:`_stop_task`), and
        * a graceful server shutdown on ``SIGTERM`` — e.g. a daemon /
          LaunchAgent restart triggered by a KISS Sorcar extension
          update — which routes through
          :meth:`RemoteAccessServer._stop_active_agent_tasks`.

        The shutdown path sets :attr:`_RunningAgentState.interrupted_by_shutdown`
        on the tab *before* injecting the interrupt, so this flag is the
        single source of truth.  Returning the shutdown-specific label
        and ``task_interrupted`` event prevents the long-standing
        mislabelling where a server restart was reported to the user as
        "Task stopped by user".

        Args:
            tab: The running tab state whose task was cancelled.

        Returns:
            ``(result_summary, task_end_event)`` — the persisted result
            string and the lifecycle end-event dict.
        """
        if tab.interrupted_by_shutdown:
            return (
                "Task interrupted by server restart/shutdown",
                {"type": "task_interrupted"},
            )
        return ("Task stopped by user", {"type": "task_stopped"})

    def _subscribe_chat_viewers(
        self,
        task_id: str,
        chat_id: str,
        *,
        source_tab_id: str,
        start_ms: int,
        client_task_id: str = "",
    ) -> None:
        """Subscribe every tab that has *chat_id* open to a new task's stream.

        Invariant: when a task is running on a chat, every tab — in
        any VS Code window or remote browser window — that has that
        chat open must see the task's events streaming live.  Tabs
        that open the chat WHILE the task is already running are
        handled by ``_replay_session`` → ``_reattach_running_chat``;
        this hook covers the tabs that opened the chat BEFORE the
        task started (e.g. the tab that ran the previous task of the
        chat, or a history viewer in a sibling window).

        Called via ``_on_task_id_allocated`` from
        :meth:`ChatSorcarAgent.run` as soon as the run's
        ``task_history`` row id exists, before any agent event is
        broadcast.  For each viewer tab (excluding the launcher,
        which ``ChatSorcarAgent.run`` already subscribed via
        ``_subscribe_tab_id``) it mirrors the launcher's start
        sequence: ``clear`` (resets the viewer's replayed content and
        per-tab stream state) followed by ``status running=True``
        (flips the viewer's spinner / stop button), after which the
        printer's per-subscriber fan-out delivers every live event.

        Args:
            task_id: The freshly allocated ``task_history`` row id.
            chat_id: The chat id the task runs on.
            source_tab_id: The tab that launched the task (skipped).
            start_ms: The agent's start timestamp (ms since epoch),
                echoed on the ``status`` broadcast so viewer tabs
                anchor their "Running …" timer correctly.
        """
        if not chat_id:
            return
        with self._state_lock:
            viewers = []
            for viewer_tab_id, viewed_chat_id in self._tab_chat_views.items():
                if viewed_chat_id != chat_id or viewer_tab_id == source_tab_id:
                    continue
                # Skip viewers that are actively running their OWN
                # task: two tabs can view the same chat and each run
                # a task concurrently (``_cmd_run`` guards per TAB,
                # not per chat).  Broadcasting ``clear`` here would
                # wipe such a tab's live transcript mid-task, and the
                # ``status running=True (startTs=this task's start)``
                # would re-anchor its "Running …" timer to the WRONG
                # task's start time.  A busy tab's transcript belongs
                # to its own task until that task ends.
                viewer_state = _RunningAgentState.running_agent_states.get(
                    viewer_tab_id,
                )
                if viewer_state is not None and viewer_state.is_task_active:
                    continue
                viewers.append(viewer_tab_id)
        for viewer_tab_id in viewers:
            self.printer.subscribe_tab(task_id, viewer_tab_id)
            self.printer.broadcast({
                "type": "clear",
                "chat_id": chat_id,
                "tabId": viewer_tab_id,
            })
            viewer_status: dict[str, Any] = {
                "type": "status",
                "running": True,
                "tabId": viewer_tab_id,
                "startTs": start_ms,
            }
            if client_task_id:
                # Echo the launcher's client-supplied taskId so the
                # CLI client's per-task status filter symmetrically
                # accepts BOTH the viewer-start (here) and the
                # viewer-end (``_broadcast_status_end_to_viewers``)
                # broadcasts — review H1 round 3.
                viewer_status["taskId"] = client_task_id
            self.printer.broadcast(viewer_status)

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
        (``task_id -> {tab_ids}``) to locate the tasks that
        *viewer_tab_id* is subscribed to, then returns another tab id
        subscribed to one of those tasks whose
        :class:`_RunningAgentState` carries a live ``stop_event``
        (i.e. the tab that actually started the task).  Returns
        ``None`` when no such tab exists.

        Every subscribed task is considered — not just the first
        match: ``JsonPrinter.cleanup_task`` intentionally preserves
        subscriber sets when a task ends, so a viewer typically holds
        stale subscriptions to FINISHED tasks alongside the one
        RUNNING task.  Stopping the scan at the first (oldest) match
        would resolve the finished task, find no peer with a live
        ``stop_event`` there, and wrongly report "no source tab" for
        a viewer that can in fact stop a running task.

        A peer with a live ``stop_event`` is only accepted when the
        task its agent is actually running (``agent._last_task_id``)
        matches the subscribed task being scanned.  A stale
        finished-task co-subscriber that has since started a brand-new
        UNRELATED task (which the viewer is NOT subscribed to) also
        carries a live ``stop_event`` — returning it would let the
        viewer's Stop kill that unrelated task (cross-task stop
        hijack, symmetric to the answer-queue hijack fixed as
        BUG-TR2-2).  Peers whose running task cannot be identified
        (no agent attached — e.g. bare test states or a task started
        before the agent slot was populated) are kept as a fallback so
        legitimate stops are not silently dropped.

        Args:
            viewer_tab_id: The subscriber/viewer tab id to look up.

        Returns:
            A peer tab id that owns the cooperative stop event for the
            running task, or ``None`` if not found.
        """
        with self.printer._lock:
            peer_lists = [
                (JsonPrinter._coerce_task_id(task_id), list(viewers))
                for task_id, viewers in self.printer._subscribers.items()
                if viewer_tab_id in viewers
            ]
        fallback: str | None = None
        with self._state_lock:
            for task_key, peers in peer_lists:
                for peer in peers:
                    if peer == viewer_tab_id:
                        continue
                    state = _RunningAgentState.running_agent_states.get(peer)
                    if state is None or state.stop_event is None:
                        continue
                    agent_task = (
                        JsonPrinter._coerce_task_id(
                            getattr(state.agent, "_last_task_id", None),
                        )
                        if state.agent is not None
                        else ""
                    )
                    if agent_task and agent_task != task_key:
                        # This peer's live stop_event belongs to a
                        # DIFFERENT task than the one the viewer is
                        # subscribed to; stopping it would hijack an
                        # unrelated task.
                        continue
                    if agent_task:
                        return peer
                    if fallback is None:
                        fallback = peer
        return fallback

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
        q = self._resolve_task_answer_queue()
        # M4 — when the tab has no answer queue (e.g. the tab was closed
        # mid-question) there is no path that can ever return a response.
        # Refuse to busy-loop forever; raise immediately so the agent
        # thread can unwind and the user-facing task can finish.
        if q is None:
            raise KeyboardInterrupt(
                "User answer queue is missing (tab closed?); aborting wait",
            )
        # Wake the blocking ``q.get`` immediately when ``stop`` fires
        # by enqueuing a sentinel from a watcher thread.  Avoids the
        # previous 0.5 s polling loop (which could delay the answer
        # delivery to the agent thread by up to half a second after
        # the user clicked Submit) without sacrificing stop-event
        # responsiveness.  The watcher exits cleanly via ``cancelled``
        # once the user's answer arrives.
        sentinel = _STOP_SENTINEL
        cancelled = threading.Event()

        def _wake_on_stop() -> None:
            # Block until either stop fires or the answer arrives
            # (signalled via ``cancelled``).  Polls ``cancelled`` on a
            # short interval so the watcher does not outlive the
            # call after a normal answer.
            while not cancelled.is_set():
                if stop.wait(0.1):
                    if not cancelled.is_set():
                        # Sentinel is identity-checked below;
                        # the queue's element type is ``str`` so
                        # this single non-string put is narrowed
                        # away by the ``is sentinel`` branch.
                        with suppress(queue.Full):
                            q.put_nowait(cast(str, sentinel))
                    return

        watcher = threading.Thread(target=_wake_on_stop, daemon=True)
        watcher.start()
        try:
            item = q.get()
        finally:
            cancelled.set()
        if item is sentinel:
            raise KeyboardInterrupt("Stopped while waiting for user")
        return item

    def _resolve_task_answer_queue(self) -> queue.Queue[str] | None:
        """Resolve the current task's user-answer queue.

        Resolves via the printer's task-id → subscriber-tabs mapping:
        picks the first subscribed tab that has a live
        ``user_answer_queue`` owned by THIS task.  Any tab subscribed
        to this task can answer, but the queue lives on the task-owner
        tab.

        A co-subscriber tab that is itself actively running a
        *different* task also carries a live ``user_answer_queue`` —
        owned by that other task.  Returning it would hijack the other
        task's answers: this task's ``ask_user_question`` would
        consume the answer the user submitted for the other task's
        question, and the other agent would never receive it.  Such
        tabs are skipped (their live agent's ``_last_task_id``
        identifies which task their queue belongs to).

        Returns:
            The owner tab's answer queue, or ``None`` when the
            thread-local ``task_id`` is unset or no subscribed tab
            carries a live queue for this task (e.g. the owner tab was
            closed).
        """
        task_key = getattr(self.printer._thread_local, "task_id", None)
        if not task_key:
            return None
        task_key = JsonPrinter._coerce_task_id(task_key)
        with self.printer._lock:
            tab_ids = list(self.printer._subscribers.get(task_key, ()))
        with self._state_lock:
            for tab_id in tab_ids:
                tab = _RunningAgentState.running_agent_states.get(tab_id)
                if tab is None or tab.user_answer_queue is None:
                    continue
                agent_task = (
                    JsonPrinter._coerce_task_id(
                        getattr(tab.agent, "_last_task_id", None),
                    )
                    if tab.agent is not None
                    else ""
                )
                if tab.is_task_active and agent_task and agent_task != task_key:
                    # This subscriber is running a DIFFERENT task; its
                    # queue belongs to that task, not this one.
                    continue
                return tab.user_answer_queue
        return None

    def _ask_user_question(self, question: str) -> str:
        """Callback for agent questions."""
        # Discard any answer already sitting in the queue: it predates
        # this question, so it can only be a stale duplicate — e.g. a
        # second viewer tab's still-open ``askUser`` modal for the
        # PREVIOUS question submitted after the agent had consumed the
        # first viewer's answer (nothing closes the other viewers'
        # modals).  Without the drain, that stale answer would answer
        # THIS question instantly and the user would never see it.
        # Held under ``_state_lock`` so the drain cannot interleave
        # with ``_cmd_user_answer``'s drain-then-put sequence.
        task_key = JsonPrinter._coerce_task_id(
            getattr(self.printer._thread_local, "task_id", None),
        )
        q = self._resolve_task_answer_queue()
        if q is not None:
            with self._state_lock:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:  # pragma: no cover — race guard
                        break
                if task_key:
                    self._pending_user_answer_tasks[id(q)] = task_key
        try:
            self.printer.broadcast({
                "type": "askUser",
                "question": question,
            })
            return self._await_user_response()
        finally:
            if q is not None:
                with self._state_lock:
                    self._pending_user_answer_tasks.pop(id(q), None)
