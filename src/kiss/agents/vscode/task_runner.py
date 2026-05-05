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
    from kiss.agents.vscode.printer import VSCodePrinter
    from kiss.agents.vscode.tab_state import _TabState

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, repo_lock
from kiss.agents.sorcar.persistence import (
    _append_chat_event,
    _save_task_extra,
    _save_task_result,
)
from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _parse_diff_hunks,
    _save_untracked_base,
    _snapshot_files,
)
from kiss.agents.vscode.tab_state import parse_task_tags
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import get_available_models

logger = logging.getLogger(__name__)


class _TaskRunnerMixin:
    """Task-lifecycle methods (run, stop, user-question callback)."""

    if TYPE_CHECKING:
        printer: VSCodePrinter
        work_dir: str
        _state_lock: threading.Lock
        _tab_states: dict[str, _TabState]

        def _get_tab(self, tab_id: str) -> _TabState: ...
        def _any_non_wt_running(self) -> bool: ...
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
        def _present_pending_worktree(
            self, tab_id: str, *, try_merge_review: bool,
        ) -> None: ...
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
        self.printer._thread_local.tab_id = tab_id
        try:
            self.printer.broadcast({"type": "status", "running": True})
            self._run_task_inner(cmd)
        finally:
            with self._state_lock:
                tab = self._tab_states.get(tab_id)
                if tab is not None:
                    tab.task_thread = None
                    tab.stop_event = None
                    tab.user_answer_queue = None
                    tab.is_task_active = False
                    tab.is_running_non_wt = False
                self.printer.broadcast({"type": "status", "running": False})

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
            tab.is_task_active = True
            stop_event = tab.stop_event
            use_worktree = tab.use_worktree
        self.printer._thread_local.stop_event = stop_event

        if tab_id and tab.agent.chat_id == "":
            tab.agent._chat_id = tab_id

        self.printer.broadcast({"type": "clear", "chat_id": tab.agent.chat_id})

        if not use_worktree:
            with self._state_lock:
                if any(
                    t.is_merging and t.use_worktree
                    for t in self._tab_states.values()
                ):
                    tab.is_task_active = False
                    self.printer.broadcast({
                        "type": "error",
                        "text": "A worktree merge is in progress. "
                        "Wait for it to finish before starting a task.",
                        "tabId": tab_id,
                    })
                    return

        pre_hunks: dict[str, list[tuple[int, int, int, int]]] = {}
        pre_untracked: set[str] = set()
        pre_file_hashes: dict[str, str] | None = None
        pre_head_sha: str | None = None
        if not use_worktree:
            with self._state_lock:
                tab.is_running_non_wt = True
                deferred = tab.deferred_snapshot
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
        try:
            self.printer.start_recording()
            self.printer._persist_agents[tab_id] = tab.agent
            tab.task_history_id = None
            subtasks = parse_task_tags(prompt)
            from kiss.agents.vscode.vscode_config import load_config as _load_cfg

            _vcfg = _load_cfg()
            _cfg_budget = float(_vcfg.get("max_budget", 100))
            _cfg_web = _vcfg.get("use_web_browser", True)

            for task_prompt in subtasks:
                try:
                    tab.agent.run(
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
                        _skip_persistence=True,
                    )
                    result_summary = self._extract_result_summary() or "No summary available"
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
                })
                break
        except BaseException:  # pragma: no cover — async interrupt before inner try
            task_end_event = task_end_event or {"type": "task_stopped"}
        finally:
            try:
                with self._state_lock:
                    tab.is_task_active = False
                self.printer._persist_agents.pop(tab_id, None)
                self.printer.stop_recording()
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
                        else:
                            with self._state_lock:
                                tab.deferred_snapshot = None
                            merge_started = self._prepare_and_start_merge(
                                work_dir, pre_hunks, pre_untracked, pre_file_hashes,
                                base_ref=pre_head_sha or "HEAD",
                                tab_id=tab_id,
                            )
                            if not merge_started:
                                changed = self._main_dirty_files()
                                if changed:
                                    self.printer.broadcast({
                                        "type": "autocommit_prompt",
                                        "tabId": tab_id,
                                        "changedFiles": changed,
                                    })
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
                        "is_parallel": tab.use_parallel,
                        "is_worktree": use_worktree,
                    },
                    task_id=tab.task_history_id,
                )
                self.printer.broadcast({"type": "tasks_updated"})
                self.printer.reset()
                if use_worktree and tab.agent._wt_pending and not tab.skip_merge:
                    try:
                        self._present_pending_worktree(
                            tab_id, try_merge_review=True,
                        )
                    except BaseException:
                        logger.debug("Worktree merge review error", exc_info=True)
                if task_end_event:  # pragma: no branch — always set
                    self.printer.broadcast(task_end_event)
                if tab.task_history_id is not None:
                    self._generate_followup_async(
                        prompt,
                        result_summary,
                        tab.task_history_id,
                    )
                tab.task_history_id = None
            except BaseException:  # pragma: no cover — cleanup interrupted
                with self._state_lock:
                    tab.is_task_active = False
                    if not use_worktree:
                        tab.is_running_non_wt = False
                logger.debug("Cleanup interrupted", exc_info=True)
                if task_end_event:
                    self.printer.broadcast(task_end_event)

    def _stop_task(self, tab_id: str = "") -> None:
        """Signal the agent to stop.

        Sets the cooperative stop event and, if the task thread doesn't
        exit promptly, forces a ``KeyboardInterrupt`` in the task thread
        using ``ctypes.pythonapi.PyThreadState_SetAsyncExc``.  This
        handles the case where the agent is blocked in an LLM API call
        or other I/O and never reaches a cooperative ``_check_stop()``
        call.

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
            tab = self._tab_states.get(tab_id)
            pairs = [(tab.stop_event, tab.task_thread)] if tab is not None else []
        for stop_event, task_thread in pairs:
            if stop_event:
                stop_event.set()
            if task_thread is not None and task_thread.is_alive():
                threading.Thread(
                    target=self._force_stop_thread,
                    args=(task_thread,),
                    daemon=True,
                ).start()

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
        tab_id = getattr(self.printer._thread_local, "tab_id", None)
        with self._state_lock:
            tab = self._tab_states.get(tab_id) if tab_id is not None else None
            q = tab.user_answer_queue if tab is not None else None
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
