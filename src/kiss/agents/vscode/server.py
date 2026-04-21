"""VS Code extension backend server for Sorcar agent.

This module provides a JSON-based stdio interface between the VS Code
extension and the Sorcar agent. Commands are read from stdin as JSON
lines, and events are written to stdout as JSON lines.
"""

from __future__ import annotations

import base64
import ctypes
import json
import logging
import os
import queue
import re
import sys
import threading
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, repo_lock
from kiss.agents.sorcar.persistence import (
    _append_chat_event,
    _get_adjacent_task_by_chat_id,
    _load_file_usage,
    _load_history,
    _load_last_model,
    _load_latest_chat_events_by_chat_id,
    _load_model_usage,
    _prefix_match_task,
    _record_file_usage,
    _record_model_usage,
    _save_last_model,
    _save_task_extra,
    _save_task_result,
    _search_history,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.browser_ui import _DISPLAY_EVENT_TYPES, BaseBrowserPrinter
from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _cleanup_merge_data,
    _git,
    _merge_data_dir,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _snapshot_files,
)
from kiss.agents.vscode.helpers import (
    clean_llm_output,
    clip_autocomplete_suggestion,
    fast_model_for,
    generate_followup_text,
    model_vendor,
    rank_file_suggestions,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import MODEL_INFO, get_available_models, get_default_model

logger = logging.getLogger(__name__)


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

ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
    ctypes.c_ulong,
    ctypes.py_object,
]


class VSCodePrinter(BaseBrowserPrinter):
    """Printer that outputs JSON events to stdout for VS Code extension.

    Inherits from BaseBrowserPrinter to get identical event parsing and
    emission (thinking_start/delta/end, text_delta/end, tool_call,
    tool_result, system_output, result). Overrides
    broadcast() to write JSON lines to stdout instead of SSE queues.
    """

    def __init__(self) -> None:
        super().__init__()
        self._stdout_lock = threading.Lock()
        self._persist_agents: dict[str, Any] = {}

    def broadcast(self, event: dict[str, Any]) -> None:
        """Write event as a JSON line to stdout, record it, and persist to DB.

        Injects ``tabId`` from thread-local storage when available so the
        frontend can route events to the correct chat tab.

        Display events are persisted to the database via
        ``_append_chat_event`` as they are created, provided a
        per-tab agent with a valid ``_last_task_id`` is registered
        in ``_persist_agents``.

        The ``_record_event`` call and the stdout write are performed
        inside a single ``_lock`` critical section so recording order
        is guaranteed to match stdout-write order even under
        concurrent broadcasts.  ``_stdout_lock`` is nested inside
        ``_lock`` for defence-in-depth against any future caller that
        writes to stdout directly.

        Args:
            event: The event dictionary to emit.
        """
        tab_id = getattr(self._thread_local, "tab_id", None)
        if tab_id is not None and "tabId" not in event:
            event = {**event, "tabId": tab_id}
        with self._lock:
            self._record_event(event)
            with self._stdout_lock:
                sys.stdout.write(json.dumps(event) + "\n")
                sys.stdout.flush()
        # Persist display events to the database as they are created
        if event.get("type") in _DISPLAY_EVENT_TYPES:
            evt_tab = event.get("tabId")
            if evt_tab is not None:
                agent = self._persist_agents.get(evt_tab)
                if agent is not None:
                    task_id = agent._last_task_id
                    if task_id is not None:
                        _append_chat_event(event, task_id=task_id)


class _TabState:
    """Per-tab state holding the agent, runtime state, and settings.

    Each chat tab owns a single ``WorktreeSorcarAgent`` so concurrent
    tabs never share mutable agent state (chat_id, last_task_id,
    worktree branch, etc.).  The ``use_worktree`` flag is passed to
    ``agent.run()`` per task — when ``False`` the agent short-circuits
    to the plain stateful code path, so no separate non-worktree agent
    instance is needed.  Runtime state (stop event, task thread,
    answer queue, merge flag) also lives here so the server needs
    only a single ``_tab_states`` dict.
    """

    __slots__ = (
        "agent",
        "use_worktree",
        "use_parallel",
        "task_history_id",
        "selected_model",
        "stop_event",
        "task_thread",
        "user_answer_queue",
        "is_merging",
        "is_running_non_wt",
    )

    def __init__(self, tab_id: str, default_model: str) -> None:
        self.agent = WorktreeSorcarAgent("Sorcar VS Code")
        self.use_worktree: bool = False
        self.use_parallel: bool = False
        self.task_history_id: int | None = None
        self.selected_model: str = default_model
        self.stop_event: threading.Event | None = None
        self.task_thread: threading.Thread | None = None
        self.user_answer_queue: queue.Queue[str] | None = None
        self.is_merging: bool = False
        self.is_running_non_wt: bool = False


class VSCodeServer:
    """Backend server for VS Code extension."""

    def __init__(self) -> None:
        self.printer = VSCodePrinter()
        self._tab_states: dict[str, _TabState] = {}
        self.work_dir = os.environ.get("KISS_WORKDIR", os.getcwd())
        persisted = _load_last_model()
        self._default_model = (
            persisted
            or os.environ.get("KISS_MODEL", "")
            or get_default_model()
        )
        # Lock ordering: _state_lock < printer._lock < printer._stdout_lock < printer._bash_lock
        self._state_lock = threading.Lock()
        # Autocomplete state — lazily initialized on first 'complete' command
        # so task processes (which never receive 'complete') don't waste
        # a daemon thread and queue.
        self._complete_seq: int = 0
        self._complete_seq_latest: int = -1
        self._complete_queue: queue.Queue[tuple[str, int, str, str]] | None = None
        self._complete_worker: threading.Thread | None = None
        # File cache — lazily populated on first 'getFiles' or 'complete'
        self._file_cache: list[str] | None = None
        self._last_active_file: str = ""
        self._last_active_content: str = ""

    def _get_tab(self, tab_id: str) -> _TabState:
        """Get or create per-tab state for the given tab.

        Each tab gets its own agent instances so concurrent tabs never
        share mutable agent state (chat_id, task_id, worktree, etc.).
        The tab_id is a frontend string identifier; the agent's chat_id
        is a string assigned by the database on first task insertion.

        Thread-safe: acquires ``_state_lock`` to protect the
        get-or-create pattern against concurrent callers.

        Args:
            tab_id: The frontend tab identifier string.

        Returns:
            The per-tab state object.
        """
        with self._state_lock:
            tab = self._tab_states.get(tab_id)
            if tab is None:
                tab = _TabState(tab_id, self._default_model)
                self._tab_states[tab_id] = tab
            return tab

    def _any_non_wt_running(self) -> bool:
        """True if any tab is running a non-worktree task on the main tree.

        Must be called with ``_state_lock`` held.

        Returns:
            True if at least one tab has ``is_running_non_wt`` set.
        """
        return any(t.is_running_non_wt for t in self._tab_states.values())

    def run(self) -> None:
        """Main loop: read commands from stdin, execute them."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
                self._handle_command(cmd)
            except json.JSONDecodeError as e:
                self.printer.broadcast({"type": "error", "text": f"Invalid JSON: {e}"})
            except Exception as e:  # pragma: no cover
                self.printer.broadcast({"type": "error", "text": str(e)})

    def _handle_command(self, cmd: dict[str, Any]) -> None:
        """Handle a command from VS Code."""

        cmd_type = cmd.get("type")

        if cmd_type == "run":
            tab_id = cmd.get("tabId", "")
            tab = self._get_tab(tab_id)
            with self._state_lock:
                if tab.task_thread is not None and tab.task_thread.is_alive():
                    self.printer.broadcast({
                        "type": "error",
                        "text": "Task already running",
                        "tabId": tab_id,
                    })
                    self.printer.broadcast({"type": "status", "running": False, "tabId": tab_id})
                    return
                # RC-NEW-1: create stop_event and queue before starting
                # the thread so _stop_task always finds a valid event.
                tab.stop_event = threading.Event()
                tab.user_answer_queue = queue.Queue(maxsize=1)
                thread = threading.Thread(
                    target=self._run_task, args=(cmd,), daemon=True
                )
                tab.task_thread = thread
                thread.start()
        elif cmd_type == "stop":
            self._stop_task(cmd.get("tabId"))
        elif cmd_type == "getModels":
            self._get_models()
        elif cmd_type == "selectModel":
            tab_id = cmd.get("tabId", "")
            tab = self._get_tab(tab_id)
            model = cmd.get("model", tab.selected_model)
            tab.selected_model = model
            with self._state_lock:
                self._default_model = model  # new tabs inherit latest selection
            _save_last_model(model)
        elif cmd_type == "getHistory":
            self._get_history(cmd.get("query"), cmd.get("offset", 0), cmd.get("generation", 0))
        elif cmd_type == "getFiles":
            self._get_files(cmd.get("prefix", ""))
        elif cmd_type == "refreshFiles":
            self._refresh_file_cache()
        elif cmd_type == "recordFileUsage":
            path = cmd.get("path", "")
            if path:
                _record_file_usage(path)
        elif cmd_type == "userAnswer":
            # Route answer to the correct tab's queue — require tabId
            ans_tab = cmd.get("tabId")
            with self._state_lock:
                ans_state = self._tab_states.get(ans_tab) if ans_tab is not None else None
                q = ans_state.user_answer_queue if ans_state is not None else None
            if q is None:
                logger.debug("userAnswer dropped: no queue for tabId=%s", ans_tab)
                return
            # Drain any stale answer, then put the new one (P2/D3 fix)
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:  # pragma: no cover — race guard
                    break
            q.put(cmd.get("answer", ""))
        elif cmd_type == "resumeSession":
            raw_id = cmd.get("chatId")
            chat_id = str(raw_id) if raw_id else ""
            if chat_id:
                self._replay_session(chat_id, cmd.get("tabId", ""))
        elif cmd_type == "mergeAction":
            self._handle_merge_action(cmd.get("action", ""), cmd.get("tabId"))
        elif cmd_type == "newChat":
            self._new_chat(cmd.get("tabId", ""))
        elif cmd_type == "complete":
            query = cmd.get("query", "")
            active_file = cmd.get("activeFile")
            active_content = cmd.get("activeFileContent")
            with self._state_lock:
                if active_file:
                    self._last_active_file = active_file
                if active_content is not None:
                    self._last_active_content = active_content
                snapshot_file = self._last_active_file
                snapshot_content = self._last_active_content
                self._complete_seq += 1
                seq = self._complete_seq
                self._complete_seq_latest = seq
            if query:
                self._ensure_complete_worker()
                self._complete_queue.put((query, seq, snapshot_file, snapshot_content))  # type: ignore[union-attr]
        elif cmd_type == "getInputHistory":
            self._get_input_history()
        elif cmd_type == "getAdjacentTask":
            adj_tab = self._get_tab(cmd.get("tabId", ""))
            # Get the current chat_id for this tab - if the agent doesn't have one,
            # look up the most recent chat_id from history for adjacent task navigation
            chat_id = adj_tab.agent.chat_id
            if chat_id == "":
                # No active chat_id in this tab, look up the most recent one from history
                entries = _load_history(limit=1)
                if entries:
                    chat_id = str(entries[0].get("chat_id", "") or "")
            self._get_adjacent_task(
                chat_id,
                cmd.get("task", ""),
                cmd.get("direction", "prev"),
            )
        elif cmd_type == "generateCommitMessage":
            threading.Thread(
                target=self._generate_commit_message, daemon=True
            ).start()
        elif cmd_type == "worktreeAction":
            action = cmd.get("action", "")
            wt_tab_id = cmd.get("tabId", "")
            try:
                result = self._handle_worktree_action(action, wt_tab_id)
            except Exception as e:
                logger.debug("Worktree action error", exc_info=True)
                result = {"success": False, "message": str(e)}
            self.printer.broadcast({"type": "worktree_result", "tabId": wt_tab_id, **result})
        else:
            self.printer.broadcast({"type": "error", "text": f"Unknown command: {cmd_type}"})

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
        # RC-NEW-3: single lock block for is_merging check + state setup
        # (no TOCTOU gap). stop_event and user_answer_queue are pre-created
        # in _handle_command (RC-NEW-1 fix).
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
            stop_event = tab.stop_event
        self.printer._thread_local.stop_event = stop_event

        # Use tab_id as chat_id for new sessions
        if tab_id and tab.agent.chat_id == "":
            tab.agent._chat_id = tab_id

        self.printer.broadcast({"type": "clear", "chat_id": tab.agent.chat_id})

        # Fix 4 (defense-in-depth): block a new non-worktree task while
        # a worktree merge is in progress on the same repo.
        if not tab.use_worktree:
            with self._state_lock:
                if any(
                    t.is_merging and t.use_worktree
                    for t in self._tab_states.values()
                ):
                    self.printer.broadcast({
                        "type": "error",
                        "text": "A worktree merge is in progress. "
                        "Wait for it to finish before starting a task.",
                        "tabId": tab_id,
                    })
                    return

        # Git snapshot captures pre-task state — only needed for
        # non-worktree merge view (worktree mode uses baseline commits
        # and its own diff; saving here would nuke another tab's data).
        pre_hunks: dict[str, list[tuple[int, int, int, int]]] = {}
        pre_untracked: set[str] = set()
        pre_file_hashes: dict[str, str] | None = None
        pre_head_sha: str | None = None
        if not tab.use_worktree:
            # BUG-55 fix: set is_running_non_wt BEFORE capturing the
            # snapshot so concurrent worktree merges are blocked during
            # the entire snapshot + task window (closes TOCTOU gap).
            with self._state_lock:
                tab.is_running_non_wt = True
            try:
                repo = GitWorktreeOps.discover_repo(Path(work_dir))
                pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
                    self._capture_pre_snapshot(work_dir, repo, tab_id)
                )
            except BaseException:
                with self._state_lock:
                    tab.is_running_non_wt = False
                raise

        # BUG-B fix: if this worktree tab has a pending branch from a
        # prior run and a non-wt agent is writing to the main tree,
        # clear the stale worktree reference (branch preserved in git)
        # so _try_setup_worktree's _release_worktree is a no-op.
        if tab.use_worktree and tab.agent._wt_pending:
            with self._state_lock:
                if self._any_non_wt_running():
                    tab.agent._merge_conflict_warning = (
                        f"Could not auto-merge branch "
                        f"'{tab.agent._wt_branch}' because another "
                        "task is running on the main working tree. "
                        "The branch is preserved for manual resolution."
                    )
                    tab.agent._wt = None

        # start_recording inside try so stop_recording always runs (P14 fix)
        result_summary = "Agent Failed Abruptly"
        task_end_event: dict[str, Any] | None = None
        try:
            self.printer.start_recording()
            self.printer._persist_agents[tab_id] = tab.agent
            tab.task_history_id = None
            subtasks = parse_task_tags(prompt)
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
                        use_worktree=tab.use_worktree,
                        _skip_persistence=True,
                    )
                    result_summary = self._extract_result_summary() or "No summary available"
                    task_end_event = {"type": "task_done"}
                except KeyboardInterrupt:
                    result_summary = "Task stopped by user"
                    task_end_event = {"type": "task_stopped"}
                    break
                except Exception as e:  # pragma: no cover
                    result_summary = f"Task failed: {e}"
                    task_end_event = {"type": "task_error", "text": str(e)}
                    break
                finally:
                    tab.task_history_id = tab.agent._last_task_id
        except BaseException:  # pragma: no cover — async interrupt before inner try
            # P14: interrupt before inner try — ensure stop_recording runs
            task_end_event = task_end_event or {"type": "task_stopped"}
        finally:
            _record_model_usage(model)
            # Entire cleanup wrapped in try/except BaseException (P13 fix)
            try:
                self.printer._persist_agents.pop(tab_id, None)
                self.printer.stop_recording()
                # BUG-61 fix: prepare the non-worktree merge view
                # BEFORE clearing is_running_non_wt.  The diff capture
                # must happen while the flag blocks concurrent worktree
                # merges — otherwise a merge can modify the working
                # tree between flag-clear and diff, corrupting the view.
                # The flag is cleared in the finally below (and in the
                # outer except handler) so it never gets stuck True
                # even if the merge view preparation raises (BUG-39
                # safety preserved).
                if not tab.use_worktree:
                    try:
                        self._prepare_and_start_merge(
                            work_dir, pre_hunks, pre_untracked, pre_file_hashes,
                            base_ref=pre_head_sha or "HEAD",
                            tab_id=tab_id,
                        )
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
                        "is_worktree": tab.use_worktree,
                    },
                    task_id=tab.task_history_id,
                )
                self.printer.broadcast({"type": "tasks_updated"})
                self.printer.reset()
                if tab.use_worktree and tab.agent._wt_pending:
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
                # BUG-39 fix: ensure flag is cleared even when cleanup
                # itself is interrupted.
                if not tab.use_worktree:
                    with self._state_lock:
                        tab.is_running_non_wt = False
                logger.debug("Cleanup interrupted", exc_info=True)
                if task_end_event:
                    self.printer.broadcast(task_end_event)

    def _start_merge_session(
        self, merge_json_path: str, tab_id: str = "",
    ) -> bool:
        """Load merge data from disk and broadcast merge_data + merge_started events.

        Args:
            merge_json_path: Path to the pending-merge.json file.
            tab_id: Frontend tab identifier.  Used to set ``is_merging``
                on the correct tab.

        Returns:
            True if a merge session was started, False otherwise.
        """
        try:
            with open(merge_json_path) as f:
                merge_data = json.load(f)
            files = merge_data.get("files", [])
            if not files:
                return False
            total_hunks = sum(len(f.get("hunks", [])) for f in files)
            if total_hunks == 0:
                return False
            # BUG-41 / RED-6 fix: use explicit tab_id parameter instead
            # of reading from thread-local (which is None on replay path).
            resolved_tab_id = tab_id or getattr(
                self.printer._thread_local, "tab_id", None,
            )
            resolved_tab: _TabState | None = None
            with self._state_lock:
                if resolved_tab_id is not None:
                    resolved_tab = self._tab_states.get(resolved_tab_id)
                    if resolved_tab is not None:
                        resolved_tab.is_merging = True
            # BUG-67 fix: if broadcast raises (e.g. BrokenPipeError),
            # clear is_merging so the tab is not permanently locked.
            try:
                self.printer.broadcast({
                    "type": "merge_data",
                    "data": merge_data,
                    "hunk_count": total_hunks,
                })
                self.printer.broadcast({"type": "merge_started"})
            except BaseException:
                with self._state_lock:
                    if resolved_tab is not None:
                        resolved_tab.is_merging = False
                raise
            return True
        except (OSError, json.JSONDecodeError, KeyError):
            logger.debug("Failed to load merge data", exc_info=True)
            return False

    def _prepare_and_start_merge(
        self,
        work_dir: str,
        pre_hunks: dict[str, list[tuple[int, int, int, int]]] | None = None,
        pre_untracked: set[str] | None = None,
        pre_file_hashes: dict[str, str] | None = None,
        base_ref: str = "HEAD",
        tab_id: str = "",
    ) -> bool:
        """Prepare a merge view and start the merge session if changes exist.

        Combines ``_prepare_merge_view`` and ``_start_merge_session``
        into a single call to eliminate the repeated prepare→check→start
        sequence.

        Args:
            work_dir: Repository root (or worktree) directory.
            pre_hunks: Pre-task diff hunks (empty dict when not applicable).
            pre_untracked: Pre-task untracked file set (empty when not applicable).
            pre_file_hashes: Pre-task MD5 hashes for change detection.
            base_ref: Git ref to diff against (default ``"HEAD"``).
                Pass a baseline commit SHA to include committed agent
                changes in the merge review.
            tab_id: Frontend tab identifier for per-tab merge data isolation.

        Returns:
            True if a merge session was started, False otherwise.
        """
        merge_dir = str(_merge_data_dir(tab_id))
        merge_result = _prepare_merge_view(
            work_dir,
            merge_dir,
            pre_hunks or {},
            pre_untracked or set(),
            pre_file_hashes,
            base_ref=base_ref,
        )
        if merge_result.get("status") != "opened":
            return False
        merge_json = os.path.join(merge_dir, "pending-merge.json")
        return self._start_merge_session(merge_json, tab_id=tab_id)

    def _start_worktree_merge_review(self, tab_id: str) -> bool:
        """Prepare and start a merge review for worktree changes.

        Builds a merge view from the worktree directory so the user can
        accept/reject individual hunks before the worktree is committed
        and merged.  When a baseline commit exists (user had dirty
        state), diffs against the baseline so that both committed and
        uncommitted agent changes are included in the review while the
        user's pre-existing dirty state is excluded.

        Args:
            tab_id: The tab whose worktree to review.

        Returns:
            True if a merge session was started, False otherwise.
        """
        tab = self._get_tab(tab_id)
        wt_dir = tab.agent._wt_dir
        if wt_dir is None or not wt_dir.exists():
            return False
        # Diff against the baseline commit when available so that
        # committed agent changes are visible in the merge review,
        # not just uncommitted ones (BUG-4 fix).
        base_ref = tab.agent._baseline_commit or "HEAD"
        try:
            return self._prepare_and_start_merge(
                str(wt_dir), base_ref=base_ref, tab_id=tab_id,
            )
        except BaseException:
            logger.debug("Worktree merge review error", exc_info=True)
            return False

    def _handle_merge_action(self, action: str, tab_id: str | None = None) -> None:
        """Handle merge accept/reject actions from the extension.

        Only ``all-done`` triggers cleanup. Individual ``accept``/``reject``
        actions are tracked on the TypeScript side; the Python server
        only needs to know when the entire merge session is finished.

        Args:
            action: The merge action string (e.g. ``"all-done"``).
            tab_id: The tab whose merge session is being finished.
        """
        if action == "all-done":
            self._finish_merge(tab_id)

    def _finish_merge(self, tab_id: str | None = None) -> None:
        """End the merge session for a specific tab.

        When a worktree task is pending, emits ``worktree_done`` so the
        user sees merge/discard buttons only after the hunk review is
        complete.

        Args:
            tab_id: The tab whose merge session is finished. When *None*,
                all merge sessions are cleared.
        """
        with self._state_lock:
            if tab_id is not None:
                tab = self._tab_states.get(tab_id)
                if tab is not None:
                    tab.is_merging = False
            else:
                for tab in self._tab_states.values():
                    tab.is_merging = False
        event: dict[str, Any] = {"type": "merge_ended"}
        if tab_id is not None:
            event["tabId"] = tab_id
        self.printer.broadcast(event)
        # Per-tab merge data cleanup — each tab has its own directory
        if tab_id is not None:
            _cleanup_merge_data(str(_merge_data_dir(tab_id)))
        else:
            _cleanup_merge_data(str(_merge_data_dir()))

        # Emit deferred worktree_done after merge review completes
        if tab_id is not None:
            self._present_pending_worktree(tab_id, try_merge_review=False)

    def _stop_task(self, tab_id: str | None = None) -> None:
        """Signal the agent to stop.

        Sets the cooperative stop event and, if the task thread doesn't
        exit promptly, forces a ``KeyboardInterrupt`` in the task thread
        using ``ctypes.pythonapi.PyThreadState_SetAsyncExc``.  This
        handles the case where the agent is blocked in an LLM API call
        or other I/O and never reaches a cooperative ``_check_stop()``
        call.

        Args:
            tab_id: The tab to stop.  When *None*, stops all running tabs.
        """
        with self._state_lock:
            if tab_id is not None:
                tab = self._tab_states.get(tab_id)
                pairs = [(tab.stop_event, tab.task_thread)] if tab is not None else []
            else:
                pairs = [
                    (t.stop_event, t.task_thread)
                    for t in self._tab_states.values()
                ]
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
                    # Thread ID not found — thread already exited
                    return
                if rc > 1:  # pragma: no cover — rare: exception set in multiple states
                    # Undo: clear the exception in all affected thread states
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
        tab = self._tab_states.get(tab_id) if tab_id is not None else None
        q = tab.user_answer_queue if tab is not None else None
        while True:
            if q is not None:
                try:
                    return q.get(timeout=0.5)
                except queue.Empty:
                    pass
            else:
                stop.wait(timeout=0.5)
            if stop.is_set():
                raise KeyboardInterrupt("Stopped while waiting for user")

    def _ask_user_question(self, question: str) -> str:
        """Callback for agent questions."""
        self.printer.broadcast({
            "type": "askUser",
            "question": question,
        })
        return self._await_user_response()

    def _get_models(self) -> None:
        """Send available models list with usage counts and pricing."""
        usage = _load_model_usage()
        models_list: list[dict[str, Any]] = []
        sort_keys: dict[str, tuple[int, float]] = {}
        for name in get_available_models():
            info = MODEL_INFO.get(name)
            if info and info.is_function_calling_supported:
                vendor_name, vendor_order = model_vendor(name)
                models_list.append({
                    "name": name,
                    "inp": info.input_price_per_1M,
                    "out": info.output_price_per_1M,
                    "uses": usage.get(name, 0),
                    "vendor": vendor_name,
                })
                price = float(info.input_price_per_1M) + float(info.output_price_per_1M)
                sort_keys[name] = (vendor_order, -price)
        models_list.sort(key=lambda m: sort_keys[m["name"]])
        self.printer.broadcast({
            "type": "models",
            "models": models_list,
            "selected": self._default_model,
        })

    def _get_history(self, query: str | None, offset: int = 0, generation: int = 0) -> None:
        """Send conversation history with pagination support."""
        if query:
            entries = _search_history(query, limit=50, offset=offset)
        else:
            entries = _load_history(limit=50, offset=offset)

        sessions = []
        for entry in entries:
            task = str(entry.get("task", ""))
            has_events = bool(entry.get("has_events", False))
            chat_id = str(entry.get("chat_id", "") or "")
            sessions.append({
                "id": chat_id,
                "title": task[:50] + "..." if len(task) > 50 else task,
                "timestamp": entry.get("timestamp", 0),
                "preview": task,
                "has_events": has_events,
            })
        self.printer.broadcast({
            "type": "history", "sessions": sessions,
            "offset": offset, "generation": generation,
        })

    def _get_input_history(self) -> None:
        """Send deduplicated task texts for arrow-key cycling.

        Loads the full persisted history so ArrowUp can traverse every
        distinct task stored in ``history.db``, not just an arbitrary
        recent subset.
        """
        entries = _load_history()
        seen: set[str] = set()
        tasks: list[str] = []
        for e in entries:
            task = str(e.get("task", "")).strip()
            if task and task not in seen:
                seen.add(task)
                tasks.append(task)
        self.printer.broadcast({"type": "inputHistory", "tasks": tasks})

    def _new_chat(self, tab_id: str) -> None:
        """Start a new chat session for the given tab.

        Resets the agent's ``chat_id`` to ``""`` so the next task starts
        a fresh session.  The tab_id (frontend key) does not change.
        Broadcasts any pending warnings (stash-pop failure or merge
        conflict) from the auto-release, then emits ``showWelcome``
        so the frontend displays the welcome messages in the tab.

        Args:
            tab_id: The frontend tab identifier.
        """
        tab = self._get_tab(tab_id)
        # BUG-65 fix: refuse while a merge review is active — calling
        # ``tab.agent.new_chat()`` would trigger ``_release_worktree``
        # which auto-commits and squash-merges, destroying the user's
        # hunk-picking intent and leaving the VS Code merge view
        # stale.  Symmetric with the ``_run_task_inner`` guard.
        with self._state_lock:
            if tab.is_merging:
                self.printer.broadcast({
                    "type": "error",
                    "text": "Cannot start a new chat while a merge review is in progress."
                            " Accept or reject all changes first.",
                    "tabId": tab_id,
                })
                return
        # BUG-44 fix: check _wt_pending regardless of use_worktree —
        # a tab may have switched modes but still have a pending
        # worktree from the previous session.
        # BUG-35 fix: if a non-worktree task is running on the main
        # tree, skip auto-release to avoid stashing its work.
        if tab.agent._wt_pending:
            with self._state_lock:
                if self._any_non_wt_running():
                    self.printer.broadcast({
                        "type": "warning",
                        "message": (
                            "Another tab is running a task on the main working "
                            "tree. The pending worktree branch will be merged "
                            "when the task finishes."
                        ),
                        "tabId": tab_id,
                    })
                    # Reset chat state without releasing the worktree
                    super(type(tab.agent), tab.agent).new_chat()
                    self.printer.broadcast({"type": "showWelcome", "tabId": tab_id})
                    return
        tab.agent.new_chat()
        # Surface warnings from auto-releasing a prior worktree
        if tab.agent._stash_pop_warning:
            self.printer.broadcast({
                "type": "warning",
                "message": tab.agent._stash_pop_warning,
                "tabId": tab_id,
            })
            tab.agent._stash_pop_warning = None
        if tab.agent._merge_conflict_warning:
            self.printer.broadcast({
                "type": "warning",
                "message": tab.agent._merge_conflict_warning,
                "tabId": tab_id,
            })
            tab.agent._merge_conflict_warning = None
        self.printer.broadcast({"type": "showWelcome", "tabId": tab_id})

    def _replay_session(self, chat_id: str, tab_id: str = "") -> None:
        """Replay recorded chat events for a previous chat session.

        Sets the tab's agent chat_id to match the resumed session.
        The tab_id (frontend key in ``_tab_states``) does not change.

        Args:
            chat_id: The string chat session identifier to replay.
            tab_id: The frontend tab identifier.
        """
        result = _load_latest_chat_events_by_chat_id(chat_id)
        if not result or not result.get("events"):
            return
        tab = self._get_tab(tab_id) if tab_id else self._get_tab(str(chat_id))
        tab.agent.resume_chat_by_id(chat_id)

        # Restore use_worktree from persisted extra data (BUG-10 fix).
        # Without this, pending worktrees are invisible after restart
        # because _emit_pending_worktree returns early when use_worktree
        # is False.
        extra_str = str(result.get("extra", "") or "")
        if extra_str:
            try:
                extra = json.loads(extra_str)
                if extra.get("is_worktree"):
                    tab.use_worktree = True
            except (json.JSONDecodeError, TypeError):
                pass

        self.printer.broadcast({
            "type": "task_events",
            "events": result["events"],
            "task": result["task"],
            "chat_id": chat_id,
            "extra": result.get("extra", ""),
            "tabId": tab_id,
        })
        self._emit_pending_worktree(tab_id)

    def _broadcast_worktree_done(self, changed: list[str], tab_id: str = "") -> None:
        """Broadcast a ``worktree_done`` event with the current worktree state.

        Args:
            changed: List of file paths changed in the worktree.
            tab_id: The tab that owns the worktree.
        """
        wt = self._get_tab(tab_id).agent
        self.printer.broadcast({
            "type": "worktree_done",
            "branch": wt._wt_branch,
            "worktreeDir": str(wt._wt_dir),
            "originalBranch": wt._original_branch,
            "changedFiles": changed,
            "hasConflict": self._check_merge_conflict(tab_id) if changed else False,
        })

    def _emit_pending_worktree(self, tab_id: str = "") -> None:
        """Emit merge review or ``worktree_done`` for a pending worktree branch.

        Called after replaying a session.  Restores worktree state
        from git (for post-restart resume) and delegates to
        :meth:`_present_pending_worktree`.

        Args:
            tab_id: The tab to check for pending worktree.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return
        self._ensure_worktree_state(tab_id)
        self._present_pending_worktree(tab_id, try_merge_review=True)

    def _present_pending_worktree(
        self, tab_id: str, *, try_merge_review: bool,
    ) -> None:
        """Auto-discard, start merge review, or emit ``worktree_done``.

        Single source of truth for post-task / post-merge-review /
        session-resume handling of a pending worktree (RED-10 fix).

        Behavior:
        - No pending worktree: return.
        - Worktree has changed files and *try_merge_review* is True:
          attempt to start a merge review; on failure broadcast
          ``worktree_done``.
        - Worktree has changed files and *try_merge_review* is False
          (merge review already finished): broadcast ``worktree_done``.
        - Worktree has no changes and no non-wt task is running:
          auto-discard.
        - Worktree has no changes but a non-wt task is running:
          broadcast ``worktree_done`` so the user is aware of the
          pending branch and can take manual action later
          (BUG-68 fix — previously silent for the post-task and
          post-merge-review paths).

        Args:
            tab_id: The tab with a pending worktree.
            try_merge_review: Whether to attempt starting a merge
                review before falling back.  Pass False after a
                merge review has already been completed.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree or not tab.agent._wt_pending:
            return
        changed = self._get_worktree_changed_files(tab_id)
        if changed and try_merge_review and self._start_worktree_merge_review(tab_id):
            return
        if not changed:
            with self._state_lock:
                non_wt_busy = self._any_non_wt_running()
            if not non_wt_busy:
                tab.agent.discard()
                return
        self._broadcast_worktree_done(changed, tab_id)

    def _ensure_worktree_state(self, tab_id: str = "") -> None:
        """Restore agent worktree state from git if not already set.

        Discovers the repo root and calls ``_restore_from_git()`` so
        that ``merge()``/``discard()`` work even after a server process
        restart where in-memory state was lost.
        Only applicable when using the worktree agent.

        Args:
            tab_id: The tab whose worktree state to restore.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return
        wt = tab.agent
        repo_root = wt._repo_root
        if repo_root is None:
            repo_root = GitWorktreeOps.discover_repo(Path(self.work_dir))
            if repo_root is None:
                return
        wt._restore_from_git(repo_root)

    def _generate_followup_async(
        self,
        task: str,
        result: str,
        task_id: int | None,
    ) -> None:
        """Generate and broadcast a follow-up suggestion in a background thread.

        The suggestion is broadcast to the webview and also appended to
        the persisted chat events so it survives panel re-creation.

        Args:
            task: The completed task description.
            result: The task result summary.
            task_id: Stable history row id for the completed task.
        """
        owner_tab = getattr(self.printer._thread_local, "tab_id", None)

        def _run() -> None:
            if owner_tab is not None:
                self.printer._thread_local.tab_id = owner_tab
            try:
                suggestion = generate_followup_text(
                    task, result, fast_model_for()
                )
                if suggestion:  # pragma: no cover — requires LLM API call
                    event: dict[str, object] = {
                        "type": "followup_suggestion",
                        "text": suggestion,
                    }
                    self.printer.broadcast(event)
                    _append_chat_event(event, task_id=task_id, task=task)
            except Exception:  # pragma: no cover — LLM API error handler
                logger.debug("Async followup generation failed", exc_info=True)

        threading.Thread(target=_run, daemon=True).start()

    def _extract_result_summary(self) -> str:
        """Extract result summary from the current recording."""
        events = self.printer.peek_recording()
        for ev in reversed(events):
            if ev.get("type") == "result":
                summary = ev.get("summary") or ev.get("text") or ""
                return str(summary)
        return ""

    def _complete_from_active_file(
        self, query: str, snapshot_file: str = "", snapshot_content: str = ""
    ) -> str:
        """Complete the trailing token of *query* using identifiers from the active file.

        Extracts single-word identifiers and dot-chained identifiers
        (e.g. ``self.method``, ``os.path.join``) from the active editor
        buffer (or falls back to reading from disk). Matches the trailing
        token of the query — which may contain dots — against all
        candidates via case-sensitive prefix matching.

        Args:
            query: The full query string from the chat input.
            snapshot_file: Atomically-captured active file path.
            snapshot_content: Atomically-captured active file content.

        Returns:
            The remaining suffix to append, or empty string if no match.
        """
        content = snapshot_content
        if not content:
            active_path = snapshot_file
            if not active_path:
                return ""
            try:
                with open(active_path) as f:
                    content = f.read(50000)
            except OSError:
                return ""

        # If the query ends with a non-word character the user has moved past
        # the identifier; don't complete a word that is no longer being typed.
        if query and not (query[-1].isalnum() or query[-1] == "_" or query[-1] == "."):
            return ""
        # Extract the trailing token (may include dots for chains)
        m = re.search(r"([\w][\w.]*)$", query)
        if not m:
            return ""
        partial = m.group(1)
        if len(partial) < 2:
            return ""
        # Extract single-word identifiers (length >= 3) and dot-chained identifiers
        words = set(re.findall(r"\b[A-Za-z_]\w{2,}\b", content))
        chains = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", content))
        candidates = words | chains

        best = ""
        for candidate in candidates:
            if candidate.startswith(partial) and len(candidate) > len(partial):
                suffix = candidate[len(partial):]
                if len(suffix) > len(best):
                    best = suffix
        return best

    def _complete_worker_loop(self) -> None:
        """Persistent worker that drains the complete queue."""
        assert self._complete_queue is not None
        q = self._complete_queue
        while True:
            item = q.get()
            # Drain to latest request (skip stale ones)
            while not q.empty():
                try:
                    item = q.get_nowait()
                except queue.Empty:  # pragma: no cover — race guard
                    break
            query, seq, snapshot_file, snapshot_content = item
            self._complete(query, seq, snapshot_file, snapshot_content)

    def _complete(
        self,
        query: str,
        seq: int = -1,
        snapshot_file: str = "",
        snapshot_content: str = "",
    ) -> None:
        """Ghost text autocomplete via fast local prefix matching.

        Args:
            query: Raw query text from the chat input.
            seq: Sequence number for this request. If a newer request has
                been issued (``seq`` no longer matches the counter), this
                call exits early to avoid broadcasting stale results.
            snapshot_file: Atomically-captured active file path.
            snapshot_content: Atomically-captured active file content.
        """
        if seq >= 0:
            with self._state_lock:
                if seq != self._complete_seq_latest:
                    return
        if not query or len(query) < 2:
            self.printer.broadcast({"type": "ghost", "suggestion": "", "query": query})
            return

        match = _prefix_match_task(query)
        if match:
            fast = match[len(query):]
        else:
            fast = self._complete_from_active_file(query, snapshot_file, snapshot_content)
        fast = clip_autocomplete_suggestion(query, fast)
        self.printer.broadcast({"type": "ghost", "suggestion": fast, "query": query})

    def _ensure_complete_worker(self) -> None:
        """Lazily start the autocomplete worker thread on first use.

        Task processes never receive ``complete`` commands, so the
        worker thread and queue are only created for service processes
        that actually need autocomplete.
        """
        if self._complete_worker is not None:
            return
        self._complete_queue = queue.Queue()
        self._complete_worker = threading.Thread(
            target=self._complete_worker_loop, daemon=True
        )
        self._complete_worker.start()

    def _refresh_file_cache(self) -> None:
        """Refresh the file cache from disk in a background thread."""
        from kiss.agents.vscode.diff_merge import _scan_files

        def _do_refresh() -> None:
            result = _scan_files(self.work_dir)
            with self._state_lock:
                self._file_cache = result

        threading.Thread(target=_do_refresh, daemon=True).start()

    def _get_files(self, prefix: str) -> None:
        """Send file list for autocomplete with usage-based sorting."""
        with self._state_lock:
            cache = self._file_cache
        if not cache:
            from kiss.agents.vscode.diff_merge import _scan_files
            cache = _scan_files(self.work_dir)
            # Double-check under lock: a concurrent _refresh_file_cache
            # thread may have published a fresher cache while we were
            # scanning — do not overwrite it with our (older) scan.
            with self._state_lock:
                if self._file_cache is None:
                    self._file_cache = cache
                else:
                    cache = self._file_cache
        usage = _load_file_usage()
        ranked = rank_file_suggestions(cache, prefix, usage)
        self.printer.broadcast({"type": "files", "files": ranked})

    def _check_merge_conflict(self, tab_id: str = "") -> bool:
        """Check if merging the worktree branch into original would conflict.

        Pure query — does **not** commit or otherwise mutate git state
        (BUG-9 fix).  Uses file-level overlap detection between:

        1. Files changed on the original branch since the fork point.
        2. Files changed in the worktree (committed + uncommitted)
           since the fork point.

        When both sides modify the same file, reports a potential
        conflict.  Also checks for dirty main working-tree files that
        overlap with the worktree changes (which would cause
        ``git merge`` to refuse).

        Args:
            tab_id: The tab whose worktree to check.

        Returns:
            True if the merge would likely fail, False otherwise.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return False
        wt = tab.agent._wt
        if wt is None or wt.original_branch is None:
            return False
        wt_dir = wt.wt_dir
        if not wt_dir.exists():
            return False

        # Determine fork points for conflict detection.
        # For the *original branch* diff we need the HEAD at worktree
        # creation time (before the baseline dirty-state commit) so that
        # the user's own dirty edits aren't reported as "original branch
        # changes".  For the *worktree* diff we use the baseline itself
        # so only agent changes are visible.
        #
        # BUG-56 fix: validate baseline with git cat-file -t before
        # using it (consistent with _resolve_base_ref's BUG-51 fix).
        # An invalid baseline would make both diff commands fail
        # silently and return False (no conflict) even when there is
        # one.
        baseline_valid = False
        if wt.baseline_commit:
            check = _git(
                str(wt_dir), "cat-file", "-t", wt.baseline_commit,
            )
            baseline_valid = (
                check.returncode == 0
                and check.stdout.strip() == "commit"
            )
        if baseline_valid:
            assert wt.baseline_commit is not None  # guarded by baseline_valid
            # baseline^ = the commit the worktree branched from (HEAD at
            # creation).  baseline = HEAD + user dirty state.
            orig_fork = f"{wt.baseline_commit}^"
            wt_fork: str = wt.baseline_commit
        else:
            mb = _git(str(wt_dir), "merge-base", "HEAD", wt.original_branch)
            if mb.returncode != 0 or not mb.stdout.strip():
                return False
            orig_fork = wt_fork = mb.stdout.strip()

        # Files changed on original branch since the fork
        orig_diff = _git(
            str(wt.repo_root), "diff", "--name-only",
            orig_fork, wt.original_branch,
        )
        orig_files = (
            set(orig_diff.stdout.strip().splitlines())
            if orig_diff.returncode == 0 else set()
        )

        # Files changed in the worktree (committed + uncommitted) since fork
        wt_diff = _git(str(wt_dir), "diff", "--name-only", wt_fork)
        wt_files = (
            set(wt_diff.stdout.strip().splitlines())
            if wt_diff.returncode == 0 else set()
        )
        wt_files.update(_capture_untracked(str(wt_dir)))

        # Check 1: both sides modified the same file → likely conflict
        if orig_files & wt_files:
            return True

        # Check 2: dirty main working-tree files that overlap with
        # worktree changes (git merge would refuse).
        # Fix 3 (BUG-37): skip this check when a non-worktree agent is
        # running — its in-flight writes would cause false-positive
        # conflict reports since they're not user edits.
        with self._state_lock:
            if self._any_non_wt_running():
                return False
        # INC-6 fix: check both unstaged AND staged files — staged
        # files that overlap with worktree changes would also cause
        # git merge to refuse.
        # BUG-70 fix: also check untracked files in main.  The auto-
        # merge flow (``stash --include-untracked`` → squash-merge →
        # ``stash pop``) fails at pop when the squash recreates an
        # untracked file, because pop can't overwrite the now-
        # existing file.
        dirty = set(GitWorktreeOps.unstaged_files(wt.repo_root))
        dirty.update(GitWorktreeOps.staged_files(wt.repo_root))
        dirty.update(_capture_untracked(str(wt.repo_root)))
        return bool(dirty & wt_files)

    @staticmethod
    def _resolve_base_ref(
        git_dir: str, baseline: str | None, original_branch: str,
        tip: str = "HEAD",
    ) -> str:
        """Resolve the base ref for worktree diff operations.

        Uses the baseline commit when available **and valid** (i.e. the
        SHA exists in the repository), otherwise falls back to
        ``git merge-base`` between *tip* and *original_branch*.

        BUG-51 fix: validates baseline SHA with ``git cat-file -t``
        before returning it.  An invalid baseline (e.g. from a
        force-pushed branch or corrupt config) is silently ignored
        so callers get a usable ref instead of a guaranteed-to-fail one.

        Args:
            git_dir: Directory to run git commands in.
            baseline: Baseline commit SHA, or ``None``.
            original_branch: The user's original branch name.
            tip: The tip ref to compute merge-base against (default ``HEAD``).

        Returns:
            A git ref string suitable for ``git diff``.
        """
        if baseline:
            check = _git(git_dir, "cat-file", "-t", baseline)
            if check.returncode == 0 and check.stdout.strip() == "commit":
                return baseline
            # Invalid baseline — fall through to merge-base
        mb = _git(git_dir, "merge-base", tip, original_branch)
        if mb.returncode == 0 and mb.stdout.strip():
            return mb.stdout.strip()
        return original_branch

    def _get_worktree_changed_files(self, tab_id: str = "") -> list[str]:
        """List files changed in the worktree vs the original branch.

        Detects both committed changes on the worktree branch and
        uncommitted changes in the worktree working tree.  When the
        worktree directory exists, runs ``git diff`` and
        ``git ls-files --others`` inside it so that uncommitted
        edits and new files are included.  Falls back to a branch-
        to-branch diff when the worktree has already been removed.

        Args:
            tab_id: The tab whose worktree to check.

        Returns:
            Sorted deduplicated list of relative file paths.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return []
        wt = tab.agent
        if not wt._original_branch:
            return []
        wt_dir = wt._wt_dir
        if wt_dir and wt_dir.exists():
            base_ref = self._resolve_base_ref(
                str(wt_dir), wt._baseline_commit, wt._original_branch,
            )
            tracked = _git(str(wt_dir), "diff", "--name-only", base_ref)
            if tracked.returncode == 0:
                files = tracked.stdout.strip().splitlines()
            else:
                # BUG-51 fix: git diff failed — fall back to
                # git status to detect any changes rather than
                # returning [] which would trigger auto-discard.
                status = _git(str(wt_dir), "status", "--porcelain")
                files = [
                    line[3:].strip()
                    for line in status.stdout.splitlines()
                    if len(line) >= 4 and line[3:].strip()
                ]
            files.extend(_capture_untracked(str(wt_dir)))
            return sorted(set(files))
        # Worktree already removed — fall back to branch diff
        if not wt._wt_branch:
            return []
        repo_root = str(wt._repo_root) if wt._repo_root else self.work_dir
        base_ref = self._resolve_base_ref(
            repo_root, wt._baseline_commit, wt._original_branch,
            tip=wt._wt_branch,
        )
        result = _git(repo_root, "diff", "--name-only",
                      base_ref,
                      wt._wt_branch)
        return result.stdout.strip().splitlines() if result.returncode == 0 else []

    def _handle_worktree_action(self, action: str, tab_id: str = "") -> dict[str, Any]:
        """Execute a worktree merge/discard/manual action.

        Restores agent worktree state from git if needed (e.g. after a
        server process restart where in-memory state was lost).

        Args:
            action: One of ``"merge"`` or ``"discard"``.
            tab_id: The tab whose worktree to act on.

        Returns:
            Dict with ``success`` bool and ``message`` string.
        """
        tab = self._get_tab(tab_id)
        if not tab.use_worktree:
            return {"success": False, "message": "Worktree mode is not enabled"}
        wt = tab.agent
        if not wt._wt_pending:
            self._ensure_worktree_state(tab_id)
        if action == "merge":
            # Fix 3 (BUG-35): refuse merge while a non-worktree task
            # is writing to the main working tree — stash_if_dirty
            # would capture the agent's in-flight edits.
            with self._state_lock:
                if self._any_non_wt_running():
                    return {
                        "success": False,
                        "message": (
                            "Another tab is running a task on the main working "
                            "tree. Wait for it to finish before merging."
                        ),
                    }
            self.printer.broadcast({
                "type": "worktree_progress",
                "message": "Generating commit message…",
            })
            msg = wt.merge()
            success = "Successfully merged" in msg
            return {"success": success, "message": msg}
        elif action == "discard":
            # INC-2 fix: guard discard's checkout against concurrent
            # non-worktree task writing to the main tree.
            with self._state_lock:
                if self._any_non_wt_running():
                    return {
                        "success": False,
                        "message": (
                            "Another tab is running a task on the main working "
                            "tree. Wait for it to finish before discarding."
                        ),
                    }
            msg = wt.discard()
            return {"success": True, "message": msg}
        return {"success": False, "message": f"Unknown action: {action}"}

    def _get_adjacent_task(self, chat_id: str, task: str, direction: str) -> None:
        """Send events for the adjacent task in the same chat session.

        Args:
            chat_id: The string chat session identifier.
            task: Current task description string (used as timestamp reference).
            direction: ``"prev"`` or ``"next"``.
        """
        result = _get_adjacent_task_by_chat_id(chat_id, task, direction)
        self.printer.broadcast({
            "type": "adjacent_task_events",
            "direction": direction,
            "task": result["task"] if result else "",
            "events": result["events"] if result else [],
        })

    def _generate_commit_message(self) -> None:
        """Generate a git commit message from current changes."""
        try:
            cached_result = _git(self.work_dir, "diff", "--cached")
            diff_text = cached_result.stdout.strip()
            if not diff_text:  # pragma: no branch — LLM API required for else
                self.printer.broadcast({
                    "type": "commitMessage",
                    "message": "",
                    "error": "No staged changes found. Stage files with 'git add' first.",
                })
                return
            self._generate_commit_message_llm(diff_text)  # pragma: no cover
        except Exception:  # pragma: no cover — LLM API error handler
            logger.debug("Commit message generation failed", exc_info=True)
            self.printer.broadcast({
                "type": "commitMessage",
                "message": "",
                "error": "Failed to generate",
            })

    def _generate_commit_message_llm(self, diff_text: str) -> None:  # pragma: no cover
        """Call LLM to generate commit message from diff text."""
        agent = KISSAgent("Commit Message Generator")
        raw = agent.run(
            model_name=fast_model_for(),
            prompt_template=(
                "Generate a nicely markdown formatted, informative git commit message for "
                "these changes. Use conventional commit format with a clear subject "
                "line (type: description) and optionally a body with bullet points "
                "for multiple changes. Return ONLY the commit message text, no "
                "quotes or markdown fences.\n\n{context}"
            ),
            arguments={"context": f"Diff:\n{diff_text}"},
            is_agentic=False,
            verbose=False,
        )
        msg = clean_llm_output(raw)
        self.printer.broadcast({"type": "commitMessage", "message": msg})


def main() -> None:  # pragma: no cover — CLI entry point
    """Main entry point for VS Code backend server."""
    server = VSCodeServer()
    server.run()


if __name__ == "__main__":
    main()
