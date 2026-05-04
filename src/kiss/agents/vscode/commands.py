"""Command handlers for the VS Code server.

Split out of ``server.py`` for organisation.  ``_CommandsMixin``
provides one ``_cmd_*`` method per frontend command type plus the
class-level ``_HANDLERS`` dispatch table consumed by
``VSCodeServer._handle_command``.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.persistence import (
    _record_file_usage,
    _record_model_usage,
)
from kiss.agents.vscode.tab_state import _TabState

if TYPE_CHECKING:
    from kiss.agents.vscode.printer import VSCodePrinter

logger = logging.getLogger(__name__)


class _CommandsMixin:
    """Methods that implement frontend command handlers."""

    if TYPE_CHECKING:
        printer: VSCodePrinter
        work_dir: str
        _state_lock: threading.Lock
        _tab_states: dict[str, _TabState]
        _default_model: str
        _complete_seq: int
        _complete_seq_latest: int
        _complete_queue: queue.Queue[tuple[str, int, str, str]] | None
        _last_active_file: str
        _last_active_content: str

        def _get_tab(self, tab_id: str) -> _TabState: ...
        def _run_task(self, cmd: dict[str, Any]) -> None: ...
        def _stop_task(self, tab_id: str = "") -> None: ...
        def _get_models(self) -> None: ...
        def _get_history(
            self, query: str | None, offset: int = 0, generation: int = 0
        ) -> None: ...
        def _get_frequent_tasks(self, limit: int = 20) -> None: ...
        def _get_files(self, prefix: str) -> None: ...
        def _refresh_file_cache(self) -> None: ...
        def _replay_session(
            self, chat_id: str, tab_id: str = "", task_id: int | None = None,
        ) -> None: ...
        def _finish_merge(self, tab_id: str = "") -> None: ...
        def _new_chat(self, tab_id: str) -> None: ...
        def _close_tab(self, tab_id: str) -> None: ...
        def _ensure_complete_worker(self) -> None: ...
        def _get_input_history(self) -> None: ...
        def _get_adjacent_task(
            self, chat_id: str, task: str, direction: str, tab_id: str = "",
        ) -> None: ...
        def _generate_commit_message(self) -> None: ...
        def _handle_worktree_action(
            self, action: str, tab_id: str = "",
        ) -> dict[str, Any]: ...
        def _handle_autocommit_action(
            self, action: str, tab_id: str = "",
        ) -> None: ...
        def _handle_delete_task(self, task_id: int) -> None: ...


    def _cmd_run(self, cmd: dict[str, Any]) -> None:
        """Start an agent task in a background thread."""
        tab_id = cmd.get("tabId", "")
        with self._state_lock:
            tab = self._tab_states.get(tab_id)
            if tab is None:
                tab = _TabState(tab_id, self._default_model)
                self._tab_states[tab_id] = tab
            if tab.task_thread is not None and tab.task_thread.is_alive():
                self.printer.broadcast({
                    "type": "error",
                    "text": "Task already running",
                    "tabId": tab_id,
                })
                self.printer.broadcast({"type": "status", "running": True, "tabId": tab_id})
                return
            if "skipMerge" in cmd:
                tab.skip_merge = bool(cmd["skipMerge"])
            tab.stop_event = threading.Event()
            tab.user_answer_queue = queue.Queue(maxsize=1)
            thread = threading.Thread(
                target=self._run_task, args=(cmd,), daemon=True
            )
            tab.task_thread = thread
            thread.start()

    def _cmd_stop(self, cmd: dict[str, Any]) -> None:
        """Stop a running task."""
        self._stop_task(cmd.get("tabId", ""))

    def _cmd_get_models(self, cmd: dict[str, Any]) -> None:
        """Send available models list."""
        self._get_models()

    def _cmd_select_model(self, cmd: dict[str, Any]) -> None:
        """Update the selected model for a tab."""
        tab_id = cmd.get("tabId", "")
        tab = self._get_tab(tab_id)
        model = cmd.get("model", tab.selected_model)
        with self._state_lock:
            tab.selected_model = model
            self._default_model = model
        _record_model_usage(model)

    def _cmd_get_history(self, cmd: dict[str, Any]) -> None:
        """Send conversation history."""
        self._get_history(cmd.get("query"), cmd.get("offset", 0), cmd.get("generation", 0))

    def _cmd_get_frequent_tasks(self, cmd: dict[str, Any]) -> None:
        """Send the top-N most-frequent tasks (default 20)."""
        self._get_frequent_tasks(int(cmd.get("limit", 20)))

    def _cmd_delete_task(self, cmd: dict[str, Any]) -> None:
        """Delete a task from the database and refresh history."""
        task_id = cmd.get("taskId")
        if task_id is not None:
            self._handle_delete_task(int(task_id))

    def _cmd_get_files(self, cmd: dict[str, Any]) -> None:
        """Send file list for autocomplete."""
        self._get_files(cmd.get("prefix", ""))

    def _cmd_refresh_files(self, cmd: dict[str, Any]) -> None:
        """Refresh the file cache."""
        self._refresh_file_cache()

    def _cmd_record_file_usage(self, cmd: dict[str, Any]) -> None:
        """Record a file access for usage-based sorting."""
        path = cmd.get("path", "")
        if path:
            _record_file_usage(path)

    def _cmd_user_answer(self, cmd: dict[str, Any]) -> None:
        """Route a user answer to the correct tab's queue."""
        ans_tab = cmd.get("tabId", "")
        with self._state_lock:
            ans_state = self._tab_states.get(ans_tab)
            q = ans_state.user_answer_queue if ans_state is not None else None
        if q is None:
            logger.debug("userAnswer dropped: no queue for tabId=%s", ans_tab)
            return
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:  # pragma: no cover — race guard
                break
        q.put(cmd.get("answer", ""))

    def _cmd_resume_session(self, cmd: dict[str, Any]) -> None:
        """Replay a previous chat session.

        When ``taskId`` is present, load that specific task instead of
        the latest task in the chat session.
        """
        raw_id = cmd.get("chatId")
        chat_id = str(raw_id) if raw_id else ""
        raw_task_id = cmd.get("taskId")
        task_id = int(raw_task_id) if raw_task_id is not None else None
        if chat_id or task_id is not None:
            self._replay_session(
                chat_id, cmd.get("tabId", ""), task_id=task_id,
            )

    def _cmd_merge_action(self, cmd: dict[str, Any]) -> None:
        """Handle merge accept/reject from the extension.

        Only ``all-done`` triggers cleanup. Individual ``accept``/``reject``
        actions are tracked on the TypeScript side; the Python server
        only needs to know when the entire merge session is finished.
        """
        if cmd.get("action", "") == "all-done":
            self._finish_merge(cmd.get("tabId", ""))

    def _cmd_close_tab(self, cmd: dict[str, Any]) -> None:
        """Clean up backend state for a closed frontend tab."""
        tab_id = cmd.get("tabId", "")
        if tab_id:
            self._close_tab(tab_id)

    def _cmd_new_chat(self, cmd: dict[str, Any]) -> None:
        """Start a new chat session."""
        self._new_chat(cmd.get("tabId", ""))

    def _cmd_complete(self, cmd: dict[str, Any]) -> None:
        """Ghost text autocomplete request."""
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

    def _cmd_get_input_history(self, cmd: dict[str, Any]) -> None:
        """Send deduplicated task texts for arrow-key cycling."""
        self._get_input_history()

    def _cmd_get_adjacent_task(self, cmd: dict[str, Any]) -> None:
        """Send events for the adjacent task in the same chat session.

        Uses only the tab's own agent chat_id.  Previously, when the tab
        had no chat_id the handler fell back to the globally-latest
        chat in history, causing arrow-key navigation in one tab to
        traverse a *different* tab's conversation (C1 fix).
        """
        tab_id = cmd.get("tabId", "")
        adj_tab = self._get_tab(tab_id)
        chat_id = adj_tab.agent.chat_id
        self._get_adjacent_task(
            chat_id,
            cmd.get("task", ""),
            cmd.get("direction", "prev"),
            tab_id,
        )

    def _cmd_generate_commit_message(self, cmd: dict[str, Any]) -> None:
        """Generate a git commit message in the background.

        Runs the generator in a daemon thread and captures the caller's
        ``tabId`` so all ``commitMessage`` events broadcast from the
        worker are tagged via ``printer._thread_local.tab_id``.  This
        prevents the result from leaking into other tabs (B5 fix).
        """
        tab_id = cmd.get("tabId", "")

        def _run() -> None:
            self.printer._thread_local.tab_id = tab_id
            self._generate_commit_message()

        threading.Thread(target=_run, daemon=True).start()

    def _cmd_worktree_action(self, cmd: dict[str, Any]) -> None:
        """Execute a worktree merge/discard action."""
        action = cmd.get("action", "")
        wt_tab_id = cmd.get("tabId", "")
        try:
            result = self._handle_worktree_action(action, wt_tab_id)
        except Exception as e:
            logger.debug("Worktree action error", exc_info=True)
            result = {"success": False, "message": str(e)}
        self.printer.broadcast({"type": "worktree_result", "tabId": wt_tab_id, **result})

    def _cmd_autocommit_action(self, cmd: dict[str, Any]) -> None:
        """Process the user's reply to an autocommit prompt."""
        self._handle_autocommit_action(
            cmd.get("action", ""), cmd.get("tabId", ""),
        )

    def _cmd_get_config(self, cmd: dict[str, Any]) -> None:
        """Send the current configuration to the frontend."""
        from kiss.agents.vscode.vscode_config import get_current_api_keys, load_config

        cfg = load_config()
        api_keys = get_current_api_keys()
        self.printer.broadcast({"type": "configData", "config": cfg, "apiKeys": api_keys})

    def _cmd_save_config(self, cmd: dict[str, Any]) -> None:
        """Save configuration and API keys from the frontend."""
        from kiss.agents.vscode.vscode_config import (
            apply_config_to_env,
            load_config,
            save_api_key_to_shell,
            save_config,
        )

        cfg = cmd.get("config", {})
        save_config(cfg)
        apply_config_to_env(cfg)

        new_work_dir = cfg.get("work_dir", "")
        if new_work_dir:
            self.work_dir = new_work_dir

        api_keys = cmd.get("apiKeys", {})
        for key_name, key_value in api_keys.items():
            if key_value:
                save_api_key_to_shell(key_name, key_value)

        self._get_models()

        new_cfg = load_config()
        self.printer.broadcast({"type": "configData", "config": new_cfg})

    def _cmd_set_skip_merge(self, cmd: dict[str, Any]) -> None:
        """Set the skip_merge flag on a tab.

        When skip_merge is True, the task completion flow will skip the
        merge review and autocommit prompt.  Used by the frontend to
        defer merge/diff until all queued tasks have finished.
        """
        tab_id = cmd.get("tabId", "")
        skip = bool(cmd.get("skip", False))
        tab = self._get_tab(tab_id)
        with self._state_lock:
            tab.skip_merge = skip

    _HANDLERS: dict[str, Any] = {
        "run": _cmd_run,
        "stop": _cmd_stop,
        "getModels": _cmd_get_models,
        "selectModel": _cmd_select_model,
        "getHistory": _cmd_get_history,
        "getFrequentTasks": _cmd_get_frequent_tasks,
        "deleteTask": _cmd_delete_task,
        "getFiles": _cmd_get_files,
        "refreshFiles": _cmd_refresh_files,
        "recordFileUsage": _cmd_record_file_usage,
        "userAnswer": _cmd_user_answer,
        "resumeSession": _cmd_resume_session,
        "mergeAction": _cmd_merge_action,
        "closeTab": _cmd_close_tab,
        "newChat": _cmd_new_chat,
        "complete": _cmd_complete,
        "getInputHistory": _cmd_get_input_history,
        "getAdjacentTask": _cmd_get_adjacent_task,
        "generateCommitMessage": _cmd_generate_commit_message,
        "worktreeAction": _cmd_worktree_action,
        "autocommitAction": _cmd_autocommit_action,
        "setSkipMerge": _cmd_set_skip_merge,
        "getConfig": _cmd_get_config,
        "saveConfig": _cmd_save_config,
    }
