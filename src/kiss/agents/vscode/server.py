"""VS Code extension backend server for Sorcar agent.

This module provides a JSON-based stdio interface between the VS Code
extension and the Sorcar agent. Commands are read from stdin as JSON
lines, and events are written to stdout as JSON lines.

The per-command handlers, task-runner, merge / worktree flow and
autocomplete logic live in sibling mixin modules.  This file keeps the
core dispatch loop, per-tab state accessors, the history / chat /
commit-message helpers, and the ``main`` CLI entry point.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
from typing import Any

from kiss.agents.sorcar.persistence import (
    _append_chat_event,
    _get_adjacent_task_by_chat_id,
    _load_history,
    _load_last_model,
    _load_latest_chat_events_by_chat_id,
    _load_model_usage,
    _search_history,
)
from kiss.agents.vscode.autocomplete import _AutocompleteMixin
from kiss.agents.vscode.commands import _CommandsMixin
from kiss.agents.vscode.diff_merge import (  # noqa: F401 (re-export for tests)
    _cleanup_merge_data,
    _git,
    _merge_data_dir,
)
from kiss.agents.vscode.helpers import (
    fast_model_for,
    generate_commit_message_from_diff,
    generate_followup_text,
    model_vendor,
)
from kiss.agents.vscode.merge_flow import _MergeFlowMixin
from kiss.agents.vscode.printer import VSCodePrinter
from kiss.agents.vscode.tab_state import _TabState, parse_task_tags
from kiss.agents.vscode.task_runner import _TaskRunnerMixin
from kiss.core.models.model_info import MODEL_INFO, get_available_models, get_default_model

__all__ = [
    "VSCodePrinter",
    "VSCodeServer",
    "_TabState",
    "main",
    "parse_task_tags",
]

logger = logging.getLogger(__name__)


class VSCodeServer(
    _CommandsMixin,
    _TaskRunnerMixin,
    _MergeFlowMixin,
    _AutocompleteMixin,
):
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
        self._state_lock = threading.Lock()
        self._complete_seq: int = 0
        self._complete_seq_latest: int = -1
        self._complete_queue: queue.Queue[tuple[str, int, str, str]] | None = None
        self._complete_worker: threading.Thread | None = None
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
            cmd: dict[str, Any] = {}
            try:
                cmd = json.loads(line)
                self._handle_command(cmd)
            except json.JSONDecodeError as e:
                self.printer.broadcast({"type": "error", "text": f"Invalid JSON: {e}"})
            except Exception as e:  # pragma: no cover
                event: dict[str, Any] = {"type": "error", "text": str(e)}
                tab_id = cmd.get("tabId") if isinstance(cmd, dict) else None
                if tab_id is not None:
                    event["tabId"] = tab_id
                self.printer.broadcast(event)

    def _handle_command(self, cmd: dict[str, Any]) -> None:
        """Dispatch a command from VS Code to the appropriate handler."""
        cmd_type: str = cmd.get("type", "")
        handler = self._HANDLERS.get(cmd_type)
        if handler is not None:
            handler(self, cmd)
        else:
            event: dict[str, Any] = {"type": "error", "text": f"Unknown command: {cmd_type}"}
            tab_id = cmd.get("tabId")
            if tab_id is not None:
                event["tabId"] = tab_id
            self.printer.broadcast(event)


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
        distinct task stored in ``sorcar.db``, not just an arbitrary
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

    def _close_tab(self, tab_id: str) -> None:
        """Clean up all backend state for a closed tab.

        Removes the tab from ``_tab_states``, cleans up per-tab printer
        state (bash buffers, recordings), and drops the persist-agent
        reference.  Does nothing if the tab is currently running a task
        or in a merge review — the frontend should stop/resolve those
        first.

        When the tab has a pending worktree, auto-merges it (just like
        starting a new task would) before removing the tab, so the
        worktree branch and directory are not orphaned.

        Args:
            tab_id: The frontend tab identifier to close.
        """
        with self._state_lock:
            tab = self._tab_states.get(tab_id)
            if tab is not None and (
                tab.is_task_active
                or tab.is_merging
                or (tab.task_thread is not None and tab.task_thread.is_alive())
            ):
                return
            self._tab_states.pop(tab_id, None)
        if tab is not None and tab.agent._wt_pending:
            try:
                tab.agent._release_worktree()
            except Exception:
                logger.debug("Worktree release on tab close failed", exc_info=True)
        self.printer.cleanup_tab(tab_id)
        self.printer._persist_agents.pop(tab_id, None)
        _cleanup_merge_data(str(_merge_data_dir(tab_id)))

    def _new_chat(self, tab_id: str) -> None:
        """Start a new chat session for the given tab.

        The ``newChat`` command is only issued by the frontend's
        ``createNewTab`` flow, which always allocates a fresh tab id
        that the backend has never seen before.  ``_get_tab`` creates a
        clean ``_TabState``, so there is no prior run state (no active
        task, no in-progress merge, no pending worktree, no carried-over
        warnings) to guard against here.

        Args:
            tab_id: The frontend tab identifier (a freshly-minted uuid).
        """
        tab = self._get_tab(tab_id)
        tab.agent.new_chat()
        self.printer.broadcast({"type": "showWelcome", "tabId": tab_id})

    def _replay_session(self, chat_id: str, tab_id: str = "") -> None:
        """Replay recorded chat events for a previous chat session.

        Sets the tab's agent chat_id to match the resumed session.
        The tab_id (frontend key in ``_tab_states``) does not change.

        When ``tab_id`` is empty the call is a no-op — the previous
        behavior of synthesizing a phantom tab keyed by ``chat_id`` and
        mutating its ``use_worktree`` flag violated per-tab state
        isolation (C2/C3 fix).

        Args:
            chat_id: The string chat session identifier to replay.
            tab_id: The frontend tab identifier.
        """
        if not tab_id:
            logger.debug("_replay_session called without tab_id; ignoring")
            return
        result = _load_latest_chat_events_by_chat_id(chat_id)
        if not result or not result.get("events"):
            return
        tab = self._get_tab(tab_id)
        tab.agent.resume_chat_by_id(chat_id)

        extra_str = str(result.get("extra", "") or "")
        if extra_str:
            try:
                extra = json.loads(extra_str)
                with self._state_lock:
                    tab.use_worktree = bool(extra.get("is_worktree"))
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

    def _get_adjacent_task(
        self, chat_id: str, task: str, direction: str, tab_id: str = "",
    ) -> None:
        """Send events for the adjacent task in the same chat session.

        Args:
            chat_id: The string chat session identifier.
            task: Current task description string (used as timestamp reference).
            direction: ``"prev"`` or ``"next"``.
            tab_id: Frontend tab identifier used to route the event.
        """
        result = _get_adjacent_task_by_chat_id(chat_id, task, direction)
        event: dict[str, Any] = {
            "type": "adjacent_task_events",
            "direction": direction,
            "task": result["task"] if result else "",
            "events": result["events"] if result else [],
            "tabId": tab_id,
        }
        self.printer.broadcast(event)

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
            msg = generate_commit_message_from_diff(diff_text)  # pragma: no cover
            self.printer.broadcast({"type": "commitMessage", "message": msg})  # pragma: no cover
        except Exception:  # pragma: no cover — LLM API error handler
            logger.debug("Commit message generation failed", exc_info=True)
            self.printer.broadcast({
                "type": "commitMessage",
                "message": "",
                "error": "Failed to generate",
            })


def main() -> None:  # pragma: no cover — CLI entry point
    """Main entry point for VS Code backend server."""
    server = VSCodeServer()
    server.run()


if __name__ == "__main__":
    main()
