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
    _chat_has_tasks,
    _delete_task,
    _get_adjacent_task_by_chat_id,
    _get_task_chat_id,
    _load_chat_events_by_task_id,
    _load_frequent_tasks,
    _load_history,
    _load_last_model,
    _load_latest_chat_events_by_chat_id,
    _load_model_usage,
    _search_history,
)
from kiss.agents.vscode.autocomplete import _AutocompleteMixin
from kiss.agents.vscode.commands import _CommandsMixin
from kiss.agents.vscode.diff_merge import (
    _cleanup_merge_data,
    _git,
    _merge_data_dir,
)
from kiss.agents.vscode.helpers import (
    generate_commit_message_from_diff,
    generate_followup_text,
    model_vendor,
)
from kiss.agents.vscode.merge_flow import _MergeFlowMixin
from kiss.agents.vscode.printer import VSCodePrinter
from kiss.agents.vscode.tab_state import _TabState, parse_task_tags
from kiss.agents.vscode.task_runner import _TaskRunnerMixin
from kiss.core.models.model_info import (
    MODEL_INFO,
    get_available_models,
    get_default_model,
    get_fast_model,
)

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

        from kiss.agents.vscode.vscode_config import get_custom_model_entry, load_config

        cfg = load_config()
        custom = get_custom_model_entry(cfg)
        if custom:
            models_list.insert(0, custom)

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
            result = str(entry.get("result", "") or "")
            session: dict[str, Any] = {
                "id": chat_id,
                "task_id": entry.get("id"),
                "title": task[:50] + "..." if len(task) > 50 else task,
                "timestamp": entry.get("timestamp", 0),
                "preview": task,
                "has_events": has_events,
                "failed": (
                    result.startswith("Task failed")
                    or result == "Agent Failed Abruptly"
                ),
            }
            # Surface sub-agent metadata so the history sidebar can
            # reopen the row as a sub-agent tab (⚡N indicator, no
            # input bar) instead of reclassifying it as a regular
            # chat tab.  Stored in ``extra`` under the ``subagent``
            # key by ``ChatSorcarAgent.run`` for sub-agent rows.
            extra_raw = str(entry.get("extra", "") or "")
            if extra_raw:
                try:
                    extra_obj = json.loads(extra_raw)
                except (json.JSONDecodeError, TypeError):
                    extra_obj = None
                if isinstance(extra_obj, dict):
                    sub = extra_obj.get("subagent")
                    if isinstance(sub, dict):
                        session["is_subagent"] = True
                        session["subagent_tab_id"] = str(
                            sub.get("tab_id", "") or ""
                        )
                        session["parent_tab_id"] = str(
                            sub.get("parent_tab_id", "") or ""
                        )
                        ti = sub.get("task_index")
                        session["task_index"] = (
                            int(ti) if isinstance(ti, int) else 0
                        )
                        session["description"] = str(
                            sub.get("description", task) or task
                        )
            sessions.append(session)
        self.printer.broadcast({
            "type": "history", "sessions": sessions,
            "offset": offset, "generation": generation,
        })

    def _handle_delete_task(self, task_id: int) -> None:
        """Delete a task and its associated events from the database
        and broadcast a ``taskDeleted`` event so any open chat tab
        that displays this task or its chat can prune its UI.

        The history sidebar is removed optimistically by the
        frontend on click, but open tabs that show the deleted task
        (as the current task or as an adjacent-task block from
        scroll-loaded chat history) need to react too.  We therefore
        look up the task's chat_id *before* deleting, and after
        deletion broadcast::

            {
                "type": "taskDeleted",
                "taskId": <int>,
                "chatId": <str>,
                "chatHasMoreTasks": <bool>,
            }

        ``chatHasMoreTasks`` lets the frontend decide whether to
        merely remove a single ``.adjacent-task`` block or close the
        whole tab (when the chat is now empty).

        Args:
            task_id: The primary key of the task_history row to
                delete.
        """
        chat_id = _get_task_chat_id(task_id)
        if not _delete_task(task_id):
            return
        self.printer.broadcast({
            "type": "taskDeleted",
            "taskId": task_id,
            "chatId": chat_id,
            "chatHasMoreTasks": _chat_has_tasks(chat_id),
        })

    def _get_frequent_tasks(self, limit: int = 50) -> None:
        """Send the top *limit* most-frequent tasks (highest count first).

        Broadcasts a ``frequentTasks`` event whose ``tasks`` field is a
        list of ``{task, count, timestamp}`` dicts ordered by ``count``
        descending.

        Args:
            limit: Maximum number of frequent tasks to return.
        """
        self.printer.broadcast({
            "type": "frequentTasks",
            "tasks": _load_frequent_tasks(limit=limit),
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
        reference.

        When the tab is currently running a task or in a merge review,
        the state is **not** removed immediately — the running agent
        must be allowed to finish (per ``USER_PREFS.md``: "Closing a
        chat tab does NOT stop a running agent task").  Instead the
        tab is marked ``frontend_closed = True`` so that
        :meth:`_dispose_if_closed` will tear it down later, once the
        last lifecycle flag drops to false.

        When the tab has a pending worktree (no active task / merge),
        the worktree is released (just like starting a new task
        would) before removing the tab, so the worktree branch and
        directory are not orphaned.

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
                tab.frontend_closed = True
                return
            self._tab_states.pop(tab_id, None)
        self._teardown_tab_resources(tab_id, tab)

    def _dispose_if_closed(self, tab_id: str) -> None:
        """Dispose *tab_id*'s state if the frontend already closed it.

        Invoked at every lifecycle transition that can flip the last
        lifecycle flag to false (task end, merge end).  Pops the state
        only when ``frontend_closed`` is set AND no lifecycle flag is
        still raised; otherwise leaves it alone.  Idempotent and safe
        to call when no state exists for *tab_id*.

        Args:
            tab_id: The frontend tab identifier.
        """
        if not tab_id:
            return
        with self._state_lock:
            tab = self._tab_states.get(tab_id)
            if tab is None or not tab.frontend_closed:
                return
            if (
                tab.is_task_active
                or tab.is_merging
                or (tab.task_thread is not None and tab.task_thread.is_alive())
            ):
                return
            self._tab_states.pop(tab_id, None)
        self._teardown_tab_resources(tab_id, tab)

    def _teardown_tab_resources(
        self, tab_id: str, tab: _TabState | None,
    ) -> None:
        """Release worktree, per-tab printer state and merge data dir.

        Shared cleanup tail used by both the immediate (:meth:`_close_tab`)
        and the deferred (:meth:`_dispose_if_closed`) disposal paths.
        Caller must have already popped *tab* from ``_tab_states``.

        Args:
            tab_id: The frontend tab identifier being disposed.
            tab: The popped tab state, or ``None`` when the tab was
                never created (e.g. ``closeTab`` for an unknown id).
        """
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

        Re-reads the last user-picked model from the database so the new
        tab uses the correct model even when the in-memory default has
        drifted (e.g. after switching between tabs with different models).

        Args:
            tab_id: The frontend tab identifier (a freshly-minted uuid).
        """
        persisted = _load_last_model()
        if persisted:
            self._default_model = persisted
        tab = self._get_tab(tab_id)
        with self._state_lock:
            tab.selected_model = self._default_model
        tab.agent.new_chat()
        self.printer.broadcast({
            "type": "showWelcome",
            "tabId": tab_id,
            "model": self._default_model,
        })

    def _replay_session(
        self, chat_id: str, tab_id: str = "", task_id: int | None = None,
    ) -> None:
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
            task_id: Optional task row ID.  When provided, load this
                specific task instead of the latest task in the chat
                session.  This is used when the user clicks a specific
                task in the history panel.
        """
        if not tab_id:
            logger.debug("_replay_session called without tab_id; ignoring")
            return
        result = None
        if task_id is not None:
            result = _load_chat_events_by_task_id(task_id)
            # If found, ensure chat_id is populated for resume_chat_by_id
            if result:
                chat_id = str(result.get("chat_id", "") or chat_id)
        if not result:
            result = _load_latest_chat_events_by_chat_id(chat_id)
        if not result or not result.get("events"):
            return

        # Inspect ``extra`` BEFORE re-attaching so we can detect a
        # sub-agent row and skip ``_reattach_running_chat`` for it.
        # Sub-agents share their parent's chat_id but run as threads
        # inside the parent's executor — the parent's ``_TabState``
        # owns the live thread, so rebinding it here would steal the
        # parent's tab.  We only need to rebind the printer's per-tab
        # alias so live events still tagged with the sub-agent's
        # original ``sub_tab_id`` reach the newly opened tab.
        extra_str = str(result.get("extra", "") or "")
        subagent_info: dict[str, object] | None = None
        is_worktree = False
        if extra_str:
            try:
                extra = json.loads(extra_str)
                if isinstance(extra, dict):
                    is_worktree = bool(extra.get("is_worktree"))
                    sub = extra.get("subagent")
                    if isinstance(sub, dict):
                        subagent_info = sub
            except (json.JSONDecodeError, TypeError):
                pass

        rebound_running = False
        if subagent_info is None:
            # Subscribe the new tab to a still-running agent's event
            # stream (under some other tab id) so its live events ALSO
            # flow here — without stealing the stream from the original
            # client.  See :meth:`_reattach_running_chat`.
            rebound_running = self._reattach_running_chat(chat_id, tab_id)

        tab = self._get_tab(tab_id)
        # Always attach the new tab's own agent to the resumed chat
        # so a follow-up prompt typed in this tab continues the same
        # chat session.  With multi-viewer subscribe semantics the
        # source tab (running the live task) keeps its own _TabState,
        # so the new tab needs its own agent linked to the chat too.
        tab.agent.resume_chat_by_id(chat_id)
        with self._state_lock:
            tab.use_worktree = is_worktree

        if subagent_info is not None:
            # Convert the freshly created regular tab into a sub-agent
            # tab on the frontend.  The ``openSubagentTab`` handler is
            # idempotent on ``tab_id`` (see main.js case
            # ``openSubagentTab``) and will simply flip the existing
            # tab's ``isSubagentTab`` / title in place.
            ti = subagent_info.get("task_index")
            self.printer.broadcast({
                "type": "openSubagentTab",
                "tab_id": tab_id,
                "parent_tab_id": subagent_info.get("parent_tab_id", "") or "",
                "description": subagent_info.get(
                    "description", result.get("task", "")
                ),
                "taskIndex": int(ti) if isinstance(ti, int) else 0,
                "isSubagentTab": True,
            })
            # Route any future events tagged with the sub-agent's
            # original sub_tab_id (the parent's running executor may
            # still be emitting them via thread-local) to the new tab.
            orig_sub_tab_id = str(subagent_info.get("tab_id", "") or "")
            if orig_sub_tab_id and orig_sub_tab_id != tab_id:
                self.printer.rebind_tab(orig_sub_tab_id, tab_id)

        self.printer.broadcast({
            "type": "task_events",
            "events": result["events"],
            "task": result["task"],
            "task_id": result.get("task_id"),
            "chat_id": chat_id,
            "extra": result.get("extra", ""),
            "tabId": tab_id,
        })
        if rebound_running:
            # Make the resumed tab show the running spinner since the
            # agent is still working.  Mirrors the broadcast that
            # ``_run_task`` emits when a task starts.
            self.printer.broadcast({
                "type": "status",
                "running": True,
                "tabId": tab_id,
            })
        self._emit_pending_worktree(tab_id)

    def _reattach_running_chat(self, chat_id: str, new_tab_id: str) -> bool:
        """Subscribe *new_tab_id* to a still-running ``_TabState`` for
        *chat_id* so its live agent's events ALSO flow to the newly
        opened tab — without stealing the stream from the original
        client.

        Looks for an entry in ``_tab_states`` whose agent has the same
        ``chat_id`` and whose ``task_thread`` is still alive (or whose
        ``is_task_active`` flag is set).  Multi-viewer fan-out is
        implemented in the printer: the original ``_TabState`` keeps
        owning the running task and the agent thread keeps tagging
        events with the original tab id, while
        :meth:`BaseBrowserPrinter.subscribe_tab` registers
        *new_tab_id* as an additional viewer so every broadcast is
        duplicated with ``tabId=new_tab_id``.  This means BOTH the
        original client (if still connected) AND the freshly-opened
        client see the streaming events.

        Args:
            chat_id: The chat id of the task the user clicked in
                history.
            new_tab_id: The freshly allocated frontend tab id.

        Returns:
            ``True`` when a live agent for *chat_id* exists and
            *new_tab_id* is now subscribed to its event stream;
            ``False`` when no live agent exists.
        """
        if not chat_id or not new_tab_id:
            return False
        with self._state_lock:
            source_tab_id: str | None = None
            for key, t in self._tab_states.items():
                if key == new_tab_id:
                    # Don't subscribe a tab to its own state.
                    continue
                agent_chat_id = getattr(t.agent, "chat_id", "") or ""
                if agent_chat_id != chat_id:
                    continue
                alive = (
                    t.task_thread is not None and t.task_thread.is_alive()
                )
                if alive or t.is_task_active:
                    source_tab_id = key
                    break
            if source_tab_id is None:
                return False
        self.printer.subscribe_tab(source_tab_id, new_tab_id)
        return True


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
                    task, result, get_fast_model()
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
            "task_id": result["task_id"] if result else None,
            "events": result["events"] if result else [],
            "tabId": tab_id,
        }
        self.printer.broadcast(event)

    def _generate_commit_message(self) -> None:
        """Generate a git commit message from current changes."""
        try:
            from pathlib import Path

            from kiss.agents.sorcar.git_worktree import GitWorktreeOps

            if GitWorktreeOps.discover_repo(Path(self.work_dir)) is None:
                self.printer.broadcast({
                    "type": "commitMessage",
                    "message": "",
                    "error": "Not a git repository.",
                })
                return
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
