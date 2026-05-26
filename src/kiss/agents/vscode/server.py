"""VS Code extension backend server for Sorcar agent.

The per-command handlers, task-runner, merge / worktree flow and
autocomplete logic live in sibling mixin modules.  This file keeps the
per-tab state accessors, the command dispatcher, and the history /
chat / commit-message helpers.

``VSCodeServer`` is consumed by :class:`RemoteAccessServer`
(:mod:`kiss.agents.vscode.web_server`), which owns the actual I/O
transports (Unix-domain socket for the local VS Code extension and
WebSocket for remote browser clients) and instantiates a
:class:`WebPrinter` whose ``broadcast`` method fans events out to
every connected client.  No stdin/stdout transport remains: the old
per-tab subprocess model has been fully replaced by the single
``kiss-web`` daemon.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from typing import Any

from kiss.agents.sorcar.persistence import (
    _append_chat_event,
    _chat_has_tasks,
    _delete_frequent_task,
    _delete_task,
    _get_adjacent_task_by_chat_id,
    _get_task_chat_id,
    _is_failed_result,
    _load_chat_events_by_task_id,
    _load_frequent_tasks,
    _load_history,
    _load_last_model,
    _load_latest_chat_events_by_chat_id,
    _load_model_usage,
    _load_subagent_rows_by_parent_task_id,
    _recover_orphaned_tasks,
    _search_history,
    _set_task_favorite,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState, parse_task_tags
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.autocomplete import _AutocompleteMixin
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
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
from kiss.agents.vscode.task_runner import _TaskRunnerMixin
from kiss.core.models.model_info import (
    MODEL_INFO,
    get_available_models,
    get_default_model,
    get_fast_model,
)

__all__ = [
    "VSCodeServer",
    "_RunningAgentState",
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

    def __init__(self, printer: BaseBrowserPrinter | None = None) -> None:
        # The transport-specific printer is owned by the caller
        # (typically :class:`RemoteAccessServer`, which passes a
        # :class:`WebPrinter`).  Defaulting to a plain
        # :class:`BaseBrowserPrinter` keeps the unit tests that
        # construct a bare ``VSCodeServer()`` working — they patch
        # ``server.printer.broadcast`` themselves to capture events.
        self.printer: BaseBrowserPrinter = printer or BaseBrowserPrinter()
        # ``running_agent_states`` is now a class attribute on
        # :class:`_RunningAgentState` (shared across every instance).
        # Reset on init so each ``VSCodeServer`` starts with a clean
        # slate (production only ever spawns one server per Python
        # process; test fixtures that instantiate the server fresh
        # need isolation from prior tests' tab state).  The server
        # still owns ``_state_lock`` to coordinate multi-step access
        # (read-then-modify, scan-then-modify) and the lifecycle
        # transitions that span both the dict and individual
        # ``_RunningAgentState`` fields.
        _RunningAgentState.running_agent_states.clear()
        # Sweep up "Agent Failed Abruptly" rows left behind by a prior
        # process that was killed mid-task (SIGKILL / VS Code reload /
        # OOM) — the Python ``finally`` in ``_TaskRunnerMixin`` cannot
        # run when the host process dies, so the only way to clear
        # the sentinel from those rows is here, on every fresh
        # server boot.  At this point ``running_agent_states`` has
        # just been cleared so no task in THIS process is active, and
        # we pass an empty active set.
        try:
            _recover_orphaned_tasks(set())
        except Exception:  # pragma: no cover — best-effort sweep
            logger.exception(
                "orphan-task recovery sweep failed; continuing startup",
            )
        self.work_dir = os.environ.get("KISS_WORKDIR", os.getcwd())
        persisted = _load_last_model()
        self._default_model = (
            persisted
            or os.environ.get("KISS_MODEL", "")
            or get_default_model()
        )
        # Share the lock that guards ``_RunningAgentState.running_agent_states``
        # so producers outside this module (parallel sub-agent spawners
        # in :class:`ChatSorcarAgent`, registration helpers in
        # :class:`WorktreeSorcarAgent`) can serialise their mutations
        # against the server's iteration loops without re-importing the
        # server class.  Using the registry's ``RLock`` (re-entrant)
        # also fixes a latent self-deadlock where an existing
        # ``with self._state_lock:`` critical section would call into
        # a helper that itself tried to re-acquire the same lock.
        self._state_lock = _RunningAgentState._registry_lock
        self._complete_seq: int = 0
        self._complete_seq_latest: int = -1
        self._complete_queue: queue.Queue[tuple[str, int, str, str, str]] | None = None
        self._complete_worker: threading.Thread | None = None
        self._file_cache: list[str] | None = None
        self._last_active_file: str = ""
        self._last_active_content: str = ""

    @property
    def _running_agent_states(self) -> dict[str, _RunningAgentState]:
        """Process-global running-agent-state dict.

        Backward-compat accessor for existing test fixtures and audit
        tests that read ``server._running_agent_states``.  The
        canonical home is the class attribute
        :attr:`kiss.agents.sorcar.running_agent_state._RunningAgentState.running_agent_states`;
        production code inside this package accesses it directly via
        :class:`_RunningAgentState`.
        """
        return _RunningAgentState.running_agent_states

    def _get_tab(self, tab_id: str) -> _RunningAgentState:
        """Get or create per-tab state for the given tab.

        Each tab gets its own agent instances so concurrent tabs never
        share mutable agent state (chat_id, task_id, worktree, etc.).
        ``tab_id`` is purely a frontend routing key (whatever uuid the
        frontend allocated for this tab); ``chat_id`` is purely the
        persistence key stored on :class:`_RunningAgentState` once a
        run starts.

        Thread-safe: acquires ``_state_lock`` to protect the
        get-or-create pattern against concurrent callers.

        Args:
            tab_id: The frontend tab identifier string.

        Returns:
            The per-tab state object.
        """
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is None:
                tab = _RunningAgentState(tab_id, self._default_model)
                _RunningAgentState.running_agent_states[tab_id] = tab
            # Lazily ensure an agent slot for tests / out-of-task
            # callers that read ``tab.agent`` directly (e.g. merge
            # / discard, worktree state inspection).  The agent is
            # transient — :meth:`_TaskRunnerMixin._run_task`'s outer
            # ``finally`` sets ``tab.agent = None`` once each task
            # completes, and :meth:`_CommandsMixin._cmd_run` allocates
            # a fresh agent before the worker thread starts.  So no
            # agent state ever survives a task boundary; the slot
            # populated here is a fresh, empty agent.
            if tab.agent is None:
                agent = WorktreeSorcarAgent("Sorcar VS Code")
                if tab.chat_id:
                    agent._chat_id = tab.chat_id
                tab.agent = agent
            return tab

    def _any_non_wt_running(self) -> bool:
        """True if any tab is running a non-worktree task on the main tree.

        Must be called with ``_state_lock`` held.

        Returns:
            True if at least one tab has ``is_running_non_wt`` set.
        """
        return any(t.is_running_non_wt for t in _RunningAgentState.running_agent_states.values())

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


    def broadcast_new_tab(self, task_id: int) -> None:
        """Ask the frontend to open a fresh chat tab and resume ``task_id``.

        The webview's ``new_tab`` handler allocates a new tab id
        (frontend-only concept) via ``createNewTab`` and then posts a
        ``resumeSession`` command back with the same ``task_id`` —
        ``_cmd_resume_session`` accepts a task-id-only resume payload.

        ``taskId=""`` keeps this a global system event so it reaches
        every connected client; otherwise ``WebPrinter.broadcast``
        would fan it out only to subscribers of the freshly-minted
        task (of which there are none until the frontend has
        received this broadcast and allocated the tab).
        """
        self.printer.broadcast(
            {"type": "new_tab", "task_id": int(task_id), "taskId": ""},
        )

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

    def _get_running_task_ids(self) -> set[int]:
        """Return the set of task_history row ids with alive worker threads.

        Scans all per-tab ``_RunningAgentState`` entries and collects the
        live task id of those whose ``task_thread`` is still alive.
        Prefers ``tab.agent._last_task_id`` (set by the agent the moment
        it allocates the task row) and falls back to
        ``tab.task_history_id`` for the post-run window when the agent
        reference has already been cleared.  Must be called WITHOUT
        holding ``_state_lock`` — acquires it internally.

        Returns:
            Set of ``task_history.id`` values that are currently running.
        """
        running: set[int] = set()
        with self._state_lock:
            for tab in _RunningAgentState.running_agent_states.values():
                tid = (
                    tab.agent._last_task_id
                    if tab.agent is not None and tab.agent._last_task_id is not None
                    else tab.task_history_id
                )
                if (
                    tid is not None
                    and tab.task_thread is not None
                    and tab.task_thread.is_alive()
                ):
                    running.add(tid)
        return running

    def _overlay_live_metrics(
        self, session: dict[str, Any], task_id: int,
    ) -> None:
        """Replace persisted metrics with live agent data for a running task.

        Scans ``_RunningAgentState.running_agent_states`` for a tab whose
        live task id matches *task_id* and overwrites the ``tokens``,
        ``cost``, and ``steps`` fields in *session* with current values
        from the running agent, including the in-progress executor's
        ``step_count``.  Must be called WITHOUT holding ``_state_lock``.

        Args:
            session: The history session dict to update in place.
            task_id: The ``task_history.id`` of the running task.
        """
        with self._state_lock:
            for tab in _RunningAgentState.running_agent_states.values():
                agent = tab.agent
                if agent is None:
                    continue
                live_tid = (
                    agent._last_task_id
                    if agent._last_task_id is not None
                    else tab.task_history_id
                )
                if live_tid != task_id:
                    continue
                session["tokens"] = int(
                    getattr(agent, "total_tokens_used", 0) or 0
                )
                session["cost"] = float(
                    getattr(agent, "budget_used", 0.0) or 0.0
                )
                steps = int(getattr(agent, "total_steps", 0) or 0)
                cur = getattr(agent, "_current_executor", None)
                if cur is not None:
                    steps += int(getattr(cur, "step_count", 0) or 0)
                session["steps"] = steps
                break

    def _get_history(self, query: str | None, offset: int = 0, generation: int = 0) -> None:
        """Send conversation history with pagination support."""
        if query:
            entries = _search_history(query, limit=50, offset=offset)
        else:
            entries = _load_history(limit=50, offset=offset)

        running_task_ids = self._get_running_task_ids()

        sessions = []
        for entry in entries:
            task = str(entry.get("task", ""))
            has_events = bool(entry.get("has_events", False))
            chat_id = str(entry.get("chat_id", "") or "")
            result = str(entry.get("result", "") or "")
            entry_id = entry.get("id")
            session: dict[str, Any] = {
                "id": chat_id,
                "task_id": entry_id,
                "title": task,
                "timestamp": entry.get("timestamp", 0),
                "preview": task,
                "has_events": has_events,
                "failed": _is_failed_result(result),
                "is_running": (
                    isinstance(entry_id, int)
                    and entry_id in running_task_ids
                ),
                "tokens": 0,
                "cost": 0.0,
                "steps": 0,
                "is_favorite": False,
                # Agent start / end timestamps in ms since epoch.
                # ``startTs`` comes from the row's own ``timestamp``
                # column (set on INSERT in ``_add_task``), ``endTs``
                # is read from the persisted ``extra`` JSON below
                # (0 means the agent has not yet recorded an end —
                # i.e. still running, or the task pre-dates the
                # endTs persistence change).  Surfaced so the
                # frontend can render the chat webview's
                # "Running …" / "Done (Xm Ys)" header from
                # agent wall-clock rather than the local client
                # wall-clock at history-load time.
                "startTs": int(
                    float(entry.get("timestamp", 0) or 0)  # type: ignore[arg-type]
                    * 1000
                ),
                "endTs": 0,
            }
            # Mark sub-agent rows so the history sidebar treats them
            # as a regular task with one rendering difference: the
            # reopened tab is styled as a sub-agent tab (purple
            # accent) and suppresses adjacent-task loading
            # for siblings in the same chat.  Persisted as just
            # ``{"parent_task_id": <int>}`` under ``extra.subagent``;
            # presence of the key implies the row is a sub-agent.
            #
            # ``extra`` also holds the post-completion metrics
            # (``tokens``, ``cost``, ``steps``) written by
            # ``_TaskRunnerMixin._run_task_inner`` so the history
            # sidebar can render each row with the same metrics
            # line as the Running tab.
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
                        pid = sub.get("parent_task_id")
                        if isinstance(pid, int):
                            session["parent_task_id"] = pid
                    try:
                        session["tokens"] = int(extra_obj.get("tokens", 0) or 0)
                    except (TypeError, ValueError):
                        session["tokens"] = 0
                    try:
                        session["cost"] = float(extra_obj.get("cost", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        session["cost"] = 0.0
                    try:
                        session["steps"] = int(extra_obj.get("steps", 0) or 0)
                    except (TypeError, ValueError):
                        session["steps"] = 0
                    session["is_favorite"] = bool(
                        extra_obj.get("is_favorite", False)
                    )
                    try:
                        end_ts_raw = extra_obj.get("endTs", 0)
                        session["endTs"] = int(end_ts_raw or 0)
                    except (TypeError, ValueError):
                        session["endTs"] = 0
                    try:
                        start_ts_raw = extra_obj.get("startTs", 0)
                        if start_ts_raw:
                            session["startTs"] = int(start_ts_raw)
                    except (TypeError, ValueError):
                        pass
            # For running tasks, overlay live metrics from the agent
            # so the history panel shows current steps/tokens/cost
            # instead of the stale persisted values (which are only
            # written at task completion).
            if session.get("is_running") and isinstance(entry_id, int):
                self._overlay_live_metrics(session, entry_id)
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

    def _handle_set_favorite(self, task_id: int, is_favorite: bool) -> None:
        """Persist the favourite flag on a task history row.

        Merges ``{"is_favorite": <bool>}`` into the row's ``extra``
        JSON column, preserving other keys (tokens, cost, steps,
        subagent metadata).  No broadcast is emitted: the originating
        webview updates its star icon optimistically on click, and
        the next ``getHistory`` refresh will reflect the persisted
        flag for all other clients.

        Args:
            task_id: Primary key of the ``task_history`` row.
            is_favorite: New value for the ``is_favorite`` flag.
        """
        _set_task_favorite(task_id, is_favorite)

    def _handle_delete_frequent_task(self, task: str) -> None:
        """Delete a row from the ``frequent_tasks`` table and rebroadcast.

        After deletion succeeds, re-emits the current frequent tasks
        list so any other open webview rerenders without the deleted
        row.  The originating webview removes the row optimistically.

        Args:
            task: The exact task description string identifying the row.
        """
        if not _delete_frequent_task(task):
            return
        self._get_frequent_tasks()

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

        Removes the tab from ``_running_agent_states``, cleans up per-tab printer
        state (bash buffers, recordings), and drops the persist-agent
        reference.

        When the tab is currently running a task or in a merge review,
        the state is **not** removed immediately — the running agent
        must be allowed to finish: closing a chat tab does NOT stop a
        running agent task.  Instead the
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
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is not None and (
                tab.is_task_active
                or tab.is_merging
                or (tab.task_thread is not None and tab.task_thread.is_alive())
            ):
                tab.frontend_closed = True
                return
            _RunningAgentState.running_agent_states.pop(tab_id, None)
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
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is None or not tab.frontend_closed:
                return
            if (
                tab.is_task_active
                or tab.is_merging
                or (tab.task_thread is not None and tab.task_thread.is_alive())
            ):
                return
            _RunningAgentState.running_agent_states.pop(tab_id, None)
        self._teardown_tab_resources(tab_id, tab)

    def _teardown_tab_resources(
        self, tab_id: str, tab: _RunningAgentState | None,
    ) -> None:
        """Release worktree, per-tab printer state and merge data dir.

        Shared cleanup tail used by both the immediate (:meth:`_close_tab`)
        and the deferred (:meth:`_dispose_if_closed`) disposal paths.
        Caller must have already popped *tab* from ``_running_agent_states``.

        Args:
            tab_id: The frontend tab identifier being disposed.
            tab: The popped tab state, or ``None`` when the tab was
                never created (e.g. ``closeTab`` for an unknown id).
        """
        if tab is not None:
            # Release any still-pending worktree branch held by the
            # tab's transient agent.  When the agent has already been
            # disposed (the common case at tab close after the task
            # ended), there is nothing to release: each task creates
            # a fresh worktree and the agent is the sole authority on
            # its state.
            try:
                wt_agent = self._ensure_wt_agent(tab)
                if wt_agent is not None and wt_agent._wt_pending:
                    wt_agent._release_worktree()
            except Exception:
                logger.debug("Worktree release on tab close failed", exc_info=True)
        # ``cleanup_tab`` removes *tab_id* from every task subscriber
        # set so a stale viewer never receives events.  Per-task state
        # (recordings, persist_agents, offsets) is owned by the agent
        # thread and cleaned up by :meth:`_TaskRunnerMixin` /
        # :meth:`ChatSorcarAgent.run`.
        self.printer.cleanup_tab(tab_id)
        _cleanup_merge_data(str(_merge_data_dir(tab_id)))

    def _new_chat(self, tab_id: str) -> None:
        """Start a new chat session for the given tab.

        The ``newChat`` command is only issued by the frontend's
        ``createNewTab`` flow, which always allocates a fresh tab id
        that the backend has never seen before.  ``_get_tab`` creates a
        clean ``_RunningAgentState``, so there is no prior run state (no active
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
            # Clear the long-lived chat identity for this tab; the
            # next ``_cmd_run`` will mint a fresh chat id when it
            # builds the per-task agent.  No agent is owned by the
            # tab outside of an active task, so there is nothing
            # else to reset here.
            tab.chat_id = ""
            tab.last_task_id = None
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
        The tab_id (frontend key in ``_running_agent_states``) does not change.

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
        if not result:
            return
        # NOTE: do NOT early-return on empty ``events``.  When a
        # sub-agent has just been spawned by ``run_parallel`` and
        # broadcasts ``new_tab`` (carrying its own ``task_id``), the
        # frontend round-trips a ``resumeSession`` back here long
        # before the async event-writer thread has flushed any
        # events to the DB.  Returning early on an empty events list
        # would skip ``_reattach_running_chat`` below, so the new
        # tab would never get subscribed to the sub-agent's live
        # event stream and subsequent broadcasts would have no
        # fan-out target.  Proceeding with an empty events list is
        # harmless: ``task_events`` with ``events=[]`` is a no-op
        # for the frontend's replay loop.

        # Inspect ``extra`` BEFORE re-attaching so we know whether to
        # flip the freshly-allocated tab into sub-agent styling.  The
        # sub-agent's own :class:`_RunningAgentState` is registered by
        # :meth:`ChatSorcarAgent._run_tasks_parallel` under the
        # sub-agent's ``sub_tab_id`` and carries
        # ``task_history_id`` mirrored from its own ``task_history``
        # row, so :meth:`_reattach_running_chat` can disambiguate it
        # from the parent (which shares ``chat_id``) by matching on
        # the row's task id.
        extra_str = str(result.get("extra", "") or "")
        subagent_info: dict[str, object] | None = None
        extra_raw: object = None
        if extra_str:
            try:
                extra_raw = json.loads(extra_str)
                if isinstance(extra_raw, dict):
                    sub = extra_raw.get("subagent")
                    if isinstance(sub, dict):
                        subagent_info = sub
            except (json.JSONDecodeError, TypeError):
                pass

        # Subscribe the new tab to a still-running agent's event
        # stream (under some other tab id) so its live events ALSO
        # flow here — without stealing the stream from the original
        # client.  See :meth:`_reattach_running_chat`.  The ``task_id``
        # kwarg disambiguates by ``task_history_id`` so the parent
        # (whose ``_RunningAgentState`` shares the sub-agent's
        # ``chat_id``) is never matched when the user clicks a
        # sub-agent row.  When ``task_id`` is ``None`` (no specific
        # row, just a chat) the regular chat-id-based scan applies.
        rebound_task_id = result.get("task_id") if result else None
        if not isinstance(rebound_task_id, int):
            rebound_task_id = None
        rebound_running = self._reattach_running_chat(
            chat_id,
            tab_id,
            task_id=rebound_task_id,
            is_subagent=subagent_info is not None,
        )

        # Loading a (completed or running-elsewhere) task into this
        # tab is a VIEW operation — no agent will run here, so we
        # MUST NOT eagerly create a fresh ``_RunningAgentState``
        # registry entry or a ``WorktreeSorcarAgent``.  The new tab
        # only needs the persisted events rendered in its webview
        # (and, when ``rebound_running`` is true, a printer
        # subscription to the live source tab — already established
        # above by ``_reattach_running_chat`` via ``subscribe_tab``).
        #
        # If a ``_RunningAgentState`` already exists for ``tab_id``
        # (e.g. the user previously launched a task in this tab and
        # is now viewing a different chat in the same tab), update it
        # in place but do NOT create one if absent.  In particular we
        # still want to: (a) associate the resumed ``chat_id`` so a
        # follow-up ``_cmd_run`` continues the same chat, and (b)
        # clear ``frontend_closed`` so a pending deferred-dispose
        # does not tear down the tab the user is actively viewing.
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is not None:
                tab.chat_id = chat_id
                tab.use_worktree = bool(
                    extra_raw.get("is_worktree")
                    if isinstance(extra_raw, dict) else False,
                )
                tab.frontend_closed = False

        if subagent_info is not None:
            # Convert the freshly created regular tab into a sub-agent
            # tab on the frontend.  The ``openSubagentTab`` handler is
            # idempotent on ``tab_id`` (see main.js case
            # ``openSubagentTab``) and will flip the existing tab's
            # ``isSubagentTab`` / title in place.  Sub-agent rows
            # persist only the parent ``task_history.id`` — the
            # description shown in the tab header is derived from the
            # row's own ``task`` column.
            #
            # ``isDone`` is decided from the task-id-keyed
            # ``ChatSorcarAgent.running_agents`` map: presence under
            # the sub-agent's own task id means its thread is still
            # running, absence means it finished.  This lets the
            # reopened tab render the same "done, no indicator" state
            # the original tab ended on (instead of pulsing ◉ purple
            # forever because no later ``subagentDone`` arrives).
            from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

            sub_task_id = result.get("task_id")
            is_done = not (
                isinstance(sub_task_id, int)
                and sub_task_id in ChatSorcarAgent.running_agents
            )
            # Look up the parent's frontend tab id so the frontend can
            # record the parent → child relationship.  Without this,
            # closing the parent tab would not cascade-close this
            # sub-agent tab (see ``closeTab`` in media/main.js, which
            # walks ``parentTabId`` chains).
            parent_tid_raw = subagent_info.get("parent_task_id")
            parent_tid: int | None = (
                parent_tid_raw if isinstance(parent_tid_raw, int) else None
            )
            parent_tab_id_for_sub = self._resolve_parent_tab_id_for_sub(
                parent_task_id=parent_tid,
                chat_id=chat_id,
                sub_tab_id=tab_id,
            )
            self.printer.broadcast({
                "type": "openSubagentTab",
                "tab_id": tab_id,
                "parent_tab_id": parent_tab_id_for_sub,
                "description": str(result.get("task", "") or ""),
                "isSubagentTab": True,
                "isDone": is_done,
            })

        if rebound_running:
            # Make the resumed tab show the running spinner since the
            # agent is still working.  Mirrors the broadcast that
            # ``_run_task`` emits when a task starts.  This is emitted
            # BEFORE ``task_events`` so the frontend's ``isRunning``
            # flag is true while ``replayTaskEvents`` runs — matching
            # the fresh-run ordering (``status:true`` → events) so a
            # resumed-running tab and a fresh-run tab initialise the
            # webview to the same state (chevron visibility, send/stop
            # buttons, timer, etc.).
            #
            # Echo the agent's persisted ``startTs`` (ms since epoch,
            # written into the ``extra`` JSON by ``_run_task_inner``)
            # so the frontend's "Running …" header at the top of the
            # chat webview reflects the agent's true elapsed time
            # rather than the client's ``Date.now()`` at history-load.
            start_ts_for_resume = 0
            if isinstance(extra_raw, dict):
                try:
                    start_ts_for_resume = int(extra_raw.get("startTs", 0) or 0)
                except (TypeError, ValueError):
                    start_ts_for_resume = 0
            self.printer.broadcast({
                "type": "status",
                "running": True,
                "tabId": tab_id,
                "startTs": start_ts_for_resume,
            })
        self.printer.broadcast({
            "type": "task_events",
            "events": result["events"],
            "task": result["task"],
            "task_id": result.get("task_id"),
            "chat_id": chat_id,
            "extra": result.get("extra", ""),
            "tabId": tab_id,
        })
        self._emit_pending_worktree(tab_id)

        # When the user loads a PARENT task from the history sidebar,
        # also reopen every sub-agent the parent fanned out via
        # ``run_parallel`` so the loaded view mirrors the live
        # execution layout (one purple sub-agent tab per fan-out
        # task).  Sub-agent rows are skipped here — clicking a
        # sub-agent row already converts the clicked tab into a
        # sub-agent tab via the ``subagent_info`` branch above; we
        # do NOT want to recursively reopen siblings on that path.
        if subagent_info is None and isinstance(rebound_task_id, int):
            self._open_persisted_subagent_tabs(
                parent_task_id=rebound_task_id, parent_tab_id=tab_id,
            )

    def _resolve_parent_tab_id_for_sub(
        self,
        *,
        parent_task_id: int | None,
        chat_id: str,
        sub_tab_id: str,
    ) -> str:
        """Return the frontend tab id of the parent agent owning the
        sub-agent currently being opened on *sub_tab_id*.

        Used to populate ``parent_tab_id`` on the ``openSubagentTab``
        broadcast so the webview can record the parent → child
        relationship that drives cascade-close (see ``closeTab`` in
        media/main.js, which walks ``parentTabId`` chains).  A blank
        return value breaks that cascade, so this helper tries every
        signal we have before giving up.

        Lookup order (each tier skips sub-agent states):

        1. **Task-id match.**  Scan
           :attr:`_RunningAgentState.running_agent_states` for a
           non-subagent state whose ``agent._last_task_id`` (set at
           run-start, before the per-subtask ``finally`` writes
           ``task_history_id``) or ``task_history_id`` equals
           *parent_task_id*.  This is the primary, unambiguous match.

        2. **Chat-id match.**  Sub-agents inherit ``chat_id`` from
           the parent (see ``ChatSorcarAgent._run_tasks_parallel``).
           Scan for non-subagent states whose ``chat_id`` matches.
           If exactly one such state exists, use it.  More than one
           is ambiguous — bail out so we don't pick the wrong tab.

        3. **Synthetic-tab-id parse.**  Live sub-agent tab ids are
           generated as ``f"task-{parent_task_id}__sub_{idx}"`` by
           :meth:`ChatSorcarAgent._run_tasks_parallel` and as
           ``f"{parent_tab_id}__sub_{sub_task_id}"`` by
           :meth:`_open_persisted_subagent_tabs`.  Split on
           ``"__sub_"`` and, if the prefix matches a known
           non-subagent ``tab_id``, use it.

        If every tier fails, log a WARNING (silent ``""`` would
        manifest as the cascade-close bug from a downstream
        feature) and return ``""``.
        """
        with self._state_lock:
            non_sub_states = [
                st for st in _RunningAgentState.running_agent_states.values()
                if not st.is_subagent
            ]

            if parent_task_id is not None:
                for st in non_sub_states:
                    st_tid = (
                        st.agent._last_task_id
                        if st.agent is not None
                        and st.agent._last_task_id is not None
                        else st.task_history_id
                    )
                    if st_tid == parent_task_id:
                        return st.tab_id

            if chat_id:
                # Exclude ``sub_tab_id`` itself: ``_replay_session``
                # calls ``_get_tab(sub_tab_id)`` before this resolver
                # runs, which pre-registers a non-subagent state for
                # the freshly opened sub-tab and copies the resumed
                # session's ``chat_id`` onto it.  Without the guard
                # below, that state would match here and we'd return
                # the sub-tab's own id, creating a self-referential
                # parent_tab_id and a self-loop in the frontend's
                # parent→child cascade-close registry.
                chat_matches = [
                    st for st in non_sub_states
                    if st.chat_id == chat_id and st.tab_id != sub_tab_id
                ]
                if len(chat_matches) == 1:
                    return chat_matches[0].tab_id

            if "__sub_" in sub_tab_id:
                prefix = sub_tab_id.rsplit("__sub_", 1)[0]
                for st in non_sub_states:
                    if st.tab_id == prefix:
                        return st.tab_id

        logger.warning(
            "Could not resolve parent tab id for sub-agent "
            "(sub_tab_id=%r, parent_task_id=%r, chat_id=%r); "
            "cascade-close from parent will not reach this sub-tab.",
            sub_tab_id, parent_task_id, chat_id,
        )
        return ""

    def _open_persisted_subagent_tabs(
        self, *, parent_task_id: int, parent_tab_id: str,
    ) -> None:
        """Broadcast ``openSubagentTab`` + ``task_events`` for every
        persisted sub-agent row whose parent is *parent_task_id*.

        The sub-tab ids are deterministic
        (``f"{parent_tab_id}__sub_{sub_task_id}"``) so that clicking
        the same parent task twice in a row updates the existing
        sub-agent tabs in place instead of stacking duplicates — the
        webview's ``openSubagentTab`` handler is idempotent on
        ``tab_id``.

        ``isDone`` is decided from
        :attr:`ChatSorcarAgent.running_agents`: presence under the
        sub-agent's own task id means its thread is still running so
        the tab should pulse the ◉ indicator; absence means the
        sub-agent has completed and the tab should render as a
        finished tab without the indicator.

        Args:
            parent_task_id: ``task_history.id`` of the parent task.
            parent_tab_id: Frontend tab id of the parent tab.  Used
                as the prefix for the deterministic sub-tab ids.
        """
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        sub_rows = _load_subagent_rows_by_parent_task_id(parent_task_id)
        for idx, row in enumerate(sub_rows):
            sub_task_id = row["task_id"]
            sub_tab_id = f"{parent_tab_id}__sub_{sub_task_id}"
            description = str(row.get("task", "") or "")
            is_done = not (
                isinstance(sub_task_id, int)
                and sub_task_id in ChatSorcarAgent.running_agents
            )
            self.printer.broadcast({
                "type": "openSubagentTab",
                "tab_id": sub_tab_id,
                "parent_tab_id": parent_tab_id,
                "description": description,
                "taskIndex": idx,
                "isSubagentTab": True,
                "isDone": is_done,
            })
            self.printer.broadcast({
                "type": "task_events",
                "events": row["events"],
                "task": description,
                "task_id": sub_task_id,
                "chat_id": row.get("chat_id", ""),
                "extra": row.get("extra", ""),
                "tabId": sub_tab_id,
            })

    def _reattach_running_chat(
        self,
        chat_id: str,
        new_tab_id: str,
        *,
        task_id: int | None = None,
        is_subagent: bool = False,
    ) -> bool:
        """Subscribe *new_tab_id* to a still-running ``_RunningAgentState``
        so its live agent's events ALSO flow to the newly opened tab —
        without stealing the stream from the original client.

        ``tab_id`` (frontend routing key) and ``chat_id`` (persistence
        key) are orthogonal: the source ``_RunningAgentState`` is
        keyed by its own tab id (whatever the frontend allocated when
        the task was launched), and the chat id is stored on the
        state.

        Matching strategy (two passes when *task_id* is given):

        1. Exact pass — when *task_id* is provided, the scan first
           tries to find a live state whose ``task_history_id``
           equals it.  This is what makes multi-view of running
           **sub-agents** work — sub-agents share their parent's
           ``chat_id`` but each carries a distinct
           ``task_history_id`` mirrored from its own ``task_history``
           row by :meth:`ChatSorcarAgent.run`.

        2. Fallback pass — if no exact task-id match is found (or
           *task_id* is ``None``), the scan matches any live state
           whose ``chat_id`` equals *chat_id* **and** which is not
           itself a sub-agent state (``is_subagent=False``).
           Excluding sub-agents from this pass guarantees that
           clicking the parent (or any regular task in the chat)
           never lands the viewer inside a sub-agent's stream by
           accident.

        Multi-viewer fan-out is implemented in the printer: the
        original ``_RunningAgentState`` keeps owning the running task
        and the agent thread keeps tagging events with the original
        (source) tab id, while
        :meth:`BaseBrowserPrinter.subscribe_tab` registers
        *new_tab_id* as an additional viewer so every broadcast is
        duplicated with ``tabId=new_tab_id``.  This means BOTH the
        original client (if still connected) AND the freshly-opened
        client see the streaming events.

        Args:
            chat_id: The chat id of the task the user clicked in
                history.
            new_tab_id: The freshly allocated frontend tab id.
            task_id: When provided, only states whose
                ``task_history_id`` equals this id are eligible.
                Used by sub-agent multi-view to disambiguate from
                the parent (which shares ``chat_id``).

        Returns:
            ``True`` when a matching live agent exists and
            *new_tab_id* is now subscribed to its event stream;
            ``False`` when no matching live agent exists.
        """
        if not new_tab_id:
            return False
        if task_id is None and not chat_id:
            return False
        with self._state_lock:
            source: _RunningAgentState | None = None
            # Pass 1 — exact ``task_history_id`` match (only when
            # caller knows the row id).  Used by sub-agent multi-view
            # so the parent's state (which shares ``chat_id``) is
            # never reached.
            if task_id is not None:
                for t in _RunningAgentState.running_agent_states.values():
                    live_tid = (
                        t.agent._last_task_id
                        if t.agent is not None and t.agent._last_task_id is not None
                        else t.task_history_id
                    )
                    if live_tid != task_id:
                        continue
                    alive = (
                        t.task_thread is not None and t.task_thread.is_alive()
                    )
                    if alive or t.is_task_active:
                        source = t
                        break
            # Pass 2 — chat-id fallback, excluding sub-agent states.
            # Excluding sub-agents here means the fallback can never
            # subscribe a regular-task viewer to a sub-agent's stream
            # (which would be the wrong row entirely).  When the row
            # the user clicked is itself a sub-agent
            # (``is_subagent=True``) the fallback is skipped
            # altogether: a sub-agent row whose thread has already
            # ended must NEVER be attached to its parent's stream.
            if source is None and chat_id and not is_subagent:
                for t in _RunningAgentState.running_agent_states.values():
                    if t.chat_id != chat_id:
                        continue
                    if t.is_subagent:
                        continue
                    alive = (
                        t.task_thread is not None and t.task_thread.is_alive()
                    )
                    if alive or t.is_task_active:
                        source = t
                        break
            if source is None:
                return False
            source_tab_id = source.tab_id
            source_task_id = (
                source.agent._last_task_id if source.agent is not None else None
            )
            if source_task_id is None:
                source_task_id = source.task_history_id
        # Subscribe the new viewer to the running task so live events
        # fan out to the freshly opened tab.  The caller still emits a
        # ``status running=true`` event before the ``task_events``
        # replay so the webview's ``isRunning`` flag is set before
        # ``replayTaskEvents`` runs; otherwise ``applyChevronState``
        # would mark every replayed panel ``.chv-hidden``.
        if source_task_id is not None and source_tab_id != new_tab_id:
            self.printer.subscribe_tab(source_task_id, new_tab_id)
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
        # Capture the task id so the worker thread can tag the
        # ``followup_suggestion`` event correctly for fan-out via the
        # subscriber map.  ``task_id`` is the parameter the caller
        # passes in (the completed task's ``task_history.id``).
        owner_task_key = str(task_id) if task_id is not None else None

        def _run() -> None:
            if owner_task_key is not None:
                self.printer._thread_local.task_id = owner_task_key
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
        self, chat_id: str, task_id: int | None, direction: str, tab_id: str = "",
    ) -> None:
        """Send events for the adjacent task in the same chat session.

        Args:
            chat_id: The string chat session identifier.
            task_id: DB row id of the current task (used as timestamp
                reference).  Using the row id (rather than the task
                text) makes navigation unambiguous when the same task
                description appears multiple times in a chat.
            direction: ``"prev"`` or ``"next"``.
            tab_id: Frontend tab identifier used to route the event.
        """
        result = _get_adjacent_task_by_chat_id(chat_id, task_id, direction)
        event: dict[str, Any] = {
            "type": "adjacent_task_events",
            "direction": direction,
            "task": result["task"] if result else "",
            "task_id": result["task_id"] if result else None,
            "events": result["events"] if result else [],
            "tabId": tab_id,
        }
        self.printer.broadcast(event)

    def _generate_commit_message(self, tab_id: str = "") -> None:
        """Generate a git commit message from current changes.

        Args:
            tab_id: Frontend tab id that requested the message; stamped
                on every emitted ``commitMessage`` event so the
                printer's "system event" routing forwards the message
                only to the originating tab.
        """
        try:
            from pathlib import Path

            from kiss.agents.sorcar.git_worktree import GitWorktreeOps

            if GitWorktreeOps.discover_repo(Path(self.work_dir)) is None:
                self.printer.broadcast({
                    "type": "commitMessage",
                    "message": "",
                    "error": "Not a git repository.",
                    "tabId": tab_id,
                })
                return
            cached_result = _git(self.work_dir, "diff", "--cached")
            diff_text = cached_result.stdout.strip()
            if not diff_text:  # pragma: no branch — LLM API required for else
                self.printer.broadcast({
                    "type": "commitMessage",
                    "message": "",
                    "error": "No staged changes found. Stage files with 'git add' first.",
                    "tabId": tab_id,
                })
                return
            msg = generate_commit_message_from_diff(diff_text)  # pragma: no cover
            self.printer.broadcast({
                "type": "commitMessage", "message": msg, "tabId": tab_id,
            })  # pragma: no cover
        except Exception:  # pragma: no cover — LLM API error handler
            logger.debug("Commit message generation failed", exc_info=True)
            self.printer.broadcast({
                "type": "commitMessage",
                "message": "",
                "error": "Failed to generate",
                "tabId": tab_id,
            })



