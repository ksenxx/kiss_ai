# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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
from collections.abc import Callable
from typing import Any, cast

from kiss.agents.sorcar.persistence import (
    _append_chat_event,
    _chat_has_tasks,
    _current_db_path,
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
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
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
from kiss.agents.vscode.json_printer import JsonPrinter, _coalesce_events
from kiss.agents.vscode.merge_flow import _MergeFlowMixin
from kiss.agents.vscode.task_runner import _TaskRunnerMixin, parse_task_tags
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


# Keys in the persisted ``extra`` JSON that capture the per-task
# snapshot of GLOBAL settings (the model picker and the worktree /
# parallel / auto-commit toggles) at the moment the task ran.  When a
# chat tab is LOADED FROM HISTORY into a tab, the live toggles in the
# webview already reflect the user's CURRENT global preferences (synced
# from ``~/.kiss/config.json`` via ``updateSetting`` / ``configData``);
# replaying these stale per-task values would stamp the toggles back
# onto the loaded task's old settings and silently make the NEXT task
# run with those old settings instead of whatever the user just picked
# globally.  ``_extra_for_replay`` strips them so the frontend's
# ``task_events`` handler cannot clobber the live toggle state.
#
# Keys deliberately preserved: ``startTs`` / ``endTs`` drive the
# "Running …" / "Done (Xm Ys)" header in the chat webview; ``work_dir``
# is per-tab routing state; ``tokens`` / ``cost`` / ``steps`` /
# ``version`` are informational read-outs of the historical run.
_REPLAY_STRIPPED_EXTRA_KEYS = (
    "model",
    "is_worktree",
    "is_parallel",
    "auto_commit_mode",
)


def _extra_for_replay(extra: object) -> str:
    """Return *extra* with global-setting keys stripped for replay.

    See :data:`_REPLAY_STRIPPED_EXTRA_KEYS` for the rationale.  Non-
    string inputs and non-dict-JSON payloads are converted to ``""``
    (the persistence layer always writes a JSON object; any other
    shape is defensive coverage against a future spread / Object.assign
    reader smuggling arbitrary keys through).  An unparseable string
    is returned as-is — the frontend's ``JSON.parse`` is wrapped in
    ``try/catch`` and ignores the payload safely.

    Args:
        extra: The persisted ``extra`` value from
            ``_load_chat_events_by_task_id`` /
            ``_load_latest_chat_events_by_chat_id``.

    Returns:
        A JSON string with the stripped keys removed (or the original
        string if no stripped key was present and it parses as a
        dict), the original string when it does not parse as JSON,
        or ``""`` when *extra* is missing, not a string, or parses
        to a non-dict value.
    """
    if not isinstance(extra, str):
        return ""
    if not extra:
        return ""
    try:
        parsed = json.loads(extra)
    except (json.JSONDecodeError, TypeError):
        return extra
    if not isinstance(parsed, dict):
        # Defensive: any non-dict JSON (list / scalar) is malformed
        # for a task's ``extra`` payload — the persistence layer
        # always writes a JSON object.  Returning ``""`` here means a
        # future frontend reader that uses spread / Object.assign
        # cannot smuggle arbitrary fields through a corrupt or
        # tampered row.  The frontend's ``if (ev.extra)`` guard
        # handles the empty case cleanly.
        return ""
    mutated = False
    for key in _REPLAY_STRIPPED_EXTRA_KEYS:
        if key in parsed:
            parsed.pop(key, None)
            mutated = True
    if not mutated:
        return extra
    return json.dumps(parsed)


def _coalesced_replay_events(events: object) -> list[dict[str, Any]]:
    """Coalesce a persisted event list for a replay broadcast.

    Persisted streams store one row per streamed token; merging
    consecutive same-type delta events (``thinking_delta`` /
    ``text_delta`` / ``system_output``) before broadcasting shrinks the
    ``task_events`` / ``adjacent_task_events`` payload and the
    frontend's replay loop by orders of magnitude while rendering
    identically.

    Args:
        events: The ``events`` value loaded from persistence (typed
            ``object`` by the loaders; always a list of event dicts).

    Returns:
        The coalesced event list.  Empty when *events* is not a list.
    """
    if not isinstance(events, list):
        return []
    return _coalesce_events(cast("list[dict[str, Any]]", events))


def _live_task_id(tab: _RunningAgentState) -> str | None:
    """Return the live ``task_history`` row id for *tab*.

    Prefers ``tab.agent._last_task_id`` (set by the agent the moment it
    allocates the task row) and falls back to ``tab.task_history_id``
    for the post-run window when the agent reference has already been
    cleared.

    Args:
        tab: The per-tab state to inspect.

    Returns:
        The live task id, or ``None`` when neither source is set.
    """
    agent = tab.agent
    if agent is not None and agent._last_task_id is not None:
        return agent._last_task_id
    return tab.task_history_id


def _tab_busy(tab: _RunningAgentState) -> bool:
    """True when *tab* must not be disposed yet OR have its per-tab
    state reset.

    A tab is busy while a task is active, a merge review is in
    progress, or its worker thread is still alive.  Shared by the
    immediate (``_close_tab``) and deferred (``_dispose_if_closed``)
    disposal paths, AND by :meth:`_replay_session` as the
    ``not _tab_busy`` gate on resetting ``tab.use_worktree`` /
    ``tab.use_parallel`` / ``tab.auto_commit_mode`` /
    ``tab.selected_model`` on a history load.  Callers must hold
    ``_state_lock`` while reading the result — the function itself
    only does plain attribute reads and is not internally locked.

    Args:
        tab: The per-tab state to inspect.

    Returns:
        True when any lifecycle flag is still raised.
    """
    return (
        tab.is_task_active
        or tab.is_merging
        or (tab.task_thread is not None and tab.task_thread.is_alive())
    )


def _subagent_is_done(sub_task_id: Any) -> bool:
    """True when the sub-agent owning *sub_task_id* is no longer running.

    Decided from the task-id-keyed :attr:`ChatSorcarAgent.running_agents`
    map: presence under the sub-agent's own task id means its thread is
    still running; absence means it finished.

    Args:
        sub_task_id: The sub-agent's ``task_history`` row id (any type;
            non-int values are treated as done).

    Returns:
        True when no live agent is registered for *sub_task_id*.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    return not (
        isinstance(sub_task_id, str)
        and sub_task_id
        and sub_task_id in ChatSorcarAgent.running_agents
    )


class VSCodeServer(
    _CommandsMixin,
    _TaskRunnerMixin,
    _MergeFlowMixin,
    _AutocompleteMixin,
):
    """Backend server for VS Code extension."""

    def __init__(self, printer: JsonPrinter | None = None) -> None:
        # The transport-specific printer is owned by the caller
        # (typically :class:`RemoteAccessServer`, which passes a
        # :class:`WebPrinter`).  Defaulting to a plain
        # :class:`JsonPrinter` keeps the unit tests that
        # construct a bare ``VSCodeServer()`` working — they patch
        # ``server.printer.broadcast`` themselves to capture events.
        self.printer: JsonPrinter = printer or JsonPrinter()
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
        # ``_tab_chat_views`` maps every frontend tab id (from ANY
        # VS Code window or browser window) to the chat id that tab
        # currently has open.  Maintained by ``_cmd_run`` (launcher
        # tabs), ``_replay_session`` (history / restored-tab viewers),
        # ``_new_chat`` and ``_teardown_tab_resources``.  When a new
        # task is allocated for a chat, ``_subscribe_chat_viewers``
        # scans this map and subscribes every tab that has the chat
        # open to the task's live event stream — the
        # ``_RunningAgentState`` registry alone cannot serve this
        # purpose because pure-viewer tabs deliberately have no
        # registry entry after a daemon restart or deferred disposal
        # (see the C2/C3 note in ``_replay_session``).  Guarded by
        # ``_state_lock``.
        self._tab_chat_views: dict[str, str] = {}
        # Maps ``id(user_answer_queue)`` to the task id of the currently
        # pending ``ask_user_question`` that owns that queue.  ``userAnswer``
        # uses this to close exactly the subscriber set for the task that
        # consumed the answer; old completed-task subscriber sets are
        # intentionally retained and must not receive unrelated close events.
        self._pending_user_answer_tasks: dict[int, str] = {}
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
        # Latest autocomplete sequence number per connection, keyed by
        # the ``connId`` that ``RemoteAccessServer`` stamps on every
        # client command (``""`` for direct callers, e.g. tests).
        # Staleness is tracked per connection so two VS Code windows
        # typing concurrently never cancel each other's in-flight
        # ghost-text requests (a single global counter let whichever
        # window typed last mark every other window's pending request
        # stale).
        self._complete_seq_latest: dict[str, int] = {}
        self._complete_queue: (
            queue.Queue[tuple[str, int, str, str, str, str]] | None
        ) = None
        self._complete_worker: threading.Thread | None = None
        # Per-work_dir file scan cache.  Keyed by the absolute work_dir
        # path the scan ran in so different chat tabs (each with their
        # own ``work_dir``) keep independent file lists for the
        # ``@``-mention autocomplete picker.  ``None`` would have meant
        # "no cache" for the legacy single-dir cache; the dict form uses
        # "key absent" instead and is initialised empty.
        self._file_cache: dict[str, list[str]] = {}
        # Last-seen active editor file path / content per connection,
        # keyed by ``connId`` (``""`` for direct callers).  Each VS
        # Code window reports its own active editor on ``complete``
        # commands; keeping the fallback snapshot per connection means
        # one window's active-file content can never leak into another
        # window's ghost-text autocomplete context.
        self._last_active_file: dict[str, str] = {}
        self._last_active_content: dict[str, str] = {}
        # Optional hook the owning ``RemoteAccessServer`` installs via
        # :meth:`set_cli_running_lookup`.  Returns ``True`` when a
        # given task id is currently being executed by the local
        # ``sorcar`` CLI (announced via ``cliTaskStart`` envelopes).
        # Used by :meth:`_replay_session` to subscribe a freshly
        # resumed webview tab to the live event stream and broadcast
        # a ``status:running`` event so the tab title shows the
        # blinking-green-circle "running" indicator.  ``None`` in
        # tests that construct ``VSCodeServer`` standalone.
        self._cli_running_lookup: Callable[[str], bool] | None = None
        # Companion snapshot hook installed by ``RemoteAccessServer``
        # via :meth:`set_cli_running_task_ids_lookup`.  Returns the
        # full set of task ids the local ``sorcar`` CLI has announced
        # as running (via ``cliTaskStart``).  Used by
        # :meth:`_get_running_task_ids` to UNION the CLI-launched
        # running tasks with the UI-launched ones, so the History
        # panel's pulsing-green-dot ``is_running`` flag is set on
        # CLI tasks too (the in-process ``_RunningAgentState``
        # registry only tracks UI-launched tasks).
        self._cli_running_task_ids_lookup: Callable[[], set[str]] | None = None

    def set_cli_running_lookup(
        self, lookup: Callable[[str], bool] | None,
    ) -> None:
        """Install the CLI-task running-lookup used by :meth:`_replay_session`.

        Called by :class:`RemoteAccessServer` so the resume path can
        detect tasks the local ``sorcar`` CLI is currently running
        and subscribe the freshly opened webview tab to their live
        event stream.  Passing ``None`` clears the hook.

        Args:
            lookup: Callable taking the task id and returning
                ``True`` when the CLI is running it.
        """
        self._cli_running_lookup = lookup

    def set_cli_running_task_ids_lookup(
        self, lookup: Callable[[], set[str]] | None,
    ) -> None:
        """Install the CLI-running-task-id snapshot used by ``_get_history``.

        Called by :class:`RemoteAccessServer` so the history listing
        can union the CLI-launched running task ids with the
        in-process UI-launched ones, making the History panel render
        the pulsing-green-dot indicator on CLI tasks as well.
        Passing ``None`` clears the hook.

        Args:
            lookup: Zero-arg callable returning a fresh snapshot set
                of CLI-launched running ``task_history`` row ids.
        """
        self._cli_running_task_ids_lookup = lookup

    def drop_connection_state(self, conn_id: str) -> None:
        """Discard per-connection autocomplete state for a closed connection.

        Called by :class:`RemoteAccessServer` when a client connection
        (one per VS Code window / browser tab) goes away, so the
        per-connection active-file snapshots and autocomplete sequence
        counters do not accumulate forever in a long-lived daemon.

        Args:
            conn_id: The connection id that was stamped (as ``connId``)
                on every command from the departed connection.  An
                empty id is ignored — it is the shared key used by
                direct callers (tests) and must survive.
        """
        if not conn_id:
            return
        with self._state_lock:
            self._last_active_file.pop(conn_id, None)
            self._last_active_content.pop(conn_id, None)
            self._complete_seq_latest.pop(conn_id, None)

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
            # completes, and the next task's
            # :meth:`_TaskRunnerMixin._run_task_inner` re-enters here
            # (via ``_get_tab``) to allocate a fresh agent before the
            # run starts.  So no agent state ever survives a task
            # boundary; the slot populated here is a fresh, empty
            # agent.
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
        # The shared routing fields are used as dict keys (tab
        # registry, per-work_dir file cache, per-connection
        # autocomplete state) throughout the handlers: a non-string
        # value (e.g. ``"tabId": [1]`` from a malformed client) raises
        # ``TypeError: unhashable type`` out of a registry lookup — or
        # silently corrupts daemon-global state (``setWorkDir`` would
        # assign the list to ``self.work_dir``) — and an exception
        # escaping this method kills the transport's whole receive
        # loop, i.e. the entire client connection.  Coerce them to
        # ``""``, the neutral value every handler already guards.
        for field in ("tabId", "workDir", "connId"):
            value = cmd.get(field)
            if value is not None and not isinstance(value, str):
                cmd[field] = ""
        cmd_type = cmd.get("type", "")
        # A non-string ``type`` (e.g. a list) is unhashable: using it
        # as a dict key would raise TypeError, which escapes to the
        # transport's receive loop and kills the whole client
        # connection.  Route it to the unknown-command branch instead.
        handler = (
            self._HANDLERS.get(cmd_type) if isinstance(cmd_type, str) else None
        )
        if handler is not None:
            handler(self, cmd)
        else:
            event: dict[str, Any] = {"type": "error", "text": f"Unknown command: {cmd_type}"}
            tab_id = cmd.get("tabId")
            if tab_id is not None:
                event["tabId"] = tab_id
            conn_id = cmd.get("connId", "")
            if conn_id:
                # Reply only to the connection that sent the unknown
                # command — other windows' webviews must not render
                # an error banner for a command they never issued.
                event["connId"] = conn_id
            self.printer.broadcast(event)


    def _get_models(self, conn_id: str = "") -> None:
        """Send available models list with usage counts and pricing.

        Stamped with the requesting connection's ``conn_id`` (when
        non-empty) so the reply reaches only the window that asked —
        one window refreshing its model picker must not repaint
        another window's picker or change its selected model.

        Args:
            conn_id: Requesting connection id (``""`` for direct callers).
        """
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

        # ``kiss-web`` can outlive VS Code windows.  A fresh VS Code
        # activation asks this long-lived daemon for ``getModels``;
        # if the user selected a different model in a previous window
        # session, that choice is persisted in ``config.json`` and must
        # take precedence over this process's stale in-memory default.
        # Honor the persisted last model whenever it is still runnable.
        available_names = {m["name"] for m in models_list}
        with self._state_lock:
            # Read the persisted last model INSIDE the lock so a
            # concurrent ``_cmd_select_model`` (which now persists
            # under the same lock) cannot leave us with a stale
            # on-disk value that would clobber the user's just-picked
            # in-memory selection.
            persisted = _load_last_model()
            if persisted in available_names:
                self._default_model = persisted

            # On a fresh installation the server is constructed before any
            # API key is configured, so ``self._default_model`` is the
            # ``"No model"`` sentinel.  Once a key becomes available (env var
            # or settings panel), ``get_available_models()`` returns real
            # models, but the cached sentinel would keep the picker stuck on
            # "No model".  Re-resolve the default whenever the cached
            # selection is no longer a valid choice so the picker recovers.
            # If the only available option is a custom endpoint, the core
            # default resolver cannot name it; choose the first available
            # entry so the picker never emits a stale unavailable selection.
            if self._default_model not in available_names:
                refreshed = get_default_model()
                if refreshed in available_names:
                    self._default_model = refreshed
                elif models_list:
                    self._default_model = str(models_list[0]["name"])
                else:
                    self._default_model = refreshed
            selected = self._default_model

        event: dict[str, Any] = {
            "type": "models",
            "models": models_list,
            "selected": selected,
        }
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

    def _get_running_task_ids(self) -> set[str]:
        """Return the set of task_history row ids with alive worker threads.

        Scans all per-tab ``_RunningAgentState`` entries and collects
        the live task id (see :func:`_live_task_id`) of those whose
        ``task_thread`` is still alive.  Acquires ``_state_lock``
        internally (re-entrant, so safe to call with it already held).

        Returns:
            Set of ``task_history.id`` values that are currently running.
        """
        running: set[str] = set()
        with self._state_lock:
            for tab in _RunningAgentState.running_agent_states.values():
                tid = _live_task_id(tab)
                if (
                    tid is not None
                    and tab.task_thread is not None
                    and tab.task_thread.is_alive()
                ):
                    running.add(tid)
        # CLI-launched tasks run in a separate ``sorcar`` process and
        # have no ``_RunningAgentState`` entry on the daemon, but the
        # ``RemoteAccessServer`` tracks them via ``cliTaskStart`` /
        # ``cliTaskEnd`` envelopes.  Merge that set in so the History
        # panel renders the pulsing-green-dot running indicator on
        # CLI-launched rows too.
        if self._cli_running_task_ids_lookup is not None:
            try:
                running.update(self._cli_running_task_ids_lookup())
            except Exception:  # pragma: no cover — defensive
                logger.exception(
                    "cli running-task lookup failed; continuing",
                )
        return running

    def _overlay_live_metrics(
        self, session: dict[str, Any], task_id: str,
    ) -> None:
        """Replace persisted metrics with live agent data for a running task.

        Scans ``_RunningAgentState.running_agent_states`` for a tab whose
        live task id matches *task_id* and overwrites the ``tokens``,
        ``cost``, and ``steps`` fields in *session* with current values
        from the running agent, including the in-progress executor's
        ``step_count``.  Acquires ``_state_lock`` internally
        (re-entrant, so safe to call with it already held).

        Args:
            session: The history session dict to update in place.
            task_id: The ``task_history.id`` of the running task.
        """
        with self._state_lock:
            for tab in _RunningAgentState.running_agent_states.values():
                agent = tab.agent
                if agent is None:
                    continue
                if _live_task_id(tab) != task_id:
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
                # Overlay run-mode metadata for the meta line.
                # The persisted ``extra`` is only written at task
                # END (see ``_TaskRunnerMixin._run_task_inner``),
                # so a still-running task has no model / flag
                # values in its row.  Read them straight off the
                # live agent / tab so the history sidebar's meta
                # line renders the same model + flags the running
                # tab actually uses.  ``agent.model_name`` is the
                # model the in-flight task was launched with
                # (``_TaskRunnerMixin._cmd_run`` plumbs this from
                # the ``run`` command's ``model`` field through to
                # the agent); we prefer it over the tab-level
                # ``selected_model`` because the user can change
                # ``selected_model`` mid-task via the model picker.
                mdl_live = getattr(agent, "model_name", "") or getattr(
                    tab, "selected_model", ""
                )
                if isinstance(mdl_live, str) and mdl_live:
                    session["model"] = mdl_live
                session["is_worktree"] = bool(
                    getattr(tab, "use_worktree", False)
                )
                session["is_parallel"] = bool(
                    getattr(tab, "use_parallel", False)
                )
                session["auto_commit_mode"] = bool(
                    getattr(tab, "auto_commit_mode", False)
                )
                break

    def _get_history(
        self,
        query: str | None,
        offset: int = 0,
        generation: int = 0,
        conn_id: str = "",
    ) -> None:
        """Send conversation history with pagination support.

        The reply is stamped with the requesting connection's
        ``conn_id`` (when non-empty) so it reaches only the VS Code
        window / browser tab that asked — one window's history search
        must not repaint another window's history panel.
        """
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
            _raw_eid = entry.get("id")
            # r4-vscode-H2 — accept legacy int task_ids from DBs that
            # escaped auto-migration.  Coerce to str so the rest of
            # the history pipeline (which assumes a string id) works
            # uniformly.
            entry_id: str | None
            if isinstance(_raw_eid, str) and _raw_eid:
                entry_id = _raw_eid
            elif isinstance(_raw_eid, int) and _raw_eid:
                entry_id = str(_raw_eid)
            else:
                entry_id = None
            is_running = entry_id is not None and entry_id in running_task_ids
            session: dict[str, Any] = {
                "id": chat_id,
                "task_id": entry_id,
                "title": task,
                "timestamp": entry.get("timestamp", 0),
                "preview": task,
                "has_events": has_events,
                # A running task's row still holds the "Agent Failed
                # Abruptly" sentinel result — don't paint it as failed
                # while it is alive.
                "failed": _is_failed_result(result) and not is_running,
                "is_running": is_running,
                "tokens": 0,
                "cost": 0.0,
                "steps": 0,
                "is_favorite": False,
                # ``work_dir`` persisted on this task's row at
                # completion time (see ``_TaskRunnerMixin`` →
                # ``_save_task_extra``).  Surfaced so the history
                # sidebar's Workspace filter checkbox can hide rows
                # whose ``work_dir`` differs from the client's
                # currently-configured workspace.  Empty string when
                # the task pre-dates the persistence change or is
                # still running.
                "work_dir": "",
                # Per-task run-mode metadata persisted by
                # ``_TaskRunnerMixin._run_task_inner`` into the row's
                # ``extra`` JSON.  Surfaced so the history sidebar can
                # render its dot-separated meta line under the
                # workspace row::
                #
                #     <model> • <wt|no-wt> • <parallel|sequential>
                #         • <auto-commit|manual-commit>
                #
                # Legacy rows that pre-date the persistence change
                # (no ``model`` field) get an empty ``model``, which
                # the frontend interprets as "render no meta line".
                "model": "",
                "is_worktree": False,
                "is_parallel": False,
                "auto_commit_mode": False,
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
                        # r3-vscode-H2: accept both UUID strings and
                        # legacy int parent ids written by pre-UUID
                        # task_history rows that escaped migration.
                        if isinstance(pid, str) and pid:
                            session["parent_task_id"] = pid
                        elif isinstance(pid, int) and pid:
                            session["parent_task_id"] = str(pid)
                    # Field-by-field numeric coercion: garbage values
                    # fall back to the zero default, numeric strings
                    # round-trip through the cast.
                    for key, cast, default in (
                        ("tokens", int, 0),
                        ("cost", float, 0.0),
                        ("steps", int, 0),
                    ):
                        try:
                            session[key] = cast(extra_obj.get(key, default) or default)
                        except (TypeError, ValueError):
                            session[key] = default
                    try:
                        session["endTs"] = int(extra_obj.get("endTs", 0) or 0)
                    except (TypeError, ValueError):
                        session["endTs"] = 0
                    session["is_favorite"] = bool(
                        extra_obj.get("is_favorite", False)
                    )
                    wd_raw = extra_obj.get("work_dir", "")
                    if isinstance(wd_raw, str):
                        session["work_dir"] = wd_raw
                    # Run-mode metadata for the meta line.
                    # ``model`` round-trips verbatim when present;
                    # the three booleans are coerced via ``bool()``
                    # so missing / non-boolean garbage falls back
                    # to ``False`` (rendered as "no-wt" /
                    # "sequential" / "manual-commit").
                    mdl_raw = extra_obj.get("model", "")
                    if isinstance(mdl_raw, str):
                        session["model"] = mdl_raw
                    session["is_worktree"] = bool(
                        extra_obj.get("is_worktree", False)
                    )
                    session["is_parallel"] = bool(
                        extra_obj.get("is_parallel", False)
                    )
                    session["auto_commit_mode"] = bool(
                        extra_obj.get("auto_commit_mode", False)
                    )
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
            if session.get("is_running") and entry_id is not None:
                self._overlay_live_metrics(session, entry_id)
            sessions.append(session)
        event: dict[str, Any] = {
            "type": "history", "sessions": sessions,
            "offset": offset, "generation": generation,
        }
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

    def _handle_delete_task(self, task_id: str) -> None:
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

    def _handle_set_favorite(self, task_id: str, is_favorite: bool) -> None:
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

    def _get_frequent_tasks(self, limit: int = 50, conn_id: str = "") -> None:
        """Send the top *limit* most-frequent tasks (highest count first).

        Emits a ``frequentTasks`` event whose ``tasks`` field is a
        list of ``{task, count, timestamp}`` dicts ordered by ``count``
        descending.  Stamped with the requesting connection's
        ``conn_id`` (when non-empty) so the reply reaches only the
        window that asked.

        Args:
            limit: Maximum number of frequent tasks to return.
            conn_id: Requesting connection id (``""`` for direct callers).
        """
        event: dict[str, Any] = {
            "type": "frequentTasks",
            "tasks": _load_frequent_tasks(limit=limit),
        }
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

    def _get_input_history(self, conn_id: str = "") -> None:
        """Send deduplicated task texts for arrow-key cycling.

        Loads the full persisted history so ArrowUp can traverse every
        distinct task stored in ``sorcar.db``, not just an arbitrary
        recent subset.  Stamped with the requesting connection's
        ``conn_id`` (when non-empty) so the reply reaches only the
        window that asked.

        Args:
            conn_id: Requesting connection id (``""`` for direct callers).
        """
        entries = _load_history()
        seen: set[str] = set()
        tasks: list[str] = []
        for e in entries:
            task = str(e.get("task", "")).strip()
            if task and task not in seen:
                seen.add(task)
                tasks.append(task)
        event: dict[str, Any] = {"type": "inputHistory", "tasks": tasks}
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

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
            if tab is not None and _tab_busy(tab):
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
            if _tab_busy(tab):
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
            #
            # "Lost slides" bug fix: when the last task ended in
            # failure / user-Stop (``_pending_review=True``) and the
            # user has not yet explicitly chosen Merge or Discard,
            # commit the partial work onto the worktree branch but
            # do NOT silently squash-merge it.  Closing a chat tab
            # while a stopped task's partial work is unreviewed must
            # not be allowed to overwrite the user's original
            # branch.  The branch survives in ``git branch`` for
            # manual recovery via ``git checkout <branch>``.
            try:
                wt_agent = self._ensure_wt_agent(tab)
                if wt_agent is not None and wt_agent._wt_pending:
                    if getattr(wt_agent, "_pending_review", False):
                        wt_agent._preserve_pending_worktree_for_review()
                    else:
                        wt_agent._release_worktree()
            except Exception:
                logger.debug("Worktree release on tab close failed", exc_info=True)
        # ``cleanup_tab`` removes *tab_id* from every task subscriber
        # set so a stale viewer never receives events.  Per-task state
        # (recordings, persist_agents, offsets) is owned by the agent
        # thread and cleaned up by :meth:`_TaskRunnerMixin` /
        # :meth:`ChatSorcarAgent.run`.
        self.printer.cleanup_tab(tab_id)
        # A disposed tab no longer views any chat: drop it from the
        # chat-viewer map so future tasks on that chat do not fan
        # events out to a tab that no longer exists.
        with self._state_lock:
            self._tab_chat_views.pop(tab_id, None)
        _cleanup_merge_data(str(_merge_data_dir(tab_id)))

    def _new_chat(self, tab_id: str) -> None:
        """Start a new chat session for the given tab.

        The ``newChat`` command is only issued by the frontend's
        ``createNewTab`` flow, which always allocates a fresh tab id
        that the backend has never seen before.  ``_get_tab`` creates a
        clean ``_RunningAgentState``, so there is no prior run state (no active
        task, no in-progress merge, no pending worktree, no carried-over
        warnings) to guard against here.

        Re-reads the last user-picked model from ``config.json`` so the
        new tab uses the correct model even when the in-memory default
        has drifted (e.g. after switching between tabs with different
        models).

        Args:
            tab_id: The frontend tab identifier (a freshly-minted uuid).
        """
        if not tab_id:
            # A malformed ``newChat`` without a tabId must not mint a
            # phantom registry entry keyed "" via ``_get_tab`` —
            # ``_cmd_close_tab`` guards against empty ids, so such an
            # entry (and its eagerly-created agent) could never be
            # disposed.  Mirror ``_replay_session``'s empty-id no-op.
            logger.debug("newChat ignored: empty tabId")
            return
        tab = self._get_tab(tab_id)
        with self._state_lock:
            # Read the persisted last model INSIDE ``_state_lock`` so a
            # concurrent ``_cmd_select_model`` (which now persists
            # under the same lock) cannot leave us with a stale
            # on-disk value that would clobber the user's just-picked
            # in-memory selection.  Without this guard,
            # ``_load_last_model()`` could read the OLD on-disk value
            # mid-flight of ``_cmd_select_model``'s disk write, then
            # ``self._default_model = persisted`` would revert the
            # just-picked model both on the daemon-wide default AND
            # on this new tab's ``selected_model``.
            persisted = _load_last_model()
            if persisted:
                self._default_model = persisted
            tab.selected_model = self._default_model
            # Clear the long-lived chat identity for this tab; the
            # next ``_cmd_run`` will mint a fresh chat id when it
            # builds the per-task agent.  No agent is owned by the
            # tab outside of an active task, so there is nothing
            # else to reset here.
            tab.chat_id = ""
            tab.last_task_id = None
            # A fresh chat has no chat id yet, so the tab views no
            # chat until ``_cmd_run`` mints one or ``_replay_session``
            # associates a resumed one.
            self._tab_chat_views.pop(tab_id, None)
            # Snapshot the model under the lock so the ``showWelcome``
            # broadcast below cannot disagree with the in-memory state
            # captured for ``tab.selected_model`` above (a concurrent
            # ``_cmd_select_model`` could otherwise mutate
            # ``self._default_model`` between the lock release and the
            # broadcast read).
            welcome_model = self._default_model
        # Drop any live-task subscriptions this tab carried from the
        # chat it previously displayed (e.g. a still-running task it
        # was viewing via ``_reattach_running_chat``).  The webview now
        # shows the welcome screen, so the old task's streaming events
        # must no longer fan out to this tab.  Resolved via ``getattr``
        # because some duck-typed test printers implement only the
        # broadcast/subscribe subset of the printer protocol.
        cleanup_tab = getattr(self.printer, "cleanup_tab", None)
        if cleanup_tab is not None:
            cleanup_tab(tab_id)
        self.printer.broadcast({
            "type": "showWelcome",
            "tabId": tab_id,
            "model": welcome_model,
        })

    def _replay_session(
        self, chat_id: str, tab_id: str = "", task_id: str | None = None,
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
            # The persisted ``task_history`` row may not exist yet —
            # ``ChatSorcarAgent.run`` writes it inside the worker
            # thread, and a tab that STARTED a task and is immediately
            # closed+reopened (or VS Code reload) can race the writer.
            # Without a row there are no events to replay, but a live
            # ``_RunningAgentState`` may still exist for this
            # ``chat_id`` / ``tab_id`` — re-subscribe the re-opened
            # webview to its stream and broadcast ``status running:true``
            # so the webview learns the task is running.  Otherwise the
            # re-opened tab would silently keep ``isRunning=false``,
            # the user's next message would be sent as a ``submit``
            # (→ ``run``) that the daemon would have to inject (the
            # tab is busy) — and after the task finishes the tab would
            # appear idle, but any input would NEVER reach the
            # appendUserMessage path because the webview never learned
            # the task was running in the first place.  Worst case is
            # silent: the user types and nothing happens.
            cleanup_tab = getattr(self.printer, "cleanup_tab", None)
            if cleanup_tab is not None:
                cleanup_tab(tab_id)
            rebound_running = self._reattach_running_chat(
                chat_id,
                tab_id,
                task_id=task_id,
                is_subagent=False,
            )
            if rebound_running:
                start_ts = self._live_task_start_ms(task_id, chat_id)
                self.printer.broadcast({
                    "type": "status",
                    "running": True,
                    "tabId": tab_id,
                    "startTs": start_ts,
                })
                # Empty task_events so the webview's replay loop
                # transitions out of its "loading" state cleanly.
                self.printer.broadcast({
                    "type": "task_events",
                    "events": [],
                    "task": "",
                    "task_id": task_id,
                    "chat_id": chat_id,
                    "extra": "",
                    "tabId": tab_id,
                })
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
        _raw_rebound_tid = result.get("task_id") if result else None
        rebound_task_id: str | None
        if isinstance(_raw_rebound_tid, str) and _raw_rebound_tid:
            rebound_task_id = _raw_rebound_tid
        elif isinstance(_raw_rebound_tid, int) and _raw_rebound_tid:
            # r4-vscode-H1 — accept legacy int payloads from DBs that
            # escaped the auto-migration; mirror the same defence
            # applied to ``parent_task_id`` in r3-vscode-H2.
            rebound_task_id = str(_raw_rebound_tid)
        else:
            rebound_task_id = None
        # The tab is navigating to (possibly) another chat: drop every
        # live-task subscription it carried from whatever it displayed
        # before, so the previous chat's still-running task does not
        # keep streaming its events into a webview that now renders a
        # different conversation.  When the loaded chat itself is
        # backed by a running task, ``_reattach_running_chat`` below
        # re-subscribes this tab to the correct stream.  Resolved via
        # ``getattr`` because some duck-typed test printers implement
        # only the broadcast/subscribe subset of the printer protocol.
        cleanup_tab = getattr(self.printer, "cleanup_tab", None)
        if cleanup_tab is not None:
            cleanup_tab(tab_id)
        rebound_running = self._reattach_running_chat(
            chat_id,
            tab_id,
            task_id=rebound_task_id,
            is_subagent=subagent_info is not None,
        )
        # Tasks launched by the local ``sorcar`` CLI never have a
        # :class:`_RunningAgentState` registry entry in this process
        # (the CLI agent runs in a separate Python process and only
        # talks to the daemon over UDS).  ``_reattach_running_chat``
        # therefore returns ``False`` for them and a webview tab
        # opened from the history sidebar would not be subscribed to
        # the live event stream and not get the blinking-green-circle
        # "running" indicator.  Consult the CLI-running-task hook the
        # :class:`RemoteAccessServer` installed via
        # :meth:`set_cli_running_lookup` and, when the click resolved
        # to a still-running CLI task, subscribe this tab under the
        # task id (so subsequent ``cliEvent`` relays fan out to it)
        # AND broadcast a ``status:running=true`` event to start the
        # indicator.  Mirrors the tail of the rebound-running branch
        # below.
        if (
            not rebound_running
            and rebound_task_id is not None
            and self._cli_running_lookup is not None
            and self._cli_running_lookup(rebound_task_id)
        ):
            self.printer.subscribe_tab(str(rebound_task_id), tab_id)
            rebound_running = True

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
                # Do NOT seed ``tab.use_worktree`` /
                # ``tab.use_parallel`` / ``tab.auto_commit_mode`` /
                # ``tab.selected_model`` from the loaded task's
                # ``extra``: those historical values are a snapshot
                # of the toggles / model at the time the task ran,
                # NOT the user's current global setting.  Loading a
                # chat is a VIEW operation — the NEXT task started in
                # this tab will overwrite ``tab.use_worktree`` /
                # ``tab.use_parallel`` / ``tab.auto_commit_mode`` from
                # the frontend's live ``useWorktree`` / ``useParallel``
                # / ``autoCommit`` flags (which mirror the user's
                # CURRENT global toggles), so any value seeded here
                # would be either stale (last-task snapshot) or
                # immediately clobbered (correct global).
                #
                # Additionally, RESET these per-tab fields to baseline
                # (= ``self._default_model`` for the model, ``False``
                # for the toggle flags).  Without this reset a tab
                # that previously ran a task — and therefore had
                # ``tab.use_worktree`` / ``tab.use_parallel`` /
                # ``tab.auto_commit_mode`` / ``tab.selected_model``
                # written by ``_run_task`` — would keep those values
                # after loading a different chat into the same tab,
                # which ``_get_history`` then reads as the running-
                # session sidebar metadata for any LIVE task in the
                # loaded chat (see ``_get_history`` running-session
                # loop earlier in this file).  Resetting here mirrors
                # ``_new_chat``'s baseline (``tab.selected_model =
                # self._default_model`` at the start of a fresh
                # chat) and keeps a history-load idempotent w.r.t.
                # the tab's prior task-runner state.
                #
                # Skip the reset when a task is actively running in
                # this tab — that scenario means the user is re-
                # rendering the live chat into the SAME tab that owns
                # the run, so the in-flight per-tab fields are the
                # source of truth for ``_get_history``'s sidebar
                # metadata and must not be clobbered.
                # Use the shared ``_tab_busy`` predicate so the guard
                # ALSO covers an in-flight worktree-merge review
                # (``tab.is_merging``).  Without that bit, a
                # ``_replay_session`` re-entry on a tab whose
                # post-task merge prompt is still open (e.g. user
                # reloads the VS Code window mid-merge) would
                # clobber ``tab.use_worktree=False`` and break
                # ``_finish_merge`` / ``_present_pending_worktree``
                # in ``merge_flow.py`` (which dispatch on
                # ``tab.use_worktree``), leaking the worktree
                # directory.
                if not _tab_busy(tab):
                    tab.use_worktree = False
                    tab.use_parallel = False
                    tab.auto_commit_mode = False
                    tab.selected_model = self._default_model
                tab.frontend_closed = False
            # Record which chat this tab now displays so a task later
            # started on the same chat (from any tab in any window)
            # subscribes this tab to its live event stream (see
            # :meth:`_TaskRunnerMixin._subscribe_chat_viewers`).
            # Tracked independently of the ``_RunningAgentState``
            # registry because viewer tabs may have no registry entry
            # (C2/C3 above).  A tab converted into a sub-agent view
            # is NOT a viewer of the (parent) chat: streaming the
            # parent's follow-up tasks into a sub-agent tab would mix
            # two different task streams in one webview.
            if subagent_info is None and chat_id:
                self._tab_chat_views[tab_id] = chat_id
            else:
                self._tab_chat_views.pop(tab_id, None)

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
            # ``isDone`` (see :func:`_subagent_is_done`) lets the
            # reopened tab render the same "done, no indicator" state
            # the original tab ended on (instead of pulsing ◉ purple
            # forever because no later ``subagentDone`` arrives).
            is_done = _subagent_is_done(result.get("task_id"))
            # Look up the parent's frontend tab id so the frontend can
            # record the parent → child relationship.  Without this,
            # closing the parent tab would not cascade-close this
            # sub-agent tab (see ``closeTab`` in media/main.js, which
            # walks ``parentTabId`` chains).
            parent_tid_raw = subagent_info.get("parent_task_id")
            # r3-vscode-H2: accept legacy int parent ids that may
            # appear in DBs that escaped migration (extra JSON in a
            # row that was never re-saved through the new typed
            # column path).  Stringify rather than drop so the
            # parent-tab resolver still has a value to match.
            if isinstance(parent_tid_raw, str) and parent_tid_raw:
                parent_tid: str | None = parent_tid_raw
            elif isinstance(parent_tid_raw, int) and parent_tid_raw:
                parent_tid = str(parent_tid_raw)
            else:
                parent_tid = None
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
            # ``extra.startTs`` is only persisted at task END
            # (``_save_task_extra`` in ``_run_task_inner``'s cleanup
            # finally) — for a STILL-RUNNING task (the only case this
            # branch handles) it is therefore always absent and the
            # value above stays 0, which the frontend ignores
            # (``ev.startTs > 0`` guard) before mis-anchoring the
            # "Running …" timer at the client's ``Date.now()``.  Fall
            # back to the live agent's in-memory start timestamp,
            # stamped by ``_TaskRunnerMixin._run_task_inner``.
            if start_ts_for_resume <= 0:
                start_ts_for_resume = self._live_task_start_ms(
                    rebound_task_id, chat_id,
                )
            self.printer.broadcast({
                "type": "status",
                "running": True,
                "tabId": tab_id,
                "startTs": start_ts_for_resume,
            })
        # Coalesce consecutive per-token delta events before shipping
        # the replay payload: persisted streams store one row per
        # streamed token, so a long task can carry tens of thousands
        # of tiny ``text_delta`` / ``thinking_delta`` events.  Merging
        # them (same contract as ``JsonPrinter.stop_recording``)
        # shrinks the JSON payload and the frontend's replay loop by
        # orders of magnitude while rendering identically.
        self.printer.broadcast({
            "type": "task_events",
            "events": _coalesced_replay_events(result["events"]),
            "task": result["task"],
            "task_id": result.get("task_id"),
            "chat_id": chat_id,
            # ``_extra_for_replay`` strips ``model`` / ``is_worktree``
            # / ``is_parallel`` / ``auto_commit_mode`` so loading a
            # chat into this tab cannot clobber the live toggle / model
            # state that mirrors the user's CURRENT global settings.
            # A follow-up task started in this tab must use whatever
            # the toggles currently say (= the global settings the
            # user just picked), not the stale per-task snapshot of
            # the historical row being loaded.  See the comment on
            # :data:`_REPLAY_STRIPPED_EXTRA_KEYS`.
            "extra": _extra_for_replay(result.get("extra", "")),
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
        if (
            subagent_info is None
            and isinstance(rebound_task_id, str)
            and rebound_task_id
        ):
            self._open_persisted_subagent_tabs(
                parent_task_id=rebound_task_id, parent_tab_id=tab_id,
            )

    def _resolve_parent_tab_id_for_sub(
        self,
        *,
        parent_task_id: str | None,
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
                    if _live_task_id(st) == parent_task_id:
                        return st.tab_id

            if chat_id:
                # Exclude ``sub_tab_id`` itself: when a non-subagent
                # state already exists for the freshly opened sub-tab
                # (e.g. the user previously ran a task in that tab),
                # ``_replay_session`` copies the resumed session's
                # ``chat_id`` onto it before this resolver runs.
                # Without the guard below, that state would match here
                # and we'd return the sub-tab's own id, creating a
                # self-referential parent_tab_id and a self-loop in
                # the frontend's parent→child cascade-close registry.
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
        self, *, parent_task_id: str, parent_tab_id: str,
    ) -> None:
        """Broadcast ``openSubagentTab`` + ``task_events`` for every
        persisted sub-agent row whose parent is *parent_task_id*.

        The sub-tab ids are deterministic
        (``f"{parent_tab_id}__sub_{sub_task_id}"``) so that clicking
        the same parent task twice in a row updates the existing
        sub-agent tabs in place instead of stacking duplicates — the
        webview's ``openSubagentTab`` handler is idempotent on
        ``tab_id``.

        ``isDone`` is decided by :func:`_subagent_is_done`: presence in
        :attr:`ChatSorcarAgent.running_agents` under the sub-agent's
        own task id means its thread is still running so the tab
        should pulse the ◉ indicator; absence means the sub-agent has
        completed and the tab should render as a finished tab without
        the indicator.

        Args:
            parent_task_id: ``task_history.id`` of the parent task.
            parent_tab_id: Frontend tab id of the parent tab.  Used
                as the prefix for the deterministic sub-tab ids.
        """
        sub_rows = _load_subagent_rows_by_parent_task_id(parent_task_id)
        for idx, row in enumerate(sub_rows):
            sub_task_id = row["task_id"]
            sub_tab_id = f"{parent_tab_id}__sub_{sub_task_id}"
            description = str(row.get("task", "") or "")
            is_done = _subagent_is_done(sub_task_id)
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
                "events": _coalesced_replay_events(row["events"]),
                "task": description,
                "task_id": sub_task_id,
                "chat_id": row.get("chat_id", ""),
                # Sub-agent tabs reopened on a parent-task history
                # load must follow the same strip rule as the parent
                # ``_replay_session`` broadcast above: a persisted
                # sub-agent ``extra`` snapshot of
                # ``model`` / ``is_worktree`` / ``is_parallel`` /
                # ``auto_commit_mode`` is just as historical as the
                # parent's, and the frontend's background-tab
                # ``task_events`` branch still reads ``extra.model``
                # into ``teTab.selectedModel`` (see ``media/main.js``
                # near the ``bgExtra.model`` block).  Stripping here
                # prevents a stale per-sub-agent model from being
                # inherited by the live model picker when the user
                # later switches to that sub-tab.
                "extra": _extra_for_replay(row.get("extra", "")),
                "tabId": sub_tab_id,
            })

    def _live_task_start_ms(
        self, task_id: str | None, chat_id: str,
    ) -> int:
        """Return the start timestamp (ms since epoch) of a live task.

        Scans :attr:`_RunningAgentState.running_agent_states` for the
        state owning the running task and reads the
        ``_task_start_ms`` attribute that
        :meth:`_TaskRunnerMixin._run_task_inner` stamps on the live
        agent at run start.  Matching mirrors
        :meth:`_reattach_running_chat`: an exact ``task_history`` row
        id match when *task_id* is given, otherwise a non-subagent
        ``chat_id`` match.

        Args:
            task_id: The ``task_history`` row id of the task, or
                ``None`` to match by chat id only.
            chat_id: The chat id of the task (used when *task_id* is
                ``None``).

        Returns:
            The agent's start timestamp in ms since epoch, or ``0``
            when no live agent (or no stamped timestamp) is found.
        """
        with self._state_lock:
            for tab in _RunningAgentState.running_agent_states.values():
                if task_id is not None:
                    if _live_task_id(tab) != task_id:
                        continue
                elif (
                    not chat_id
                    or tab.chat_id != chat_id
                    or tab.is_subagent
                ):
                    continue
                start_ms = int(
                    getattr(tab.agent, "_task_start_ms", 0) or 0
                )
                if start_ms > 0:
                    return start_ms
        return 0

    def _reattach_running_chat(
        self,
        chat_id: str,
        new_tab_id: str,
        *,
        task_id: str | None = None,
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
        :meth:`JsonPrinter.subscribe_tab` registers
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
                    if _live_task_id(t) != task_id:
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
            source_task_id = _live_task_id(source)
        # Subscribe the new viewer to the running task so live events
        # fan out to the freshly opened tab.  The caller still emits a
        # ``status running=true`` event before the ``task_events``
        # replay so the webview's ``isRunning`` flag is set before
        # ``replayTaskEvents`` runs; otherwise ``applyChevronState``
        # would mark every replayed panel ``.chv-hidden``.
        # ``source_tab_id == new_tab_id`` (the launcher tab replaying
        # its OWN running chat, e.g. a webview restore) is subscribed
        # too: ``_replay_session`` just dropped the tab's previous
        # subscriptions via ``cleanup_tab``, so the owner must be
        # re-registered.  ``subscribe_tab`` is idempotent.
        if source_task_id is not None:
            self.printer.subscribe_tab(source_task_id, new_tab_id)
        return True


    def _generate_followup_async(
        self,
        task: str,
        result: str,
        task_id: str | None,
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
        # Capture the database the task id belongs to NOW: this thread
        # outlives the request (LLM call takes seconds), and a stale
        # numeric id must never be written into a *different* database
        # (swapped by test fixtures / a daemon restart) where it would
        # resolve to an unrelated task's row.
        origin_db_path = _current_db_path()

        def _run() -> None:
            if owner_task_key is not None:
                self.printer._thread_local.task_id = owner_task_key
            try:
                suggestion = generate_followup_text(
                    task, result, get_fast_model()
                )
                if suggestion:  # pragma: no cover — requires LLM API call
                    if _current_db_path() != origin_db_path:
                        return
                    event: dict[str, object] = {
                        "type": "followup_suggestion",
                        "text": suggestion,
                    }
                    self.printer.broadcast(event)
                    _append_chat_event(
                        event,
                        task_id=task_id,
                        task=task,
                        origin_db_path=origin_db_path,
                    )
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
        self, chat_id: str, task_id: str | None, direction: str, tab_id: str = "",
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
            "events": (
                _coalesced_replay_events(result["events"]) if result else []
            ),
            "tabId": tab_id,
        }
        self.printer.broadcast(event)

    def _generate_commit_message(
        self, tab_id: str = "", *, work_dir: str = "",
    ) -> None:
        """Generate a git commit message from current changes.

        Args:
            tab_id: Frontend tab id that requested the message; stamped
                on every emitted ``commitMessage`` event so the
                printer's "system event" routing forwards the message
                only to the originating tab.
            work_dir: The tab's working directory.  Preferred over the
                daemon-wide ``self.work_dir`` because the shared
                ``kiss-web`` daemon may have been launched from (or
                synced to) a different — possibly non-git — folder than
                the window that owns this tab, which would otherwise
                yield a misleading "Not a git repository." error.  Falls
                back to ``self.work_dir`` when empty.
        """
        work_dir = work_dir or self.work_dir
        try:
            from pathlib import Path

            from kiss.agents.sorcar.git_worktree import GitWorktreeOps

            if GitWorktreeOps.discover_repo(Path(work_dir)) is None:
                self.printer.broadcast({
                    "type": "commitMessage",
                    "message": "",
                    "error": "Not a git repository.",
                    "tabId": tab_id,
                })
                return
            cached_result = _git(work_dir, "diff", "--cached")
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



