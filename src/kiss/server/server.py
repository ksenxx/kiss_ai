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
(:mod:`kiss.server.web_server`), which owns the actual I/O
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
import math
import os
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.persistence import (
    _append_chat_event,
    _current_db_path,
    _delete_frequent_task,
    _get_adjacent_task_by_chat_id,
    _history_date_range,
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
from kiss.agents.sorcar.running_agent_state import (
    _RunningAgentState,
    _tab_busy,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import (
    MODEL_INFO,
    get_default_model,
    get_fast_model,
)
from kiss.server.autocomplete import (
    _AutocompleteMixin,
    ranked_function_calling_models,
)
from kiss.server.commands import _CommandsMixin
from kiss.server.diff_merge import (
    _cleanup_merge_data,
    _git,
    _merge_data_dir,
)
from kiss.server.helpers import (
    generate_commit_message_from_diff,
    generate_followup_text,
    model_vendor,
)
from kiss.server.json_printer import JsonPrinter, _coalesce_events
from kiss.server.merge_flow import _MergeFlowMixin
from kiss.server.task_runner import _TaskRunnerMixin, parse_task_tags

__all__ = [
    "VSCodeServer",
    "parse_task_tags",
]

logger = logging.getLogger(__name__)


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
    if not isinstance(extra, str) or not extra:
        return ""
    try:
        parsed = json.loads(extra)
    except (json.JSONDecodeError, TypeError):
        return extra
    if not isinstance(parsed, dict):
        return ""
    if not any(key in parsed for key in _REPLAY_STRIPPED_EXTRA_KEYS):
        return extra
    return json.dumps(
        {key: value for key, value in parsed.items() if key not in _REPLAY_STRIPPED_EXTRA_KEYS}
    )


def _coerce_id(value: object) -> str | None:
    """Coerce a DB row id that may be a str or a legacy int to a string.

    Accepts legacy int ids from databases that escaped the UUID
    auto-migration (r3-vscode-H2 / r4-vscode-H1/H2) and stringifies
    them so the rest of the pipeline works uniformly with string ids.

    Args:
        value: The raw id value read from a DB row or persisted JSON.

    Returns:
        The non-empty string id, or ``None`` when *value* is missing,
        empty, zero, a bool, or of any other type.
    """
    if isinstance(value, str) and value:
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value:
        return str(value)
    return None


def _safe_start_ms(value: object) -> int:
    """Convert a persisted ``timestamp`` (seconds) to epoch milliseconds.

    SQLite's dynamic typing lets the non-STRICT ``REAL NOT NULL``
    timestamp column hold TEXT or non-finite floats in hand-edited or
    third-party-corrupted rows.  A raw ``int(float(value) * 1000)``
    raises ``ValueError``/``TypeError`` on such text and ``OverflowError``
    on infinity, which would abort the entire history response.  This
    helper degrades a single corrupt timestamp to ``0`` instead.

    Args:
        value: The raw ``timestamp`` value read from a history row.

    Returns:
        Epoch milliseconds as an ``int``, or ``0`` when *value* is
        missing, non-numeric, or non-finite.
    """
    try:
        seconds = float(value or 0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(seconds):
        return 0
    try:
        return int(seconds * 1000)
    except (OverflowError, ValueError):
        return 0


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
    evs = cast("list[dict[str, Any]]", events)
    for ev in evs:
        if "ts" not in ev:
            legacy = ev.get("_timestamp")
            if isinstance(legacy, (int, float)) and 0 < legacy <= 8.64e12:
                ev["ts"] = int(legacy * 1000)
    return _coalesce_events(evs)


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
    live_id = getattr(agent, "_last_task_id", None) if agent is not None else None
    if live_id is not None:
        return str(live_id)
    return tab.task_history_id


def broadcast_to_conn(
    printer: Any,
    event: dict[str, Any],
    conn_id: str,
) -> None:
    """Broadcast *event* on *printer*, stamped with *conn_id* when non-empty.

    Stamping ``connId`` makes the printer deliver the event ONLY to the
    requesting connection (the VS Code window / browser tab whose user
    triggered the command), so one window's request never repaints — or
    pops a banner in — another window's UI; ``""`` broadcasts to all.
    Shared by :meth:`VSCodeServer._broadcast_to_conn` and
    ``RemoteAccessServer._broadcast_to_conn`` (web_server.py).

    Args:
        printer: Any printer exposing ``broadcast(event)``.
        event: The event payload to broadcast (mutated in place).
        conn_id: Requesting connection id (``""`` reaches all).
    """
    if conn_id:
        event["connId"] = conn_id
    printer.broadcast(event)


def _subagent_is_done(sub_task_id: Any) -> bool:
    """True when the sub-agent owning *sub_task_id* is no longer running.

    Consults BOTH registries that track a live run, each under its own
    lock, because they are updated at different moments of the task
    lifecycle (S3-09): the agent publishes its ``task_history`` row
    before inserting itself into
    :attr:`ChatSorcarAgent.running_agents`, and at shutdown it pops
    that map before the per-tab :class:`_RunningAgentState` is
    unregistered.  Checking only the map therefore reported a task as
    done during the startup gap while :meth:`_reattach_running_chat`
    (which scans live ``_RunningAgentState`` entries) simultaneously
    reattached it as running.  A task is considered running when its
    id is in the map OR a live tab state (alive worker thread or
    ``is_task_active``) owns the same task id — the same liveness
    predicate reattachment uses.

    Args:
        sub_task_id: The sub-agent's ``task_history`` row id (any type;
            non-str values are treated as done).

    Returns:
        True when no live agent is registered for *sub_task_id*.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    if not (isinstance(sub_task_id, str) and sub_task_id):
        return True
    with ChatSorcarAgent._running_agents_lock:
        if sub_task_id in ChatSorcarAgent.running_agents:
            return False
    with _RunningAgentState._registry_lock:
        for tab in _RunningAgentState.running_agent_states.values():
            if _live_task_id(tab) != sub_task_id:
                continue
            alive = tab.task_thread is not None and tab.task_thread.is_alive()
            if alive or tab.is_task_active:
                return False
    return True


class VSCodeServer(
    _CommandsMixin,
    _TaskRunnerMixin,
    _MergeFlowMixin,
    _AutocompleteMixin,
):
    """Backend server for VS Code extension."""

    _tab_opened_task_ids: dict[str, str] = {}
    _cli_running_lookup: Callable[[str], bool] | None = None
    _orphan_sweep_thread: threading.Thread | None = None

    def __init__(self, printer: JsonPrinter | None = None) -> None:
        self.printer: JsonPrinter = printer or JsonPrinter()
        boot_ts = time.time()
        still_running: set[str] = set()
        with _RunningAgentState._registry_lock:
            for tab in _RunningAgentState.running_agent_states.values():
                thread = tab.task_thread
                if not (tab.is_task_active and thread is not None and thread.is_alive()):
                    continue
                th_id = tab.task_history_id
                if th_id is None and tab.agent is not None:
                    th_id = getattr(tab.agent, "_last_task_id", None)
                if th_id:
                    still_running.add(str(th_id))
            if still_running:
                logger.warning(
                    "New VSCodeServer clearing a registry with %d live "
                    "task(s); their rows are exempt from the orphan "
                    "sweep: %s",
                    len(still_running),
                    ", ".join(sorted(still_running)),
                )
            _RunningAgentState.running_agent_states.clear()
        self._orphan_sweep_thread = threading.Thread(
            target=self._run_orphan_sweep,
            args=(still_running, boot_ts),
            name="orphan-task-sweep",
            daemon=True,
        )
        self._orphan_sweep_thread.start()
        self.work_dir = os.environ.get("KISS_WORKDIR", os.getcwd())
        self._tab_chat_views: dict[str, str] = {}
        self._tab_opened_task_ids: dict[str, str] = {}
        self._pending_user_answer_tasks: dict[int, str] = {}
        persisted = _load_last_model()
        self._default_model = persisted or os.environ.get("KISS_MODEL", "") or get_default_model()
        self._state_lock = _RunningAgentState._registry_lock
        self._complete_seq: int = 0
        self._complete_seq_latest: dict[str, int] = {}
        self._complete_queue: queue.Queue[tuple[str, int, str, str, str, str, str]] | None = None
        self._complete_worker: threading.Thread | None = None
        self._file_cache: dict[str, list[str]] = {}
        self._last_active_file: dict[str, str] = {}
        self._last_active_content: dict[str, str] = {}
        self._cli_running_lookup: Callable[[str], bool] | None = None
        self._cli_running_task_ids_lookup: Callable[[], set[str]] | None = None

    @staticmethod
    def _run_orphan_sweep(still_running: set[str], boot_ts: float) -> None:
        """Run the orphan-task recovery sweep (background thread body).

        Rewrites the ``"Agent Failed Abruptly"`` sentinel on
        ``task_history`` rows abandoned by a prior, now-dead process
        (see :func:`_recover_orphaned_tasks`).  Executed on the
        ``orphan-task-sweep`` daemon thread started by ``__init__`` so
        that SQLite lock contention (``busy_timeout`` is 30 s) or a
        slow sweep over a large database can never delay server
        startup — the UDS / WSS listeners must bind promptly after an
        ``install.sh`` daemon restart.  Best-effort: failures are
        logged and never propagate.

        Because the sweep runs asynchronously, a task can legitimately
        START (inserting a fresh sentinel row via ``_add_task``) after
        ``__init__`` returned but before this thread has executed its
        UPDATE.  Such a row belongs to the LIVE process and is not an
        orphan — rewriting it would mislabel a running task as
        ``"Task terminated unexpectedly (process killed)"`` and defeat
        the pre-emptive shutdown persistence in
        :meth:`RemoteAccessServer._stop_active_agent_tasks` (which
        conditions on the sentinel still being present).  The *boot_ts*
        cut-off scopes the sweep to rows created strictly before this
        server instance was constructed.

        Args:
            still_running: Task-history row ids owned by worker
                threads still alive in this process; exempt from the
                sweep.
            boot_ts: Epoch-seconds timestamp captured in ``__init__``
                before this thread was spawned.  Only rows whose
                ``timestamp`` column is strictly older are eligible
                for the sweep.
        """
        from kiss.agents.sorcar.persistence import _close_thread_db

        try:
            _recover_orphaned_tasks(still_running, created_before=boot_ts)
        except Exception:  # pragma: no cover — best-effort sweep
            logger.exception(
                "orphan-task recovery sweep failed; continuing startup",
            )
        finally:
            _close_thread_db()

    def set_cli_running_lookup(
        self,
        lookup: Callable[[str], bool] | None,
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
        self,
        lookup: Callable[[], set[str]] | None,
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
            if tab.agent is None:
                agent = WorktreeSorcarAgent("Sorcar VS Code")
                if tab.chat_id:
                    agent._chat_id = tab.chat_id
                tab.agent = agent
            return tab

    def _any_non_wt_running(self, repo_root: Path | None = None) -> bool:
        """True if a non-worktree task is running on *repo_root*'s main tree.

        Must be called with ``_state_lock`` held.

        A non-worktree task only occupies the main working tree of the
        repository its ``work_dir`` resolves into (recorded on the tab
        as ``non_wt_repo_root`` when the task starts).  A task running
        in a *different* repository, in a non-git directory, or inside
        a linked ``.kiss-worktrees`` worktree (whose ``git rev-parse
        --show-toplevel`` is the worktree itself, not the main tree)
        never touches *repo_root*'s main working tree, so it must not
        block worktree merges there.

        Args:
            repo_root: The main repository root the caller is about to
                stash/checkout/merge.  ``None`` means "any main tree"
                and preserves the conservative pre-repo-aware behavior
                (used when the caller cannot name its repository).

        Returns:
            True if at least one tab is running a non-worktree task
            whose main working tree is *repo_root* (or, when
            *repo_root* is ``None``, any non-worktree task at all).
        """
        for t in _RunningAgentState.running_agent_states.values():
            if not t.is_running_non_wt:
                continue
            if repo_root is None:
                return True
            t_root = t.non_wt_repo_root
            if t_root is None:
                # The non-worktree task is not inside any git repo, so
                # it cannot be modifying repo_root's main working tree.
                continue
            try:
                if t_root.resolve() == repo_root.resolve():
                    return True
            except OSError:  # pragma: no cover — unresolvable path
                return True
        return False

    def _handle_command(self, cmd: dict[str, Any]) -> None:
        """Dispatch a command from VS Code to the appropriate handler."""
        for field in ("tabId", "workDir", "connId"):
            value = cmd.get(field)
            if value is not None and not isinstance(value, str):
                cmd[field] = ""
        cmd_type = cmd.get("type", "")
        handler = self._HANDLERS.get(cmd_type) if isinstance(cmd_type, str) else None
        if handler is not None:
            handler(self, cmd)
        else:
            event: dict[str, Any] = {"type": "error", "text": f"Unknown command: {cmd_type}"}
            tab_id = cmd.get("tabId")
            if tab_id is not None:
                event["tabId"] = tab_id
            self._broadcast_to_conn(event, cmd.get("connId", ""))

    def _broadcast_to_conn(
        self,
        event: dict[str, Any],
        conn_id: str,
    ) -> None:
        """Broadcast *event*, stamped with *conn_id* when non-empty.

        Args:
            event: The event payload to broadcast (mutated in place).
            conn_id: Requesting connection id (``""`` reaches all).
        """
        broadcast_to_conn(self.printer, event, conn_id)

    def _refresh_default_model(self, valid: set[str] | None = None) -> None:
        """Re-read the persisted last model and adopt it as the default.

        ``kiss-web`` can outlive VS Code windows.  A fresh VS Code
        activation asks this long-lived daemon for ``getModels``; if
        the user selected a different model in a previous window
        session, that choice is persisted in ``config.json`` and must
        take precedence over this process's stale in-memory default.

        The persisted value is read INSIDE ``_state_lock`` (an RLock,
        so callers already holding it may call this freely) so a
        concurrent ``_cmd_select_model`` — which persists under the
        same lock — cannot leave us with a stale on-disk value that
        would clobber the user's just-picked in-memory selection.

        Args:
            valid: When given, adopt the persisted model only when it
                is in this set of currently-runnable model names.
        """
        with self._state_lock:
            persisted = _load_last_model()
            if persisted and (valid is None or persisted in valid):
                self._default_model = persisted

    def _printer_cleanup_tab(self, tab_id: str) -> None:
        """Drop the printer's per-tab subscriptions/state for *tab_id*.

        Resolved via ``getattr`` because some duck-typed test printers
        implement only the broadcast/subscribe subset of the printer
        protocol.

        Args:
            tab_id: The frontend tab identifier to clean up.
        """
        cleanup_tab = getattr(self.printer, "cleanup_tab", None)
        if cleanup_tab is not None:
            cleanup_tab(tab_id)

    def _printer_close_owner_ui(self, tab_id: str) -> None:
        """Take *tab_id*'s interactive UI off the other clients' screens.

        Resolved via ``getattr`` for the same reason as
        :meth:`_printer_cleanup_tab`.

        Args:
            tab_id: The frontend tab being disposed.
        """
        close_owner_ui = getattr(self.printer, "close_owner_ui", None)
        if close_owner_ui is not None:
            close_owner_ui(tab_id)

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
        for name in ranked_function_calling_models():
            info = MODEL_INFO[name]
            models_list.append(
                {
                    "name": name,
                    "inp": info.input_price_per_1M,
                    "out": info.output_price_per_1M,
                    "uses": usage.get(name, 0),
                    "vendor": model_vendor(name)[0],
                }
            )

        from kiss.core.vscode_config import get_custom_model_entry, load_config

        cfg = load_config()
        custom = get_custom_model_entry(cfg)
        if custom:
            models_list.insert(0, custom)

        available_names = {m["name"] for m in models_list}
        with self._state_lock:
            self._refresh_default_model(available_names)

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
        self._broadcast_to_conn(event, conn_id)

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
                if tid is not None and tab.task_thread is not None and tab.task_thread.is_alive():
                    running.add(tid)
        if self._cli_running_task_ids_lookup is not None:
            try:
                running.update(self._cli_running_task_ids_lookup())
            except Exception:  # pragma: no cover — defensive
                logger.exception(
                    "cli running-task lookup failed; continuing",
                )
        return running

    def _overlay_live_metrics(
        self,
        session: dict[str, Any],
        task_id: str,
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
                session["tokens"] = int(getattr(agent, "total_tokens_used", 0) or 0)
                session["cost"] = float(getattr(agent, "budget_used", 0.0) or 0.0)
                steps = int(getattr(agent, "total_steps", 0) or 0)
                cur = getattr(agent, "_current_executor", None)
                if cur is not None:
                    steps += int(getattr(cur, "step_count", 0) or 0)
                session["steps"] = steps
                mdl_live = getattr(agent, "model_name", "") or getattr(tab, "selected_model", "")
                if isinstance(mdl_live, str) and mdl_live:
                    session["model"] = mdl_live
                session["is_worktree"] = bool(getattr(tab, "use_worktree", False))
                session["is_parallel"] = bool(getattr(tab, "use_parallel", False))
                session["auto_commit_mode"] = bool(getattr(tab, "auto_commit_mode", False))
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
            entry_id = _coerce_id(entry.get("id"))
            is_running = entry_id is not None and entry_id in running_task_ids
            session: dict[str, Any] = {
                "id": chat_id,
                "task_id": entry_id,
                "title": task,
                "timestamp": entry.get("timestamp", 0),
                "preview": task,
                "has_events": has_events,
                "failed": _is_failed_result(result) and not is_running,
                "is_running": is_running,
                "tokens": 0,
                "cost": 0.0,
                "steps": 0,
                "is_favorite": False,
                "work_dir": "",
                "model": "",
                "is_worktree": False,
                "is_parallel": False,
                "auto_commit_mode": False,
                "startTs": _safe_start_ms(entry.get("timestamp", 0)),
                "endTs": 0,
            }
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
                        pid = _coerce_id(sub.get("parent_task_id"))
                        if pid is not None:
                            session["parent_task_id"] = pid
                    # ``OverflowError`` must be caught alongside the
                    # usual coercion errors: Python's JSON parser
                    # accepts ``Infinity``/huge numbers in hand-edited
                    # ``extra`` payloads and one corrupt row must not
                    # abort the entire history response (S3-13/R7).
                    for key, cast, default in (
                        ("tokens", int, 0),
                        ("cost", float, 0.0),
                        ("steps", int, 0),
                    ):
                        try:
                            session[key] = cast(extra_obj.get(key, default) or default)
                        except (TypeError, ValueError, OverflowError):
                            session[key] = default
                    try:
                        session["endTs"] = int(extra_obj.get("endTs", 0) or 0)
                    except (TypeError, ValueError, OverflowError):
                        session["endTs"] = 0
                    session["is_favorite"] = bool(extra_obj.get("is_favorite", False))
                    wd_raw = extra_obj.get("work_dir", "")
                    if isinstance(wd_raw, str):
                        session["work_dir"] = wd_raw
                    mdl_raw = extra_obj.get("model", "")
                    if isinstance(mdl_raw, str):
                        session["model"] = mdl_raw
                    session["is_worktree"] = bool(extra_obj.get("is_worktree", False))
                    session["is_parallel"] = bool(extra_obj.get("is_parallel", False))
                    session["auto_commit_mode"] = bool(extra_obj.get("auto_commit_mode", False))
                    try:
                        start_ts_raw = extra_obj.get("startTs", 0)
                        if start_ts_raw:
                            session["startTs"] = int(start_ts_raw)
                    except (TypeError, ValueError, OverflowError):
                        pass
            if session.get("is_running") and entry_id is not None:
                self._overlay_live_metrics(session, entry_id)
            sessions.append(session)
        min_ts, max_ts = _history_date_range()
        event: dict[str, Any] = {
            "type": "history",
            "sessions": sessions,
            "offset": offset,
            "generation": generation,
            "dateRange": {"min": min_ts, "max": max_ts},
        }
        self._broadcast_to_conn(event, conn_id)

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
        self._broadcast_to_conn(event, conn_id)

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
        self._broadcast_to_conn(event, conn_id)

    def _close_tab(self, tab_id: str) -> None:
        """Clean up all backend state for a closed tab.

        Removes the tab from
        ``_RunningAgentState.running_agent_states``, cleans up per-tab printer
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
        self,
        tab_id: str,
        tab: _RunningAgentState | None,
    ) -> None:
        """Release worktree, per-tab printer state and merge data dir.

        Shared cleanup tail used by both the immediate (:meth:`_close_tab`)
        and the deferred (:meth:`_dispose_if_closed`) disposal paths.
        Caller must have already popped *tab* from
        ``_RunningAgentState.running_agent_states``.

        Retiring the worktree here can strand work — a rejected
        pre-commit hook leaves the changes in the worktree directory,
        and a conflicting merge leaves them on the branch — and the
        agent records where to find them as a pending warning.  Those
        warnings are flushed before the printer is torn down, because
        after that there is nothing left to say it on and the user
        would never learn their work survived.

        Args:
            tab_id: The frontend tab identifier being disposed.
            tab: The popped tab state, or ``None`` when the tab was
                never created (e.g. ``closeTab`` for an unknown id).
        """
        if tab is not None:
            try:
                wt_agent = self._ensure_wt_agent(tab)
                if wt_agent is not None and wt_agent._wt_pending:
                    if getattr(wt_agent, "_pending_review", False):
                        wt_agent._preserve_pending_worktree_for_review()
                    else:
                        wt_agent._release_worktree()
                    wt_agent._flush_warnings(self.printer)
            except Exception:
                logger.debug("Worktree release on tab close failed", exc_info=True)
        self._printer_close_owner_ui(tab_id)
        self._printer_cleanup_tab(tab_id)
        with self._state_lock:
            self._tab_chat_views.pop(tab_id, None)
            self._tab_opened_task_ids.pop(tab_id, None)
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
            logger.debug("newChat ignored: empty tabId")
            return
        tab = self._get_tab(tab_id)
        with self._state_lock:
            self._refresh_default_model()
            tab.selected_model = self._default_model
            tab.chat_id = ""
            tab.last_task_id = None
            self._tab_chat_views.pop(tab_id, None)
            self._tab_opened_task_ids.pop(tab_id, None)
            welcome_model = self._default_model
        self._printer_cleanup_tab(tab_id)
        self.printer.broadcast(
            {
                "type": "showWelcome",
                "tabId": tab_id,
                "model": welcome_model,
            }
        )

    def _replay_session(
        self,
        chat_id: str,
        tab_id: str = "",
        task_id: str | None = None,
    ) -> None:
        """Replay recorded chat events for a previous chat session.

        Sets the tab's agent chat_id to match the resumed session.
        The tab_id (frontend key in
        ``_RunningAgentState.running_agent_states``) does not change.

        When ``tab_id`` is empty the call is a no-op — the previous
        behavior of synthesizing a phantom tab keyed by ``chat_id`` and
        mutating its ``use_worktree`` flag violated per-tab state
        isolation (C2/C3 fix).

        Loading a chat never touches the tab's ``use_worktree`` /
        ``use_parallel`` / ``auto_commit_mode`` / ``selected_model``:
        those mirror the toolbar toggles, which are global UI state the
        user owns.  Clearing them made a history click silently switch
        auto-commit off, which in turn made the pending-worktree
        handling below present a diff/merge review the user had opted
        out of.

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
        with self._state_lock:
            if task_id:
                self._tab_opened_task_ids[tab_id] = str(task_id)
            else:
                self._tab_opened_task_ids.pop(tab_id, None)
        result = None
        if task_id is not None:
            result = _load_chat_events_by_task_id(task_id)
            if result:
                chat_id = str(result.get("chat_id", "") or chat_id)
        if not result:
            result = _load_latest_chat_events_by_chat_id(chat_id)
        if not result:
            self._printer_cleanup_tab(tab_id)
            rebound_running = self._reattach_running_chat(
                chat_id,
                tab_id,
                task_id=task_id,
                is_subagent=False,
            )
            if rebound_running:
                start_ts = self._live_task_start_ms(task_id, chat_id)
                self.printer.broadcast(
                    {
                        "type": "status",
                        "running": True,
                        "tabId": tab_id,
                        "startTs": start_ts,
                    }
                )
                self.printer.broadcast(
                    {
                        "type": "task_events",
                        "events": [],
                        "task": "",
                        "task_id": task_id,
                        "chat_id": chat_id,
                        "extra": "",
                        "tabId": tab_id,
                    }
                )
            with self._state_lock:
                tab = _RunningAgentState.running_agent_states.get(tab_id)
                is_sub_view = tab is not None and tab.is_subagent
                if tab is not None:
                    if chat_id:
                        tab.chat_id = chat_id
                    tab.frontend_closed = False
                if chat_id and not is_sub_view:
                    self._tab_chat_views[tab_id] = chat_id
            return

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

        rebound_task_id = _coerce_id(result.get("task_id") if result else None)
        self._printer_cleanup_tab(tab_id)
        rebound_running = self._reattach_running_chat(
            chat_id,
            tab_id,
            task_id=rebound_task_id,
            is_subagent=subagent_info is not None,
        )
        if (
            not rebound_running
            and rebound_task_id is not None
            and self._cli_running_lookup is not None
            and self._cli_running_lookup(rebound_task_id)
        ):
            self.printer.subscribe_tab(str(rebound_task_id), tab_id)
            rebound_running = True
        if (
            not rebound_running
            and rebound_task_id is not None
            and self.printer.has_ui_mirror_for_task(rebound_task_id)
        ):
            # The task is over but another window is still holding a
            # merge review or an auto-commit prompt open on it.
            # Joining replays that UI into this tab, so opening the
            # chat here shows the same question instead of nothing.
            self.printer.subscribe_tab(str(rebound_task_id), tab_id)

        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is not None:
                tab.chat_id = chat_id
                tab.frontend_closed = False
            if subagent_info is None and chat_id:
                self._tab_chat_views[tab_id] = chat_id
            else:
                self._tab_chat_views.pop(tab_id, None)

        if subagent_info is not None:
            is_done = _subagent_is_done(result.get("task_id"))
            parent_tid = _coerce_id(subagent_info.get("parent_task_id"))
            parent_tab_id_for_sub = self._resolve_parent_tab_id_for_sub(
                parent_task_id=parent_tid,
                chat_id=chat_id,
                sub_tab_id=tab_id,
            )
            self.printer.broadcast(
                {
                    "type": "openSubagentTab",
                    "tab_id": tab_id,
                    "parent_tab_id": parent_tab_id_for_sub,
                    "description": str(result.get("task", "") or ""),
                    "task_id": result.get("task_id"),
                    "isSubagentTab": True,
                    "isDone": is_done,
                }
            )

        if rebound_running:
            start_ts_for_resume = 0
            if isinstance(extra_raw, dict):
                try:
                    start_ts_for_resume = int(extra_raw.get("startTs", 0) or 0)
                except (TypeError, ValueError):
                    start_ts_for_resume = 0
            if start_ts_for_resume <= 0:
                start_ts_for_resume = self._live_task_start_ms(
                    rebound_task_id,
                    chat_id,
                )
            self.printer.broadcast(
                {
                    "type": "status",
                    "running": True,
                    "tabId": tab_id,
                    "startTs": start_ts_for_resume,
                }
            )
        self.printer.broadcast(
            {
                "type": "task_events",
                "events": _coalesced_replay_events(result["events"]),
                "task": result["task"],
                "task_id": result.get("task_id"),
                "chat_id": chat_id,
                "extra": _extra_for_replay(result.get("extra", "")),
                "tabId": tab_id,
            }
        )
        self._emit_pending_worktree(tab_id)

        if subagent_info is None and isinstance(rebound_task_id, str) and rebound_task_id:
            self._open_persisted_subagent_tabs(
                parent_task_id=rebound_task_id,
                parent_tab_id=tab_id,
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
                st for st in _RunningAgentState.running_agent_states.values() if not st.is_subagent
            ]

            if parent_task_id is not None:
                for st in non_sub_states:
                    if _live_task_id(st) == parent_task_id:
                        return st.tab_id

            if chat_id:
                chat_matches = [
                    st for st in non_sub_states if st.chat_id == chat_id and st.tab_id != sub_tab_id
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
            sub_tab_id,
            parent_task_id,
            chat_id,
        )
        return ""

    def _open_persisted_subagent_tabs(
        self,
        *,
        parent_task_id: str,
        parent_tab_id: str,
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
            if not is_done:
                self._reattach_running_chat(
                    str(row.get("chat_id", "") or ""),
                    sub_tab_id,
                    task_id=str(sub_task_id),
                    is_subagent=True,
                )
            self.printer.broadcast(
                {
                    "type": "openSubagentTab",
                    "tab_id": sub_tab_id,
                    "parent_tab_id": parent_tab_id,
                    "description": description,
                    "task_id": sub_task_id,
                    "taskIndex": idx,
                    "isSubagentTab": True,
                    "isDone": is_done,
                }
            )
            self.printer.broadcast(
                {
                    "type": "task_events",
                    "events": _coalesced_replay_events(row["events"]),
                    "task": description,
                    "task_id": sub_task_id,
                    "chat_id": row.get("chat_id", ""),
                    "extra": _extra_for_replay(row.get("extra", "")),
                    "tabId": sub_tab_id,
                }
            )
            if not is_done and _subagent_is_done(sub_task_id):
                self.printer.broadcast(
                    {
                        "type": "subagentDone",
                        "tab_id": sub_tab_id,
                        "tabId": "",
                    }
                )

    def _live_task_start_ms(
        self,
        task_id: str | None,
        chat_id: str,
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
                elif not chat_id or tab.chat_id != chat_id or tab.is_subagent:
                    continue
                start_ms = int(getattr(tab.agent, "_task_start_ms", 0) or 0)
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
            if task_id is not None:
                for t in _RunningAgentState.running_agent_states.values():
                    if _live_task_id(t) != task_id:
                        continue
                    alive = t.task_thread is not None and t.task_thread.is_alive()
                    if alive or t.is_task_active:
                        source = t
                        break
            if source is None and chat_id and not is_subagent:
                for t in _RunningAgentState.running_agent_states.values():
                    if t.chat_id != chat_id:
                        continue
                    if t.is_subagent:
                        continue
                    alive = t.task_thread is not None and t.task_thread.is_alive()
                    if alive or t.is_task_active:
                        source = t
                        break
            if source is None:
                return False
            source_task_id = _live_task_id(source)
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
        owner_task_key = str(task_id) if task_id is not None else None
        origin_db_path = _current_db_path()

        def _run() -> None:
            if owner_task_key is not None:
                self.printer._thread_local.task_id = owner_task_key
            try:
                suggestion = generate_followup_text(task, result, get_fast_model())
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
            finally:
                # The task's subscriber set was kept alive (a bounded
                # linger) solely so this broadcast could still fan out
                # after ``cleanup_task``.  The follow-up is the last
                # post-task event, so release the lease as soon as it
                # is delivered (or failed) instead of waiting out the
                # full linger.
                if owner_task_key is not None:
                    try:
                        self.printer.cleanup_task(
                            owner_task_key, subscriber_linger_seconds=0,
                        )
                    except Exception:
                        logger.debug(
                            "Follow-up subscriber release failed",
                            exc_info=True,
                        )

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
        self,
        chat_id: str,
        task_id: str | None,
        direction: str,
        tab_id: str = "",
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
            "events": (_coalesced_replay_events(result["events"]) if result else []),
            "tabId": tab_id,
        }
        self.printer.broadcast(event)

    def _generate_commit_message(
        self,
        tab_id: str = "",
        *,
        work_dir: str = "",
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
                self.printer.broadcast(
                    {
                        "type": "commitMessage",
                        "message": "",
                        "error": "Not a git repository.",
                        "tabId": tab_id,
                    }
                )
                return
            cached_result = _git(work_dir, "diff", "--cached")
            diff_text = cached_result.stdout.strip()
            if not diff_text:  # pragma: no branch — LLM API required for else
                self.printer.broadcast(
                    {
                        "type": "commitMessage",
                        "message": "",
                        "error": "No staged changes found. Stage files with 'git add' first.",
                        "tabId": tab_id,
                    }
                )
                return
            msg = generate_commit_message_from_diff(diff_text)  # pragma: no cover
            self.printer.broadcast(
                {
                    "type": "commitMessage",
                    "message": msg,
                    "tabId": tab_id,
                }
            )  # pragma: no cover
        except Exception:  # pragma: no cover — LLM API error handler
            logger.debug("Commit message generation failed", exc_info=True)
            self.printer.broadcast(
                {
                    "type": "commitMessage",
                    "message": "",
                    "error": "Failed to generate",
                    "tabId": tab_id,
                }
            )
