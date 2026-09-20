# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""VS Code extension backend server for Sorcar agent.

The per-command handlers, task-runner, worktree flow and
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
import shutil
import threading
import time
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.persistence import (
    _chat_first_tasks,
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
from kiss.core import config as config_module
from kiss.core.models.model_info import (
    MODEL_INFO,
    get_default_model,
    list_custom_models,
)
from kiss.core.utils import is_root_dir
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.autocomplete import (
    _AutocompleteMixin,
    ranked_function_calling_models,
)
from kiss.server.commands import _CommandsMixin
from kiss.server.diff_merge import _git
from kiss.server.helpers import (
    generate_commit_message_from_diff,
    model_vendor,
)
from kiss.server.json_printer import (
    JsonPrinter,
    _coalesce_events,
    with_task_settings_event,
)
from kiss.server.merge_flow import _MergeFlowMixin
from kiss.server.tab_registry import TabRegistry
from kiss.server.task_runner import (
    _subtask_metrics,
    _TaskRunnerMixin,
    parse_task_tags,
)

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


def _start_ts_from_extra(extra: object) -> int:
    """Return the ``startTs`` (ms since the epoch) persisted in *extra*.

    A task row's ``extra`` JSON carries the run's wall-clock start
    (stamped by ``ChatSorcarAgent.run`` when it allocated the row, or
    by the task runner's own start for a daemon-run task — both fall
    inside the parent's tool call for a sub-agent).  The frontend
    attributes a re-announced sub-agent to the fan-out call
    (``run_parallel`` / ``run_agent`` tool call) that was running when
    it started, so ``openSubagentTab`` announcements carry this stamp.

    Args:
        extra: The persisted ``extra`` value (a JSON string), or an
            already parsed dict.

    Returns:
        The start stamp, or ``0`` when *extra* has none or is
        malformed.
    """
    parsed: object = extra
    if isinstance(extra, str):
        if not extra:
            return 0
        try:
            parsed = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            return 0
    if not isinstance(parsed, dict):
        return 0
    try:
        start_ts = int(parsed.get("startTs", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return start_ts if start_ts > 0 else 0


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

    A task is running while its registered agent state is still active
    or its worker thread is alive — the same liveness predicate
    reattachment uses.

    Args:
        sub_task_id: The sub-agent's ``task_history`` row id (any type;
            non-str values are treated as done).

    Returns:
        True when no live agent is registered for *sub_task_id*.
    """
    if not (isinstance(sub_task_id, str) and sub_task_id):
        return True
    with agent_state.STATE_LOCK:
        state = agent_state.get(sub_task_id)
        return state is None or not (state.is_task_active or state.thread_alive())


def _cleanup_legacy_merge_artifacts() -> None:
    """Delete review snapshots left behind by the removed diff review.

    Prior releases snapshotted dirty and untracked files (up to 2 MB
    each) under ``{artifact_root}/merge_dir/<tab>/`` while preparing
    the interactive diff/merge review, and deleted them when each
    review ended.  With the review workflow removed, nothing writes —
    or would ever delete — that tree, so an upgrade (or a restart
    mid-review) would strand potentially sensitive file copies
    forever.  Removing the whole directory once at server construction
    retires the legacy data.
    """
    legacy = config_module._artifact_root() / "merge_dir"
    try:
        shutil.rmtree(legacy)
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("Legacy merge_dir cleanup failed", exc_info=True)


def _prewarm_task_dependencies() -> None:
    """Warm the lazily imported modules a first task would pay for.

    The first task after a daemon start used to spend several seconds
    between its ``task_history`` row allocation and the agent's first
    event: ``KISSAgent._reset`` imports the concrete model class (and
    its provider SDK) on demand, and ``_run_task_inner`` loads the
    VS Code config module lazily.  Importing them here — on a
    background daemon thread started from ``VSCodeServer.__init__`` —
    moves that cost to server startup, where nobody is waiting on it.

    Best-effort: a missing optional SDK or an import error must never
    affect server startup, so every import failure is only logged.
    """
    import importlib

    for mod in (
        "kiss.core.models.anthropic_model",
        "kiss.core.models.openai_compatible_model",
        "kiss.core.models.openai_compatible_model2",
        "kiss.core.models.gemini_model",
        "kiss.core.vscode_config",
    ):
        try:
            importlib.import_module(mod)
        except Exception:
            logger.debug("Prewarm import of %s failed", mod, exc_info=True)
    try:
        from kiss.core.models.model_info import get_available_models

        get_available_models()
    except Exception:
        logger.debug("Prewarm of the model registry failed", exc_info=True)


class MainTreeClaim:
    """One published main-tree mutation claim (see ``_claim_main_tree``).

    Carries the identity needed for interrupt-safe release: ``owner``
    lets the admission check and later claimants heal a claim whose
    thread died without releasing (an injected stop skipped its
    ``finally``), and the object's own identity makes release
    conditional — a stale cleanup can never pop a successor's claim.
    """

    __slots__ = ("key", "reason", "owner")

    def __init__(self, key: Path, reason: str, owner: threading.Thread) -> None:
        self.key = key
        self.reason = reason
        self.owner = owner


class VSCodeServer(
    _CommandsMixin,
    _TaskRunnerMixin,
    _MergeFlowMixin,
    _AutocompleteMixin,
):
    """Backend server for VS Code extension."""

    _orphan_sweep_thread: threading.Thread | None = None

    def __init__(self, printer: JsonPrinter | None = None) -> None:
        self.printer: JsonPrinter = printer or JsonPrinter()
        _cleanup_legacy_merge_artifacts()
        boot_ts = time.time()
        still_running: set[str] = set()
        # ``agent_states`` is process-global, so this constructor must
        # evict only the finished states a previous server left behind:
        # clearing the whole registry detached the live tasks of a
        # server that is still serving, orphaning their worktrees and
        # stranding their history rows (F08-5).
        with agent_state.STATE_LOCK:
            for task_id, state in list(agent_state.agent_states.items()):
                if state.busy():
                    still_running.add(task_id)
                else:
                    del agent_state.agent_states[task_id]
            if still_running:
                logger.warning(
                    "New VSCodeServer kept %d live task(s) registered; "
                    "their rows are exempt from the orphan sweep: %s",
                    len(still_running),
                    ", ".join(sorted(still_running)),
                )
        self._orphan_sweep_thread = threading.Thread(
            target=self._run_orphan_sweep,
            args=(still_running, boot_ts),
            name="orphan-task-sweep",
            daemon=True,
        )
        self._orphan_sweep_thread.start()
        threading.Thread(
            target=_prewarm_task_dependencies,
            name="task-dependency-prewarm",
            daemon=True,
        ).start()
        # The daemon-wide last-resort work dir.  A GUI-launched daemon
        # (launchd/systemd, `open -a`) can inherit cwd `/`, and a
        # poisoned environment can carry a root in KISS_WORKDIR;
        # falling back to a filesystem root would let every unstamped
        # command (and the @-mention file scan) span the whole disk,
        # so degrade to the user's home directory instead.
        _fallback_wd = os.environ.get("KISS_WORKDIR") or os.getcwd()
        if is_root_dir(_fallback_wd):
            _fallback_wd = os.path.expanduser("~")
        self.work_dir = _fallback_wd
        # The canonical shared tab registry (mirrored by every client).
        # The path is resolved through the persistence module's
        # redirectable KISS dir so tests point it at a scratch home.
        self.tab_registry = TabRegistry(
            Path(_persistence._KISS_DIR) / "tabs.json",
        )
        # The printer's local-UDS talk bookkeeping only knows which
        # connection addressed which tab and whether a chat webview is
        # attached; whether a tab is SHOWN by a local webview is decided
        # at talk time by ``_local_tab_shown`` from the canonical facts
        # (duck-typed: only the daemon's ``WebPrinter`` has the hook).
        # The bound method reads ``self.tab_registry`` on every call,
        # so ``use_private_tab_registry`` swaps are honoured.
        install_visibility = getattr(self.printer, "set_local_tab_visibility", None)
        if install_visibility is not None:
            install_visibility(self._local_tab_shown)
        self._tab_chat_views: dict[str, str] = {}
        # Rebind surviving chat views from the persisted registry so a
        # follow-up ``run`` after a daemon restart continues the tab's
        # chat instead of silently starting a fresh one.
        self._tab_chat_views.update(self.tab_registry.bindings())
        self._tab_opened_task_ids: dict[str, str] = {}
        self._tab_models: dict[str, str] = {}
        self._commit_msg_tabs: set[str] = set()
        self._autocommit_tabs: set[str] = set()
        # Per-repository main-tree mutation claims (resolved repo root
        # → ``MainTreeClaim``), guarded by ``_state_lock``.  A
        # main-tree mutator (Discard, manual Git Commit) publishes its
        # claim in the SAME locked section as its
        # ``_any_non_wt_running`` busy check, and non-worktree task
        # admission refuses to start while a claim is held — closing
        # the check-without-claim TOCTOU in which a direct task could
        # start (and begin writing) between the mutator's busy check
        # and its ``git reset``/``git add`` (gpt-5.6-sol review,
        # findings 2 and 3).
        self._main_tree_claims: dict[Path, MainTreeClaim] = {}
        persisted = _load_last_model()
        self._default_model = persisted or os.environ.get("KISS_MODEL", "") or get_default_model()
        self._state_lock = agent_state.STATE_LOCK
        # Raised (under ``_state_lock``) by the graceful-shutdown
        # sweep (``RemoteAccessServer._stop_active_agent_tasks``):
        # from then on ``_cmd_run``'s pre-start handshake cancels new
        # runs instead of starting their worker threads, so no run
        # can start AFTER the sweep and execute untrusted setup with
        # no watchdog (audit0903 F1).
        self._shutdown_stopping: bool = False
        self._complete_seq: int = 0
        self._complete_seq_latest: dict[str, int] = {}
        self._complete_queue: (
            queue.Queue[tuple[str, int, str, str | None, str, str, str]] | None
        ) = None
        self._complete_worker: threading.Thread | None = None
        self._file_cache: dict[str, list[str]] = {}
        self._last_active_file: dict[str, str] = {}
        self._last_active_content: dict[str, str] = {}

    def use_private_tab_registry(self, path: Path) -> None:
        """Own a private tab registry at *path* instead of the canonical one.

        The constructor binds every server to ``KISS_HOME/tabs.json``,
        the registry all clients mirror.  That file has exactly ONE
        owner — a registry loads it once and publishes its complete
        in-memory list on every mutation, so a second live registry on
        the same file overwrites the first one's tabs with its stale
        snapshot.  An EMBEDDED server that shares the KISS home with
        the canonical daemon (the channel launcher's private-UDS
        daemon, ``_kiss_web_launcher._ensure_api_server``) therefore
        keeps its transient tabs in a registry of its own.

        Call before serving any client.  The chat views rebound from
        the canonical registry in the constructor are replaced by
        *path*'s, so a run on this server never continues a canonical
        tab's chat.

        Args:
            path: The JSON file backing this server's registry (a
                fresh or launcher-private location, never the
                canonical daemon's ``tabs.json``).
        """
        self.tab_registry = TabRegistry(path)
        self._tab_chat_views = dict(self.tab_registry.bindings())

    def _local_tab_shown(
        self, tab_id: str, interested: bool, webview_attached: bool,
    ) -> bool:
        """Decide whether a local UDS webview shows *tab_id* right now.

        The rule behind ``WebPrinter.shown_local_uds_tabs`` — the
        daemon-native talk playback decision.  It is evaluated at talk
        time from the canonical facts instead of from bookkeeping that
        mirrors them, which is what makes the decision immune to the
        interleavings of a close with a re-registration, a ``ready``
        sync or a ``resumeSession`` reopen: no copy exists that could
        go stale.  *tab_id* is always a talk target, i.e. a tab the
        printer currently has subscribed to a live task.

        * A tab listed in the canonical registry is shown by EVERY
          attached chat webview (clients mirror the whole registry
          from ``tabs_state``), so it counts as soon as a webview is
          attached — per-connection interest is irrelevant, which is
          why a prune racing a reopen cannot hide the reopened tab.
        * Otherwise some UDS peer must have addressed the tab
          (*interested*).  If the tab runs a task of its own — a
          ``run_agent`` dispatch's ``api-…`` tab, a ``run_parallel``
          child's synthetic tab, a placeholder the registry refused at
          its cap, or a registry tab with no webview attached whose
          headless owner (``daemon_client.run``) is still connected —
          it counts while that state is not ``frontend_closed``, the
          flag every close path (``_close_tab``, displacement, the
          deferred ``_dispose_if_closed``) raises and every reopen
          (``_replay_session``) clears.
        * A tab with no state of its own is a VIEWER of another tab's
          task (a sub-agent tab the client opened for a running child,
          a tab resuming a running chat before its registry
          publication lands): being a talk target proves it is still
          subscribed, and closing a viewer unsubscribes it
          (``_teardown_tab_resources`` → ``cleanup_tab``), so it counts
          unless it is a registry tab that no webview shows.

        Args:
            tab_id: The frontend tab identifier a talk event targets.
            interested: Some UDS connection addressed *tab_id*.
            webview_attached: Some UDS connection hosts a chat webview.
        """
        in_registry = self.tab_registry.has_tab(tab_id)
        if in_registry and webview_attached:
            return True
        if not interested:
            return False
        with self._state_lock:
            state = agent_state.find_by_tab(tab_id)
            if state is not None:
                return not state.frontend_closed
        return not in_registry

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
            # The file-picker request token is normally popped by the
            # scan that answers it, but a scan that failed (A-C1) or
            # is still in flight when the window closes would leave
            # the departed connection's entry behind forever.
            self._files_request_map().pop(conn_id, None)

    def _broadcast_tabs_state(self) -> None:
        """Broadcast the canonical tab snapshot to every client.

        Emitted after every registry mutation.  Clients reconcile
        their local tab bar against the full snapshot (idempotent and
        self-healing), so deltas are never needed.  The explicit empty
        ``tabId`` stamp routes the event through the printer's
        verbatim all-clients path — never recorded or persisted, and
        immune to the thread-local task-id injection when a mutation
        happens inside a task thread.
        """
        self.printer.broadcast({
            "type": "tabs_state",
            "tabs": self.tab_registry.snapshot(),
            "tabId": "",
        })

    def _registry_update_tab(
        self,
        tab_id: str,
        *,
        chat_id: str | None = None,
        title: str | None = None,
        work_dir: str | None = None,
        scope_work_dir: str | None = None,
        task_id: str | None = None,
        create: bool = False,
    ) -> int:
        """Update the shared registry and broadcast when it changed.

        Binding a chat displaces any other tab bound to the same chat
        (the registry enforces the one-tab-per-chat invariant); the
        displaced tabs' server-side state is released here exactly as
        an explicit ``closeTab`` would.

        Returns:
            This publication's OWN generation token, captured inside
            the registry's locked update (``0`` when nothing was
            published).  A caller that may need to UNDO its own
            publication (``_cmd_run``'s compensating close) passes it
            to ``TabRegistry.close_tab_if_generation`` so a later
            publication by anyone else invalidates the undo.  A
            post-hoc ``generation()`` lookup would be racy — it could
            observe a LATER publisher's stamp and hand the undo a
            token that deletes that publisher's tab (gpt-5.6-sol
            review 3, introduced bug 1).

        Args:
            tab_id: The shared tab identifier.
            chat_id: New chat binding (``None`` keeps the current one).
            title: New title (``None``/empty keeps the current one).
            work_dir: New working directory (``None``/empty keeps it).
            scope_work_dir: The workspace-scope directory that decides
                which client tab bars show the tab, distinct from
                *work_dir* (``None``/empty keeps the current value;
                clients fall back to *work_dir* when it is empty).
            task_id: The specific historical task the tab shows
                (``None`` keeps the current value, ``""`` clears it).
            create: Register the tab first when it is unknown.
        """
        changed, displaced, generation = self.tab_registry.update_tab(
            tab_id, chat_id=chat_id, title=title,
            work_dir=work_dir, scope_work_dir=scope_work_dir,
            task_id=task_id, create=create,
        )
        for old_tab_id, removal_token in displaced:
            self._prune_local_uds_tab(old_tab_id)
            self._drop_tab_state(old_tab_id, removal_token=removal_token)
        if changed:
            self._broadcast_tabs_state()
        return generation

    def ready_tab_sync(
        self, restored: list[dict[str, str]],
    ) -> list[tuple[str, str, str]]:
        """Synchronize a (re)connecting client with the tab registry.

        Adopts the client's legacy ``restoredTabs`` when the registry
        is still empty (one-time migration from pre-registry clients),
        rebinds the in-memory chat views from the registry, and
        broadcasts the canonical ``tabs_state`` snapshot.

        Args:
            restored: Sanitized ``restoredTabs`` entries from the
                client's ``ready`` command.

        Returns:
            ``(tab_id, chat_id, task_id)`` triples for every
            chat-bound registry tab — the caller replays each so all
            clients converge on the same transcripts.  ``task_id`` is
            the specific historical task the tab was resumed to
            (``""`` when the tab tracks the chat's latest task);
            replaying it verbatim keeps a tab pinned to an older task
            from being silently switched to the chat's latest task by
            any client's reconnect.
        """
        self.tab_registry.merge_if_empty(restored)
        bound = self.tab_registry.bound_tabs()
        with self._state_lock:
            for tab_id, chat_id, _task_id in bound:
                self._tab_chat_views.setdefault(tab_id, chat_id)
        self._broadcast_tabs_state()
        return bound

    def _tab_model(self, tab_id: str) -> str:
        """Return the model selected for *tab_id* (default when unset).

        Args:
            tab_id: The frontend tab identifier string.

        Returns:
            The tab's selected model name.
        """
        with self._state_lock:
            return self._tab_models.get(tab_id, "") or self._default_model

    def _any_non_wt_running(
        self,
        repo_root: Path | None = None,
        *,
        exclude: AgentState | None = None,
    ) -> bool:
        """True if a non-worktree task is running on *repo_root*'s main tree.

        Must be called with ``_state_lock`` held.

        A non-worktree task only occupies the main working tree of the
        repository its ``work_dir`` resolves into (recorded on the
        state as ``non_wt_repo_root`` when the task starts).  A task running
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
            exclude: A state whose own admission must not count as
                busy — the finishing task whose post-task auto-commit
                runs while its ``is_running_non_wt`` is still set
                checks only for OTHER occupants (gpt-5.6-sol review 2,
                missed wiring 2).

        Returns:
            True if at least one state is running a non-worktree task
            whose main working tree is *repo_root* (or, when
            *repo_root* is ``None``, any non-worktree task at all).
        """
        for s in agent_state.agent_states.values():
            if s is exclude or not s.is_running_non_wt:
                continue
            if repo_root is None:
                return True
            s_root = s.non_wt_repo_root
            if s_root is None:
                # The non-worktree task is not inside any git repo, so
                # it cannot be modifying repo_root's main working tree.
                continue
            try:
                if s_root.resolve() == repo_root.resolve():
                    return True
            except OSError:  # pragma: no cover — unresolvable path
                return True
        return False

    def _claim_main_tree(
        self,
        repo_root: Path,
        reason: str,
        holder: list[MainTreeClaim] | None = None,
    ) -> bool:
        """Publish an exclusive main-tree mutation claim for *repo_root*.

        Must be called with ``_state_lock`` held, in the SAME locked
        section as the caller's ``_any_non_wt_running`` busy check —
        that is what makes the check-and-claim pair atomic against
        non-worktree task admission (which refuses to start while a
        claim is held, see ``_main_tree_claim_reason``).

        Interrupt safety (gpt-5.6-sol review 3, introduced bug 2): the
        server stops a task by injecting ``KeyboardInterrupt`` at an
        arbitrary bytecode boundary of the task thread — including
        between this call returning and the caller recording the claim
        in a local for its ``finally``.  The claim is therefore
        appended to *holder* BEFORE it is published: the caller
        installs its releasing ``try``/``finally`` around this call,
        so at every boundary the claim is either not yet published or
        already release-armed.  A claim stranded anyway (an injection
        that skips the caller's ``finally`` entirely) heals when its
        owner thread dies — see the liveness checks below and in
        :meth:`_main_tree_claim_reason`.

        Args:
            repo_root: The main repository root about to be mutated.
            reason: Human-readable operation name for refusal messages
                (e.g. ``"discard"``, ``"manual commit"``).
            holder: Release-arming list owned by the caller's
                ``finally``; the new claim is appended before
                publication.

        Returns:
            True when the claim was published; False when another
            main-tree mutation already holds a live claim on
            *repo_root*.
        """
        try:
            key = repo_root.resolve()
        except OSError:  # pragma: no cover — unresolvable path
            key = repo_root
        existing = self._main_tree_claims.get(key)
        if existing is not None:
            if existing.owner.is_alive():
                return False
            # The claiming thread died without releasing (its finally
            # was skipped by an injected stop): the claim is stale, and
            # honouring it would wedge this repository until restart.
            del self._main_tree_claims[key]
        claim = MainTreeClaim(key, reason, threading.current_thread())
        if holder is not None:
            holder.append(claim)
        self._main_tree_claims[key] = claim
        return True

    def _release_main_tree_claim(self, claim: MainTreeClaim) -> None:
        """Withdraw *claim* if it is still the published one.

        Must be called with ``_state_lock`` held.  The identity check
        makes release conditional on the releasing operation: a stale
        cleanup can never pop an unrelated successor's claim (the
        successor may have healed and replaced a stranded claim).

        Args:
            claim: The claim object appended by ``_claim_main_tree``.
        """
        if self._main_tree_claims.get(claim.key) is claim:
            del self._main_tree_claims[claim.key]

    def _main_tree_claim_reason(self, repo_root: Path | None) -> str | None:
        """Return the active main-tree claim on *repo_root*, if any.

        Must be called with ``_state_lock`` held.  A claim whose owner
        thread has died (its release was skipped by an injected stop)
        is treated as released — and healed here — so a stranded claim
        can never refuse task admission until restart.

        Args:
            repo_root: The main repository root a non-worktree task is
                about to write, or ``None`` when the task's
                ``work_dir`` is not inside a git repository (then no
                main-tree mutation can conflict with it).

        Returns:
            The claiming operation's reason string, or ``None`` when
            the main tree is unclaimed.
        """
        if repo_root is None:
            return None
        try:
            key = repo_root.resolve()
        except OSError:  # pragma: no cover — unresolvable path
            key = repo_root
        claim = self._main_tree_claims.get(key)
        if claim is None:
            return None
        if not claim.owner.is_alive():
            del self._main_tree_claims[key]
            return None
        return claim.reason

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

        from kiss.core.vscode_config import (
            _parse_custom_headers,
            get_custom_model_entry,
            load_config,
        )

        cfg = load_config()
        custom = get_custom_model_entry(cfg)
        if custom:
            models_list.insert(0, custom)

        # Settings-panel custom models (~/.kiss/MY_MODELS.json entries
        # carrying an endpoint) are selectable right away — they are not
        # in MODEL_INFO until the next restart, so the picker lists them
        # from the file, shaped exactly like the config-based custom
        # entry above.  Names already listed (a catalog model the entry
        # merely overrides) are left as the catalog reported them.
        listed_names = {m["name"] for m in models_list}
        for cm in list_custom_models():
            if not cm["endpoint"] or cm["name"] in listed_names:
                continue
            models_list.insert(0, {
                "name": cm["name"],
                "inp": 0,
                "out": 0,
                "uses": usage.get(cm["name"], 0),
                "vendor": "Custom",
                "endpoint": cm["endpoint"],
                "api_key": cm["api_key"],
                "extra_headers": _parse_custom_headers(cm["headers"]),
            })

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

        Scans the agent-state registry and collects the task id of
        every state whose ``task_thread`` is still alive.  Acquires
        ``_state_lock`` internally (re-entrant, so safe to call with
        it already held).

        Returns:
            Set of ``task_history.id`` values that are currently running.
        """
        running: set[str] = set()
        with self._state_lock:
            for state in agent_state.agent_states.values():
                # thread_alive() (C-R4) additionally counts a created-
                # but-not-yet-started worker as running: between
                # ``_cmd_run`` installing the thread and the worker
                # starting, the task is real and its live metrics
                # should already win over the persisted row.
                if state.thread_alive():
                    running.add(state.task_id)
        return running

    def _overlay_live_metrics(
        self,
        session: dict[str, Any],
        task_id: str,
    ) -> None:
        """Replace persisted metrics with live agent data for a running task.

        Looks up *task_id* in the agent-state registry and overwrites
        the ``tokens``, ``cost``, and ``steps`` fields in *session*
        with current values from the running agent, including the
        in-progress executor's ``step_count``.  Acquires
        ``_state_lock`` internally (re-entrant, so safe to call with
        it already held).

        The usage triple is read through ONE
        :func:`_subtask_metrics` call (``usage_snapshot()`` on a
        ``RelentlessAgent``, per-attribute fallback on plain agents):
        three separate property reads each sum the append-only usage
        ledger afresh, and a concurrent attribution between two of
        those reads shows the monitor an impossible mix (e.g. the old
        cost with the new tokens/steps).

        Args:
            session: The history session dict to update in place.
            task_id: The ``task_history.id`` of the running task.
        """
        with self._state_lock:
            state = agent_state.get(task_id)
            agent = state.agent if state is not None else None
            if state is None or agent is None:
                return
            tokens, cost, steps = _subtask_metrics(agent)
            session["tokens"] = tokens
            session["cost"] = cost
            cur = getattr(agent, "_current_executor", None)
            if cur is not None:
                steps += int(getattr(cur, "step_count", 0) or 0)
            session["steps"] = steps
            mdl_live = getattr(agent, "model_name", "")
            if isinstance(mdl_live, str) and mdl_live:
                session["model"] = mdl_live
            session["is_worktree"] = state.use_worktree
            session["is_parallel"] = state.use_parallel
            session["auto_commit_mode"] = state.auto_commit_mode

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
        # The chat-panel headers in the History sidebar show each chat's
        # FIRST task, which may be older than any row on this page.
        first_tasks = _chat_first_tasks([str(s["id"]) for s in sessions])
        for session in sessions:
            session["chat_first_task"] = first_tasks.get(str(session["id"]), "")
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
        """Close a tab: remove it from the registry and drop its state.

        The tab's state is marked ``frontend_closed`` BEFORE the
        registry removal (``_drop_tab_state`` marks it again while
        deciding busy/teardown).  Ordering matters: a concurrent
        ``_cmd_run`` publishing this tab re-checks the flag right
        after its ``_registry_update_tab(..., create=True)`` and
        undoes the recreate when the flag is up.  Mark-then-remove
        here plus recreate-then-recheck there makes every
        interleaving converge on "tab closed" — with the old
        remove-then-mark order a run could recreate the tab after the
        removal yet re-check before the mark, leaving the registry
        showing a tab whose backend state the deferred disposal later
        retired (gpt-5.6-sol review, finding 4).

        Args:
            tab_id: The frontend tab identifier to close.
        """
        with self._state_lock:
            state = agent_state.find_by_tab(tab_id)
            if state is not None:
                state.frontend_closed = True
        removal_token = self.tab_registry.close_tab(tab_id)
        if removal_token:
            # Registry tab: prune its local-UDS interest (hygiene) and
            # broadcast.  The prune is not the playback decision —
            # ``_local_tab_shown`` decides registry tabs from the
            # registry plus "a webview is attached" — so a prune that
            # lands after a concurrent reopen's re-registration cannot
            # hide the reopened tab, and a stale re-registration after
            # it cannot resurrect the closed one.  Prune BEFORE the
            # broadcast so a client that sees the snapshot never
            # observes stale bookkeeping.
            self._prune_local_uds_tab(tab_id)
            self._broadcast_tabs_state()
        # No prune for a tab the registry did not list (a sub-agent
        # viewer, a ``run_agent`` ``api-…`` tab, a duplicate close):
        # interest IS part of the decision for those, and a stale
        # duplicate close landing inside a reopen would otherwise strip
        # the reopened tab's interest.  ``_drop_tab_state`` below
        # retires such a tab through its agent state (``frontend_closed``)
        # or its subscriptions (``cleanup_tab``); the interest entry
        # itself is reconciled by the connection's next ``ready``.
        # The removal token lets the cleanup tail stand down when a
        # later publication (a concurrent ``resumeSession`` reopen)
        # has legitimately taken the tab over.  A close that found the
        # tab ABSENT (a token-0 duplicate close) reads the publication
        # clock instead, so a reopen republishing the tab after this
        # close observed it gone makes the tail stand down too — the
        # stale duplicate used to run unconditionally and re-marked
        # the reopened state ``frontend_closed`` / tore down the
        # re-subscribed viewer.  A publication landing between the
        # ``close_tab`` above and this clock read can stamp a
        # generation at or below the reading; ``_tab_reopened_since``
        # closes that gap by also treating registry PRESENCE as a
        # reopen (presence after an absent-close is always a later
        # republication).  A tab that was never published (sub-agent
        # tabs) has generation 0 and is never present, so those are
        # still dropped unconditionally.
        self._drop_tab_state(
            tab_id,
            removal_token=removal_token or self.tab_registry.clock(),
        )

    def _prune_local_uds_tab(self, tab_id: str) -> None:
        """Drop a registry-removed tab from the local-UDS interest sets.

        Bookkeeping hygiene for a tab the registry just removed (close
        or displacement): it bounds the interest a long-lived
        connection accumulates as the user opens and closes tabs.  The
        playback decision for registry tabs does not depend on
        interest — see :meth:`_local_tab_shown` — so the prune can
        never hide a concurrently reopened tab.  Never called for tabs
        outside the registry, whose interest is part of the decision.
        Duck-typed like ``cleanup_tab``: only the daemon's
        :class:`~kiss.server.web_server.WebPrinter` tracks local UDS
        tabs.

        Args:
            tab_id: The frontend tab identifier removed from clients.
        """
        prune = getattr(self.printer, "prune_local_uds_tab", None)
        if prune is not None:
            prune(tab_id)

    def _tab_reopened_since(self, tab_id: str, token: int) -> bool:
        """True when *tab_id* was legitimately reopened after *token*.

        The stand-down predicate of the close/displacement cleanup
        tails.  A reopen is visible in either of two ways:

        * a publication stamped a generation newer than *token*
          (:meth:`TabRegistry.republished_since`) — covers a removal
          token and a clock observation alike; or
        * the tab is PRESENT in the registry.  For a real removal
          token this is implied by the first clause (the removal
          deleted the row, so presence requires a later publication).
          For a token-0 duplicate close it closes the observation gap:
          a reopen publishing between the close's ``close_tab`` (which
          found the tab absent) and its ``clock()`` read stamps a
          generation at or below the observation, yet its row proves
          the reopen happened after the close's registry step.

        Both observations are drawn in ONE registry-locked section
        (:meth:`TabRegistry.reopened_since`): two separate calls left
        a seam where a rowless publication landing between them was
        invisible to both — the stale generation read and a still-false
        row presence (gpt-5.6-sol round-6 review, finding 1).

        Args:
            tab_id: The frontend tab identifier being cleaned up.
            token: The cleanup's removal token or clock observation.
        """
        return self.tab_registry.reopened_since(tab_id, token)

    def _commit_replay_publication(
        self,
        tab_id: str,
        chat_id: str,
        publication: int,
        source: AgentState | None,
        fallback_publication: int = 0,
    ) -> None:
        """Commit a replay's backend state iff *publication* still owns the tab.

        ``_replay_session`` publishes the reopened tab first and
        commits the backend state (clearing ``frontend_closed``,
        binding ``_tab_chat_views``) afterwards, outside any single
        lock.  Two stale-close races live in that window:

        * a concurrent ``_close_tab`` can REMOVE the replay's own
          publication before this commit runs — committing anyway
          leaves the registry saying "closed" while the backend says
          "open", matching neither serial order (review finding 1);
        * a token-0 duplicate close whose clock observation predates
          the viewer attach can run its cleanup tail (which removes
          the freshly installed printer subscription) between the
          attach and the publication — the published tab then never
          receives the running task's fan-out (review finding 2).

        Both close under one ``_state_lock`` section: the commit
        stands down unless the registry row still exists and nobody
        republished since *publication* (a newer publication performs
        its own commit; an absent row means a close owns the final
        state), and while it does own the row it (re)installs the
        viewer subscription for a still-live *source* —
        ``subscribe_tab`` is idempotent, so an undisturbed attach is
        unaffected.  Lock order: ``STATE_LOCK`` → registry leaf lock /
        printer locks, the established edges.

        A tab the registry could not admit (``publication == 0``: the
        registry is at capacity) has no row, so the row-based checks
        cannot qualify it — yet a token-0 close can complete in the
        same window and its cleanup must not be overwritten
        (gpt-5.6-sol round-2 review, finding 1).  Such a commit is
        qualified by *fallback_publication* instead: the rowless token
        ``_replay_session`` stamped via
        :meth:`TabRegistry.stamp_unregistered` BEFORE attempting the
        row publication.  The token-0 close's cleanup tail retires the
        stamp (:meth:`TabRegistry.retire_unregistered`) in the same
        ``_state_lock`` section as its other teardown, so the commit's
        equality check (``generation(tab_id) == fallback_publication``)
        fails exactly when a close (or any newer publication) landed
        after the stamp — a closed tab stays closed, with no chat-view
        or subscription recreation.

        Args:
            tab_id: The reopened frontend tab.
            chat_id: The chat the tab was rebound to.
            publication: The generation token the replay's own
                ``_registry_update_tab`` returned.  ``0`` means nothing
                was published (the registry is at capacity); the commit
                is then qualified by *fallback_publication*.
            source: The still-running state the tab was attached to as
                a viewer, or ``None`` when the chat has no live task.
            fallback_publication: The rowless publication token stamped
                before the row publication was attempted, or ``0`` when
                none could be stamped (the commit then stands down on
                the capacity path).
        """
        with self._state_lock:
            if publication > 0:
                if not self.tab_registry.has_tab(tab_id):
                    return
                if self.tab_registry.republished_since(tab_id, publication):
                    return
            else:
                if fallback_publication <= 0:
                    return
                if (
                    self.tab_registry.generation(tab_id)
                    != fallback_publication
                ):
                    # A token-0 close retired the stamp, or a newer
                    # publication (which performs its own commit)
                    # superseded it: this stale commit owns nothing.
                    return
            if source is not None and (
                source.is_task_active or source.thread_alive()
            ):
                self.printer.subscribe_tab(source.task_id, tab_id)
            state = agent_state.find_by_tab(tab_id)
            if state is not None:
                state.frontend_closed = False
            self._tab_chat_views[tab_id] = chat_id

    def _publish_replay_reopen(
        self,
        tab_id: str,
        chat_id: str,
        task_id: str | None,
        source: AgentState | None,
        *,
        title: str | None = None,
    ) -> None:
        """Stamp, publish and commit a tab reopened by ``_replay_session``.

        The rowless token is stamped BEFORE the row publication: if the
        registry is at capacity (publication 0, no row), the commit is
        qualified by this token instead, which a racing token-0 close
        retires (round-2 finding 1).  It is stamped under
        ``_state_lock`` so it is totally ordered against every close
        teardown's single destructive ``_state_lock`` section — it can
        no longer land between a teardown's ownership check and its
        cleanup (gpt-5.6-sol round-6 review, finding 1).

        Args:
            tab_id: The reopened frontend tab.
            chat_id: The chat the tab is bound to.
            task_id: The task to record on the registry row (falsy
                values publish an empty task id).
            source: The live state the viewer should subscribe to, if any.
            title: New row title, or ``None`` to keep the current one.
        """
        with self._state_lock:
            fallback = self.tab_registry.stamp_unregistered(tab_id)
        publication = self._registry_update_tab(
            tab_id,
            chat_id=chat_id,
            title=title,
            task_id=str(task_id) if task_id else "",
            create=True,
        )
        self._commit_replay_publication(
            tab_id, chat_id, publication, source,
            fallback_publication=fallback,
        )

    def _drop_tab_state(
        self, tab_id: str, removal_token: int | None = None,
    ) -> None:
        """Clean up all backend state for a tab no longer shown.

        Shared by :meth:`_close_tab` and the chat-bind displacement
        path in :meth:`_registry_update_tab` (the one-tab-per-chat
        invariant removes the previously bound tab from the registry;
        its backend state is released here).

        *removal_token* carries the registry removal's identity: this
        cleanup runs OUTSIDE the registry lock, so a later publication
        (a ``resumeSession`` reopen, a chat takeover re-binding the
        same id) can land in between and legitimately own the tab
        again.  Retiring the backend state then would produce a
        registry/state combination matching neither serial order
        (gpt-5.6-sol review 3, missed wiring 1) — so the drop stands
        down when :meth:`TabRegistry.republished_since` reports a
        newer publication.  A token-0 duplicate close (the registry
        did not list the tab) passes the clock OBSERVATION drawn by
        :meth:`TabRegistry.close_tab_or_observe` instead, so a reopen
        republishing the tab after the close observed it absent makes
        this drop stand down exactly like a real removal's tail; a
        tab that was never published (sub-agent tabs) has generation
        0 — never greater than any token — and is dropped
        unconditionally, as before.  ``None`` (direct callers with no
        registry ordering) keeps the unconditional behaviour.

        Removes the tab from
        the agent-state registry, cleans up per-tab printer
        state (bash buffers, recordings), and drops the persist-agent
        reference.

        When the tab is currently running a task or a merge/discard,
        the state is **not** removed immediately — the running agent
        must be allowed to finish: closing a chat tab does NOT stop a
        running agent task.  Instead the
        tab is marked ``frontend_closed = True`` so that
        :meth:`_dispose_if_closed` will tear it down later, once the
        last lifecycle flag drops to false.

        When the tab has a pending worktree (no active task / merge),
        the worktree is released (just like starting a new task
        would) before removing the tab, so the worktree branch and
        directory are not orphaned.  The state is claimed
        (``is_merging``) for that disposal and unregistered only
        afterwards — see :meth:`_teardown_tab_resources` for why the
        order matters.

        Args:
            tab_id: The frontend tab identifier being dropped.
        """
        busy = False
        with self._state_lock:
            if removal_token is not None and self._tab_reopened_since(
                tab_id, removal_token
            ):
                # A later publication reopened the tab between the
                # registry removal and this cleanup; it owns the state
                # now (its own rebind runs under this same lock).
                return
            state = agent_state.find_by_tab(tab_id)
            is_subagent_tab = (
                state is not None and state.is_subagent
            ) or "__sub_" in tab_id
            if state is not None:
                state.frontend_closed = True
                if state.busy():
                    busy = True
                    # Retire any rowless capacity-replay stamp HERE, at
                    # the close's linearization point: the busy path
                    # returns below without reaching
                    # ``_teardown_tab_resources``, whose retire would
                    # only run at the deferred ``_dispose_if_closed`` —
                    # but a pending capacity commit qualified by that
                    # stamp would clear ``frontend_closed`` first,
                    # reopening the closed backend view and preventing
                    # that deferred disposal from ever running
                    # (gpt-5.6-sol round-3 review, finding 1).  The
                    # retire re-verifies ownership and drops the stamp
                    # in ONE registry-locked step
                    # (:meth:`TabRegistry.finalize_removal`): a replay
                    # stamped BEFORE it stands down at its commit,
                    # while one stamped after it (or between the guard
                    # above and this line) survives and re-bumps the
                    # generation, superseding the close as on the
                    # immediate path (gpt-5.6-sol round-6 review,
                    # finding 1).  A no-op when the tab has a registry
                    # row (the row's own publication token belongs to
                    # its owner).
                    if removal_token is not None:
                        self.tab_registry.finalize_removal(
                            tab_id, removal_token,
                        )
                    else:
                        self.tab_registry.retire_unregistered(tab_id)
                else:
                    state.is_merging = True
                    # Published with every claim that retires a
                    # worktree so shutdown's ``_await_active_merges``
                    # waits for it (see ``_finalize_pending_worktree``).
                    state.merge_thread = threading.current_thread()
        # Sub-agent tabs are not in the registry, so their close
        # cannot mirror via ``tabs_state``: broadcast a canonical
        # close event instead.  Every client removes the tab, so a
        # torn-down shared per-tab printer subscription cannot starve
        # a client that still shows the tab.
        if is_subagent_tab:
            self._broadcast_subagent_close(tab_id)
        if busy:
            return
        self._teardown_tab_resources(tab_id, state, removal_token=removal_token)

    def _broadcast_subagent_close(self, tab_id: str) -> None:
        """Tell every client to close the sub-agent tab *tab_id*.

        Sub-agent tabs are derived state shared under ONE tab id by
        every client, but they never live in the tab registry — so a
        close on one client must be mirrored with this dedicated
        broadcast (clients apply it without echoing ``closeTab`` back).

        Args:
            tab_id: The shared sub-agent tab identifier.
        """
        self.printer.broadcast({
            "type": "closeSubagentTab",
            "tab_id": tab_id,
            # The explicit tabId stamp routes the event through the
            # printer's verbatim all-clients path (never recorded).
            "tabId": "",
        })

    def _dispose_if_closed(self, tab_id: str) -> None:
        """Dispose *tab_id*'s state if the frontend already closed it.

        Invoked at every lifecycle transition that can flip the last
        lifecycle flag to false (task end, merge end).  Pops the state
        only when ``frontend_closed`` is set AND no lifecycle flag is
        still raised; otherwise leaves it alone.  Idempotent and safe
        to call when no state exists for *tab_id*.

        The claim carries an ownership token, exactly like the close
        paths: the teardown runs OUTSIDE the locks, so a capacity
        replay can stamp a fresh rowless generation and commit a
        reopen after the claim — an unqualified teardown would then
        unregister the reopened state and delete its fresh stamp and
        chat view (gpt-5.6-sol round-4 review, finding 1).  The claim
        therefore stands down when the tab has a registry row (only a
        later republication can have created one), retires any
        pre-claim rowless replay stamp and draws a clock observation
        in ONE registry-locked step
        (:meth:`TabRegistry.retire_unregistered_and_observe`), and
        hands the observation to the teardown as its removal token: a
        replay stamped before the claim stands down at its own commit
        (the stamp is gone), while one stamped after it makes the
        teardown stand down.

        Args:
            tab_id: The frontend tab identifier.
        """
        if not tab_id:
            return
        with self._state_lock:
            state = agent_state.find_by_tab(tab_id)
            if state is None or not state.frontend_closed:
                return
            if state.busy():
                return
            if self.tab_registry.has_tab(tab_id):
                # A row exists only through a publication AFTER the
                # close that set ``frontend_closed`` (the close removed
                # the row): the reopen owns the tab, and its commit
                # clears the flag under this same lock.
                return
            state.is_merging = True
            state.merge_thread = threading.current_thread()
            token = self.tab_registry.retire_unregistered_and_observe(tab_id)
        self._teardown_tab_resources(tab_id, state, removal_token=token)

    def _teardown_tab_resources(
        self,
        tab_id: str,
        state: AgentState | None,
        removal_token: int | None = None,
    ) -> None:
        """Release worktree and per-tab printer state.

        *removal_token* (when given) re-verifies before every
        destructive step that no later publication has reopened the
        tab; see :meth:`_drop_tab_state`.  The re-checks close the
        window between that method's initial check and this tail.

        Shared cleanup tail used by both the immediate (:meth:`_close_tab`)
        and the deferred (:meth:`_dispose_if_closed`) disposal paths.
        The caller must have claimed *state* for this disposal by
        setting ``is_merging`` under ``_state_lock`` (so a concurrent
        close or lifecycle transition sees it busy and defers instead
        of retiring the same worktree twice); the claim is released
        and the state unregistered here.

        The worktree is retired BEFORE the state leaves the registry.
        ``JsonPrinter.live_worktree_branches`` — the reclaim exclusion
        set every other task in this process builds — only sees
        registered states, and ``reclaim_orphaned_worktrees`` exempts
        this process's own pid from the owner protection, so an
        unregistered agent's worktree is fair game for the next
        reclaim.  When the retire keeps the worktree but cannot make
        that decision durable (:meth:`WorktreeSorcarAgent.retire_for_disposal`
        returns False: the ``kiss-preserve`` marker could not be
        written), the state therefore stays registered — closed,
        idle, still naming the branch — until a later disposal attempt
        (another close, a chat rebind displacing the tab, the next
        ``_dispose_if_closed``) writes the marker and drops it.

        Retiring the worktree here can strand work — a rejected
        pre-commit hook leaves the changes in the worktree directory,
        and a conflicting merge leaves them on the branch — and the
        agent records where to find them as a pending warning.  Those
        warnings are flushed before the printer is torn down, because
        after that there is nothing left to say it on and the user
        would never learn their work survived.

        Args:
            tab_id: The frontend tab identifier being disposed.
            state: The claimed agent state, or ``None`` when the tab
                never ran a task (e.g. ``closeTab`` for an unknown
                id).
        """
        wt_agent = state.agent if state is not None else None
        claim_retained = False
        if wt_agent is not None and getattr(wt_agent, "_wt_pending", False):
            try:
                claim_retained = not wt_agent.retire_for_disposal()
                wt_agent._flush_warnings(self.printer)
            except Exception:  # pragma: no cover — git/printer failure
                logger.debug("Worktree release on tab close failed", exc_info=True)
                claim_retained = bool(getattr(wt_agent, "_wt_pending", False))
        # ONE ``_state_lock`` section for every destructive step, with
        # ONE atomic ownership decision (``finalize_removal``: re-check
        # + rowless stamp retirement under a single registry lock
        # acquisition) at its head.  Splitting the tail into two locked
        # sections, each guarded by its own registry reads, left seams
        # where a rowless capacity replay could stamp between a guard
        # and the destructive step it protected — the stale teardown
        # then unregistered the reopened state or erased the fresh
        # stamp (gpt-5.6-sol round-6 review, finding 1).  Now a stamp
        # lands strictly before the decision (the teardown stands down
        # wholesale; the replay's commit, which also takes
        # ``_state_lock``, wins) or strictly after it (the stamp
        # survives and the commit replays into a fully torn-down tab —
        # the serial "closed, then reopened" order).
        with self._state_lock:
            if state is not None:
                state.is_merging = False
                state.merge_thread = None
            if removal_token is not None:
                if not self.tab_registry.finalize_removal(
                    tab_id, removal_token,
                ):
                    # Reopened since the claim (e.g. while the worktree
                    # was being retired): the new publication owns this
                    # state — and any fresh rowless stamp — from here on.
                    return
            else:
                # Direct callers with no registry ordering keep the
                # unconditional behaviour; the retire is a no-op when
                # the tab has a registry row (the row's own publication
                # token must survive for its owner).
                self.tab_registry.retire_unregistered(tab_id)
            if state is not None:
                if claim_retained:
                    logger.warning(
                        "Tab %s closed but its worktree's keep-for-review "
                        "decision is not durable yet; keeping its state "
                        "registered so the worktree stays protected",
                        tab_id,
                    )
                else:
                    agent_state.unregister(state.task_id, state)
            self._printer_cleanup_tab(tab_id)
            self._tab_chat_views.pop(tab_id, None)
            self._tab_opened_task_ids.pop(tab_id, None)
            self._tab_models.pop(tab_id, None)

    def _new_chat(self, tab_id: str) -> None:
        """Start a new chat session for the given tab.

        The ``newChat`` command is only issued by the frontend's
        ``createNewTab`` flow, which always allocates a fresh tab id
        that the backend has never seen before, so there is no prior
        run state (no active task, no in-progress merge, no pending
        worktree, no carried-over warnings) to guard against here.

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
        with self._state_lock:
            self._refresh_default_model()
            self._tab_models[tab_id] = self._default_model
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
        The tab_id (frontend routing key) does not change.

        When ``tab_id`` is empty the call is a no-op — the previous
        behavior of synthesizing a phantom tab keyed by ``chat_id`` and
        mutating its ``use_worktree`` flag violated per-tab state
        isolation (C2/C3 fix).

        Loading a chat never touches the tab's ``use_worktree`` /
        ``use_parallel`` / ``auto_commit_mode`` / ``selected_model``:
        those mirror the toolbar toggles, which are global UI state the
        user owns.  Clearing them made a history click silently switch
        auto-commit off, which in turn changed how the pending-worktree
        handling below finalizes the branch.

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
            rebound_state = self._attach_viewer_to_running_chat(
                chat_id,
                tab_id,
                task_id=task_id,
                is_subagent=False,
            )
            if rebound_state is not None:
                start_ts = self._live_task_start_ms(task_id, chat_id)
                self._broadcast_viewer_running(tab_id, rebound_state, start_ts)
                # The task runs but has no history row yet: the live
                # in-memory recording is the only copy of what it has
                # already broadcast (the events table is written
                # asynchronously).  Snapshot by the LIVE state's task
                # id — the caller's *task_id* is None for a plain chat
                # resume, and a pre-history-row run's recording (e.g.
                # its setup-failure result) is keyed by the
                # provisional id the state carries (audit0903 F4).
                live_events = self.printer.peek_recording_for_task(
                    rebound_state.task_id,
                )
                events_payload: dict[str, Any] = {
                    "type": "task_events",
                    # The run has no ``task_history`` row to read the
                    # task text from yet, but the fixed task panel of
                    # a (re)connecting client is repainted from this
                    # field — use the prompt ``_cmd_run`` stamped on
                    # the state at submit time so the panel is not
                    # blanked for the whole setup window.
                    "task": rebound_state.last_user_prompt,
                    "task_id": task_id,
                    "chat_id": chat_id,
                    "extra": "",
                    "tabId": tab_id,
                }
                self.printer.broadcast(
                    {**events_payload, "events": live_events},
                )
                self._finalize_viewer_attach(
                    tab_id, rebound_state, live_events, events_payload,
                )
            with self._state_lock:
                state = agent_state.find_by_tab(tab_id)
                is_sub_view = state is not None and state.is_subagent
            # Publish the reopen BEFORE clearing ``frontend_closed``:
            # a stale duplicate close's cleanup tail orders itself on
            # the registry clock (``republished_since``), so with
            # publish-first it either sees this publication and stands
            # down, or runs entirely before it — in which case the
            # commit below lands last and the reopened tab does not
            # end up marked closed.  The commit itself is qualified by
            # THIS publication's generation (see
            # ``_commit_replay_publication``): a close that removed
            # the publication, or a newer publication, owns the tab
            # from here on.
            if chat_id and not is_sub_view:
                self._publish_replay_reopen(
                    tab_id, chat_id, task_id, rebound_state,
                )
            else:
                with self._state_lock:
                    state = agent_state.find_by_tab(tab_id)
                    if state is not None:
                        state.frontend_closed = False
            self._emit_pending_ask(tab_id)
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
        rebound_state = self._attach_viewer_to_running_chat(
            chat_id,
            tab_id,
            task_id=rebound_task_id,
            is_subagent=subagent_info is not None,
        )
        if rebound_state is not None:
            # The task is still running, so the events table lags
            # behind it: display events reach the database through an
            # asynchronous writer, and a tab resumed moments after the
            # task started (the round trip a freshly spawned
            # ``run_parallel`` sub-agent's ``new_tab`` triggers) would
            # replay an EMPTY transcript and permanently miss every
            # event from before this subscription.  The printer's live
            # in-memory recording is the authoritative copy while the
            # task runs; events recorded after this snapshot reach the
            # tab through the fan-out the reattach above just set up.
            # Known micro-window: recording and fan-out are two steps
            # of one broadcast, so an event recorded just before this
            # snapshot can also fan out just after the replay below and
            # render twice.  That window is a thread preemption inside
            # a single broadcast (microseconds); the alternative — the
            # events-table read this replaces — lost the whole
            # transcript head for the async writer's full lag.
            live_events = self.printer.peek_recording_for_task(
                rebound_task_id,
            )
            if live_events:
                result["events"] = live_events
        if subagent_info is None and chat_id:
            # A resumed chat binds + titles the tab for EVERY client:
            # the shared registry is what makes a history click on one
            # client rename the same tab everywhere.  The selected
            # task is persisted too, so the ready replay path keeps a
            # tab pinned to an older task instead of silently
            # switching every client to the chat's latest task.
            # Published BEFORE the ``frontend_closed`` clear below: a
            # stale duplicate close's cleanup tail orders itself on
            # the registry clock, so it either sees this publication
            # and stands down or runs entirely before the commit,
            # which then lands last — the reopened tab can no longer
            # end up published yet marked closed.  The commit is
            # qualified by THIS publication's generation (see
            # ``_commit_replay_publication``): a close that removed
            # the publication, or a newer publication, owns the tab.
            self._publish_replay_reopen(
                tab_id, chat_id, task_id, rebound_state,
                title=str(result.get("task", "") or ""),
            )
        else:
            with self._state_lock:
                state = agent_state.find_by_tab(tab_id)
                if state is not None:
                    state.frontend_closed = False
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
                    "startTs": _start_ts_from_extra(extra_raw),
                }
            )

        if rebound_state is not None:
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
            self._broadcast_viewer_running(
                tab_id, rebound_state, start_ts_for_resume,
            )
        replayed_events = with_task_settings_event(
            _coalesced_replay_events(result["events"]), result,
        )
        replay_payload: dict[str, Any] = {
            "type": "task_events",
            "task": result["task"],
            "task_id": result.get("task_id"),
            "chat_id": chat_id,
            "extra": _extra_for_replay(result.get("extra", "")),
            "tabId": tab_id,
        }
        self.printer.broadcast({**replay_payload, "events": replayed_events})
        if rebound_state is not None:
            self._finalize_viewer_attach(
                tab_id, rebound_state, replayed_events, replay_payload,
            )
        self._emit_pending_ask(tab_id)
        self._emit_pending_worktree(tab_id)

        if subagent_info is None and isinstance(rebound_task_id, str) and rebound_task_id:
            self._open_persisted_subagent_tabs(
                parent_task_id=rebound_task_id,
                parent_tab_id=tab_id,
            )

    def _emit_pending_ask(self, tab_id: str) -> None:
        """Re-broadcast a still-pending ask-user question to *tab_id*.

        Session replays (``resumeSession``) repaint a tab's transcript
        but the ``askUser`` modal is a live event: a client that
        connects or reloads while the tab's task is blocked inside
        ``ask_user_question`` would otherwise never see the question.
        Called after every ``task_events`` replay broadcast so such
        clients converge on the same modal every other client shows.

        The pending state is resolved exactly like ``userAnswer``
        routing (:meth:`_resolve_user_answer_state`): the state
        launched from *tab_id* itself, else the state of any task the
        tab is subscribed to.  The broadcast happens under
        ``_state_lock`` — the same lock ``_cmd_user_answer`` holds
        while clearing ``pending_ask_question`` — so the re-emitted
        ``askUser`` can never be ordered after the answer's
        ``askUserDone`` (which is broadcast after the lock is
        released), guaranteeing no client is left with a stale modal.

        Args:
            tab_id: Frontend tab id whose viewers should (re)show the
                modal.  The event is tabId-stamped, so every connected
                client mirroring the shared tab renders it.
        """
        if not tab_id:
            return
        with self._state_lock:
            state = self._resolve_user_answer_state(tab_id)
            question = state.pending_ask_question if state is not None else ""
            if question:
                self.printer.broadcast(
                    {
                        "type": "askUser",
                        "question": question,
                        "tabId": tab_id,
                    }
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
           the agent-state registry for a non-subagent state registered
           under *parent_task_id*.  This is the primary, unambiguous
           match.

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
            if parent_task_id is not None:
                parent = agent_state.get(parent_task_id)
                if parent is not None and not parent.is_subagent and parent.tab_id:
                    return parent.tab_id

            non_sub_states = [
                st
                for st in agent_state.agent_states.values()
                if not st.is_subagent and st.tab_id
            ]

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
        the agent-state registry under the sub-agent's
        own task id means its thread is still running so the tab
        should show the spinner; absence means the sub-agent has
        completed and the tab should render as a finished tab with the
        green tick.

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
                # A still-running sub-agent's events table lags behind
                # the live run (asynchronous writer); its in-memory
                # recording holds the full transcript so far.  Events
                # recorded after this snapshot reach the tab through
                # the fan-out the reattach above just set up.
                live_events = self.printer.peek_recording_for_task(
                    str(sub_task_id),
                )
                if live_events:
                    row["events"] = live_events
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
                    # The frontend attributes the row to the fan-out
                    # call that was running when it started.
                    "startTs": _start_ts_from_extra(row.get("extra", "")),
                }
            )
            self.printer.broadcast(
                {
                    "type": "task_events",
                    "events": with_task_settings_event(
                        _coalesced_replay_events(row["events"]), row,
                    ),
                    "task": description,
                    "task_id": sub_task_id,
                    "chat_id": row.get("chat_id", ""),
                    "extra": _extra_for_replay(row.get("extra", "")),
                    "tabId": sub_tab_id,
                }
            )
            self._emit_pending_ask(sub_tab_id)
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

        Scans the agent-state registry for the state owning the running task and reads the
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
            for state in agent_state.agent_states.values():
                if task_id is not None:
                    if state.task_id != task_id:
                        continue
                elif not chat_id or state.chat_id != chat_id or state.is_subagent:
                    continue
                start_ms = int(getattr(state.agent, "_task_start_ms", 0) or 0)
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
        """Boolean facade over :meth:`_attach_viewer_to_running_chat`.

        Kept for the callers (and tests) that only need to know
        WHETHER a live task was attached; ``_replay_session`` uses the
        state-returning method directly because its post-broadcast
        liveness re-check needs the state object itself.

        Args:
            chat_id: The chat id of the task the user clicked in
                history.
            new_tab_id: The freshly allocated frontend tab id.
            task_id: When provided, only states whose task id equals
                this are eligible.
            is_subagent: Skip the chat-id fallback pass (sub-agent
                views must match by task id alone).

        Returns:
            ``True`` when a matching live agent exists and
            *new_tab_id* is now subscribed to its event stream.
        """
        return (
            self._attach_viewer_to_running_chat(
                chat_id,
                new_tab_id,
                task_id=task_id,
                is_subagent=is_subagent,
            )
            is not None
        )

    def _attach_viewer_to_running_chat(
        self,
        chat_id: str,
        new_tab_id: str,
        *,
        task_id: str | None = None,
        is_subagent: bool = False,
    ) -> AgentState | None:
        """Subscribe *new_tab_id* to a still-running agent state
        so its live agent's events ALSO flow to the newly opened tab —
        without stealing the stream from the original client.

        ``tab_id`` (frontend routing key) and ``chat_id`` (persistence
        key) are orthogonal: the source state is keyed by its task id,
        and the tab id and chat id are stored on the state.

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
        original agent state keeps owning the running task
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
            The live source state *new_tab_id* is now subscribed to,
            or ``None`` when no matching live agent exists.  Callers
            that broadcast an optimistic ``status running=true`` for
            the attach re-check THIS object's liveness afterwards
            (:meth:`_broadcast_viewer_running`) — the object survives
            the printer bridge's mid-run re-keying, which a task-id
            lookup would not.
        """
        if not new_tab_id:
            return None
        if task_id is None and not chat_id:
            return None
        with self._state_lock:
            source: AgentState | None = None
            if task_id is not None:
                candidate = agent_state.get(task_id)
                # thread_alive() (C-R4) additionally counts a created-
                # but-not-yet-started worker as live: a task in the
                # run-startup window is real, and a viewer resuming
                # its chat must attach to it rather than be treated
                # as opening a finished session.
                if candidate is not None and (
                    candidate.is_task_active or candidate.thread_alive()
                ):
                    source = candidate
            if source is None and chat_id and not is_subagent:
                for t in agent_state.agent_states.values():
                    if t.chat_id != chat_id or t.is_subagent:
                        continue
                    if t.thread_alive() or t.is_task_active:
                        source = t
                        break
            if source is None:
                return None
            source_task_id = source.task_id
        self.printer.subscribe_tab(source_task_id, new_tab_id)
        return source

    def _viewer_owns_other_busy_run(
        self,
        tab_id: str,
        source: AgentState,
    ) -> bool:
        """True when *tab_id* currently owns a busy run other than *source*.

        Guard for the viewer-attach status broadcasts (audit0903 F3):
        a replay can be delayed past the point where the user starts a
        NEW run on the very tab that was attaching, and status events
        are not generation-qualified at the frontend — a stale
        ``running=true`` would overwrite the newer run's timer with
        the old task's ``startTs``, and a stale ``running=false``
        would kill its spinner/Stop button.  Must be called under
        ``_state_lock`` so the check is serialized against
        ``_cmd_run``'s state installation.

        Args:
            tab_id: The viewer tab about to receive a status event.
            source: The (old) task state the status event is about.

        Returns:
            ``True`` when the tab's current state is a different busy
            run — the status event must then be suppressed.
        """
        viewer_state = agent_state.find_by_tab(tab_id)
        return (
            viewer_state is not None
            and viewer_state is not source
            and viewer_state.busy()
        )

    def _broadcast_viewer_running(
        self,
        tab_id: str,
        source: AgentState,
        start_ts: int,
    ) -> None:
        """Flip *tab_id* to running for the task it just attached to.

        Emitted optimistically right after
        :meth:`_attach_viewer_to_running_chat`.  The check-and-
        broadcast pair runs under ``_state_lock`` so it is serialized
        against ``_cmd_run`` installing a NEWER run on the same tab
        (audit0903 F3): once the tab owns a different busy run, this
        stale ``running=true`` — whose ``startTs`` is the OLD task's —
        is suppressed instead of overwriting the newer run's timer; a
        newer install that lands after this broadcast emits its own
        ``running=true`` afterwards and wins.

        The source-died correction lives in
        :meth:`_finalize_viewer_attach`, which ``_replay_session``
        calls AFTER delivering the transcript, so an attached viewer
        never ends on a bare status boolean (audit0903 F4).

        Args:
            tab_id: The viewer tab that just attached.
            source: The live state returned by
                :meth:`_attach_viewer_to_running_chat`.
            start_ts: The task's start timestamp (ms epoch, 0 when
                unknown), echoed on the ``running=true`` broadcast.
        """
        with self._state_lock:
            if self._viewer_owns_other_busy_run(tab_id, source):
                return
            self.printer.broadcast(
                {
                    "type": "status",
                    "running": True,
                    "tabId": tab_id,
                    "startTs": start_ts,
                }
            )

    def _finalize_viewer_attach(
        self,
        tab_id: str,
        source: AgentState,
        replayed_events: list[dict[str, Any]],
        events_payload: dict[str, Any],
    ) -> None:
        """Correct *tab_id*'s status when *source* died during the attach.

        ``_replay_session`` resolves the live task under
        ``_state_lock``, subscribes the viewer and replays the
        transcript — with the lock released between the steps.  The
        task can finish inside that window: its end-of-run fan-out
        (``_TaskRunnerMixin._broadcast_status_end_to_viewers``) reads
        the subscriber map BEFORE the subscription lands, so nothing
        would ever send this tab ``running=false`` and its spinner
        (and follow-up input routed as ``appendUserMessage`` against a
        finished task) would survive the dead task forever.

        Called AFTER the replay's ``task_events`` broadcast, this
        re-checks the source OBJECT under the lock (the printer bridge
        re-keys states mid-run, so a key lookup could misread a live
        task as finished) and corrects the viewer.  When the
        transcript it just replayed predates the death — no terminal
        ``result`` event in it — the live recording is re-snapshot and
        re-broadcast first, so the terminal result reaches the viewer
        BEFORE the corrective ``running=false`` and the viewer never
        ends with only status booleans (audit0903 F4).  An end that
        commits after this re-check necessarily runs with the
        subscription already registered, so the normal fan-out
        delivers both the result and the terminal status.

        Both the snapshot and the correction are suppressed when the
        tab meanwhile owns a DIFFERENT busy run (audit0903 F3): the
        newer run's own lifecycle broadcasts are authoritative for
        the tab.

        Args:
            tab_id: The viewer tab that attached.
            source: The state the viewer attached to.
            replayed_events: The events the replay just delivered.
            events_payload: The replay's ``task_events`` payload minus
                ``events`` — reused verbatim for the corrective
                terminal snapshot so both broadcasts describe the same
                task/chat/tab.
        """
        with self._state_lock:
            if source.is_task_active or source.thread_alive():
                return
            if self._viewer_owns_other_busy_run(tab_id, source):
                return
            if not any(
                ev.get("type") == "result" for ev in replayed_events
            ):
                events = self.printer.peek_recording_for_task(
                    source.task_id,
                )
                if any(ev.get("type") == "result" for ev in events):
                    self.printer.broadcast(
                        {**events_payload, "events": events},
                    )
            self.printer.broadcast(
                {
                    "type": "status",
                    "running": False,
                    "tabId": tab_id,
                }
            )

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
            "events": (
                with_task_settings_event(
                    _coalesced_replay_events(result["events"]), result,
                )
                if result
                else []
            ),
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
