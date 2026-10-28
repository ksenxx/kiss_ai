# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backend server for the Sorcar agent (``AgentServer``).

The per-command handlers, task-runner, worktree flow and
autocomplete logic live in sibling mixin modules.  This file keeps the
per-tab state accessors, the command dispatcher, and the history /
chat / commit-message helpers.

``AgentServer`` is consumed by :class:`RemoteAccessServer`
(:mod:`kiss.server.web_server`), which owns the actual I/O transport
(one WSS endpoint: token-authenticated for local Python clients,
password-authenticated for remote browsers) and instantiates a
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
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar._concurrency import _race_delay, locked_call
from kiss.agents.sorcar.machines import (
    HISTORY_PAGE,
    LOCAL_MACHINE,
    HistoryPage,
    Machine,
    load_machines,
    local_machine_name,
    machine_label,
    merged_history,
)
from kiss.agents.sorcar.persistence import (
    _chat_first_tasks,
    _chat_summaries,
    _get_adjacent_task_by_chat_id,
    _history_date_range,
    _is_failed_result,
    _load_chat_events_by_task_id,
    _load_history,
    _load_input_history,
    _load_last_model,
    _load_latest_chat_events_by_chat_id,
    _load_model_usage,
    _load_subagent_rows_by_parent_task_id,
    _recover_orphaned_tasks,
)
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
from kiss.server.browser_tab import BrowserTabService
from kiss.server.commands import _CommandsMixin
from kiss.server.file_index import FileIndexRegistry
from kiss.server.helpers import model_vendor
from kiss.server.json_printer import (
    JsonPrinter,
    _coalesce_events,
    with_synthesized_events,
)
from kiss.server.merge_flow import _MergeFlowMixin
from kiss.server.tab_registry import TabRegistry
from kiss.server.task_runner import _TaskRunnerMixin
from kiss.server.task_update import mark_legacy_updates_as_side_channels
from kiss.server.terminal_tab import TerminalService

__all__ = ["AgentServer"]

logger = logging.getLogger(__name__)


_REPLAY_STRIPPED_EXTRA_KEYS = (
    "model",
    "is_parallel",
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


def _extra_dict(extra: object) -> dict[str, Any]:
    """Parse a task row's ``extra`` JSON into a dict.

    Args:
        extra: The persisted ``extra`` value (a JSON string), an
            already parsed dict, or anything else.

    Returns:
        The parsed object (or *extra* itself when it already is a
        dict); ``{}`` when *extra* is missing, not a string, not JSON,
        or not a JSON object.
    """
    if isinstance(extra, dict):
        return extra
    if not isinstance(extra, str) or not extra:
        return {}
    try:
        parsed = json.loads(extra)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
    try:
        start_ts = int(_extra_dict(extra).get("startTs", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(start_ts, 0)


def _subagent_done_event(tab_id: str, scope: dict[str, Any]) -> dict[str, Any]:
    """Build the ``subagentDone`` event closing sub-agent tab *tab_id*.

    Args:
        tab_id: The sub-agent tab to close.
        scope: ``{"connId": ...}`` (and ``replayId``) to address one
            connection, else empty.

    Returns:
        The event dict to broadcast.
    """
    return {"type": "subagentDone", "tab_id": tab_id, "tabId": "", **scope}


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
        legacy = ev.get("_timestamp")
        if "_timestamp" in ev and not isinstance(legacy, (int, float)):
            # TEXT/BLOB junk from the dynamically typed SQLite column:
            # bytes are not JSON-serializable, so the key must not reach
            # the frame sizing or the transport.
            del ev["_timestamp"]
        elif "ts" not in ev and isinstance(legacy, (int, float)) and 0 < legacy <= 8.64e12:
            ev["ts"] = int(legacy * 1000)
    return _bound_replay_events(_coalesce_events(evs))


#: A replayed delta event's ``text`` is cut to this many characters
#: (head and tail kept).  Coalescing merges a legacy transcript's
#: uncapped bash chunks into one event, so the per-row cap of the
#: persistence loader is not enough on its own.
MAX_REPLAY_EVENT_TEXT_CHARS = 200_000
#: Budget, in serialized bytes, for the events of one replay frame.
#: A client drops its socket above 64 MiB (``_MAX_LINE_BYTES``) and a
#: laptop on a tunnel must not pull hundreds of megabytes per
#: reconnect; beyond the budget the
#: oldest events are left out and a note says so.
MAX_REPLAY_FRAME_BYTES = 16 * 1024 * 1024
_REPLAY_DELTA_TYPES = frozenset({"thinking_delta", "text_delta", "system_output"})


def _clip_text(text: str, limit: int) -> str:
    """Return *text* cut to *limit* characters keeping its head and tail."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return text[:half] + f"\n... [{omitted} characters omitted from replay] ...\n" + text[-half:]


def _event_bytes(ev: dict[str, Any]) -> int:
    """Size of *ev* as the transport sends it (``json.dumps`` default).

    Measured on the serialized form, not on character counts: the
    default encoder escapes every non-ASCII character to ``\\uXXXX``,
    so a transcript of CJK text or emoji is three to six times its
    character count on the wire.
    """
    return len(json.dumps(ev))


def _bound_replay_events(evs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fit a coalesced replay under the per-event and per-frame caps.

    Clips every delta event's ``text`` to
    :data:`MAX_REPLAY_EVENT_TEXT_CHARS`, then walks the list from the
    newest event back and keeps only the suffix that fits in
    :data:`MAX_REPLAY_FRAME_BYTES` serialized; a ``system_output``
    note replaces the omitted prefix so the reader knows the
    transcript is cut.  The newest event is always kept (a ``result``
    larger than the whole budget still reaches the client rather than
    leaving the transcript empty).

    Args:
        evs: Coalesced events, oldest first (clipped entries are
            replaced in this list).

    Returns:
        The bounded event list.
    """
    total = 0
    kept_from = 0
    for i in range(len(evs) - 1, -1, -1):
        ev = evs[i]
        text = ev.get("text")
        if (
            ev.get("type") in _REPLAY_DELTA_TYPES
            and isinstance(text, str)
            and len(text) > MAX_REPLAY_EVENT_TEXT_CHARS
        ):
            # Copied, not edited in place: an un-merged event may be
            # the printer's live recording entry.
            ev = evs[i] = {**ev, "text": _clip_text(text, MAX_REPLAY_EVENT_TEXT_CHARS)}
        total += _event_bytes(ev)
        if total > MAX_REPLAY_FRAME_BYTES and i < len(evs) - 1:
            kept_from = i + 1
            break
    if kept_from == 0:
        return evs
    kept = evs[kept_from:]
    note: dict[str, Any] = {
        "type": "system_output",
        "text": (
            f"[{kept_from} earlier events are not shown: the transcript "
            f"exceeds the {MAX_REPLAY_FRAME_BYTES >> 20} MB replay limit]"
        ),
    }
    if kept and "ts" in kept[0]:
        note["ts"] = kept[0]["ts"]
    if not any(ev.get("type") in ("usage_info", "result") for ev in kept):
        # The run's totals live in its newest usage event; when that
        # fell in the omitted prefix (a long tool output followed it)
        # the reader would see no cost at all.  Carry it across: the
        # event is cumulative, so the latest one alone is exact.
        for ev in reversed(evs[:kept_from]):
            if ev.get("type") == "usage_info":
                kept = [ev, *kept]
                break
    return [note, *kept]


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
    return locked_call(agent_state.STATE_LOCK, _no_live_state_locked, sub_task_id)


def _no_live_state_locked(task_id: str) -> bool:
    """Under ``STATE_LOCK``: no registered state for *task_id* is running."""
    state = agent_state.get(task_id)
    return state is None or not state.running()


def _is_side_channel_row(row: dict[str, object]) -> bool:
    """True when the persisted sub-agent *row* is a side channel.

    A side channel is a sub-agent whose result is delivered into its
    parent's transcript (the ``/ask`` answerer): its nested tab is
    closed when the run ends and must not be re-opened by a replay.
    The mark travels in the row's synthesized ``extra`` JSON as
    ``subagent.side_channel`` (see ``persistence._row_to_extra_json``).

    Args:
        row: A sub-agent row as returned by
            ``_load_subagent_rows_by_parent_task_id``.

    Returns:
        True when the row's ``subagent`` extra carries
        ``side_channel: true``.
    """
    sub = _extra_dict(row.get("extra")).get("subagent")
    return isinstance(sub, dict) and bool(sub.get("side_channel"))


def _prewarm_task_dependencies() -> None:
    """Warm the lazily imported modules a first task would pay for.

    The first task after a daemon start used to spend several seconds
    between its ``task_history`` row allocation and the agent's first
    event: ``KISSAgent._reset`` imports the concrete model class (and
    its provider SDK) on demand, and ``_run_task_inner`` loads the
    config module lazily.  Importing them here — on a
    background daemon thread started from ``AgentServer.__init__`` —
    moves that cost to server startup, where nobody is waiting on it.

    Best-effort: a missing optional SDK or an import error must never
    affect server startup, so every import failure is only logged.

    The provider modules go through :func:`kiss.core.models.import_provider_module`,
    the lock the task threads' own lazy imports take: a first task that
    starts while this thread is still importing would otherwise walk
    the ``openai`` package's cyclic import graph from the other end and
    die at once with ``_DeadlockError``.
    """
    from kiss.core.models import import_provider_module

    for mod in (
        "kiss.core.models.anthropic_model",
        "kiss.core.models.openai_compatible_model",
        "kiss.core.models.openai_compatible_model2",
        "kiss.core.models.gemini_model",
        "kiss.core.user_config",
    ):
        try:
            import_provider_module(mod)
        except Exception:
            logger.debug("Prewarm import of %s failed", mod, exc_info=True)
    try:
        from kiss.core.models.model_info import get_available_models

        get_available_models()
    except Exception:
        logger.debug("Prewarm of the model registry failed", exc_info=True)


class AgentServer(
    _CommandsMixin,
    _TaskRunnerMixin,
    _MergeFlowMixin,
    _AutocompleteMixin,
):
    """Backend server behind the daemon's one WSS endpoint."""

    _orphan_sweep_thread: threading.Thread | None = None

    def __init__(self, printer: JsonPrinter | None = None) -> None:
        self.printer: JsonPrinter = printer or JsonPrinter()
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
                    "New AgentServer kept %d live task(s) registered; "
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
        # The machine's browser streamed as a tab on every surface; idle
        # (no thread, no browser) until the first ``browserOpen``.
        self.browser_tabs = BrowserTabService(
            self.printer,
            Path(_persistence._KISS_DIR) / "browser-tab-profile",
        )
        # The remote webapp's terminal tabs: one pty shell each, owned
        # by the connection that opened it (``terminalOpen``).
        self.terminals = TerminalService(self.printer)
        self._tab_chat_views: dict[str, str] = {}
        # Running sub-agent tabs the user closed by hand; see
        # ``_remember_closed_subagent_tab``.
        self._closed_subagent_tabs: set[str] = set()
        # Rebind surviving chat views from the persisted registry so a
        # follow-up ``run`` after a daemon restart continues the tab's
        # chat instead of silently starting a fresh one.
        self._tab_chat_views.update(self.tab_registry.bindings())
        # ``tabs_state`` publication bookkeeping, all under the
        # condition: every registry mutation requests a generation,
        # one sender at a time ships snapshots until the generation on
        # the wire covers every request; see ``_broadcast_tabs_state``.
        self._tabs_state_cond = threading.Condition()
        self._tabs_state_requested = 0
        self._tabs_state_sent = 0
        self._tabs_state_sender: threading.Thread | None = None
        self._tab_opened_task_ids: dict[str, str] = {}
        self._tab_models: dict[str, str] = {}
        self._autocommit_tabs: set[str] = set()
        persisted = _load_last_model()
        self._default_model = persisted or os.environ.get("KISS_MODEL", "") or get_default_model()
        # The newest ``selectModel`` pick as ``(generation, model)``
        # (set under ``_state_lock``), and the count of picks whose
        # write to the file is still in flight: ``_refresh_default_model``
        # adopts the persisted model only when no pick is pending and
        # none landed while it read the file, and a pick's own write
        # ends by writing the newest pick when a later one superseded it.
        self._model_pick: tuple[int, str] = (0, "")
        self._model_picks_pending = 0
        self._state_lock = agent_state.STATE_LOCK
        # ``_cmd_run`` waits here while ``_history_deletes`` is non-zero
        # (see ``_CommandsMixin``).
        self._history_delete_done = threading.Condition(self._state_lock)
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
        self._file_index = FileIndexRegistry()
        # Connection id -> token of its latest ``getFiles`` request.
        # Entries are short-lived: each is removed once the request it
        # names has been answered or abandoned (autocomplete.py), so
        # the map only ever holds connections with a build in flight.
        self._files_latest_request: dict[str, object] = {}
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
        the canonical daemon (the channel launcher's private-endpoint
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

    @staticmethod
    def _run_orphan_sweep(still_running: set[str], boot_ts: float) -> None:
        """Run the orphan-task recovery sweep (background thread body).

        Rewrites the ``"Agent Failed Abruptly"`` sentinel on
        ``task_history`` rows abandoned by a prior, now-dead process
        (see :func:`_recover_orphaned_tasks`).  Executed on the
        ``orphan-task-sweep`` daemon thread started by ``__init__`` so
        that SQLite lock contention (``busy_timeout`` is 30 s) or a
        slow sweep over a large database can never delay server
        startup — the WSS listeners must bind promptly after an
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

        The same thread then stamps ``is_side_channel`` on task-update
        child rows persisted by releases that predate the flag (see
        :func:`mark_legacy_updates_as_side_channels`); without the stamp
        every chat reload re-opens each finished periodic update as a
        dead sub-agent tab.  It is another ``UPDATE`` behind the
        persistence write lock, so it must live here and never on the
        listener-binding path.

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
        try:
            stamped = mark_legacy_updates_as_side_channels()
            if stamped:
                logger.info(
                    "Stamped %d legacy task-update rows as side channels",
                    stamped,
                )
        except Exception:
            logger.warning("could not stamp legacy task-update rows", exc_info=True)
        finally:
            _close_thread_db()

    def drop_connection_state(self, conn_id: str) -> None:
        """Discard per-connection autocomplete state for a closed connection.

        Called by :class:`RemoteAccessServer` when a client connection
        (one per browser tab) goes away, so the
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
            self._files_latest_request.pop(conn_id, None)

    def _broadcast_tabs_state(self) -> None:
        """Send the full tab registry snapshot to EVERY connected client.

        Called after every registry mutation (open, close, rebind,
        retitle).  Each client reconciles
        their local tab bar against the full snapshot (idempotent and
        self-healing), so deltas are never needed.  The explicit empty
        ``tabId`` stamp routes the event through the printer's
        verbatim all-clients path — never recorded or persisted, and
        immune to the thread-local task-id injection when a mutation
        happens inside a task thread.

        Commands from different connections run concurrently, and a
        thread descheduled between its snapshot and its send could
        publish a snapshot OLDER than one another thread already sent;
        clients apply every snapshot verbatim, so the stale one closed
        the newer thread's freshly opened tab on every surface
        (conc1018 H2-F1).  Hence the mutation only REQUESTS a
        generation (``_tabs_state_requested``), and a single sender at
        a time (``_tabs_state_sender``) takes the snapshot and ships
        it, looping while the generation on the wire
        (``_tabs_state_sent``) is behind the requests
        (:meth:`_send_tabs_state_settled`): a snapshot taken by the
        sender after a request always reflects that mutation, and two
        snapshots never overtake each other.  Nobody waits for another
        thread's in-flight send (a close or a publication landing
        inside a run's publication returns at once): when a sender is
        busy, the request alone makes its next snapshot cover the
        caller's mutation.
        """
        locked_call(self._tabs_state_cond, self._tabs_state_request)
        self._send_tabs_state_settled()

    def _publish_tabs_state(self) -> None:
        """Broadcast the registry, and return only once a snapshot taken
        after the caller's mutation is on the wire.

        A publication's snapshot holds its own tab, and must precede
        the tab's status and transcript broadcasts: a mirroring client
        never receives events for a tab it has not yet seen in
        ``tabs_state``.  The caller waits for the current sender to
        ship every request made so far (its own among them); when
        there is no live sender (the sender was interrupted, or has
        quit), the caller sends it itself.
        """
        self._broadcast_tabs_state()
        while not locked_call(self._tabs_state_cond, self._await_tabs_state_sent):
            self._send_tabs_state_settled()

    def _send_tabs_state_settled(self, resend_if_interrupted: bool = True) -> None:
        """Send snapshots until the generation on the wire covers every request.

        Returns at once when another live thread is sending (its loop
        covers this thread's request).  Otherwise this thread is the
        sender until its loop finds nothing left to send, where it
        hands the role back under the condition.  A registry mutation
        can run on a task thread, where the runner's Stop injects
        ``KeyboardInterrupt`` at any bytecode (see ``kiss.core.locked``),
        so the role is also handed back in the ``finally``; a request
        the interrupted sender leaves unsent is resent by a helper
        thread, which nobody can interrupt, because the requester may
        be a close that is long gone and waits for nothing.

        Args:
            resend_if_interrupted: Spawn the helper thread on an
                exceptional exit with a request left unsent.  The
                helper itself passes ``False`` so a printer that fails
                every send does not respawn helpers forever.
        """
        try:
            while True:
                generation = locked_call(self._tabs_state_cond, self._tabs_state_next)
                if generation is None:
                    return
                self.printer.broadcast(
                    {
                        "type": "tabs_state",
                        "tabs": self.tab_registry.snapshot(),
                        "tabId": "",
                    }
                )
                locked_call(self._tabs_state_cond, self._tabs_state_mark_sent, generation)
        finally:
            unsent = locked_call(self._tabs_state_cond, self._tabs_state_abandon)
            if unsent and resend_if_interrupted:
                threading.Thread(
                    target=self._send_tabs_state_settled,
                    args=(False,),
                    name="tabs-state-resend",
                    daemon=True,
                ).start()

    def _tabs_state_request(self) -> int:
        """Under ``_tabs_state_cond``: request a snapshot taken from now on."""
        self._tabs_state_requested += 1
        return self._tabs_state_requested

    def _tabs_state_next(self) -> int | None:
        """Under ``_tabs_state_cond``: claim or keep the sender role.

        Returns:
            The generation the next snapshot will cover, or ``None``
            when nothing is left to send (the role is handed back) or
            another live thread holds the role.
        """
        me = threading.current_thread()
        sender = self._tabs_state_sender
        if sender is not None and sender is not me and sender.is_alive():
            return None
        if self._tabs_state_sent >= self._tabs_state_requested:
            self._tabs_state_sender = None
            self._tabs_state_cond.notify_all()
            return None
        self._tabs_state_sender = me
        return self._tabs_state_requested

    def _tabs_state_mark_sent(self, generation: int) -> None:
        """Under ``_tabs_state_cond``: a snapshot covering *generation* is on the wire."""
        self._tabs_state_sent = max(self._tabs_state_sent, generation)
        self._tabs_state_cond.notify_all()

    def _tabs_state_abandon(self) -> bool:
        """Under ``_tabs_state_cond``: hand back the sender role if this thread holds it.

        Returns:
            Whether a request was left unsent (always ``False`` after a
            normal exit, where :meth:`_tabs_state_next` handed the
            role back only once everything was sent).
        """
        if self._tabs_state_sender is not threading.current_thread():
            return False
        self._tabs_state_sender = None
        self._tabs_state_cond.notify_all()
        return self._tabs_state_sent < self._tabs_state_requested

    def _await_tabs_state_sent(self) -> bool:
        """Under ``_tabs_state_cond``: wait until every request so far is on the wire.

        Returns:
            ``False`` when no live sender exists before then: the
            caller sends itself.
        """
        generation = self._tabs_state_requested
        while self._tabs_state_sent < generation:
            sender = self._tabs_state_sender
            if sender is None or not sender.is_alive():
                return False
            self._tabs_state_cond.wait(timeout=1.0)
        return True

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
            tab_id,
            chat_id=chat_id,
            title=title,
            work_dir=work_dir,
            scope_work_dir=scope_work_dir,
            task_id=task_id,
            create=create,
        )
        for old_tab_id, removal_token in displaced:
            self._drop_tab_state(old_tab_id, removal_token=removal_token)
        if changed:
            self._publish_tabs_state()
        return generation

    def ready_tab_sync(
        self,
        restored: list[dict[str, str]],
    ) -> tuple[list[tuple[str, str, str]], bool]:
        """Synchronize a (re)connecting client with the tab registry.

        Adopts the client's legacy ``restoredTabs`` when the registry
        is still empty (one-time migration from pre-registry clients),
        rebinds the in-memory chat views from the registry, and
        broadcasts the canonical ``tabs_state`` snapshot.

        Args:
            restored: Sanitized ``restoredTabs`` entries from the
                client's ``ready`` command.

        Returns:
            ``(bound, adopted)``.  ``bound`` holds ``(tab_id, chat_id,
            task_id)`` triples for every chat-bound registry tab — the
            caller replays each to the (re)connecting client so it
            converges on the transcripts the other clients show.
            ``task_id`` is the specific historical task the tab was
            resumed to (``""`` when the tab tracks the chat's latest
            task); replaying it verbatim keeps a tab pinned to an
            older task from being silently switched to the chat's
            latest task by any client's reconnect.  ``adopted`` is
            ``True`` when this call seeded the empty registry from
            *restored*: the tabs are then new to EVERY connected
            client, so the caller broadcasts their replays instead of
            scoping them to the announcing connection.
        """
        adopted = self.tab_registry.merge_if_empty(restored)
        bound = self.tab_registry.bound_tabs()
        with self._state_lock:
            for tab_id, chat_id, _task_id in bound:
                self._tab_chat_views.setdefault(tab_id, chat_id)
        self._publish_tabs_state()
        return bound, adopted

    def _tab_model(self, tab_id: str) -> str:
        """Return the model selected for *tab_id* (default when unset).

        Args:
            tab_id: The frontend tab identifier string.

        Returns:
            The tab's selected model name.
        """
        return locked_call(self._state_lock, self._tab_model_locked, tab_id)

    def _tab_model_locked(self, tab_id: str) -> str:
        """Under ``_state_lock``: *tab_id*'s pick or the default model."""
        return self._tab_models.get(tab_id, "") or self._default_model

    def _tab_chat_id_locked(self, tab_id: str) -> str:
        """Under ``_state_lock``: the chat *tab_id* shows, or ``""``.

        The running task's chat when the tab is bound to one, else the
        chat a resumed (finished) session associated with the tab
        (``_tab_chat_views``).
        """
        state = agent_state.find_by_tab(tab_id) if tab_id else None
        chat_id = state.chat_id if state is not None else ""
        return chat_id or self._tab_chat_views.get(tab_id, "")

    def _handle_command(self, cmd: dict[str, Any]) -> None:
        """Dispatch a client command to the appropriate handler."""
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

    def _refresh_default_model(self, valid: set[str] | None = None) -> None:
        """Re-read the persisted last model and adopt it as the default.

        ``kiss-web`` can outlive its clients.  A fresh page load asks
        this long-lived daemon for ``getModels``; if the user selected
        a different model in a previous session, that choice is
        persisted in ``config.json`` and must
        take precedence over this process's stale in-memory default.

        The file is read OUTSIDE ``_state_lock`` (the lock is on the
        path of every tab's commands, and ``_cmd_select_model`` must
        not hold it while its write waits for the history database);
        call this before taking the lock, not under it.  The in-memory
        default is newer than the file while a pick of this process is
        still being persisted, and a pick that lands while the file is
        read replaces ``_model_pick``: in both cases the file is left
        alone, so it never clobbers a fresh pick.

        Args:
            valid: When given, adopt the persisted model only when it
                is in this set of currently-runnable model names.
        """
        with self._state_lock:
            if self._model_picks_pending:
                return
            pick = self._model_pick
        persisted = _load_last_model()
        with self._state_lock:
            if persisted and (valid is None or persisted in valid) and self._model_pick == pick:
                self._default_model = persisted

    def _printer_cleanup_tab(self, tab_id: str, keep_task_id: str = "") -> None:
        """Drop the printer's per-tab subscriptions/state for *tab_id*.

        Resolved via ``getattr`` because some duck-typed test printers
        implement only the broadcast/subscribe subset of the printer
        protocol.

        Args:
            tab_id: The frontend tab identifier to clean up.
            keep_task_id: A task whose subscription of *tab_id* is
                kept (see :meth:`JsonPrinter.cleanup_tab`).
        """
        cleanup_tab = getattr(self.printer, "cleanup_tab", None)
        if cleanup_tab is not None:
            cleanup_tab(tab_id, keep_task_id)

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
        catalog_names = ranked_function_calling_models()
        for name in catalog_names:
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

        from kiss.core.user_config import (
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
            models_list.insert(
                0,
                {
                    "name": cm["name"],
                    "inp": 0,
                    "out": 0,
                    "uses": usage.get(cm["name"], 0),
                    "vendor": "Custom",
                    "endpoint": cm["endpoint"],
                    "api_key": cm["api_key"],
                    "extra_headers": _parse_custom_headers(cm["headers"]),
                },
            )

        # A SEA whose ``register_as_model()`` returns True (``autorouter``,
        # ``bestrouter``) is listed under its command name but is not a
        # model: picking it runs every task through the SEA
        # (``_resolve_sea_model`` in the task runner) on catalog models —
        # so the entries are offered only when at least one catalog model
        # is runnable, and each is a valid pick but never the fallback a
        # stale pick lands on (that stays the first real model).  They
        # head the list.
        routers = sea_commands.model_seas() if catalog_names else {}
        available_names = {m["name"] for m in models_list} | set(routers)
        self._refresh_default_model(available_names)
        with self._state_lock:
            if self._default_model not in available_names:
                refreshed = get_default_model()
                if refreshed in available_names:
                    self._default_model = refreshed
                elif models_list:
                    self._default_model = str(models_list[0]["name"])
                else:
                    self._default_model = refreshed
            selected = self._default_model

        models_list[:0] = [
            {
                "name": name,
                "inp": 0,
                "out": 0,
                "uses": usage.get(name, 0),
                "vendor": "Router",
                "cost_label": "routes by its own protocol",
            }
            for name in routers
        ]

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
        in-progress executor's ``step_count``, and sets
        ``awaiting_answer`` when the task is blocked on an
        ``ask_user_question`` (the History panel then shows a ``?``
        instead of the running spinner).  Acquires ``_state_lock``
        internally (re-entrant, so safe to call with it already held).

        The usage triple is read through ONE
        :func:`~kiss.agents.sorcar.sorcar_agent._live_agent_usage` call
        (the banked ledger via ``usage_snapshot()`` on a
        ``RelentlessAgent`` plus the in-flight executor session, each a
        coherent snapshot): three separate property reads each sum the
        append-only usage ledger afresh, and a concurrent attribution
        between two of those reads shows the monitor an impossible mix
        (e.g. the old cost with the new tokens/steps).  The executor's
        spend is included for all three fields, so the History row of a
        running task shows the same figure as its chat header rather
        than only the previous sessions' banked total.

        Args:
            session: The history session dict to update in place.
            task_id: The ``task_history.id`` of the running task.
        """
        from kiss.agents.sorcar.sorcar_agent import _live_agent_usage

        with self._state_lock:
            state = agent_state.get(task_id)
            if state is None:
                return
            session["awaiting_answer"] = bool(state.pending_ask_question)
            agent = state.agent
            if agent is None:
                return
            cost, tokens, steps = _live_agent_usage(agent)
            session["tokens"] = tokens
            session["cost"] = cost
            session["steps"] = steps or int(getattr(agent, "step_count", 0) or 0)
            mdl_live = getattr(agent, "model_name", "")
            if isinstance(mdl_live, str) and mdl_live:
                session["model"] = mdl_live
            session["is_parallel"] = state.use_parallel

    def _get_history(
        self,
        query: str | None,
        offset: int = 0,
        generation: int = 0,
        conn_id: str = "",
        tag: str = "",
        machine: str = "",
        work_dir: str = "",
    ) -> None:
        """Send conversation history with pagination support.

        ``tag`` (the history panel's tag dropdown) restricts the page to
        tasks carrying that classification tag; ``""`` means every task.

        ``machine`` (the panel's machine dropdown) picks whose tasks the
        page holds: ``""`` every registered machine's and this daemon's,
        ``LOCAL_MACHINE`` this daemon's only (what another daemon asks
        for), else the named remote machine's only.

        ``work_dir`` (the panel's WORKDIR dropdown) restricts the page
        to the tasks that ran in that exact directory.  The options
        the dropdown offers travel in the page-one reply as
        ``workDirs``: the same directories as the chat header's
        working-directory drop-down, see :meth:`_work_dir_options`
        (later pages of the same view carry no such field).  A
        directory is a path on this machine, so a set
        ``work_dir`` scopes the page to this daemon's rows and asks no
        remote machine.

        The reply is stamped with the requesting connection's
        ``conn_id`` (when non-empty) so it reaches only the browser
        tab that asked — one tab's history search must not repaint
        another tab's history panel.
        """
        running_task_ids = self._get_running_task_ids()
        registered = load_machines()
        remote_only = bool(machine) and machine != LOCAL_MACHINE and not work_dir
        if machine == LOCAL_MACHINE or work_dir:
            machines: list[Machine] = []
        elif remote_only:
            machines = [m for m in registered if m.name == machine]
        else:
            machines = registered
        # With remote machines in the page it is cut from the union of
        # every machine's rows (``merged_history``), which needs the
        # local rows before the page too, not just the page itself.
        limit, local_offset = (offset + HISTORY_PAGE, 0) if machines else (HISTORY_PAGE, offset)
        entries = (
            []
            if remote_only
            else _load_history(
                limit=limit,
                offset=local_offset,
                tag=tag,
                query=query or "",
                work_dir=work_dir,
                running_task_ids=running_task_ids,
            )
        )

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
                "failed": (_is_failed_result(result, entry.get("success")) and not is_running),
                "is_running": is_running,
                "awaiting_answer": False,
                "tokens": 0,
                "cost": 0.0,
                "steps": 0,
                "is_favorite": False,
                "work_dir": "",
                "model": "",
                "is_parallel": False,
                # A cron-launched task (persistence ``is_cron``): the
                # CHATS list hides it unless its Cron chip is on.
                "is_cron": False,
                "startTs": _safe_start_ms(entry.get("timestamp", 0)),
                "endTs": 0,
            }
            extra_obj = _extra_dict(entry.get("extra"))
            if extra_obj:
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
                    ("max_budget", float, 0.0),
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
                session["is_parallel"] = bool(extra_obj.get("is_parallel", False))
                session["is_cron"] = bool(extra_obj.get("is_cron", False))
                for key in ("tags", "sea"):
                    raw = extra_obj.get(key, "")
                    session[key] = raw if isinstance(raw, str) else ""
                start_ts = _start_ts_from_extra(extra_obj)
                if start_ts:
                    session["startTs"] = start_ts
            if session.get("is_running") and entry_id is not None:
                self._overlay_live_metrics(session, entry_id)
            sessions.append(session)
        # The chat-panel headers in the History sidebar show each chat's
        # summary (``chat_summaries``, once a task has finished) or else
        # its FIRST task, which may be older than any row on this page,
        # and the launch time of its LATEST task (``chat_summaries``).
        chat_ids = [str(s["id"]) for s in sessions]
        first_tasks = _chat_first_tasks(chat_ids)
        summaries = _chat_summaries(chat_ids)
        for session in sessions:
            chat_id = str(session["id"])
            session["chat_first_task"] = first_tasks.get(chat_id, "")
            summary = summaries.get(chat_id, {})
            session["chat_summary"] = summary.get("summary", "")
            session["chat_last_launched"] = summary.get("last_launched", 0)
            session["chat_last_finished"] = summary.get("last_finished", 0)
        min_ts, max_ts = (None, None) if remote_only else _history_date_range()
        if machines or remote_only:
            page = merged_history(
                HistoryPage(sessions, min_ts, max_ts),
                machines,
                query or "",
                tag,
                offset,
                HISTORY_PAGE,
            )
            sessions, min_ts, max_ts = page.rows, page.date_min, page.date_max
        event: dict[str, Any] = {
            "type": "history",
            "sessions": sessions,
            "offset": offset,
            "generation": generation,
            "dateRange": {"min": min_ts, "max": max_ts},
            "machines": [m.name for m in registered],
            # What the CHATS panel calls each registered machine: its
            # own host name when it was registered by its address.
            "machineLabels": {m.name: machine_label(m) for m in registered},
            "localMachine": local_machine_name(),
        }
        # The WORKDIR dropdown's options, sent with page one of a view
        # only; the later pages of a scroll change nothing.
        if offset == 0:
            event["workDirs"] = self._work_dir_options()
        self._broadcast_to_conn(event, conn_id)

    def _work_dir_options(self) -> list[str]:
        """Return the directories the history panel's WORKDIR filter offers.

        The same list as the chat header's working-directory drop-down:
        every directory adopted before (``config.json``
        ``recent_work_dirs``, :func:`kiss.core.user_config.recent_work_dirs`),
        most recently adopted first.  Directories that no longer exist
        are left out.  The daemon's current working directory is added
        at the front when it is not already listed (never recorded, or
        no longer existing): it is the one in use.  The directories the
        history merely ran in (sub-agent worktrees, cron run
        directories) are not offered: a filter on those would be noise.
        """
        from kiss.core.user_config import recent_work_dirs

        paths = [str(entry["path"]) for entry in recent_work_dirs()]
        current = self.work_dir
        listed = {os.path.normpath(p) for p in paths}
        if current and os.path.normpath(current) not in listed:
            paths.insert(0, current)
        return paths

    def _get_input_history(self, conn_id: str = "") -> None:
        """Send deduplicated composer texts for arrow-key cycling.

        Loads the full persisted history — every task stored in
        ``history.db`` plus every message typed into a running task
        (``steer_inputs``) — so ArrowUp can traverse each distinct
        text, not just an arbitrary recent subset.  Stamped with the
        requesting connection's ``conn_id`` (when non-empty) so the
        reply reaches only the window that asked.

        Args:
            conn_id: Requesting connection id (``""`` for direct callers).
        """
        tasks = list(dict.fromkeys(t for raw in _load_input_history() if (t := raw.strip())))
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
            self._broadcast_tabs_state()
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
        # generation at or below the reading; ``reopened_since``
        # closes that gap by also treating registry PRESENCE as a
        # reopen (presence after an absent-close is always a later
        # republication).  A tab that was never published (sub-agent
        # tabs) has generation 0 and is never present, so those are
        # still dropped unconditionally.
        self._drop_tab_state(
            tab_id,
            removal_token=removal_token or self.tab_registry.clock(),
        )

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
                if self.tab_registry.generation(tab_id) != fallback_publication:
                    # A token-0 close retired the stamp, or a newer
                    # publication (which performs its own commit)
                    # superseded it: this stale commit owns nothing.
                    return
            if source is not None and source.running():
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
            tab_id,
            chat_id,
            publication,
            source,
            fallback_publication=fallback,
        )

    def _drop_tab_state(
        self,
        tab_id: str,
        removal_token: int | None = None,
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
        :meth:`TabRegistry.clock` right after :meth:`TabRegistry.close_tab`
        instead, so a reopen republishing the tab after the close
        observed it absent makes this drop stand down exactly like a
        real removal's tail; a tab that was never published (sub-agent
        tabs) has generation 0 — never greater than any token — and is
        dropped unconditionally, as before.  ``None`` (direct callers with no
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
        # Reached from the task thread too (``_registry_update_tab``
        # displacing an older tab at task startup), so the locked
        # section goes through ``locked_call`` (``kiss.core.locked``).
        teardown, state = locked_call(
            self._state_lock,
            self._drop_tab_state_locked,
            tab_id,
            removal_token,
        )
        if teardown:
            self._teardown_tab_resources(tab_id, state, removal_token=removal_token)

    def _drop_tab_state_locked(
        self,
        tab_id: str,
        removal_token: int | None,
    ) -> tuple[bool, AgentState | None]:
        """Under ``_state_lock``: the close's linearization point of :meth:`_drop_tab_state`.

        Returns:
            ``(True, state)`` when the caller must run the unlocked
            teardown tail for the claimed *state*; ``(False, None)``
            when the drop stood down or the tab is busy.
        """
        if removal_token is not None and self.tab_registry.reopened_since(
            tab_id,
            removal_token,
        ):
            # A later publication reopened the tab between the
            # registry removal and this cleanup (a newer generation,
            # or — for a token-0 duplicate close — a registry row
            # published after the close found the tab absent); it owns
            # the state now (its own rebind runs under this same lock).
            return False, None
        state = agent_state.find_by_tab(tab_id)
        is_subagent_tab = (state is not None and state.is_subagent) or "__sub_" in tab_id
        busy = False
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
                        tab_id,
                        removal_token,
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
        # close event instead.  Every client removes the tab, so
        # a torn-down shared per-tab printer subscription cannot
        # starve a client that still shows the tab.  Recorded and
        # broadcast in the same locked step as
        # ``_open_persisted_subagent_tabs`` checks the record and
        # announces a live child, so the close is ordered wholly
        # before that announcement (which then stands down) or
        # wholly after it (and closes the tab it opened).
        if is_subagent_tab:
            self._broadcast_subagent_close(tab_id)
            self._remember_closed_subagent_tab(tab_id)
        if busy:
            # The task keeps running with no tab showing it.  Drop
            # the tab's printer subscriptions now (the task's
            # recording is per task and survives): the viewer
            # registry then lists only OPEN tabs, which is what
            # ``SorcarAgent._subagent_parent_tab_id`` relies on to
            # parent later spawns under the tab a history reopen
            # gave the chat instead of this closed one.  Under the
            # lock, after the reopen check above: a replay that
            # republished this id either landed before (this close
            # stood down) or commits after, re-subscribing the tab
            # (``_commit_replay_publication``).
            self._printer_cleanup_tab(tab_id)
            return False, None
        return True, state

    def _remember_closed_subagent_tab(self, tab_id: str) -> None:
        """Under ``_state_lock``: record that the user closed the running sub-agent tab *tab_id*.

        :meth:`_open_persisted_subagent_tabs` then leaves the tab closed
        on a replay (a surface connecting mid-run) until the sub-agent
        ends or a client reopens the tab explicitly (``resumeSession``,
        see :meth:`_replay_session`).  Entries of finished sub-agents
        are pruned here, so the set never outgrows the number of
        running sub-agents.  Every access to the set happens under
        ``_state_lock``.

        Args:
            tab_id: The shared sub-agent tab identifier
                (``{parent_tab_id}__sub_{task_id}``).
        """
        self._closed_subagent_tabs.add(tab_id)
        self._closed_subagent_tabs.difference_update(
            {
                t
                for t in self._closed_subagent_tabs
                if _no_live_state_locked(t.rsplit("__sub_", 1)[-1])
            }
        )

    def _broadcast_subagent_close(self, tab_id: str) -> None:
        """Tell every client to close the sub-agent tab *tab_id*.

        Sub-agent tabs are derived state shared under ONE tab id by
        every client, but they never live in the tab registry — so a
        close on one client must be mirrored with this dedicated
        broadcast (clients apply it without echoing ``closeTab`` back).

        Args:
            tab_id: The shared sub-agent tab identifier.
        """
        self.printer.broadcast(
            {
                "type": "closeSubagentTab",
                "tab_id": tab_id,
                # The explicit tabId stamp routes the event through the
                # printer's verbatim all-clients path (never recorded).
                "tabId": "",
            }
        )

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
        # Runs on the task thread at every lifecycle transition, so
        # the claim goes through ``locked_call`` (``kiss.core.locked``).
        claim = locked_call(self._state_lock, self._claim_closed_tab_locked, tab_id)
        if claim is not None:
            state, token = claim
            self._teardown_tab_resources(tab_id, state, removal_token=token)

    def _claim_closed_tab_locked(self, tab_id: str) -> tuple[AgentState, int] | None:
        """Under ``_state_lock``: claim a closed, idle *tab_id* for disposal.

        Returns:
            ``(state, removal_token)`` for :meth:`_teardown_tab_resources`,
            or ``None`` when the tab is open, busy, reopened or gone.
        """
        state = agent_state.find_by_tab(tab_id)
        if state is None or not state.frontend_closed:
            return None
        if state.busy():
            return None
        if self.tab_registry.has_tab(tab_id):
            # A row exists only through a publication AFTER the
            # close that set ``frontend_closed`` (the close removed
            # the row): the reopen owns the tab, and its commit
            # clears the flag under this same lock.
            return None
        state.is_merging = True
        state.merge_thread = threading.current_thread()
        return state, self.tab_registry.retire_unregistered_and_observe(tab_id)

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
        that decision durable (:meth:`SorcarAgent.retire_for_disposal`
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
        try:
            if wt_agent is not None and getattr(wt_agent, "_wt_pending", False):
                try:
                    wt_agent.retire_for_disposal()
                    wt_agent._flush_warnings(self.printer)
                except Exception:  # pragma: no cover — git/printer failure
                    logger.debug("Worktree release on tab close failed", exc_info=True)
        finally:
            # The claim is released whatever the retire did — a
            # ``BaseException`` escaping it (the stop interrupt the
            # model layer raises for a failed auto-commit message when
            # the thread's stop signal is set) once left ``is_merging``
            # stuck, and every later main-tree run on the repo was
            # refused as "a worktree merge in progress".  The agent
            # keeps ``_wt`` exactly while its claim is retained
            # (``retire_for_disposal`` returns ``_wt is None``), so the
            # keep/unregister decision reads that instead of a result
            # the interrupted retire never returned.
            self._finish_tab_teardown(tab_id, state, removal_token)

    def _finish_tab_teardown(
        self,
        tab_id: str,
        state: AgentState | None,
        removal_token: int | None,
    ) -> None:
        """Release the disposal claim and drop the tab's server-side state.

        The locked tail of :meth:`_teardown_tab_resources`; see there
        for the ownership rules.  The state stays registered when its
        agent still holds a worktree claim (the retire kept the
        worktree but could not make that decision durable, or never
        finished), so a later disposal attempt can retry.

        Args:
            tab_id: The frontend tab identifier being disposed.
            state: The claimed agent state, or ``None`` when the tab
                never ran a task.
            removal_token: The registry observation taken at claim
                time, or ``None`` for callers with no registry ordering.
        """
        wt_agent = state.agent if state is not None else None
        claim_retained = bool(wt_agent is not None and getattr(wt_agent, "_wt_pending", False))
        # On the task thread for a displaced or deferred-disposed tab:
        # ``locked_call``, never ``with`` (see ``kiss.core.locked``).
        locked_call(
            self._state_lock,
            self._finish_tab_teardown_locked,
            tab_id,
            state,
            removal_token,
            claim_retained,
        )

    def _finish_tab_teardown_locked(
        self,
        tab_id: str,
        state: AgentState | None,
        removal_token: int | None,
        claim_retained: bool,
    ) -> None:
        """Under ``_state_lock``: the destructive tail of :meth:`_finish_tab_teardown`."""
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
        if state is not None:
            state.is_merging = False
            state.merge_thread = None
        if removal_token is not None:
            if not self.tab_registry.finalize_removal(
                tab_id,
                removal_token,
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

    def _pin_opened_task(self, tab_id: str, task_id: str | None) -> None:
        """Record which task *tab_id* was opened on, under ``_state_lock``.

        Args:
            tab_id: The frontend tab id.
            task_id: The task the tab is pinned to; ``""``/``None``
                unpins it (the tab tracks the chat's latest task).
        """
        with self._state_lock:
            if task_id:
                self._tab_opened_task_ids[tab_id] = str(task_id)
            else:
                self._tab_opened_task_ids.pop(tab_id, None)

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
        self._refresh_default_model()
        with self._state_lock:
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

    def _defer_replay(
        self,
        chat_id: str,
        tab_id: str,
        task_id: str | None,
        conn_id: str,
    ) -> None:
        """Offer *tab_id*'s transcript to one client instead of sending it.

        A (re)connecting client's ``ready`` replays only the tab it is
        showing; every other chat-bound registry tab goes through here,
        which reads nothing from the events table.  The client learns
        the tab's running state (its header spinner), any pending
        question, and gets a ``replay_deferred`` event it answers with
        a connection-scoped ``resumeSession`` when the user opens the
        tab.  Twenty open tabs used to cost a reconnecting laptop
        twenty full transcripts (hundreds of megabytes over a tunnel)
        and froze the daemon while it loaded and serialized them.

        The running state is always re-sent, whatever flag the client
        kept from before its outage: ``running:true`` for a tab whose
        task is still going (the attached source's optimistic status,
        or the tab's own run through :meth:`_broadcast_replay_status`
        when the attach resolved nothing), ``running:false`` for an
        idle tab, so a background tab's spinner matches the daemon
        even though the client missed the run's start or end.

        Args:
            chat_id: The chat the registry binds *tab_id* to.
            tab_id: The frontend tab id.
            task_id: The task the tab is pinned to (``""``/``None``
                when it tracks the chat's latest task).
            conn_id: The connection the deferral is scoped to.
        """
        if not tab_id:
            return
        self._pin_opened_task(tab_id, task_id)
        source, _source_task_id, _recording = self._attach_viewer_to_running_chat(
            chat_id,
            tab_id,
            task_id=task_id or None,
        )
        with self._state_lock:
            # Decided and sent under the lock, like every viewer status
            # broadcast: a run the user starts on this tab meanwhile
            # publishes its ``running:true`` under the same lock, and a
            # stale ``running:false`` after it would kill its spinner.
            owner_decides = source is None or self._viewer_owns_other_busy_run(
                tab_id, source,
            )
            if owner_decides:
                # Nothing live attached, or the tab's own busy run
                # outranks the pinned source it views (the optimistic
                # status below would be withheld): the owner decides,
                # and only the (re)connecting client needs the answer.
                self._broadcast_replay_status(tab_id, {"connId": conn_id})
        if not owner_decides:
            assert source is not None
            self._broadcast_viewer_running(
                tab_id,
                source,
                int(getattr(source.agent, "_task_start_ms", 0) or 0),
            )
            # The source may have ended inside the attach window: its
            # end-of-run fan-out already told EVERY client
            # ``running:false`` BEFORE the optimistic ``running:true``
            # above (which, like the tab, reaches every surface), so
            # the correction must reach every client too — scoped to
            # one connection it stranded the other surfaces' spinners
            # (audit 2026-10-12).  No transcript was replayed, so the
            # terminal result is not re-sent here.
            self._finalize_viewer_attach(tab_id, source, [], {}, resend_result=False)
        self.printer.broadcast(
            {
                "type": "replay_deferred",
                "tabId": tab_id,
                "chatId": chat_id,
                "taskId": task_id or "",
                "connId": conn_id,
            }
        )
        self._emit_pending_ask(tab_id)
        self._emit_pending_worktree(tab_id)
        # The tab's sub-agent tabs are announced too (a running one
        # must be open on every surface), their transcripts deferred
        # the same way.
        parent_task_id = task_id or ""
        if not parent_task_id:
            row = _load_latest_chat_events_by_chat_id(chat_id, with_events=False)
            parent_task_id = (_coerce_id(row.get("task_id")) or "") if row else ""
        if parent_task_id:
            self._open_persisted_subagent_tabs(
                parent_task_id=parent_task_id,
                parent_tab_id=tab_id,
                conn_id=conn_id,
                transcripts=False,
            )

    def _replay_session(
        self,
        chat_id: str,
        tab_id: str = "",
        task_id: str | None = None,
        conn_id: str = "",
        replay_id: str = "",
    ) -> None:
        """Replay recorded chat events for a previous chat session.

        Sets the tab's agent chat_id to match the resumed session.
        The tab_id (frontend routing key) does not change.

        When ``tab_id`` is empty the call is a no-op — the previous
        behavior of synthesizing a phantom tab keyed by ``chat_id``
        violated per-tab state isolation (C2/C3 fix).

        Loading a chat never touches the tab's ``use_parallel`` /
        ``selected_model``: those mirror the toolbar toggles, which
        are global UI state the user owns.

        Args:
            chat_id: The string chat session identifier to replay.
            tab_id: The frontend tab identifier.
            task_id: Optional task row ID.  When provided, load this
                specific task instead of the latest task in the chat
                session.  This is used when the user clicks a specific
                task in the history panel.
            conn_id: When non-empty, the ``task_events`` transcripts are
                delivered ONLY to this connection.  A (re)connecting
                client's ``ready`` resumes every bound registry tab to
                rebuild its own view; broadcasting those replays would
                make every other window rebuild transcripts it already
                shows.  ``""`` (a user's history click) reaches every
                client, since the other windows mirror the same tab and
                have no other way to learn its new transcript.  The tab
                binding (``tabs_state``) and a live task's
                ``running:true`` are always broadcast; the status
                that precedes a transcript with nothing live attached
                (:meth:`_broadcast_replay_status`) is scoped like the
                transcript.
            replay_id: When non-empty, echoed as ``replayId`` on every
                frame of the answer (the transcript, the status or
                ``subagentDone`` standing in for it, the sub-agent
                announcements and transcripts that follow), so a
                daemon that forwarded the resume for one of its
                clients (:mod:`kiss.server.remote_tabs`) can tell the
                answer from the tab's other frames.
        """
        if not tab_id:
            logger.debug("_replay_session called without tab_id; ignoring")
            return
        replay_scope: dict[str, Any] = {"connId": conn_id} if conn_id else {}
        if replay_id:
            replay_scope["replayId"] = replay_id
        self._pin_opened_task(tab_id, task_id)
        row = None
        if task_id is not None:
            row = _load_chat_events_by_task_id(task_id)
            if row:
                chat_id = str(row.get("chat_id", "") or chat_id)
        if not row:
            row = _load_latest_chat_events_by_chat_id(chat_id)
        no_row = not row
        # A run allocates its history row after setup, so the chat can
        # be live without a row: replay it as an empty row whose
        # transcript is the live recording snapshot below.
        result: dict[str, Any] = row or {
            "task": "",
            "task_id": task_id,
            "chat_id": chat_id,
            "extra": "",
            "events": [],
        }
        extra_raw = _extra_dict(result.get("extra"))
        sub = extra_raw.get("subagent")
        subagent_info: dict[str, object] | None = sub if isinstance(sub, dict) else None

        rebound_task_id = _coerce_id(result.get("task_id"))
        rebound_state, attached_task_id, recording = self._attach_viewer_to_running_chat(
            chat_id,
            tab_id,
            task_id=rebound_task_id,
            is_subagent=subagent_info is not None,
            replace_subscriptions=True,
        )
        _race_delay()  # test hook: widens the attach-to-commit window
        publish_tab = subagent_info is None and bool(chat_id)
        if no_row:
            # Without a row the live recording is the only copy of what
            # the run has broadcast (the events table is written
            # asynchronously), and a pre-row run's recording (e.g. its
            # setup-failure result) is keyed by the provisional id the
            # state carries — not by the caller's *task_id*, which is
            # None for a plain chat resume (audit0903 F4).  The task
            # panel heading comes from the prompt ``_cmd_run`` stamped
            # on the state, so it is not blanked for the whole setup
            # window.  A sub-agent view tab is not a viewer of the
            # (parent) chat, matching the ``subagent_info`` guard a
            # row provides.
            rebound_task_id = attached_task_id or None
            with self._state_lock:
                state = agent_state.find_by_tab(tab_id)
                publish_tab = publish_tab and not (state is not None and state.is_subagent)
            if rebound_state is not None:
                result["task"] = rebound_state.last_user_prompt
        if publish_tab:
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
            # Published before the status and transcript broadcasts
            # below, so a mirroring client never receives events for
            # a tab it has not yet seen in ``tabs_state``.
            self._publish_replay_reopen(
                tab_id,
                chat_id,
                task_id,
                rebound_state,
                title=None if no_row else str(result.get("task", "") or ""),
            )
        elif subagent_info is None:
            with self._state_lock:
                state = agent_state.find_by_tab(tab_id)
                if state is not None:
                    state.frontend_closed = False
                self._tab_chat_views.pop(tab_id, None)
        if no_row and rebound_state is None:
            # Nothing to replay and nothing running: say so, or the
            # client keeps waiting for a replay that never comes
            # (the page holds a registry chat open until its
            # state is known, see ``statusKnown`` in main.js).
            self.printer.broadcast(
                {
                    "type": "status",
                    "running": False,
                    "tabId": tab_id,
                    **replay_scope,
                }
            )
            self._emit_pending_ask(tab_id)
            return

        if subagent_info is not None:
            is_done = _subagent_is_done(result.get("task_id"))
            if is_done and subagent_info.get("side_channel"):
                # A finished side channel (the /ask answerer) has
                # nothing to show: its answer sits in the parent's
                # transcript.  The client that asked for this replay
                # opened the tab on ``new_tab`` before the run ended
                # (or restored it from its saved state); tell it to
                # close the tab instead of re-opening it finished.
                self.printer.broadcast(_subagent_done_event(tab_id, replay_scope))
                return
            announcement = self._subagent_announcement(
                result,
                None,
                _coerce_id(subagent_info.get("parent_task_id")) or "",
                tab_id,
                is_done,
                replay_scope,
            )
            # A client asked for this sub-agent's tab by name: a tab the
            # user had closed while it ran is open again.  Committed and
            # announced in ONE lock section, like
            # ``_open_persisted_subagent_tabs`` and
            # ``_commit_replay_publication``: a concurrent ``_close_tab``
            # of this tab runs its whole tail (mark closed, broadcast the
            # close, remember it, unsubscribe) under this same lock, so
            # it lands wholly before — the idempotent ``subscribe_tab``
            # here reinstalls the fan-out it removed — or wholly after,
            # closing the tab announced here.  Three separate sections
            # let a close slip between the attach above and this commit,
            # leaving the tab open on the requesting client yet
            # unsubscribed, its transcript frozen (conc1018 H1-F1).
            # Scoped like the transcript: a deferred child's fetch by
            # one window must not reopen the tab in the others (a user
            # may have closed it there).
            with self._state_lock:
                self._closed_subagent_tabs.discard(tab_id)
                state = agent_state.find_by_tab(tab_id)
                if state is not None:
                    state.frontend_closed = False
                self._tab_chat_views.pop(tab_id, None)
                if rebound_state is not None and rebound_state.running():
                    self.printer.subscribe_tab(rebound_state.task_id, tab_id)
                self.printer.broadcast(announcement)

        if rebound_state is not None:
            # The live agent's stamp (``_cmd_run``'s start, stamped by
            # ``_run_task_inner``) is what the originating client's
            # ``status running:true`` carried, so a viewer attaching
            # mid-run counts from the same instant; the row's
            # ``startTs`` covers a live state without a stamp (a
            # sub-agent).
            start_ts_for_resume = int(
                getattr(rebound_state.agent, "_task_start_ms", 0) or 0,
            ) or _start_ts_from_extra(extra_raw)
            self._broadcast_viewer_running(
                tab_id,
                rebound_state,
                start_ts_for_resume,
            )
        elif subagent_info is None:
            self._broadcast_replay_status(tab_id, replay_scope)
        replay_payload: dict[str, Any] = {
            "type": "task_events",
            "task": result["task"],
            "task_id": result.get("task_id"),
            "chat_id": chat_id,
            "extra": _extra_for_replay(result.get("extra", "")),
            "tabId": tab_id,
            **replay_scope,
        }
        # While the task runs the events table lags behind it: display
        # events reach the database through an asynchronous writer, and
        # a tab resumed moments after the task started (the round trip
        # a freshly spawned ``run_parallel`` sub-agent's ``new_tab``
        # triggers) would replay an EMPTY transcript.  The printer's
        # live in-memory recording is the authoritative copy while the
        # task runs.  Snapshotted after the reattach above subscribed
        # the tab, so each live event is either in this snapshot or
        # fanned out after the replay (see ``replay_snapshot``).  The
        # attach's recording is used only when it belongs to the row
        # being replayed (a chat-id fallback can attach another task).
        # Checked against the id captured with the recording, not the
        # state's current ``task_id``: a sequential run re-keys the same
        # state to its next subtask, possibly before this check.
        if attached_task_id != rebound_task_id:
            recording = None
        with self.printer.replay_snapshot(
            rebound_task_id if rebound_state is not None else "",
            conn_id,
            recording,
        ) as live_events:
            if live_events:
                result["events"] = live_events
            replayed_events = with_synthesized_events(
                _coalesced_replay_events(result["events"]),
                result,
            )
            self.printer.broadcast(
                {**replay_payload, "events": replayed_events},
            )
        if rebound_state is not None:
            self._finalize_viewer_attach(
                tab_id,
                rebound_state,
                replayed_events,
                replay_payload,
            )
        self._emit_pending_ask(tab_id)
        self._emit_pending_worktree(tab_id)

        # A resumed sub-agent may itself have fanned out: its running
        # descendants' tabs must reach this client too (its finished
        # ones stay history, reopened from its own fan-out panel).  A
        # connection-scoped replay (a ready, a deferred tab's fetch)
        # announces the children and defers their transcripts; a
        # history click replays them in full for every window.
        if rebound_task_id:
            self._open_persisted_subagent_tabs(
                parent_task_id=rebound_task_id,
                parent_tab_id=tab_id,
                conn_id=conn_id,
                replay_id=replay_id,
                live_only=subagent_info is not None,
                transcripts=not conn_id,
            )

    def _emit_pending_ask(self, tab_id: str) -> None:
        """Re-broadcast a still-pending ask-user question to *tab_id*.

        Session replays (``resumeSession``) repaint a tab's transcript
        but the ``askUser`` prompt is a live event: a client that
        connects or reloads while the tab's task is blocked inside
        ``ask_user_question`` would otherwise never see the question.
        Called after every ``task_events`` replay broadcast so such
        clients converge on the same modal every other client shows.

        The pending state is resolved exactly like ``userAnswer``
        routing (:meth:`_resolve_user_answer_state`): the state
        launched from *tab_id* itself, else the state of any task the
        tab is subscribed to.  The broadcast happens under
        ``_state_lock`` — the same lock ``_deliver_user_answer`` holds
        while clearing ``pending_ask_question`` and broadcasting the
        answer's ``askUserDone`` — so the re-emitted ``askUser`` is
        ordered either strictly before that close (and closed by it)
        or strictly after the question was cleared (and not emitted at
        all), guaranteeing no client is left with a stale modal.

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
        sub_task_id: str = "",
    ) -> str:
        """Return the frontend tab id of the parent agent owning the
        sub-agent currently being opened on *sub_tab_id*.

        Used to populate ``parent_tab_id`` on the ``openSubagentTab``
        broadcast so the page can record the parent → child
        relationship that drives cascade-close (see ``closeTab`` in
        media/main.js, which walks ``parentTabId`` chains).  A blank
        return value breaks that cascade, so this helper tries every
        signal we have before giving up.

        Lookup order:

        0. **Tab-id suffix match.**  Every chat page names a sub-agent's
           tab ``f"{parent_tab_id}__sub_{task_id}"`` after the parent
           tab it was spawned under (``subagentTabIdFor`` in
           media/main.js), so a *sub_tab_id* ending in
           ``__sub_{sub_task_id}`` settles the parent: the prefix.
           This is the only tier that can name a parent which is
           itself a sub-agent (a nested fan-out's grandchild; the
           tiers below skip sub-agent states and would hand the
           grandchild to the top-level chat tab instead).  The
           ``task`` prefix of a parentless spawn means "no parent".

        Then, each tier skipping sub-agent states:

        1. **Task-id match.**  Scan
           the agent-state registry for a non-subagent state registered
           under *parent_task_id*.  This is the primary, unambiguous
           match.

        2. **Chat-id match.**  Sub-agents inherit ``chat_id`` from
           the parent (see ``agent_dispatch.inherit_from_parent``).
           Scan for non-subagent states whose ``chat_id`` matches.
           If exactly one such state exists, use it.  More than one
           is ambiguous — bail out so we don't pick the wrong tab.

        3. **Synthetic-tab-id parse.**  Sub-agent tab ids are
           generated as ``f"{parent_tab_id}__sub_{sub_task_id}"`` by
           :meth:`_open_persisted_subagent_tabs`.  Split on
           ``"__sub_"`` and, if the prefix matches a known
           non-subagent ``tab_id``, use it.

        If every tier fails, log a WARNING (silent ``""`` would
        manifest as the cascade-close bug from a downstream
        feature) and return ``""``.
        """
        suffix = f"__sub_{sub_task_id}" if sub_task_id else ""
        if suffix and sub_tab_id.endswith(suffix):
            prefix = sub_tab_id[: -len(suffix)]
            # A live helper can outlive its parent. Keep its stable tab id,
            # but attach the visible tab to the nearest live ancestor.
            # Finished history keeps its real parent so a hand-close of
            # that parent still cascades.
            if not _subagent_is_done(sub_task_id):
                while "__sub_" in prefix:
                    ancestor, ancestor_task_id = prefix.rsplit("__sub_", 1)
                    if not _subagent_is_done(ancestor_task_id):
                        break
                    prefix = ancestor
            if prefix and prefix != "task":
                return prefix
        with self._state_lock:
            if parent_task_id is not None:
                parent = agent_state.get(parent_task_id)
                if parent is not None and not parent.is_subagent and parent.tab_id:
                    return parent.tab_id

            non_sub_states = [
                st for st in agent_state.agent_states.values() if not st.is_subagent and st.tab_id
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

    def _subagent_announcement(
        self,
        row: dict[str, Any],
        idx: int | None,
        parent_task_id: str,
        sub_tab_id: str,
        is_done: bool,
        scope: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the ``openSubagentTab`` event replaying a persisted sub-agent.

        Args:
            row: The sub-agent's row from
                :func:`_load_subagent_rows_by_parent_task_id` (or the
                one :meth:`_replay_session` loaded by task id).
            idx: The row's position among its parent's children
                (numbers the tab title), or ``None`` when the client
                asked for this one tab by name and keeps its title.
            parent_task_id: ``task_history.id`` of the parent task.
            sub_tab_id: The deterministic sub-agent tab id.
            is_done: Whether the sub-agent has finished.
            scope: ``{"connId": ...}`` to address one connection, else
                empty.

        Returns:
            The event dict to broadcast.
        """
        return {
            "type": "openSubagentTab",
            "tab_id": sub_tab_id,
            "parent_tab_id": self._resolve_parent_tab_id_for_sub(
                parent_task_id=parent_task_id,
                chat_id=str(row.get("chat_id", "") or ""),
                sub_tab_id=sub_tab_id,
                sub_task_id=str(row["task_id"]),
            ),
            "description": str(row.get("task", "") or ""),
            "task_id": row["task_id"],
            **({} if idx is None else {"taskIndex": idx}),
            "isSubagentTab": True,
            "isDone": is_done,
            # The frontend attributes the row to the fan-out call that
            # was running when it started.
            "startTs": _start_ts_from_extra(row.get("extra", "")),
            **scope,
        }

    def _open_persisted_subagent_tabs(
        self,
        *,
        parent_task_id: str,
        parent_tab_id: str,
        conn_id: str = "",
        replay_id: str = "",
        live_only: bool = False,
        ancestors: frozenset[str] = frozenset(),
        transcripts: bool = True,
    ) -> None:
        """Replay child tabs, including live descendants of completed parents.

        Finished direct children retain normal history behavior. Below a
        completed child, replay only live tasks, attached to a surviving ancestor.
        A running child whose tab the user closed
        (:meth:`_remember_closed_subagent_tab`) is treated the same way:
        not reopened, its live descendants still announced.

        The sub-tab ids are deterministic
        (``f"{parent_tab_id}__sub_{sub_task_id}"``) so that clicking
        the same parent task twice in a row updates the existing
        sub-agent tabs in place instead of stacking duplicates — the
        page's ``openSubagentTab`` handler is idempotent on
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
            conn_id: When non-empty, the sub-agent tabs and their
                transcripts are delivered only to this connection —
                the client whose ``ready`` replays the parent tab (see
                :meth:`_replay_session`).  The other clients opened
                these tabs from their own replay; re-announcing them
                would also re-open a sub-tab a user closed by hand.
                A reattached still-running sub-agent's live events
                keep fanning out to every viewer as before.
            replay_id: Echoed as ``replayId`` on every frame, as
                :meth:`_replay_session` does for the parent's.
            live_only: Omit completed descendants when walking through a
                finished ancestor; their live children must still be reachable.
            ancestors: Task ids already on the replay path.  A child that
                is also its own ancestor (a ``parent_task_id`` cycle in the
                history table) is skipped, or the descent would never end.
            transcripts: ``False`` announces the tabs (and attaches the
                running ones) but sends a ``replay_deferred`` offer in
                place of each ``task_events`` frame, reading nothing
                from the events table: the client fetches a child's
                transcript when the user opens its tab.  Every
                connection-scoped replay (a ``ready``, a deferred tab's
                fetch) does this; a history click still replays all.
        """
        scope: dict[str, Any] = {"connId": conn_id} if conn_id else {}
        if replay_id:
            scope["replayId"] = replay_id
        ancestors = ancestors | {parent_task_id}
        sub_rows = _load_subagent_rows_by_parent_task_id(
            parent_task_id,
            with_events=transcripts,
        )
        for idx, row in enumerate(sub_rows):
            sub_task_id = row["task_id"]
            if str(sub_task_id) in ancestors:
                continue
            sub_tab_id = f"{parent_tab_id}__sub_{sub_task_id}"
            description = str(row.get("task", "") or "")
            is_done = _subagent_is_done(sub_task_id)
            recording: list[dict[str, Any]] | None = None
            with self._state_lock:
                if is_done:
                    self._closed_subagent_tabs.discard(sub_tab_id)
                user_closed = sub_tab_id in self._closed_subagent_tabs
                if not (is_done or user_closed):
                    # Attached and announced in the same locked step as
                    # the closed check: a user close (recorded and
                    # broadcast under this lock, see ``_drop_tab_state``)
                    # lands wholly before — this replay stands down —
                    # or wholly after, closing the tab opened here.
                    _, _, recording = self._attach_viewer_to_running_chat(
                        str(row.get("chat_id", "") or ""),
                        sub_tab_id,
                        task_id=str(sub_task_id),
                        is_subagent=True,
                    )
                    self.printer.broadcast(
                        self._subagent_announcement(
                            row,
                            idx,
                            parent_task_id,
                            sub_tab_id,
                            False,
                            scope,
                        )
                    )
            if is_done or user_closed:
                # No tab of its own here, but its live descendants must
                # still reach this client (under the nearest open
                # ancestor, which the id chain names).
                self._open_persisted_subagent_tabs(
                    parent_task_id=str(sub_task_id),
                    parent_tab_id=sub_tab_id,
                    conn_id=conn_id,
                    replay_id=replay_id,
                    live_only=True,
                    ancestors=ancestors,
                    transcripts=transcripts,
                )
                if user_closed or live_only or _is_side_channel_row(row):
                    # Stays closed: a running sub-agent whose tab the
                    # user closed, finished history below a finished
                    # ancestor, or a finished side channel (the /ask
                    # answerer, whose answer sits in the parent's
                    # transcript and which has no fan-out panel to
                    # keep its tab closed).  A client that missed the
                    # close or completion still closes its copy.
                    self.printer.broadcast(_subagent_done_event(sub_tab_id, scope))
                    continue
                # A finished direct child: history the client's fan-out
                # panel keeps closed or reopens.
                self.printer.broadcast(
                    self._subagent_announcement(
                        row,
                        idx,
                        parent_task_id,
                        sub_tab_id,
                        True,
                        scope,
                    )
                )
            if not transcripts:
                self.printer.broadcast(
                    {
                        "type": "replay_deferred",
                        "tabId": sub_tab_id,
                        "chatId": str(row.get("chat_id", "") or ""),
                        "taskId": str(sub_task_id),
                        **scope,
                    }
                )
            else:
                # A still-running sub-agent's events table lags behind
                # the live run (asynchronous writer); its in-memory
                # recording holds the full transcript so far.
                # Snapshotted after the reattach above subscribed the
                # tab, so each live event is either in this snapshot or
                # fanned out after the replay.
                with self.printer.replay_snapshot(
                    "" if is_done else sub_task_id,
                    conn_id,
                    recording,
                ) as live_events:
                    if live_events:
                        row["events"] = live_events
                    self.printer.broadcast(
                        {
                            "type": "task_events",
                            "events": with_synthesized_events(
                                _coalesced_replay_events(row["events"]),
                                row,
                            ),
                            "task": description,
                            "task_id": sub_task_id,
                            "chat_id": row.get("chat_id", ""),
                            "extra": _extra_for_replay(row.get("extra", "")),
                            "tabId": sub_tab_id,
                            **scope,
                        }
                    )
            self._emit_pending_ask(sub_tab_id)
            if is_done:
                continue
            is_done = _subagent_is_done(sub_task_id)
            if is_done:
                self.printer.broadcast(_subagent_done_event(sub_tab_id, scope))
            # Preserve the original ID chain even when a completed ancestor
            # no longer has a visible tab. Completion can race with replay.
            self._open_persisted_subagent_tabs(
                parent_task_id=str(sub_task_id),
                parent_tab_id=sub_tab_id,
                conn_id=conn_id,
                replay_id=replay_id,
                live_only=is_done,
                ancestors=ancestors,
                transcripts=transcripts,
            )

    def _attach_viewer_to_running_chat(
        self,
        chat_id: str,
        new_tab_id: str,
        *,
        task_id: str | None = None,
        is_subagent: bool = False,
        replace_subscriptions: bool = False,
    ) -> tuple[AgentState | None, str, list[dict[str, Any]] | None]:
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
            is_subagent: Skip the chat-id fallback pass (sub-agent
                views must match by task id alone).
            replace_subscriptions: First drop every printer
                subscription/state of *new_tab_id* (a session replay
                rebinding the tab) EXCEPT its subscription to the
                resolved live task.  The tab is shared by every
                window, and a ``connId``-scoped replay re-sends the
                transcript to one window only: unsubscribing and
                re-subscribing the tab would let an event the task
                emits in between reach no window, so the other
                windows showing the tab would silently lose it.

        Returns:
            ``(source, source_task_id, recording)``: the live source
            state *new_tab_id* is now subscribed to (``None`` when no
            matching live agent exists), the task id it was subscribed
            under (``""`` when none; unlike ``source.task_id`` it does
            not change when a sequential run re-keys the state to its
            next subtask), and that task's recording list as returned
            by :meth:`JsonPrinter.subscribe_tab` in the same lock hold
            (``None`` when there is none).  Callers that broadcast an
            optimistic ``status running=true`` for the attach re-check
            THIS state object's liveness afterwards
            (:meth:`_broadcast_viewer_running`) — the object survives
            the printer bridge's mid-run re-keying, which a task-id
            lookup would not.  A replay passes the recording to
            :meth:`JsonPrinter.replay_snapshot`, so the run stopping
            its recording or being cleaned up before the snapshot
            cannot drop events the tab missed.
        """
        if not new_tab_id:
            return None, "", None
        recording: list[dict[str, Any]] | None = None
        source: AgentState | None = None
        # One ``_state_lock`` section resolves the source AND rewires
        # the printer: the printer bridge re-keys a run to its
        # persisted id (and the runner then subscribes the launching
        # tab to it) under the same lock, so a cleanup run after the
        # release could keep an obsolete provisional id and drop the
        # fresh persisted-id subscription.  Lock order ``STATE_LOCK``
        # → printer locks; ``subscribe_tab``'s catch-up broadcast only
        # schedules sends, so nothing here blocks under the lock.
        with self._state_lock:
            if task_id is not None:
                candidate = agent_state.get(task_id)
                # ``running()`` counts a created-but-not-yet-started
                # worker as live: a task in the run-startup window is
                # real, and a viewer resuming its chat must attach to
                # it rather than be treated as opening a finished
                # session.
                if candidate is not None and candidate.running():
                    source = candidate
            if source is None and chat_id and not is_subagent:
                # The tab's own run first: two tasks of one chat can
                # run at once (the chat opened in two tabs, a run
                # started in each), and a tab attached to its
                # sibling's run instead of its own was reported
                # neither running nor idle on a reconnect, since the
                # optimistic status of a source the tab does not own
                # is suppressed (``_viewer_owns_other_busy_run``).
                own = agent_state.find_by_tab(new_tab_id)
                if (
                    own is not None
                    and own.chat_id == chat_id
                    and not own.is_subagent
                    and own.running()
                ):
                    source = own
                else:
                    for t in agent_state.agent_states.values():
                        if t.chat_id != chat_id or t.is_subagent:
                            continue
                        if t.running():
                            source = t
                            break
            source_task_id = source.task_id if source is not None else ""
            if replace_subscriptions:
                self._printer_cleanup_tab(new_tab_id, source_task_id)
            if source is not None:
                recording = self.printer.subscribe_tab(
                    source_task_id,
                    new_tab_id,
                )
        return source, source_task_id, recording

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
        return viewer_state is not None and viewer_state is not source and viewer_state.busy()

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

        ``attached: true`` marks the broadcast as a viewer catching up
        on a run already going (a reload's replay, a history open) as
        opposed to a run starting: the page gives a STARTING run
        the window (``main.js`` ``showChatForRun``) and must not do so
        for a replay.

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
                    "attached": True,
                }
            )

    def _broadcast_replay_status(
        self,
        tab_id: str,
        replay_scope: dict[str, Any],
    ) -> None:
        """Settle *tab_id*'s running flag when its replay attached to nothing live.

        The page settles a tab's running flag only from ``status``
        events (a replay skips ``task_done``), so a replay that
        carries none leaves whatever flag the tab had — wrong for
        every client that was disconnected while the flag changed.
        A replay of a finished task (a history row, nothing live
        attached) used to carry no status: a tab that saw its task's
        ``running:true``, lost its connection and missed the task's
        end got the finished transcript while still marked running,
        kept a Stop button, and landed at the end of the transcript
        instead of on its task panel (``main.js``
        ``landOpenedTranscript``).  And a deferred replay
        (:meth:`_defer_replay`) of a tab whose own run the attach did
        not resolve (bound to a chat other than the run's) sent
        neither state, so a background tab's spinner stayed stale.

        The tab's OWNER decides, under ``_state_lock`` so the check is
        serialized against ``_cmd_run`` installing a run on the tab:
        a tab that owns a live run is told ``running:true`` again
        (``attached``, like :meth:`_broadcast_viewer_running`, so the
        page does not treat the re-send as a run starting; the run's
        own start stamp so the timer keeps counting from the same
        instant), any other tab ``running:false``.  A worktree merge
        in progress on the tab is not a run: the page shows no spinner
        for it and ``_cmd_run`` refuses runs during it, so only
        ``running()`` — not ``busy()`` — withholds the idle status.

        Sent BEFORE a transcript so the page lands the replay with the
        flag settled, and stamped with the replay's scope (a
        connection-scoped ready replay stays scoped, a forwarded
        resume keeps its ``replayId``).

        Args:
            tab_id: The tab whose replay attached to nothing live.
            replay_scope: The ``connId`` / ``replayId`` stamps of the
                replay's frames.
        """
        with self._state_lock:
            owner = agent_state.find_by_tab(tab_id)
            status: dict[str, Any] = {
                "type": "status",
                "running": False,
                "tabId": tab_id,
                **replay_scope,
            }
            if owner is not None and owner.running():
                status["running"] = True
                status["startTs"] = int(getattr(owner.agent, "_task_start_ms", 0) or 0)
                status["attached"] = True
            self.printer.broadcast(status)

    def _finalize_viewer_attach(
        self,
        tab_id: str,
        source: AgentState,
        replayed_events: list[dict[str, Any]],
        events_payload: dict[str, Any],
        resend_result: bool = True,
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
                task/chat/tab and reach the same connections (a
                ``connId``-scoped replay stays scoped).
            resend_result: ``False`` for a deferred replay
                (:meth:`_defer_replay`), which sent no transcript and so
                has no terminal result to re-send — only the status is
                corrected.
        """
        with self._state_lock:
            if source.running():
                return
            if self._viewer_owns_other_busy_run(tab_id, source):
                return
            if resend_result and not any(ev.get("type") == "result" for ev in replayed_events):
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
                with_synthesized_events(
                    _coalesced_replay_events(result["events"]),
                    result,
                )
                if result
                else []
            ),
            "tabId": tab_id,
        }
        self.printer.broadcast(event)
