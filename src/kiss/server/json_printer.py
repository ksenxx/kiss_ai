# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared browser UI components for KISS agent viewers.

The printer is **task-centric**: every piece of per-stream state
(recordings, usage offsets, bash buffering, persistence) is keyed by
``task_id`` rather than by the frontend tab id.  Multiple browser tabs
viewing the same task subscribe to the task's event stream via
``_subscribers[task_id] -> {tab_id, ...}``.

The agent thread sets ``_thread_local.task_id`` once its
``task_history.id`` has been allocated; from that point every
``broadcast()`` is recorded under the task id, persisted under the task
id, and fanned out to every subscriber tab (each copy stamped with its
own ``tabId``).  Events with an explicit ``tabId`` already set on the
payload are treated as "system" events targeted at a specific tab and
are forwarded directly without recording or persistence.
"""

import base64
import json
import logging
import os
import re
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from kiss.agents.sorcar.persistence import _queue_chat_event
from kiss.core import stop_signal
from kiss.core.printer import (
    Printer,
    extract_extras,
    extract_path_and_lang,
    parse_result_yaml,
    truncate_result,
)
from kiss.server import agent_state

logger = logging.getLogger(__name__)

_OffsetT = TypeVar("_OffsetT", int, float)

#: How many finished task ids are remembered so a late write cannot
#: resurrect their per-task state.  Bounded, so the guard itself can
#: never grow without limit.
_CLOSED_TASK_MEMORY = 256

_DISPLAY_EVENT_TYPES = frozenset(
    {
        "clear",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_delta",
        "text_end",
        "tool_call",
        "tool_result",
        "system_output",
        "result",
        "system_prompt",
        "prompt",
        "task_done",
        "task_error",
        "task_stopped",
        "task_interrupted",
        "followup_suggestion",
        # The finished ``/ask`` side-channel answer, delivered into
        # the OWNER task's transcript (``commands._broadcast_ask_answer``).
        "ask_answer",
        "autocommit_done",
        "warning",
        # Persisted so replays repopulate the chat header's tokens/cost
        # metrics: without it a transcript reloaded mid-run (or after a
        # stop/error, when no ``result`` event exists) shows only the
        # step count in the status row.
        "usage_info",
        # Persisted so replays repopulate the static task panel's
        # settings info (model, worktree / parallel modes, budget,
        # start time, chat / task / parent ids); broadcast once per
        # run by ``ChatSorcarAgent.run``.
        "task_settings",
    }
)

#: Event types that, when broadcast WITH an explicit ``tabId`` AND a
#: ``taskId``, are still recorded into that task's in-memory recording
#: and persisted into its ``events`` rows (a tabId-stripped copy, filed
#: under the event's own ``taskId`` — see
#: :meth:`JsonPrinter._keep_tab_stamped_task_event`).  Every other
#: tabId-stamped event is a transient targeted broadcast.  These are the
#: events a task's transcript must keep although they are emitted from
#: outside the task's own thread: the ``/ask`` prompt echo
#: (``commands._echo_injected_prompt``), its ``ask_answer`` reply
#: (``commands._broadcast_ask_answer``) and tab-targeted ``result``
#: events (``task_runner._broadcast_failure_result``).
TAB_STAMPED_TASK_EVENT_TYPES = frozenset({"prompt", "result", "ask_answer"})

# Tools whose ``tool_call`` event names a file the agent CHANGED (as
# opposed to merely read).  Used to track, per task, which files the
# task modified so the end-of-task auto-commit can also commit repos
# other than the tab's work_dir one.
_FILE_MUTATING_TOOLS = frozenset({"Write", "Edit"})

#: Image formats the chat webview renders inline in a tool's event
#: panel (as ``data:`` URIs, so replays keep working after the file —
#: often inside a discarded worktree — is gone).
_IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}

#: At most this many images are embedded per ``tool_result`` event, and
#: only files up to this size — the events are persisted per task, so
#: a tool result naming dozens of screenshots must not balloon the DB.
_MAX_RESULT_IMAGES = 3
_MAX_RESULT_IMAGE_BYTES = 2 * 1024 * 1024

#: Cap on the TOTAL base64 image payload embedded across one task's
#: events.  Replays send a task's whole event list in a single frame
#: and the VS Code extension drops its socket above 32 MiB
#: (``AgentClient.ts``), so unbounded per-event embedding would make an
#: image-heavy task's history unloadable.  12 MiB leaves ample room
#: for the textual events sharing the frame.
_MAX_TASK_IMAGE_B64_BYTES = 12 * 1024 * 1024

#: A file counts as "generated by this tool call" when its mtime is no
#: older than the call's ``tool_call`` broadcast minus this slack
#: (clock skew, files finalized a moment before the event was stamped).
_RESULT_IMAGE_RECENCY_SLACK_SECS = 5.0
#: Fallback recency window for result paths seen without a preceding
#: ``tool_call`` (e.g. third-party message routes).
_RESULT_IMAGE_FALLBACK_WINDOW_SECS = 60.0

#: Path-looking tokens ending in a known image extension.  Quote and
#: bracket characters terminate a token so ``(tmp/plot.png)`` and
#: markdown ``![...](x.png)`` yield the bare path; an optional drive
#: prefix keeps ``C:\shots\x.png`` whole; the trailing lookahead
#: rejects longer names like ``x.png.bak`` while still accepting a
#: sentence-ending period (``Saved output.png.``).
#:
#: The leading lookbehind (``not preceded by a token character``) pins
#: every match attempt to the START of a token, which makes the scan
#: linear in the text size.  Without it the engine attempted a match
#: at every offset inside a token and backtracked to the token's end
#: each time — quadratic — and a tool result containing one long
#: unbroken token (``media/vosk.js`` embeds a 5.77 M-char base64
#: blob) pinned the GIL for hours, freezing the whole server process
#: (2026-09-12 outage).  Semantics are unchanged: a token's longest
#: valid image suffix is found from its first character either way.
_IMAGE_PATH_RE = re.compile(
    r"(?<![^\s\"'`<>|:;,()\[\]{}])"
    r"(?:(?<![\w.-])[A-Za-z]:[\\/])?[^\s\"'`<>|:;,()\[\]{}]+"
    r"\.(?:png|jpe?g|gif|webp|bmp|svg)(?![\w-]|\.[\w-])",
    re.IGNORECASE,
)

#: Quoted image paths, which may contain spaces (``saved to "/home/A
#: User/shot.png"``).  Scanned before the bare-token pass, and their
#: spans are blanked out of the text so the bare pass cannot re-match
#: a truncated tail of the same path.
_QUOTED_IMAGE_PATH_RE = re.compile(
    r"[\"']([^\"'\n]+\.(?:png|jpe?g|gif|webp|bmp|svg))[\"']",
    re.IGNORECASE,
)


def _extract_image_path_candidates(text: str) -> list[str]:
    """Return image-file paths mentioned in *text*, deduplicated.

    Quoted paths (which may contain spaces) are collected first, then
    bare path tokens from the remaining text.  Protocol-relative URL
    leftovers (``//host/x.png`` — the colon of ``https://`` terminates
    the bare token) are dropped; other non-existent candidates are
    filtered later by the caller's ``stat`` check.

    Args:
        text: Arbitrary tool-result text to scan.

    Returns:
        The unique path candidates, quoted mentions first, each group
        in first-mention order.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []

    def _add(candidate: str) -> None:
        if candidate.startswith("//") or candidate in seen:
            return
        seen.add(candidate)
        out.append(candidate)

    for match in _QUOTED_IMAGE_PATH_RE.finditer(text):
        _add(match.group(1))
    remaining = _QUOTED_IMAGE_PATH_RE.sub(" ", text)
    for match in _IMAGE_PATH_RE.finditer(remaining):
        _add(match.group(0))
    return out


def stamp_event_ts(event: dict[str, Any]) -> None:
    """Stamp *event* with its wall-clock emission time, in place.

    Adds a ``ts`` field (ms since the epoch) when the event does not
    already carry one, so live transports, in-memory recordings, and
    the persisted DB rows all agree on WHEN the event happened and the
    chat webview (extension and remote web app alike) can render the
    compact per-panel timestamp badge — including on replays, which
    keep the original stamp.

    Args:
        event: The event dictionary to stamp (mutated in place).
    """
    if "ts" not in event:
        event["ts"] = int(time.time() * 1000)


def _coalesce_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive delta events of the same type to reduce storage size.

    Consecutive thinking_delta, text_delta, and system_output events are
    combined by concatenating their ``text`` fields.

    Args:
        events: List of event dicts to coalesce.

    Returns:
        A new list with consecutive same-type delta events merged.
    """
    if not events:
        return events
    result: list[dict[str, Any]] = []
    merge_types = ("thinking_delta", "text_delta", "system_output")
    for ev in events:
        t = ev.get("type", "")
        if (
            result
            and t == result[-1].get("type")
            and t in merge_types
            and "text" in ev
            and "text" in result[-1]
        ):
            result[-1] = {**result[-1], "text": result[-1]["text"] + ev["text"]}
        else:
            result.append(ev)
    return result


def _task_settings_event_from_session(
    session: dict[str, Any],
) -> dict[str, Any] | None:
    """Synthesize a ``task_settings`` display event from a session dict.

    Tasks that ran before the ``task_settings`` event existed (or ran
    without a broadcasting printer) have no such event persisted, yet
    their settings live in the ``task_history`` row.  This builds the
    event the live run would have broadcast from the session dict the
    persistence loaders return (``{task, task_id, chat_id, events,
    extra}``), so replays and shares can repopulate the static task
    panel's settings info for every task.

    Args:
        session: A loader session dict.  ``extra`` is the JSON string
            synthesized by ``_row_to_extra_json``.

    Returns:
        The event dict, or None when *session* has no task id or its
        ``extra`` does not parse to a dict.
    """
    task_id = str(session.get("task_id") or "")
    if not task_id:
        return None
    extra_raw = session.get("extra")
    if not isinstance(extra_raw, str) or not extra_raw:
        return None
    try:
        extra = json.loads(extra_raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(extra, dict):
        return None
    settings: dict[str, Any] = {
        "model": str(extra.get("model") or ""),
        "work_dir": str(extra.get("work_dir") or ""),
        "is_parallel": bool(extra.get("is_parallel", False)),
        "is_worktree": bool(extra.get("is_worktree", False)),
        "chat_id": str(session.get("chat_id") or ""),
        "task_id": task_id,
    }
    try:
        start_ts = int(extra.get("startTs", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        start_ts = 0
    if start_ts > 0:
        settings["start_ts"] = start_ts
    try:
        max_budget = float(extra.get("max_budget", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        max_budget = 0.0
    if max_budget > 0:
        settings["max_budget"] = max_budget
    sub = extra.get("subagent")
    parent_id = str(sub.get("parent_task_id") or "") if isinstance(sub, dict) else ""
    settings["is_subagent"] = bool(parent_id)
    if parent_id:
        settings["parent_task_id"] = parent_id
    return {"type": "task_settings", "settings": settings, "taskId": task_id}


def with_task_settings_event(
    events: list[dict[str, Any]],
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return *events* with a leading ``task_settings`` event ensured.

    Used by every replay / share reply builder: when the persisted
    stream already carries the run's own ``task_settings`` event the
    list is returned unchanged; otherwise the event synthesized from
    *session* (see :func:`_task_settings_event_from_session`) is
    prepended.  When nothing can be synthesized the list is returned
    unchanged.

    Args:
        events: The task's replay events (already coalesced or raw).
        session: The loader session dict the events came from.

    Returns:
        The events list, never None.
    """
    if any(ev.get("type") == "task_settings" for ev in events):
        return events
    synthesized = _task_settings_event_from_session(session)
    if synthesized is None:
        return events
    return [synthesized, *events]


class _BashState:
    """Bash buffering state for streaming output.

    Buffers bash output fragments and flushes them periodically to
    avoid overwhelming the frontend with tiny events.
    """

    __slots__ = (
        "buffer",
        "timer",
        "generation",
        "last_flush",
        "streamed",
        "flush_lock",
    )

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.timer: threading.Timer | None = None
        self.generation: int = 0
        self.last_flush: float = 0.0
        self.streamed: bool = False
        self.flush_lock = threading.Lock()


class _PrinterThreadLocal(threading.local):
    """Per-thread printer state whose ``stop_event`` is process-visible.

    ``stop_event`` is a property over :mod:`kiss.core.stop_signal`
    rather than plain thread-local storage, so the single assignment
    that binds a stop event to a task thread (``task_runner``,
    ``chat_sorcar_agent``'s fan-out workers, ``sorcar_agent``'s
    sub-agents) also publishes it to code *below* the agent.  Model
    adapters need it to abort a stream that has gone silent: without it
    a stop is only noticed when the agent next prints, which left task
    ``709ebce3`` unstoppable for 178 seconds
    (``reports/stop_button_delay_2026-08-05.html``).  Keeping one
    storage location — instead of publishing to two — means the flag
    the agent polls and the flag the model watches can never disagree.
    """

    @property
    def stop_event(self) -> threading.Event | None:
        """The calling thread's stop event, or ``None`` when unbound."""
        return stop_signal.get_thread_stop_event()

    @stop_event.setter
    def stop_event(self, event: threading.Event | None) -> None:
        stop_signal.set_thread_stop_event(event)


class JsonPrinter(Printer):
    """Base printer for browser-based UIs (task-id keyed).

    The current block type (``_current_block_type``) is stored in
    thread-local storage so concurrent task threads can each route
    their streamed tokens to the correct (thinking vs text) panel
    without corrupting each other.  Recording and bash buffering are
    per-task (keyed by ``task_id``) so one task's ``stop_recording()``
    or ``reset()`` does not destroy another task's state.

    The set of frontend tabs that should receive a task's events is
    looked up from ``_subscribers[task_id]``.  A tab subscribes via
    :meth:`subscribe_tab` (e.g. when the user opens the task in a new
    browser tab) and unsubscribes via :meth:`cleanup_tab` (when the tab
    closes).
    """

    @property
    def _current_block_type(self) -> str:
        return getattr(self._thread_local, "_cbt", "")

    @_current_block_type.setter
    def _current_block_type(self, value: str) -> None:
        self._thread_local._cbt = value

    @property
    def _bash_state(self) -> _BashState:
        """Return the bash buffering state for the current task.

        Each task gets its own ``_BashState`` so concurrent tasks
        cannot corrupt each other's bash buffer, ``streamed`` flag,
        generation counter, or flush timer.  The caller must hold
        ``_bash_lock`` when accessing this in multi-threaded code.
        """
        key = self._task_key()
        bs = self._bash_states.get(key)
        if bs is None:
            bs = _BashState()
            self._bash_states[key] = bs
        return bs

    def _bash_task_may_create_state(self, key: str) -> bool:
        """Return whether a NEW ``_BashState`` may be created for *key*.

        The ``_closed_tasks`` tombstone set is bounded, so after
        :data:`_CLOSED_TASK_MEMORY` later task cleanups an old task's
        tombstone is evicted and "not in ``_closed_tasks``" no longer
        proves the task was never closed — a sufficiently delayed
        ``bash_stream`` fragment could then recreate the freed state
        and broadcast stale output (gpt-5.6-sol conc review,
        finding 5).  Until the first eviction, tombstone absence IS
        proof, and creation stays unconditional (also the behaviour
        every pre-existing direct-printer test relies on).  After it,
        creation additionally requires the task to be a POSITIVELY
        ACTIVE producer in the agent-state registry
        (``is_task_active``) — a check that cannot forget, no matter
        how much later the straggler arrives.  Thread liveness is
        deliberately NOT accepted as a substitute: normal task
        finalization clears ``is_task_active`` and then runs
        ``cleanup_task`` on the SAME runner thread, so an already
        cleaned-up task's state can still be registered with a live
        thread while the runner executes its post-task tail — a
        ``thread_alive()`` fallback let exactly that tail recreate the
        freed state and emit stale output once its tombstone was
        evicted (gpt-5.6-sol round-2 review, finding 2).  Every real
        producer (UI runs, sub-agents, standalone runs) is registered
        active by ``agent_task_allocated`` — UI runs additionally by
        the task runner — before its first bash fragment, so the
        active-only gate never drops live output.

        Called with ``_lock`` and ``_bash_lock`` held, which is why it
        reads the registry dict directly instead of via
        ``agent_state.get`` — taking ``STATE_LOCK`` here would create
        a ``_lock`` → ``STATE_LOCK`` edge inverting the established
        ``STATE_LOCK`` → printer-lock order.  The lock-free point read
        is safe: it is a single dict lookup plus a plain attribute
        read, and both race directions converge
        (a task observed live here is unregistered only at/after its
        cleanup, whose tombstone mark this caller's ``_lock`` section
        already checked; a task observed dead can only stay dead).

        Args:
            key: The non-empty task key a fragment wants state for.

        Returns:
            ``True`` when creating a ``_BashState`` for *key* is safe.
        """
        if not self._closed_tasks_evicted:
            return True
        state = agent_state.agent_states.get(key)
        return state is not None and state.is_task_active

    def __init__(self) -> None:
        self._thread_local = _PrinterThreadLocal()
        self._lock = threading.Lock()
        self._bash_lock = threading.Lock()
        self._bash_states: dict[str, _BashState] = {}
        self._tokens_offsets: dict[str, int] = {}
        self._budget_offsets: dict[str, float] = {}
        self._steps_offsets: dict[str, int] = {}
        # Per-task wall-clock time of the latest ``tool_call``
        # broadcast, so ``_emit_tool_result`` can tell images the tool
        # CREATED during this call apart from old images it merely
        # mentioned.  Consumed (popped) by the call's own result so a
        # later unpaired result falls back to the short recency window
        # instead of an arbitrarily old cutoff.  Guarded by
        # ``self._lock``.
        self._tool_call_started: dict[str, float] = {}
        # Per-task total of base64 image bytes already embedded, so an
        # image-heavy task stops embedding before its replayed event
        # list outgrows the extension's frame limit (see
        # _MAX_TASK_IMAGE_B64_BYTES).  Guarded by ``self._lock``.
        self._image_b64_used: dict[str, int] = {}
        # Task ids whose state cleanup_task already freed, newest last
        # and capped at _CLOSED_TASK_MEMORY entries.  A late usage
        # offset write for one of them is dropped instead of leaking a
        # dict entry nothing would pop again.
        self._closed_tasks: dict[str, None] = {}
        # Flips (monotonically) the first time the cap above evicts a
        # tombstone.  From then on "key not in _closed_tasks" no longer
        # proves the task was never closed, so the bash_stream branch's
        # state CREATION additionally requires a positive agent-state
        # liveness check (see _bash_task_may_create_state) — the
        # bounded tombstone set stays bounded without the guarantee
        # ever expiring (gpt-5.6-sol conc review, finding 5).
        self._closed_tasks_evicted = False
        self._recordings: dict[str, list[dict[str, Any]]] = {}
        # task id → (tab_id, conn_id) of the UI tab the task was
        # launched from; set via register_task_ui when a task runs in
        # a UI tab.  Deliberately TASK-SCOPED: cleanup_task drops the
        # entry the moment a task ends, which is precisely why
        # _transient_targets resolves post-task broadcasts from the
        # (longer-lived) subscriber set instead.  That lifetime
        # difference is a load-bearing part of the printer's routing
        # contract.  Read under self._lock.
        self._task_ui: dict[str, tuple[str, str]] = {}
        self._subscribers: dict[str, set[str]] = {}
        self._subscriber_expiry: dict[str, float] = {}
        # Tabs whose picker currently shows a running agent's model
        # instead of their user's pick, and the model each running task
        # switched itself to (see broadcast_agent_model_pick).
        self._model_override_tabs: set[str] = set()
        self._task_model_override: dict[str, str] = {}
        # Serialises {override-state update + modelPick broadcast} in
        # subscribe_tab's catch-up, broadcast_agent_model_pick, and
        # restore_model_pick.  Without it a restore could interleave
        # between a catch-up's state write (under ``_lock``) and its
        # broadcast (after ``_lock``), leaving the tab's picker showing
        # a dead agent's model while the tab is no longer in
        # ``_model_override_tabs`` — an inversion no later restore can
        # repair.  Held ACROSS the broadcast, which is safe: modelPick
        # events carry a ``tabId`` so ``broadcast`` never re-enters the
        # model-pick methods or this lock (base printer: early return;
        # WebPrinter: worktree tracking + socket send; MemoryPrinter:
        # list append), and no caller of the three methods holds any
        # printer lock.  Order: ``_model_pick_lock`` → ``_lock``.
        self._model_pick_lock = threading.Lock()
        # Absolute paths of files each task changed through the
        # file-mutating tools (Write / Edit), keyed by task id.
        # Consumed by the task-runner's end-of-task auto-commit so
        # changes landing OUTSIDE the tab's work_dir repository are
        # committed too (see _autocommit_changed_repos).  Tracked
        # in memory because event persistence is asynchronous — the
        # DB may not yet hold the last tool_call rows when the task's
        # finally block runs.
        self._changed_paths: dict[str, set[str]] = {}

    @staticmethod
    def _coerce_task_id(value: Any) -> str:
        """Return *value* normalised to the printer's task-id string key.

        Accepts ``str`` and ``int`` (``task_history.id``).  Returns
        ``""`` for ``None``/empty input so callers can treat
        "no task" and "task id unset" uniformly.
        """
        if value is None or value == "":
            return ""
        return str(value)

    def _task_key(self) -> str:
        """Return the thread-local task key for per-task state lookups.

        Used for per-task usage offsets, recordings, and bash state.
        Falls back to ``""`` for threads without a ``task_id`` set
        (e.g. unit tests or pre-task lifecycle code paths).
        """
        return self._coerce_task_id(
            getattr(self._thread_local, "task_id", None),
        )

    def subscribe_tab(self, task_id: Any, tab_id: str) -> None:
        """Subscribe *tab_id* to receive every event broadcast for *task_id*.

        Used by the server when the user opens a chat tab that is
        backed by a running task: the tab subscribes to the task's
        event stream so live events flow to that tab.  Idempotent.

        Args:
            task_id: The task identifier (``task_history.id`` int or
                its string form).
            tab_id: The frontend tab id to subscribe.
        """
        key = self._coerce_task_id(task_id)
        if not key or not tab_id:
            return
        # _model_pick_lock keeps the catch-up's state write and its
        # broadcast atomic w.r.t. restore_model_pick: a restore racing
        # this subscribe used to discard the tab's membership and
        # broadcast the user's model FIRST, then the resumed catch-up
        # broadcast repainted the dead agent's model with the tab no
        # longer in _model_override_tabs — a stale label no later
        # restore could fix (restore early-returns for non-members).
        with self._model_pick_lock:
            with self._lock:
                self._sweep_expired_subscribers()
                viewers = self._subscribers.get(key)
                if viewers is None:
                    viewers = set()
                    self._subscribers[key] = viewers
                viewers.add(tab_id)
                # A tab joining a task whose agent already switched
                # models missed that one-shot event, and would
                # otherwise sit on the wrong label until the task
                # ended.
                catch_up = self._task_model_override.get(key, "")
                if catch_up:
                    self._model_override_tabs.add(tab_id)
            if catch_up:
                self.broadcast_model_pick(catch_up, "agent", tab_id)

    def register_task_ui(
        self,
        task_id: Any,
        tab_id: str,
        conn_id: str = "",
    ) -> None:
        """Attach the UI tab (and its connection) running *task_id*.

        Called by the server when a task is launched from a UI tab:
        the tab id and the connection id of the launching client are
        recorded on the printer so the task's event stream is fanned
        out to that tab (via :meth:`subscribe_tab`) and the owning
        connection stays identifiable for the task's whole life.

        Args:
            task_id: The task identifier.
            tab_id: The frontend tab id the task runs in.
            conn_id: The id of the client connection that launched the
                task (``""`` for direct callers / tests).
        """
        key = self._coerce_task_id(task_id)
        if not key or not tab_id:
            return
        with self._lock:
            self._task_ui[key] = (tab_id, conn_id)
        self.subscribe_tab(task_id, tab_id)

    def agent_task_allocated(
        self,
        agent: Any,
        task_id: Any,
        chat_id: str = "",
    ) -> None:
        """Register (or re-key) *agent*'s run under its allocated task id.

        Duck-typed bridge called by ``ChatSorcarAgent.run`` the moment
        the run's ``task_history`` row id exists.  When the server
        pre-registered a state for this agent (a UI-launched run), the
        state is re-keyed to the persisted id; otherwise (parallel
        sub-agents, standalone runs) a fresh state is created from the
        calling thread's context.

        Args:
            agent: The live agent instance.
            task_id: The freshly allocated ``task_history`` row id.
            chat_id: The chat id the run belongs to.
        """
        key = self._coerce_task_id(task_id)
        if not key:
            return
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_agent(agent)
            if state is None:
                sub_info = getattr(agent, "_subagent_info", None)
                parent_task_id: str | None = None
                if isinstance(sub_info, dict):
                    parent_task_id = str(sub_info.get("parent_task_id") or "")
                state = agent_state.AgentState(
                    key,
                    agent=agent,
                    tab_id=str(getattr(agent, "_tab_id", "") or ""),
                    parent_task_id=parent_task_id,
                    stop_event=stop_signal.get_thread_stop_event(),
                    task_thread=threading.current_thread(),
                    is_task_active=True,
                )
                agent_state.register(state)
            else:
                agent_state.rekey(state, key)
                state.is_task_active = True
                if state.stop_event is None:
                    state.stop_event = stop_signal.get_thread_stop_event()
                if state.task_thread is None:
                    state.task_thread = threading.current_thread()
            if chat_id:
                state.chat_id = chat_id

    def agent_task_finished(self, agent: Any, task_id: Any) -> None:
        """Mark *agent*'s run as finished and drop non-server states.

        Duck-typed bridge called from ``ChatSorcarAgent.run``'s
        ``finally``.  Server-owned states (UI-launched runs) are left
        entirely to the server's own task lifecycle — the task runner
        still does persistence / autocommit / worktree post-processing
        after ``run()`` returns, so flipping ``is_task_active`` here
        would open a window where a concurrent merge/discard races the
        runner.  States the bridge created itself (sub-agents,
        standalone runs) are deactivated and removed here.

        Args:
            agent: The live agent instance.
            task_id: The task id the run was registered under.
        """
        key = self._coerce_task_id(task_id)
        with agent_state.STATE_LOCK:
            state = agent_state.get(key)
            if state is None or state.agent is not agent:
                state = agent_state.find_by_agent(agent)
            if state is None or state.server_owned:
                return
            state.is_task_active = False
            state.task_thread = None
            agent_state.unregister(state.task_id, state)

    def drain_pending_user_messages(self) -> list[str]:
        """Return and clear the current task's queued follow-up prompts.

        Duck-typed bridge called by the agent's pre-step hook.  Also
        emits a durable ``recordOnly`` prompt echo for every message
        whose live echo could not be attributed to a task id at
        queueing time, so the echo lands in the correct trajectory.

        Returns:
            The queued user messages, oldest first.  Empty when the
            calling thread has no task or nothing is queued.
        """
        state = agent_state.get(self._task_key())
        if state is None:
            return []
        with agent_state.STATE_LOCK:
            queued = list(state.pending_user_messages)
            state.pending_user_messages.clear()
            deferred = list(state.unattributed_prompt_echoes)
            state.unattributed_prompt_echoes.clear()
        for msg in deferred:
            try:
                self.broadcast(
                    {"type": "prompt", "text": msg, "recordOnly": True},
                )
            except Exception:
                # Requeue so the durable echo is retried on the next
                # drain instead of being lost forever.
                logger.debug(
                    "recordOnly prompt echo broadcast failed",
                    exc_info=True,
                )
                with agent_state.STATE_LOCK:
                    state.unattributed_prompt_echoes.append(msg)
        return queued

    def has_pending_user_messages(self) -> bool:
        """True when the current task has undrained follow-up prompts.

        Duck-typed bridge consulted by the agent's ``finish`` guard so
        a follow-up the user typed mid-step is injected before the
        task is allowed to end.
        """
        state = agent_state.get(self._task_key())
        if state is None:
            return False
        with agent_state.STATE_LOCK:
            return bool(state.pending_user_messages)

    def live_worktree_branches(self) -> set[str]:
        """Return the ``kiss/wt-*`` branches owned by live agents.

        Duck-typed bridge used by ``WorktreeSorcarAgent`` so its
        orphaned-worktree reclaim pass never adopts a branch another
        live agent is still using.
        """
        branches: set[str] = set()
        for state in agent_state.snapshot():
            wt = getattr(state.agent, "_wt", None) if state.agent else None
            if wt is not None:
                branches.add(wt.branch)
        return branches

    def _fanout_targets(self, task_id: Any) -> list[str]:
        """Return a snapshot of subscriber tab ids for *task_id*.

        Args:
            task_id: The task identifier from the event's ``taskId``.

        Returns:
            List of subscriber tab ids that should receive a copy of
            the event.  Empty when *task_id* is falsy or has no
            subscribers.
        """
        key = self._coerce_task_id(task_id)
        if not key:
            return []
        with self._lock:
            self._sweep_expired_subscribers()
            viewers = self._subscribers.get(key)
            if not viewers:
                return []
            return list(viewers)

    def tasks_for_tab(self, tab_id: str) -> list[str]:
        """Return the task keys *tab_id* is currently subscribed to.

        The inverse of :meth:`_fanout_targets`.  A tab is normally
        subscribed to the one task it launched or is viewing, but a
        finished task's subscriber set lingers for a while (see
        :meth:`cleanup_task`), so right after a new run starts the list
        can name both the old and the new task; callers pick the live
        one via the agent-state registry.

        Args:
            tab_id: The frontend tab id.

        Returns:
            The subscribed task keys (``task_history`` ids as strings),
            empty when *tab_id* is subscribed to nothing.
        """
        with self._lock:
            self._sweep_expired_subscribers()
            return [
                key
                for key, viewers in self._subscribers.items()
                if tab_id in viewers
            ]

    def _transient_targets(
        self, task_id: Any, tab_id: str = "",
    ) -> tuple[str, list[str]]:
        """Resolve every tab id watching a task, for transient broadcasts.

        The task is identified by the calling thread's task id when
        one is bound, else by the explicit *task_id* fallback — the
        latter covers calls made off the agent's run thread and calls
        near task teardown, when the thread-local key has already
        been cleared.  The watching tabs come from the subscriber
        registry, which :meth:`cleanup_task` keeps alive for a few
        minutes after the task ends precisely so post-task broadcasts
        still reach their tabs (``_task_ui`` by contrast is dropped
        at teardown, so it is deliberately not consulted here).

        All tabs are treated uniformly — the tab a task was launched
        from is subscribed like any viewer (see
        :meth:`register_task_ui`), so no owner/viewer distinction
        exists.  *tab_id* is simply one more uniform target, for
        callers whose printer never saw a subscription (plain
        recording printers in tests).

        Args:
            task_id: Explicit task id used when the calling thread
                has no thread-local ``task_id`` bound.
            tab_id: Extra tab id to include (deduplicated; ``""`` is
                ignored).

        Returns:
            ``(task_key, targets)``: the resolved task key (thread-
            local first, else the coerced *task_id* fallback — handed
            back so callers that also need the key, e.g.
            :meth:`broadcast_agent_model_pick`, never re-derive it and
            risk resolving a different key than the one the targets
            were computed for), and the sorted, deduplicated,
            non-empty tab ids — empty when no watching tab is
            resolvable at all.
        """
        task_key = self._task_key() or self._coerce_task_id(task_id)
        targets = {t for t in self._fanout_targets(task_key) if t}
        if tab_id:
            targets.add(tab_id)
        return task_key, sorted(targets)

    def broadcast_transient(
        self,
        event: dict[str, Any],
        task_id: Any = None,
        tab_id: str = "",
    ) -> None:
        """Broadcast one ``tabId``-stamped copy of *event* per watching tab.

        The printer-side "transient, all-watching-tabs" primitive:
        the caller supplies a plain event (no ``tabId``) plus the ids
        identifying its task, and the printer resolves the watching
        tabs itself (see :meth:`_transient_targets`) and broadcasts
        one copy per tab.  The explicit per-copy ``tabId`` is what
        makes the event transient: ``broadcast`` implementations
        deliver such events only to clients (which filter by
        ``tabId``) and never record or persist them, so replaying a
        finished conversation cannot resurrect them.

        When no watching tab is resolvable at all, ONE copy stamped
        with *tab_id* (possibly ``""``) is still broadcast: the stamp
        preserves the transient no-record semantics, and printers
        that render events locally regardless of the stamp still
        show it.

        Args:
            event: The event to broadcast; must not carry ``tabId``.
            task_id: Explicit task id used when the calling thread
                has no thread-local ``task_id`` bound (off-thread
                calls, task teardown).
            tab_id: Extra tab id treated exactly like every resolved
                watcher, and the sole (possibly empty) stamp of the
                fallback copy when nothing is resolvable.
        """
        _task_key, targets = self._transient_targets(task_id, tab_id)
        for target in targets or [tab_id]:
            self.broadcast({**event, "tabId": target})

    def broadcast_model_pick(
        self,
        model: str,
        source: str,
        tab_id: str,
    ) -> None:
        """Show *model* in the model picker of *tab_id*.

        ``modelPick`` carries an explicit ``tabId`` so it is routed as a
        transient system event: delivered verbatim to every connected
        client (which filters on ``tabId``) and never recorded into the
        task's event log, so replaying a finished conversation cannot
        resurrect a stale picker label.

        Args:
            model: Model name to display.
            source: ``"agent"`` for the display-only model a running
                agent switched itself to, or ``"restore"`` for the
                user's own pick coming back when the task ends.
            tab_id: The tab whose picker to update.
        """
        if not model or not tab_id:
            return
        self.broadcast(
            {
                "type": "modelPick",
                "model": model,
                "source": source,
                "tabId": tab_id,
            },
        )

    def broadcast_agent_model_pick(
        self, model: str, tab_id: str, task_id: Any = None,
    ) -> None:
        """Show a running agent's *model* in every tab watching its task.

        The launching tab plus every viewer subscribed to the agent's
        task (history-resume tabs, chat viewers) get the override, so
        each window watching the agent sees what it is actually
        running.  Every other tab keeps showing its own user's pick.
        Target resolution is shared with :meth:`broadcast_transient`
        (see :meth:`_transient_targets`); this method additionally
        remembers each target so :meth:`restore_model_pick` can hand
        the picker back, which is why it does not simply delegate to
        the plain transient primitive.

        Each target is remembered so the picker can be handed back
        when the task ends — and only then, which is why a task whose
        agent never switched models costs nothing.

        Args:
            model: The model the agent just switched to.
            tab_id: The tab the agent's task was launched in (``""``
                when the agent runs outside a tab).
            task_id: Optional explicit task id used to look up the
                viewer tabs when the calling thread has no
                thread-local ``task_id`` bound (e.g. a call made off
                the agent's run thread).  Ignored when the
                thread-local key is available, which is the normal
                on-thread case.
        """
        if not model:
            return
        # _model_pick_lock: the target registration and the broadcasts
        # form one atomic step w.r.t. restore_model_pick, so a restore
        # that observes a target in _model_override_tabs always puts
        # its own broadcast AFTER the agent broadcast it overrides.
        with self._model_pick_lock:
            # The task key is resolved ONCE, by _transient_targets
            # itself (D-R5): re-deriving it here duplicated the
            # resolution rule and could drift from the key the targets
            # were computed for.
            task_key, targets = self._transient_targets(task_id, tab_id)
            with self._lock:
                self._model_override_tabs.update(targets)
                # Never (re)store the override of a task cleanup_task
                # already closed: the entry would resurrect the
                # completed task's metadata and every LATER subscriber
                # would catch up to the dead agent's model, with no
                # cleanup left to pop it (gpt-5.6-sol conc review,
                # finding 4).  cleanup_task marks _closed_tasks and
                # pops the override under this same _model_pick_lock →
                # _lock pair, so the check cannot be interleaved.  The
                # transient broadcast below still goes out to the
                # lingering subscribers — post-task broadcasts reaching
                # their tabs during the linger window is the
                # subscriber set's documented purpose (see
                # _transient_targets).
                if task_key and task_key not in self._closed_tasks:
                    self._task_model_override[task_key] = model
            for target in targets:
                self.broadcast_model_pick(model, "agent", target)

    def restore_model_pick(self, model: str, tab_id: str) -> None:
        """Put *tab_id*'s own picker back to *model*, if an agent took it.

        A no-op for a tab that never showed an override, so an ordinary
        task ends without putting anything extra on the wire.

        Args:
            model: The model the user picked for this tab.
            tab_id: The tab whose picker to hand back.
        """
        # _model_pick_lock: membership check, discard, and broadcast
        # are atomic w.r.t. the two agent-pick writers, so the wire
        # order of modelPick events always matches the final
        # _model_override_tabs state (see the lock's __init__ comment).
        with self._model_pick_lock:
            with self._lock:
                if tab_id not in self._model_override_tabs:
                    return
                self._model_override_tabs.discard(tab_id)
            self.broadcast_model_pick(model, "restore", tab_id)

    def _inject_task_id(self, event: dict[str, Any]) -> dict[str, Any]:
        """Return *event* with ``taskId`` injected from thread-local storage.

        If *event* already has ``taskId`` set, it is returned unchanged.
        Otherwise the thread-local ``task_id`` (when set) is copied in.

        Args:
            event: The event dictionary.

        Returns:
            The (possibly augmented) event dictionary.
        """
        if event.get("taskId") is not None:
            return event
        key = self._task_key()
        if key:
            return {**event, "taskId": key}
        return event

    def _keep_tab_stamped_task_event(self, event: dict[str, Any]) -> bool:
        """Record and persist a ``tabId``-stamped event under its own task.

        Only the :data:`TAB_STAMPED_TASK_EVENT_TYPES` that also carry a
        ``taskId`` are kept; every other tabId-stamped event is a
        transient targeted broadcast.  A tabId-stripped copy is appended
        to that task's in-memory recording and queued for persistence
        under the event's OWN ``taskId`` — the emitters of these events
        (``commands._echo_injected_prompt``,
        ``commands._broadcast_ask_answer``,
        ``task_runner._broadcast_failure_result``) always stamp the
        persisted ``task_history`` row id.  Filing by that id rather than
        through :meth:`_persist_event` (which resolves the task's LIVE
        agent) matters for the ``/ask`` answer: it may well arrive after
        the task it answers has finished and its agent was cleared, and
        it must still survive a history reopen.

        Args:
            event: The tabId-stamped event (not mutated).

        Returns:
            ``True`` when the event was recorded and persisted, ``False``
            when it was a transient targeted broadcast.
        """
        if (
            event.get("type") not in TAB_STAMPED_TASK_EVENT_TYPES
            or not event.get("taskId")
        ):
            return False
        record = {k: v for k, v in event.items() if k != "tabId"}
        with self._lock:
            self._record_event(record)
        if record.get("type") in _DISPLAY_EVENT_TYPES:
            _queue_chat_event(record, task_id=str(record["taskId"]))
        return True

    def _persist_event(self, event: dict[str, Any]) -> None:
        """Persist a display event to the database if applicable.

        Looks up the agent state registered for ``event["taskId"]``
        and, when its agent has already published a ``last_task_id``,
        enqueues the event for asynchronous persistence via
        ``_queue_chat_event``.  The id is read through the agent's
        property, which takes the same lock the publishing assignment
        takes; it answers ``""`` for an agent that has not run yet,
        and an event can never be filed under an empty id.

        Args:
            event: The event dictionary (must already have ``taskId``
                injected when applicable).
        """
        if event.get("type") not in _DISPLAY_EVENT_TYPES:
            return
        key = self._coerce_task_id(event.get("taskId"))
        if not key:
            return
        state = agent_state.get(key)
        agent = state.agent if state is not None else None
        task_id = getattr(agent, "last_task_id", "")
        if task_id:
            _queue_chat_event(event, task_id=str(task_id))

    def _read_offset(
        self, offsets: dict[str, _OffsetT], default: _OffsetT,
    ) -> _OffsetT:
        """Read the current task's entry of a usage-offset dict.

        Args:
            offsets: The task-keyed offset dict to read.
            default: Value to report when the task has no entry.

        Returns:
            The current task's offset, or *default*.
        """
        with self._lock:
            return offsets.get(self._task_key(), default)

    def _write_offset(
        self, offsets: dict[str, _OffsetT], value: _OffsetT,
    ) -> None:
        """Store the current task's entry of a usage-offset dict.

        Held under ``self._lock`` — the same lock ``cleanup_task``
        pops these dicts under — and silently dropped for a task that
        has already been cleaned up.  Writers are not limited to the
        task's own thread: ``_attribute_sub_usage`` folds a finished
        sub-agent's spend into its parent's offsets from the
        sub-agent's thread, so a write can land after the parent's
        cleanup.  Without the guard that write re-creates an entry
        nothing will ever pop again (R09-7).

        Args:
            offsets: The task-keyed offset dict to write.
            value: The new offset for the current task.
        """
        key = self._task_key()
        with self._lock:
            if key in self._closed_tasks:
                return
            offsets[key] = value

    def set_usage_offsets(
        self, task_id: Any, budget: float, tokens: int, steps: int,
    ) -> None:
        """Store all three usage offsets of an explicitly named task.

        The ``budget_offset`` / ``tokens_offset`` / ``steps_offset``
        setters key on the CALLING thread's task, which is right for the
        task's own agent thread but not for a server or side-channel
        thread folding a child's spend into a parent it does not run:
        ``task_update`` runs the ``/update`` child on a worker thread, and
        its attribution to the parent landed under the worker's (empty)
        key, so the parent's tab under-counted until the parent's own
        thread rewrote its offset.  Naming the task writes it where the
        parent's ``usage_info`` reads.  Dropped for a cleaned-up task,
        like :meth:`_write_offset`.

        Args:
            task_id: The parent task whose offsets change.
            budget: The parent's banked USD total.
            tokens: The parent's banked token total.
            steps: The parent's banked step total.
        """
        key = self._coerce_task_id(task_id)
        with self._lock:
            if key in self._closed_tasks:
                return
            self._budget_offsets[key] = budget
            self._tokens_offsets[key] = tokens
            self._steps_offsets[key] = steps

    @property
    def tokens_offset(self) -> int:
        """Per-task token-count offset used when broadcasting ``usage_info``.

        Backed by a ``task_id``-keyed dict so concurrent tasks never
        clobber each other's accumulated tokens.
        """
        return self._read_offset(self._tokens_offsets, 0)

    @tokens_offset.setter
    def tokens_offset(self, value: int) -> None:
        self._write_offset(self._tokens_offsets, value)

    @property
    def budget_offset(self) -> float:
        """Per-task dollar-budget offset used when broadcasting ``usage_info``."""
        return self._read_offset(self._budget_offsets, 0.0)

    @budget_offset.setter
    def budget_offset(self, value: float) -> None:
        self._write_offset(self._budget_offsets, value)

    @property
    def steps_offset(self) -> int:
        """Per-task step-count offset used when broadcasting ``usage_info``."""
        return self._read_offset(self._steps_offsets, 0)

    @steps_offset.setter
    def steps_offset(self, value: int) -> None:
        self._write_offset(self._steps_offsets, value)

    def cleanup_tab(self, tab_id: str) -> None:
        """Remove *tab_id* from every subscriber and override set.

        Should be called when a frontend tab is closed.  The
        underlying per-task state (recording, bash buffer, offsets)
        is NOT touched here: those belong to the task, not the tab,
        and survive a tab close so a freshly-opened tab on the same
        task can still pick up the running stream.  Call
        :meth:`cleanup_task` to drop the per-task state when the task
        itself ends.

        This also runs when a tab merely re-subscribes (session
        replay, new chat), so it must stay safe to call on a live tab.

        Args:
            tab_id: The frontend tab identifier to drop.
        """
        if not tab_id:
            return
        # _model_pick_lock: this is a WRITER of the model-override
        # lifecycle state, so it must be serialized with the
        # subscribe/pick/restore trio.  Without it a
        # broadcast_agent_model_pick that had already snapshotted its
        # targets could resume after this cleanup and re-add the
        # cleaned tab to _model_override_tabs (and emit a trailing
        # agent modelPick to it) — resurrecting state no later restore
        # could fix (gpt-5.6-sol conc review, finding 4).  Order:
        # _model_pick_lock → _lock, the established edge.
        with self._model_pick_lock, self._lock:
            self._model_override_tabs.discard(tab_id)
            self._sweep_expired_subscribers()
            for task_key in list(self._subscribers.keys()):
                viewers = self._subscribers[task_key]
                viewers.discard(tab_id)
                if not viewers:
                    self._subscribers.pop(task_key, None)
                    self._subscriber_expiry.pop(task_key, None)

    def cleanup_task(
        self,
        task_id: Any,
        subscriber_linger_seconds: float = 300.0,
    ) -> None:
        """Remove all per-task state for *task_id* to free memory.

        Called by the task-runner once a task has fully terminated.
        Cancels any pending bash flush timer and drops the per-task
        recording, persist-agent, and usage-offset entries.

        Bash-state teardown synchronizes with in-flight flushes in two
        steps: the popped state's generation is bumped (under
        ``_bash_lock``, where every flush path re-checks it), so a
        flush that copied text but has not yet passed the generation
        re-check discards it; then the state's ``flush_lock`` is
        acquired and released (after ``_bash_lock`` is dropped, so the
        lock order matches the flush paths), so a flush that already
        passed its re-check and is broadcasting finishes BEFORE this
        method returns.  After ``cleanup_task`` returns, no stale
        ``system_output`` for the task can be broadcast.  The
        ``_closed_tasks`` mark (under ``_lock``) lands BEFORE the bash
        pop, so a straggler ``print(type="bash_stream")`` — whose
        guard checks the mark and creates state atomically under
        ``_lock`` — can never re-create the entry under the dead key.

        The subscriber set is preserved for ``subscriber_linger_seconds``
        so a broadcast that lands just after the task ends still fans
        out to the originating tab.
        Expired sets are pruned opportunistically (no timer thread per
        task) by every subscriber-map operation — previously they were
        kept for the tab's whole lifetime, leaking one entry per
        completed task in long-lived tabs.  A tab that closes earlier
        is still removed immediately via :meth:`cleanup_tab`.

        Args:
            task_id: The task identifier whose state should be freed.
            subscriber_linger_seconds: How long the task's subscriber
                set survives to serve post-task broadcasts; ``<= 0``
                prunes synchronously.
        """
        key = self._coerce_task_id(task_id)
        if not key:
            return
        # The ``_closed_tasks`` mark (under ``_lock``, below) is taken
        # BEFORE the bash state is popped (under ``_bash_lock``): the
        # ``bash_stream`` branch of :meth:`print` checks the mark and
        # creates its state atomically under ``_lock``, so with this
        # order every state a straggler fragment could create either
        # sees the mark (and is dropped) or still exists when the pop
        # below runs — no interleaving can resurrect an entry keyed by
        # a dead task id.
        #
        # _model_pick_lock: this pops ``_task_model_override`` — model-
        # override lifecycle state — so it must be serialized with the
        # subscribe/pick/restore trio.  Without it a
        # broadcast_agent_model_pick paused between its target snapshot
        # and its state write could resume after this cleanup and
        # recreate the completed task's override, feeding a dead
        # agent's model to every later subscriber (gpt-5.6-sol conc
        # review, finding 4).  Order: _model_pick_lock → _lock.
        with self._model_pick_lock, self._lock:
            self._recordings.pop(key, None)
            self._changed_paths.pop(key, None)
            self._task_model_override.pop(key, None)
            self._tokens_offsets.pop(key, None)
            self._budget_offsets.pop(key, None)
            self._steps_offsets.pop(key, None)
            self._tool_call_started.pop(key, None)
            self._image_b64_used.pop(key, None)
            self._closed_tasks.pop(key, None)
            self._closed_tasks[key] = None
            while len(self._closed_tasks) > _CLOSED_TASK_MEMORY:
                self._closed_tasks.pop(next(iter(self._closed_tasks)))
                self._closed_tasks_evicted = True
            # The launching-tab entry dies WITH the task (unlike the
            # subscriber set, which lingers below to serve post-task
            # broadcasts): _transient_targets must never route through
            # a tab whose task already ended.
            self._task_ui.pop(key, None)
            if key in self._subscribers:
                if subscriber_linger_seconds <= 0:
                    self._subscribers.pop(key, None)
                    self._subscriber_expiry.pop(key, None)
                else:
                    self._subscriber_expiry[key] = time.monotonic() + subscriber_linger_seconds
            self._sweep_expired_subscribers()
        with self._bash_lock:
            bs = self._bash_states.pop(key, None)
            if bs is not None:
                if bs.timer is not None:
                    bs.timer.cancel()
                bs.generation += 1
                bs.buffer.clear()
        if bs is not None:
            # Wait out a flush that passed its generation re-check
            # before the bump and is still broadcasting under
            # ``flush_lock`` — its output belongs to the task's
            # lifetime and must land before cleanup completes.
            with bs.flush_lock:
                pass

    def _sweep_expired_subscribers(self) -> None:
        """Drop subscriber sets whose post-task linger has expired.

        Must be called with ``self._lock`` held.  Cheap when nothing
        is pending (the expiry map only holds completed tasks still
        inside their linger window), so every subscriber-map operation
        can afford to call it — this replaces a per-task timer thread.
        """
        if not self._subscriber_expiry:
            return
        now = time.monotonic()
        for key, deadline in list(self._subscriber_expiry.items()):
            if now >= deadline:
                del self._subscriber_expiry[key]
                self._subscribers.pop(key, None)

    def reset(self) -> None:
        """Reset internal streaming state for a new turn.

        Holds the per-task ``flush_lock`` across the generation bump so
        an in-flight flush that already passed its generation re-check
        (and is broadcasting under ``flush_lock``) finishes before the
        new turn starts — after ``reset()`` returns, no stale bash text
        from the previous turn can be broadcast.
        """
        self._current_block_type = ""
        with self._bash_lock:
            # Non-creating lookup, like _flush_bash and the tool_call /
            # tool_result branches: there is nothing to reset when the
            # task has no bash state, and creating one here retained an
            # entry under the "" key of a task-less thread that
            # cleanup_task can never remove (R09-7).
            bs = self._bash_states.get(self._task_key())
        if bs is None:
            return
        with bs.flush_lock:
            with self._bash_lock:
                bs.generation += 1
                bs.buffer.clear()
                bs.streamed = False
                if bs.timer is not None:
                    bs.timer.cancel()
                    bs.timer = None

    def _timer_flush_for_task(self, task_id: str | None) -> None:
        """Timer callback that sets the thread-local task_id and flushes bash.

        Used by the bash-stream buffering timer so the flushed event
        is attributed to the right task even when the timer runs on a
        worker thread that has no thread-local task_id of its own.

        Args:
            task_id: The task identifier that owns the bash buffer, or
                ``None`` when no task context is available.
        """
        if task_id is not None:
            self._thread_local.task_id = task_id
        self._flush_bash()

    def _flush_bash(self) -> None:
        """Flush the bash buffer.

        Captures the generation counter inside ``_bash_lock`` along with
        the buffered text.  After releasing the lock, re-checks the
        generation (inside a second ``_bash_lock`` acquisition) while
        holding the state's per-task ``flush_lock``: if ``reset()`` ran
        in between (incrementing the generation), the captured text is
        stale and is discarded.  W2-F5: the ``broadcast()`` itself
        happens under ``flush_lock`` but NOT under the printer-global
        ``_bash_lock`` — ``reset()`` also takes ``flush_lock`` before
        bumping the generation, so the reset-vs-flush TOCTOU stays
        closed while a slow transport ``broadcast`` (socket sends in
        ``WebPrinter``) no longer blocks every other task's
        ``print(type="bash_stream")`` behind ``_bash_lock``.

        Uses a NON-creating state lookup: a straggler flush (e.g. a
        timer callback that fired before ``cleanup_task`` could cancel
        it) must not resurrect the just-freed ``_BashState`` — the
        ``_bash_state`` property would re-insert it into
        ``_bash_states`` keyed by a dead task id, leaking it forever
        and allowing a stale ``system_output`` broadcast attributed to
        the finished task.  A missing state has nothing to flush.
        """
        with self._bash_lock:
            bs = self._bash_states.get(self._task_key())
            if bs is None:
                return
            gen = bs.generation
            if bs.timer is not None:
                bs.timer.cancel()
                bs.timer = None
            text = "".join(bs.buffer) if bs.buffer else ""
            bs.buffer.clear()
            bs.last_flush = time.monotonic()
        if text:
            with bs.flush_lock:
                with self._bash_lock:
                    if bs.generation != gen:
                        return
                self.broadcast({"type": "system_output", "text": text})

    def start_recording(self) -> None:
        """Start recording broadcast events for the current task.

        No-op when no thread-local ``task_id`` is set.
        """
        key = self._task_key()
        if not key:
            return
        with self._lock:
            self._recordings[key] = []

    def ensure_recording_for_task(self, task_id: Any) -> None:
        """Make sure an event recording exists for *task_id*.

        Unlike :meth:`start_recording` this is keyed explicitly (no
        thread-local binding needed) and never clears an existing
        recording.  Used by the task runner for a run that fails in
        SETUP — before ``ChatSorcarAgent.run`` ever started the run's
        recording — so its terminal ``result`` can be recorded under
        the run's (possibly provisional) task id and replayed to a
        viewer that attached inside the end-of-run race window
        (audit0903 F4).

        Args:
            task_id: The task identifier (``task_history.id`` int, its
                string form, or a provisional client/registry id).
        """
        key = self._coerce_task_id(task_id)
        if not key:
            return
        with self._lock:
            self._recordings.setdefault(key, [])

    @staticmethod
    def _filter_and_coalesce(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter to display events and merge consecutive deltas.

        Args:
            raw: Unfiltered list of recorded events.

        Returns:
            Display-relevant events with consecutive deltas merged.
        """
        filtered = [e for e in raw if e.get("type") in _DISPLAY_EVENT_TYPES]
        return _coalesce_events(filtered)

    def stop_recording(self) -> list[dict[str, Any]]:
        """Stop recording for the current task and return its display events.

        Returns:
            List of display-relevant events with consecutive deltas
            merged.  Empty when no recording is active.
        """
        key = self._task_key()
        if not key:
            return []
        with self._lock:
            raw = self._recordings.pop(key, [])
        return self._filter_and_coalesce(raw)

    def peek_recording(self) -> list[dict[str, Any]]:
        """Return a snapshot of the current task's recording.

        Used for periodic crash-recovery flushes: the caller can
        persist a snapshot of events to the database while recording
        continues.

        Returns:
            List of display-relevant events with consecutive deltas
            merged.  Empty when no recording is active.
        """
        return self.peek_recording_for_task(self._task_key())

    def peek_recording_for_task(self, task_id: Any) -> list[dict[str, Any]]:
        """Return a snapshot of *task_id*'s in-memory recording.

        Like :meth:`peek_recording` but keyed explicitly instead of by
        the calling thread's task binding.  Used by the server when a
        tab opens a STILL-RUNNING task (e.g. a freshly spawned
        ``run_parallel`` sub-agent): the task's events reach the
        database through an asynchronous writer, so a replay loaded
        from the events table can miss the transcript head — the live
        recording is the authoritative copy while the task runs.

        Args:
            task_id: The task identifier (``task_history.id`` int or
                its string form).

        Returns:
            List of display-relevant events with consecutive deltas
            merged.  Empty when the task has no active recording.
        """
        key = self._coerce_task_id(task_id)
        if not key:
            return []
        with self._lock:
            rec = self._recordings.get(key)
            raw = list(rec) if rec is not None else []
        return self._filter_and_coalesce(raw)

    def _record_event(self, event: dict[str, Any]) -> None:
        """Append *event* to the active recording for its task.

        Looks up the recording list by the event's ``taskId``, falling
        back to the thread-local ``task_id``.  Must be called with
        ``self._lock`` held.
        """
        key = self._coerce_task_id(
            event.get("taskId") or getattr(self._thread_local, "task_id", None),
        )
        if not key:
            return
        rec = self._recordings.get(key)
        if rec is not None:
            rec.append(event)

    def _track_changed_path(self, event: dict[str, Any]) -> None:
        """Record the file path of a mutating ``tool_call`` under its task.

        Only ``Write`` / ``Edit`` calls are tracked — the tools whose
        ``path`` names a file the agent changed (``Read`` events carry
        a path too but change nothing; ``Bash`` changes cannot be
        attributed to paths).  Must be called with ``self._lock`` held.

        Args:
            event: A broadcast event, already task-id-injected.
        """
        if event.get("type") != "tool_call":
            return
        if event.get("name") not in _FILE_MUTATING_TOOLS:
            return
        path = event.get("path")
        key = self._coerce_task_id(event.get("taskId"))
        if not path or not key:
            return
        self._changed_paths.setdefault(key, set()).add(str(path))

    def pop_changed_paths(self, task_id: Any) -> set[str]:
        """Return and clear the file paths *task_id*'s tools changed.

        Called once by the task-runner's end-of-task auto-commit.
        Popping (rather than reading) keeps the map from accumulating
        entries for tasks whose runner never consumed them.

        Args:
            task_id: The task identifier whose changed paths to take.

        Returns:
            The set of absolute path strings recorded for the task
            (empty when nothing was tracked).
        """
        key = self._coerce_task_id(task_id)
        if not key:
            return set()
        with self._lock:
            return self._changed_paths.pop(key, set())

    def broadcast(self, event: dict[str, Any]) -> None:
        """Inject the thread-local taskId, record, and persist the event.

        Subclasses that own a transport (WSS / UDS sockets, etc.) add
        their own emission logic AFTER calling the recording /
        persistence path — see :class:`WebPrinter` in
        ``web_server.py``.  The default implementation here is
        sufficient for tests that only need the recording and
        persistence side effects.

        A ``recordOnly`` marker (a durable copy of a prompt echo that
        was already rendered live at queueing time — see
        ``SorcarAgent._drain_pending_user_messages``) is stripped
        before recording; this default implementation has no transport,
        so record + persist is exactly the marker's semantics.

        Args:
            event: The event dictionary to broadcast.
        """
        stamp_event_ts(event)
        event.pop("recordOnly", None)
        if "tabId" in event:
            self._keep_tab_stamped_task_event(event)
            return
        event = self._inject_task_id(event)
        with self._lock:
            self._record_event(event)
            self._track_changed_path(event)
        self._persist_event(event)

    def _cost_with_offset(self, cost: Any) -> Any:
        """Add the per-task budget offset to a ``"$…"`` cost string.

        Non-dollar or malformed costs (e.g. ``"N/A"``, ``"$abc"``) are
        returned verbatim so a junk value never raises out of the
        emitting agent thread.

        Args:
            cost: The raw cost value (usually a ``"$1.2345"`` string).

        Returns:
            The offset-adjusted cost string, or *cost* unchanged.
        """
        if isinstance(cost, str) and cost.startswith("$"):
            try:
                return f"${float(cost[1:]) + self.budget_offset:.4f}"
            except ValueError:
                pass
        return cost

    def _broadcast_result(
        self,
        text: str,
        total_tokens: int = 0,
        cost: str = "N/A",
        step_count: int = 0,
    ) -> None:
        cost = self._cost_with_offset(cost)
        total_tokens = total_tokens + self.tokens_offset
        step_count = step_count + self.steps_offset
        event: dict[str, Any] = {
            "type": "result",
            "text": text or "(no result)",
            "total_tokens": total_tokens,
            "cost": cost,
            "step_count": step_count,
        }
        parsed = parse_result_yaml(text) if text else None
        if parsed:
            event["success"] = parsed.get("success")
            event["is_continue"] = bool(parsed.get("is_continue", False))
            event["summary"] = str(parsed["summary"])
        self.broadcast(event)

    def _check_stop(self) -> None:
        ev = getattr(self._thread_local, "stop_event", None)
        if ev is not None and ev.is_set():
            raise KeyboardInterrupt("Agent stop requested")

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Render content by broadcasting events to connected clients.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "tool_call",
                "tool_result", "result", "message").
            **kwargs: Additional options such as tool_input, is_error, cost,
                total_tokens.

        Returns:
            str: Always the empty string.
        """
        self._check_stop()
        if type == "text":
            from io import StringIO

            from rich.console import Console

            buf = StringIO()
            Console(file=buf, highlight=False, width=120, no_color=True).print(content)
            text = buf.getvalue()
            if text.strip():
                self.broadcast({"type": "text_delta", "text": text})
            return ""
        if type in ("system_prompt", "prompt"):
            self.broadcast({"type": type, "text": str(content)})
            return ""
        if type == "message":
            self._handle_message(content, **kwargs)
            return ""
        if type == "bash_stream":
            text = ""
            gen = 0
            with self._lock:
                # No-resurrection guard, closing the one straggler
                # path the sibling guards (_flush_bash,
                # _emit_tool_result, _format_tool_call,
                # _collect_result_images, _write_offset) left open: a
                # bash fragment arriving after ``cleanup_task`` must
                # not re-create the freed ``_BashState`` — the creating
                # ``_bash_state`` property would insert an entry keyed
                # by a dead task id that nothing ever pops (a permanent
                # leak), and its immediate flush (``last_flush`` 0.0)
                # would broadcast post-task ``system_output`` to the
                # lingering subscriber set.  The check and the create
                # are ATOMIC w.r.t. ``cleanup_task``, which marks
                # ``_closed_tasks`` under ``_lock`` BEFORE popping the
                # state under ``_bash_lock``: either the mark precedes
                # this block (fragment dropped) or the state created
                # here still exists when the pop runs.  Lock order
                # ``_lock`` → ``_bash_lock`` is new and acyclic — no
                # path acquires ``_lock`` while holding ``_bash_lock``
                # (broadcasts run under ``flush_lock`` only).  The
                # task-less ""-key path is unaffected: cleanup_task
                # never closes "".
                key = self._task_key()
                if key in self._closed_tasks:
                    return ""
                with self._bash_lock:
                    if (
                        key
                        and key not in self._bash_states
                        and not self._bash_task_may_create_state(key)
                    ):
                        return ""
                    bs = self._bash_state
                    bs.buffer.append(str(content))
                    gen = bs.generation
                    if time.monotonic() - bs.last_flush >= 0.1:
                        if bs.timer is not None:
                            bs.timer.cancel()
                            bs.timer = None
                        text = "".join(bs.buffer)
                        bs.buffer.clear()
                        bs.last_flush = time.monotonic()
                    elif bs.timer is None:
                        owner_task = getattr(self._thread_local, "task_id", None)
                        bs.timer = threading.Timer(
                            0.1,
                            partial(self._timer_flush_for_task, owner_task),
                        )
                        bs.timer.daemon = True
                        bs.timer.start()
            if text:
                with bs.flush_lock:
                    stale = False
                    with self._bash_lock:
                        stale = bs.generation != gen
                    if not stale:
                        self.broadcast(
                            {"type": "system_output", "text": text},
                        )
            with self._bash_lock:
                # Use the state captured above — the creating
                # ``_bash_state`` property would resurrect a state
                # that ``cleanup_task`` freed while ``broadcast`` was
                # running, leaking it forever under a dead task id.
                bs.streamed = True
            return ""
        if type == "tool_call":
            self._flush_bash()
            with self._bash_lock:
                live = self._bash_states.get(self._task_key())
                if live is not None:
                    live.streamed = False
            self.broadcast({"type": "text_end"})
            self._format_tool_call(
                str(content),
                kwargs.get("tool_input", {}),
                call_id=kwargs.get("call_id"),
            )
            return ""
        if type == "tool_result":
            self._emit_tool_result(
                content,
                tool_name=kwargs.get("tool_name", ""),
                is_error=kwargs.get("is_error", False),
                tool_input=kwargs.get("tool_input"),
                interrupted=bool(kwargs.get("interrupted", False)),
            )
            return ""
        if type == "usage_info":
            raw_tokens = kwargs.get("total_tokens", 0)
            raw_cost = kwargs.get("cost", "N/A")
            raw_steps = kwargs.get("total_steps", 0)
            total_tokens = raw_tokens + self.tokens_offset
            total_steps = raw_steps + self.steps_offset
            total_cost = self._cost_with_offset(raw_cost)
            event: dict[str, Any] = {
                "type": "usage_info",
                "text": str(content),
                "total_tokens": total_tokens,
                "cost": total_cost,
                "total_steps": total_steps,
            }
            # Per-step provenance for the cost report: the model that
            # served the step (a ``set_model`` switch is otherwise
            # invisible in task_history) and its prompt-cache read
            # tokens (0 = the static prefix missed the cache).
            for key in ("cache_read", "model"):
                if key in kwargs:
                    event[key] = kwargs[key]
            self.broadcast(event)
            return ""
        if type == "result":
            self.broadcast({"type": "text_end"})
            self._broadcast_result(
                str(content),
                kwargs.get("total_tokens", 0),
                kwargs.get("cost", "N/A"),
                kwargs.get("step_count", 0),
            )
            return ""
        return ""

    def _emit_tool_result(
        self,
        content: Any,
        *,
        tool_name: str,
        is_error: Any,
        tool_input: Any,
        interrupted: bool = False,
    ) -> None:
        """Broadcast a ``tool_result`` event with the shared treatment.

        Single emission path for both ``print(type="tool_result")`` and
        the message-object route of :meth:`_handle_message`, so the two
        cannot drift apart (W3-D4): every ``tool_result`` event carries
        ``tool_name`` (downstream consumers key panel labels and
        highlighting on it), ``finish`` results are suppressed, and
        already-streamed bash output is deduplicated.

        Args:
            content: The tool's return value.
            tool_name: Name of the tool that produced the result.
            is_error: Whether the result represents an error.
            tool_input: The originating tool input dict when available
                (used to stamp ``path`` / ``start_line`` for Read
                results).
            interrupted: ``True`` when the user stopped the tool call
                through its panel's Stop button.  The event is flagged
                ``interrupted`` and keeps its content even for a Bash
                call whose output was streamed (normally blanked as a
                duplicate), so the panel can show that the call was
                cut short rather than ending silently.
        """
        self._flush_bash()
        show_result = tool_name != "finish"
        with self._bash_lock:
            # Non-creating lookup: a tool result arriving after
            # ``cleanup_task`` must not resurrect the freed state.
            live = self._bash_states.get(self._task_key())
            streamed = live.streamed if live is not None else False
            if live is not None:
                live.streamed = False
        # Consume the pairing tool_call's start time: this result ends
        # the call, and a LATER result arriving without its own
        # tool_call (message routes, extra blocks) must use the short
        # fallback window rather than this call's stale cutoff.
        with self._lock:
            started = self._tool_call_started.pop(self._task_key(), None)
        result_content = (
            "" if streamed and not interrupted else truncate_result(str(content))
        )
        if show_result:
            event: dict[str, Any] = {
                "type": "tool_result",
                "content": result_content,
                "is_error": is_error,
                "tool_name": tool_name,
            }
            if interrupted:
                event["interrupted"] = True
            if isinstance(tool_input, dict):
                path = tool_input.get("file_path") or tool_input.get("path")
                if path:
                    event["path"] = str(path)
                start_line = tool_input.get("start_line")
                if isinstance(start_line, int) and start_line >= 1:
                    event["start_line"] = start_line
            if not is_error:
                images = self._collect_result_images(str(content), tool_input, started)
                if images:
                    event["images"] = images
            self.broadcast(event)

    def _task_work_dir(self) -> str:
        """Best-effort working directory of the current task.

        Duck-typed lookup through the agent registered for the task
        (``RelentlessAgent`` and its subclasses expose ``work_dir``);
        answers ``""`` when no task/agent is known, in which case
        relative image paths are resolved against the process cwd.

        Returns:
            The task's work dir, or ``""`` when unknown.
        """
        state = agent_state.get(self._task_key())
        agent = state.agent if state is not None else None
        work_dir = getattr(agent, "work_dir", "") or ""
        return str(work_dir)

    def _collect_result_images(
        self, content: str, tool_input: Any, started: float | None
    ) -> list[dict[str, str]]:
        """Embed images a tool call generated as base64 display payloads.

        Scans the tool's return text — plus the call's ``file_path`` /
        ``path`` argument, covering ``screenshot(file_path=...)`` and
        ``Write`` of an SVG — for image-file paths, and keeps only
        files that were created or modified while this tool call ran
        (file mtime >= the ``tool_call`` broadcast time minus slack).
        The recency gate keeps images that are merely *mentioned* — an
        ``ls`` of an old assets directory, a grep hit on ``logo.png``
        — out of the event panel.  The bytes are embedded (not linked)
        because worktree runs delete the file's directory after the
        merge, which would break transcript replays.  A per-task byte
        budget (``_MAX_TASK_IMAGE_B64_BYTES``) stops embedding before
        the task's replayed event list outgrows the extension's frame
        limit.

        Args:
            content: The tool's full return text (untruncated).
            tool_input: The originating tool input dict when available.
            started: Wall-clock time of the pairing ``tool_call``
                broadcast, or ``None`` when this result has no pairing
                call (then the short fallback window applies).

        Returns:
            Up to ``_MAX_RESULT_IMAGES`` dicts with ``path`` (as the
            tool named it), ``mime``, and ``b64`` keys, oldest mention
            first.  Empty when nothing qualifies.
        """
        candidates: list[str] = []
        if isinstance(tool_input, dict):
            for key in ("file_path", "path"):
                arg = tool_input.get(key)
                if isinstance(arg, str) and arg:
                    candidates.append(arg)
        candidates.extend(_extract_image_path_candidates(content))
        candidates = [
            c for c in candidates if os.path.splitext(c)[1].lower() in _IMAGE_MIME_BY_EXT
        ]
        if not candidates:
            return []
        if started is not None:
            cutoff = started - _RESULT_IMAGE_RECENCY_SLACK_SECS
        else:
            cutoff = time.time() - _RESULT_IMAGE_FALLBACK_WINDOW_SECS
        work_dir = self._task_work_dir()
        images: list[dict[str, str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            path = candidate
            if not os.path.isabs(path) and work_dir:
                path = os.path.join(work_dir, path)
            try:
                resolved = Path(path).resolve()
                key = str(resolved)
                if key in seen:
                    continue
                seen.add(key)
                if not resolved.is_file():
                    continue
                st = resolved.stat()
                if (
                    st.st_mtime < cutoff
                    or st.st_size == 0
                    or st.st_size > _MAX_RESULT_IMAGE_BYTES
                ):
                    continue
                b64 = base64.b64encode(resolved.read_bytes()).decode("ascii")
            except OSError:
                logger.debug("Skipping unreadable result image", exc_info=True)
                continue
            task_key = self._task_key()
            with self._lock:
                used = self._image_b64_used.get(task_key, 0)
                if used + len(b64) > _MAX_TASK_IMAGE_B64_BYTES:
                    logger.debug(
                        "Per-task image budget exhausted; not embedding %s",
                        candidate,
                    )
                    continue
                # Same guards as _write_offset / _tool_call_started:
                # no empty key (nothing pops ""), no resurrection
                # after cleanup_task.
                if task_key and task_key not in self._closed_tasks:
                    self._image_b64_used[task_key] = used + len(b64)
            images.append(
                {
                    "path": candidate,
                    "mime": _IMAGE_MIME_BY_EXT[os.path.splitext(candidate)[1].lower()],
                    "b64": b64,
                }
            )
            if len(images) >= _MAX_RESULT_IMAGES:
                break
        return images

    def token_callback(self, token: str) -> None:
        """Broadcast a streamed token as a delta event.

        Args:
            token: The text token to broadcast.
        """
        self._check_stop()
        if token:
            delta_type = (
                "thinking_delta" if self._current_block_type == "thinking" else "text_delta"
            )
            self.broadcast({"type": delta_type, "text": token})

    def thinking_callback(self, is_start: bool) -> None:
        """Handle thinking-block boundary events.

        Sets ``_current_block_type`` so that subsequent ``token_callback``
        tokens are routed to the thinking panel, and broadcasts
        ``thinking_start`` / ``thinking_end`` events.

        Args:
            is_start: ``True`` when a thinking block starts, ``False`` when it ends.
        """
        if is_start:
            self._current_block_type = "thinking"
            self.broadcast({"type": "thinking_start"})
        else:
            self._current_block_type = ""
            self.broadcast({"type": "thinking_end"})

    def _format_tool_call(
        self,
        name: str,
        tool_input: dict[str, Any],
        call_id: Any = None,
    ) -> None:
        """Broadcast a ``tool_call`` event for *name*.

        Args:
            name: The tool's name.
            tool_input: The call's arguments (rendered into the event's
                ``path`` / ``command`` / ``content`` / diff fields).
            call_id: The agent's per-call id
                (``kiss.core.tool_interrupt.ToolCallToken.call_id``),
                stamped as ``callId`` so the panel's Stop button can
                name exactly this call in its ``interruptTool`` command.
        """
        key = self._task_key()
        with self._lock:
            # Same guard as _write_offset: a straggler tool_call for a
            # cleaned-up task must not re-create an entry nothing pops.
            # Taskless calls (key "") are not recorded at all — no
            # cleanup_task ever pops "".
            if key and key not in self._closed_tasks:
                self._tool_call_started[key] = time.time()
        file_path, lang = extract_path_and_lang(tool_input)
        event: dict[str, Any] = {"type": "tool_call", "name": name}
        if isinstance(call_id, int) and not isinstance(call_id, bool):
            event["callId"] = call_id
        if file_path:
            event["path"] = file_path
            event["lang"] = lang
        if desc := tool_input.get("description"):
            event["description"] = str(desc)
        if command := tool_input.get("command"):
            event["command"] = str(command)
        if content := tool_input.get("content"):
            event["content"] = str(content)
        old_string = tool_input.get("old_string")
        new_string = tool_input.get("new_string")
        if old_string is not None:
            event["old_string"] = str(old_string)
        if new_string is not None:
            event["new_string"] = str(new_string)
        extras = extract_extras(tool_input)
        if extras:
            event["extras"] = extras
        self.broadcast(event)

    def _handle_message(self, message: Any, **kwargs: Any) -> None:
        if hasattr(message, "subtype") and hasattr(message, "data"):
            if message.subtype == "tool_output":
                text = message.data.get("content", "")
                if text:
                    self.broadcast({"type": "system_output", "text": text})
        elif hasattr(message, "result"):
            budget_used = kwargs.get("budget_used", 0.0)
            self._broadcast_result(
                message.result,
                kwargs.get("total_tokens_used", 0),
                f"${budget_used:.4f}" if budget_used else "N/A",
            )
        elif hasattr(message, "content"):
            blocks = [
                block
                for block in message.content
                if hasattr(block, "is_error") and hasattr(block, "content")
            ]
            shared_input = kwargs.get("tool_input") if len(blocks) == 1 else None
            for block in blocks:
                self._emit_tool_result(
                    block.content,
                    tool_name=(getattr(block, "tool_name", "") or kwargs.get("tool_name", "")),
                    is_error=bool(block.is_error),
                    tool_input=shared_input,
                )
