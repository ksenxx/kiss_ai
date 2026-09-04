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

import json
import logging
import threading
import time
from functools import partial
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

# Tools whose ``tool_call`` event names a file the agent CHANGED (as
# opposed to merely read).  Used to track, per task, which files the
# task modified so the end-of-task auto-commit can also commit repos
# other than the tab's work_dir one.
_FILE_MUTATING_TOOLS = frozenset({"Write", "Edit"})


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

    def __init__(self) -> None:
        self._thread_local = _PrinterThreadLocal()
        self._lock = threading.Lock()
        self._bash_lock = threading.Lock()
        self._bash_states: dict[str, _BashState] = {}
        self._tokens_offsets: dict[str, int] = {}
        self._budget_offsets: dict[str, float] = {}
        self._steps_offsets: dict[str, int] = {}
        # Task ids whose state cleanup_task already freed, newest last
        # and capped at _CLOSED_TASK_MEMORY entries.  A late usage
        # offset write for one of them is dropped instead of leaking a
        # dict entry nothing would pop again.
        self._closed_tasks: dict[str, None] = {}
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
        with self._lock:
            self._sweep_expired_subscribers()
            viewers = self._subscribers.get(key)
            if viewers is None:
                viewers = set()
                self._subscribers[key] = viewers
            viewers.add(tab_id)
            # A tab joining a task whose agent already switched models
            # missed that one-shot event, and would otherwise sit on the
            # wrong label until the task ended.
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
        # The task key is resolved ONCE, by _transient_targets itself
        # (D-R5): re-deriving it here duplicated the resolution rule
        # and could drift from the key the targets were computed for.
        task_key, targets = self._transient_targets(task_id, tab_id)
        with self._lock:
            self._model_override_tabs.update(targets)
            if task_key:
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
        with self._lock:
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
        ``system_output`` for the task can be broadcast.

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
        with self._lock:
            self._recordings.pop(key, None)
            self._changed_paths.pop(key, None)
            self._task_model_override.pop(key, None)
            self._tokens_offsets.pop(key, None)
            self._budget_offsets.pop(key, None)
            self._steps_offsets.pop(key, None)
            self._closed_tasks.pop(key, None)
            self._closed_tasks[key] = None
            while len(self._closed_tasks) > _CLOSED_TASK_MEMORY:
                self._closed_tasks.pop(next(iter(self._closed_tasks)))
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
            if event.get("type") in ("prompt", "result") and event.get("taskId"):
                record = {k: v for k, v in event.items() if k != "tabId"}
                with self._lock:
                    self._record_event(record)
                self._persist_event(record)
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
            with self._bash_lock:
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
            self._format_tool_call(str(content), kwargs.get("tool_input", {}))
            return ""
        if type == "tool_result":
            self._emit_tool_result(
                content,
                tool_name=kwargs.get("tool_name", ""),
                is_error=kwargs.get("is_error", False),
                tool_input=kwargs.get("tool_input"),
            )
            return ""
        if type == "usage_info":
            raw_tokens = kwargs.get("total_tokens", 0)
            raw_cost = kwargs.get("cost", "N/A")
            raw_steps = kwargs.get("total_steps", 0)
            total_tokens = raw_tokens + self.tokens_offset
            total_steps = raw_steps + self.steps_offset
            total_cost = self._cost_with_offset(raw_cost)
            self.broadcast(
                {
                    "type": "usage_info",
                    "text": str(content),
                    "total_tokens": total_tokens,
                    "cost": total_cost,
                    "total_steps": total_steps,
                }
            )
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
        result_content = "" if streamed else truncate_result(str(content))
        if show_result:
            event: dict[str, Any] = {
                "type": "tool_result",
                "content": result_content,
                "is_error": is_error,
                "tool_name": tool_name,
            }
            if isinstance(tool_input, dict):
                path = tool_input.get("file_path") or tool_input.get("path")
                if path:
                    event["path"] = str(path)
                start_line = tool_input.get("start_line")
                if isinstance(start_line, int) and start_line >= 1:
                    event["start_line"] = start_line
            self.broadcast(event)

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

    def _format_tool_call(self, name: str, tool_input: dict[str, Any]) -> None:
        file_path, lang = extract_path_and_lang(tool_input)
        event: dict[str, Any] = {"type": "tool_call", "name": name}
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
