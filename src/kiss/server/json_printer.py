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

import threading
import time
from functools import partial
from typing import Any

from kiss.agents.sorcar.persistence import _queue_chat_event
from kiss.core import stop_signal
from kiss.core.printer import (
    Printer,
    extract_extras,
    extract_path_and_lang,
    parse_result_yaml,
    truncate_result,
)

_DISPLAY_EVENT_TYPES = frozenset({
    "clear", "thinking_start", "thinking_delta", "thinking_end",
    "text_delta", "text_end", "tool_call", "tool_result",
    "system_output", "result", "system_prompt", "prompt",
    "task_done", "task_error", "task_stopped", "task_interrupted",
    "followup_suggestion",
    "autocommit_done",
    "warning",
})


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


class _BashState:
    """Bash buffering state for streaming output.

    Buffers bash output fragments and flushes them periodically to
    avoid overwhelming the frontend with tiny events.
    """

    __slots__ = (
        "buffer", "timer", "generation", "last_flush", "streamed",
        "flush_lock",
    )

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.timer: threading.Timer | None = None
        self.generation: int = 0
        self.last_flush: float = 0.0
        self.streamed: bool = False
        self.flush_lock = threading.Lock()


# Interactive-UI events that put something on screen until it is
# answered, mapped to the events that take it back off again.  A
# mirror lives exactly as long as one of these is on screen.
_UI_CLOSE_EVENTS: dict[str, tuple[str, ...]] = {
    "merge_ended": ("merge_data", "merge_started", "merge_nav"),
    "autocommit_done": ("autocommit_prompt",),
    "worktree_result": ("worktree_done",),
}
_UI_OPEN_EVENTS: frozenset[str] = frozenset(
    event_type
    for closed in _UI_CLOSE_EVENTS.values()
    for event_type in closed
)


class _UiMirror:
    """One tab's interactive UI, mirrored onto other clients' tabs.

    A chat can be open in several frontend tabs at once (the VS Code
    window that launched the task, a browser on a phone, a second
    laptop).  Blocking UIs — the merge/diff review, the auto-commit
    prompt, the worktree merge/discard strip — are owned by exactly
    one tab: the on-disk merge artifacts, the server-side hunk cursor
    and the git working tree all hang off that tab's id.  This record
    names the other tabs that must render the same UI, so opening and
    closing it stays in lock step across every client.

    Attributes:
        viewer_tab_ids: Tabs of other clients showing the same task.
        work_dir: The owner's working directory, used when an action
            arrives from a viewer whose own folder may differ.
        task_key: The task the UI belongs to, so a tab that joins
            the task later can be shown the same UI.
        open_events: The latest still-unanswered event of each open
            type, replayed verbatim to such a late joiner.
    """

    __slots__ = ("viewer_tab_ids", "work_dir", "task_key", "open_events")

    def __init__(self, work_dir: str, task_key: str) -> None:
        self.viewer_tab_ids: list[str] = []
        self.work_dir = work_dir
        self.task_key = task_key
        self.open_events: dict[str, dict[str, Any]] = {}


def _orphaned_ui_close_events(
    owner_tab_id: str, mirror: _UiMirror,
) -> list[dict[str, Any]]:
    """Return the events that take *mirror*'s UI off the viewers' screens.

    Used when the owner tab disappears with a UI still open.  One
    closing event per still-open UI, per viewer, carrying the reason so
    the user sees why the buttons went away rather than watching them
    vanish.

    Args:
        owner_tab_id: The tab that owned the UI and has now gone.
        mirror: Its mirror record, already detached from the printer.

    Returns:
        Ready-to-broadcast closing events.
    """
    closers = {
        close_type
        for close_type, opened in _UI_CLOSE_EVENTS.items()
        if any(open_type in mirror.open_events for open_type in opened)
    }
    return [
        {
            "type": close_type,
            "tabId": viewer_tab_id,
            "mirrorOf": owner_tab_id,
            "success": False,
            "message": "The window that opened this closed it.",
        }
        for viewer_tab_id in mirror.viewer_tab_ids
        for close_type in sorted(closers)
    ]


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
        self._recordings: dict[str, list[dict[str, Any]]] = {}
        self._persist_agents: dict[str, Any] = {}
        self._subscribers: dict[str, set[str]] = {}
        self._subscriber_expiry: dict[str, float] = {}
        # Tabs whose picker currently shows a running agent's model
        # instead of their user's pick, and the model each running task
        # switched itself to (see broadcast_agent_model_pick).
        self._model_override_tabs: set[str] = set()
        self._task_model_override: dict[str, str] = {}
        # Interactive UIs (merge review, auto-commit prompt, worktree
        # strip) owned by one tab but mirrored onto the tabs of every
        # other client viewing the same task.  Keyed by owner tab id.
        self._ui_mirrors: dict[str, _UiMirror] = {}

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
            pending_ui = self._join_ui_mirrors(key, tab_id)
        if catch_up:
            self.broadcast_model_pick(catch_up, "agent", tab_id)
        for event in pending_ui:
            self.broadcast(event)

    def _join_ui_mirrors(
        self, task_key: str, tab_id: str,
    ) -> list[dict[str, Any]]:
        """Show *tab_id* the interactive UIs already open on *task_key*.

        A client that opens a chat while another client is mid-review
        (or sitting on an unanswered auto-commit prompt) has to be
        caught up, or the question stays invisible in that window until
        somebody else answers it.  Must be called with :attr:`_lock`
        held; the returned events are broadcast by the caller once the
        lock is released.

        Args:
            task_key: The coerced task id *tab_id* just subscribed to.
            tab_id: The joining tab.

        Returns:
            Copies of the open UI events, stamped for *tab_id*.
        """
        pending: list[dict[str, Any]] = []
        for owner_tab_id, mirror in self._ui_mirrors.items():
            if mirror.task_key != task_key or owner_tab_id == tab_id:
                continue
            if tab_id not in mirror.viewer_tab_ids:
                mirror.viewer_tab_ids.append(tab_id)
            pending.extend(
                {**event, "tabId": tab_id, "mirrorOf": owner_tab_id}
                for event in mirror.open_events.values()
            )
        return pending

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

    def open_ui_mirror(
        self,
        owner_tab_id: str,
        viewer_tab_ids: list[str],
        work_dir: str = "",
        task_key: str = "",
    ) -> None:
        """Mirror *owner_tab_id*'s interactive UI onto other clients' tabs.

        Called whenever a blocking UI opens (merge review, auto-commit
        prompt, worktree strip).  The viewer set is a snapshot rather
        than a live query of :attr:`_subscribers`, because a review the
        user leaves open for a while outlives the task's subscriber
        set (``cleanup_task`` expires it a few minutes after the task
        ends) — yet the closing event must still reach the very tabs
        that were shown the UI.  Re-opening refreshes the snapshot
        while keeping whatever is already on screen.

        Args:
            owner_tab_id: The tab that owns the UI and its on-disk
                state.
            viewer_tab_ids: Tabs of other clients viewing the same
                task.  Duplicates and the owner itself are ignored.
            work_dir: The owner's working directory.
            task_key: The task the UI belongs to, used to catch up a
                tab that joins the task while the UI is open.
        """
        if not owner_tab_id:
            return
        with self._lock:
            mirror = self._ui_mirrors.get(owner_tab_id)
            if mirror is None:
                mirror = _UiMirror(work_dir, task_key)
                self._ui_mirrors[owner_tab_id] = mirror
            else:
                mirror.work_dir = work_dir
                mirror.task_key = task_key
            # Tabs already shown an earlier phase are KEPT: the
            # subscriber set expires a few minutes after the task ends,
            # so re-reading it late would silently drop the very tabs
            # still displaying the review this phase follows.  Closed
            # tabs are removed by cleanup_tab, so nothing accumulates.
            for tab_id in viewer_tab_ids:
                if (
                    tab_id
                    and tab_id != owner_tab_id
                    and tab_id not in mirror.viewer_tab_ids
                ):
                    mirror.viewer_tab_ids.append(tab_id)

    def close_ui_mirror(self, owner_tab_id: str) -> None:
        """Drop *owner_tab_id*'s mirror without notifying anyone.

        For the case where opening the UI failed and the owner has
        already rolled its own state back, so there is nothing on any
        screen to take down.

        Args:
            owner_tab_id: The tab whose UI never opened.
        """
        with self._lock:
            self._ui_mirrors.pop(owner_tab_id, None)

    def ui_mirror_owner(self, tab_id: str, open_event: str = "") -> str:
        """Return the tab owning the interactive UI shown on *tab_id*.

        An action (accept a hunk, commit, discard a worktree) carries
        the tab id of the client that produced it.  When that client is
        only mirroring someone else's UI, the action has to be applied
        to the owner — the tab that holds the merge cursor, the
        on-disk merge artifacts and the repository.

        Args:
            tab_id: The tab id carried by the inbound command.
            open_event: The event type that put the acted-on UI on
                screen (e.g. ``"merge_data"``).  A tab watching two
                chats can mirror two owners at once, so the action is
                matched to the owner actually showing that UI.

        Returns:
            The owner tab id, or *tab_id* itself when it owns its UI
            (or no mirror is registered for it).
        """
        if not tab_id:
            return tab_id
        with self._lock:
            if tab_id in self._ui_mirrors:
                return tab_id
            for owner_tab_id, mirror in self._ui_mirrors.items():
                if tab_id not in mirror.viewer_tab_ids:
                    continue
                if open_event and open_event not in mirror.open_events:
                    continue
                return owner_tab_id
        return tab_id

    def ui_mirror_work_dir(self, owner_tab_id: str) -> str:
        """Return the working directory recorded for *owner_tab_id*'s UI.

        Args:
            owner_tab_id: The tab that owns the UI.

        Returns:
            The owner's working directory, or ``""`` when unknown.
        """
        with self._lock:
            mirror = self._ui_mirrors.get(owner_tab_id)
        return mirror.work_dir if mirror is not None else ""

    def ui_mirror_tabs(self, owner_tab_id: str) -> list[str]:
        """Return every tab that renders *owner_tab_id*'s UI, owner first.

        Args:
            owner_tab_id: The tab that owns the UI.

        Returns:
            ``[owner, *viewers]``.  Just ``[owner]`` when nothing
            mirrors it.
        """
        with self._lock:
            mirror = self._ui_mirrors.get(owner_tab_id)
            viewers = list(mirror.viewer_tab_ids) if mirror is not None else []
        return [owner_tab_id, *viewers]

    def broadcast_tab_ui(self, event: dict[str, Any]) -> None:
        """Broadcast a tab-scoped UI *event* to every mirroring tab.

        Tab-stamped events are routed verbatim to all clients, which
        each keep only the copy naming a tab they know.  Tab ids are
        per-client, so one stamped copy is emitted per mirroring tab;
        copies for viewers also carry ``mirrorOf`` naming the owner, so
        the transports can tell an original from its mirror.

        Args:
            event: A UI event carrying an owner ``tabId``.
        """
        owner_tab_id = event.get("tabId", "")
        if not owner_tab_id:
            self.broadcast(event)
            return
        targets = self._track_ui_event(owner_tab_id, event)
        for target in targets:
            copy = {**event, "tabId": target}
            if target != owner_tab_id:
                copy["mirrorOf"] = owner_tab_id
            self.broadcast(copy)

    def _track_ui_event(
        self, owner_tab_id: str, event: dict[str, Any],
    ) -> list[str]:
        """Record what *owner_tab_id* has on screen and return the targets.

        An opening event replaces the previous one of its kind; a
        closing event retires the events it answers.  Once nothing is
        on screen the mirror is dropped, so a tab that merely watched
        an old review can never be mistaken for a viewer of a new one.

        The state change and the target snapshot happen under one hold
        of the lock: a tab that joins mid-way must either be caught up
        by :meth:`_join_ui_mirrors` or be in this event's target list —
        splitting the two would let it fall between them and be shown
        an opening it is never told to close (or a close for an opening
        it never saw).

        Args:
            owner_tab_id: The tab that owns the UI.
            event: The UI event being broadcast.

        Returns:
            The tabs to send this event to, owner first.
        """
        event_type = event.get("type", "")
        with self._lock:
            mirror = self._ui_mirrors.get(owner_tab_id)
            if mirror is None:
                return [owner_tab_id]
            targets = [owner_tab_id, *mirror.viewer_tab_ids]
            if event_type in _UI_OPEN_EVENTS:
                mirror.open_events[event_type] = event
            answered = _UI_CLOSE_EVENTS.get(event_type, ())
            for answered_type in answered:
                mirror.open_events.pop(answered_type, None)
            if answered and not mirror.open_events:
                del self._ui_mirrors[owner_tab_id]
        return targets

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

    def broadcast_agent_model_pick(self, model: str, tab_id: str) -> None:
        """Show a running agent's *model* in every tab watching its task.

        The launching tab plus every viewer subscribed to the agent's
        task (history-resume tabs, chat viewers) get the override, so
        each window watching the agent sees what it is actually
        running.  Every other tab keeps showing its own user's pick.

        Each target is remembered so the picker can be handed back
        when the task ends — and only then, which is why a task whose
        agent never switched models costs nothing.

        Args:
            model: The model the agent just switched to.
            tab_id: The tab the agent's task was launched in (``""``
                when the agent runs outside a tab, e.g. from the CLI).
        """
        if not model:
            return
        task_key = self._task_key()
        targets = set(self._fanout_targets(task_key))
        if tab_id:
            targets.add(tab_id)
        with self._lock:
            self._model_override_tabs |= targets
            if task_key:
                self._task_model_override[task_key] = model
        for target in sorted(targets):
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

        Looks up the agent registered for ``event["taskId"]`` and, when
        present with a non-None ``_last_task_id``, enqueues the event
        for asynchronous persistence via ``_queue_chat_event``.

        Args:
            event: The event dictionary (must already have ``taskId``
                injected when applicable).
        """
        if event.get("type") not in _DISPLAY_EVENT_TYPES:
            return
        key = self._coerce_task_id(event.get("taskId"))
        if not key:
            return
        with self._lock:
            agent = self._persist_agents.get(key)
        if agent is None:
            return
        task_id = getattr(agent, "_last_task_id", None)
        if task_id is not None:
            _queue_chat_event(event, task_id=task_id)

    @property
    def tokens_offset(self) -> int:
        """Per-task token-count offset used when broadcasting ``usage_info``.

        Backed by a ``task_id``-keyed dict so concurrent tasks never
        clobber each other's accumulated tokens.
        """
        return self._tokens_offsets.get(self._task_key(), 0)

    @tokens_offset.setter
    def tokens_offset(self, value: int) -> None:
        self._tokens_offsets[self._task_key()] = value

    @property
    def budget_offset(self) -> float:
        """Per-task dollar-budget offset used when broadcasting ``usage_info``."""
        return self._budget_offsets.get(self._task_key(), 0.0)

    @budget_offset.setter
    def budget_offset(self, value: float) -> None:
        self._budget_offsets[self._task_key()] = value

    @property
    def steps_offset(self) -> int:
        """Per-task step-count offset used when broadcasting ``usage_info``."""
        return self._steps_offsets.get(self._task_key(), 0)

    @steps_offset.setter
    def steps_offset(self, value: int) -> None:
        self._steps_offsets[self._task_key()] = value

    def cleanup_tab(self, tab_id: str) -> None:
        """Remove *tab_id* from every subscriber, override and mirror set.

        Should be called when a frontend tab is closed.  The
        underlying per-task state (recording, bash buffer, offsets)
        is NOT touched here: those belong to the task, not the tab,
        and survive a tab close so a freshly-opened tab on the same
        task can still pick up the running stream.  Call
        :meth:`cleanup_task` to drop the per-task state when the task
        itself ends.

        A UI the tab OWNS is deliberately left alone — this also runs
        when a tab merely re-subscribes (session replay, new chat), and
        a review open in other windows must survive that.  Disposal of
        the tab itself goes through :meth:`close_owner_ui`.

        Args:
            tab_id: The frontend tab identifier to drop.
        """
        if not tab_id:
            return
        with self._lock:
            self._model_override_tabs.discard(tab_id)
            for mirror in self._ui_mirrors.values():
                if tab_id in mirror.viewer_tab_ids:
                    mirror.viewer_tab_ids.remove(tab_id)
            self._sweep_expired_subscribers()
            for task_key in list(self._subscribers.keys()):
                viewers = self._subscribers[task_key]
                viewers.discard(tab_id)
                if not viewers:
                    self._subscribers.pop(task_key, None)
                    self._subscriber_expiry.pop(task_key, None)

    def close_owner_ui(self, owner_tab_id: str) -> None:
        """Take *owner_tab_id*'s interactive UI off the other clients.

        Called when the owner tab is disposed for good.  The tab that
        held the merge cursor, the on-disk artifacts and the repository
        is gone, so a button left on the other screens could only act
        on nothing — or, once the mirror is forgotten, on the wrong
        repository.

        Args:
            owner_tab_id: The tab being disposed.
        """
        with self._lock:
            orphaned = self._ui_mirrors.pop(owner_tab_id, None)
        if orphaned is None:
            return
        for event in _orphaned_ui_close_events(owner_tab_id, orphaned):
            self.broadcast(event)

    def has_ui_mirror_for_task(self, task_id: Any) -> bool:
        """Return whether a UI of *task_id* is still waiting to be answered.

        Args:
            task_id: The task identifier to look for.

        Returns:
            True when some tab owns an open UI belonging to that task.
        """
        key = self._coerce_task_id(task_id)
        if not key:
            return False
        with self._lock:
            return any(
                mirror.task_key == key and mirror.open_events
                for mirror in self._ui_mirrors.values()
            )

    def cleanup_task(
        self, task_id: Any, subscriber_linger_seconds: float = 300.0,
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
        so any post-task broadcasts (e.g. the async
        ``followup_suggestion``) still fan out to the originating tab.
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
            self._task_model_override.pop(key, None)
            self._tokens_offsets.pop(key, None)
            self._budget_offsets.pop(key, None)
            self._steps_offsets.pop(key, None)
            self._persist_agents.pop(key, None)
            if key in self._subscribers:
                if subscriber_linger_seconds <= 0:
                    self._subscribers.pop(key, None)
                    self._subscriber_expiry.pop(key, None)
                else:
                    self._subscriber_expiry[key] = (
                        time.monotonic() + subscriber_linger_seconds
                    )
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
            bs = self._bash_state
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
        key = self._task_key()
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
            event.get("taskId")
            or getattr(self._thread_local, "task_id", None),
        )
        if not key:
            return
        rec = self._recordings.get(key)
        if rec is not None:
            rec.append(event)

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
                        0.1, partial(self._timer_flush_for_task, owner_task),
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
            self.broadcast({
                "type": "usage_info",
                "text": str(content),
                "total_tokens": total_tokens,
                "cost": total_cost,
                "total_steps": total_steps,
            })
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
            shared_input = (
                kwargs.get("tool_input") if len(blocks) == 1 else None
            )
            for block in blocks:
                self._emit_tool_result(
                    block.content,
                    tool_name=(
                        getattr(block, "tool_name", "")
                        or kwargs.get("tool_name", "")
                    ),
                    is_error=bool(block.is_error),
                    tool_input=shared_input,
                )
