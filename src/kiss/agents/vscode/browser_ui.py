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
    "task_done", "task_error", "task_stopped",
    "followup_suggestion",
    "autocommit_done",
})


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

    __slots__ = ("buffer", "timer", "generation", "last_flush", "streamed")

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.timer: threading.Timer | None = None
        self.generation: int = 0
        self.last_flush: float = 0.0
        self.streamed: bool = False


class BaseBrowserPrinter(Printer):
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
    closes) or :meth:`unsubscribe_tab`.
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
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._bash_lock = threading.Lock()
        # All per-task maps are keyed by ``str(task_id)``.  The empty
        # string ``""`` is used as a fallback key for events that
        # happen on a thread with no thread-local ``task_id`` set
        # (e.g. very early task lifecycle, or unit tests).
        self._bash_states: dict[str, _BashState] = {}
        self._tokens_offsets: dict[str, int] = {}
        self._budget_offsets: dict[str, float] = {}
        self._steps_offsets: dict[str, int] = {}
        self._recordings: dict[str, list[dict[str, Any]]] = {}
        self._persist_agents: dict[str, Any] = {}
        # ``_subscribers`` maps a task_id (str) to the set of frontend
        # tab ids that should receive every broadcast for that task.
        # The agent thread emits events tagged with ``taskId`` only;
        # the transport layer (e.g. ``WebPrinter.broadcast``) fans
        # each event out to every subscriber tab, stamping the
        # tab-specific ``tabId`` on each fan-out copy.  The single
        # recording per task is shared by all subscribed tabs.
        self._subscribers: dict[str, set[str]] = {}

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
            viewers = self._subscribers.get(key)
            if viewers is None:
                viewers = set()
                self._subscribers[key] = viewers
            viewers.add(tab_id)

    def unsubscribe_tab(self, task_id: Any, tab_id: str) -> None:
        """Remove *tab_id* from the subscriber set for *task_id*.

        Args:
            task_id: The task identifier (``task_history.id`` int or
                its string form).
            tab_id: The frontend tab id to unsubscribe.
        """
        key = self._coerce_task_id(task_id)
        if not key or not tab_id:
            return
        with self._lock:
            viewers = self._subscribers.get(key)
            if viewers is None:
                return
            viewers.discard(tab_id)
            if not viewers:
                self._subscribers.pop(key, None)

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
            viewers = self._subscribers.get(key)
            if not viewers:
                return []
            return list(viewers)

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
        # Look up the registered agent under ``_lock`` so a concurrent
        # ``cleanup_task`` (which pops from ``_persist_agents`` under
        # the same lock) cannot remove the entry between our ``get``
        # and our use of the returned agent.  Holding the lock also
        # serialises us against ``ChatSorcarAgent.run`` registering
        # a fresh agent under the same key.
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
        """Remove *tab_id* from every subscriber set.

        Should be called when a frontend tab is closed.  The
        underlying per-task state (recording, bash buffer, offsets)
        is NOT touched here: those belong to the task, not the tab,
        and survive a tab close so a freshly-opened tab on the same
        task can still pick up the running stream.  Call
        :meth:`cleanup_task` to drop the per-task state when the task
        itself ends.

        Args:
            tab_id: The frontend tab identifier to drop.
        """
        if not tab_id:
            return
        with self._lock:
            for task_key in list(self._subscribers.keys()):
                viewers = self._subscribers[task_key]
                viewers.discard(tab_id)
                if not viewers:
                    self._subscribers.pop(task_key, None)

    def cleanup_task(self, task_id: Any) -> None:
        """Remove all per-task state for *task_id* to free memory.

        Called by the task-runner once a task has fully terminated.
        Cancels any pending bash flush timer and drops the per-task
        recording, persist-agent, and usage-offset entries.  The
        subscriber set is **intentionally preserved** so any post-
        task broadcasts (e.g. the async ``followup_suggestion``) still
        fan out to the originating tab; subscriber cleanup happens
        when the frontend tab itself closes via :meth:`cleanup_tab`.

        Args:
            task_id: The task identifier whose state should be freed.
        """
        key = self._coerce_task_id(task_id)
        if not key:
            return
        with self._bash_lock:
            bs = self._bash_states.pop(key, None)
            if bs is not None and bs.timer is not None:
                bs.timer.cancel()
        with self._lock:
            self._recordings.pop(key, None)
            self._tokens_offsets.pop(key, None)
            self._budget_offsets.pop(key, None)
            self._steps_offsets.pop(key, None)
            self._persist_agents.pop(key, None)

    def reset(self) -> None:
        """Reset internal streaming state for a new turn."""
        self._current_block_type = ""
        with self._bash_lock:
            self._bash_state.generation += 1
            self._bash_state.buffer.clear()
            self._bash_state.streamed = False
            if self._bash_state.timer is not None:
                self._bash_state.timer.cancel()
                self._bash_state.timer = None

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
        generation inside a second ``_bash_lock`` acquisition: if
        ``reset()`` ran in between (incrementing the generation), the
        captured text is stale and is discarded.  The ``broadcast()``
        call is made while still holding the second lock to close the
        TOCTOU window that would otherwise allow ``reset()`` +
        ``start_recording()`` to slip in between the generation check
        and the broadcast.
        """
        with self._bash_lock:
            bs = self._bash_state
            gen = bs.generation
            if bs.timer is not None:
                bs.timer.cancel()
                bs.timer = None
            text = "".join(bs.buffer) if bs.buffer else ""
            bs.buffer.clear()
            bs.last_flush = time.monotonic()
        if text:
            with self._bash_lock:
                if self._bash_state.generation != gen:
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

        Args:
            event: The event dictionary to broadcast.
        """
        event = self._inject_task_id(event)
        with self._lock:
            self._record_event(event)
        self._persist_event(event)

    def _broadcast_result(
        self,
        text: str,
        total_tokens: int = 0,
        cost: str = "N/A",
        step_count: int = 0,
    ) -> None:
        # Apply per-task offsets so sub-agent cost / tokens / steps that
        # were accumulated into the printer (e.g. by ``run_parallel``)
        # are included in the final result panel.  Otherwise the parent
        # agent's displayed cost would be smaller than the sum of its
        # sub-agents' costs.  Matches the offset arithmetic in the
        # ``usage_info`` branch of :meth:`WebPrinter.print`.
        if isinstance(cost, str) and cost.startswith("$"):
            try:
                cost = f"${float(cost[1:]) + self.budget_offset:.4f}"
            except ValueError:
                pass
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
            with self._bash_lock:
                bs = self._bash_state
                bs.buffer.append(str(content))
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
                self.broadcast({"type": "system_output", "text": text})
            with self._bash_lock:
                self._bash_state.streamed = True
            return ""
        if type == "tool_call":
            self._flush_bash()
            with self._bash_lock:
                self._bash_state.streamed = False
            self.broadcast({"type": "text_end"})
            self._format_tool_call(str(content), kwargs.get("tool_input", {}))
            return ""
        if type == "tool_result":
            self._flush_bash()
            tool_name = kwargs.get("tool_name", "")
            # Show every tool's return value (so the user sees the output
            # of run_parallel, ask_user_question, update_settings, the
            # WebUseTool methods, etc.) EXCEPT ``finish`` -- the agentic
            # loop renders that one as a dedicated "result" panel right
            # after, so a tool_result here would just be a duplicate.
            show_result = tool_name != "finish"
            with self._bash_lock:
                streamed = self._bash_state.streamed
                self._bash_state.streamed = False
            result_content = "" if streamed else truncate_result(str(content))
            if show_result:
                self.broadcast(
                    {
                        "type": "tool_result",
                        "content": result_content,
                        "is_error": kwargs.get("is_error", False),
                    }
                )
            return ""
        if type == "usage_info":
            raw_tokens = kwargs.get("total_tokens", 0)
            raw_cost = kwargs.get("cost", "N/A")
            raw_steps = kwargs.get("total_steps", 0)
            total_tokens = raw_tokens + self.tokens_offset
            total_steps = raw_steps + self.steps_offset
            if isinstance(raw_cost, str) and raw_cost.startswith("$"):
                total_cost = f"${float(raw_cost[1:]) + self.budget_offset:.4f}"
            else:
                total_cost = raw_cost
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
            for block in message.content:
                if hasattr(block, "is_error") and hasattr(block, "content"):
                    self.broadcast(
                        {
                            "type": "tool_result",
                            "content": truncate_result(str(block.content)),
                            "is_error": bool(block.is_error),
                        }
                    )
