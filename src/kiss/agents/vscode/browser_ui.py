"""Shared browser UI components for KISS agent viewers."""

import threading
import time
from functools import partial
from typing import Any

from kiss.agents.sorcar.persistence import _append_chat_event
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
    """Base printer for browser-based UIs.

    The current block type (``_current_block_type``) is stored in
    thread-local storage so concurrent task threads can each route
    their streamed tokens to the correct (thinking vs text) panel
    without corrupting each other.  Recording and bash buffering are
    per-tab (keyed by ``tab_id``) so one task's ``stop_recording()``
    or ``reset()`` does not destroy another tab's state.
    """


    @property
    def _current_block_type(self) -> str:
        return getattr(self._thread_local, "_cbt", "")

    @_current_block_type.setter
    def _current_block_type(self, value: str) -> None:
        self._thread_local._cbt = value

    @property
    def _bash_state(self) -> _BashState:
        """Return the bash buffering state for the current thread's tab.

        Each tab gets its own ``_BashState`` so concurrent tasks on
        different tabs cannot corrupt each other's bash buffer,
        ``streamed`` flag, generation counter, or flush timer.

        The caller should hold ``_bash_lock`` when accessing this in
        multi-threaded production code.
        """
        key = getattr(self._thread_local, "tab_id", None) or ""
        bs = self._bash_states.get(key)
        if bs is None:
            bs = _BashState()
            self._bash_states[key] = bs
        return bs

    def __init__(self) -> None:
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._bash_lock = threading.Lock()
        self._bash_states: dict[str, _BashState] = {}
        self._tokens_offsets: dict[str, int] = {}
        self._budget_offsets: dict[str, float] = {}
        self._steps_offsets: dict[str, int] = {}
        self._recordings: dict[str, list[dict[str, Any]]] = {}
        self._persist_agents: dict[str, Any] = {}

    def _tab_key(self) -> str:
        """Return the thread-local tab key for per-tab state lookups.

        Used for per-tab usage offsets, recordings, and bash state.
        Falls back to the empty string for threads without a tab_id
        (e.g. unit tests that do not set ``_thread_local.tab_id``).
        """
        return getattr(self._thread_local, "tab_id", None) or ""

    def _inject_tab_id(self, event: dict[str, Any]) -> dict[str, Any]:
        """Return *event* with ``tabId`` injected from thread-local storage.

        If the current thread has a ``tab_id`` set and the event does
        not already contain a ``tabId`` key, returns a shallow copy of
        *event* with ``tabId`` added.  Otherwise returns *event* as-is.

        Args:
            event: The event dictionary.

        Returns:
            The (possibly augmented) event dictionary.
        """
        tab_id = getattr(self._thread_local, "tab_id", None)
        if tab_id is not None and "tabId" not in event:
            return {**event, "tabId": tab_id}
        return event

    def _persist_event(self, event: dict[str, Any]) -> None:
        """Persist a display event to the database if applicable.

        Checks whether *event* is a display event type, looks up the
        per-tab agent from ``_persist_agents``, and appends the event
        to the database via ``_append_chat_event`` when a valid
        ``_last_task_id`` is present.

        Args:
            event: The event dictionary (must already have ``tabId``
                injected).
        """
        if event.get("type") not in _DISPLAY_EVENT_TYPES:
            return
        evt_tab = event.get("tabId")
        if evt_tab is None:
            return
        agent = self._persist_agents.get(evt_tab)
        if agent is None:
            return
        task_id = agent._last_task_id
        if task_id is not None:
            _append_chat_event(event, task_id=task_id)

    @property
    def tokens_offset(self) -> int:
        """Per-tab token-count offset used when broadcasting ``usage_info``.

        Backed by a ``tab_id``-keyed dict so concurrent tasks on
        different tabs never clobber each other's accumulated tokens
        (A7 fix).
        """
        return self._tokens_offsets.get(self._tab_key(), 0)

    @tokens_offset.setter
    def tokens_offset(self, value: int) -> None:
        self._tokens_offsets[self._tab_key()] = value

    @property
    def budget_offset(self) -> float:
        """Per-tab dollar-budget offset used when broadcasting ``usage_info``."""
        return self._budget_offsets.get(self._tab_key(), 0.0)

    @budget_offset.setter
    def budget_offset(self, value: float) -> None:
        self._budget_offsets[self._tab_key()] = value

    @property
    def steps_offset(self) -> int:
        """Per-tab step-count offset used when broadcasting ``usage_info``."""
        return self._steps_offsets.get(self._tab_key(), 0)

    @steps_offset.setter
    def steps_offset(self, value: int) -> None:
        self._steps_offsets[self._tab_key()] = value

    def cleanup_tab(self, tab_id: str) -> None:
        """Remove all per-tab state for *tab_id* to free memory.

        Should be called when a tab is closed on the frontend.  Cancels
        any pending bash flush timer and removes the tab's entries from
        ``_bash_states`` and ``_recordings``.

        Args:
            tab_id: The frontend tab identifier to clean up.
        """
        key = tab_id or ""
        with self._bash_lock:
            bs = self._bash_states.pop(key, None)
            if bs is not None and bs.timer is not None:
                bs.timer.cancel()
        with self._lock:
            self._recordings.pop(key, None)
            self._tokens_offsets.pop(key, None)
            self._budget_offsets.pop(key, None)
            self._steps_offsets.pop(key, None)

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

    def _timer_flush_for_tab(self, tab_id: str | None) -> None:
        """Timer callback that sets the thread-local tab_id and flushes bash.

        Used by the bash-stream buffering timer.  Replaces the former
        closure ``_timer_flush`` so that ``self`` is not captured from
        an enclosing scope.

        Args:
            tab_id: The tab identifier that owns the bash buffer, or
                None when no tab context is available.
        """
        if tab_id is not None:
            self._thread_local.tab_id = tab_id
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
        """Start recording broadcast events for the current tab."""
        key = self._tab_key()
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
        """Stop recording for the current tab and return its display events.

        Returns:
            List of display-relevant events with consecutive deltas merged.
        """
        key = self._tab_key()
        with self._lock:
            raw = self._recordings.pop(key, [])
        return self._filter_and_coalesce(raw)

    def peek_recording(self) -> list[dict[str, Any]]:
        """Return a snapshot of the current tab's recording without stopping it.

        Used for periodic crash-recovery flushes: the caller can persist
        a snapshot of events to the database while recording continues.

        Returns:
            List of display-relevant events with consecutive deltas merged.
        """
        key = self._tab_key()
        with self._lock:
            rec = self._recordings.get(key)
            raw = list(rec) if rec is not None else []
        return self._filter_and_coalesce(raw)

    def _record_event(self, event: dict[str, Any]) -> None:
        """Append event to the active recording for the event's tab.

        Looks up the recording list by ``tabId`` from the event (set by
        ``VSCodePrinter.broadcast``), falling back to the thread-local
        ``tab_id``.  This ensures timer-thread broadcasts (bash flush)
        are routed to the correct tab's recording.

        Must be called with ``self._lock`` held.
        """
        key = event.get("tabId") or getattr(self._thread_local, "tab_id", None) or ""
        rec = self._recordings.get(key)
        if rec is not None:
            rec.append(event)

    def broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast an event and record it.

        Args:
            event: The event dictionary to broadcast.
        """
        with self._lock:
            self._record_event(event)

    def _broadcast_result(
        self,
        text: str,
        total_tokens: int = 0,
        cost: str = "N/A",
        step_count: int = 0,
    ) -> None:
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
                    owner_tab = getattr(self._thread_local, "tab_id", None)
                    bs.timer = threading.Timer(
                        0.1, partial(self._timer_flush_for_tab, owner_tab),
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
            core_tools = {"Bash", "Read", "Edit", "Write"}
            show_result = tool_name in core_tools or kwargs.get("is_error", False)
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
