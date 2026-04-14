"""Shared browser UI components for KISS agent viewers."""

import queue
import threading
import time
from typing import Any

from kiss.core.printer import (
    Printer,
    StreamEventParser,
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
    """Per-tab bash buffering state.

    Each chat tab gets its own buffer so concurrent tasks don't
    interleave bash output or tag it with the wrong tabId.
    """

    __slots__ = ("buffer", "timer", "generation", "last_flush", "streamed")

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.timer: threading.Timer | None = None
        self.generation: int = 0
        self.last_flush: float = 0.0
        self.streamed: bool = False


class BaseBrowserPrinter(StreamEventParser, Printer):
    # -- S12 fix: redirect StreamEventParser state to thread-local --------
    # StreamEventParser stores _current_block_type, _tool_name, and
    # _tool_json_buffer as instance attributes.  Since there is ONE printer
    # shared by all task threads, concurrent streaming corrupts these.
    # Properties redirect reads/writes to thread-local storage.

    @property  # type: ignore[override]
    def _current_block_type(self) -> str:  # type: ignore[override]
        """Per-thread block type for stream parsing."""
        return getattr(self._thread_local, "_current_block_type", "")

    @_current_block_type.setter
    def _current_block_type(self, value: str) -> None:
        self._thread_local._current_block_type = value

    @property  # type: ignore[override]
    def _tool_name(self) -> str:  # type: ignore[override]
        """Per-thread tool name being parsed."""
        return getattr(self._thread_local, "_tool_name", "")

    @_tool_name.setter
    def _tool_name(self, value: str) -> None:
        self._thread_local._tool_name = value

    @property  # type: ignore[override]
    def _tool_json_buffer(self) -> str:  # type: ignore[override]
        """Per-thread JSON buffer for tool input being accumulated."""
        return getattr(self._thread_local, "_tool_json_buffer", "")

    @_tool_json_buffer.setter
    def _tool_json_buffer(self, value: str) -> None:
        self._thread_local._tool_json_buffer = value

    # -- S13 fix: redirect token/budget/steps offsets to thread-local -----
    # relentless_agent.py sets these between sub-sessions; with concurrent
    # tasks on different tabs the offsets leak across tabs.

    @property
    def tokens_offset(self) -> int:
        """Per-thread token offset for usage_info events."""
        return getattr(self._thread_local, "tokens_offset", 0)

    @tokens_offset.setter
    def tokens_offset(self, value: int) -> None:
        self._thread_local.tokens_offset = value

    @property
    def budget_offset(self) -> float:
        """Per-thread budget offset for usage_info events."""
        return getattr(self._thread_local, "budget_offset", 0.0)

    @budget_offset.setter
    def budget_offset(self, value: float) -> None:
        self._thread_local.budget_offset = value

    @property
    def steps_offset(self) -> int:
        """Per-thread steps offset for usage_info events."""
        return getattr(self._thread_local, "steps_offset", 0)

    @steps_offset.setter
    def steps_offset(self, value: int) -> None:
        self._thread_local.steps_offset = value

    def __init__(self) -> None:
        # _thread_local MUST be created before StreamEventParser.__init__
        # because the S12 properties redirect attribute writes to it.
        self._thread_local = threading.local()
        StreamEventParser.__init__(self)
        self._client_queue: queue.Queue[dict[str, Any]] | None = None
        self._lock = threading.Lock()
        self._bash_lock = threading.Lock()
        self._bash_states: dict[str, _BashState] = {}
        self.stop_event = threading.Event()
        self._recordings: dict[int, list[dict[str, Any]]] = {}
        self._recording_owners: dict[int, str] = {}

    def _get_bash(self) -> _BashState:
        """Return the bash state for the current thread's tab.

        Creates the state on first access.  Must be called with
        ``_bash_lock`` held.
        """
        tab_id: str = getattr(self._thread_local, "tab_id", "")
        bs = self._bash_states.get(tab_id)
        if bs is None:
            bs = _BashState()
            self._bash_states[tab_id] = bs
        return bs

    def reset(self) -> None:
        """Reset internal streaming and tool-parsing state for a new turn."""
        self.reset_stream_state()
        with self._bash_lock:
            bs = self._get_bash()
            bs.generation += 1
            bs.buffer.clear()
            bs.streamed = False
            if bs.timer is not None:
                bs.timer.cancel()
                bs.timer = None

    def _flush_bash(self) -> None:
        """Flush the bash buffer for the current thread's tab."""
        # RC5 fix: drain entirely inside lock, no generation TOCTOU
        with self._bash_lock:
            bs = self._get_bash()
            if bs.timer is not None:
                bs.timer.cancel()
                bs.timer = None
            text = "".join(bs.buffer) if bs.buffer else ""
            bs.buffer.clear()
            bs.last_flush = time.monotonic()
        if text:
            self.broadcast({"type": "system_output", "text": text})

    def start_recording(
        self, recording_id: int | None = None, tab_id: str | None = None
    ) -> None:
        """Start recording broadcast events.

        Uses an explicit *recording_id* to avoid thread-ID reuse corruption.
        Falls back to thread ident when no ID is given (backward compat).

        When *tab_id* is provided, only events whose ``tabId`` matches
        are recorded.  Events without a ``tabId`` are still recorded to
        all active recordings.

        Args:
            recording_id: Unique identifier for this recording session.
            tab_id: Optional tab owner — restricts which events are recorded.
        """
        key = recording_id if recording_id is not None else threading.current_thread().ident
        with self._lock:
            if key is not None:  # pragma: no branch – always set for alive threads
                self._recordings[key] = []
                if tab_id is not None:
                    self._recording_owners[key] = tab_id

    def stop_recording(self, recording_id: int | None = None) -> list[dict[str, Any]]:
        """Stop recording and return its display events.

        Args:
            recording_id: The recording ID passed to start_recording.

        Returns:
            List of display-relevant events with consecutive deltas merged.
        """
        key = recording_id if recording_id is not None else threading.current_thread().ident
        assert key is not None
        with self._lock:
            raw = self._recordings.pop(key, [])
            self._recording_owners.pop(key, None)
        filtered = [e for e in raw if e.get("type") in _DISPLAY_EVENT_TYPES]
        return _coalesce_events(filtered)

    def peek_recording(self, recording_id: int) -> list[dict[str, Any]]:
        """Return a snapshot of the current recording without stopping it.

        Used for periodic crash-recovery flushes: the caller can persist
        a snapshot of events to the database while recording continues.

        Args:
            recording_id: The recording ID passed to start_recording.

        Returns:
            List of display-relevant events with consecutive deltas merged.
        """
        with self._lock:
            raw = list(self._recordings.get(recording_id, []))
        filtered = [e for e in raw if e.get("type") in _DISPLAY_EVENT_TYPES]
        return _coalesce_events(filtered)

    def _record_event(self, event: dict[str, Any]) -> None:
        """Append event to matching active recordings.

        When a recording has an owner ``tab_id``, only events whose
        ``tabId`` matches the owner are appended.  Events without a
        ``tabId`` (global events like models/history responses) are
        skipped for owned recordings since they belong to no specific
        tab.  Recordings without an owner receive all events.

        Must be called with ``self._lock`` held.
        """
        ev_tab = event.get("tabId")
        for key, events_list in self._recordings.items():
            owner = self._recording_owners.get(key)
            if owner is not None and ev_tab != owner:
                continue
            events_list.append(event)

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send an SSE event dict to the connected client.

        The event is also appended to every active per-thread recording.

        Args:
            event: The event dictionary to broadcast.
        """
        with self._lock:
            self._record_event(event)
            if self._client_queue is not None:
                self._client_queue.put(event)

    def add_client(self) -> queue.Queue[dict[str, Any]]:
        """Register the SSE client and return its event queue.

        Only one client is supported. A new connection replaces any
        previous one.

        Returns:
            queue.Queue[dict[str, Any]]: A queue that will receive broadcast events.
        """
        cq: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._client_queue = cq
        return cq

    def remove_client(self, cq: queue.Queue[dict[str, Any]]) -> None:
        """Unregister the SSE client's event queue.

        Only clears the queue if *cq* is the current client (handles
        reconnection races where the old connection tears down after a
        new one has already connected).

        Args:
            cq: The client queue to remove.
        """
        with self._lock:
            if self._client_queue is cq:
                self._client_queue = None

    def has_clients(self) -> bool:
        """Return True if a client is currently connected."""
        return self._client_queue is not None

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
            event["summary"] = str(parsed["summary"])
        self.broadcast(event)

    def _check_stop(self) -> None:
        ev = getattr(self._thread_local, "stop_event", None) or self.stop_event
        if ev.is_set():
            raise KeyboardInterrupt("Agent stop requested")

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Render content by broadcasting SSE events to connected browser clients.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "stream_event",
                "tool_call", "tool_result", "result", "message").
            **kwargs: Additional options such as tool_input, is_error, cost,
                total_tokens.

        Returns:
            str: Extracted text from stream events, or empty string.
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
        if type == "stream_event":
            return self.parse_stream_event(content)
        if type == "message":
            self._handle_message(content, **kwargs)
            return ""
        if type == "bash_stream":
            text = ""
            with self._bash_lock:
                bs = self._get_bash()
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

                    def _timer_flush(tid: int | None = owner_tab) -> None:
                        if tid is not None:
                            self._thread_local.tab_id = tid
                        self._flush_bash()

                    bs.timer = threading.Timer(0.1, _timer_flush)
                    bs.timer.daemon = True
                    bs.timer.start()
            if text:
                self.broadcast({"type": "system_output", "text": text})
            with self._bash_lock:
                self._get_bash().streamed = True
            return ""
        if type == "tool_call":
            self._flush_bash()
            with self._bash_lock:
                self._get_bash().streamed = False
            self.broadcast({"type": "text_end"})
            self._format_tool_call(str(content), kwargs.get("tool_input", {}))
            return ""
        if type == "tool_result":
            self._flush_bash()
            tool_name = kwargs.get("tool_name", "")
            core_tools = {"Bash", "Read", "Edit", "Write"}
            show_result = tool_name in core_tools or kwargs.get("is_error", False)
            with self._bash_lock:
                bs = self._get_bash()
                streamed = bs.streamed
                bs.streamed = False
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
        """Broadcast a streamed token as an SSE delta event to browser clients.

        Args:
            token: The text token to broadcast.
        """
        self._check_stop()
        if token:
            delta_type = (
                "thinking_delta" if self._current_block_type == "thinking" else "text_delta"
            )
            self.broadcast({"type": delta_type, "text": token})

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

    def _on_thinking_start(self) -> None:
        self.broadcast({"type": "thinking_start"})

    def _on_thinking_end(self) -> None:
        self.broadcast({"type": "thinking_end"})

    def _on_tool_use_end(self, name: str, tool_input: dict) -> None:
        self._format_tool_call(name, tool_input)

    def _on_text_block_end(self) -> None:
        self.broadcast({"type": "text_end"})

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
