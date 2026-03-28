"""Shared browser UI components for KISS agent viewers."""

import logging
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

logger = logging.getLogger(__name__)




_DISPLAY_EVENT_TYPES = frozenset({
    "clear", "thinking_start", "thinking_delta", "thinking_end",
    "text_delta", "text_end", "tool_call", "tool_result",
    "system_output", "result", "system_prompt", "prompt", "usage_info",
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


class BaseBrowserPrinter(StreamEventParser, Printer):
    def __init__(self) -> None:
        StreamEventParser.__init__(self)
        self._client_queue: queue.Queue[dict[str, Any]] | None = None
        self._lock = threading.Lock()
        self._bash_lock = threading.Lock()
        self._bash_buffer: list[str] = []
        self._bash_last_flush = 0.0
        self._bash_flush_timer: threading.Timer | None = None
        self._bash_streamed = False
        self.stop_event = threading.Event()
        self._thread_local = threading.local()
        self._recordings: dict[int, list[dict[str, Any]]] = {}

    def reset(self) -> None:
        """Reset internal streaming and tool-parsing state for a new turn."""
        self.reset_stream_state()
        self._bash_streamed = False
        with self._bash_lock:
            self._bash_buffer.clear()
            if self._bash_flush_timer is not None:
                self._bash_flush_timer.cancel()
                self._bash_flush_timer = None

    def _flush_bash(self) -> None:
        with self._bash_lock:
            if self._bash_flush_timer is not None:
                self._bash_flush_timer.cancel()
                self._bash_flush_timer = None
            if self._bash_buffer:
                text = "".join(self._bash_buffer)
                self._bash_buffer.clear()
                self._bash_last_flush = time.monotonic()
            else:
                text = ""
        if text:
            self.broadcast({"type": "system_output", "text": text})

    def start_recording(self) -> None:
        """Start recording broadcast events for the calling thread.

        Each thread gets its own independent recording buffer, so concurrent
        agent threads do not interfere with each other's recordings.
        """
        tid = threading.current_thread().ident
        with self._lock:
            if tid is not None:  # pragma: no branch – always set for alive threads
                self._recordings[tid] = []

    def stop_recording(self) -> list[dict[str, Any]]:
        """Stop recording for the calling thread and return its display events.

        Returns:
            List of display-relevant events with consecutive deltas merged.
        """
        tid = threading.current_thread().ident
        assert tid is not None
        with self._lock:
            raw = self._recordings.pop(tid, [])
        filtered = [e for e in raw if e.get("type") in _DISPLAY_EVENT_TYPES]
        return _coalesce_events(filtered)

    def _record_event(self, event: dict[str, Any]) -> None:
        """Append event to all active per-thread recordings.

        Must be called with ``self._lock`` held.
        """
        for events_list in self._recordings.values():
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
        with self._lock:
            return self._client_queue is not None

    def _broadcast_result(
        self,
        text: str,
        total_tokens: int = 0,
        cost: str = "N/A",
    ) -> None:
        event: dict[str, Any] = {
            "type": "result",
            "text": text or "(no result)",
            "total_tokens": total_tokens,
            "cost": cost,
        }
        parsed = parse_result_yaml(text) if text else None
        if parsed:
            event["success"] = parsed.get("success")
            event["summary"] = str(parsed["summary"])
        self.broadcast(event)

    def _check_stop(self) -> None:
        ev = getattr(self._thread_local, "stop_event", None)
        if ev is not None:
            if ev.is_set():
                raise KeyboardInterrupt("Agent stop requested")
        elif self.stop_event.is_set():
            raise KeyboardInterrupt("Agent stop requested")

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Render content by broadcasting SSE events to connected browser clients.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "stream_event",
                "tool_call", "tool_result", "result", "usage_info", "message").
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
        if type == "usage_info":
            self.broadcast({"type": "usage_info", "text": str(content).strip()})
            return ""
        if type == "bash_stream":
            text = ""
            with self._bash_lock:
                self._bash_buffer.append(str(content))
                if time.monotonic() - self._bash_last_flush >= 0.1:
                    if self._bash_flush_timer is not None:
                        self._bash_flush_timer.cancel()
                        self._bash_flush_timer = None
                    text = "".join(self._bash_buffer)
                    self._bash_buffer.clear()
                    self._bash_last_flush = time.monotonic()
                elif self._bash_flush_timer is None:
                    self._bash_flush_timer = threading.Timer(0.1, self._flush_bash)
                    self._bash_flush_timer.daemon = True
                    self._bash_flush_timer.start()
            if text:
                self.broadcast({"type": "system_output", "text": text})
            self._bash_streamed = True
            return ""
        if type == "tool_call":
            self._flush_bash()
            self._bash_streamed = False
            self.broadcast({"type": "text_end"})
            self._format_tool_call(str(content), kwargs.get("tool_input", {}))
            return ""
        if type == "tool_result":
            self._flush_bash()
            tool_name = kwargs.get("tool_name", "")
            core_tools = {"Bash", "Read", "Edit", "Write"}
            show_result = tool_name in core_tools or kwargs.get("is_error", False)
            result_content = "" if self._bash_streamed else truncate_result(str(content))
            self._bash_streamed = False
            if show_result:
                self.broadcast(
                    {
                        "type": "tool_result",
                        "content": result_content,
                        "is_error": kwargs.get("is_error", False),
                    }
                )
            return ""
        if type == "result":
            self.broadcast({"type": "text_end"})
            self._broadcast_result(
                str(content),
                kwargs.get("total_tokens", 0),
                kwargs.get("cost", "N/A"),
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
