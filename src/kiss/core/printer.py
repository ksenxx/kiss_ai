"""Abstract base class and shared utilities for KISS agent printers."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

_logger = logging.getLogger(__name__)


def parse_result_yaml(raw: str) -> dict[str, Any] | None:
    """Parse a YAML result string and return the dict if it has a 'summary' key.

    Used by both console and browser printers to extract structured result data
    from agent finish() output.

    Args:
        raw: Raw result string, potentially YAML-formatted.

    Returns:
        The parsed dict if valid YAML with a 'summary' key, else None.
    """
    try:
        data = yaml.safe_load(raw)
    except Exception:
        _logger.debug("Exception caught", exc_info=True)
        return None
    if isinstance(data, dict) and "summary" in data:
        return data
    return None


LANG_MAP = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "rb": "ruby",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "h": "c",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "xml": "xml",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "md": "markdown",
}

MAX_RESULT_LEN = 3000

KNOWN_KEYS = {"file_path", "path", "content", "command", "old_string", "new_string", "description"}


def lang_for_path(path: str) -> str:
    """Map a file path to its syntax-highlighting language name.

    Args:
        path: File path whose extension determines the language.

    Returns:
        str: Language name (e.g. "python", "javascript"), or the raw extension,
            or "text" if no extension is present.
    """
    ext = Path(path).suffix.lstrip(".")
    return LANG_MAP.get(ext, ext or "text")


def truncate_result(content: str) -> str:
    """Truncate long content to MAX_RESULT_LEN, keeping the first and last halves.

    Args:
        content: The string to truncate.

    Returns:
        str: The original string if short enough, otherwise the first and last
            halves joined by a truncation marker.
    """
    if len(content) <= MAX_RESULT_LEN:
        return content
    half = MAX_RESULT_LEN // 2
    return content[:half] + "\n... (truncated) ...\n" + content[-half:]


def extract_path_and_lang(tool_input: dict) -> tuple[str, str]:
    """Extract the file path and inferred language from a tool input dict.

    Args:
        tool_input: Dictionary of tool call arguments, checked for "file_path"
            or "path" keys.

    Returns:
        tuple[str, str]: A (file_path, language) pair. Language defaults to
            "text" if no path is found.
    """
    file_path = str(tool_input.get("file_path") or tool_input.get("path") or "")
    lang = lang_for_path(file_path) if file_path else "text"
    return file_path, lang


def extract_extras(tool_input: dict) -> dict[str, str]:
    """Extract non-standard keys from a tool input dict for display.

    Args:
        tool_input: Dictionary of tool call arguments.

    Returns:
        dict[str, str]: Keys not in KNOWN_KEYS mapped to their string values.
    """
    return {k: str(v) for k, v in tool_input.items() if k not in KNOWN_KEYS}


class StreamEventParser:
    """Shared parser for LLM stream events used by both console and browser printers.

    Tracks block type and tool JSON buffer state, dispatching parsed events
    to subclass callbacks.
    """

    def __init__(self) -> None:
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""

    def reset_stream_state(self) -> None:
        """Reset block type and tool buffer state."""
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""

    def parse_stream_event(self, event: Any) -> str:
        """Parse a stream event, dispatch to on_* callbacks, return extracted text.

        Args:
            event: An event object with an `event` dict attribute.

        Returns:
            str: Any text content extracted from text or thinking deltas.
        """
        evt = event.event
        evt_type = evt.get("type", "")
        text = ""

        if evt_type == "content_block_start":
            block = evt.get("content_block", {})
            block_type = block.get("type", "")
            self._current_block_type = block_type
            if block_type == "thinking":
                self._on_thinking_start()
            elif block_type == "tool_use":
                self._tool_name = block.get("name", "?")
                self._tool_json_buffer = ""
                self._on_tool_use_start(self._tool_name)

        elif evt_type == "content_block_delta":
            delta = evt.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "thinking_delta":
                text = delta.get("thinking", "")
            elif delta_type == "text_delta":
                text = delta.get("text", "")
            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                self._tool_json_buffer += partial
                self._on_tool_json_delta(partial)

        elif evt_type == "content_block_stop":
            block_type = self._current_block_type
            if block_type == "thinking":
                self._on_thinking_end()
            elif block_type == "tool_use":
                try:
                    tool_input = json.loads(self._tool_json_buffer)
                except (json.JSONDecodeError, ValueError):
                    logging.getLogger(__name__).debug("Exception caught", exc_info=True)
                    tool_input = {"_raw": self._tool_json_buffer}
                self._on_tool_use_end(self._tool_name, tool_input)
            else:
                self._on_text_block_end()
            self._current_block_type = ""

        return text

    def _on_thinking_start(self) -> None:
        """Called when a thinking block starts."""

    def _on_thinking_end(self) -> None:
        """Called when a thinking block ends."""

    def _on_tool_use_start(self, name: str) -> None:
        """Called when a tool_use block starts."""

    def _on_tool_json_delta(self, partial: str) -> None:
        """Called for each partial JSON delta in a tool_use block."""

    def _on_tool_use_end(self, name: str, tool_input: dict) -> None:
        """Called when a tool_use block ends with the parsed input."""

    def _on_text_block_end(self) -> None:
        """Called when a non-thinking, non-tool block ends."""


class Printer(ABC):
    @abstractmethod
    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Render content to the output destination.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "stream_event",
                "tool_call", "tool_result", "result", "message").
            **kwargs: Additional type-specific options (e.g. tool_input, is_error).

        Returns:
            str: Any extracted text (e.g. streamed text deltas), or empty string.
        """

    @abstractmethod
    def token_callback(self, token: str) -> None:
        """Handle a single streamed token from the LLM.

        Args:
            token: The text token to process.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the printer's internal streaming state between messages."""


class MultiPrinter(Printer):
    def __init__(self, printers: list[Printer]) -> None:
        self.printers = printers

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Dispatch a print call to all child printers.

        Args:
            content: The content to display.
            type: Content type forwarded to each child printer.
            **kwargs: Additional options forwarded to each child printer.

        Returns:
            str: The first non-empty result from child printers.
        """
        result = ""
        for p in self.printers:
            r = p.print(content, type=type, **kwargs)
            if not result and r:
                result = r
        return result

    def token_callback(self, token: str) -> None:
        """Forward a streamed token to all child printers.

        Args:
            token: The text token to forward.
        """
        for p in self.printers:
            p.token_callback(token)

    def reset(self) -> None:
        """Reset streaming state on all child printers."""
        for p in self.printers:
            p.reset()
