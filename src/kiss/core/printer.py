"""Abstract base class and shared utilities for KISS agent printers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

LANG_MAP = {
    "py": "python", "js": "javascript", "ts": "typescript",
    "sh": "bash", "bash": "bash", "zsh": "bash",
    "rb": "ruby", "rs": "rust", "go": "go",
    "java": "java", "c": "c", "cpp": "cpp", "h": "c",
    "json": "json", "yaml": "yaml", "yml": "yaml",
    "toml": "toml", "xml": "xml", "html": "html",
    "css": "css", "sql": "sql", "md": "markdown",
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
        dict[str, str]: Keys not in KNOWN_KEYS mapped to their string values
            (truncated to 200 chars).
    """
    extras: dict[str, str] = {}
    for k, v in tool_input.items():
        if k not in KNOWN_KEYS:
            val = str(v)
            if len(val) > 200:
                val = val[:200] + "..."
            extras[k] = val
    return extras


class Printer(ABC):
    @abstractmethod
    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Render content to the output destination.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "stream_event",
                "tool_call", "tool_result", "result", "usage_info", "message").
            **kwargs: Additional type-specific options (e.g. tool_input, is_error).

        Returns:
            str: Any extracted text (e.g. streamed text deltas), or empty string.
        """

    @abstractmethod
    async def token_callback(self, token: str) -> None:
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
            str: The result from the last child printer.
        """
        result = ""
        for p in self.printers:
            result = p.print(content, type=type, **kwargs)
        return result

    async def token_callback(self, token: str) -> None:
        """Forward a streamed token to all child printers.

        Args:
            token: The text token to forward.
        """
        for p in self.printers:
            await p.token_callback(token)

    def reset(self) -> None:
        """Reset streaming state on all child printers."""
        for p in self.printers:
            p.reset()
