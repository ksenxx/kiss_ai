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
    ext = Path(path).suffix.lstrip(".")
    return LANG_MAP.get(ext, ext or "text")


def truncate_result(content: str) -> str:
    if len(content) <= MAX_RESULT_LEN:
        return content
    half = MAX_RESULT_LEN // 2
    return content[:half] + "\n... (truncated) ...\n" + content[-half:]


def extract_path_and_lang(tool_input: dict) -> tuple[str, str]:
    file_path = str(tool_input.get("file_path") or tool_input.get("path") or "")
    lang = lang_for_path(file_path) if file_path else "text"
    return file_path, lang


def extract_extras(tool_input: dict) -> dict[str, str]:
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
        pass

    @abstractmethod
    async def token_callback(self, token: str) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class MultiPrinter(Printer):
    def __init__(self, printers: list[Printer]) -> None:
        self.printers = printers

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        result = ""
        for p in self.printers:
            result = p.print(content, type=type, **kwargs)
        return result

    async def token_callback(self, token: str) -> None:
        for p in self.printers:
            await p.token_callback(token)

    def reset(self) -> None:
        for p in self.printers:
            p.reset()
