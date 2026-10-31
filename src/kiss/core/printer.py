# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Abstract base class and shared utilities for KISS agent printers."""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from kiss.core.models.model import parse_binary_attachments

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


# Commas only as thousands separators: "$1,25" is no figure (not 125).
_COST_RE = re.compile(r"\$?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")


def parse_cost(value: Any) -> float | None:
    """Return the USD figure of a cost field: a number, or a ``"$1.2345"`` string.

    Args:
        value: The field (``cost`` of a ``usage_info`` / ``result`` /
            ``llm_call`` event, or a display cost).

    Returns:
        The figure; ``None`` for ``"N/A"``, a bool, or anything else
        that is not one.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _COST_RE.fullmatch(value.strip()) if isinstance(value, str) else None
    return float(match.group(1).replace(",", "")) if match else None


def event_cost(event: dict[str, Any]) -> float | None:
    """Return the USD cost an event carries: its exact ``cost_usd``, else its ``cost``.

    Events carry the unrounded figure as ``cost_usd`` next to the
    four-decimal display string in ``cost``; the string is the fallback
    for events without it (older versions, late-charge deltas).

    Args:
        event: A ``usage_info`` / ``result`` event, or a charge delta.

    Returns:
        The cost; ``None`` when neither field is one.
    """
    exact = event.get("cost_usd")
    if isinstance(exact, (int, float)) and not isinstance(exact, bool):
        return float(exact)
    return parse_cost(event.get("cost"))


def cost_fields(
    cost: Any, cost_usd: float | None, budget_offset: float
) -> tuple[Any, float | None]:
    """Add the task's *budget_offset* to a cost; return both renderings.

    The ``"$x.xxxx"`` string is what the UI shows; the float is what a
    ``run_agent`` / ``run_parallel`` parent charges to its ledger
    (``daemon_client._net_totals``), so it must not go through the
    four-decimal rounding of the string.  A cost that is no figure
    (``"N/A"``, ``"$abc"``) without *cost_usd* passes through
    verbatim so a junk value never raises out of the emitting agent
    thread.

    Args:
        cost: The raw display cost (usually a ``"$1.2345"`` string).
        cost_usd: The unrounded USD figure when the emitter knows it;
            parsed from *cost* otherwise.
        budget_offset: The task's banked USD total to add.

    Returns:
        ``(display, exact)``: the offset-adjusted cost string (or
        *cost* unchanged) and the offset-adjusted USD float, ``None``
        when neither input yields a number.
    """
    exact = float(cost_usd) if cost_usd is not None else parse_cost(cost)
    if exact is None:
        return cost, None
    exact += budget_offset
    return f"${exact:.4f}", exact


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

    Also strips out base64 binary-attachment payloads emitted by the ``Read``
    tool so the terminal/chat page never has to render kilobytes of base64.

    Args:
        content: The string to truncate.

    Returns:
        str: The original string if short enough, otherwise the first and last
            halves joined by a truncation marker.
    """
    content, _ = parse_binary_attachments(content)
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


class Printer(ABC):
    @abstractmethod
    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        """Render content to the output destination.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "tool_call",
                "tool_result", "result").
            **kwargs: Additional type-specific options (e.g. tool_input, is_error).
        """

    @abstractmethod
    def token_callback(self, token: str) -> None:
        """Handle a single streamed token from the LLM.

        Args:
            token: The text token to process.
        """

    def thinking_callback(self, is_start: bool) -> None:
        """Handle thinking-block boundary events from the LLM.

        Called with ``True`` when a thinking block starts and ``False``
        when it ends.  Printers use this to switch between thinking and
        text display modes so that ``token_callback`` tokens are routed
        to the correct panel.

        The default implementation is a no-op.

        Args:
            is_start: ``True`` for block start, ``False`` for block end.
        """

    def thread_binding(self) -> Any:
        """Return the calling thread's task binding for :meth:`adopt_thread_binding`.

        A printer that keeps per-task state in thread-local storage
        (``JsonPrinter`` routes every event by the thread's ``task_id``)
        returns whatever a helper thread must adopt to print on the
        task's behalf.  The default has nothing to carry and returns
        ``None``.

        Returns:
            An opaque binding, or ``None``.
        """
        return None

    def adopt_thread_binding(self, binding: Any) -> None:
        """Bind the calling thread to the task of *binding*.

        Args:
            binding: A value returned by :meth:`thread_binding` on the
                task's own thread.  The default ignores it.
        """
