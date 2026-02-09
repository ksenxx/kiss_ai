"""Console output formatting for Claude Coding Agent."""

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

_LANG_MAP = {
    "py": "python", "js": "javascript", "ts": "typescript",
    "sh": "bash", "bash": "bash", "zsh": "bash",
    "rb": "ruby", "rs": "rust", "go": "go",
    "java": "java", "c": "c", "cpp": "cpp", "h": "c",
    "json": "json", "yaml": "yaml", "yml": "yaml",
    "toml": "toml", "xml": "xml", "html": "html",
    "css": "css", "sql": "sql", "md": "markdown",
}
_MAX_RESULT_LEN = 3000


class ConsolePrinter:
    """Handles all terminal output for Claude Coding Agent.

    API:
        print_stream_event(event) -> str:
            Handle a streaming event and print to terminal.
            Returns extracted text content for token callbacks.

        print_message(message, **context) -> None:
            Print a complete message (SystemMessage, UserMessage, or ResultMessage).
            For ResultMessage, pass step_count, budget_used, total_tokens_used.
    """

    def __init__(self, file: Any = None) -> None:
        self._console = Console(highlight=False, file=file)
        self._file = file or sys.stdout
        self._mid_line = False
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""

    def reset(self) -> None:
        self._mid_line = False
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""

    @staticmethod
    def _lang_for_path(path: str) -> str:
        ext = Path(path).suffix.lstrip(".")
        return _LANG_MAP.get(ext, ext or "text")

    def _flush_newline(self) -> None:
        if self._mid_line:
            self._file.write("\n")
            self._file.flush()
            self._mid_line = False

    def _stream_delta(self, text: str, **kwargs: Any) -> None:
        self._console.print(text, end="", highlight=False, **kwargs)
        if text:
            self._mid_line = not text.endswith("\n")

    def _format_tool_call(self, name: str, tool_input: dict[str, Any]) -> None:
        file_path = str(
            tool_input.get("file_path") or tool_input.get("path") or ""
        )
        lang = self._lang_for_path(file_path) if file_path else "text"
        parts: list[Any] = []

        if file_path:
            parts.append(Text(file_path, style="bold cyan"))

        desc = tool_input.get("description")
        if desc:
            parts.append(Text(str(desc), style="italic"))

        command = tool_input.get("command")
        if command:
            parts.append(
                Syntax(str(command), "bash", theme="monokai", word_wrap=True)
            )

        content = tool_input.get("content")
        if content:
            parts.append(
                Syntax(
                    str(content), lang, theme="monokai",
                    line_numbers=True, word_wrap=True,
                )
            )

        old_string = tool_input.get("old_string")
        new_string = tool_input.get("new_string")
        if old_string is not None:
            parts.append(Text("old:", style="bold red"))
            parts.append(
                Syntax(str(old_string), lang, theme="monokai", word_wrap=True)
            )
        if new_string is not None:
            parts.append(Text("new:", style="bold green"))
            parts.append(
                Syntax(str(new_string), lang, theme="monokai", word_wrap=True)
            )

        skip = {
            "file_path", "path", "content", "command",
            "old_string", "new_string", "description",
        }
        for k, v in tool_input.items():
            if k not in skip:
                val = str(v)
                if len(val) > 200:
                    val = val[:200] + "..."
                parts.append(Text(f"{k}: {val}", style="dim"))

        self._console.print(
            Panel(
                Group(*parts) if parts else Text("(no arguments)"),
                title=f"[bold blue]{name}[/bold blue]",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def _print_tool_result(self, content: str, is_error: bool) -> None:
        display = content
        if len(display) > _MAX_RESULT_LEN:
            half = _MAX_RESULT_LEN // 2
            display = (
                display[:half]
                + "\n... (truncated) ...\n"
                + display[-half:]
            )
        if is_error:
            self._console.rule("FAILED", style="red", align="center")
        else:
            self._console.rule("OK", style="green", align="center")
        for line in display.splitlines():
            self._file.write(line + "\n")
            self._file.flush()
        style = "red" if is_error else "green"
        self._console.rule(style=style)

    def print_stream_event(self, event: Any) -> str:
        """Handle a streaming event, print to terminal in real-time.

        Args:
            event: A StreamEvent (or any object with an `event` dict attribute).

        Returns:
            Extracted text content (for token callbacks).
        """
        evt = event.event
        evt_type = evt.get("type", "")
        text = ""

        if evt_type == "content_block_start":
            block = evt.get("content_block", {})
            block_type = block.get("type", "")
            self._current_block_type = block_type
            if block_type == "thinking":
                self._flush_newline()
                self._console.rule(
                    "Thinking", style="dim cyan", align="center"
                )
                self._console.print()
            elif block_type == "tool_use":
                self._tool_name = block.get("name", "?")
                self._tool_json_buffer = ""
                self._flush_newline()
                self._console.print(
                    f"[bold blue]{self._tool_name}[/bold blue] ", end="",
                )
                self._mid_line = True

        elif evt_type == "content_block_delta":
            delta = evt.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "thinking_delta":
                text = delta.get("thinking", "")
                self._stream_delta(text, style="dim cyan italic")
            elif delta_type == "text_delta":
                text = delta.get("text", "")
                self._stream_delta(text)
            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                self._tool_json_buffer += partial
                self._stream_delta(partial, style="dim")

        elif evt_type == "content_block_stop":
            block_type = self._current_block_type
            if block_type == "thinking":
                self._flush_newline()
                self._console.rule(style="dim cyan")
                self._console.print()
            elif block_type == "tool_use":
                self._flush_newline()
                try:
                    tool_input = json.loads(self._tool_json_buffer)
                except (json.JSONDecodeError, ValueError):
                    tool_input = {"_raw": self._tool_json_buffer}
                self._format_tool_call(self._tool_name, tool_input)
            else:
                self._flush_newline()
            self._current_block_type = ""

        return text

    def print_message(
        self,
        message: Any,
        step_count: int = 0,
        budget_used: float = 0.0,
        total_tokens_used: int = 0,
    ) -> None:
        """Print a complete message to terminal.

        Uses duck typing to detect message kind:
          - SystemMessage: has `subtype` and `data` attributes.
          - ResultMessage: has `result` attribute.
          - UserMessage: has `content` list with tool-result blocks.

        Args:
            message: A SystemMessage, UserMessage, or ResultMessage.
            step_count: Agent step count (for ResultMessage subtitle).
            budget_used: Budget used in USD (for ResultMessage subtitle).
            total_tokens_used: Total tokens used (for ResultMessage subtitle).
        """
        if hasattr(message, "subtype") and hasattr(message, "data"):
            self._print_system(message)
        elif hasattr(message, "result"):
            self._print_result(
                message, step_count, budget_used, total_tokens_used
            )
        elif hasattr(message, "content"):
            self._print_tool_results(message)

    def _print_system(self, message: Any) -> None:
        if message.subtype == "tool_output":
            text = message.data.get("content", "")
            if text:
                self._file.write(text)
                self._file.flush()
                self._mid_line = not text.endswith("\n")

    def _print_result(
        self,
        message: Any,
        step_count: int,
        budget_used: float,
        total_tokens_used: int,
    ) -> None:
        cost_str = f"${budget_used:.4f}" if budget_used else "N/A"
        subtitle = (
            f"steps={step_count}  "
            f"tokens={total_tokens_used}  cost={cost_str}"
        )
        self._flush_newline()
        self._console.print(
            Panel(
                message.result or "(no result)",
                title="Result",
                subtitle=subtitle,
                border_style="bold green",
                padding=(1, 2),
            )
        )

    def print_usage_info(self, usage_info: str) -> None:
        self._flush_newline()
        md = Markdown(usage_info.strip())
        self._console.print(
            Panel(
                md,
                border_style="dim",
                padding=(0, 1),
            ),
            style="dim",
        )

    def _print_tool_results(self, message: Any) -> None:
        for block in message.content:
            if hasattr(block, "is_error") and hasattr(block, "content"):
                content = (
                    block.content
                    if isinstance(block.content, str)
                    else str(block.content)
                )
                self._flush_newline()
                self._print_tool_result(content, bool(block.is_error))
