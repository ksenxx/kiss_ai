"""Console output formatting for KISS agents."""

import json
import sys
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from kiss.core.printer import (
    Printer,
    extract_extras,
    extract_path_and_lang,
    truncate_result,
)


class ConsolePrinter(Printer):
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

    def _flush_newline(self) -> None:
        if self._mid_line:
            self._file.write("\n")
            self._file.flush()
            self._mid_line = False

    def _stream_delta(self, text: str, **kwargs: Any) -> None:
        self._console.print(text, end="", highlight=False, **kwargs)
        if text:
            self._mid_line = not text.endswith("\n")

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        if type == "text":
            self._flush_newline()
            self._console.print(content, **kwargs)
            return ""
        if type == "prompt":
            self._flush_newline()
            self._console.print(
                Panel(
                    str(content),
                    title="[bold]Prompt[/bold]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
            return ""
        if type == "stream_event":
            return self._handle_stream_event(content)
        if type == "message":
            self._handle_message(content, **kwargs)
            return ""
        if type == "usage_info":
            self._flush_newline()
            self._console.print(
                Panel(
                    Text(str(content).strip(), style="dim italic"),
                    border_style="dim",
                    padding=(0, 1),
                    expand=True,
                )
            )
            return ""
        if type == "tool_call":
            self._flush_newline()
            self._format_tool_call(str(content), kwargs.get("tool_input", {}))
            return ""
        if type == "tool_result":
            self._flush_newline()
            self._print_tool_result(str(content), kwargs.get("is_error", False))
            return ""
        if type == "result":
            self._flush_newline()
            cost = kwargs.get("cost", "N/A")
            step_count = kwargs.get("step_count", 0)
            total_tokens = kwargs.get("total_tokens", 0)
            self._console.print(
                Panel(
                    str(content) or "(no result)",
                    title="Result",
                    subtitle=f"steps={step_count}  tokens={total_tokens}  cost={cost}",
                    border_style="bold green",
                    padding=(1, 2),
                )
            )
            return ""
        return ""

    async def token_callback(self, token: str) -> None:
        self._stream_delta(token)

    def _format_tool_call(self, name: str, tool_input: dict[str, Any]) -> None:
        file_path, lang = extract_path_and_lang(tool_input)
        parts: list[Any] = []

        if file_path:
            parts.append(Text(file_path, style="bold cyan"))
        if desc := tool_input.get("description"):
            parts.append(Text(str(desc), style="italic"))
        if command := tool_input.get("command"):
            parts.append(Syntax(str(command), "bash", theme="monokai", word_wrap=True))
        if content := tool_input.get("content"):
            parts.append(
                Syntax(str(content), lang, theme="monokai", line_numbers=True, word_wrap=True)
            )

        old_string = tool_input.get("old_string")
        new_string = tool_input.get("new_string")
        if old_string is not None:
            parts.append(Text("old:", style="bold red"))
            parts.append(Syntax(str(old_string), lang, theme="monokai", word_wrap=True))
        if new_string is not None:
            parts.append(Text("new:", style="bold green"))
            parts.append(Syntax(str(new_string), lang, theme="monokai", word_wrap=True))

        for k, v in extract_extras(tool_input).items():
            parts.append(Text(f"{k}: {v}", style="dim"))

        self._console.print(
            Panel(
                Group(*parts) if parts else Text("(no arguments)"),
                title=f"[bold blue]{name}[/bold blue]",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def _print_tool_result(self, content: str, is_error: bool) -> None:
        display = truncate_result(content)
        style = "red" if is_error else "green"
        self._console.rule("FAILED" if is_error else "OK", style=style, align="center")
        for line in display.splitlines():
            self._file.write(line + "\n")
            self._file.flush()
        self._console.rule(style=style)

    def _handle_stream_event(self, event: Any) -> str:
        evt = event.event
        evt_type = evt.get("type", "")
        text = ""

        if evt_type == "content_block_start":
            block = evt.get("content_block", {})
            block_type = block.get("type", "")
            self._current_block_type = block_type
            if block_type == "thinking":
                self._flush_newline()
                self._console.rule("Thinking", style="dim cyan", align="center")
                self._console.print()
            elif block_type == "tool_use":
                self._tool_name = block.get("name", "?")
                self._tool_json_buffer = ""
                self._flush_newline()
                self._console.print(f"[bold blue]{self._tool_name}[/bold blue] ", end="")
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

    def _handle_message(self, message: Any, **kwargs: Any) -> None:
        if hasattr(message, "subtype") and hasattr(message, "data"):
            if message.subtype == "tool_output":
                text = message.data.get("content", "")
                if text:
                    self._file.write(text)
                    self._file.flush()
                    self._mid_line = not text.endswith("\n")
        elif hasattr(message, "result"):
            step_count = kwargs.get("step_count", 0)
            budget_used = kwargs.get("budget_used", 0.0)
            total_tokens_used = kwargs.get("total_tokens_used", 0)
            cost_str = f"${budget_used:.4f}" if budget_used else "N/A"
            self._flush_newline()
            self._console.print(
                Panel(
                    message.result or "(no result)",
                    title="Result",
                    subtitle=(
                        f"steps={step_count}  tokens={total_tokens_used}  cost={cost_str}"
                    ),
                    border_style="bold green",
                    padding=(1, 2),
                )
            )
        elif hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "is_error") and hasattr(block, "content"):
                    content = (
                        block.content if isinstance(block.content, str) else str(block.content)
                    )
                    self._flush_newline()
                    self._print_tool_result(content, bool(block.is_error))
