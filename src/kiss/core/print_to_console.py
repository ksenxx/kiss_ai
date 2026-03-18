"""Console output formatting for KISS agents."""

import logging
import sys
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from kiss.core.printer import (
    Printer,
    StreamEventParser,
    extract_extras,
    extract_path_and_lang,
    parse_result_yaml,
    truncate_result,
)

logger = logging.getLogger(__name__)

class ConsolePrinter(StreamEventParser, Printer):
    def __init__(self, file: Any = None) -> None:
        StreamEventParser.__init__(self)
        self._console = Console(highlight=False, file=file)
        self._file = file or sys.stdout
        self._mid_line = False

    def reset(self) -> None:
        """Reset internal streaming and tool-parsing state for a new turn."""
        self._mid_line = False
        self.reset_stream_state()

    @staticmethod
    def _format_result_content(raw: str) -> Group | str:
        data = parse_result_yaml(raw)
        if data is None:
            return raw
        parts: list[Any] = []
        if "success" in data:
            style = "bold green" if data["success"] else "bold red"
            label = "PASSED" if data["success"] else "FAILED"
            parts.append(Text(f"Status: {label}", style=style))
            parts.append(Text(""))
        parts.append(Markdown(str(data["summary"])))
        return Group(*parts)

    def _flush_newline(self) -> None:
        if self._mid_line:
            self._file.write("\n")
            self._file.flush()
            self._mid_line = False

    def _stream_delta(self, text: str, **kwargs: Any) -> None:
        self._console.print(text, end="", highlight=False, markup=False, **kwargs)
        if text:
            self._mid_line = not text.endswith("\n")

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Render content to the console using Rich formatting.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "stream_event",
                "tool_call", "tool_result", "result", "usage_info", "message").
            **kwargs: Additional options such as tool_input, is_error, cost,
                step_count, total_tokens.

        Returns:
            str: Extracted text from stream events, or empty string.
        """
        if type == "text":
            self._flush_newline()
            self._console.print(content, markup=False, **kwargs)
            return ""
        if type == "system_prompt":
            self._flush_newline()
            self._console.print(
                Panel(
                    Text(str(content)),
                    title="[bold]System Prompt[/bold]",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )
            return ""
        if type == "prompt":
            self._flush_newline()
            self._console.print(
                Panel(
                    Text(str(content)),
                    title="[bold]Prompt[/bold]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
            return ""
        if type == "stream_event":
            return self.parse_stream_event(content)
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
        if type == "bash_stream":
            self._file.write(str(content))
            self._file.flush()
            self._mid_line = not str(content).endswith("\n")
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
            body = self._format_result_content(str(content)) if content else "(no result)"
            self._console.print(
                Panel(
                    body,
                    title="Result",
                    subtitle=f"steps={step_count}  tokens={total_tokens}  cost={cost}",
                    border_style="bold green",
                    padding=(1, 2),
                )
            )
            return ""
        return ""

    async def token_callback(self, token: str) -> None:
        """Stream a single token to the console, styled by current block type.

        Args:
            token: The text token to display.
        """
        if self._current_block_type == "thinking":
            self._stream_delta(token, style="dim cyan italic")
        else:
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

    def _on_thinking_start(self) -> None:
        self._flush_newline()
        self._console.rule("Thinking", style="dim cyan", align="center")
        self._console.print()

    def _on_thinking_end(self) -> None:
        self._flush_newline()
        self._console.rule(style="dim cyan")
        self._console.print()

    def _on_tool_use_start(self, name: str) -> None:
        self._flush_newline()
        self._console.print(f"[bold blue]{name}[/bold blue] ", end="")
        self._mid_line = True

    def _on_tool_json_delta(self, partial: str) -> None:
        self._stream_delta(partial, style="dim")

    def _on_tool_use_end(self, name: str, tool_input: dict) -> None:
        self._flush_newline()
        self._format_tool_call(name, tool_input)

    def _on_text_block_end(self) -> None:
        self._flush_newline()

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
            body = self._format_result_content(message.result) if message.result else "(no result)"
            self._console.print(
                Panel(
                    body,
                    title="Result",
                    subtitle=(f"steps={step_count}  tokens={total_tokens_used}  cost={cost_str}"),
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
