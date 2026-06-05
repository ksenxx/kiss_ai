# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Console output formatting for KISS agents."""

import sys
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from kiss.core.printer import (
    Printer,
    extract_extras,
    extract_path_and_lang,
    parse_result_yaml,
    truncate_result,
)


class ConsolePrinter(Printer):
    def __init__(self, file: Any = None) -> None:
        self._console = Console(highlight=False, file=file)
        self._file = file or sys.stdout
        self._mid_line = False
        self._bash_streamed = False
        self._current_block_type = ""
        # Per-task usage offsets, mirroring ``JsonPrinter``.  The agentic
        # loop sets these (``printer.tokens_offset = ...`` etc.) so that
        # sub-agent / continued-session usage accumulated into the parent
        # is included in the Result panel and usage line -- exactly like
        # the webview status bar.  Default 0 for a simple single-agent run.
        self.tokens_offset = 0
        self.budget_offset = 0.0
        self.steps_offset = 0

    def reset(self) -> None:
        """Reset internal streaming state for a new turn."""
        self._mid_line = False
        self._bash_streamed = False
        self._current_block_type = ""

    def _apply_budget_offset(self, cost: Any) -> Any:
        """Add the accumulated budget offset to a ``$x.xxxx`` cost string.

        Mirrors ``JsonPrinter`` so the Result panel and usage line on the
        console include sub-agent / continued-session spend.  Non-dollar
        values (e.g. ``"N/A"``) are returned unchanged.

        Args:
            cost: The cost value, typically a ``"$<float>"`` string.

        Returns:
            The cost with ``budget_offset`` added when it is a dollar
            string, otherwise the original value.
        """
        if isinstance(cost, str) and cost.startswith("$"):
            try:
                return f"${float(cost[1:]) + self.budget_offset:.4f}"
            except ValueError:
                return cost
        return cost

    @staticmethod
    def _format_result_content(raw: str) -> Group | Markdown:
        data = parse_result_yaml(raw)
        if data is None:
            return Markdown(raw)
        parts: list[Any] = []
        # Mirror the webview Result panel exactly (see ``main.js`` "result"
        # case): a continuation shows a yellow "Status: Continue", an
        # explicit failure shows a red "Status: FAILED", and a plain
        # success shows NO status banner at all.
        if data.get("is_continue"):
            parts.append(Text("Status: Continue", style="bold yellow"))
            parts.append(Text(""))
        elif data.get("success") is False:
            parts.append(Text("Status: FAILED", style="bold red"))
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
            type: Content type (e.g. "text", "prompt", "tool_call",
                "tool_result", "result", "message").
            **kwargs: Additional options such as tool_input, is_error, cost,
                total_tokens.

        Returns:
            str: Always the empty string.
        """
        if type == "text":
            # Match JsonPrinter: silently drop empty / whitespace-only
            # text so a blank line never appears in the terminal when nothing
            # would have been shown in the browser.
            if not str(content).strip():
                return ""
            self._flush_newline()
            # Default ``markup=True`` to mirror the browser printer, which
            # also lets Rich parse markup before flattening to plain text.
            self._console.print(content, **kwargs)
            return ""
        if type == "system_prompt":
            self._flush_newline()
            self._console.print(
                Panel(
                    Markdown(str(content)),
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
                    Markdown(str(content)),
                    title="[bold]Prompt[/bold]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
            return ""
        if type == "message":
            self._handle_message(content, **kwargs)
            return ""
        if type == "bash_stream":
            self._file.write(str(content))
            self._file.flush()
            self._mid_line = not str(content).endswith("\n")
            self._bash_streamed = True
            return ""
        if type == "tool_call":
            self._flush_newline()
            self._bash_streamed = False
            self._format_tool_call(str(content), kwargs.get("tool_input", {}))
            return ""
        if type == "tool_result":
            # Match JsonPrinter: show every tool's return value
            # so the console mirrors the webview.  Suppress only the
            # ``finish`` tool result -- the agentic loop renders that as
            # a dedicated "result" panel immediately after, so emitting
            # it here would be a duplicate.
            is_error = bool(kwargs.get("is_error", False))
            tool_name = kwargs.get("tool_name", "")
            if tool_name != "finish":
                self._flush_newline()
                self._print_tool_result(str(content), is_error=is_error)
            return ""
        if type == "usage_info":
            # Match JsonPrinter: surface per-step usage info so the
            # terminal user sees the same token / cost / step updates as
            # the webview status bar.  The ``content`` string is already
            # a human-readable summary built by KISSAgent.
            text = str(content)
            if text.strip():
                self._flush_newline()
                self._console.print(text, style="dim", highlight=False)
            return ""
        if type == "result":
            self._flush_newline()
            cost = self._apply_budget_offset(kwargs.get("cost", "N/A"))
            total_tokens = kwargs.get("total_tokens", 0) + self.tokens_offset
            step_count = kwargs.get("step_count", 0) + self.steps_offset
            body = self._format_result_content(str(content)) if content else "(no result)"
            self._console.print(
                Panel(
                    body,
                    title="Result",
                    subtitle=f"tokens={total_tokens:,}  cost={cost}  steps={step_count:,}",
                    border_style="bold green",
                    padding=(1, 2),
                )
            )
            return ""
        return ""

    def token_callback(self, token: str) -> None:
        """Stream a single token to the console, styled by current block type.

        Args:
            token: The text token to display.
        """
        if self._current_block_type == "thinking":
            self._stream_delta(token, style="dim cyan italic")
        else:
            self._stream_delta(token)

    def thinking_callback(self, is_start: bool) -> None:
        """Handle thinking-block boundary events.

        Sets ``_current_block_type`` so ``token_callback`` uses the correct
        style, and prints ruler lines to bracket thinking output.

        Args:
            is_start: ``True`` when a thinking block starts, ``False`` when it ends.
        """
        self._flush_newline()
        if is_start:
            self._current_block_type = "thinking"
            self._console.rule("Thinking", style="dim cyan", align="center")
        else:
            self._current_block_type = ""
            self._console.rule(style="dim cyan")
        self._console.print()

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

    def _print_tool_result(self, content: str, is_error: bool = False) -> None:
        label = "FAILED" if is_error else "RESULT"
        style = "red" if is_error else "green"
        self._console.rule(label, style=style, align="center")
        if not self._bash_streamed:
            display = truncate_result(content)
            for line in display.splitlines():
                self._file.write(line + "\n")
                self._file.flush()
        self._bash_streamed = False
        self._console.rule(style=style)

    def _handle_message(self, message: Any, **kwargs: Any) -> None:
        if hasattr(message, "subtype") and hasattr(message, "data"):
            if message.subtype == "tool_output":
                text = message.data.get("content", "")
                if text:
                    self._file.write(text)
                    self._file.flush()
                    self._mid_line = not text.endswith("\n")
        elif hasattr(message, "result"):
            budget_used = kwargs.get("budget_used", 0.0)
            total_tokens_used = kwargs.get("total_tokens_used", 0) + self.tokens_offset
            cost_str = self._apply_budget_offset(
                f"${budget_used:.4f}" if budget_used else "N/A"
            )
            self._flush_newline()
            body = self._format_result_content(message.result) if message.result else "(no result)"
            self._console.print(
                Panel(
                    body,
                    title="Result",
                    subtitle=(f"tokens={total_tokens_used:,}  cost={cost_str}"),
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
                    self._print_tool_result(content, is_error=bool(block.is_error))
