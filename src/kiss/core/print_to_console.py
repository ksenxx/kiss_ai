# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Console output formatting for KISS agents."""

import sys
import threading
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from kiss.core.html_render import html_to_rich
from kiss.core.printer import (
    Printer,
    extract_extras,
    extract_path_and_lang,
    parse_result_yaml,
    truncate_result,
)

_NOTIFICATION_STYLES = {
    "error": ("yellow", "✕ ERROR"),
    "warning": ("yellow", "⚠ WARNING"),
    "info": ("yellow", "ℹ INFO"),
}


class _PrinterThreadState(threading.local):
    """Per-thread streaming state for :class:`ConsolePrinter`.

    ``bash_streamed`` (is a bash RESULT rule currently open?),
    ``block_type`` (thinking vs normal text) and the three usage offsets
    (what the printing agent has already spent in earlier sub-sessions)
    describe the *agent* that is printing, not the terminal.  Parallel
    sub-agents each run on their own thread and share one printer, so
    keeping these per thread is what stops one sub-agent from closing
    another's bash block, restyling another's tokens, or contributing
    its own totals to another's result panel.  ``threading.local`` runs
    ``__init__`` once per thread, so every thread starts from the same
    defaults.
    """

    def __init__(self) -> None:
        self.bash_streamed = False
        self.block_type = ""
        self.tokens_offset = 0
        self.budget_offset = 0.0
        self.steps_offset = 0


class ConsolePrinter(Printer):
    """Rich-formatted console printer, safe to share across agent threads.

    ``run_tasks_parallel`` forwards one printer object verbatim to every
    parallel sub-agent, and the live usage monitor prints from a third
    daemon thread, so this class faces the same fan-out ``JsonPrinter``
    does.  It is protected the same way: state that belongs to the
    printing agent is thread-local, the shared cursor position
    (``_mid_line``, a property of the one output stream) stays shared,
    and every public entry point holds ``_lock`` for its whole body so a
    panel and its rules are emitted as one uninterrupted unit.
    """

    def __init__(self, file: Any = None) -> None:
        self._console = Console(highlight=False, file=file)
        self._explicit_file: Any = file
        self._lock = threading.RLock()
        self._thread_state = _PrinterThreadState()
        self._mid_line = False

    @property
    def tokens_offset(self) -> int:
        """Tokens *this thread's* agent spent before its current session."""
        return self._thread_state.tokens_offset

    @tokens_offset.setter
    def tokens_offset(self, value: int) -> None:
        self._thread_state.tokens_offset = value

    @property
    def budget_offset(self) -> float:
        """Dollars *this thread's* agent spent before its current session."""
        return self._thread_state.budget_offset

    @budget_offset.setter
    def budget_offset(self, value: float) -> None:
        self._thread_state.budget_offset = value

    @property
    def steps_offset(self) -> int:
        """Steps *this thread's* agent took before its current session."""
        return self._thread_state.steps_offset

    @steps_offset.setter
    def steps_offset(self, value: int) -> None:
        self._thread_state.steps_offset = value

    @property
    def _bash_streamed(self) -> bool:
        """Whether *this thread* has an open streamed-bash RESULT rule."""
        return self._thread_state.bash_streamed

    @_bash_streamed.setter
    def _bash_streamed(self, value: bool) -> None:
        self._thread_state.bash_streamed = value

    @property
    def _current_block_type(self) -> str:
        """The block (``"thinking"`` or ``""``) *this thread* is streaming."""
        return self._thread_state.block_type

    @_current_block_type.setter
    def _current_block_type(self, value: str) -> None:
        self._thread_state.block_type = value

    @property
    def _file(self) -> Any:
        """Output stream — resolved lazily so a later ``sys.stdout`` swap is honoured.

        Returns the explicit *file* passed to ``__init__`` when one
        was provided, otherwise the current ``sys.stdout`` at access
        time (matching the lazy resolution Rich's :class:`Console`
        does for its own writes).
        """
        return self._explicit_file if self._explicit_file is not None else sys.stdout

    def reset(self) -> None:
        """Reset internal streaming state for a new turn."""
        with self._lock:
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
        if data.get("is_continue"):
            parts.append(Text("Status: Continue", style="bold yellow"))
            parts.append(Text(""))
        elif data.get("success") is False:
            parts.append(Text("Status: FAILED", style="bold red"))
            parts.append(Text(""))
        parts.append(html_to_rich(str(data["summary"])))
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

        The whole render is serialized on ``_lock`` so that an event made
        of several writes (a rule, a panel, a raw body, a closing rule)
        reaches the terminal as one unit even when parallel sub-agents
        share this printer.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "tool_call",
                "tool_result", "result", "message").
            **kwargs: Additional options such as tool_input, is_error, cost,
                total_tokens.

        Returns:
            str: Always the empty string.
        """
        with self._lock:
            return self._render(content, type, **kwargs)

    def _render(self, content: Any, type: str, **kwargs: Any) -> str:
        """Render one event; the caller must hold ``_lock``."""
        if type == "text":
            if not str(content).strip():
                return ""
            self._flush_newline()
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
            if not self._bash_streamed:
                self._flush_newline()
                self._console.rule("RESULT", style="green", align="center")
            self._file.write(str(content))
            self._file.flush()
            self._mid_line = not str(content).endswith("\n")
            self._bash_streamed = True
            return ""
        if type == "tool_call":
            self._flush_newline()
            if self._bash_streamed:
                self._console.rule(style="green")
                self._bash_streamed = False
            self._format_tool_call(str(content), kwargs.get("tool_input", {}))
            return ""
        if type == "tool_result":
            is_error = bool(kwargs.get("is_error", False))
            tool_name = kwargs.get("tool_name", "")
            tool_input = kwargs.get("tool_input")
            if tool_name != "finish":
                self._flush_newline()
                self._print_tool_result(
                    str(content),
                    is_error=is_error,
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
            return ""
        if type == "usage_info":
            text = str(content)
            if text.strip():
                self._flush_newline()
                self._console.print(text, style="dim", highlight=False)
            return ""
        if type == "notification":
            severity = str(kwargs.get("severity", "info")).lower()
            border, label = _NOTIFICATION_STYLES.get(severity, _NOTIFICATION_STYLES["info"])
            message = str(content) if content is not None else ""
            parts: list[Any] = [Text(message, style="yellow")]
            progress_message = kwargs.get("progress_message") or ""
            if progress_message:
                parts.append(Text(str(progress_message), style="yellow dim"))
            self._flush_newline()
            self._console.print(
                Panel(
                    Group(*parts),
                    title=f"[bold]{label}[/bold]",
                    border_style=border,
                    padding=(0, 1),
                )
            )
            return ""
        if type == "result":
            cost = self._apply_budget_offset(kwargs.get("cost", "N/A"))
            total_tokens = kwargs.get("total_tokens", 0) + self.tokens_offset
            step_count = kwargs.get("step_count", 0) + self.steps_offset
            self._print_result_panel(
                content,
                f"tokens={total_tokens:,}  cost={cost}  steps={step_count:,}",
            )
            return ""
        return ""

    def _print_result_panel(self, raw: Any, subtitle: str) -> None:
        """Render the green "Result" panel shared by result and message events."""
        self._flush_newline()
        body = self._format_result_content(str(raw)) if raw else "(no result)"
        self._console.print(
            Panel(
                body,
                title="Result",
                subtitle=subtitle,
                border_style="bold green",
                padding=(1, 2),
            )
        )

    def token_callback(self, token: str) -> None:
        """Stream a single token to the console, styled by current block type.

        Args:
            token: The text token to display.
        """
        with self._lock:
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
        with self._lock:
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

    @staticmethod
    def _should_syntax_highlight_read(
        tool_name: str,
        is_error: bool,
        tool_input: dict[str, Any] | None,
        display: str,
    ) -> bool:
        """Return True iff a ``Read`` tool_result should be syntax-highlighted.

        The output of the ``Read`` tool is the textual content of the
        file the model asked to read.  The console renders that
        content with syntax highlighting derived from the file
        extension (matching the language picker
        used by ``_format_tool_call`` for the inverse direction —
        ``Write`` / ``Edit`` inputs).  Non-content results (errors,
        the ``(file is empty)`` sentinel, the binary-attachment
        header) are NOT real file body so they are left as plain
        text so the user can still read the diagnostic message.
        """
        if tool_name != "Read" or is_error or not tool_input:
            return False
        file_path = tool_input.get("file_path") or tool_input.get("path")
        if not file_path:
            return False
        stripped = display.lstrip()
        if stripped.startswith("Error:"):
            return False
        if display.strip() == "(file is empty)":
            return False
        if display.startswith("Read binary file "):
            return False
        return True

    def _print_tool_result(
        self,
        content: str,
        is_error: bool = False,
        tool_name: str = "",
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        label = "FAILED" if is_error else "RESULT"
        style = "red" if is_error else "green"
        if not self._bash_streamed:
            self._console.rule(label, style=style, align="center")
            display = truncate_result(content)
            if self._should_syntax_highlight_read(
                tool_name=tool_name,
                is_error=is_error,
                tool_input=tool_input,
                display=display,
            ):
                assert tool_input is not None
                _, lang = extract_path_and_lang(tool_input)
                start_line = tool_input.get("start_line", 1)
                if not isinstance(start_line, int) or start_line < 1:
                    start_line = 1
                self._console.print(
                    Syntax(
                        display,
                        lang,
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=True,
                        start_line=start_line,
                    )
                )
            else:
                for line in display.splitlines():
                    self._file.write(line + "\n")
                    self._file.flush()
            self._console.rule(style=style)
        else:
            if is_error:
                self._console.rule(label, style=style, align="center")
            else:
                self._console.rule(style=style)
        self._bash_streamed = False

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
            cost_str = self._apply_budget_offset(f"${budget_used:.4f}" if budget_used else "N/A")
            self._print_result_panel(
                message.result, f"tokens={total_tokens_used:,}  cost={cost_str}"
            )
        elif hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "is_error") and hasattr(block, "content"):
                    content = (
                        block.content if isinstance(block.content, str) else str(block.content)
                    )
                    self._flush_newline()
                    self._print_tool_result(content, is_error=bool(block.is_error))
