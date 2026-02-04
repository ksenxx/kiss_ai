# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Simple formatter implementation using Rich for terminal output."""

import sys
from collections.abc import Generator
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Heading, Markdown
from rich.panel import Panel
from rich.text import Text

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.formatter import Formatter


def _left_aligned_heading(self: Any, console: Any, options: Any) -> Generator[Any]:
    """Render a heading with left-aligned text for Rich markdown output.

    Args:
        self: The Heading instance.
        console: The Rich console instance.
        options: The console options.

    Yields:
        Rich renderable objects for the heading.
    """
    self.text.justify = "left"
    if self.tag == "h1":
        yield Panel(self.text, box=box.HEAVY, style="markdown.h1.border")
    else:
        if self.tag == "h2":
            yield Text("")
        yield self.text


Heading.__rich_console__ = _left_aligned_heading  # type: ignore[method-assign]


class SimpleFormatter(Formatter):
    """Simple formatter implementation using Rich for colorful terminal output."""

    def __init__(self) -> None:
        """Initialize the SimpleFormatter with Rich console instances."""
        self.color = sys.stdout.isatty()
        self._console = Console() if self.color else None
        self._stderr_console = Console(stderr=True) if self.color else None

    def format_message(self, message: dict[str, Any]) -> str:
        """Format a single message as a string.

        Args:
            message: A dictionary containing message data with 'role' and 'content' keys.

        Returns:
            str: The formatted message string, or empty string if not verbose.
        """
        if DEFAULT_CONFIG.agent.verbose:
            return f'\n## role="{message.get("role", "")}" #\n{message.get("content", "")}\n'
        return ""

    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format a list of messages as a single string.

        Args:
            messages: A list of message dictionaries.

        Returns:
            str: The formatted messages joined by newlines, or empty string if not verbose.
        """
        if DEFAULT_CONFIG.agent.verbose:
            return "\n".join(self.format_message(m) for m in messages)
        return ""

    def print_message(self, message: dict[str, Any]) -> None:
        """Print a single message with Rich formatting.

        Args:
            message: A dictionary containing message data with 'role' and 'content' keys.
        """
        if DEFAULT_CONFIG.agent.verbose:
            role = message.get("role", "")
            content = message.get("content", "")
            if self._console:
                self._console.print(Markdown(f'\n## role="{role}" #'), style="bold")
                self._console.print(Markdown(content))
            else:
                print(f'\n## role="{role}" #')
                print(content)

    def print_messages(self, messages: list[dict[str, Any]]) -> None:
        """Print a list of messages with Rich formatting.

        Args:
            messages: A list of message dictionaries.
        """
        if DEFAULT_CONFIG.agent.verbose:
            if self._console:
                self._console.print()
                self._console.print(Markdown("\n#  Agent Messages #"), style="bold")
                for message in messages:
                    self.print_message(message)
            else:
                print("\n#  Agent Messages #")
                print(self.format_messages(messages), end="")

    def print_status(self, message: str) -> None:
        """Print a status message in green.

        Args:
            message: The status message to print.
        """
        if DEFAULT_CONFIG.agent.verbose:
            if self._console:
                self._console.print(message, style="green")
            else:
                print(message)

    def print_error(self, message: str) -> None:
        """Print an error message in red to stderr.

        Args:
            message: The error message to print.
        """
        if DEFAULT_CONFIG.agent.verbose:
            if self._stderr_console:
                self._stderr_console.print(message, style="red")
            else:
                print(message, file=sys.stderr)

    def print_warning(self, message: str) -> None:
        """Print a warning message in yellow.

        Args:
            message: The warning message to print.
        """
        if DEFAULT_CONFIG.agent.verbose:
            if self._console:
                self._console.print(message, style="yellow")
            else:
                print(message)

    def print_label_and_value(self, label: str, value: str) -> None:
        """Print a label and value pair with Rich formatting.

        Args:
            label: The label text.
            value: The value text.
        """
        if DEFAULT_CONFIG.agent.verbose:
            md = Markdown(f"__**{label}**__: {value}")
            if self._console:
                self._console.print(md)
            else:
                print(md)
