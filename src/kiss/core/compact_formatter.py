# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Compact formatter implementation using terminal output."""

import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.formatter import Formatter

LINE_LENGTH = 100

class CompactFormatter(Formatter):
    """Compact formatter that displays truncated single-line messages."""

    def __init__(self) -> None:
        """Initialize the CompactFormatter with Rich console instances."""
        self.color = sys.stdout.isatty()
        self._console = Console() if self.color else None
        self._stderr_console = Console(stderr=True) if self.color else None

    def format_message(self, message: dict[str, Any]) -> str:
        """Format a single message as a truncated single line.

        Args:
            message: A dictionary containing message data with 'role' and 'content' keys.

        Returns:
            str: The formatted truncated message string, or empty string if not verbose.
        """
        if DEFAULT_CONFIG.agent.verbose:
            content = message.get("content", "").replace(chr(10), chr(92) + "n")
            return f'[{message.get("role", "unknown")}]: {content}'[:LINE_LENGTH] + " ..."
        return ""

    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format a list of messages as truncated single lines.

        Args:
            messages: A list of message dictionaries.

        Returns:
            str: The formatted messages joined by newlines, or empty string if not verbose.
        """
        if DEFAULT_CONFIG.agent.verbose:
            return "\n".join(self.format_message(m) for m in messages)
        return ""

    def print_message(self, message: dict[str, Any]) -> None:
        """Print a single message in compact format.

        Args:
            message: A dictionary containing message data with 'role' and 'content' keys.
        """
        if DEFAULT_CONFIG.agent.verbose:
            print(self.format_message(message))

    def print_messages(self, messages: list[dict[str, Any]]) -> None:
        """Print a list of messages in compact format.

        Args:
            messages: A list of message dictionaries.
        """
        if DEFAULT_CONFIG.agent.verbose:
            print(self.format_messages(messages))

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
