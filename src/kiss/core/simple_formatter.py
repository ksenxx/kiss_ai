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

from kiss.core.formatter import Formatter


def _left_aligned_heading(self: Any, console: Any, options: Any) -> Generator[Any]:
    self.text.justify = "left"
    if self.tag == "h1":
        yield Panel(self.text, box=box.HEAVY, style="markdown.h1.border")
    else:
        if self.tag == "h2":
            yield Text("")
        yield self.text


Heading.__rich_console__ = _left_aligned_heading  # type: ignore[method-assign]


class SimpleFormatter(Formatter):
    def __init__(self) -> None:
        self.color = sys.stdout.isatty()
        self._console = Console() if self.color else None
        self._stderr_console = Console(stderr=True) if self.color else None

    def format_message(self, message: dict[str, str]) -> str:
        return f'\n# role="{message.get("role", "")}" #\n{message.get("content", "")}\n'

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(self.format_message(m) for m in messages)

    def print_message(self, message: dict[str, str]) -> None:
        if self._console:
            self._console.print(Markdown(f'\n# role="{message.get("role", "")}" #'), style="bold")
            self._console.print(Markdown(message.get("content", "")))
        else:
            print(f'\n# role="{message.get("role", "")}" #')
            print(message.get("content", ""))

    def print_messages(self, messages: list[dict[str, str]]) -> None:
        if self._console:
            self._console.print(Markdown("\n#  Agent Messages #"), style="bold")
            for message in messages:
                self.print_message(message)
        else:
            print("\n#  Agent Messages #")
            print(self.format_messages(messages), end="")

    def print_status(self, message: str) -> None:
        if self._console:
            self._console.print(message, style="green")
        else:
            print(message)

    def print_error(self, message: str) -> None:
        if self._stderr_console:
            self._stderr_console.print(message, style="red")
        else:
            print(message, file=sys.stderr)

    def print_warning(self, message: str) -> None:
        if self._console:
            self._console.print(message, style="yellow")
        else:
            print(message)
