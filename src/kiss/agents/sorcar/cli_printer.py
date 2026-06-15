# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Recording printer used by the ``sorcar`` CLI.

The plain :class:`~kiss.core.print_to_console.ConsolePrinter` renders
rich panels to the terminal but does NOT persist any events to the
chat database.  As a consequence a task launched from the CLI shows
nothing but a synthesised ``prompt`` and ``result`` when the user
later opens the session in the chat webview — every intermediate
``tool_call``, ``tool_result``, ``text_delta`` etc. is silently lost.

:class:`RecordingConsolePrinter` extends
:class:`~kiss.agents.vscode.json_printer.JsonPrinter` (which already
records every display event into the task-keyed recording and queues
it to the ``events`` table via ``_queue_chat_event``) and forwards
each :meth:`print`, :meth:`token_callback`, :meth:`thinking_callback`,
and :meth:`reset` call to an internal :class:`ConsolePrinter` so the
terminal user still sees the Rich panels.  Installing this printer on
the CLI run path therefore makes a CLI run fully replayable in the
chat webview without losing the live terminal experience.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.sorcar import cli_daemon_bridge
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.core.print_to_console import ConsolePrinter


class RecordingConsolePrinter(JsonPrinter):
    """A console printer that ALSO records events into the chat DB.

    Inherits the JsonPrinter's recording / persistence machinery (so
    every display event reaches the ``events`` table the chat webview
    reads) and wraps a :class:`ConsolePrinter` so the same events also
    render to the terminal as Rich panels.

    The per-task usage offsets (``tokens_offset`` / ``budget_offset`` /
    ``steps_offset``) are mirrored onto the inner ConsolePrinter so
    that the green Result panel printed to the terminal carries the
    same totals the persisted ``result`` event carries — the agentic
    loop only sets the offset on the outer printer.
    """

    def __init__(self) -> None:
        super().__init__()
        self._console = ConsolePrinter()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record + persist the event AND forward it to the daemon.

        Inherits :meth:`JsonPrinter.broadcast` for the in-process
        record/persist side effects.  After persistence, forwards a
        copy of the (task-id-injected) event to the local daemon's
        UDS endpoint via :mod:`cli_daemon_bridge`, so any chat
        webview currently subscribed to the task's chat id receives
        the event live instead of having to wait for the next page
        reload to pick it up from the DB.

        Args:
            event: The event dictionary to broadcast.
        """
        super().broadcast(event)
        injected = self._inject_task_id(event)
        if injected.get("taskId"):
            cli_daemon_bridge.send_event(injected)

    @property
    def tokens_offset(self) -> int:
        """Mirror :attr:`JsonPrinter.tokens_offset` (task-keyed)."""
        return JsonPrinter.tokens_offset.fget(self)  # type: ignore[attr-defined,no-any-return]

    @tokens_offset.setter
    def tokens_offset(self, value: int) -> None:
        JsonPrinter.tokens_offset.fset(self, value)  # type: ignore[attr-defined]
        self._console.tokens_offset = value

    @property
    def budget_offset(self) -> float:
        """Mirror :attr:`JsonPrinter.budget_offset` (task-keyed)."""
        return JsonPrinter.budget_offset.fget(self)  # type: ignore[attr-defined,no-any-return]

    @budget_offset.setter
    def budget_offset(self, value: float) -> None:
        JsonPrinter.budget_offset.fset(self, value)  # type: ignore[attr-defined]
        self._console.budget_offset = value

    @property
    def steps_offset(self) -> int:
        """Mirror :attr:`JsonPrinter.steps_offset` (task-keyed)."""
        return JsonPrinter.steps_offset.fget(self)  # type: ignore[attr-defined,no-any-return]

    @steps_offset.setter
    def steps_offset(self, value: int) -> None:
        JsonPrinter.steps_offset.fset(self, value)  # type: ignore[attr-defined]
        self._console.steps_offset = value

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Record/persist the event AND render it to the terminal.

        Delegates to :meth:`JsonPrinter.print` for the
        recording/persistence side effects and to the wrapped
        :class:`ConsolePrinter` for terminal rendering.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "tool_call",
                "tool_result", "result", "message").
            **kwargs: Additional options (tool_input, is_error, cost,
                total_tokens, etc.) forwarded to both printers.

        Returns:
            str: Always the empty string (matches the base contract).
        """
        super().print(content, type=type, **kwargs)
        self._console.print(content, type=type, **kwargs)
        return ""

    def token_callback(self, token: str) -> None:
        """Broadcast the token as a delta AND stream it to the terminal.

        Args:
            token: The text token.
        """
        super().token_callback(token)
        self._console.token_callback(token)

    def thinking_callback(self, is_start: bool) -> None:
        """Broadcast thinking-boundary AND render ruler line to terminal.

        Args:
            is_start: ``True`` at thinking-block start, ``False`` at end.
        """
        super().thinking_callback(is_start)
        self._console.thinking_callback(is_start)

    def reset(self) -> None:
        """Reset both the recording state and the inner console state."""
        super().reset()
        self._console.reset()
