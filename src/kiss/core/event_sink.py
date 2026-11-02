# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The printer of the agent running on this thread, for records made outside the agent.

A tool, a SEA hook or a gate stage knows nothing of the agent that
called it, but what it did (a finish-gate stage, its duration, model,
cost and outcome) belongs in that agent's event log next to the model
calls and tool results.  :class:`~kiss.core.kiss_agent.KISSAgent` binds
its printer to the thread for the length of a run; :func:`publish`
prints a record through it, or drops the record when no agent runs on
the thread (a hook exercised directly in a test).  A helper thread the
hook starts takes the printer and the printer's own task binding along
(:func:`thread_context`, :func:`adopt_thread_context`): the printer files
a record under the task bound to the recording thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from kiss.core.printer import Printer

_state = threading.local()


def bind_printer(printer: Printer | None) -> Printer | None:
    """Make *printer* the thread's current printer; return the previous one (to restore)."""
    previous = getattr(_state, "printer", None)
    _state.printer = printer
    return previous


def current_printer() -> Printer | None:
    """Return the printer of the agent running on this thread, or ``None``."""
    return getattr(_state, "printer", None)


def thread_context() -> tuple[Printer | None, Any]:
    """Return the calling thread's printer and task binding, for :func:`adopt_thread_context`."""
    printer = current_printer()
    return printer, (None if printer is None else printer.thread_binding())


def adopt_thread_context(context: tuple[Printer | None, Any]) -> None:
    """Make the calling thread record as the thread *context* was taken on.

    Args:
        context: A :func:`thread_context`: the printer and its task binding.
    """
    printer, binding = context
    bind_printer(printer)
    if printer is not None:
        printer.adopt_thread_binding(binding)


def publish(record_type: str, **fields: Any) -> None:
    """Print a framework record (``gate_stage``, ...) through the thread's current printer.

    Args:
        record_type: The event type; the daemon persists the types it
            knows as records (``kiss.server.json_printer.RECORD_EVENT_TYPES``).
        **fields: The record's fields (numbers, strings, bools, lists, dicts).
    """
    printer = current_printer()
    if printer is not None:
        printer.print("", type=record_type, **fields)
