# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A tool served natively records through the event sink like one on the agent's thread.

Audit 1101, C1 F2.  A Codex tool call reaches the agent on a bridge
thread, where ``_execute_tool_on_helper_thread`` re-binds the stop
event and the printer's task routing, but not the event sink's printer
(``kiss.core.event_sink``, a ``threading.local``): a gate stage the
tool or a hook published there was dropped, and a worker the tool
started with ``thread_context()`` inherited no printer either.  Real
``ToolBridge`` over a socket, real printer, nothing mocked.
"""

from __future__ import annotations

import threading
from typing import Any

from kiss.core.base import collecting_trajectories
from kiss.core.event_sink import adopt_thread_context, current_printer, publish, thread_context
from kiss.core.kiss_agent import KISSAgent
from kiss.core.printer import Printer
from kiss.tests.core.test_audit1018_codex_bridge_stop import _call, _reply, _schema
from kiss.tests.core.test_f1_codex_bridge_findings import _bridge_of


class _Recorder(Printer):
    """A real ``Printer`` keeping every event, with the thread each came from."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any], int]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        """Record the event and the recording thread.

        Args:
            content: The content to display.
            type: Content type (``tool_call``, ``gate_stage``, ...).
            **kwargs: Type-specific fields.
        """
        self.events.append((type, kwargs, threading.get_ident()))

    def token_callback(self, token: str) -> None:
        """Ignore streamed tokens.

        Args:
            token: The text token (unused).
        """


SEEN: list[Printer | None] = []


def publishes_stage() -> str:
    """Publish a gate stage from the tool's thread and from a worker it starts."""
    SEEN.append(current_printer())
    publish("gate_stage", stage="from-the-tool", success=True)
    context = thread_context()

    def work() -> None:
        adopt_thread_context(context)
        publish("gate_stage", stage="from-a-worker", success=True)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    return "published"


def test_a_natively_served_tool_publishes_through_the_agents_printer() -> None:
    SEEN.clear()
    printer = _Recorder()
    with collecting_trajectories(save=False):
        agent = KISSAgent("native event sink")
        agent._reset("codex/gpt-5.5", True, 10, 10.0, None, printer=printer, verbose=False)
        agent.function_map = {"publishes_stage": publishes_stage}
        agent._cached_tools_schema = _schema("publishes_stage")
        with agent._native_tools_exposed():
            bridge = _bridge_of(agent)
            reply = _reply(_call(bridge, "publishes_stage"))
    assert reply == {"text": "published"}
    assert SEEN == [printer], "the tool's thread must see the agent's printer"
    stages = [fields["stage"] for kind, fields, _ in printer.events if kind == "gate_stage"]
    assert stages == ["from-the-tool", "from-a-worker"]
    kinds = [kind for kind, _, _ in printer.events]
    assert kinds == ["tool_call", "gate_stage", "gate_stage", "tool_result"]
    # Nothing was recorded on this (the agent's) thread: every record came
    # from the bridge's thread or the tool's worker.
    assert all(thread != threading.get_ident() for _, _, thread in printer.events)
