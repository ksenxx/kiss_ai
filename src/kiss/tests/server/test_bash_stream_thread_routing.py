# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Server-only tests extracted from
``kiss.tests.agents.sorcar.test_bash_stream_thread_routing``.

Moved here because their full dependency closure touches only
kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).  The
calling-thread test, which does not exercise kiss.server, stays in
the original module.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.server.json_printer import JsonPrinter


class _CapturingPrinter(JsonPrinter):
    """JsonPrinter that additionally captures every broadcast event.

    Per project test policy the transport is captured by subclassing
    ``broadcast()`` (see ``test_printer_equivalence.py``); the base
    class recording/persistence path still runs unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[dict[str, Any]] = []
        self._captured_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the (taskId-injected) event and capture a copy."""
        event = self._inject_task_id(event)
        with self._captured_lock:
            self.captured.append(dict(event))
        super().broadcast(event)


class BashStreamTaskRoutingTest(unittest.TestCase):
    """Bash streamed output must be attributed to the calling task."""

    TASK_ID = "task-42"

    def _run_bash_with_streaming(
        self,
    ) -> tuple[_CapturingPrinter, list[dict[str, Any]], str]:
        """Run a streaming Bash command through a real JsonPrinter.

        Mirrors the production wiring: the agent thread sets the
        thread-local ``task_id``, starts recording, and passes a
        ``stream_callback`` that calls ``printer.print(...,
        type="bash_stream")`` — exactly what
        ``SorcarAgent._get_tools._stream`` does.

        Returns:
            Tuple of (printer, recorded display events, Bash result).
        """
        printer = _CapturingPrinter()
        printer._thread_local.task_id = self.TASK_ID
        printer.start_recording()

        def _stream(text: str) -> None:
            printer.print(text, type="bash_stream")

        tools = UsefulTools(stream_callback=_stream)
        result = tools.Bash(
            "echo hello_stream; sleep 0.25; echo world_stream",
            description="stream two lines with a pause",
            timeout_seconds=30,
        )
        printer.print("Read", type="tool_call", tool_input={})
        events = printer.stop_recording()
        return printer, events, result

    def test_streamed_output_is_recorded_under_the_task(self) -> None:
        """The task's recording must contain the streamed bash output.

        Before the fix the reader thread recorded ``system_output``
        under the ``""`` fallback key, so the task's own recording —
        what the webview replays into the tool panel's bash sub panel
        and what gets persisted to history — was missing the output.
        """
        _, events, result = self._run_bash_with_streaming()
        self.assertIn("hello_stream", result)
        self.assertIn("world_stream", result)
        sys_out = "".join(
            e.get("text", "") for e in events if e.get("type") == "system_output"
        )
        self.assertIn("hello_stream", sys_out)
        self.assertIn("world_stream", sys_out)

    def test_streamed_events_carry_the_task_id(self) -> None:
        """Every broadcast ``system_output`` event must carry ``taskId``.

        An event without ``taskId`` is treated by ``WebPrinter`` as a
        global system event and is sent verbatim to EVERY connected
        client — that is the "garbage in the chat webview" symptom.
        """
        printer, _, _ = self._run_bash_with_streaming()
        sys_events = [
            e for e in printer.captured if e.get("type") == "system_output"
        ]
        self.assertTrue(sys_events, "no system_output events were broadcast")
        for ev in sys_events:
            self.assertEqual(
                ev.get("taskId"),
                self.TASK_ID,
                f"system_output event lost its task attribution: {ev!r}",
            )
