# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""V5-3 / V5-4: the ``result`` event of a real :class:`JsonPrinter`.

* V5-3: a ``bash_stream`` fragment printed within 0.1 s of the previous
  one arms a flush timer.  ``print(type="result")`` used to broadcast
  the result without flushing it, so the timer delivered (and the
  recording persisted) a ``system_output`` below the final answer.
* V5-4: ``result`` and ``usage_info`` used to read the three usage
  offsets in three separate ``_lock`` holds; a ``set_usage_offsets``
  from a side-channel thread between two reads produced one event whose
  cost came from one generation of offsets and whose token / step totals
  from the next.  Pinned with a settrace :class:`Pause` on the second
  read; after the fix the three are read under one hold and the pause
  never triggers.
* The runner's own failure ``result`` (``AgentServer._broadcast_failure_result``:
  a stop, an agent error) bypassed ``print(type="result")`` and so
  that flush; it now flushes the failed task's buffer by task id.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any

from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.tests.server.pause import Pause

_TASK = "v5-result-task"


class _CapturingPrinter(JsonPrinter):
    """A real printer whose broadcasts are kept in order."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        event = self._inject_task_id(event)
        with self._lock:
            self.captured.append(dict(event))
        super().broadcast(event)


class TestResultFlushesPendingBashOutput(unittest.TestCase):
    """V5-3: nothing streamed by the task lands after its ``result``."""

    def test_no_system_output_after_result(self) -> None:
        printer = _CapturingPrinter()
        printer._thread_local.task_id = _TASK
        printer.start_recording()
        printer.print("Bash", type="tool_call", tool_input={"command": "echo"})
        printer.print("first ", type="bash_stream")
        printer.print("second", type="bash_stream")
        printer.print("done", type="result")
        time.sleep(0.4)
        recorded = printer.stop_recording()
        for events in (printer.captured, recorded):
            types = [e["type"] for e in events]
            self.assertIn("result", types)
            self.assertEqual(
                types[types.index("result") + 1 :], [], f"events after result: {types}"
            )
            streamed = "".join(e["text"] for e in events if e["type"] == "system_output")
            self.assertEqual(streamed, "first second")


class TestRunnerFailureResultFlushesPendingBashOutput(unittest.TestCase):
    """The runner's own terminal ``result`` (a stop, an error) flushes the task's output too.

    ``_broadcast_failure_result`` runs on the worker thread AFTER
    ``ChatSorcarAgent.run`` restored the thread's previous task
    binding, so the flush has to be keyed by the failed task's id, not
    by the thread.
    """

    def _streamed_then_failed(self, task_id: str) -> _CapturingPrinter:
        printer = _CapturingPrinter()
        printer._thread_local.task_id = task_id
        printer.start_recording()
        printer.print("Bash", type="tool_call", tool_input={"command": "echo"})
        printer.print("first ", type="bash_stream")
        printer.print("second", type="bash_stream")
        printer._thread_local.task_id = ""
        AgentServer(printer)._broadcast_failure_result(
            agent=None, result_summary="Task stopped by user",
            task_history_id=task_id, tab_id="v5-tab",
        )
        time.sleep(0.4)
        return printer

    def test_no_system_output_after_the_runner_result(self) -> None:
        printer = self._streamed_then_failed(_TASK)
        printer._thread_local.task_id = _TASK
        recorded = printer.stop_recording()
        for events in (printer.captured, recorded):
            types = [e["type"] for e in events]
            self.assertEqual(types[types.index("result") + 1 :], [], f"after result: {types}")
            streamed = [e for e in events if e["type"] == "system_output"]
            self.assertEqual("".join(e["text"] for e in streamed), "first second")
            self.assertTrue(all(e.get("taskId") == _TASK for e in streamed), streamed)

    def test_a_task_without_streamed_output_just_gets_the_result(self) -> None:
        printer = _CapturingPrinter()
        AgentServer(printer)._broadcast_failure_result(
            agent=None, result_summary="Task failed: boom",
            task_history_id="v5-no-bash", tab_id="v5-tab",
        )
        AgentServer(printer)._broadcast_failure_result(
            agent=None, result_summary="Task failed: setup", task_history_id=None, tab_id="v5-tab",
        )
        self.assertEqual(
            [(e["type"], e.get("taskId"), e.get("tabId")) for e in printer.captured],
            [("result", "v5-no-bash", None), ("result", None, "v5-tab")],
        )


class TestResultReadsOffsetsAtomically(unittest.TestCase):
    """V5-4: cost, tokens and steps of one event come from one generation."""

    def test_offsets_written_mid_event_do_not_mix(self) -> None:
        printer = _CapturingPrinter()
        printer.set_usage_offsets(_TASK, 1.0, 1, 1)

        def emit_result() -> None:
            printer._thread_local.task_id = _TASK
            printer.print(
                "done", type="result", total_tokens=0, cost="$0.0000", step_count=0, cost_usd=0.0,
            )

        def second_read(frame_locals: dict[str, Any]) -> bool:
            return frame_locals.get("offsets") is printer._tokens_offsets

        emitter = threading.Thread(target=emit_result, name="v5-emitter", daemon=True)
        with Pause(JsonPrinter._read_offset.__code__, second_read) as hold:
            emitter.start()
            # Unfixed: held between the budget read and the token read.
            # Fixed: one locked read, the hold never triggers.
            hold.paused.wait(1)
            printer.set_usage_offsets(_TASK, 2.0, 2, 2)
        emitter.join(10)
        self.assertFalse(emitter.is_alive())
        results = [e for e in printer.captured if e["type"] == "result"]
        self.assertEqual(len(results), 1)
        seen = (results[0]["cost_usd"], results[0]["total_tokens"], results[0]["step_count"])
        self.assertIn(seen, [(1.0, 1, 1), (2.0, 2, 2)])
        self.assertEqual(results[0]["cost"], f"${seen[0]:.4f}")


if __name__ == "__main__":
    unittest.main()
