# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Per-task printer state must not outlive its task.

R09-7.  :class:`JsonPrinter` keeps several dicts keyed by task id.
``cleanup_task`` frees them all under ``self._lock``, but two of the
lifetimes had holes:

* The three usage-offset dicts were written by plain property setters
  with **no** lock, while ``cleanup_task`` pops them under one.  A
  write that lands after the cleanup re-creates the entry, and nothing
  will ever pop it again — one permanently leaked entry per
  occurrence, in a daemon that runs for weeks.  The writers are not
  only the task's own thread: ``_attribute_sub_usage`` folds a
  finished sub-agent's spend into the parent's offsets, so a
  ``run_parallel`` sub-agent that outlives its parent's cleanup writes
  exactly there.

Real printers, real threads, real bash streaming — nothing is mocked.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from kiss.server.json_printer import _CLOSED_TASK_MEMORY, JsonPrinter


class TestPrinterTaskStateLifetime(unittest.TestCase):
    """Task-keyed printer state is freed exactly when the task ends."""

    def _offset_keys(self, printer: JsonPrinter) -> set[str]:
        """Every task key currently holding a usage offset."""
        return (
            set(printer._tokens_offsets)
            | set(printer._budget_offsets)
            | set(printer._steps_offsets)
        )

    def test_offset_write_after_cleanup_does_not_leak_a_dead_key(
        self,
    ) -> None:
        """A straggler write must not resurrect a finished task."""
        printer = JsonPrinter()
        printer._thread_local.task_id = "T1"
        printer.tokens_offset = 5
        printer.budget_offset = 1.25
        printer.steps_offset = 2
        assert self._offset_keys(printer) == {"T1"}

        printer.cleanup_task("T1")
        assert self._offset_keys(printer) == set()

        printer.tokens_offset = 7
        printer.budget_offset = 2.5
        printer.steps_offset = 3
        assert self._offset_keys(printer) == set(), (
            "a write after cleanup_task re-created the dead task's "
            "offsets; nothing pops them again so they leak forever"
        )
        assert printer.tokens_offset == 0
        assert printer.budget_offset == 0.0
        assert printer.steps_offset == 0

    def test_straggler_subagent_thread_cannot_resurrect_a_task(self) -> None:
        """A sub-agent finishing after its parent must not leak either.

        Mirrors ``_attribute_sub_usage``: the sub-agent runs on its own
        thread, bound to the parent's task id, and folds its spend into
        the parent's offsets.  The barrier makes it land strictly after
        the parent's ``cleanup_task``.
        """
        printer = JsonPrinter()
        printer._thread_local.task_id = "T2"
        printer.tokens_offset = 10

        cleaned = threading.Event()
        wrote = threading.Event()

        def sub_agent() -> None:
            printer._thread_local.task_id = "T2"
            cleaned.wait(10)
            printer.tokens_offset = 99
            printer.budget_offset = 9.5
            printer.steps_offset = 9
            wrote.set()

        thread = threading.Thread(target=sub_agent, daemon=True)
        thread.start()
        printer.cleanup_task("T2")
        cleaned.set()
        assert wrote.wait(10), "the sub-agent thread never wrote"
        thread.join(10)

        assert self._offset_keys(printer) == set(), (
            "a sub-agent that outlived its parent task leaked the "
            f"parent's offset entries: {self._offset_keys(printer)}"
        )

    def test_live_task_offsets_survive_another_tasks_cleanup(self) -> None:
        """Cleanup is per task: a running task keeps its own offsets."""
        printer = JsonPrinter()
        printer._thread_local.task_id = "keep"
        printer.tokens_offset = 4
        printer._thread_local.task_id = "drop"
        printer.tokens_offset = 8

        printer.cleanup_task("drop")

        printer._thread_local.task_id = "keep"
        assert printer.tokens_offset == 4, (
            "cleaning one task disturbed another task's offsets"
        )
        printer.tokens_offset = 6
        assert printer.tokens_offset == 6, (
            "a live task must still be able to update its offsets"
        )

    def test_closed_task_memory_is_bounded(self) -> None:
        """The guard against late writes cannot itself grow forever."""
        printer = JsonPrinter()
        total = _CLOSED_TASK_MEMORY + 10
        for index in range(total):
            printer._thread_local.task_id = f"T{index}"
            printer.tokens_offset = index
            printer.cleanup_task(f"T{index}")

        assert len(printer._closed_tasks) == _CLOSED_TASK_MEMORY, (
            "the finished-task memory grew past its cap: "
            f"{len(printer._closed_tasks)}"
        )
        assert f"T{total - 1}" in printer._closed_tasks, (
            "the most recently finished task must still be remembered"
        )
        assert "T0" not in printer._closed_tasks, (
            "the oldest finished task should have been evicted first"
        )
        assert self._offset_keys(printer) == set()

    def test_straggler_cannot_resurrect_a_task_the_memory_forgot(self) -> None:
        """A straggler later than the bounded memory still finds no home.

        Once ``_CLOSED_TASK_MEMORY`` later cleanups evict a finished
        task's tombstone, its absence proves nothing: per-task state is
        then created only for a task the agent-state registry holds
        active.  The delayed fold of a finished child
        (``set_usage_offsets``), a thread still bound to the task, its
        model pick, its tool call and its image all leave no entry.
        """
        printer = JsonPrinter()
        dead = SimpleNamespace(last_task_id="dead")
        printer.agent_task_allocated(dead, "dead")
        printer.set_usage_offsets("dead", 1.0, 10, 1)
        assert self._offset_keys(printer) == {"dead"}
        printer.agent_task_finished(dead, "dead")
        for index in range(_CLOSED_TASK_MEMORY):
            printer.cleanup_task(f"later-{index}")
        assert "dead" not in printer._closed_tasks

        printer.set_usage_offsets("dead", 1.25, 100, 2)
        printer.adopt_thread_binding("dead")
        self.addCleanup(printer.adopt_thread_binding, "")
        printer.tokens_offset = 7
        printer.broadcast_agent_model_pick("some-model", "tab", "dead")
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "shot.png"
            image.write_bytes(b"png")
            printer.print("Bash", type="tool_call", tool_input={"command": "x"})
            printer.print(str(image), type="tool_result", tool_name="Bash", tool_input={})
        assert self._offset_keys(printer) == set()
        assert "dead" not in printer._task_model_override
        assert "dead" not in printer._tool_call_started
        assert "dead" not in printer._image_b64_used

        live = SimpleNamespace(last_task_id="live")
        printer.agent_task_allocated(live, "live")
        self.addCleanup(printer.agent_task_finished, live, "live")
        printer.set_usage_offsets("live", 2.0, 20, 2)
        assert self._offset_keys(printer) == {"live"}, (
            "a task the registry holds active must still store its offsets"
        )


if __name__ == "__main__":
    unittest.main()
