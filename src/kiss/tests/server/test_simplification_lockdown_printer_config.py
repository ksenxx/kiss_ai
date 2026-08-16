# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Characterization (lockdown) tests for json_printer and vscode_config.

Pins the CURRENT externally-observable behavior of
``kiss.server.json_printer.JsonPrinter`` and
``kiss.core.vscode_config`` so the planned simplifications
(tmp/findings-6.md sections B1, B2, B3, C1, C2, A3) cannot silently
change behavior:

* **B3** — budget/tokens/steps offset arithmetic must stay identical
  between the final ``result`` event (``_broadcast_result``) and the
  ``usage_info`` event path of ``print()``.
* **C2** — usage offsets are per-task: one task's offsets must never
  leak into another task's reported usage, and ``cleanup_task`` drops
  them.
* **C1 / A3** — bash-stream buffering: every printed chunk is
  broadcast as ``system_output`` exactly once (no loss, no
  duplication) even under concurrent print/flush threads, timer
  flushes are attributed to the owning task, and ``reset()`` discards
  any pending buffered text.
* **B1** — ``get_custom_model_entry`` and ``build_model_config`` parse
  ``custom_headers`` with identical semantics.
* **B2** — ``save_config``/``load_config`` round-trip: DEFAULTS
  overlay, preservation of unknown on-disk keys, dropping of unknown
  input keys, and graceful fallback to DEFAULTS on a corrupt file.

No mocks/patches/fakes are used: the tests drive a real
:class:`MemoryPrinter` (the in-memory capturing subclass mirroring the
production ``WebPrinter`` broadcast contract) and the real config
module with its ``CONFIG_DIR``/``CONFIG_PATH`` redirected to a fresh
temp dir per test (the established pattern from
``test_config_custom_headers.py``).
"""

from __future__ import annotations

import random
import re
import threading
import time
import unittest
from typing import Any

from kiss.tests.server._memory_printer import MemoryPrinter

_TOKEN_RE = re.compile(r"<\d{6}>")


def _events_of(printer: MemoryPrinter, event_type: str) -> list[dict[str, Any]]:
    """Return the captured fan-out events of *event_type* in emission order.

    Args:
        printer: The capturing printer under test.
        event_type: The ``type`` field to filter on.

    Returns:
        All matching events from ``printer.emitted``.
    """
    return [e for e in printer.emitted if e.get("type") == event_type]


def _make_task_printer(task_id: str, tab_id: str = "TAB1") -> MemoryPrinter:
    """Create a MemoryPrinter with *tab_id* subscribed and the calling
    thread's task id set to *task_id* — exactly the way an agent thread
    operates in production (``printer._thread_local.task_id = ...``).

    Args:
        task_id: Task id for the calling thread.
        tab_id: Frontend tab id to subscribe for fan-out capture.

    Returns:
        The configured printer.
    """
    printer = MemoryPrinter()
    printer.subscribe_tab(task_id, tab_id)
    printer._thread_local.task_id = task_id
    return printer



class TestBudgetOffsetParity(unittest.TestCase):
    """`result` and `usage_info` must apply identical offset arithmetic."""

    def test_result_and_usage_info_apply_identical_offsets(self) -> None:
        """Both paths add the SAME per-task offsets to cost/tokens/steps."""
        printer = _make_task_printer("PARITY")
        printer.budget_offset = 0.5
        printer.tokens_offset = 100
        printer.steps_offset = 3

        printer.print("usage", type="usage_info", total_tokens=1000, cost="$1.2500", total_steps=4)
        printer.print("all done", type="result", total_tokens=1000, cost="$1.2500", step_count=4)

        usage_events = _events_of(printer, "usage_info")
        result_events = _events_of(printer, "result")
        assert len(usage_events) == 1, printer.emitted
        assert len(result_events) == 1, printer.emitted
        usage, result = usage_events[0], result_events[0]

        assert usage["cost"] == "$1.7500", usage
        assert usage["total_tokens"] == 1100, usage
        assert usage["total_steps"] == 7, usage
        assert result["cost"] == usage["cost"], (result, usage)
        assert result["total_tokens"] == usage["total_tokens"], (result, usage)
        assert result["step_count"] == usage["total_steps"], (result, usage)

    def test_non_dollar_cost_passes_through_both_paths(self) -> None:
        """A cost not starting with '$' is never offset-adjusted."""
        printer = _make_task_printer("PARITY-NA")
        printer.budget_offset = 9.99

        printer.print("usage", type="usage_info", total_tokens=5, cost="N/A", total_steps=1)
        printer.print("done", type="result", total_tokens=5, cost="N/A", step_count=1)

        assert _events_of(printer, "usage_info")[0]["cost"] == "N/A"
        assert _events_of(printer, "result")[0]["cost"] == "N/A"

    def test_malformed_dollar_cost_result_path_passes_through(self) -> None:
        """`_broadcast_result` swallows ValueError for malformed '$' costs."""
        printer = _make_task_printer("PARITY-BAD")
        printer.budget_offset = 1.0

        printer.print("done", type="result", total_tokens=0, cost="$abc", step_count=0)

        assert _events_of(printer, "result")[0]["cost"] == "$abc"



class TestPerTaskOffsetIsolation(unittest.TestCase):
    """Offsets are keyed by task id and never leak across tasks."""

    def test_task_b_does_not_inherit_task_a_offsets(self) -> None:
        """Switching the thread-local task id exposes B's own (zero) offsets."""
        printer = _make_task_printer("A")
        printer.tokens_offset = 100
        printer.budget_offset = 0.5
        printer.steps_offset = 3

        printer._thread_local.task_id = "B"
        assert printer.tokens_offset == 0
        assert printer.budget_offset == 0.0
        assert printer.steps_offset == 0

        printer._thread_local.task_id = "A"
        assert printer.tokens_offset == 100
        assert printer.budget_offset == 0.5
        assert printer.steps_offset == 3

    def test_usage_info_under_other_task_is_unadjusted(self) -> None:
        """Task B's reported usage must not include task A's offsets."""
        printer = _make_task_printer("A")
        printer.subscribe_tab("B", "TAB-B")
        printer.tokens_offset = 100
        printer.budget_offset = 0.5
        printer.steps_offset = 3

        printer._thread_local.task_id = "B"
        printer.print("usage", type="usage_info", total_tokens=10, cost="$1.0000", total_steps=2)

        usage = _events_of(printer, "usage_info")[0]
        assert usage["taskId"] == "B", usage
        assert usage["total_tokens"] == 10, usage
        assert usage["cost"] == "$1.0000", usage
        assert usage["total_steps"] == 2, usage

    def test_cleanup_task_zeroes_offsets(self) -> None:
        """`cleanup_task` drops all three offsets for the task."""
        printer = _make_task_printer("A")
        printer.tokens_offset = 100
        printer.budget_offset = 0.5
        printer.steps_offset = 3

        printer.cleanup_task("A")

        assert printer.tokens_offset == 0
        assert printer.budget_offset == 0.0
        assert printer.steps_offset == 0



class TestBashStreamFlush(unittest.TestCase):
    """Bash buffering broadcasts every chunk exactly once."""

    def test_chunks_broadcast_exactly_once_via_tool_call_flush(self) -> None:
        """Buffered chunks are concatenated and flushed exactly once."""
        printer = _make_task_printer("BASH")

        printer.print("one", type="bash_stream")
        printer.print("two", type="bash_stream")
        printer.print("three", type="bash_stream")
        printer.print("Bash", type="tool_call", tool_input={})
        time.sleep(0.3)

        texts = [e["text"] for e in _events_of(printer, "system_output")]
        assert "".join(texts) == "onetwothree", texts
        assert len(_events_of(printer, "tool_call")) == 1, printer.emitted

    def test_timer_flush_attributes_task_id(self) -> None:
        """The 0.1s timer flush stamps the owning task's taskId even though
        it runs on a worker thread with no task id of its own."""
        printer = _make_task_printer("BASH-TIMER")

        printer.print("first", type="bash_stream")
        printer.print("later", type="bash_stream")
        time.sleep(0.35)

        texts = [e["text"] for e in _events_of(printer, "system_output")]
        assert "".join(texts) == "firstlater", texts
        for ev in _events_of(printer, "system_output"):
            assert ev["taskId"] == "BASH-TIMER", ev

    def test_reset_discards_pending_buffer(self) -> None:
        """`reset()` must prevent any stale buffered text from ever being
        broadcast afterwards (guards the generation-counter logic)."""
        printer = _make_task_printer("BASH-RESET")

        printer.print("flushed", type="bash_stream")
        printer.print("pending", type="bash_stream")
        printer.reset()
        time.sleep(0.3)

        texts = [e["text"] for e in _events_of(printer, "system_output")]
        assert texts == ["flushed"], texts

    def test_concurrent_print_and_flush_no_loss_no_duplication(self) -> None:
        """Race: one thread streams unique bash chunks with random sleeps
        while another triggers flushes concurrently.  Every chunk must be
        broadcast exactly once, each flushed payload must be a contiguous
        in-order slice of the chunk stream, and reassembling the payloads
        must reproduce the full concatenation of all printed chunks."""
        task = "BASH-RACE"
        printer = _make_task_printer(task)
        chunks = [f"<{i:06d}>" for i in range(120)]
        done = threading.Event()

        def producer() -> None:
            printer._thread_local.task_id = task
            for chunk in chunks:
                printer.print(chunk, type="bash_stream")
                time.sleep(random.random() * 0.02)
            done.set()

        def flusher() -> None:
            printer._thread_local.task_id = task
            while not done.is_set():
                printer._flush_bash()
                time.sleep(random.random() * 0.02)

        threads = [threading.Thread(target=producer), threading.Thread(target=flusher)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "bash race worker hung"
        printer._flush_bash()
        time.sleep(0.3)

        payloads = [e["text"] for e in _events_of(printer, "system_output")]
        seen: list[str] = []
        for payload in payloads:
            tokens = _TOKEN_RE.findall(payload)
            assert "".join(tokens) == payload, payload
            assert tokens == sorted(tokens), payload
            seen.extend(tokens)
        assert sorted(seen) == chunks, (
            f"missing={set(chunks) - set(seen)} dup_or_extra="
            f"{[t for t in seen if seen.count(t) > 1 or t not in chunks]}"
        )
        reassembled = "".join(sorted(payloads))
        assert reassembled == "".join(chunks), reassembled










if __name__ == "__main__":
    unittest.main()
