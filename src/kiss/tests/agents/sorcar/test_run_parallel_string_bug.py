"""Bug reproducer + regression test for run_parallel character-iteration bug.

Bug: ``run_tasks_parallel(tasks, ...)`` does ``for i, task in enumerate(tasks)``.
If the LLM passes ``tasks`` as a bare string (e.g. ``"hello"``) instead of
``["hello"]``, Python iterates the string character-by-character and the
function spawns one sub-agent per character.

Fix: coerce a bare string into a single-element list (``str`` -> ``[str]``)
and reject other non-list types with ``TypeError``.

No mocks, patches, fakes, or test doubles.  These tests run cheaply
because they trigger ``ValueError`` from ``ThreadPoolExecutor(max_workers=0)``
*after* the broadcast loop (which is where the bug manifests) — so no real
LLM call is made.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent, run_tasks_parallel
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter


class _CapturePrinter(BaseBrowserPrinter):
    """Real BaseBrowserPrinter subclass that captures all broadcast events."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[dict[str, Any]] = []
        self._capture_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record event then delegate to parent for normal recording."""
        event = self._inject_tab_id(event)
        with self._capture_lock:
            self.captured.append(event)
        super().broadcast(event)


def _count_open_subagent_events(printer: _CapturePrinter) -> int:
    return sum(
        1 for e in printer.captured if e.get("type") == "openSubagentTab"
    )


class TestRunParallelStringBug:
    """Reproduce + verify the fix for the str-vs-list[str] iteration bug."""

    def test_string_input_does_not_iterate_characters(self) -> None:
        """A bare string must not produce one sub-agent per character.

        With the bug, ``"hello"`` produces 5 ``openSubagentTab`` broadcasts
        (one per character).  With the fix, it produces at most 1 broadcast
        (coerced to ``["hello"]``) or 0 (rejected with ``TypeError``).

        We use ``max_workers=0`` so ``ThreadPoolExecutor`` raises before
        any actual LLM call is made; the broadcast loop runs first.
        """
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-strbug"

        try:
            run_tasks_parallel(
                "hello",  # type: ignore[arg-type]  # the bug: not a list
                max_workers=0,
                printer=printer,
            )
        except (ValueError, TypeError):
            pass  # ValueError from max_workers=0, TypeError from fix

        count = _count_open_subagent_events(printer)
        assert count <= 1, (
            f"String input 'hello' produced {count} sub-agent broadcasts — "
            "the bug iterates the string character-by-character."
        )

    def test_long_string_input_not_iterated(self) -> None:
        """Longer realistic-looking task string also must not be iterated."""
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-long"

        task_str = "Summarize file foo.py and reply with the result."
        try:
            run_tasks_parallel(
                task_str,  # type: ignore[arg-type]
                max_workers=0,
                printer=printer,
            )
        except (ValueError, TypeError):
            pass

        count = _count_open_subagent_events(printer)
        assert count <= 1, (
            f"String input of length {len(task_str)} produced {count} "
            "sub-agent broadcasts — character-iteration bug present."
        )

    def test_run_parallel_closure_rejects_string(self) -> None:
        """The ``run_parallel`` tool exposed to the LLM must also handle str."""
        agent = SorcarAgent("test-string-bug")
        agent._use_web_tools = False
        agent._is_parallel = True
        tools = agent._get_tools()
        run_parallel = next(
            t for t in tools if getattr(t, "__name__", "") == "run_parallel"
        )

        # Either raise TypeError, or accept by coercion (but then with
        # max_workers=0 the ThreadPoolExecutor surfaces ValueError after
        # the broadcast loop — still no character iteration).
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-closure"
        agent.printer = printer

        try:
            run_parallel("hello world", max_workers=0)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            pass

        count = _count_open_subagent_events(printer)
        assert count <= 1, (
            f"run_parallel closure produced {count} sub-agent broadcasts "
            f"from string input — bug present."
        )

    def test_non_list_non_str_rejected(self) -> None:
        """Other non-list inputs (e.g. dict, int) must raise TypeError."""
        with pytest.raises(TypeError):
            run_tasks_parallel({"k": "v"}, max_workers=1)  # type: ignore[arg-type]

    def test_list_of_strings_still_works(self) -> None:
        """Normal list input still goes through the broadcast loop correctly."""
        printer = _CapturePrinter()
        printer._thread_local.tab_id = "parent-normal"

        try:
            run_tasks_parallel(
                ["task A", "task B"],
                max_workers=0,
                printer=printer,
            )
        except (ValueError, TypeError):
            pass

        count = _count_open_subagent_events(printer)
        assert count == 2, (
            f"Normal list input should produce 2 broadcasts, got {count}"
        )
