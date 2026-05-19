"""Integration tests for multi-viewer streaming.

When more than one client is viewing the same running task, every
broadcast emitted by the agent must reach all of them.  The mechanism
is :meth:`BaseBrowserPrinter.subscribe_tab`, which adds a frontend
``tab_id`` to the set of subscribers for a ``task_id``.  ``broadcast()``
tags the event with the agent thread's ``taskId`` and emits one stamped
copy per subscriber tab.  System events that already carry an explicit
``tabId`` are forwarded verbatim and never fanned out.

This file pins the contract end-to-end against :class:`MemoryPrinter`,
a minimal in-memory subclass that mirrors the production
:class:`WebPrinter` broadcast logic verbatim and captures every
emission into ``printer.emitted`` for inspection.
"""

from __future__ import annotations

import threading

from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


def _emit_on(printer: MemoryPrinter, task_id: str, event: dict) -> None:
    """Set the thread-local task_id and broadcast *event* — exactly the
    pattern an agent thread uses on every broadcast."""

    def runner() -> None:
        printer._thread_local.task_id = task_id
        printer.broadcast(event)

    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "broadcast worker hung"


class TestMultiViewerFanout:
    """Subscribed tab ids each receive a stamped copy of every
    broadcast; recording happens exactly once per task."""

    def test_two_subscribers_get_their_own_tagged_copy(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("TASK1", "T1")
        printer.subscribe_tab("TASK1", "T2")

        _emit_on(printer, "TASK1", {"type": "text_delta", "text": "hi"})

        assert len(printer.emitted) == 2, (
            f"Expected 2 broadcasts (T1 + T2); got "
            f"{len(printer.emitted)}: {printer.emitted}"
        )
        tab_ids = sorted(str(e.get("tabId")) for e in printer.emitted)
        assert tab_ids == ["T1", "T2"], tab_ids
        for ev in printer.emitted:
            assert ev["type"] == "text_delta"
            assert ev["text"] == "hi"
            assert ev["taskId"] == "TASK1"

    def test_recording_is_one_entry_per_task_event(self) -> None:
        """Fan-out copies must NOT inflate the per-task recording or
        be persisted — only the canonical event keyed by ``taskId`` is.
        """
        printer = MemoryPrinter()
        # Enable recording for the running task.
        printer._recordings["TASK1"] = []
        printer.subscribe_tab("TASK1", "T1")
        printer.subscribe_tab("TASK1", "T2")

        _emit_on(printer, "TASK1", {"type": "text_delta", "text": "hi"})

        rec = printer._recordings.get("TASK1", [])
        assert len(rec) == 1, (
            f"Expected exactly one recorded event, got {len(rec)}"
        )
        assert rec[0].get("taskId") == "TASK1"

    def test_subscribe_is_idempotent(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("TASK1", "T1")
        printer.subscribe_tab("TASK1", "T1")
        printer.subscribe_tab("TASK1", "T1")

        _emit_on(printer, "TASK1", {"type": "text_delta", "text": "x"})

        tab_ids = [str(e.get("tabId")) for e in printer.emitted]
        assert tab_ids == ["T1"], tab_ids

    def test_no_subscribers_emits_nothing(self) -> None:
        """When no tab is subscribed to the task, the event is
        recorded / persisted but no wire copy is emitted."""
        printer = MemoryPrinter()

        _emit_on(printer, "TASK1", {"type": "text_delta", "text": "x"})

        assert printer.emitted == []

    def test_cleanup_tab_removes_tab_from_every_subscriber_set(self) -> None:
        """``cleanup_tab`` must drop the tab from every
        ``_subscribers`` set so the map cannot grow unboundedly across
        closed tabs."""
        printer = MemoryPrinter()
        printer.subscribe_tab("TASK1", "T1")
        printer.subscribe_tab("TASK1", "T2")
        printer.subscribe_tab("TASK2", "T1")

        printer.cleanup_tab("T1")
        assert "T1" not in printer._subscribers.get("TASK1", set())
        assert "TASK2" not in printer._subscribers, (
            "TASK2's only subscriber was T1; the empty set must be "
            "pruned by cleanup_tab."
        )

    def test_unsubscribe_drops_only_the_named_tab(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("TASK1", "T1")
        printer.subscribe_tab("TASK1", "T2")

        printer.unsubscribe_tab("TASK1", "T1")

        assert printer._subscribers["TASK1"] == {"T2"}

    def test_explicit_tabid_event_bypasses_fanout(self) -> None:
        """Events that already carry ``tabId`` are forwarded verbatim
        once, never duplicated for subscribers."""
        printer = MemoryPrinter()
        printer.subscribe_tab("TASK1", "T1")
        printer.subscribe_tab("TASK1", "T2")

        _emit_on(printer, "TASK1", {
            "type": "status", "running": True, "tabId": "T1",
        })

        assert len(printer.emitted) == 1
        assert printer.emitted[0]["tabId"] == "T1"
        assert printer.emitted[0]["type"] == "status"
