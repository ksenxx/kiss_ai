"""Integration tests for multi-viewer streaming.

When more than one client is viewing the same running task, every
broadcast emitted by the agent must reach all of them.  The
mechanism is :meth:`BaseBrowserPrinter.subscribe_tab`, which
registers an additional viewer tab id under a canonical source tab
id.  ``broadcast()`` then emits the primary event once (recorded
and persisted under the source tab id) and emits a fan-out copy
per viewer (same payload, only ``tabId`` replaced; no recording, no
persistence) so that every connected client whose frontend filters
by its own tab id renders the event.

This file pins the contract end-to-end against
:class:`MemoryPrinter`, a minimal in-memory subclass that mirrors
the production :class:`WebPrinter` broadcast logic verbatim and
captures every emission into ``printer.emitted`` for inspection.
"""

from __future__ import annotations

import threading

from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


def _emit_on(printer: MemoryPrinter, tab_id: str, event: dict) -> None:
    """Set the thread-local tab id and broadcast *event* — exactly the
    pattern ``_run_task`` uses on every agent thread."""

    def runner() -> None:
        printer._thread_local.tab_id = tab_id
        printer.broadcast(event)

    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "broadcast worker hung"


class TestMultiViewerFanout:
    """Subscribed viewer tab ids each receive a tagged copy of the
    broadcast; the source tab id receives exactly one copy; recording
    happens once."""

    def test_two_subscribers_get_their_own_tagged_copy(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("T1", "T2")
        printer.subscribe_tab("T1", "T3")

        _emit_on(printer, "T1", {"type": "text_delta", "text": "hi"})

        assert len(printer.emitted) == 3, (
            f"Expected 3 broadcasts (T1 + T2 + T3); got "
            f"{len(printer.emitted)}: {printer.emitted}"
        )
        tab_ids = sorted(str(e.get("tabId")) for e in printer.emitted)
        assert tab_ids == ["T1", "T2", "T3"], tab_ids
        for ev in printer.emitted:
            assert ev["type"] == "text_delta"
            assert ev["text"] == "hi"

    def test_recording_captures_only_primary_event(self) -> None:
        """Fan-out copies must NOT inflate the per-tab recording or be
        persisted — only the primary (source-tagged) event is."""
        printer = MemoryPrinter()
        # Enable recording for the source tab (recording is lazy —
        # `_record_event` no-ops unless the per-tab list is present).
        # Also enable for the viewer tabs to prove fan-out does NOT
        # accidentally feed their recording lists.
        printer._recordings["T1"] = []
        printer._recordings["T2"] = []
        printer._recordings["T3"] = []
        printer.subscribe_tab("T1", "T2")
        printer.subscribe_tab("T1", "T3")

        _emit_on(printer, "T1", {"type": "text_delta", "text": "hi"})

        rec = printer._recordings.get("T1", [])
        assert len(rec) == 1, (
            f"Expected exactly one recorded primary event, got {len(rec)}"
        )
        assert rec[0].get("tabId") == "T1"
        assert printer._recordings.get("T2", []) == []
        assert printer._recordings.get("T3", []) == []

    def test_subscribe_is_idempotent(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("T1", "T2")
        printer.subscribe_tab("T1", "T2")
        printer.subscribe_tab("T1", "T2")

        _emit_on(printer, "T1", {"type": "text_delta", "text": "x"})

        tab_ids = sorted(str(e.get("tabId")) for e in printer.emitted)
        assert tab_ids == ["T1", "T2"], tab_ids

    def test_subscribing_to_self_is_a_no_op(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("T1", "T1")

        _emit_on(printer, "T1", {"type": "text_delta", "text": "x"})

        tab_ids = [str(e.get("tabId")) for e in printer.emitted]
        assert tab_ids == ["T1"], tab_ids

    def test_no_subscribers_emits_only_primary(self) -> None:
        printer = MemoryPrinter()

        _emit_on(printer, "T1", {"type": "text_delta", "text": "x"})

        assert len(printer.emitted) == 1
        assert printer.emitted[0]["tabId"] == "T1"

    def test_cleanup_tab_removes_source_and_viewer_entries(self) -> None:
        """``cleanup_tab`` must drop the tab from ``_subscribers`` both
        as a key (source) and as a member (viewer) so the map cannot
        grow unboundedly across closed/reopened tabs."""
        printer = MemoryPrinter()
        printer.subscribe_tab("T1", "T2")
        printer.subscribe_tab("T1", "T3")
        printer.subscribe_tab("T9", "T2")

        printer.cleanup_tab("T2")
        assert "T2" not in printer._subscribers.get("T1", set())
        assert "T9" not in printer._subscribers, (
            "T9's only subscriber was T2; the empty set must be "
            "pruned by cleanup_tab."
        )

        printer.cleanup_tab("T1")
        assert "T1" not in printer._subscribers

    def test_fanout_resolves_through_alias(self) -> None:
        """Subscribing under a canonical id must still fan out when
        the broadcast is tagged with an aliased source id (e.g. a
        sub-agent's ``orig_sub_tab_id`` rebound via ``rebind_tab``)."""
        printer = MemoryPrinter()
        printer.rebind_tab("ORIG", "T1")
        printer.subscribe_tab("T1", "T2")

        _emit_on(printer, "ORIG", {"type": "text_delta", "text": "x"})

        tab_ids = sorted(str(e.get("tabId")) for e in printer.emitted)
        assert tab_ids == ["T1", "T2"], tab_ids


class TestMultiViewerOrdering:
    """Fan-out is sequential and the primary event is always written
    first so a viewer never sees its copy before the source client."""

    def test_primary_event_is_written_before_fanout_copies(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("T1", "T2")
        printer.subscribe_tab("T1", "T3")

        _emit_on(printer, "T1", {"type": "text_delta", "text": "x"})

        assert len(printer.emitted) == 3
        assert printer.emitted[0].get("tabId") == "T1", (
            "Primary event MUST be written before fan-out copies."
        )
