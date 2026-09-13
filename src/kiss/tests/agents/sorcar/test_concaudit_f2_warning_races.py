# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pending-warning races in ``WorktreeSorcarAgent`` (F2 findings 1 and 4).

Finding 1 — ``add_warning`` must combine with the pending merge warning
under ONE hold of ``_warning_lock``.  The old inline code in
``task_runner._release_worktree_without_merging`` read the slot under
the lock, released it, and wrote ``reason + old`` back through
``_set_warnings``.  A ``_flush_warnings`` landing in the gap took and
broadcast ``old``; the write then put it back, so ``old`` reached the
user twice.  A ``_set_warnings(B)`` landing in the gap was overwritten.

Finding 4 — ``_flush_warnings`` clears the slot, then broadcasts outside
the lock.  When the broadcast raised, it restored the taken warning
only if the slot was still empty, so a warning set concurrently while
the broadcast was failing made the taken one vanish: neither delivered
nor retained.  Both the stash and the merge slot had the defect.

These tests use real threads.  The finding-4 tests force the exact
interleaving with a printer that blocks inside ``broadcast`` until the
concurrent write has landed, then raises.  The finding-1 tests hammer
the fixed read-modify-write with concurrent flushes and writers.
"""

from __future__ import annotations

import threading
from typing import Any

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


class _RecordingPrinter:
    """Minimal broadcast sink: collects every warning message."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.messages.append(str(event.get("message", "")))


def _lines(messages: list[str]) -> list[str]:
    """Split every broadcast message into its newline-joined parts."""
    return [line for msg in messages for line in msg.split("\n")]


class _BlockingBrokenPrinter:
    """Blocks inside ``broadcast`` until released, then raises."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def broadcast(self, event: dict[str, Any]) -> None:
        self.entered.set()
        assert self.release.wait(30)
        raise RuntimeError("printer is broken")


def test_add_warning_prepend_under_concurrent_flush_delivers_old_once() -> None:
    """The pending warning is broadcast exactly once around a prepend."""
    agent = WorktreeSorcarAgent("f2-add-warning-race")
    printer = _RecordingPrinter()
    stop = threading.Event()

    def flusher() -> None:
        while not stop.is_set():
            agent._flush_warnings(printer)

    flushers = [threading.Thread(target=flusher) for _ in range(4)]
    for t in flushers:
        t.start()
    try:
        for i in range(300):
            # ``_set_warnings`` replaces the slot, so only store the
            # round's "old" warning once the previous round was flushed.
            while agent._merge_conflict_warning is not None:
                pass
            agent._set_warnings(merge=f"OLD-{i}")
            agent.add_warning(f"REASON-{i}", prepend=True)
    finally:
        stop.set()
        for t in flushers:
            t.join(timeout=30)
            assert not t.is_alive()
    agent._flush_warnings(printer)

    lines = _lines(printer.messages)
    for i in range(300):
        assert lines.count(f"OLD-{i}") == 1, (i, printer.messages)
        assert lines.count(f"REASON-{i}") == 1, (i, printer.messages)
    # Whenever both parts of one round were flushed together, the
    # reason came first.
    for msg in printer.messages:
        if "\n" in msg:
            first, second = msg.split("\n", 1)
            assert first.startswith("REASON-"), msg
            assert second.startswith("OLD-"), msg


def test_add_warning_never_loses_a_concurrent_set() -> None:
    """A ``_set_warnings`` racing ``add_warning`` is never overwritten."""
    agent = WorktreeSorcarAgent("f2-add-warning-vs-set")
    printer = _RecordingPrinter()
    n = 500
    go = threading.Barrier(2)

    def adder() -> None:
        go.wait()
        for i in range(n):
            agent.add_warning(f"ADD-{i}")

    def setter() -> None:
        go.wait()
        for i in range(n):
            agent.add_warning(f"SET-{i}", prepend=True)

    threads = [threading.Thread(target=adder), threading.Thread(target=setter)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive()
    agent._flush_warnings(printer)
    lines = _lines(printer.messages)
    for i in range(n):
        assert lines.count(f"ADD-{i}") == 1, i
        assert lines.count(f"SET-{i}") == 1, i


def test_add_warning_on_empty_slot_sets_text() -> None:
    """With nothing pending, both orders store *text* alone."""
    agent = WorktreeSorcarAgent("f2-add-warning-empty")
    agent.add_warning("A")
    assert agent._merge_conflict_warning == "A"
    agent._set_warnings(merge="")
    agent.add_warning("B", prepend=True)
    assert agent._merge_conflict_warning == "B"
    agent.add_warning("C")
    assert agent._merge_conflict_warning == "B\nC"


def _flush_fails_while_slot_is_rewritten(
    agent: WorktreeSorcarAgent, slot: str, taken: str, concurrent: str,
) -> None:
    """Run a failing flush of *taken* while *concurrent* lands in *slot*."""
    agent._set_warnings(**{slot: taken})
    printer = _BlockingBrokenPrinter()
    flusher = threading.Thread(target=agent._flush_warnings, args=(printer,))
    flusher.start()
    try:
        assert printer.entered.wait(30)
        # The flush has taken *taken* and is stuck in the broadcast.
        agent._set_warnings(**{slot: concurrent})
    finally:
        printer.release.set()
        flusher.join(timeout=30)
        assert not flusher.is_alive()


def test_failed_merge_broadcast_keeps_both_old_and_concurrent_warning() -> None:
    """Finding 4, merge slot: the taken warning is prepended, not dropped."""
    agent = WorktreeSorcarAgent("f2-flush-restore-merge")
    _flush_fails_while_slot_is_rewritten(agent, "merge", "OLD-MERGE", "NEW-MERGE")
    assert agent._merge_conflict_warning == "OLD-MERGE\nNEW-MERGE"
    good = _RecordingPrinter()
    agent._flush_warnings(good)
    assert good.messages == ["OLD-MERGE\nNEW-MERGE"]


def test_failed_stash_broadcast_keeps_both_old_and_concurrent_warning() -> None:
    """Finding 4, stash slot: the taken warning is prepended, not dropped."""
    agent = WorktreeSorcarAgent("f2-flush-restore-stash")
    _flush_fails_while_slot_is_rewritten(agent, "stash", "OLD-STASH", "NEW-STASH")
    assert agent._stash_pop_warning == "OLD-STASH\nNEW-STASH"
    good = _RecordingPrinter()
    agent._flush_warnings(good)
    assert good.messages == ["OLD-STASH\nNEW-STASH"]


def test_failed_broadcast_with_empty_slot_restores_plain_warning() -> None:
    """Without a concurrent write, the failed warning is put back as is."""
    agent = WorktreeSorcarAgent("f2-flush-restore-plain")
    agent._set_warnings(stash="S", merge="M")

    class _Broken:
        def broadcast(self, event: dict[str, Any]) -> None:
            raise RuntimeError("printer is broken")

    agent._flush_warnings(_Broken())
    assert agent._stash_pop_warning == "S"
    assert agent._merge_conflict_warning == "M"
