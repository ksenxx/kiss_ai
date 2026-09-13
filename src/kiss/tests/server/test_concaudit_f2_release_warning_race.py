# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_release_worktree_without_merging`` must prefix the pending warning atomically.

``kiss.server.task_runner._release_worktree_without_merging`` used to
read ``agent._merge_conflict_warning`` under ``_warning_lock``, release
the lock, and write ``reason + old`` back through ``_set_warnings``.  A
``_flush_warnings`` landing in the gap (the agent thread starting
``run()`` while the server releases the previous worktree) took and
broadcast ``old``; the write then put it back and the user saw the
"work is in the worktree directory" warning twice.

This test drives the real production path with a real git repository
and worktree: Auto-commit is off and the worktree has an uncommitted
file, so ``_preserve_pending_worktree_for_review`` keeps the directory
and records the "Auto-commit is disabled ..." warning, which the
release then has to prefix with the blocked-merge reason.  Concurrent
flushes race the release from the moment the preserve outcome is
recorded; the preserve warning must be broadcast exactly once.

The exact flush-between-read-and-write interleaving is not forced
here: the old code had no observable state between its read and its
write to synchronise on, and a spinning flusher takes the preserve
warning before the release even reaches the read.  This test therefore
exercises the fixed production path end-to-end; the atomicity of the
combine itself is hammered (and shown to catch a split read/write) in
``test_concaudit_f2_warning_races.py``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _WorktreeCleanupOutcome,
)
from kiss.server.task_runner import _release_worktree_without_merging
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-concaudit-f2-release-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _RecordingPrinter:
    """Minimal broadcast sink: collects every warning message."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.messages.append(str(event.get("message", "")))


def test_preserve_warning_is_delivered_once_under_concurrent_flush(
    env: IsolatedKissHome,
) -> None:
    """Concurrent flushes never re-deliver the warning the release prefixes."""
    agent = WorktreeSorcarAgent("f2-release-race")
    agent.auto_commit_enabled = False
    assert agent._try_setup_worktree(env.repo, str(env.repo)) is not None
    wt = agent._wt
    assert wt is not None
    (wt.wt_dir / "agent.txt").write_text("agent work\n", encoding="utf-8")
    branch = agent._wt_branch
    assert branch

    printer = _RecordingPrinter()
    done = threading.Event()

    def flush_from_preserve_on() -> None:
        # ``_last_preserve_outcome`` is recorded right before the
        # preserve warning is stored, so flushing from that moment on
        # overlaps the release's read-modify-write of that warning.
        while agent._last_preserve_outcome is None and not done.is_set():
            pass
        while not done.is_set():
            agent._flush_warnings(printer)

    def release() -> None:
        try:
            _release_worktree_without_merging(agent, True)
        finally:
            done.set()

    releaser = threading.Thread(target=release)
    flushers = [threading.Thread(target=flush_from_preserve_on) for _ in range(4)]
    releaser.start()
    for t in flushers:
        t.start()
    for t in [releaser, *flushers]:
        t.join(timeout=120)
        assert not t.is_alive()
    agent._flush_warnings(printer)

    assert agent._last_preserve_outcome is (
        _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT
    )
    assert wt.wt_dir.exists(), "worktree with the only copy of the work removed"
    joined = "\n".join(printer.messages)
    preserve = "Auto-commit is disabled"
    reason = f"Could not auto-merge branch '{branch}'"
    assert joined.count(preserve) == 1, printer.messages
    assert joined.count(reason) == 1, printer.messages
    assert str(wt.wt_dir) in joined, printer.messages
    # A flush may legitimately take the preserve warning before the
    # release prefixes it (then both arrive as separate messages, each
    # once); when they arrive together, the reason comes first.
    with_reason = next(m for m in printer.messages if reason in m)
    if preserve in with_reason:
        assert with_reason.index(reason) < with_reason.index(preserve), with_reason
