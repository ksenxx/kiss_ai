# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the area-D redundancy/race-condition fixes.

Covers (real :class:`AgentServer`, real :class:`AgentState` registry,
real threads, real on-disk git repos and worktrees — no mocks, patches
or fakes):

- D-RC1: ``AgentState.running`` (the predicate ``appendUserMessage``
  uses) must treat a created-but-not-yet-started worker thread as
  live (``Thread.is_alive()`` is False before ``start()``), so an
  ``appendUserMessage`` arriving in the ``_cmd_run`` startup window
  is queued instead of dropped.
- D-RC2: ``TabRegistry.open_tab`` decides opened/exists/full
  atomically under the registry lock; ``_cmd_open_tab`` no longer
  re-probes ``has_tab`` unlocked, so a concurrent ``closeTab`` can
  never turn a benign re-announce into a spurious "Tab limit reached"
  rejection.
- D-RC3 (obsolete): the worktree-occupancy check and task-admission
  gate were removed together with the main-tree claims; a worktree
  now exists only through the ``create_worktree`` tool.
- D-R1: ``_cmd_set_work_dir`` and ``_cmd_save_config`` share one
  work-dir update implementation (``_apply_new_work_dir``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.file_index import FileIndex, FileIndexRegistry
from kiss.server.server import AgentServer
from kiss.server.tab_registry import _MAX_TABS, OpenTabOutcome, TabRegistry

from ._memory_printer import MemoryPrinter


def _noop() -> None:
    """Body for worker threads that must exist but do nothing."""


class TestDRC1TaskAcceptsInputStartupWindow:
    """A created-but-unstarted worker thread must count as a live task."""

    def test_created_but_unstarted_thread_accepts_input(self) -> None:
        """The exact S3-05 window: ``_cmd_run`` installs ``task_thread``
        and broadcasts BEFORE ``thread.start()``; ``is_alive()`` is
        False then, but the task is real and must accept input."""
        state = AgentState("rr-d-rc1-a", tab_id="rr-d-rc1-a-tab")
        state.task_thread = threading.Thread(target=_noop, daemon=True)
        assert state.task_thread.is_alive() is False, (
            "precondition: an unstarted thread reports is_alive()=False"
        )
        with agent_state.STATE_LOCK:
            assert state.running() is True, (
                "a follow-up typed during the run-startup window was "
                "treated as addressed to a dead task and dropped"
            )

    def test_finished_and_active_states(self) -> None:
        """Branch coverage: finished-thread and active states."""
        finished = AgentState("rr-d-rc1-b", tab_id="rr-d-rc1-b-tab")
        thread = threading.Thread(target=_noop, daemon=True)
        thread.start()
        thread.join()
        finished.task_thread = thread
        with agent_state.STATE_LOCK:
            assert finished.running() is False, (
                "a started-and-finished worker thread is a dead task"
            )

        active = AgentState("rr-d-rc1-c", tab_id="rr-d-rc1-c-tab")
        active.is_task_active = True
        with agent_state.STATE_LOCK:
            assert active.running() is True

    def test_append_user_message_queued_during_startup_window(self) -> None:
        """End-to-end: ``appendUserMessage`` from another connection in
        the startup window lands in ``pending_user_messages``."""
        server = AgentServer(printer=MemoryPrinter())
        state = AgentState(
            "rr-d-rc1-e2e", tab_id="rr-d-rc1-e2e-tab", server_owned=True,
        )
        state.task_thread = threading.Thread(target=_noop, daemon=True)
        agent_state.register(state)
        try:
            server._cmd_append_user_message({
                "tabId": "rr-d-rc1-e2e-tab",
                "prompt": "follow-up during startup",
            })
            assert state.pending_user_messages == [
                "follow-up during startup",
            ], "the startup-window follow-up was dropped or misrouted"
        finally:
            agent_state.unregister(state.task_id, state)


class TestDRC2OpenTabTriState:
    """``open_tab`` decides opened/exists/full atomically."""

    def test_tristate_outcomes_and_truthiness(self, tmp_path: Path) -> None:
        reg = TabRegistry(tmp_path / "tabs.json")
        opened = reg.open_tab("t1", "one")
        assert opened is OpenTabOutcome.OPENED
        assert bool(opened) is True, (
            "OPENED must stay truthy for legacy boolean callers"
        )
        exists = reg.open_tab("t1", "one again")
        assert exists is OpenTabOutcome.EXISTS
        assert bool(exists) is False
        assert reg.open_tab("") is OpenTabOutcome.EXISTS, (
            "a blank id opens nothing but must not be reported as FULL"
        )
        for i in range(_MAX_TABS - 1):
            assert reg.open_tab(f"fill-{i}") is OpenTabOutcome.OPENED
        full = reg.open_tab("overflow")
        assert full is OpenTabOutcome.FULL
        assert bool(full) is False
        # The registry really is unchanged by the refusal.
        assert not reg.has_tab("overflow")

    def test_full_registry_rejects_unknown_tab(self, tmp_path: Path) -> None:
        """The FULL outcome still produces ``openTabRejected``."""
        printer = MemoryPrinter()
        server = AgentServer(printer=printer)
        server.tab_registry = TabRegistry(tmp_path / "tabs.json")
        for i in range(_MAX_TABS):
            assert server.tab_registry.open_tab(f"pre-{i}")
        printer.emitted.clear()
        server._cmd_open_tab({"tabId": "brand-new", "connId": "c1"})
        rejected = [
            e for e in printer.emitted if e.get("type") == "openTabRejected"
        ]
        assert len(rejected) == 1
        assert rejected[0]["tabId"] == "brand-new"

    def test_reannounce_never_rejected_despite_concurrent_close(
        self, tmp_path: Path,
    ) -> None:
        """D-RC2 end-to-end: with the registry AT its cap and a client
        re-announcing a tab that exists, a concurrent ``closeTab`` of
        that same tab must never surface as ``openTabRejected`` — the
        outcome is decided atomically, so every interleaving yields
        EXISTS (still registered) or OPENED (close freed a slot)."""
        printer = MemoryPrinter()
        server = AgentServer(printer=printer)
        server.tab_registry = TabRegistry(tmp_path / "tabs.json")
        for i in range(_MAX_TABS - 1):
            assert server.tab_registry.open_tab(f"cap-{i}")
        assert server.tab_registry.open_tab("racy-tab")

        for _ in range(30):
            printer.emitted.clear()
            closer = threading.Thread(
                target=server.tab_registry.close_tab, args=("racy-tab",),
            )
            closer.start()
            server._cmd_open_tab({"tabId": "racy-tab", "connId": "c1"})
            closer.join()
            rejected = [
                e for e in printer.emitted
                if e.get("type") == "openTabRejected"
            ]
            assert rejected == [], (
                "a concurrent closeTab turned a re-announce of an "
                "existing tab into a spurious 'Tab limit reached'"
            )
            # Re-arm: make sure the tab is registered again for the
            # next round (the open may have raced ahead of the close).
            server.tab_registry.close_tab("racy-tab")
            assert server.tab_registry.open_tab("racy-tab")


def _wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll *predicate* until it holds or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def _private_registry(server: AgentServer, tmp_path: Path) -> FileIndexRegistry:
    """Replace the server's file-index registry with one rooted in *tmp_path*."""
    server._file_index.stop()
    registry = FileIndexRegistry(
        home=str(tmp_path / "home"), cache_dir=tmp_path / "cache",
    )
    server._file_index = registry
    return registry


def _drain(registry: FileIndexRegistry, tmp_path: Path) -> None:
    """Block until every build queued so far on *registry* has run.

    The registry's single worker serves jobs in FIFO order, so a
    sentinel job on an unrelated directory completing proves the
    earlier jobs (or their absence) have been decided.
    """
    done = threading.Event()
    registry.ensure(str(tmp_path / "sentinel"), done.set)
    assert done.wait(10.0), "the file-index worker never ran the sentinel"


def _index_of(registry: FileIndexRegistry, work_dir: str) -> FileIndex | None:
    root, _ = registry.root_for(work_dir)
    with registry._lock:
        return registry._indexes.get(root)



class TestDR1SharedWorkDirUpdate:
    """Both work-dir update paths share ``_apply_new_work_dir``.

    Adopting a different directory pre-warms its ``@``-mention index
    (``_file_index.ensure(new_dir)``); re-announcing the current one or
    naming a filesystem root queues nothing.
    """

    def test_set_work_dir_updates_state(self, tmp_path: Path) -> None:
        server = AgentServer(printer=MemoryPrinter())
        registry = _private_registry(server, tmp_path)
        try:
            # ``work_dir`` is an optional printer attribute the update
            # mirrors onto when present (hasattr-guarded in production).
            setattr(server.printer, "work_dir", "")
            new_dir = tmp_path / "workspace-a"
            new_dir.mkdir()
            (new_dir / "a.py").write_text("a = 1\n")
            server._cmd_set_work_dir({"workDir": str(new_dir), "connId": "c1"})
            assert server.work_dir == str(new_dir)
            assert getattr(server.printer, "work_dir") == str(new_dir)
            _wait_for(lambda: registry.view_for(str(new_dir)) is not None)
            index = _index_of(registry, str(new_dir))
            assert index is not None and "a.py" in index.paths, (
                "changing the work dir must pre-warm the new directory's index"
            )

            # Unchanged dir: no build is queued, the index stays as is.
            server._cmd_set_work_dir({"workDir": str(new_dir), "connId": "c1"})
            _drain(registry, tmp_path)
            assert _index_of(registry, str(new_dir)) is index, (
                "re-announcing the same work dir must not rebuild its index"
            )

            # A filesystem root is refused: nothing adopted, nothing queued.
            server._cmd_set_work_dir({"workDir": "/", "connId": "c1"})
            assert server.work_dir == str(new_dir)
            _drain(registry, tmp_path)
            assert registry.home not in registry._indexes, (
                "a refused root must not queue an index build of the home dir"
            )
        finally:
            registry.stop()

    def test_save_config_updates_state_identically(
        self, tmp_path: Path,
    ) -> None:
        server = AgentServer(printer=MemoryPrinter())
        registry = _private_registry(server, tmp_path)
        try:
            setattr(server.printer, "work_dir", "")
            new_dir = tmp_path / "workspace-b"
            new_dir.mkdir()
            (new_dir / "b.py").write_text("b = 1\n")
            server._cmd_save_config({
                "config": {"work_dir": str(new_dir)},
                "apiKeys": {},
                "connId": "c1",
            })
            assert server.work_dir == str(new_dir)
            assert getattr(server.printer, "work_dir") == str(new_dir)
            _wait_for(lambda: registry.view_for(str(new_dir)) is not None)
            index = _index_of(registry, str(new_dir))
            assert index is not None and "b.py" in index.paths
        finally:
            registry.stop()
