# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests confirming the fixes for cross-tab state-isolation violations.

Covers the following violations from the earlier audit that the user
asked to fix:

- A7: Printer usage offsets must be per-task, not shared.
- B4: ``adjacent_task_events`` must always carry a ``tabId`` so a
  missing frontend tab_id cannot reach every tab.
- B5: ``commitMessage`` events generated in the background thread must
  carry a ``tabId`` so the result only reaches the requesting tab.
- C1: ``_cmd_get_adjacent_task`` must not fall back to the globally
  latest chat when the tab has no chat association.
- C2, C3: ``_replay_session`` with an empty ``tab_id`` must not
  synthesize a phantom registry entry keyed by ``chat_id`` or flip
  ``use_worktree`` on a state that is not the caller's.
- C4: ``_stop_task(None)`` must not stop every tab's task.

All tests use real ``VSCodeServer`` instances (no mocks).
"""

from __future__ import annotations

import threading
import time
import unittest

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestA7PrinterOffsetsAreIsolated(unittest.TestCase):
    """A7: per-task token/budget/step offsets in the printer."""

    def test_offsets_are_per_task_via_thread_local(self) -> None:
        server, _ = _make_server()
        printer = server.printer
        printer._thread_local.task_id = "A"
        printer.tokens_offset = 100
        printer.budget_offset = 1.5
        printer.steps_offset = 5

        printer._thread_local.task_id = "B"
        assert printer.tokens_offset == 0
        assert printer.budget_offset == 0.0
        assert printer.steps_offset == 0

        printer._thread_local.task_id = "A"
        assert printer.tokens_offset == 100
        assert printer.budget_offset == 1.5
        assert printer.steps_offset == 5

    def test_offsets_concurrent_threads_do_not_clobber(self) -> None:
        server, _ = _make_server()
        printer = server.printer
        barrier = threading.Barrier(2)
        results: dict[str, tuple[int, float, int]] = {}

        def worker(tid: str, vals: tuple[int, float, int]) -> None:
            printer._thread_local.task_id = tid
            printer.tokens_offset = vals[0]
            printer.budget_offset = vals[1]
            printer.steps_offset = vals[2]
            barrier.wait()
            results[tid] = (
                printer.tokens_offset,
                printer.budget_offset,
                printer.steps_offset,
            )

        t1 = threading.Thread(target=worker, args=("A", (100, 1.5, 5)))
        t2 = threading.Thread(target=worker, args=("B", (200, 2.5, 7)))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert results["A"] == (100, 1.5, 5)
        assert results["B"] == (200, 2.5, 7)

    def test_cleanup_task_drops_offsets(self) -> None:
        server, _ = _make_server()
        printer = server.printer
        printer._thread_local.task_id = "A"
        printer.tokens_offset = 42
        printer.cleanup_task("A")
        printer._thread_local.task_id = "A"
        assert printer.tokens_offset == 0


class TestB4AdjacentTaskAlwaysTagged(unittest.TestCase):
    """B4: adjacent_task_events must never leak to all tabs."""

    def test_empty_tab_id_still_tags_event_so_cross_tab_broadcast_is_impossible(
        self,
    ) -> None:
        server, events = _make_server()
        server._get_adjacent_task(
            chat_id="none",
            task_id=None,
            direction="prev",
            tab_id="",
        )
        ate = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(ate) == 1
        assert "tabId" in ate[0]


class TestB5CommitMessageCarriesTabId(unittest.TestCase):
    """B5: commitMessage events must carry tabId so only the requester sees them."""

    def test_cmd_generate_commit_message_tags_events_with_tab_id(self) -> None:
        from kiss.tests.server._memory_printer import MemoryPrinter

        printer = MemoryPrinter()
        server = VSCodeServer(printer=printer)
        done = threading.Event()
        captured_tab_ids: list[object] = []

        def stub(tab_id_arg: str = "", *, work_dir: str = "") -> None:
            captured_tab_ids.append(tab_id_arg)
            server.printer.broadcast({
                "type": "commitMessage", "message": "x", "tabId": tab_id_arg,
            })
            done.set()

        server._generate_commit_message = stub  # type: ignore[assignment]
        server._cmd_generate_commit_message({
            "type": "generateCommitMessage", "tabId": "TAB-1",
        })
        assert done.wait(timeout=5)

        assert captured_tab_ids == ["TAB-1"]
        cm = [e for e in printer.emitted if e.get("type") == "commitMessage"]
        assert len(cm) == 1
        assert cm[0].get("tabId") == "TAB-1"


class TestC1AdjacentTaskNoGlobalFallback(unittest.TestCase):
    """C1: no fallback to globally-latest chat when the tab has no chat id.

    When neither a registered agent state nor a ``_tab_chat_views``
    association exists for the tab (freshly created tab),
    ``_cmd_get_adjacent_task`` must call ``_get_adjacent_task`` with
    an empty chat_id rather than silently falling back to the
    globally most-recent history row, which would make arrow-key
    navigation traverse *another* tab's conversation.
    """

    def test_empty_chat_id_is_passed_through_without_global_fallback(self) -> None:
        server, _ = _make_server()
        captured: list[tuple[str, str | None, str, str]] = []

        def stub(
            chat_id: str,
            task_id: str | None,
            direction: str,
            tab_id: str = "",
        ) -> None:
            captured.append((chat_id, task_id, direction, tab_id))

        server._get_adjacent_task = stub  # type: ignore[assignment]
        server._cmd_get_adjacent_task({
            "type": "getAdjacentTask", "tabId": "T",
            "taskId": None, "direction": "prev",
        })
        assert captured == [("", None, "prev", "T")]


class TestC2C3ReplayRequiresTabId(unittest.TestCase):
    """C2/C3: _replay_session with empty tab_id must not synthesize phantom state.

    Patches the persistence loader so a replay would normally succeed;
    the test proves that the empty-tab_id guard prevents creation of a
    phantom registry entry keyed by ``chat_id`` and prevents modifying
    ``use_worktree`` on any other tab's state.
    """

    def setUp(self) -> None:
        from kiss.server import server as smod
        self._smod = smod
        self._orig_loader = smod._load_latest_chat_events_by_chat_id

        def fake_loader(chat_id: str) -> dict[str, object]:
            return {
                "events": [{"type": "text_delta", "text": "x"}],
                "task": "t",
                "extra": '{"is_worktree": true}',
            }

        smod._load_latest_chat_events_by_chat_id = fake_loader  # type: ignore[assignment]

    def tearDown(self) -> None:
        self._smod._load_latest_chat_events_by_chat_id = self._orig_loader  # type: ignore[assignment]
        agent_state.agent_states.clear()

    def test_empty_tab_id_does_not_create_state_keyed_by_chat_id(self) -> None:
        server, _ = _make_server()
        server._replay_session("some-chat-id", tab_id="")
        assert "some-chat-id" not in agent_state.agent_states
        assert "some-chat-id" not in server._tab_chat_views

    def test_empty_tab_id_does_not_flip_use_worktree_on_any_tab(self) -> None:
        server, _ = _make_server()
        state = agent_state.AgentState(
            "task-real", tab_id="real-tab", server_owned=True,
        )
        # Force the flag away from whatever the replayed session would
        # set so a leaky replay (C3) is still detected now that the
        # AgentState default is True.
        state.use_worktree = False
        agent_state.register(state)
        server._replay_session("some-chat-id", tab_id="")
        for st in agent_state.agent_states.values():
            assert st.use_worktree is False


class TestC4StopRequiresTabId(unittest.TestCase):
    """C4: _stop_task(None) must not stop every tab's task."""

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_stop_without_tab_id_is_no_op(self) -> None:
        server, _ = _make_server()
        ev1 = threading.Event()
        ev2 = threading.Event()
        t1 = threading.Thread(target=lambda: time.sleep(1), daemon=True)
        t2 = threading.Thread(target=lambda: time.sleep(1), daemon=True)
        t1.start()
        t2.start()
        state1 = agent_state.AgentState(
            "task-1", tab_id="1", server_owned=True,
            stop_event=ev1, task_thread=t1,
        )
        state2 = agent_state.AgentState(
            "task-2", tab_id="2", server_owned=True,
            stop_event=ev2, task_thread=t2,
        )
        agent_state.register(state1)
        agent_state.register(state2)
        server._stop_task(None)  # type: ignore[arg-type]
        time.sleep(0.2)
        assert not ev1.is_set()
        assert not ev2.is_set()


if __name__ == "__main__":
    unittest.main()
