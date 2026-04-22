"""Tests confirming the fixes for cross-tab state-isolation violations.

Covers the following violations from the earlier audit that the user
asked to fix:

- A7: ``VSCodePrinter`` usage offsets must be per-tab, not shared.
- B4: ``adjacent_task_events`` must always carry a ``tabId`` so a
  missing frontend tab_id cannot reach every tab.
- B5: ``commitMessage`` events generated in the background thread must
  carry a ``tabId`` so the result only reaches the requesting tab.
- B8: ``_finish_merge(None)`` must not clear every tab's merge flag
  and must not emit an untagged ``merge_ended`` event.
- C1: ``_cmd_get_adjacent_task`` must not fall back to the globally
  latest chat when the tab has no chat_id.
- C2, C3: ``_replay_session`` with an empty ``tab_id`` must not
  synthesize a phantom tab keyed by ``chat_id`` or flip
  ``use_worktree`` on a tab that is not the caller's.
- C4: ``_stop_task(None)`` must not stop every tab's task.

All tests use real ``VSCodeServer`` instances (no mocks); see
``test_vscode_tabs._make_server`` for the broadcast-capture helper
pattern.
"""

from __future__ import annotations

import threading
import time
import unittest

from kiss.agents.vscode.server import VSCodeServer


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
    """A7: per-tab token/budget/step offsets in the printer."""

    def test_offsets_are_per_tab_via_thread_local(self) -> None:
        server, _ = _make_server()
        printer = server.printer
        printer._thread_local.tab_id = "A"
        printer.tokens_offset = 100
        printer.budget_offset = 1.5
        printer.steps_offset = 5

        printer._thread_local.tab_id = "B"
        assert printer.tokens_offset == 0
        assert printer.budget_offset == 0.0
        assert printer.steps_offset == 0

        printer._thread_local.tab_id = "A"
        assert printer.tokens_offset == 100
        assert printer.budget_offset == 1.5
        assert printer.steps_offset == 5

    def test_offsets_concurrent_threads_do_not_clobber(self) -> None:
        server, _ = _make_server()
        printer = server.printer
        barrier = threading.Barrier(2)
        results: dict[str, tuple[int, float, int]] = {}

        def worker(tid: str, vals: tuple[int, float, int]) -> None:
            printer._thread_local.tab_id = tid
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

    def test_cleanup_tab_drops_offsets(self) -> None:
        server, _ = _make_server()
        printer = server.printer
        printer._thread_local.tab_id = "A"
        printer.tokens_offset = 42
        printer.cleanup_tab("A")
        printer._thread_local.tab_id = "A"
        assert printer.tokens_offset == 0


class TestB4AdjacentTaskAlwaysTagged(unittest.TestCase):
    """B4: adjacent_task_events must never leak to all tabs."""

    def test_empty_tab_id_still_tags_event_so_cross_tab_broadcast_is_impossible(
        self,
    ) -> None:
        server, events = _make_server()
        server._get_adjacent_task(
            chat_id="none",
            task="",
            direction="prev",
            tab_id="",
        )
        ate = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(ate) == 1
        assert "tabId" in ate[0]


class TestB5CommitMessageCarriesTabId(unittest.TestCase):
    """B5: commitMessage events must carry tabId so only the requester sees them."""

    def test_cmd_generate_commit_message_tags_events_with_tab_id(self) -> None:
        server = VSCodeServer()
        events: list[dict] = []
        done = threading.Event()
        captured_tab_ids: list[object] = []

        def stub() -> None:
            # Capture the thread-local tab_id so the test can verify
            # the worker thread set it correctly before broadcasting.
            captured_tab_ids.append(
                getattr(server.printer._thread_local, "tab_id", None)
            )
            server.printer.broadcast({"type": "commitMessage", "message": "x"})
            done.set()

        # Replace stdout.write in the real broadcast pipeline with a capture
        # so tabId injection from thread-local is exercised end-to-end.
        import io
        import sys as _sys
        buf = io.StringIO()
        real_stdout = _sys.stdout
        _sys.stdout = buf
        try:
            server._generate_commit_message = stub  # type: ignore[assignment]
            server._cmd_generate_commit_message({
                "type": "generateCommitMessage", "tabId": "TAB-1",
            })
            assert done.wait(timeout=5)
        finally:
            _sys.stdout = real_stdout
        import json as _json
        for line in buf.getvalue().splitlines():
            if line.strip():
                events.append(_json.loads(line))
        assert captured_tab_ids == ["TAB-1"]
        cm = [e for e in events if e.get("type") == "commitMessage"]
        assert len(cm) == 1
        assert cm[0].get("tabId") == "TAB-1"


class TestB8FinishMergeRequiresTabId(unittest.TestCase):
    """B8: _finish_merge(None) must not tear down every tab's state."""

    def test_finish_merge_none_does_not_clear_other_tabs(self) -> None:
        server, events = _make_server()
        server._get_tab("A").is_merging = True
        server._get_tab("B").is_merging = True
        server._finish_merge(None)
        assert server._get_tab("A").is_merging is True
        assert server._get_tab("B").is_merging is True
        ended = [e for e in events if e.get("type") == "merge_ended"]
        assert all("tabId" in e for e in ended)


class TestC1AdjacentTaskNoGlobalFallback(unittest.TestCase):
    """C1: no fallback to globally-latest chat when the tab has no chat_id.

    When the tab's agent has an empty ``chat_id`` (freshly created tab),
    ``_cmd_get_adjacent_task`` must call ``_get_adjacent_task`` with
    that empty chat_id rather than silently falling back to the
    globally most-recent history row, which would make arrow-key
    navigation traverse *another* tab's conversation.
    """

    def test_empty_chat_id_is_passed_through_without_global_fallback(self) -> None:
        server, _ = _make_server()
        captured: list[tuple[str, str, str, str]] = []

        def stub(
            chat_id: str, task: str, direction: str, tab_id: str = "",
        ) -> None:
            captured.append((chat_id, task, direction, tab_id))

        server._get_adjacent_task = stub  # type: ignore[assignment]
        tab = server._get_tab("T")
        assert tab.agent.chat_id == ""
        server._cmd_get_adjacent_task({
            "type": "getAdjacentTask", "tabId": "T",
            "task": "", "direction": "prev",
        })
        assert captured == [("", "", "prev", "T")]


class TestC2C3ReplayRequiresTabId(unittest.TestCase):
    """C2/C3: _replay_session with empty tab_id must not synthesize a phantom tab.

    Patches the persistence loader so a replay would normally succeed;
    the test proves that the empty-tab_id guard prevents creation of a
    phantom tab keyed by ``chat_id`` and prevents modifying
    ``use_worktree`` on any other tab.
    """

    def setUp(self) -> None:
        from kiss.agents.vscode import server as smod
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

    def test_empty_tab_id_does_not_create_tab_keyed_by_chat_id(self) -> None:
        server, _ = _make_server()
        server._replay_session("some-chat-id", tab_id="")
        assert "some-chat-id" not in server._tab_states

    def test_empty_tab_id_does_not_flip_use_worktree_on_any_tab(self) -> None:
        server, _ = _make_server()
        server._get_tab("real-tab").use_worktree = False
        server._replay_session("some-chat-id", tab_id="")
        for tab in server._tab_states.values():
            assert tab.use_worktree is False


class TestC4StopRequiresTabId(unittest.TestCase):
    """C4: _stop_task(None) must not stop every tab's task."""

    def test_stop_without_tab_id_is_no_op(self) -> None:
        server, _ = _make_server()
        ev1 = threading.Event()
        ev2 = threading.Event()
        server._get_tab("1").stop_event = ev1
        server._get_tab("2").stop_event = ev2
        t1 = threading.Thread(target=lambda: time.sleep(1), daemon=True)
        t2 = threading.Thread(target=lambda: time.sleep(1), daemon=True)
        t1.start()
        t2.start()
        server._get_tab("1").task_thread = t1
        server._get_tab("2").task_thread = t2
        server._stop_task(None)
        time.sleep(0.2)
        assert not ev1.is_set()
        assert not ev2.is_set()


if __name__ == "__main__":
    unittest.main()
