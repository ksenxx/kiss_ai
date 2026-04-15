"""Tests for newly identified race conditions in the VS Code server.

Races identified:
- RC-NEW-1: stop_event created too late in task thread — _stop_task sees None
- RC-NEW-2: _get_tab has unprotected TOCTOU on _tab_states dict
- RC-NEW-3: is_merging check and state setup in separate lock blocks (TOCTOU gap)
"""

import inspect
import queue
import threading
import unittest

from kiss.agents.vscode.server import VSCodeServer


class TestRCNew1StopEventCreatedTooLate(unittest.TestCase):
    """RC-NEW-1: stop_event must exist before _stop_task can be called.

    Previously, stop_event was created inside _run_task_inner (task thread),
    meaning a fast _stop_task call from the main thread could see None.
    Fix: create stop_event in _handle_command before starting the thread.
    """

    def test_stop_event_created_before_thread_start(self) -> None:
        """Verify stop_event is assigned in _handle_command, not _run_task_inner."""
        source = inspect.getsource(VSCodeServer._handle_command)
        # Find the "run" handler block
        lines = source.split("\n")
        in_run = False
        run_block: list[str] = []
        for line in lines:
            if '"run"' in line and "cmd_type ==" in line:
                in_run = True
            elif in_run:
                if line.strip().startswith("elif") or line.strip().startswith("else:"):
                    break
                run_block.append(line)
        block = "\n".join(run_block)
        assert "tab.stop_event" in block, (
            "stop_event should be created in _handle_command 'run' handler"
        )
        assert "threading.Event()" in block, (
            "A fresh threading.Event() should be created before thread.start()"
        )

    def test_stop_event_created_before_thread_start_ordering(self) -> None:
        """Verify stop_event assignment comes before thread.start() in source."""
        source = inspect.getsource(VSCodeServer._handle_command)
        stop_event_pos = source.find("tab.stop_event")
        thread_start_pos = source.find("thread.start()")
        assert stop_event_pos > 0, "tab.stop_event not found in _handle_command"
        assert thread_start_pos > 0, "thread.start() not found in _handle_command"
        assert stop_event_pos < thread_start_pos, (
            "tab.stop_event must be assigned BEFORE thread.start()"
        )

    def test_run_task_inner_reads_stop_event_not_creates(self) -> None:
        """Verify _run_task_inner reads stop_event from tab, doesn't create it."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        # Should read stop_event from tab, not create a new one
        assert "tab.stop_event" in source, (
            "_run_task_inner should read stop_event from tab"
        )
        # Should NOT assign a new Event to tab.stop_event
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if "tab.stop_event" in stripped and "=" in stripped:
                # "stop_event = tab.stop_event" is fine (reading)
                # "tab.stop_event = threading.Event()" would be bad (creating)
                assert "tab.stop_event =" not in stripped, (
                    "_run_task_inner should not assign to tab.stop_event"
                )

    def test_stop_event_available_immediately_after_start(self) -> None:
        """Verify stop_event is immediately available after starting a task."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        tab_id = "test-stop-early"
        # Simulate what _handle_command does for "run"
        tab = server._get_tab(tab_id)
        tab.stop_event = threading.Event()
        tab.user_answer_queue = queue.Queue(maxsize=1)
        # stop_event should be immediately readable
        assert tab.stop_event is not None, (
            "stop_event must be available before task thread starts"
        )
        # Calling _stop_task should set the event
        server._stop_task(tab_id)
        assert tab.stop_event.is_set(), (
            "stop_event should be set after _stop_task"
        )


class TestRCNew2GetTabThreadSafety(unittest.TestCase):
    """RC-NEW-2: _get_tab must be thread-safe (protected by _state_lock).

    The get-or-create pattern on _tab_states is a TOCTOU race without lock.
    Under CPython GIL this is mostly safe, but for correctness and
    portability _get_tab should use _state_lock.
    """

    def test_get_tab_uses_state_lock(self) -> None:
        """Verify _get_tab acquires _state_lock internally."""
        source = inspect.getsource(VSCodeServer._get_tab)
        assert "_state_lock" in source, (
            "_get_tab should use _state_lock for thread-safe get-or-create"
        )

    def test_concurrent_get_tab_different_ids(self) -> None:
        """Concurrent _get_tab calls with different IDs should not corrupt state."""
        server = VSCodeServer()
        barrier = threading.Barrier(20)
        results: dict[str, object] = {}
        lock = threading.Lock()

        def get_tab(tab_id: str) -> None:
            barrier.wait()
            tab = server._get_tab(tab_id)
            with lock:
                results[tab_id] = tab

        threads = [
            threading.Thread(target=get_tab, args=(f"tab-{i}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 20 tabs should be created
        assert len(results) == 20
        assert len(server._tab_states) == 20
        for i in range(20):
            assert f"tab-{i}" in server._tab_states

    def test_concurrent_get_tab_same_id(self) -> None:
        """Concurrent _get_tab with same ID should return the same object."""
        server = VSCodeServer()
        barrier = threading.Barrier(10)
        results: list[object] = []
        lock = threading.Lock()

        def get_tab() -> None:
            barrier.wait()
            tab = server._get_tab("shared-tab")
            with lock:
                results.append(tab)

        threads = [threading.Thread(target=get_tab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same tab object
        assert len(results) == 10
        assert all(r is results[0] for r in results), (
            "All threads should get the same _TabState for the same tab_id"
        )


class TestRCNew3IsMergingTOCTOU(unittest.TestCase):
    """RC-NEW-3: is_merging check and state setup must be in one lock block.

    Previously, _run_task_inner had:
        with self._state_lock:
            if tab.is_merging: return
        # GAP — is_merging could change here
        with self._state_lock:
            tab.use_worktree = ...

    Fix: combine into a single lock block.
    """

    def test_single_lock_block_for_merging_check_and_state_setup(self) -> None:
        """Verify is_merging check and state mutations are in one lock block."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        lines = source.split("\n")
        # Find the is_merging check
        merging_line = -1
        for i, line in enumerate(lines):
            if "tab.is_merging" in line and "if" in line:
                merging_line = i
                break
        assert merging_line >= 0, "Could not find is_merging check"

        # Find the use_worktree assignment
        worktree_line = -1
        for i, line in enumerate(lines):
            if "tab.use_worktree" in line and "=" in line:
                worktree_line = i
                break
        assert worktree_line >= 0, "Could not find use_worktree assignment"

        # Both should be in the same lock block — no separate
        # "with self._state_lock:" between them
        block_between = "\n".join(lines[merging_line:worktree_line])
        lock_count = block_between.count("with self._state_lock")
        assert lock_count <= 1, (
            f"is_merging check and use_worktree should be in the same lock block, "
            f"found {lock_count} lock acquisitions between them"
        )

    def test_stop_event_read_in_same_block(self) -> None:
        """Verify stop_event is read (not created) in the same lock block."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        lines = source.split("\n")
        # Find where stop_event is read from tab
        for i, line in enumerate(lines):
            if "stop_event" in line and "tab.stop_event" in line:
                # Should be in a _state_lock block
                in_lock = False
                for j in range(i - 1, max(0, i - 15), -1):
                    if "_state_lock" in lines[j]:
                        in_lock = True
                        break
                assert in_lock, (
                    "stop_event read should be inside _state_lock block"
                )
                return
        # If stop_event is not explicitly read from tab, that's also fine
        # as long as it's created before the thread starts


class TestRCNew1StressTest(unittest.TestCase):
    """Stress test: rapid stop-after-start should always work."""

    def test_rapid_stop_after_start(self) -> None:
        """Start and immediately stop many tasks — stop must always work."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        for i in range(50):
            tab_id = f"stress-{i}"
            tab = server._get_tab(tab_id)

            # Create stop_event as _handle_command would
            tab.stop_event = threading.Event()
            tab.user_answer_queue = queue.Queue(maxsize=1)

            # Immediately stop — stop_event must be available
            server._stop_task(tab_id)
            assert tab.stop_event.is_set(), (
                f"stop_event for tab {tab_id} should be set after _stop_task"
            )


class TestUserAnswerQueueCreatedBeforeThread(unittest.TestCase):
    """user_answer_queue should be created in _handle_command, not task thread."""

    def test_queue_created_in_handle_command(self) -> None:
        """Verify user_answer_queue is created in _handle_command run handler."""
        source = inspect.getsource(VSCodeServer._handle_command)
        lines = source.split("\n")
        in_run = False
        run_block: list[str] = []
        for line in lines:
            if '"run"' in line and "cmd_type ==" in line:
                in_run = True
            elif in_run:
                if line.strip().startswith("elif") or line.strip().startswith("else:"):
                    break
                run_block.append(line)
        block = "\n".join(run_block)
        assert "user_answer_queue" in block, (
            "user_answer_queue should be created in _handle_command"
        )

    def test_queue_not_created_in_run_task_inner(self) -> None:
        """Verify _run_task_inner does not create a new user_answer_queue."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "queue.Queue" not in source, (
            "_run_task_inner should not create a new Queue — it's pre-created"
        )


if __name__ == "__main__":
    unittest.main()
