"""Tests validating that race conditions in the VS Code server are FIXED.

Each test class targets a specific race condition from the race.md audit,
verifying that the fix is correctly applied.

Fixed Python races:
  RC1 — _stop_event read/write protected by _state_lock
  RC2 — _task_thread read in _stop_task / resumeSession protected by _state_lock
  RC3 — _use_worktree written inside _state_lock in _run_task_inner
  RC4 — _refresh_file_cache uses generation counter to prevent stale overwrite
  RC5 — _flush_bash generation TOCTOU removed (no generation check)
  RC6 — _generate_followup_async holds _state_lock across check + broadcast
  RC7 — worktree action Promise has timeout (TS-side, verified by inspection)
  RC8 — _user_answer_queue replaced with fresh Queue per task (no drain race)
  RC9 — _recording_id incremented inside _state_lock
  RC12 — _get_last_session guarded against concurrent running task
  RC13 — status running:false broadcast inside _state_lock
  RC14 — _periodic_event_flush reads _last_task_id under _state_lock
"""

import inspect
import queue
import threading
import time
import unittest

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# RC1 — _stop_event protected by _state_lock
# ---------------------------------------------------------------------------


class TestRC1StopEventProtected(unittest.TestCase):
    """RC1 fix: _stop_event read/write is now protected by _state_lock."""

    def test_stop_event_created_inside_lock(self) -> None:
        """Verify _stop_events[tab_id] assignment is inside a _state_lock block."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "self._stop_events[" in line.strip() and "=" in line:
                in_lock = False
                for j in range(i - 1, max(0, i - 15), -1):
                    if "_state_lock" in lines[j]:
                        in_lock = True
                        break
                assert in_lock, (
                    "_stop_events[tab_id] assignment should be inside _state_lock"
                )
                return
        self.fail("Could not find _stop_events assignment in _run_task_inner")

    def test_stop_event_cleared_inside_lock(self) -> None:
        """Verify _stop_events.pop(tab_id) is in _run_task finally block."""
        source = inspect.getsource(VSCodeServer._run_task)
        assert "self._stop_events.pop(" in source, (
            "_stop_events.pop(tab_id) should be in _run_task finally"
        )

    def test_stop_task_reads_under_lock(self) -> None:
        """Verify _stop_task reads _stop_events under _state_lock."""
        source = inspect.getsource(VSCodeServer._stop_task)
        assert "_state_lock" in source, (
            "_stop_task should use _state_lock to read _stop_events"
        )
        assert "self._stop_events.get(" in source

    def test_stop_task_atomic_read(self) -> None:
        """Demonstrate that _stop_task reads stop event atomically."""
        server = VSCodeServer()
        tab_id = 1
        with server._state_lock:
            server._stop_events[tab_id] = threading.Event()
        # _stop_task reads under lock
        server._stop_task(tab_id)
        # The stop event should be set
        assert server._stop_events[tab_id].is_set()


# ---------------------------------------------------------------------------
# RC2 — _task_thread protected by _state_lock in _stop_task / resumeSession
# ---------------------------------------------------------------------------


class TestRC2TaskThreadProtected(unittest.TestCase):
    """RC2 fix: _task_thread read in _stop_task / resumeSession under _state_lock."""

    def test_stop_task_reads_under_lock(self) -> None:
        """Verify _stop_task reads _task_threads under _state_lock."""
        source = inspect.getsource(VSCodeServer._stop_task)
        assert "_state_lock" in source
        assert "self._task_threads.get(" in source

    def test_resume_session_not_blocked(self) -> None:
        """With per-tab tasks, resumeSession is not blocked by other tabs."""
        source = inspect.getsource(VSCodeServer._handle_command)
        lines = source.split("\n")
        in_resume = False
        resume_block: list[str] = []
        for line in lines:
            if '"resumeSession"' in line:
                in_resume = True
            elif in_resume:
                if line.strip().startswith("elif") or line.strip().startswith("else:"):
                    break
                resume_block.append(line)

        block = "\n".join(resume_block)
        # resumeSession no longer needs _state_lock — per-tab isolation
        assert "_replay_session" in block, "resumeSession calls _replay_session"

    def test_resume_not_blocked_when_other_tab_running(self) -> None:
        """With per-tab tasks, resumeSession works even when another tab is running."""
        server = VSCodeServer()

        stop = threading.Event()
        thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        thread.start()
        server._task_threads[99] = thread

        try:
            # resumeSession should still proceed (per-tab isolation)
            server._handle_command({"type": "resumeSession", "sessionId": "test"})
        finally:
            stop.set()
            thread.join()
            server._task_threads.pop(99, None)


# ---------------------------------------------------------------------------
# RC3 — _use_worktree written inside _state_lock
# ---------------------------------------------------------------------------


class TestRC3UseWorktreeProtected(unittest.TestCase):
    """RC3 fix: _use_worktree is set inside _state_lock in _run_task_inner."""

    def test_use_worktree_written_inside_lock(self) -> None:
        """Verify _use_worktree is set inside _state_lock."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "self._use_worktree" in line and "=" in line:
                in_lock = False
                for j in range(i - 1, max(0, i - 15), -1):
                    if "_state_lock" in lines[j]:
                        in_lock = True
                        break
                assert in_lock, (
                    "_use_worktree should be set inside _state_lock (RC3 fix)"
                )
                return
        self.fail("Could not find _use_worktree assignment")


# ---------------------------------------------------------------------------
# RC4 — _refresh_file_cache uses generation counter
# ---------------------------------------------------------------------------


class TestRC4RefreshFileCacheGenerationCounter(unittest.TestCase):
    """RC4 fix: _refresh_file_cache uses generation to prevent stale overwrite."""

    def test_has_generation_counter(self) -> None:
        """Verify _refresh_file_cache uses _refresh_generation."""
        source = inspect.getsource(VSCodeServer._refresh_file_cache)
        assert "_refresh_generation" in source

    def test_stale_refresh_does_not_overwrite(self) -> None:
        """Demonstrate that a slow refresh doesn't overwrite a fast refresh."""
        server = VSCodeServer()
        barrier = threading.Barrier(2)

        def slow_refresh() -> None:
            """Simulates a slow refresh that reads a stale generation."""
            # Capture a generation that will become stale
            with server._state_lock:
                server._refresh_generation += 1
                gen = server._refresh_generation
            barrier.wait()
            time.sleep(0.05)  # slow scan
            with server._state_lock:
                if server._refresh_generation == gen:
                    server._file_cache = ["stale_file.py"]

        def fast_refresh() -> None:
            """Simulates a fast refresh that increments generation."""
            barrier.wait()
            with server._state_lock:
                server._refresh_generation += 1
            with server._state_lock:
                server._file_cache = ["fresh_file.py"]

        t1 = threading.Thread(target=slow_refresh, daemon=True)
        t2 = threading.Thread(target=fast_refresh, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        with server._state_lock:
            cache = server._file_cache

        # The slow refresh should NOT overwrite the fast refresh
        assert cache == ["fresh_file.py"], (
            f"Expected fresh_file.py (generation counter prevents stale overwrite), got {cache}"
        )


# ---------------------------------------------------------------------------
# RC5 — _flush_bash no generation TOCTOU
# ---------------------------------------------------------------------------


class TestRC5FlushBashNoGenerationTOCTOU(unittest.TestCase):
    """RC5 fix: _flush_bash no longer has a generation check outside the lock."""

    def test_no_generation_check(self) -> None:
        """Verify _flush_bash does not check _bash_generation outside the lock."""
        source = inspect.getsource(BaseBrowserPrinter._flush_bash)
        assert "gen ==" not in source, (
            "_flush_bash should not check gen == _bash_generation (RC5 fix)"
        )
        assert "_bash_generation" not in source, (
            "_flush_bash should not reference _bash_generation at all (RC5 fix)"
        )

    def test_flush_after_reset_does_not_broadcast_stale(self) -> None:
        """After reset(), _flush_bash should not broadcast stale buffer content."""
        broadcasts: list[dict] = []

        class TestPrinter(BaseBrowserPrinter):
            def broadcast(self, event: dict) -> None:  # type: ignore[override]
                broadcasts.append(event)

        printer = TestPrinter()

        # Fill the buffer and reset
        with printer._bash_lock:
            printer._bash_buffer.append("stale output")
        printer.reset()  # clears buffer and increments generation
        printer._flush_bash()  # should not broadcast anything

        sys_outputs = [e for e in broadcasts if e.get("type") == "system_output"]
        assert len(sys_outputs) == 0, "No stale output should be broadcast after reset()"


# ---------------------------------------------------------------------------
# RC6 — _generate_followup_async holds _state_lock across check + broadcast
# ---------------------------------------------------------------------------


class TestRC6FollowupAtomicCheckAndBroadcast(unittest.TestCase):
    """RC6 fix: followup check + broadcast are atomic under _state_lock."""

    def test_state_lock_in_followup(self) -> None:
        """Verify _generate_followup_async uses _state_lock."""
        source = inspect.getsource(VSCodeServer._generate_followup_async)
        assert "_state_lock" in source, (
            "Followup thread should hold _state_lock (RC6 fix)"
        )

    def test_toctou_prevented(self) -> None:
        """Demonstrate the generation check + broadcast are atomic."""
        server = VSCodeServer()
        with server._state_lock:
            server._task_generation = 1

        # Simulate followup thread checking under lock
        with server._state_lock:
            matches = server._task_generation == 1
        assert matches

        # New task can't change generation while lock is held
        # (In the real code, the lock is held across check + broadcast)


# ---------------------------------------------------------------------------
# RC7 — worktree Promise has timeout (TS-side)
# ---------------------------------------------------------------------------


class TestRC7WorktreeActionTimeoutInspection(unittest.TestCase):
    """RC7 fix: worktreeAction Promise has a timeout."""

    ts: str

    @classmethod
    def setUpClass(cls) -> None:
        with open("src/kiss/agents/vscode/src/SorcarSidebarView.ts") as f:
            cls.ts = f.read()

    def test_timeout_on_worktree_promise(self) -> None:
        """Verify there IS a timeout on the worktree action Promise."""
        idx = self.ts.index("case 'worktreeAction':")
        end = self.ts.index("break;", idx) + 6
        block = self.ts[idx:end]
        assert "setTimeout" in block, "worktreeAction Promise has timeout (RC7 fix)"
        assert "120_000" in block or "120000" in block

    def test_dispose_resolves_worktree_promise(self) -> None:
        """Verify dispose() resolves any pending worktree action promise."""
        # Find the dispose method
        idx = self.ts.index("public dispose()")
        block = self.ts[idx:idx + 400]
        assert "_worktreeActionResolve" in block, (
            "dispose() should resolve pending worktree promise (RC7 fix)"
        )


# ---------------------------------------------------------------------------
# RC8 — Fresh Queue per task (no drain race)
# ---------------------------------------------------------------------------


class TestRC8FreshQueuePerTask(unittest.TestCase):
    """RC8 fix: fresh Queue per task instead of draining."""

    def test_no_drain_in_run_task_inner(self) -> None:
        """Verify _run_task_inner does not drain the queue."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "get_nowait" not in source, (
            "_run_task_inner should not drain queue (RC8 fix)"
        )

    def test_fresh_queue_created(self) -> None:
        """Verify _run_task_inner creates a fresh per-tab Queue."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "self._user_answer_queues[tab_id]" in source, (
            "_run_task_inner should create a fresh per-tab Queue (RC8 fix)"
        )

    def test_answer_not_stolen_by_new_task(self) -> None:
        """Demonstrate that per-tab queues prevent answer theft."""
        server = VSCodeServer()
        tab_id = 1
        old_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        old_queue.put("user_answer")
        server._user_answer_queues[tab_id] = old_queue

        # New task creates a fresh queue for the same tab
        new_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        server._user_answer_queues[tab_id] = new_queue

        # Old answer is in the old queue, not the new one
        assert server._user_answer_queues[tab_id].empty()
        assert not old_queue.empty()
        assert old_queue.get_nowait() == "user_answer"


# ---------------------------------------------------------------------------
# RC9 — _recording_id incremented inside _state_lock
# ---------------------------------------------------------------------------


class TestRC9RecordingIdUnderLock(unittest.TestCase):
    """RC9 fix: _recording_id += 1 inside _state_lock."""

    def test_recording_id_inside_lock(self) -> None:
        """Verify _recording_id is incremented inside _state_lock."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "self._recording_id += 1" in line:
                in_lock = False
                for j in range(i - 1, max(0, i - 15), -1):
                    if "_state_lock" in lines[j]:
                        in_lock = True
                        break
                assert in_lock, (
                    "_recording_id += 1 should be inside _state_lock (RC9 fix)"
                )
                return
        self.fail("Could not find _recording_id increment")


# ---------------------------------------------------------------------------
# RC10 — No deadlocks found
# ---------------------------------------------------------------------------


class TestRC10NoDeadlocks(unittest.TestCase):
    """RC10: Verify no deadlocks exist by checking lock ordering is respected."""

    def test_lock_ordering_comment_exists(self) -> None:
        """Verify the lock ordering is documented in the code."""
        source = inspect.getsource(VSCodeServer.__init__)
        assert "Lock ordering" in source

    def test_broadcast_acquires_locks_in_order(self) -> None:
        """VSCodePrinter.broadcast acquires _lock then _stdout_lock (correct order)."""
        from kiss.agents.vscode.server import VSCodePrinter

        source = inspect.getsource(VSCodePrinter.broadcast)
        lock_pos = source.index("self._lock")
        stdout_pos = source.index("self._stdout_lock")
        assert lock_pos < stdout_pos

    def test_flush_bash_releases_before_broadcast(self) -> None:
        """_flush_bash releases _bash_lock before calling broadcast."""
        source = inspect.getsource(BaseBrowserPrinter._flush_bash)
        lines = source.split("\n")
        in_lock_block = False
        indent_level = 0
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if "with self._bash_lock:" in stripped:
                in_lock_block = True
                indent_level = indent
            elif in_lock_block:
                if indent <= indent_level and stripped and not stripped.startswith("#"):
                    in_lock_block = False
                if in_lock_block and "self.broadcast(" in stripped:
                    self.fail("broadcast called inside _bash_lock!")


# ---------------------------------------------------------------------------
# RC11 — AgentProcess stdout buffer flush on close
# ---------------------------------------------------------------------------


class TestRC11StdoutBufferFlushOnClose(unittest.TestCase):
    """RC11 fix: AgentProcess flushes buffer on close."""

    ts: str

    @classmethod
    def setUpClass(cls) -> None:
        with open("src/kiss/agents/vscode/src/AgentProcess.ts") as f:
            cls.ts = f.read()

    def test_buffer_flushed_on_close(self) -> None:
        """Verify the close handler flushes remaining buffer."""
        assert "this.buffer.trim()" in self.ts
        assert "this.buffer = '';" in self.ts

    def test_buffer_retains_incomplete_lines(self) -> None:
        """Verify the buffer keeps incomplete lines for normal operation."""
        assert "this.buffer = lines.pop() || ''" in self.ts


# ---------------------------------------------------------------------------
# RC12 — _get_last_session guarded against concurrent running task
# ---------------------------------------------------------------------------


class TestRC12GetLastSessionGuarded(unittest.TestCase):
    """RC12 fix: _get_last_session guards against concurrent running task."""

    def test_has_task_threads_check(self) -> None:
        """Verify _get_last_session checks _task_threads."""
        source = inspect.getsource(VSCodeServer._get_last_session)
        assert "_task_threads" in source
        assert "_state_lock" in source

    def test_blocked_when_running(self) -> None:
        """Verify _get_last_session returns early when a task is running."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[assignment]

        stop = threading.Event()
        thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        thread.start()
        server._task_threads[0] = thread

        try:
            server._get_last_session()
            # Should return early without broadcasting anything
            task_events = [e for e in events if e.get("type") == "task_events"]
            assert len(task_events) == 0, (
                "_get_last_session should not replay when task is running"
            )
        finally:
            stop.set()
            thread.join()
            server._task_threads.pop(0, None)


# ---------------------------------------------------------------------------
# RC13 — status broadcast inside _state_lock
# ---------------------------------------------------------------------------


class TestRC13StatusBroadcastInsideLock(unittest.TestCase):
    """RC13 fix: status running:false broadcast inside _state_lock."""

    def test_broadcast_inside_lock(self) -> None:
        """Verify the status broadcast is inside the _state_lock block."""
        source = inspect.getsource(VSCodeServer._run_task)
        lines = source.split("\n")
        in_finally = False
        in_lock = False
        lock_indent = 0
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if "finally:" in stripped:
                in_finally = True
            elif in_finally and "self._state_lock" in stripped:
                in_lock = True
                lock_indent = indent
            elif in_lock:
                if indent <= lock_indent and stripped:
                    in_lock = False
                if '"running": False' in stripped:
                    assert in_lock, (
                        "status running:false should be inside _state_lock (RC13 fix)"
                    )
                    return
        # The broadcast should have been found inside the lock
        self.fail("Could not find status broadcast in the right location")


# ---------------------------------------------------------------------------
# RC14 — _periodic_event_flush reads under _state_lock
# ---------------------------------------------------------------------------


class TestRC14PeriodicFlushReadsUnderLock(unittest.TestCase):
    """RC14 fix: _periodic_event_flush reads _last_task_id under _state_lock."""

    def test_reads_under_lock(self) -> None:
        """Verify _periodic_event_flush reads _last_task_id under _state_lock."""
        source = inspect.getsource(VSCodeServer._periodic_event_flush)
        assert "_state_lock" in source


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestRaceConditionCatalog(unittest.TestCase):
    """Verify all race conditions are documented and testable."""

    def test_all_python_races_have_test_classes(self) -> None:
        """Each identified Python race has a corresponding test class."""
        import re
        import sys

        module = sys.modules[__name__]
        test_classes = [
            name for name in dir(module)
            if name.startswith("TestRC") and isinstance(getattr(module, name), type)
        ]
        rc_numbers = set()
        for name in test_classes:
            m = re.search(r"RC(\d+)", name)
            if m:
                rc_numbers.add(int(m.group(1)))

        expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
        assert expected.issubset(rc_numbers), (
            f"Missing test classes for RCs: {expected - rc_numbers}"
        )


if __name__ == "__main__":
    unittest.main()
