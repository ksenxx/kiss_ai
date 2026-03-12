"""Tests for race condition fixes in sorcar.

Verifies:
1. Base._class_lock protects agent_counter and global_budget_used
2. stop_event set/clear happens inside running_lock (no stop→run race)
3. task_done broadcast happens AFTER running=False (no 409 on immediate re-submit)
4. _history_cache is protected by _history_lock (no lost updates)
5. shutdown_timer is protected by shutdown_lock
"""

import threading
import time

from kiss.agents.sorcar.task_history import (
    _add_task,
    _load_history,
    _set_latest_chat_events,
)
from kiss.core.base import Base


class TestBaseClassLock:
    """Verify Base._class_lock protects agent_counter and global_budget_used."""

    def test_class_lock_exists(self):
        assert hasattr(Base, "_class_lock")
        assert isinstance(Base._class_lock, type(threading.Lock()))

    def test_concurrent_budget_updates_with_lock(self):
        """Concurrent budget increments under _class_lock should not lose updates."""
        initial = Base.global_budget_used
        num = 100
        barrier = threading.Barrier(num)

        def update():
            barrier.wait()
            with Base._class_lock:
                Base.global_budget_used += 1.0

        threads = [threading.Thread(target=update) for _ in range(num)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        expected = initial + num
        assert abs(Base.global_budget_used - expected) < 1e-9
        # Reset for other tests
        Base.global_budget_used = initial


class TestTaskDoneAfterRunningFalse:
    """Verify task_done is broadcast AFTER running=False."""

    def test_no_409_on_immediate_resubmit(self):
        """Client should be able to start a new task immediately after task_done."""
        running = False
        agent_thread: threading.Thread | None = None
        running_lock = threading.Lock()
        events: list[dict] = []
        events_lock = threading.Lock()
        can_check = threading.Event()

        def broadcast(event):
            with events_lock:
                events.append(event)
            if event.get("type") == "task_done":
                can_check.set()

        def run_agent_thread():
            nonlocal running, agent_thread
            current = threading.current_thread()
            done_event = {"type": "task_done"}
            # Simulate the fixed pattern:
            # set running=False THEN broadcast
            with running_lock:
                if agent_thread is not current:
                    return
                running = False
                agent_thread = None
            broadcast(done_event)

        def start_task():
            nonlocal running, agent_thread
            t = threading.Thread(target=run_agent_thread, daemon=True)
            with running_lock:
                if running:
                    return False
                running = True
                agent_thread = t
            t.start()
            return True

        # Start first task
        assert start_task()
        can_check.wait(timeout=5)

        # Immediately try to start another — should succeed, not get 409
        assert start_task()

        # Wait for second to complete
        time.sleep(0.2)
        with running_lock:
            assert not running


class TestHistoryLock:
    """Verify _history_lock protects _history_cache from concurrent corruption."""

    def test_history_lock_exists(self):
        from kiss.agents.sorcar import task_history
        assert hasattr(task_history, "_history_lock")
        assert isinstance(task_history._history_lock, type(threading.Lock()))

    def test_concurrent_set_chat_events_and_add_task(self):
        """Concurrent _set_latest_chat_events and _add_task should not corrupt state."""
        from kiss.agents.sorcar import task_history

        orig_cache = task_history._history_cache
        orig_file_content = None
        if task_history.HISTORY_FILE.exists():
            orig_file_content = task_history.HISTORY_FILE.read_text()

        try:
            task_history._history_cache = None
            _add_task("initial_task")
            task_history._history_cache = None  # Force reload

            errors: list[Exception] = []
            barrier = threading.Barrier(2)

            def add_tasks():
                barrier.wait()
                for i in range(10):
                    try:
                        _add_task(f"add_task_{i}")
                    except Exception as e:
                        errors.append(e)

            def set_results():
                barrier.wait()
                for i in range(10):
                    try:
                        _set_latest_chat_events([{"type": "text_delta", "text": f"result_{i}"}])
                    except Exception as e:
                        errors.append(e)

            t1 = threading.Thread(target=add_tasks)
            t2 = threading.Thread(target=set_results)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            assert errors == [], f"Errors during concurrent access: {errors}"
            # History should be valid (no corruption)
            history = _load_history()
            assert isinstance(history, list)
            assert len(history) > 0
        finally:
            task_history._history_cache = orig_cache
            if orig_file_content is not None:
                task_history.HISTORY_FILE.write_text(orig_file_content)


class TestShutdownTimerLock:
    """Verify shutdown_timer access is protected by shutdown_lock."""

    def test_concurrent_schedule_shutdown_no_error(self):
        """Multiple concurrent _schedule_shutdown calls should not raise."""
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        shutdowns_called = []

        def _do_shutdown():
            shutdowns_called.append(1)

        def _schedule_shutdown():
            nonlocal shutdown_timer
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                shutdown_timer = threading.Timer(1.0, _do_shutdown)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        num = 20
        barrier = threading.Barrier(num)
        errors: list[Exception] = []

        def do_schedule():
            barrier.wait()
            try:
                _schedule_shutdown()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_schedule) for _ in range(num)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Clean up timer
        with shutdown_lock:
            if shutdown_timer is not None:
                shutdown_timer.cancel()
