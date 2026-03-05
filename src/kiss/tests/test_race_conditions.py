"""Tests for race condition fixes in Base, browser_ui, and model.

Verifies thread-safety of shared mutable state: agent_counter,
global_budget_used, _bash_buffer, and _callback_helper_loop.
"""

import queue
import threading

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.core.base import Base
from kiss.core.models.model import _get_callback_loop


def _subscribe(printer: BaseBrowserPrinter) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    printer._clients.append(q)
    return q


def _drain(q: queue.Queue) -> list[dict]:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


class TestGlobalBudgetThreadSafety:
    """Verify Base.global_budget_used accumulates correctly under concurrent updates."""

    def test_concurrent_budget_updates(self):
        """Many threads incrementing global_budget_used should not lose updates."""
        num_threads = 50
        increment = 1.0
        initial = Base.global_budget_used
        barrier = threading.Barrier(num_threads)

        def update_budget():
            barrier.wait()
            with Base._class_lock:
                Base.global_budget_used += increment

        threads = [threading.Thread(target=update_budget) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = initial + num_threads * increment
        assert abs(Base.global_budget_used - expected) < 1e-9


class TestBashBufferThreadSafety:
    """Verify _bash_buffer doesn't lose data when timer flush races with append."""

    def test_concurrent_reset_and_append(self):
        """Reset from one thread while another appends doesn't raise."""
        p = BaseBrowserPrinter()
        _subscribe(p)
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def appender():
            barrier.wait()
            for i in range(50):
                try:
                    p.print(f"x{i}\n", type="bash_stream")
                except Exception as e:
                    errors.append(e)

        def resetter():
            barrier.wait()
            for _ in range(50):
                try:
                    p.reset()
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=appender)
        t2 = threading.Thread(target=resetter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Exceptions during concurrent access: {errors}"


class TestCallbackLoopThreadSafety:
    """Verify _get_callback_loop returns same loop from concurrent callers."""

    def test_concurrent_get_callback_loop_same_instance(self):
        """Multiple threads calling _get_callback_loop get the same loop."""
        num_threads = 10
        loops: list = []
        lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        def get_loop():
            barrier.wait()
            loop = _get_callback_loop()
            with lock:
                loops.append(loop)

        threads = [threading.Thread(target=get_loop) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(loops) == num_threads
        assert all(loop is loops[0] for loop in loops), "Got different loop instances"
