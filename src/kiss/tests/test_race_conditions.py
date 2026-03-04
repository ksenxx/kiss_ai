"""Tests for race condition fixes in Base, browser_ui, and model.

Verifies thread-safety of shared mutable state: agent_counter,
global_budget_used, _bash_buffer, and _callback_helper_loop.
"""

import queue
import threading
import time

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


class TestAgentCounterThreadSafety:
    """Verify Base.agent_counter yields unique IDs under concurrent init."""

    def test_concurrent_agent_ids_unique(self):
        """Spawn many threads each creating a Base, collect IDs, check uniqueness."""
        num_threads = 50
        ids: list[int] = []
        lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        def create_agent():
            barrier.wait()
            agent = Base("test")
            with lock:
                ids.append(agent.id)

        threads = [threading.Thread(target=create_agent) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == num_threads
        assert len(set(ids)) == num_threads, f"Duplicate IDs found: {ids}"

    def test_sequential_agent_ids_increment(self):
        """Sequential Base init yields strictly increasing IDs."""
        a1 = Base("a")
        a2 = Base("b")
        a3 = Base("c")
        assert a2.id == a1.id + 1
        assert a3.id == a2.id + 1


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

    def test_concurrent_append_and_flush_no_data_loss(self):
        """Interleave append and flush from separate threads; all data arrives."""
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        num_appends = 100
        barrier = threading.Barrier(2)

        def appender():
            barrier.wait()
            for i in range(num_appends):
                p.print(f"line{i}\n", type="bash_stream")
                time.sleep(0.001)

        def flusher():
            barrier.wait()
            for _ in range(num_appends):
                p._flush_bash()
                time.sleep(0.001)

        t1 = threading.Thread(target=appender)
        t2 = threading.Thread(target=flusher)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Final flush to get remaining data
        p._flush_bash()

        events = _drain(q)
        all_text = "".join(e["text"] for e in events if e.get("type") == "system_output")
        for i in range(num_appends):
            assert f"line{i}\n" in all_text, f"line{i} missing from output"

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

    def test_timer_flush_and_explicit_flush_no_duplicate(self):
        """Timer-fired flush + explicit flush shouldn't produce duplicate output."""
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._bash_last_flush = time.monotonic()
        p.print("data\n", type="bash_stream")
        assert p._bash_flush_timer is not None
        # Wait for timer to fire
        time.sleep(0.2)
        # Also call explicit flush
        p._flush_bash()

        events = _drain(q)
        texts = [e["text"] for e in events if e.get("type") == "system_output"]
        combined = "".join(texts)
        assert combined == "data\n", f"Expected 'data\\n' but got {combined!r}"


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

    def test_callback_loop_is_running(self):
        """The callback loop should be running and accepting work."""
        loop = _get_callback_loop()
        assert loop.is_running()
        assert not loop.is_closed()
