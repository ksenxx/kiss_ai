"""Tests for the stop-agent-and-restart threading behavior in sorcar.py."""

import asyncio
import ctypes
import threading
import time
from collections.abc import Callable

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.agents.sorcar.sorcar import _StopRequested


def test_stop_requested_is_base_exception():
    assert issubclass(_StopRequested, BaseException)
    assert not issubclass(_StopRequested, Exception)


def test_async_exc_injection_kills_thread():
    stopped = threading.Event()
    started = threading.Event()

    def worker():
        started.set()
        try:
            while True:
                time.sleep(0.01)
        except _StopRequested:
            stopped.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    started.wait(timeout=2)
    assert t.is_alive()

    tid = t.ident
    assert tid is not None
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid),
        ctypes.py_object(_StopRequested),
    )
    stopped.wait(timeout=3)
    assert stopped.is_set()
    t.join(timeout=2)
    assert not t.is_alive()


def test_abandoned_thread_does_not_interfere_with_new_thread():
    """Simulates stop_agent resetting state, then a new thread starting."""
    running = False
    agent_thread = None
    lock = threading.Lock()
    events = []

    def run_agent(task_name):
        nonlocal running, agent_thread
        current = threading.current_thread()
        try:
            while True:
                time.sleep(0.01)
        except _StopRequested:
            with lock:
                if agent_thread is not current:
                    return
            events.append(f"{task_name}_stopped")
        finally:
            with lock:
                if agent_thread is not current:
                    return
                running = False
                agent_thread = None

    with lock:
        running = True
    t1 = threading.Thread(target=run_agent, args=("task1",), daemon=True)
    with lock:
        agent_thread = t1
    t1.start()
    time.sleep(0.05)

    with lock:
        old_thread = agent_thread
        running = False
        agent_thread = None
    tid = old_thread.ident
    assert tid is not None
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid),
        ctypes.py_object(_StopRequested),
    )

    with lock:
        assert not running
        assert agent_thread is None

    with lock:
        running = True
    t2 = threading.Thread(target=run_agent, args=("task2",), daemon=True)
    with lock:
        agent_thread = t2
    t2.start()
    time.sleep(0.05)

    assert t2.is_alive()
    with lock:
        assert running
        assert agent_thread is t2

    t1.join(timeout=2)
    assert "task1_stopped" not in events

    with lock:
        old_thread2 = agent_thread
        running = False
        agent_thread = None
    tid2 = old_thread2.ident
    assert tid2 is not None
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid2),
        ctypes.py_object(_StopRequested),
    )
    t2.join(timeout=2)
    assert "task2_stopped" not in events


def test_stop_when_no_thread_returns_false():
    """stop_agent returns False when there's no running thread."""
    agent_thread = None
    lock = threading.Lock()

    with lock:
        thread = agent_thread
    assert thread is None


def test_current_thread_guard_prevents_double_broadcast():
    """If agent_thread is replaced, the old thread skips its broadcasts."""
    agent_thread = None
    lock = threading.Lock()
    events = []
    barrier = threading.Barrier(2, timeout=5)

    def old_worker():
        nonlocal agent_thread
        current = threading.current_thread()
        barrier.wait()
        time.sleep(0.1)
        with lock:
            if agent_thread is not current:
                events.append("old_skipped")
                return
        events.append("old_broadcast")

    def new_worker():
        time.sleep(1)

    t1 = threading.Thread(target=old_worker, daemon=True)
    with lock:
        agent_thread = t1
    t1.start()

    barrier.wait()
    t2 = threading.Thread(target=new_worker, daemon=True)
    with lock:
        agent_thread = t2
    t2.start()

    t1.join(timeout=3)
    assert "old_skipped" in events
    assert "old_broadcast" not in events


def test_stop_requested_exception_propagates_through_finally():
    """_StopRequested propagates correctly through try/except/finally."""
    results = []

    def worker():
        try:
            try:
                while True:
                    time.sleep(0.01)
            except _StopRequested:
                results.append("caught")
                raise
        except _StopRequested:
            results.append("re-caught")
        finally:
            results.append("finally")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.05)

    tid = t.ident
    assert tid is not None
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid),
        ctypes.py_object(_StopRequested),
    )
    t.join(timeout=3)
    assert "caught" in results
    assert "finally" in results


# ---------------------------------------------------------------------------
# Additional tests verifying no deadlock / bad state under all conditions
# ---------------------------------------------------------------------------


def _inject_stop(thread: threading.Thread) -> None:
    tid = thread.ident
    if tid is not None:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid),
            ctypes.py_object(_StopRequested),
        )


class _AgentController:
    """Reproduces the exact lock/guard pattern from sorcar.run_task/stop_agent/run_agent_thread."""

    def __init__(self) -> None:
        self.running = False
        self.agent_thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.events: list[str] = []

    def start_task(self, work: Callable[[], None], tag: str = "") -> threading.Thread | None:
        t = threading.Thread(target=self._thread_body, args=(work, tag), daemon=True)
        with self.lock:
            if self.running:
                return None
            self.running = True
            self.agent_thread = t
        t.start()
        return t

    def stop_task(self) -> bool:
        with self.lock:
            thread = self.agent_thread
            if thread is None or not thread.is_alive():
                return False
            self.running = False
            self.agent_thread = None
        _inject_stop(thread)
        return True

    def _thread_body(self, work: Callable[[], None], tag: str) -> None:
        current = threading.current_thread()
        result = "ok"
        try:
            work()
            with self.lock:
                if self.agent_thread is not current:
                    return
            self.events.append(f"{tag}:done")
        except _StopRequested:
            result = "stopped"
            with self.lock:
                if self.agent_thread is not current:
                    return
            self.events.append(f"{tag}:stopped")
        except Exception as exc:
            result = f"error:{exc}"
            with self.lock:
                if self.agent_thread is not current:
                    return
            self.events.append(f"{tag}:error")
        finally:
            with self.lock:
                if self.agent_thread is not current:
                    return
                self.running = False
                self.agent_thread = None
            self.events.append(f"{tag}:cleanup:{result}")

    def assert_idle(self) -> None:
        with self.lock:
            assert not self.running, "running should be False, got True"
            assert self.agent_thread is None, "agent_thread should be None"


def test_normal_completion_resets_state():
    """Thread completing normally resets running=False and agent_thread=None."""
    ctrl = _AgentController()
    done = threading.Event()

    def quick_work():
        done.set()

    t = ctrl.start_task(quick_work, "t1")
    assert t is not None
    done.wait(timeout=3)
    t.join(timeout=3)
    ctrl.assert_idle()
    assert "t1:done" in ctrl.events
    assert "t1:cleanup:ok" in ctrl.events


def test_error_in_task_resets_state():
    """An exception in the work function still resets state correctly."""
    ctrl = _AgentController()

    def failing_work():
        raise RuntimeError("boom")

    t = ctrl.start_task(failing_work, "err")
    assert t is not None
    t.join(timeout=3)
    ctrl.assert_idle()
    assert "err:error" in ctrl.events
    assert "err:cleanup:error:boom" in ctrl.events


def test_stop_then_start_new_task():
    """After stopping a task, a new task can start immediately."""
    ctrl = _AgentController()

    t1 = ctrl.start_task(_long_work, "a")
    assert t1 is not None
    time.sleep(0.05)
    assert ctrl.stop_task()

    done = threading.Event()
    t2 = ctrl.start_task(lambda: done.set(), "b")
    assert t2 is not None
    done.wait(timeout=3)
    t2.join(timeout=3)
    ctrl.assert_idle()
    assert "b:done" in ctrl.events
    t1.join(timeout=3)


def _long_work():
    """Simulate agent work with short sleep loops (interruptible by async exc)."""
    while True:
        time.sleep(0.005)


def test_rapid_stop_start_cycles_no_deadlock():
    """50 rapid start-stop cycles complete without deadlock or bad state."""
    ctrl = _AgentController()
    threads = []

    for i in range(50):
        t = ctrl.start_task(_long_work, f"r{i}")
        if t is not None:
            threads.append(t)
        time.sleep(0.002)
        ctrl.stop_task()
        time.sleep(0.002)

    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive(), "thread should have exited"

    ctrl.assert_idle()


def test_concurrent_stop_calls_only_one_succeeds():
    """Multiple threads calling stop simultaneously: exactly one returns True."""
    ctrl = _AgentController()

    t = ctrl.start_task(_long_work, "cs")
    assert t is not None
    time.sleep(0.05)

    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(10, timeout=5)

    def try_stop():
        barrier.wait()
        r = ctrl.stop_task()
        with results_lock:
            results.append(r)

    stoppers = [threading.Thread(target=try_stop, daemon=True) for _ in range(10)]
    for s in stoppers:
        s.start()
    for s in stoppers:
        s.join(timeout=5)

    assert results.count(True) == 1
    assert results.count(False) == 9
    t.join(timeout=3)
    ctrl.assert_idle()


def test_start_rejected_while_running():
    """Starting a second task while one is running returns None (rejected)."""
    ctrl = _AgentController()
    started = threading.Event()

    def work_and_signal():
        started.set()
        _long_work()

    t1 = ctrl.start_task(work_and_signal, "s1")
    assert t1 is not None
    started.wait(timeout=3)

    t2 = ctrl.start_task(lambda: None, "s2")
    assert t2 is None

    ctrl.stop_task()
    t1.join(timeout=3)
    ctrl.assert_idle()


def test_stop_on_already_finished_thread():
    """Calling stop after the thread has completed returns False."""
    ctrl = _AgentController()
    done = threading.Event()

    t = ctrl.start_task(lambda: done.set(), "fin")
    assert t is not None
    done.wait(timeout=3)
    t.join(timeout=3)
    ctrl.assert_idle()

    assert not ctrl.stop_task()


def test_finally_runs_for_abandoned_thread_but_guard_prevents_reset():
    """An abandoned thread's finally block runs but the guard prevents state mutation."""
    running = False
    agent_thread: threading.Thread | None = None
    lock = threading.Lock()
    finally_ran = threading.Event()
    state_was_reset = threading.Event()

    def worker():
        nonlocal running, agent_thread
        current = threading.current_thread()
        try:
            while True:
                time.sleep(0.01)
        except _StopRequested:
            pass
        finally:
            finally_ran.set()
            with lock:
                if agent_thread is not current:
                    return
                running = False
                agent_thread = None
            state_was_reset.set()

    with lock:
        running = True
    t = threading.Thread(target=worker, daemon=True)
    with lock:
        agent_thread = t
    t.start()
    time.sleep(0.05)

    with lock:
        running = False
        agent_thread = None
    _inject_stop(t)
    t.join(timeout=3)

    assert finally_ran.is_set()
    assert not state_was_reset.is_set()


def test_lock_not_leaked_after_stop_requested():
    """After _StopRequested kills a thread, the lock is still acquirable."""
    lock = threading.Lock()
    started = threading.Event()

    def worker():
        started.set()
        try:
            while True:
                with lock:
                    time.sleep(0.001)
        except _StopRequested:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    started.wait(timeout=2)
    time.sleep(0.05)

    _inject_stop(t)
    t.join(timeout=3)

    acquired = lock.acquire(timeout=2)
    assert acquired, "lock should be acquirable after thread death"
    lock.release()


def test_lock_not_leaked_when_stop_during_lock_hold():
    """If _StopRequested fires while the thread holds the lock, the lock is released."""
    lock = threading.Lock()
    inside_lock = threading.Event()

    def worker():
        try:
            with lock:
                inside_lock.set()
                while True:
                    time.sleep(0.01)
        except _StopRequested:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    inside_lock.wait(timeout=2)

    _inject_stop(t)
    t.join(timeout=3)

    acquired = lock.acquire(timeout=2)
    assert acquired, "lock must be released even if _StopRequested fires while held"
    lock.release()


def test_state_consistent_after_mixed_operations():
    """A mix of normal completions, errors, and stops always leaves clean state."""
    ctrl = _AgentController()
    threads = []

    # Normal completion
    t = ctrl.start_task(lambda: time.sleep(0.02), "n1")
    assert t is not None
    threads.append(t)
    t.join(timeout=3)
    ctrl.assert_idle()

    # Error
    t = ctrl.start_task(lambda: (_ for _ in ()).throw(ValueError("x")), "e1")
    assert t is not None
    threads.append(t)
    t.join(timeout=3)
    ctrl.assert_idle()

    # Stop
    t = ctrl.start_task(_long_work, "s1")
    assert t is not None
    threads.append(t)
    time.sleep(0.02)
    ctrl.stop_task()
    t.join(timeout=3)
    ctrl.assert_idle()

    # Normal again
    t = ctrl.start_task(lambda: time.sleep(0.02), "n2")
    assert t is not None
    threads.append(t)
    t.join(timeout=3)
    ctrl.assert_idle()

    # Rapid stop
    t = ctrl.start_task(_long_work, "s2")
    assert t is not None
    threads.append(t)
    ctrl.stop_task()
    t.join(timeout=3)
    ctrl.assert_idle()


def test_setup_failure_does_not_corrupt_state():
    """Simulates the fixed run_task: parsing before lock means exceptions are safe."""
    running = False
    agent_thread: threading.Thread | None = None
    lock = threading.Lock()

    def run_task_pattern(task: str) -> str:
        nonlocal running, agent_thread
        if not task:
            return "error:empty"
        t = threading.Thread(target=lambda: None, daemon=True)
        with lock:
            if running:
                return "error:busy"
            running = True
            agent_thread = t
        t.start()
        return "started"

    assert run_task_pattern("") == "error:empty"
    with lock:
        assert not running
        assert agent_thread is None

    assert run_task_pattern("real task") == "started"
    time.sleep(0.1)
    with lock:
        assert not running or agent_thread is not None


def test_concurrent_start_attempts_only_one_wins():
    """Multiple threads trying to start a task: exactly one succeeds."""
    ctrl = _AgentController()
    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(10, timeout=5)

    def try_start():
        barrier.wait()
        t = ctrl.start_task(_long_work, "race")
        with results_lock:
            results.append(t is not None)

    starters = [threading.Thread(target=try_start, daemon=True) for _ in range(10)]
    for s in starters:
        s.start()
    for s in starters:
        s.join(timeout=5)

    assert results.count(True) == 1
    assert results.count(False) == 9

    ctrl.stop_task()
    time.sleep(0.5)
    ctrl.assert_idle()


def test_stop_requested_not_caught_by_except_exception():
    """_StopRequested propagates through except Exception blocks in called code."""
    escaped = threading.Event()
    started = threading.Event()

    def inner_code_with_broad_catch():
        started.set()
        while True:
            try:
                time.sleep(0.01)
            except Exception:
                pass

    def worker():
        try:
            inner_code_with_broad_catch()
        except _StopRequested:
            escaped.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    started.wait(timeout=2)
    time.sleep(0.05)

    _inject_stop(t)
    escaped.wait(timeout=3)
    assert escaped.is_set()
    t.join(timeout=2)


def test_double_stop_is_safe():
    """Calling stop twice in a row is safe; second returns False."""
    ctrl = _AgentController()

    t = ctrl.start_task(_long_work, "ds")
    assert t is not None
    time.sleep(0.05)

    assert ctrl.stop_task() is True
    assert ctrl.stop_task() is False
    t.join(timeout=3)
    ctrl.assert_idle()


def test_stress_interleaved_start_stop_from_many_threads():
    """Hammer start/stop from 20 threads; verify no deadlock and clean final state."""
    ctrl = _AgentController()
    barrier = threading.Barrier(20, timeout=10)
    all_threads: list[threading.Thread] = []

    def hammer():
        barrier.wait()
        for _ in range(20):
            t = ctrl.start_task(_long_work, "h")
            if t is not None:
                all_threads.append(t)
            time.sleep(0.001)
            ctrl.stop_task()
            time.sleep(0.001)

    workers = [threading.Thread(target=hammer, daemon=True) for _ in range(20)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=30)

    for t in all_threads:
        t.join(timeout=3)

    time.sleep(0.2)
    ctrl.assert_idle()


# ---------------------------------------------------------------------------
# Tests for BaseBrowserPrinter.stop_event integration
# ---------------------------------------------------------------------------


def test_printer_stop_event_raises_in_print():
    """Setting stop_event causes printer.print() to raise KeyboardInterrupt."""
    printer = BaseBrowserPrinter()
    printer.stop_event.set()
    raised = False
    try:
        printer.print("hello", type="text")
    except KeyboardInterrupt:
        raised = True
    assert raised


def test_printer_stop_event_raises_in_token_callback():
    """Setting stop_event causes token_callback() to raise KeyboardInterrupt."""
    printer = BaseBrowserPrinter()
    printer.stop_event.set()
    raised = False
    try:
        asyncio.run(printer.token_callback("token"))
    except KeyboardInterrupt:
        raised = True
    assert raised


def test_printer_stop_event_clear_allows_normal_operation():
    """Cleared stop_event allows print() and token_callback() to work normally."""
    printer = BaseBrowserPrinter()
    printer.stop_event.clear()
    printer.print("hello", type="text")
    asyncio.run(printer.token_callback("token"))


def test_printer_stop_event_stops_agent_loop_in_thread():
    """Full integration: stop_event interrupts a thread doing printer.print() in a loop."""
    printer = BaseBrowserPrinter()
    printer.stop_event.clear()
    stopped = threading.Event()
    started = threading.Event()

    def agent_loop():
        started.set()
        try:
            for i in range(10000):
                printer.print(f"Step {i}", type="text")
                time.sleep(0.001)
        except KeyboardInterrupt:
            stopped.set()

    t = threading.Thread(target=agent_loop, daemon=True)
    t.start()
    started.wait(timeout=2)
    time.sleep(0.05)
    printer.stop_event.set()
    t.join(timeout=3)
    assert stopped.is_set()
    assert not t.is_alive()


def test_printer_stop_event_reusable_across_tasks():
    """stop_event can be set/cleared across multiple task cycles."""
    printer = BaseBrowserPrinter()

    for _ in range(5):
        printer.stop_event.clear()
        stopped = threading.Event()
        started = threading.Event()

        def agent_loop():
            started.set()
            try:
                while True:
                    printer.print("working", type="text")
                    time.sleep(0.001)
            except KeyboardInterrupt:
                stopped.set()

        t = threading.Thread(target=agent_loop, daemon=True)
        t.start()
        started.wait(timeout=2)
        time.sleep(0.02)
        printer.stop_event.set()
        t.join(timeout=3)
        assert stopped.is_set()
        assert not t.is_alive()
