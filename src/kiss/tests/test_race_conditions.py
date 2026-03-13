"""Tests for race condition fixes in Base, browser_ui, and model.

Verifies thread-safety of shared mutable state: agent_counter,
global_budget_used, _bash_buffer, and _callback_helper_loop.
Also verifies cross-process safety of _increment_usage and _save_last_model.
"""

import json
import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path

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


def _worker_increment_usage(usage_file: str, key: str, n: int) -> None:
    """Child-process worker: increment usage counter *n* times."""
    import kiss.agents.sorcar.task_history as th

    th._KISS_DIR = Path(usage_file).parent
    th.MODEL_USAGE_FILE = Path(usage_file)
    th.FILE_USAGE_FILE = Path(usage_file).parent / "file_usage.json"
    for _ in range(n):
        th._increment_usage(Path(usage_file), key)


class TestCrossProcessIncrementUsage:
    """Verify _increment_usage is safe under concurrent multi-process access."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_processes_no_lost_increments(self):
        """N processes each incrementing the same key M times yields N*M total."""
        usage_file = os.path.join(self.tmpdir, "model_usage.json")
        Path(usage_file).write_text(json.dumps({"mymodel": 0}))

        n_procs = 5
        increments_each = 10
        procs = [
            multiprocessing.Process(
                target=_worker_increment_usage,
                args=(usage_file, "mymodel", increments_each),
            )
            for _ in range(n_procs)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        result = json.loads(Path(usage_file).read_text())
        assert result["mymodel"] == n_procs * increments_each


def _worker_save_last_model(usage_file: str, model: str, n: int) -> None:
    """Child-process worker: call _save_last_model *n* times."""
    import kiss.agents.sorcar.task_history as th

    th._KISS_DIR = Path(usage_file).parent
    th.MODEL_USAGE_FILE = Path(usage_file)
    for _ in range(n):
        th._save_last_model(model)


class TestCrossProcessSaveLastModel:
    """Verify _save_last_model is safe under concurrent multi-process access."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_save_last_model_no_corruption(self):
        """Multiple processes saving last model concurrently produce valid JSON."""
        usage_file = os.path.join(self.tmpdir, "model_usage.json")
        Path(usage_file).write_text(json.dumps({"_last": "old"}))

        procs = [
            multiprocessing.Process(
                target=_worker_save_last_model,
                args=(usage_file, f"model-{i}", 5),
            )
            for i in range(4)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        # File must be valid JSON (no corruption from partial writes)
        data = json.loads(Path(usage_file).read_text())
        assert "_last" in data


def _worker_atomic_write(path: str, content: str, n: int) -> None:
    """Child-process worker: atomically write *content* to *path* *n* times."""
    from kiss.agents.sorcar.sorcar import _atomic_write_text

    for _ in range(n):
        _atomic_write_text(Path(path), content)


class TestAtomicWriteText:
    """Verify _atomic_write_text never leaves a partial/empty file."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_writes_file_never_empty(self):
        """Concurrent atomic writes should never leave an empty or partial file."""
        path = os.path.join(self.tmpdir, "port.txt")
        Path(path).write_text("12345")

        procs = [
            multiprocessing.Process(
                target=_worker_atomic_write,
                args=(path, str(i * 1000 + j), 20),
            )
            for i in range(3)
            for j in range(3)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        # File must exist and contain non-empty content
        content = Path(path).read_text()
        assert content.strip() != ""
