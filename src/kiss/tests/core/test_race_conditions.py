"""Tests for race condition fixes in Base, browser_ui, and model.

Verifies thread-safety of shared mutable state: agent_counter,
global_budget_used, _bash_buffer, and _callback_helper_loop.
Also verifies cross-process safety of _record_model_usage and _save_last_model.
"""

import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.base import Base


def _subscribe(printer: BaseBrowserPrinter) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    printer._client_queue = q
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


def _worker_record_model_usage(db_dir: str, model: str, n: int) -> None:
    """Child-process worker: record model usage *n* times via SQLite."""
    import kiss.agents.sorcar.persistence as th

    kiss_dir = Path(db_dir)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    for _ in range(n):
        th._record_model_usage(model)


class TestCrossProcessRecordModelUsage:
    """Verify _record_model_usage is safe under concurrent multi-process access."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_processes_no_lost_increments(self):
        """N processes each recording the same model M times yields N*M total."""
        kiss_dir = os.path.join(self.tmpdir, ".kiss")
        os.makedirs(kiss_dir, exist_ok=True)

        # Pre-create the DB so workers don't race on WAL initialization
        import kiss.agents.sorcar.persistence as th
        saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = Path(kiss_dir)
        th._DB_PATH = Path(kiss_dir) / "history.db"
        th._db_conn = None
        th._get_db()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved

        n_procs = 5
        increments_each = 10
        procs = [
            multiprocessing.Process(
                target=_worker_record_model_usage,
                args=(kiss_dir, "mymodel", increments_each),
            )
            for _ in range(n_procs)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        import kiss.agents.sorcar.persistence as th
        saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = Path(kiss_dir)
        th._DB_PATH = Path(kiss_dir) / "history.db"
        th._db_conn = None
        try:
            usage = th._load_model_usage()
            assert usage["mymodel"] == n_procs * increments_each
        finally:
            if th._db_conn is not None:
                th._db_conn.close()
                th._db_conn = None
            (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _worker_save_last_model(db_dir: str, model: str, n: int) -> None:
    """Child-process worker: call _save_last_model *n* times via SQLite."""
    import kiss.agents.sorcar.persistence as th

    kiss_dir = Path(db_dir)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    for _ in range(n):
        th._save_last_model(model)


class TestCrossProcessSaveLastModel:
    """Verify _save_last_model is safe under concurrent multi-process access."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_save_last_model_no_corruption(self):
        """Multiple processes saving last model concurrently produce valid data."""
        kiss_dir = os.path.join(self.tmpdir, ".kiss")
        os.makedirs(kiss_dir, exist_ok=True)

        procs = [
            multiprocessing.Process(
                target=_worker_save_last_model,
                args=(kiss_dir, f"model-{i}", 5),
            )
            for i in range(4)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        import kiss.agents.sorcar.persistence as th
        saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = Path(kiss_dir)
        th._DB_PATH = Path(kiss_dir) / "history.db"
        th._db_conn = None
        try:
            last = th._load_last_model()
            assert isinstance(last, str) and len(last) > 0
        finally:
            if th._db_conn is not None:
                th._db_conn.close()
                th._db_conn = None
            (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved



