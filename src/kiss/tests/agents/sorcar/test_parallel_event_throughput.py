# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Concurrency tests for the background event-writer queue.

These tests verify that ``_queue_chat_event`` + ``_flush_chat_events``
correctly persist every event from N parallel writer threads with dense
``seq`` numbering, and that the producer-side per-call latency is sub-
millisecond (i.e. the writer thread takes the SQL hit, not the agent
threads).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str) -> tuple:
    """Point persistence at a temp DB and force a fresh writer thread."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    # _close_db drains+stops the writer thread, bumps the generation,
    # clears the next-seq cache, and resets the connection cache.
    th._close_db()
    return saved


def _restore(saved: tuple) -> None:
    th._close_db()
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestParallelEventThroughput:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parallel_writers_persist_all_events_with_dense_seq(self) -> None:
        """Eight threads × 500 events → 4000 rows with seq [0..3999]."""
        task_id, _ = th._add_task("bench-task", "")
        n_threads = 8
        n_events = 500

        def worker() -> None:
            for i in range(n_events):
                th._queue_chat_event(
                    {"type": "text_delta", "tabId": "t1", "content": f"x{i}"},
                    task_id=task_id,
                )

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        th._flush_chat_events()

        db = th._get_db()
        row = db.execute(
            "SELECT COUNT(*) AS c FROM events WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        assert row["c"] == n_threads * n_events

        seqs = [
            r["seq"]
            for r in db.execute(
                "SELECT seq FROM events WHERE task_id = ? ORDER BY seq",
                (task_id,),
            ).fetchall()
        ]
        assert seqs == list(range(n_threads * n_events))

        has_events = db.execute(
            "SELECT has_events FROM task_history WHERE id = ?",
            (task_id,),
        ).fetchone()["has_events"]
        assert has_events == 1

    def test_queue_call_is_sub_millisecond(self) -> None:
        """``_queue_chat_event`` must not block on the SQL write."""
        task_id, _ = th._add_task("latency-task", "")
        # Prime the writer thread so the very first enqueue isn't charged
        # for thread startup.
        th._queue_chat_event(
            {"type": "text_delta", "tabId": "t1", "content": "warmup"},
            task_id=task_id,
        )
        th._flush_chat_events()

        samples_ns: list[int] = []
        for i in range(500):
            t0 = time.perf_counter_ns()
            th._queue_chat_event(
                {"type": "text_delta", "tabId": "t1", "content": f"x{i}"},
                task_id=task_id,
            )
            samples_ns.append(time.perf_counter_ns() - t0)

        samples_ns.sort()
        median_us = samples_ns[len(samples_ns) // 2] / 1000.0
        p99_us = samples_ns[int(len(samples_ns) * 0.99)] / 1000.0
        # Median must be sub-millisecond; p99 well under 5ms even on
        # slow CI runners.  Synchronous ``_append_chat_event`` measured
        # 200µs–2.5ms median in the benchmark — this confirms the
        # producer-side cost is now bounded by ``json.dumps`` + queue
        # put rather than SQL commit.
        assert median_us < 1000.0, f"median {median_us}µs >= 1ms"
        assert p99_us < 5000.0, f"p99 {p99_us}µs >= 5ms"

        th._flush_chat_events()

    def test_flush_is_idempotent_when_queue_empty(self) -> None:
        """``_flush_chat_events`` must be a no-op when nothing is queued."""
        # No events enqueued yet — writer thread may not even be running.
        th._flush_chat_events()
        th._flush_chat_events()
        # After enqueue + flush + flush, the second flush is still safe.
        task_id, _ = th._add_task("idempotent-task", "")
        th._queue_chat_event(
            {"type": "text_delta", "tabId": "t1", "content": "x"},
            task_id=task_id,
        )
        th._flush_chat_events()
        th._flush_chat_events()
        db = th._get_db()
        assert (
            db.execute(
                "SELECT COUNT(*) FROM events WHERE task_id = ?", (task_id,),
            ).fetchone()[0]
            == 1
        )

    def test_append_chat_event_flushes_pending_queued_events(self) -> None:
        """Synchronous append must land AFTER prior queued events."""
        task_id, _ = th._add_task("order-task", "")
        for i in range(20):
            th._queue_chat_event(
                {"type": "text_delta", "tabId": "t1", "content": f"q{i}"},
                task_id=task_id,
            )
        # Now insert a synchronous event; it must have the highest seq.
        th._append_chat_event(
            {"type": "task_done", "tabId": "t1", "content": "final"},
            task_id=task_id,
        )
        db = th._get_db()
        rows = db.execute(
            "SELECT seq, event_json FROM events WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ).fetchall()
        assert len(rows) == 21
        assert rows[-1]["seq"] == 20
        assert '"task_done"' in rows[-1]["event_json"]

    def test_save_task_result_drains_queue_first(self) -> None:
        """``_save_task_result`` must see every queued event in history."""
        task_id, _ = th._add_task("result-task", "")
        for i in range(50):
            th._queue_chat_event(
                {"type": "text_delta", "tabId": "t1", "content": f"q{i}"},
                task_id=task_id,
            )
        th._save_task_result("done", task_id=task_id)
        db = th._get_db()
        count = db.execute(
            "SELECT COUNT(*) FROM events WHERE task_id = ?", (task_id,),
        ).fetchone()[0]
        assert count == 50
        result = db.execute(
            "SELECT result, has_events FROM task_history WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert result["result"] == "done"
        assert result["has_events"] == 1
