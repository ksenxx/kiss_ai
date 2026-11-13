# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for race conditions in sorcar/ and server/.

These tests deliberately interleave concurrent threads to reproduce
specific race conditions identified in
``kiss.agents.sorcar.persistence`` and
``kiss.server.autocomplete``.

Each test inserts a small ``random.uniform(0, 0.05)`` sleep at the
suspected racing statement, then runs the involved threads many times
to make the failure deterministic.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.tests.persistence_helpers import close_db

_PEER_SCRIPT = """
import sys

from kiss.agents.sorcar import persistence

persistence._save_task_result(sys.argv[2], task_id=sys.argv[1])
"""


def _redirect(tmpdir: str):
    """Point persistence at a temporary DB and reset cached state."""
    old = (th._DB_PATH, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._invalidate_chat_context_cache("")
    return old


def _restore(saved):
    (th._DB_PATH, th._KISS_DIR) = saved
    th._invalidate_chat_context_cache("")


class TestChatContextCacheStaleWrite:
    """Race 1: ``_load_chat_context_text`` writes stale data into cache.

    Reader R1 misses the cache, runs the SQL read (under the read
    lock) and gets snapshot ``D1``.  Before R1 stores the result, a
    writer W commits a new task and invalidates the cache.  A second
    reader R2 misses, runs the SQL read, gets ``D2`` (with W's new
    task) and stores ``D2`` into the cache.  R1 then stores ``D1``,
    permanently overwriting the fresh ``D2`` and returning stale data
    on every subsequent call until another invalidation happens.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        th._stop_event_writer()
        close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stale_cache_write(self) -> None:
        chat_id = "race-chat"
        th._add_task("task-zero", chat_id=chat_id)

        r1_in_sql = threading.Event()
        let_r1_finish = threading.Event()

        original = th._load_chat_context
        call_count = {"n": 0}
        cc_lock = threading.Lock()

        def staged_load_chat_context(cid: str):
            with cc_lock:
                call_count["n"] += 1
                n = call_count["n"]
            data = original(cid)
            if n == 1:
                r1_in_sql.set()
                time.sleep(random.uniform(0.001, 0.05))
                let_r1_finish.wait(timeout=5)
            return data

        th._load_chat_context = staged_load_chat_context  # type: ignore[assignment]

        try:
            def reader_one() -> None:
                th._load_chat_context_text(chat_id)

            t_r1 = threading.Thread(target=reader_one)
            t_r1.start()
            assert r1_in_sql.wait(timeout=5)

            th._add_task("task-one", chat_id=chat_id)

            def reader_two() -> None:
                th._load_chat_context_text(chat_id)

            t_r2 = threading.Thread(target=reader_two)
            t_r2.start()
            t_r2.join(timeout=5)

            assert "task-one" in th._load_chat_context_text(chat_id)

            let_r1_finish.set()
            t_r1.join(timeout=5)

            cached_after = th._load_chat_context_text(chat_id)
            assert "task-one" in cached_after, (
                "stale chat-context cache survived a concurrent invalidation"
            )
        finally:
            th._load_chat_context = original  # type: ignore[assignment]
            let_r1_finish.set()


class TestChatContextCachePeerCommit:
    """Race 2 (A11-5 astra / A11-4 fable): a peer process's commit left the cached text stale.

    The invalidation of :func:`_invalidate_chat_context_cache` reaches
    this process only; a result another process committed (a CLI run,
    a database sync) was served from the cache for ever after.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        th._stop_event_writer()
        close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_a_peer_process_commit_refreshes_the_cached_text(self) -> None:
        chat_id = "peer-chat"
        task_id, _ = th._add_task("peer task", chat_id=chat_id)
        th._save_task_result("old result", task_id=task_id)
        assert th._load_chat_context_text(chat_id) == "peer task\nold result"

        peer = subprocess.run(
            [sys.executable, "-c", _PEER_SCRIPT, task_id, "new result from peer"],
            env={**os.environ, "KISS_HOME": str(th._KISS_DIR)},
            capture_output=True, text=True, timeout=120, check=False,
        )
        assert peer.returncode == 0, peer.stderr
        assert th._load_chat_context(chat_id)[0]["result"] == "new result from peer"
        assert th._load_chat_context_text(chat_id) == "peer task\nnew result from peer"

        # Nothing committed since: the text comes from the cache, no query.
        statements: list[str] = []
        db = th._get_db()
        db.set_trace_callback(statements.append)
        try:
            assert th._load_chat_context_text(chat_id) == "peer task\nnew result from peer"
        finally:
            db.set_trace_callback(None)
        assert not [s for s in statements if s.lstrip().upper().startswith("SELECT")], statements
