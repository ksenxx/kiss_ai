# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the ``steer_inputs`` table in history.db.

Text typed into a RUNNING task's composer (steer mode) never gets a
``task_history`` row, so it is remembered in ``steer_inputs`` and
merged into the two composer-history queries:

* ``_prefix_match_tasks`` — the prefix completions / ghost text.
* ``_load_input_texts`` — the ArrowUp history.

Every test runs against a real sqlite file in a temp directory.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.tests.persistence_helpers import close_db


def _redirect(tmpdir: str) -> tuple[Path, Path]:
    """Redirect the persistence DB to a temp dir; return saved state."""
    saved = (th._DB_PATH, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    return saved


def _restore(saved: tuple[Path, Path]) -> None:
    th._DB_PATH, th._KISS_DIR = saved


def _set_task_timestamp(task_id: str, ts: float) -> None:
    th._get_db().execute(
        "UPDATE task_history SET timestamp = ? WHERE id = ?", (ts, task_id),
    )


def _set_steer_timestamp(text: str, ts: float) -> None:
    th._get_db().execute(
        "UPDATE steer_inputs SET timestamp = ? WHERE text = ?", (ts, text),
    )


def _peer_adds_a_task_and_reads(
    go: threading.Event, done: threading.Event, texts: set[str], newest: list[float],
) -> None:
    """Another process's turn: commit a task, then read the history as the page does.

    A raw connection stands for the other process (``_rw_lock`` orders
    this process only).  *texts* receives the texts the read returned
    and *newest* the watermark the reply would carry.
    """
    assert go.wait(timeout=15)
    peer = sqlite3.connect(str(th._DB_PATH), timeout=30)
    try:
        peer.execute(
            "INSERT INTO task_history (id, chat_id, timestamp, task, result) "
            "VALUES ('peer-task', 'peer-chat', ?, 'peer prompt', 'Agent Failed Abruptly')",
            (time.time(),),
        )
        peer.commit()
        rows = peer.execute("SELECT task, timestamp FROM task_history").fetchall()
    finally:
        peer.close()
    texts.update(str(task) for task, _ts in rows)
    newest.append(max(float(ts) for _task, ts in rows))
    done.set()


def _input_history(limit: int = th._MAX_INPUT_HISTORY, since: float = 0.0) -> list[str]:
    """The texts of ``_load_input_texts(limit, since)``, the ArrowUp history's order."""
    return [text for text, _used in th._load_input_texts(limit, since)]


def _steer_rows() -> list[tuple[str, float]]:
    rows = th._get_db().execute(
        "SELECT text, timestamp FROM steer_inputs ORDER BY timestamp",
    ).fetchall()
    return [(r["text"], r["timestamp"]) for r in rows]


class TestSteerInputs:
    """Behavioural tests for recording and querying steer-mode texts."""

    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.saved = _redirect(self.tmp)

    def teardown_method(self) -> None:
        close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_table_created(self) -> None:
        row = th._get_db().execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='steer_inputs'"
        ).fetchone()
        assert row is not None

    def test_record_upserts_and_refreshes_timestamp(self) -> None:
        th._record_steer_input("focus on the tests")
        _set_steer_timestamp("focus on the tests", 1.0)
        th._record_steer_input("focus on the tests")
        rows = _steer_rows()
        assert [r[0] for r in rows] == ["focus on the tests"]
        assert rows[0][1] > 1.0

    def test_blank_text_ignored(self) -> None:
        for blank in ("", "   ", "\n\t"):
            th._record_steer_input(blank)
        assert _steer_rows() == []

    def test_cap_evicts_oldest(self) -> None:
        saved_cap = th._MAX_STEER_INPUTS
        th._MAX_STEER_INPUTS = 3
        try:
            for i in range(3):
                th._record_steer_input(f"msg {i}")
                _set_steer_timestamp(f"msg {i}", float(i))
            # A repeat of an existing text must not evict anything.
            th._record_steer_input("msg 1")
            assert sorted(r[0] for r in _steer_rows()) == [
                "msg 0", "msg 1", "msg 2",
            ]
            # A brand-new text beyond the cap evicts the oldest (msg 0).
            th._record_steer_input("msg 3")
            assert sorted(r[0] for r in _steer_rows()) == [
                "msg 1", "msg 2", "msg 3",
            ]
        finally:
            th._MAX_STEER_INPUTS = saved_cap

    def test_a_write_delayed_at_the_lock_lands_after_the_watermark(self) -> None:
        """A steer input that waited for the write lock is newer than a reader's watermark.

        The composer page refreshes its history with ``since`` set to
        the newest use it was sent (``_load_input_texts(since=...)``).
        A write that stamped its time, then waited for the write lock
        while a newer text committed and a reader took that text's time
        as its watermark, would commit with an OLDER time: every later
        delta skips it and only a full reload shows it.  ``sys.settrace``
        is thread-local: it parks only the delayed writer, at the entry
        of ``_capped_upsert``, before it takes the write lock.
        """
        th._get_db()
        parked, resume = threading.Event(), threading.Event()
        upsert = th._capped_upsert.__code__

        def trace(frame, event, arg):  # type: ignore[no-untyped-def]
            if frame.f_code is upsert and event == "call" and not parked.is_set():
                parked.set()
                assert resume.wait(timeout=15)
            return None

        def delayed() -> None:
            sys.settrace(trace)
            try:
                th._record_steer_input("delayed steer")
            finally:
                sys.settrace(None)
                th._close_thread_db()

        thread = threading.Thread(target=delayed, daemon=True)
        thread.start()
        assert parked.wait(timeout=15), "the delayed writer never reached the upsert"
        th._record_steer_input("newer steer")
        watermark = max(used for _text, used in th._load_input_texts())
        resume.set()
        thread.join(30)
        assert not thread.is_alive(), "the delayed writer did not finish"
        assert sorted(r[0] for r in _steer_rows()) == ["delayed steer", "newer steer"]
        assert [t for t, _used in th._load_input_texts(since=watermark)] == [
            "delayed steer",
        ], "regression: the delayed write carries a time older than the watermark"

    def test_a_task_stamps_its_time_under_the_databases_write_lock(self) -> None:
        """A task another process overtook between its time stamp and its INSERT is not lost.

        ``_rw_lock`` orders this process only.  ``_add_task`` used to
        evaluate ``time.time()`` and only then let the INSERT take the
        database's write lock; a peer process committing a newer task in
        between, and a reader taking that task's time as its watermark,
        left the task to commit with an OLDER time: every later delta
        (``timestamp > watermark``) skipped it.  The connection's trace
        callback fires when the INSERT starts, after its parameters are
        evaluated and before it takes the lock: the peer's turn.  The
        page's view (what the reader got, plus the delta since its
        watermark) must hold both prompts.
        """
        go, done = threading.Event(), threading.Event()
        texts: set[str] = set()
        newest: list[float] = []
        peer = threading.Thread(
            target=_peer_adds_a_task_and_reads, args=(go, done, texts, newest), daemon=True,
        )
        peer.start()
        db = th._get_db()

        def trace(statement: str) -> None:
            if statement.startswith("INSERT INTO task_history") and not go.is_set():
                go.set()
                done.wait(timeout=2)  # held off by the lock once it is taken first

        db.set_trace_callback(trace)
        try:
            th._add_task("overtaken prompt")
        finally:
            db.set_trace_callback(None)
        assert done.wait(timeout=30), "the peer never finished"
        peer.join(5)
        delta = {t for t, _used in th._load_input_texts(since=newest[0])}
        assert {"overtaken prompt", "peer prompt"} <= texts | delta, (texts, delta)

    def test_prefix_match_merges_steer_and_tasks_by_recency(self) -> None:
        old_task, _chat = th._add_task("add a README")
        _set_task_timestamp(old_task, 10.0)
        th._record_steer_input("add tests for the parser")
        _set_steer_timestamp("add tests for the parser", 20.0)
        new_task, _chat = th._add_task("add a CHANGELOG")
        _set_task_timestamp(new_task, 30.0)
        th._record_steer_input("unrelated steer text")

        assert th._prefix_match_tasks("add") == [
            "add a CHANGELOG",
            "add tests for the parser",
            "add a README",
        ]
        assert th._prefix_match_tasks("add t") == ["add tests for the parser"]
        assert th._prefix_match_tasks("add", limit=2) == [
            "add a CHANGELOG", "add tests for the parser",
        ]

    def test_prefix_match_dedups_text_present_in_both_tables(self) -> None:
        task_id, _chat = th._add_task("run the linter")
        _set_task_timestamp(task_id, 10.0)
        th._record_steer_input("run the linter")
        _set_steer_timestamp("run the linter", 50.0)
        other, _chat = th._add_task("run the tests")
        _set_task_timestamp(other, 20.0)

        # The steer copy is newer, so the shared text ranks first — once.
        assert th._prefix_match_tasks("run") == [
            "run the linter", "run the tests",
        ]

    def test_prefix_match_excludes_exact_query_and_escapes_glob(self) -> None:
        th._record_steer_input("fix [bug] *now*?")
        th._record_steer_input("fix [bug] *now*? please")
        assert th._prefix_match_tasks("fix [bug] *now*?") == [
            "fix [bug] *now*? please",
        ]
        # Case-sensitive, like the task_history match.
        assert th._prefix_match_tasks("FIX") == []

    def test_prefix_match_ignores_subagent_rows_but_keeps_steer(self) -> None:
        parent, _chat = th._add_task("deploy the service")
        _set_task_timestamp(parent, 10.0)
        th._add_task(
            "deploy the sub-agent", extra={"parent_task_id": parent},
        )
        th._record_steer_input("deploy carefully")
        assert th._prefix_match_tasks("deploy") == [
            "deploy carefully", "deploy the service",
        ]

    def test_load_input_history_merges_and_orders(self) -> None:
        t1, _chat = th._add_task("first task")
        _set_task_timestamp(t1, 10.0)
        th._record_steer_input("steer one")
        _set_steer_timestamp("steer one", 20.0)
        t2, _chat = th._add_task("second task")
        _set_task_timestamp(t2, 30.0)
        th._record_steer_input("first task")
        _set_steer_timestamp("first task", 40.0)
        th._add_task("hidden sub-agent", extra={"parent_task_id": t2})

        assert _input_history() == [
            "first task", "second task", "steer one",
        ]

    def test_load_input_history_empty_db(self) -> None:
        assert _input_history() == []
