# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regressions for area-E redundancy/race fixes in
``sorcar/persistence.py``.

Covers audit findings:

* **E-R2** — ``_load_subagent_rows_by_parent_task_id`` and
  ``_get_adjacent_task_by_chat_id`` rebuilt the 5-key session dict
  inline instead of calling ``_events_session_dict``; both must keep
  returning the exact session shape.
* **E-R5** — ``_add_task`` and ``_save_task_extra`` duplicated the
  subagent/parent_task_id extraction protocol with drift; both now
  share ``_extract_parent_task_id`` and consistently reject a payload
  carrying BOTH keys.
* **E-RC3** — the ``events`` table had no UNIQUE(task_id, seq), so a
  second process's inserts made this process's ``_next_seq_cache``
  stale and duplicate ``(task_id, seq)`` rows were written silently.
  Now ``idx_ev_task_seq`` refuses the duplicate, the writer retries
  once with re-read sequence numbers, and pre-index databases that
  already hold duplicates are resequenced before the index is built.

Every test uses a REAL temporary SQLite database, REAL threads and
REAL child processes.  No mocks, patches or doubles, no LLM calls.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import tempfile
import unittest
from pathlib import Path

import kiss.agents.sorcar.persistence as th

_PARENT_ID = "a" * 32
_OTHER_ID = "b" * 32


def _redirect(tmpdir: Path) -> tuple:
    """Point the persistence module at a private KISS home."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR, th._owner_state)
    kiss_dir = tmpdir / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    th._owner_state = None
    return saved


def _restore(saved: tuple) -> None:
    """Undo :func:`_redirect`."""
    th._close_db()
    (th._DB_PATH, th._db_conn, th._KISS_DIR, th._owner_state) = saved


def _event_seq_writer(kiss_dir: str, task_id: str, count: int) -> None:
    """Child process: append *count* events to *task_id* and flush.

    A fresh process has an empty ``_next_seq_cache``, so it re-reads
    ``MAX(seq)`` from the shared database — exactly the cross-process
    writer whose inserts make the parent process's cache stale.
    """
    import kiss.agents.sorcar.persistence as child_th

    child_th._KISS_DIR = Path(kiss_dir)
    child_th._DB_PATH = child_th._KISS_DIR / "sorcar.db"
    child_th._db_conn = None
    child_th._owner_state = None
    for i in range(count):
        child_th._queue_chat_event({"type": "child", "i": i}, task_id)
    child_th._flush_chat_events(task_id)
    child_th._close_db()


class _PersistenceTestCase(unittest.TestCase):
    """Base fixture giving each test a private database."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_rr_e_db_"))
        self.saved = _redirect(self.tmp)
        th._close_db()

    def tearDown(self) -> None:
        _restore(self.saved)

    def _all_event_seqs(self, task_id: str) -> list[int]:
        db = sqlite3.connect(th._DB_PATH)
        try:
            rows = db.execute(
                "SELECT seq FROM events WHERE task_id = ? ORDER BY seq",
                (task_id,),
            ).fetchall()
        finally:
            db.close()
        return [r[0] for r in rows]

    def _parent_column(self, task_id: str) -> str:
        db = sqlite3.connect(th._DB_PATH)
        try:
            row = db.execute(
                "SELECT parent_task_id FROM task_history WHERE id = ?",
                (task_id,),
            ).fetchone()
        finally:
            db.close()
        assert row is not None
        return str(row[0])


class TestExtractParentTaskId(_PersistenceTestCase):
    """E-R5 — one shared extraction protocol for both writers."""

    def test_add_task_rejects_both_keys(self) -> None:
        with self.assertRaises(ValueError):
            th._add_task("t", extra={
                "subagent": {"parent_task_id": _PARENT_ID},
                "parent_task_id": _OTHER_ID,
            })

    def test_add_task_rejects_both_keys_even_with_none_values(self) -> None:
        # Drift fixed: _add_task used value-is-None checks while
        # _save_task_extra used key presence.  The stricter key-
        # presence rule now applies to both.
        with self.assertRaises(ValueError):
            th._add_task("t", extra={
                "subagent": None,
                "parent_task_id": _OTHER_ID,
            })

    def test_save_task_extra_rejects_both_keys(self) -> None:
        task_id, _chat = th._add_task("t")
        with self.assertRaises(ValueError):
            th._save_task_extra(
                {
                    "subagent": {"parent_task_id": _PARENT_ID},
                    "parent_task_id": _OTHER_ID,
                },
                task_id=task_id,
            )

    def test_add_task_accepts_every_single_key_shape(self) -> None:
        nested, _ = th._add_task(
            "nested", extra={"subagent": {"parent_task_id": _PARENT_ID}},
        )
        self.assertEqual(self._parent_column(nested), _PARENT_ID)
        bare, _ = th._add_task("bare", extra={"subagent": _PARENT_ID})
        self.assertEqual(self._parent_column(bare), _PARENT_ID)
        flat, _ = th._add_task("flat", extra={"parent_task_id": _PARENT_ID})
        self.assertEqual(self._parent_column(flat), _PARENT_ID)
        none, _ = th._add_task("none")
        self.assertEqual(self._parent_column(none), "")
        garbage, _ = th._add_task("garbage", extra={"subagent": 42})
        self.assertEqual(self._parent_column(garbage), "")

    def test_save_task_extra_writes_parent_from_both_shapes(self) -> None:
        a, _ = th._add_task("a")
        th._save_task_extra({"parent_task_id": _PARENT_ID}, task_id=a)
        self.assertEqual(self._parent_column(a), _PARENT_ID)
        b, _ = th._add_task("b")
        th._save_task_extra(
            {"subagent": {"parent_task_id": _PARENT_ID}}, task_id=b,
        )
        self.assertEqual(self._parent_column(b), _PARENT_ID)
        # A malformed value never clobbers an existing parent link.
        th._save_task_extra({"parent_task_id": "junk"}, task_id=b)
        self.assertEqual(self._parent_column(b), _PARENT_ID)

    def test_save_task_extra_still_rejects_is_favorite(self) -> None:
        task_id, _chat = th._add_task("t")
        with self.assertRaises(ValueError):
            th._save_task_extra({"is_favorite": True}, task_id=task_id)


class TestSessionDictShapes(_PersistenceTestCase):
    """E-R2 — both loaders return the exact 5-key session shape."""

    _KEYS = {"task", "task_id", "events", "chat_id", "extra"}

    def test_load_subagent_rows_shape_and_events(self) -> None:
        parent_id, chat = th._add_task("parent")
        sub_id, _ = th._add_task(
            "sub", chat_id=chat, extra={"parent_task_id": parent_id},
        )
        th._queue_chat_event({"type": "hello"}, sub_id)
        th._flush_chat_events(sub_id)
        rows = th._load_subagent_rows_by_parent_task_id(parent_id)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(set(row.keys()), self._KEYS)
        self.assertEqual(row["task_id"], sub_id)
        self.assertEqual(row["task"], "sub")
        self.assertEqual(row["chat_id"], chat)
        events = row["events"]
        assert isinstance(events, list)
        self.assertEqual(events[0]["type"], "hello")
        self.assertIn("parent_task_id", str(row["extra"]))
        self.assertEqual(th._load_subagent_rows_by_parent_task_id(""), [])

    def test_get_adjacent_task_shape_both_directions(self) -> None:
        first_id, chat = th._add_task("first")
        second_id, _ = th._add_task("second", chat_id=chat)
        th._queue_chat_event({"type": "greeting"}, first_id)
        th._flush_chat_events(first_id)

        prev = th._get_adjacent_task_by_chat_id(chat, second_id, "prev")
        assert prev is not None
        self.assertEqual(set(prev.keys()), self._KEYS)
        self.assertEqual(prev["task_id"], first_id)
        self.assertEqual(prev["task"], "first")
        self.assertEqual(prev["chat_id"], chat)
        events = prev["events"]
        assert isinstance(events, list)
        self.assertEqual(events[0]["type"], "greeting")
        self.assertNotEqual(prev["extra"], "")

        nxt = th._get_adjacent_task_by_chat_id(chat, first_id, "next")
        assert nxt is not None
        self.assertEqual(set(nxt.keys()), self._KEYS)
        self.assertEqual(nxt["task_id"], second_id)
        self.assertIsNone(
            th._get_adjacent_task_by_chat_id(chat, first_id, "prev"),
        )


class TestUniqueEventSeq(_PersistenceTestCase):
    """E-RC3 — cross-process writers can no longer duplicate seqs."""

    def test_unique_index_exists_on_fresh_database(self) -> None:
        th._get_db()
        db = sqlite3.connect(th._DB_PATH)
        try:
            row = db.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND name='idx_ev_task_seq'"
            ).fetchone()
        finally:
            db.close()
        assert row is not None
        self.assertIn("UNIQUE", row[0].upper())

    def test_stale_cache_recovers_after_foreign_process_writes(self) -> None:
        task_id, _chat = th._add_task("shared task")
        for i in range(3):
            th._queue_chat_event({"type": "parent", "phase": 1, "i": i},
                                 task_id)
        th._flush_chat_events(task_id)
        # _next_seq_cache in THIS process now says next=3.

        ctx = multiprocessing.get_context()
        proc = ctx.Process(
            target=_event_seq_writer,
            args=(str(th._KISS_DIR), task_id, 3),
        )
        proc.start()
        proc.join(timeout=60)
        self.assertEqual(proc.exitcode, 0)
        self.assertEqual(self._all_event_seqs(task_id), list(range(6)))

        # The parent's cached next-seq (3) is now stale.  Without the
        # unique index these writes landed as DUPLICATE seqs 3..5; the
        # index refuses them and the writer retries with re-read seqs.
        for i in range(3):
            th._queue_chat_event({"type": "parent", "phase": 2, "i": i},
                                 task_id)
        th._flush_chat_events(task_id)

        seqs = self._all_event_seqs(task_id)
        self.assertEqual(seqs, list(range(9)))
        session = th._load_chat_events_by_task_id(task_id)
        assert session is not None
        events = session["events"]
        assert isinstance(events, list)
        self.assertEqual(len(events), 9)
        # Replay order: the parent's phase-2 events come after the
        # child's — no interleaving through reused sequence numbers.
        self.assertEqual(
            [e["type"] for e in events],
            ["parent"] * 3 + ["child"] * 3 + ["parent"] * 3,
        )

    def test_migration_resequences_preexisting_duplicates(self) -> None:
        # Build a database whose events table already carries
        # duplicate (task_id, seq) rows — what two pre-fix processes
        # left behind — then reopen it through the persistence layer.
        task_id, _chat = th._add_task("legacy task")
        th._queue_chat_event({"type": "ev", "i": 0}, task_id)
        th._queue_chat_event({"type": "ev", "i": 1}, task_id)
        th._flush_chat_events(task_id)
        th._close_db()

        raw = sqlite3.connect(th._DB_PATH)
        try:
            raw.execute("DROP INDEX idx_ev_task_seq")
            # A duplicate of seq 1 (a second process's stale-cache
            # write) and an exact duplicate pair at seq 5.
            raw.execute(
                "INSERT INTO events (task_id, seq, event_json, timestamp) "
                "VALUES (?, 1, '{\"type\": \"dup\", \"i\": 2}', 3.0)",
                (task_id,),
            )
            raw.execute(
                "INSERT INTO events (task_id, seq, event_json, timestamp) "
                "VALUES (?, 5, '{\"type\": \"dup\", \"i\": 3}', 4.0)",
                (task_id,),
            )
            raw.execute(
                "INSERT INTO events (task_id, seq, event_json, timestamp) "
                "VALUES (?, 5, '{\"type\": \"dup\", \"i\": 4}', 5.0)",
                (task_id,),
            )
            raw.commit()
        finally:
            raw.close()

        # Reopening runs _init_tables → _apply_index_ddl, which must
        # dedupe THEN build the unique index — without losing a row.
        th._get_db()
        self.assertEqual(self._all_event_seqs(task_id), list(range(5)))
        db = sqlite3.connect(th._DB_PATH)
        try:
            index_row = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' "
                "AND name='idx_ev_task_seq'"
            ).fetchone()
            ordered = db.execute(
                "SELECT event_json FROM events WHERE task_id = ? "
                "ORDER BY seq",
                (task_id,),
            ).fetchall()
        finally:
            db.close()
        self.assertIsNotNone(index_row)
        # First occurrence (by insertion id) keeps its position; every
        # original row survives.
        self.assertEqual(len(ordered), 5)
        self.assertIn("dup", ordered[2][0])

    def test_dedupe_is_a_noop_on_a_clean_database(self) -> None:
        # Covers _dedupe_event_seqs's no-duplicates branch, which the
        # production path only reaches through an index-creation
        # failure that implies duplicates exist.
        task_id, _chat = th._add_task("clean task")
        th._queue_chat_event({"type": "ev"}, task_id)
        th._flush_chat_events(task_id)
        db = th._get_db()
        th._dedupe_event_seqs(db)
        self.assertEqual(self._all_event_seqs(task_id), [0])


if __name__ == "__main__":
    unittest.main()
