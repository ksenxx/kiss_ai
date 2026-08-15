# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking behavior before simplification.

Covers the exact code paths simplified in ``persistence.py``:

* legacy-schema migration (index creation, parent remap, flag coercion,
  orphan-event dropping),
* safe numeric coercers (``_safe_int`` / ``_safe_float``),
* ``_add_task`` / ``_save_task_extra`` parent-id shapes and error paths,
* ``_shutdown_persist_in_flight_results`` sentinel rewrite,
* prefix matching helpers.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestSafeCoercers:
    """Edge cases of the finite-aware numeric coercers."""

    def test_safe_int(self) -> None:
        assert th._safe_int(None) == 0
        assert th._safe_int("") == 0
        assert th._safe_int("abc", 7) == 7
        assert th._safe_int(float("nan"), 3) == 3
        assert th._safe_int(float("inf")) == 0
        assert th._safe_int("42") == 42
        assert th._safe_int(4.9) == 4

    def test_safe_float(self) -> None:
        assert th._safe_float(None) == 0.0
        assert th._safe_float("") == 0.0
        assert th._safe_float("abc", 1.5) == 1.5
        assert th._safe_float(float("nan")) == 0.0
        assert th._safe_float(float("-inf"), 2.0) == 2.0
        assert th._safe_float("3.25") == 3.25


class TestLegacySchemaMigration(_TempDbTestBase):
    """Old INTEGER-id schema is ported in place, indexes included."""

    def _make_legacy_db(self) -> sqlite3.Connection:
        th._ensure_kiss_dir()
        conn = sqlite3.connect(
            str(th._DB_PATH), check_same_thread=False, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                task TEXT NOT NULL,
                has_events INTEGER DEFAULT 0,
                result TEXT DEFAULT '',
                chat_id CHAR(32) DEFAULT '',
                extra TEXT DEFAULT ''
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
        """)
        return conn

    def test_migration_end_to_end(self) -> None:
        conn = self._make_legacy_db()
        now = time.time()
        conn.execute(
            "INSERT INTO task_history (id, timestamp, task, result, chat_id,"
            " extra) VALUES (1, ?, 'parent', 'ok', 'chatA', ?)",
            (now, '{"model": "m1", "cost": 1.5, "is_parallel": "false",'
                  ' "is_worktree": "0", "tokens": "12"}'),
        )
        conn.execute(
            "INSERT INTO task_history (id, timestamp, task, result, chat_id,"
            " extra) VALUES (2, ?, 'child', 'ok', 'chatA', ?)",
            (now + 1, '{"subagent": {"parent_task_id": 1}}'),
        )
        conn.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (1, 0, '{}', ?)", (now,),
        )
        conn.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (99, 0, '{}', ?)", (now,),
        )
        assert th._migrate_old_schema_if_needed(conn) is True
        rows = conn.execute(
            "SELECT * FROM task_history ORDER BY timestamp"
        ).fetchall()
        assert len(rows) == 2
        parent, child = rows
        assert th.is_task_history_id(parent["id"])
        assert parent["model"] == "m1"
        assert parent["cost"] == 1.5
        assert parent["tokens"] == 12
        assert parent["is_parallel"] == 0
        assert parent["is_worktree"] == 0
        assert child["parent_task_id"] == parent["id"]
        evs = conn.execute("SELECT task_id FROM events").fetchall()
        assert len(evs) == 1
        assert evs[0]["task_id"] == parent["id"]
        idx = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        for name in (
            "idx_th_timestamp", "idx_th_task", "idx_th_chat_id",
            "idx_th_parent_task_id", "idx_ev_task_id",
        ):
            assert name in idx
        assert th._migrate_old_schema_if_needed(conn) is False
        conn.close()


class TestAddTaskAndExtra(_TempDbTestBase):
    """Parent-id shapes, NaN sanitisation, and error paths."""

    def test_add_task_nan_cost_and_uuid_ids(self) -> None:
        task_id, chat_id = th._add_task(
            "t1", extra={"model": "m", "cost": float("nan"), "tokens": 5},
        )
        assert th.is_task_history_id(task_id)
        assert th.is_task_history_id(chat_id)
        entry = th._load_history(limit=1)[0]
        assert entry["cost"] == 0.0
        assert entry["tokens"] == 5
        assert entry["model"] == "m"
        assert entry["result"] == "Agent Failed Abruptly"

    def test_add_task_parent_shapes(self) -> None:
        parent_id, chat_id = th._add_task("parent")
        c1, _ = th._add_task(
            "c1", chat_id, extra={"parent_task_id": parent_id})
        c2, _ = th._add_task(
            "c2", chat_id, extra={"subagent": {"parent_task_id": parent_id}})
        c3, _ = th._add_task("c3", chat_id, extra={"subagent": parent_id})
        subs = th._load_subagent_rows_by_parent_task_id(parent_id)
        assert [s["task_id"] for s in subs] == [c1, c2, c3]
        with pytest.raises(ValueError):
            th._add_task("bad", chat_id, extra={
                "parent_task_id": parent_id,
                "subagent": {"parent_task_id": parent_id},
            })

    def test_save_task_extra_paths(self) -> None:
        task_id, _ = th._add_task("t")
        th._save_task_extra(
            {"cost": float("inf"), "steps": 3, "endTs": 9,
             "unknown_key": "x"},
            task_id=task_id,
        )
        entry = th._load_history(limit=1)[0]
        assert entry["cost"] == 0.0
        assert entry["steps"] == 3
        assert entry["end_ts"] == 9
        with pytest.raises(ValueError):
            th._save_task_extra({"is_favorite": True}, task_id=task_id)
        parent_id, _ = th._add_task("p")
        with pytest.raises(ValueError):
            th._save_task_extra(
                {"parent_task_id": parent_id, "subagent": parent_id},
                task_id=task_id,
            )
        th._save_task_extra({"parent_task_id": "nope"}, task_id=task_id)
        assert th._load_subagent_rows_by_parent_task_id(parent_id) == []
        th._save_task_extra({"parent_task_id": parent_id}, task_id=task_id)
        subs = th._load_subagent_rows_by_parent_task_id(parent_id)
        assert [s["task_id"] for s in subs] == [task_id]


class TestShutdownPersist(_TempDbTestBase):
    """Pre-emptive sentinel rewrite touches only sentinel rows."""

    def test_rewrites_only_sentinel_rows(self) -> None:
        t1, _ = th._add_task("t1")
        t2, _ = th._add_task("t2")
        t3, _ = th._add_task("t3")
        th._save_task_result("done", task_id=t2)
        assert th._shutdown_persist_in_flight_results(set()) == 0
        assert th._shutdown_persist_in_flight_results({t1, t2, t3}) == 2
        results = {
            e["id"]: e["result"] for e in th._load_history()
        }
        assert results[t1] == "Task interrupted by server restart/shutdown"
        assert results[t2] == "done"
        assert results[t3] == "Task interrupted by server restart/shutdown"

    def test_recover_orphaned_tasks(self) -> None:
        t1, _ = th._add_task("t1")
        t2, _ = th._add_task("t2")
        # A row whose owning process is still alive is not an orphan;
        # clearing ``owner`` makes t1 look like the leftover of a
        # prior, now-dead process, which is what the sweep is for.
        th._get_db().execute(
            "UPDATE task_history SET owner = '' WHERE id = ?", (t1,)
        )
        assert th._recover_orphaned_tasks({t2}) == 1
        results = {e["id"]: e["result"] for e in th._load_history()}
        assert results[t1] == "Task terminated unexpectedly (process killed)"
        assert results[t2] == "Agent Failed Abruptly"


class TestPrefixMatch(_TempDbTestBase):
    """GLOB-escaped prefix matching, dedup, most-recent-first."""

    def test_prefix_match(self) -> None:
        th._add_task("fix the bug")
        time.sleep(0.01)
        th._add_task("fix the bug")
        time.sleep(0.01)
        th._add_task("fix the docs")
        time.sleep(0.01)
        th._add_task("weird [*] chars?")
        assert th._prefix_match_tasks("fix the") == [
            "fix the docs", "fix the bug",
        ]
        assert th._prefix_match_tasks("fix the", limit=1) == ["fix the docs"]
        assert th._prefix_match_tasks("weird [*] ch", limit=1) == ["weird [*] chars?"]
        assert th._prefix_match_tasks("", limit=1) == []
        assert th._prefix_match_tasks("fix", limit=0) == []


class TestChatContextCache(_TempDbTestBase):
    """Cache round-trip and invalidation on add/save."""

    def test_cache_invalidation(self) -> None:
        task_id, chat_id = th._add_task("taskA")
        th._save_task_result("resA", task_id=task_id)
        assert th._load_chat_context_text(chat_id) == "taskA\nresA"
        assert th._load_chat_context_text(chat_id) == "taskA\nresA"
        t2, _ = th._add_task("taskB", chat_id)
        th._save_task_result("resB", task_id=t2)
        assert th._load_chat_context_text(chat_id) == (
            "taskA\nresA\ntaskB\nresB"
        )


def test_nan_never_reaches_json(tmp_path: Path) -> None:
    """_dumps_extra sanitises non-finite floats to null."""
    out = th._dumps_extra({"cost": float("nan"), "n": [float("inf"), 1]})
    assert out == '{"cost": null, "n": [null, 1]}'
