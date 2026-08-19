"""End-to-end reproducing tests for bugs flagged by the gpt-5.5 review.

Each test reproduces a CRITICAL or HIGH bug from
``tmp/review_persistence.md``, ``tmp/review_vscode.md`` or
``tmp/review_sorcar_other.md``.  After the fix is in place every test
should pass; before the fix, each test failed deterministically.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar import persistence


@pytest.fixture
def temp_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Generator[Path]:
    """Point persistence at a temp DB and reset per-thread connection cache."""
    db_path = tmp_path / "sorcar.db"
    monkeypatch.setattr(persistence, "_DB_PATH", db_path)
    persistence._close_db()
    yield db_path
    persistence._close_db()


def _make_legacy_db(path: Path) -> None:
    """Create an old-schema (INTEGER id + extra JSON) DB at *path*."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.executescript(
        """
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
        """
    )
    conn.close()


def test_persist_bug1_migration_rolls_back_on_failure(
    temp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-migration must leave the legacy DB structurally intact.

    Inject an exception after the new tables are created but before the
    final rename; the legacy ``task_history`` (INTEGER id, ``extra``
    column) must still be reachable, not a half-converted hybrid.
    """
    _make_legacy_db(temp_db)
    conn = sqlite3.connect(str(temp_db), isolation_level=None)
    conn.execute(
        "INSERT INTO task_history (timestamp, task, result, extra) "
        "VALUES (?, ?, ?, ?)",
        (1.0, "legacy task", "ok", json.dumps({"model": "gpt-5"})),
    )
    conn.close()

    original_loads = persistence.json.loads

    def _crash_loads(s: str, *a: Any, **kw: Any) -> Any:
        if s == '{"model": "gpt-5"}':
            raise RuntimeError("simulated mid-migration crash")
        return original_loads(s, *a, **kw)

    monkeypatch.setattr(persistence.json, "loads", _crash_loads)
    with pytest.raises(RuntimeError, match="simulated"):
        persistence._get_db()
    monkeypatch.undo()
    persistence._close_db()

    conn = sqlite3.connect(str(temp_db), isolation_level=None)
    cols = {
        r[1]: (r[2] or "").upper()
        for r in conn.execute("PRAGMA table_info(task_history)").fetchall()
    }
    conn.close()
    assert cols.get("id", "").upper() == "INTEGER"
    assert "extra" in cols


def test_persist_bug2_migration_is_idempotent_after_prior_crash(
    temp_db: Path,
) -> None:
    """Leftover ``task_history__new`` from a prior crash must not block retry.

    Simulates: a previous migration crashed AFTER creating the temp
    tables but BEFORE the rename.  The next ``_get_db()`` boot must
    drop the stale tables and re-attempt cleanly.
    """
    _make_legacy_db(temp_db)
    conn = sqlite3.connect(str(temp_db), isolation_level=None)
    conn.execute(
        "INSERT INTO task_history (timestamp, task, result, extra) "
        "VALUES (?, ?, ?, ?)",
        (1.0, "legacy", "ok", "{}"),
    )
    conn.execute(
        "CREATE TABLE task_history__new (id TEXT PRIMARY KEY)"
    )
    conn.execute(
        "CREATE TABLE events__new (id INTEGER PRIMARY KEY)"
    )
    conn.close()

    db = persistence._get_db()
    rows = db.execute(
        "SELECT id, task, result FROM task_history"
    ).fetchall()
    assert len(rows) == 1
    assert persistence.is_task_history_id(rows[0]["id"])
    assert rows[0]["task"] == "legacy"


def test_persist_bug3_migration_handles_missing_events_table(
    temp_db: Path,
) -> None:
    """A legacy DB with no ``events`` table must still migrate cleanly."""
    _make_legacy_db(temp_db)
    conn = sqlite3.connect(str(temp_db), isolation_level=None)
    conn.execute(
        "INSERT INTO task_history (timestamp, task, result, extra) "
        "VALUES (?, ?, ?, ?)",
        (1.0, "task A", "ok", "{}"),
    )
    conn.execute("DROP TABLE events")
    conn.close()

    db = persistence._get_db()
    rows = db.execute("SELECT id, task FROM task_history").fetchall()
    assert len(rows) == 1
    assert persistence.is_task_history_id(rows[0]["id"])
    cols = {
        r[1]: (r[2] or "").upper()
        for r in db.execute("PRAGMA table_info(events)").fetchall()
    }
    assert cols.get("task_id", "").upper() == "TEXT"


def test_persist_bug11_add_task_rejects_legacy_int_parent_task_id(
    temp_db: Path,
) -> None:
    """A legacy integer ``parent_task_id`` must NOT be written as ``"123"``.

    The new column is UUID-hex; a numeric string would never match any
    real id, silently breaking parent/child sub-agent row lookups.
    """
    task_id, _ = persistence._add_task(
        "child task", "", extra={"subagent": {"parent_task_id": 123}}
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row["parent_task_id"] == ""


def test_persist_bug11_add_task_accepts_valid_uuid_parent_task_id(
    temp_db: Path,
) -> None:
    """A real 32-char hex UUID parent must round-trip."""
    parent = uuid.uuid4().hex
    child_id, _ = persistence._add_task(
        "child", "", extra={"subagent": {"parent_task_id": parent}}
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (child_id,),
    ).fetchone()
    assert row["parent_task_id"] == parent


def test_persist_bug11_save_task_extra_rejects_bogus_parent(
    temp_db: Path,
) -> None:
    """``_save_task_extra`` must also validate the parent id shape."""
    task_id, _ = persistence._add_task("task", "")
    persistence._save_task_extra(
        {"subagent": {"parent_task_id": "not-a-uuid"}},
        task_id=task_id,
        task=None,
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row["parent_task_id"] == ""


def test_persist_bug14_history_dict_exposes_typed_columns(
    temp_db: Path,
) -> None:
    """Consumers must see ``model``/``cost``/``tokens`` as top-level keys."""
    task_id, _ = persistence._add_task(
        "task A", "",
        extra={"model": "gpt-5", "cost": 1.25, "tokens": 4242},
    )
    entries = persistence._load_history()
    matched = [e for e in entries if e["id"] == task_id]
    assert len(matched) == 1
    e = matched[0]
    assert e["model"] == "gpt-5"
    assert e["cost"] == pytest.approx(1.25)
    assert e["tokens"] == 4242
    assert "extra" in e
    extra_parsed = json.loads(cast(str, e["extra"]))
    assert extra_parsed["model"] == "gpt-5"


def test_sorcar_bug1_is_task_history_id_contract() -> None:
    """``is_task_history_id`` is the canonical id-shape predicate."""
    assert persistence.is_task_history_id(uuid.uuid4().hex)
    assert not persistence.is_task_history_id("")
    assert not persistence.is_task_history_id(None)
    assert not persistence.is_task_history_id(123)
    assert not persistence.is_task_history_id(
        uuid.uuid4().hex.upper()
    )
    assert not persistence.is_task_history_id(str(uuid.uuid4()))


def test_sorcar_bug3_run_tasks_parallel_guards_none_parent(
    temp_db: Path,
) -> None:
    """A parent with no task id must not yield ``task-None__sub_*``.

    Drives the real fan-out engine with a parent whose
    ``_last_task_id`` is ``None`` and reads the tab id the child was
    actually given.
    """
    import threading

    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    class _Printer:
        """Thread-local-only printer: the engine needs nothing else."""

        def __init__(self) -> None:
            self._thread_local = threading.local()

    seen: list[str] = []
    original_run = ChatSorcarAgent.run

    def _record_tab(
        self: ChatSorcarAgent,
        prompt_template: str = "",
        **kwargs: Any,
    ) -> str:
        seen.append(str(getattr(self, "_tab_id", "")))
        return "success: true\nsummary: done"

    parent = ChatSorcarAgent("round1-parent")
    parent._last_task_id = None
    parent.printer = cast(Any, _Printer())
    try:
        ChatSorcarAgent.run = _record_tab  # type: ignore[method-assign]
        parent._run_tasks_parallel(["only task"], max_workers=1)
    finally:
        ChatSorcarAgent.run = original_run  # type: ignore[method-assign]

    assert len(seen) == 1
    assert seen[0].endswith("__sub_0")
    assert "None" not in seen[0], (
        f"sub-agent tab id leaked a None parent id: {seen[0]!r}"
    )
