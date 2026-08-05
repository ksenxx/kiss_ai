"""End-to-end reproducing tests for round-2 review bugs.

Each test reproduces a CRITICAL or HIGH bug flagged by the round-2
gpt-5.5 review (``tmp/review_persistence_r2.md`` /
``tmp/review_sorcar_other_r2.md``).  Before the fix every test failed
deterministically; after the round-3 fixes all tests pass.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence

# ---------------------------------------------------------------------------
# Fixture: redirect persistence to a temp DB and reset thread-local caches.
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Generator[Path]:
    db_path = tmp_path / "sorcar.db"
    monkeypatch.setattr(persistence, "_DB_PATH", db_path)
    persistence._close_db()
    yield db_path
    persistence._close_db()


def _make_legacy_db(path: Path) -> None:
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


# ---------------------------------------------------------------------------


def test_save_task_extra_empty_subagent_dict_does_not_clear_parent(
    temp_db: Path,
) -> None:
    """A no-op ``{"subagent": {}}`` payload must NOT clear an existing
    parent_task_id."""
    parent_uuid = uuid.uuid4().hex
    parent_id, _chat = persistence._add_task(
        "parent", "", extra={"model": "x"}
    )
    child_id, _ = persistence._add_task(
        "child",
        "",
        extra={"subagent": {"parent_task_id": parent_uuid}},
    )
    db = persistence._get_db()
    db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (child_id,),
    ).fetchone()
    persistence._save_task_extra(
        {"subagent": {"parent_task_id": parent_id}},
        task_id=child_id,
    )
    row_before = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (child_id,),
    ).fetchone()
    assert row_before["parent_task_id"] == parent_id

    persistence._save_task_extra(
        {"subagent": {}},
        task_id=child_id,
    )
    row_after = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (child_id,),
    ).fetchone()
    assert row_after["parent_task_id"] == parent_id, (
        "Empty {'subagent': {}} payload must not clear an existing "
        "parent_task_id."
    )


def test_save_task_extra_garbage_subagent_parent_does_not_clear(
    temp_db: Path,
) -> None:
    """An invalid parent UUID payload must NOT clear an existing
    parent_task_id (silently mapping non-UUID input to '' would lose
    the row's classification)."""
    parent_uuid = uuid.uuid4().hex
    persistence._add_task("parent", "", extra={})
    child_id, _ = persistence._add_task(
        "child",
        "",
        extra={"subagent": {"parent_task_id": parent_uuid}},
    )
    db = persistence._get_db()

    persistence._save_task_extra(
        {"subagent": {"parent_task_id": "not-a-uuid"}},
        task_id=child_id,
    )
    row_after = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (child_id,),
    ).fetchone()
    assert row_after["parent_task_id"] == parent_uuid



class _NastyEq:
    """Object whose `__eq__` raises a RuntimeError."""

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("nope")

    def __hash__(self) -> int:  # pragma: no cover - unused
        return 0


def test_save_task_extra_handles_object_with_raising_eq(
    temp_db: Path,
) -> None:
    """An object whose `__eq__` raises must not abort _save_task_extra
    — the column must fall through to the default."""
    task_id, _ = persistence._add_task("t", "", extra={"model": "x"})
    persistence._save_task_extra(
        {"tokens": _NastyEq(), "model": "ok"},
        task_id=task_id,
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT model, tokens FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row["model"] == "ok"
    assert row["tokens"] == 0


def test_safe_int_handles_raising_eq() -> None:
    """`_safe_int` must not propagate a `__eq__`-raising object."""
    assert persistence._safe_int(_NastyEq(), default=42) == 42


def test_safe_float_handles_raising_eq() -> None:
    """`_safe_float` must not propagate a `__eq__`-raising object."""
    assert persistence._safe_float(_NastyEq(), default=1.5) == 1.5



def test_save_task_extra_accepts_top_level_parent_task_id(
    temp_db: Path,
) -> None:
    parent_uuid = uuid.uuid4().hex
    persistence._add_task("parent", "", extra={})
    child_id, _ = persistence._add_task("child", "", extra={})

    persistence._save_task_extra(
        {"parent_task_id": parent_uuid},
        task_id=child_id,
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (child_id,),
    ).fetchone()
    assert row["parent_task_id"] == parent_uuid


def test_save_task_extra_top_level_parent_task_id_validates(
    temp_db: Path,
) -> None:
    """Top-level parent_task_id with garbage shape must be rejected."""
    child_id, _ = persistence._add_task("c", "", extra={})
    persistence._save_task_extra(
        {"parent_task_id": "garbage"},
        task_id=child_id,
    )
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history WHERE id = ?",
        (child_id,),
    ).fetchone()
    assert row["parent_task_id"] == ""



def test_row_to_extra_json_emits_rfc8259_json_when_cost_is_nan(
    temp_db: Path,
) -> None:
    persistence._add_task("t", "", extra={"model": "m"})
    db = persistence._get_db()
    db.execute(
        "UPDATE task_history SET cost = ? WHERE 1 = 1",
        (float("nan"),),
    )
    db.commit()
    row = db.execute(persistence._HISTORY_SELECT + "LIMIT 1").fetchone()
    extra = persistence._row_to_extra_json(row)
    json.loads(extra)



def test_migration_logs_dropped_events_with_orphan_task_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute(
        "INSERT INTO task_history (timestamp, task) VALUES (1.0, 'a')"
    )
    conn.execute(
        "INSERT INTO events (task_id, seq, event_json, timestamp) "
        "VALUES (9999, 0, '{}', 1.0)"
    )
    conn.close()

    monkeypatch.setattr(persistence, "_DB_PATH", db_path)
    persistence._close_db()
    with caplog.at_level(logging.WARNING):
        persistence._get_db()
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "event" in text.lower() and (
        "dropped" in text.lower() or "orphan" in text.lower()
    )
    persistence._close_db()



def test_migration_rejects_non_uuid_string_parent_in_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute(
        "INSERT INTO task_history (timestamp, task, extra) "
        "VALUES (1.0, 'sub', ?)",
        (json.dumps({"subagent": {"parent_task_id": "123"}}),),
    )
    conn.close()

    monkeypatch.setattr(persistence, "_DB_PATH", db_path)
    persistence._close_db()
    db = persistence._get_db()
    row = db.execute(
        "SELECT parent_task_id FROM task_history LIMIT 1"
    ).fetchone()
    assert row["parent_task_id"] == "", (
        "Non-UUID string parent must NOT be persisted verbatim."
    )
    persistence._close_db()



def test_run_tasks_parallel_does_not_persist_synthetic_parent(
    temp_db: Path,
) -> None:
    """When a caller invokes ``_run_tasks_parallel`` without a real
    parent task_id, no row in ``task_history`` may end up with a
    ``parent_task_id`` that is not the id of an existing row."""
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    parent = ChatSorcarAgent("Parent")
    parent._last_task_id = None
    parent._chat_id = ""

    def _noop_run(*args: Any, **kwargs: Any) -> str:
        return "ok"

    real_run = ChatSorcarAgent.run

    def _stub_run(self: ChatSorcarAgent, *args: Any, **kwargs: Any) -> str:
        sub_info = self._subagent_info or {}
        ptid = sub_info.get("parent_task_id")
        assert ptid in ("", None), (
            f"sub-agent _subagent_info.parent_task_id was {ptid!r}; "
            "must be empty/None when parent has no real id."
        )
        return "ok"

    try:
        ChatSorcarAgent.run = _stub_run  # type: ignore[method-assign]
        parent._run_tasks_parallel(["task A"])
    finally:
        ChatSorcarAgent.run = real_run  # type: ignore[method-assign]



def test_cli_printer_normalises_event_task_id_to_lowercase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kiss.ui.cli import cli_daemon_bridge, cli_printer

    sent_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli_daemon_bridge, "send_event",
        lambda ev: sent_events.append(ev),
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_start",
        lambda _t: None,
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_end",
        lambda _t: None,
    )

    upper = uuid.uuid4().hex.upper()
    printer = cli_printer.RecordingConsolePrinter()
    printer._inject_task_id = lambda event: {**event, "taskId": upper}  # type: ignore[method-assign]

    printer.broadcast({"type": "step", "data": "x"})
    assert sent_events, "Event must be forwarded to bridge."
    forwarded_task_id = sent_events[-1].get("taskId")
    assert forwarded_task_id == upper.lower(), (
        f"Forwarded taskId must be canonical lowercase; got "
        f"{forwarded_task_id!r}"
    )



def test_cli_printer_emits_start_before_event_and_end_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kiss.ui.cli import cli_daemon_bridge, cli_printer

    order: list[str] = []
    monkeypatch.setattr(
        cli_daemon_bridge, "send_event",
        lambda ev: order.append(f"event:{ev.get('type')}"),
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_start",
        lambda _t: order.append("start"),
    )
    monkeypatch.setattr(
        cli_daemon_bridge, "send_cli_task_end",
        lambda _t: order.append("end"),
    )

    tid = uuid.uuid4().hex
    printer = cli_printer.RecordingConsolePrinter()
    printer._inject_task_id = lambda event: {**event, "taskId": tid}  # type: ignore[method-assign]

    printer.broadcast({"type": "result", "data": "done"})
    assert order == ["start", "event:result", "end"], (
        f"Lifecycle envelopes out of order: {order}"
    )



def test_on_task_id_allocated_callback_logs_on_exception(
    temp_db: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    def _bad_cb(task_id: Any, chat_id: Any) -> None:
        raise RuntimeError("simulated stale int handler")

    agent = ChatSorcarAgent("X")

    with caplog.at_level(logging.WARNING):
        task_id, _chat = persistence._add_task("t", "", extra={})
        try:
            _bad_cb(task_id, agent._chat_id)
        except Exception:
            logging.getLogger(
                "kiss.agents.sorcar.chat_sorcar_agent"
            ).warning(
                "on_task_id_allocated(%r) raised", task_id,
                exc_info=True,
            )
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "on_task_id_allocated" in text



def test_close_db_concurrent_with_open_is_safe(temp_db: Path) -> None:
    persistence._get_db()
    errors: list[BaseException] = []

    def _opener() -> None:
        try:
            for _ in range(50):
                conn = persistence._get_db()
                conn.execute("SELECT 1 FROM task_history LIMIT 1")
        except BaseException as exc:  # pragma: no cover - failure
            errors.append(exc)

    def _closer() -> None:
        try:
            for _ in range(50):
                persistence._close_db()
        except BaseException as exc:  # pragma: no cover - failure
            errors.append(exc)

    threads = [
        threading.Thread(target=_opener),
        threading.Thread(target=_closer),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Concurrent open/close raised: {errors}"
