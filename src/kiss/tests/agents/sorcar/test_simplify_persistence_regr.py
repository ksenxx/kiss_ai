# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking behavior before simplification.

Covers the exact code paths simplified in ``persistence.py`` and
``cli_steering.py``:

* legacy-schema migration (index creation, parent remap, flag coercion,
  orphan-event dropping),
* safe numeric coercers (``_safe_int`` / ``_safe_float``),
* ``_add_task`` / ``_save_task_extra`` parent-id shapes and error paths,
* ``_delete_task`` recursive cascade,
* ``_shutdown_persist_in_flight_results`` sentinel rewrite,
* prefix matching helpers,
* ``cli_steering`` box-geometry helpers.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.ui.cli.cli_steering import (
    _box_body_h,
    _box_h_for,
    _box_top_row,
    _partial_suffix_len,
)


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
        # Orphan event: task_id 99 has no task row -> must be dropped.
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
        # r6-persistence-H3: string "false"/"0" flags coerce to 0.
        assert parent["is_parallel"] == 0
        assert parent["is_worktree"] == 0
        assert child["parent_task_id"] == parent["id"]
        evs = conn.execute("SELECT task_id FROM events").fetchall()
        assert len(evs) == 1
        assert evs[0]["task_id"] == parent["id"]
        # All five indexes must exist after migration.
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
        # Second call is a no-op on the new schema.
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
        # Garbage parent id must not re-parent the row.
        th._save_task_extra({"parent_task_id": "nope"}, task_id=task_id)
        assert th._load_subagent_rows_by_parent_task_id(parent_id) == []
        th._save_task_extra({"parent_task_id": parent_id}, task_id=task_id)
        subs = th._load_subagent_rows_by_parent_task_id(parent_id)
        assert [s["task_id"] for s in subs] == [task_id]


class TestDeleteCascade(_TempDbTestBase):
    """Recursive sub-agent cascade delete, events included."""

    def test_grandchildren_and_events_deleted(self) -> None:
        parent_id, chat_id = th._add_task("parent")
        child_id, _ = th._add_task(
            "child", chat_id, extra={"parent_task_id": parent_id})
        grand_id, _ = th._add_task(
            "grand", chat_id, extra={"parent_task_id": child_id})
        other_id, other_chat = th._add_task("other")
        for tid in (parent_id, child_id, grand_id, other_id):
            th._append_chat_event({"type": "x"}, task_id=tid)
        assert th._delete_task(parent_id) is True
        db = th._get_db()
        left = db.execute("SELECT id FROM task_history").fetchall()
        assert [r["id"] for r in left] == [other_id]
        evs = db.execute("SELECT DISTINCT task_id FROM events").fetchall()
        assert [r["task_id"] for r in evs] == [other_id]
        assert th._delete_task(parent_id) is False
        assert th._chat_has_tasks(chat_id) is False
        assert th._chat_has_tasks(other_chat) is True


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
    """Cache round-trip and invalidation on add/save/delete."""

    def test_cache_invalidation(self) -> None:
        task_id, chat_id = th._add_task("taskA")
        th._save_task_result("resA", task_id=task_id)
        assert th._load_chat_context_text(chat_id) == "taskA\nresA"
        # Cached second call.
        assert th._load_chat_context_text(chat_id) == "taskA\nresA"
        t2, _ = th._add_task("taskB", chat_id)
        th._save_task_result("resB", task_id=t2)
        assert th._load_chat_context_text(chat_id) == (
            "taskA\nresA\ntaskB\nresB"
        )
        th._delete_task(t2)
        assert th._load_chat_context_text(chat_id) == "taskA\nresA"


class TestBoxGeometry:
    """cli_steering geometry helpers used by the anchored input box."""

    def test_box_body_h(self) -> None:
        assert _box_body_h("") == 3
        assert _box_body_h("one line") == 3
        assert _box_body_h("a\nb\nc\nd") == 4

    def test_box_h_for(self) -> None:
        assert _box_h_for("") == 5
        assert _box_h_for("a\nb\nc\nd\ne") == 7

    def test_box_top_row(self) -> None:
        assert _box_top_row(40) == 36
        assert _box_top_row(40, 10) == 31
        assert _box_top_row(3, 10) == 2  # clamped

    def test_partial_suffix_len(self) -> None:
        assert _partial_suffix_len("abc\x1b[20", "\x1b[201~") == 4
        assert _partial_suffix_len("abc", "\x1b[201~") == 0
        assert _partial_suffix_len("x\x1b", "\x1b[201~") == 1


def test_nan_never_reaches_json(tmp_path: Path) -> None:
    """_dumps_extra sanitises non-finite floats to null."""
    out = th._dumps_extra({"cost": float("nan"), "n": [float("inf"), 1]})
    assert out == '{"cost": null, "n": [null, 1]}'


class _PipeStdin:
    """Real pipe-backed stdin replacement (no mocks) for input loops."""

    def __enter__(self) -> _PipeStdin:
        self._r, self._w = os.pipe()
        self._saved_stdin = sys.stdin
        self._file = os.fdopen(self._r, "rb", buffering=0)
        sys.stdin = self._file  # type: ignore[assignment]
        return self

    def write(self, data: bytes) -> None:
        os.write(self._w, data)

    def __exit__(self, *exc: object) -> None:
        sys.stdin = self._saved_stdin
        try:
            self._file.close()
        except OSError:
            pass
        try:
            os.close(self._w)
        except OSError:
            pass


class TestAnchoredReplLoops:
    """End-to-end stdin pump loops driven through a real pipe."""

    def test_read_idle_line_submit(self) -> None:
        from kiss.ui.cli.cli_steering import AnchoredRepl
        with _PipeStdin() as stdin:
            repl = AnchoredRepl()
            stdin.write(b"hello world\r")
            line = repl.read_idle_line()
        assert line == "hello world"
        assert repl.box.history[-1] == "hello world"

    def test_read_idle_line_eof(self) -> None:
        from kiss.ui.cli.cli_steering import AnchoredRepl
        with _PipeStdin() as stdin:
            repl = AnchoredRepl()
            stdin.write(b"\x04")  # Ctrl+D on empty buffer
            line = repl.read_idle_line()
        assert line is None

    def test_read_idle_line_stdin_closed(self) -> None:
        from kiss.ui.cli.cli_steering import AnchoredRepl
        with _PipeStdin() as stdin:
            repl = AnchoredRepl()
            os.close(stdin._w)
            line = repl.read_idle_line()
            stdin._w = -1
        assert line is None

    def test_run_steering_loop_submit_and_done(self) -> None:
        from kiss.ui.cli.cli_steering import AnchoredRepl
        submitted: list[str] = []
        idle_calls: list[int] = []

        def on_submit(line: str) -> None:
            submitted.append(line)

        def on_abort() -> None:
            pass

        def is_done() -> bool:
            return bool(submitted)

        def on_idle() -> None:
            idle_calls.append(1)

        with _PipeStdin() as stdin:
            repl = AnchoredRepl()
            stdin.write(b"steer me\r")
            repl.run_steering_loop(on_submit, on_abort, is_done, on_idle)
        assert submitted == ["steer me"]


class TestEventDispatcherRender:
    """Dispatcher renders real events through a real ConsolePrinter."""

    def _make(self) -> Any:
        from kiss.core.print_to_console import ConsolePrinter
        from kiss.ui.cli.cli_client import _EventDispatcher
        return _EventDispatcher(ConsolePrinter(), tab_id="tab1")

    def test_text_and_thinking_deltas(self, capsys: pytest.CaptureFixture[str]) -> None:
        d = self._make()
        d.dispatch({"type": "text_delta", "text": "alpha "})
        d.dispatch({"type": "thinking_delta", "text": "beta"})
        d.dispatch({"type": "text_end"})
        out = capsys.readouterr().out  # type: ignore[union-attr]
        assert "alpha" in out
        assert "beta" in out

    def test_thinking_start_end_no_crash(self) -> None:
        d = self._make()
        d.dispatch({"type": "thinking_start"})
        d.dispatch({"type": "thinking_end"})

    def test_foreign_tab_events_dropped(self, capsys: pytest.CaptureFixture[str]) -> None:
        d = self._make()
        d.dispatch({"type": "text_delta", "text": "SECRET",
                    "tabId": "other"})
        out = capsys.readouterr().out  # type: ignore[union-attr]
        assert "SECRET" not in out

    def test_tool_call_input_reconstruction(self, capsys: pytest.CaptureFixture[str]) -> None:
        d = self._make()
        d.dispatch({
            "type": "tool_call", "name": "Edit", "path": "/tmp/f.py",
            "description": "desc-here", "old_string": "aaa",
            "new_string": "bbb", "extras": {"k1": "v1"},
        })
        out = capsys.readouterr().out  # type: ignore[union-attr]
        assert "Edit" in out

    def test_status_taskid_filter(self) -> None:
        d = self._make()
        d.current_task_id = "task-A"  # type: ignore[attr-defined]
        d.dispatch({"type": "status", "running": True, "taskId": "task-B"})
        assert not d.task_active.is_set()  # type: ignore[attr-defined]
        d.dispatch({"type": "status", "running": True, "taskId": "task-A"})
        assert d.task_active.is_set()  # type: ignore[attr-defined]


class TestClientDisconnectPaths:
    """Early-bail behavior when the daemon connection is gone."""

    def _client(self) -> Any:
        from kiss.core.print_to_console import ConsolePrinter
        from kiss.ui.cli.cli_client import CliClient
        c = CliClient(Path("/nonexistent.sock"), "/tmp", "tabX",
                      ConsolePrinter())
        c._closed.set()
        return c

    def test_request_cli_info_disconnected(self) -> None:
        from kiss.ui.cli.cli_client import _request_cli_info
        reply = _request_cli_info(self._client(), "help")  # type: ignore[arg-type]
        assert reply["error"] is True
        assert "Daemon connection lost" in reply["errorMessage"]
        assert reply["subtype"] == "help"

    def test_request_models_disconnected(self) -> None:
        from kiss.ui.cli.cli_client import _request_models
        assert _request_models(self._client()) == []  # type: ignore[arg-type]

    def test_submit_task_disconnected(self, capsys: pytest.CaptureFixture[str]) -> None:
        from kiss.ui.cli.cli_client import _submit_task
        client = self._client()
        _submit_task(client, "do it")  # type: ignore[arg-type]
        out = capsys.readouterr().out  # type: ignore[union-attr]
        assert "Daemon connection lost" in out
        assert "Time:" in out
        assert client.dispatcher.current_task_id == ""  # type: ignore[attr-defined]
