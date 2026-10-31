# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Failed tasks stay out of the evaluation's "successful tasks" corpus.

``load_past_tasks`` used to skip only the ``Task failed``/``Task
interrupted`` prefixes and three exact markers.  In-process errors are
persisted differently (``KISS Error: ...``, ``<p>KISSError: ...``,
``<h3>Partial result: KISS Error: ...``), and a ``finish(success=False)``
run keeps a normal-looking HTML summary with ``success = 0``; all of them
were indexed as successful pages and used as probe targets.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from kiss.core.memoryfield.evaluate import is_failed_result, load_past_tasks

_LONG = "<p>" + "x" * 400 + "</p>"


def _make_db(path: Path) -> Path:
    """Create a ``history.db`` whose ``task_history`` has the ``success`` column."""
    rows = [
        ("ok-null", "KISS Error mentioned later is fine: " + _LONG, None),
        ("ok-one", _LONG, 1),
        ("kiss-error", "KISS Error: Agent Sorcar VS Code Session-1 exceeded budget" + _LONG, None),
        ("p-kisserror", "<p>KISSError: KISS Error: Non-retryable error</p>" + _LONG, None),
        ("partial", "<h3>Partial result: KISS Error: Agent X stalled</h3>" + _LONG, None),
        ("refusal", "ModelRefusalError: declined" + _LONG, None),
        ("verdict-false", "<h3>Task cancelled at your request</h3>" + _LONG, 0),
        ("task-failed", "Task failed: boom" + _LONG, None),
    ]
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TABLE task_history (id TEXT PRIMARY KEY, timestamp REAL NOT NULL,"
            " task TEXT NOT NULL, result TEXT DEFAULT '', parent_task_id TEXT DEFAULT '',"
            " success INTEGER)"
        )
        for i, (task_id, result, success) in enumerate(rows):
            conn.execute(
                "INSERT INTO task_history VALUES (?, ?, ?, ?, '', ?)",
                (task_id, 1000.0 - i, f"task {task_id}", result, success),
            )
    return path


def test_error_results_and_false_verdicts_are_not_loaded(tmp_path: Path) -> None:
    tasks = load_past_tasks(_make_db(tmp_path / "history.db"), limit=100, min_result_chars=20)
    assert [t.task_id for t in tasks] == ["ok-null", "ok-one"]


def test_is_failed_result_covers_every_persisted_failure_shape() -> None:
    assert is_failed_result("<p>done</p>", 0)
    for result in (
        "Task failed",
        "Task interrupted by server restart/shutdown",
        "Agent Failed Abruptly",
        "Task terminated unexpectedly (process killed)",
        "Task stopped by user",
        "KISS Error: Agent Sorcar exceeded budget",
        "<p>KISSError: KISS Error: Non-retryable error</p>",
        "<h3>Partial result: KISS Error: Agent X stalled</h3>",
    ):
        assert is_failed_result(result), result
    assert not is_failed_result("<p>All done.</p>", 1)
    assert not is_failed_result("<p>All done.</p>")
    assert not is_failed_result("<p>The KISS Error: line is quoted mid-text.</p>")
