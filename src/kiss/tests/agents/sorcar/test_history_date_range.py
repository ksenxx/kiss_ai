# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the History sidebar's date-range auto-fill.

The redesigned History filter bar pre-fills its From/To date inputs
with the FIRST and LAST task dates stored in ``~/.kiss/sorcar.db``.
The backend side of that feature is:

1. ``persistence._history_date_range()`` — returns the ``(min, max)``
   ``task_history.timestamp`` pair over the same row set the sidebar
   lists (i.e. excluding sub-agent rows), or ``(None, None)`` when
   the table has no listable rows.

2. ``VSCodeServer._get_history`` — stamps that pair onto every
   ``history`` event as ``dateRange: {"min": ..., "max": ...}`` so
   the webview can fill the inputs.

These tests drive the real persistence layer against a temp sqlite
DB and the real ``VSCodeServer`` broadcast path — no mocks of
project code (only the printer broadcast is captured in-memory,
exactly like ``test_subagent_history_click.py``).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.server import VSCodeServer


def _redirect(tmpdir: str) -> tuple:
    """Redirect the persistence DB to a temp dir; return saved state."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore(saved: tuple) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer whose broadcasts go into an in-memory list."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        ev = server.printer._inject_task_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        with lock:
            events.append(ev)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _set_timestamp(task_id: str, ts: float) -> None:
    """Force a task_history row's timestamp to a known value."""
    db = th._get_db()
    db.execute(
        "UPDATE task_history SET timestamp = ? WHERE id = ?", (ts, task_id)
    )
    db.commit()


class TestHistoryDateRange:
    """``_history_date_range`` and the ``history`` event's dateRange."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_db_returns_none_pair(self) -> None:
        assert th._history_date_range() == (None, None)

    def test_min_max_over_regular_rows(self) -> None:
        id1, _ = th._add_task("first task", chat_id="c1")
        id2, _ = th._add_task("middle task", chat_id="c2")
        id3, _ = th._add_task("last task", chat_id="c3")
        _set_timestamp(id1, 1_000.0)
        _set_timestamp(id2, 2_000.0)
        _set_timestamp(id3, 3_000.0)
        assert th._history_date_range() == (1_000.0, 3_000.0)

    def test_subagent_rows_are_ignored(self) -> None:
        parent_id, chat_id = th._add_task("parent task")
        _set_timestamp(parent_id, 5_000.0)
        sub_id, _ = th._add_task("sub task", chat_id=chat_id)
        th._save_task_extra(
            {
                "model": "m",
                "work_dir": "/tmp",
                "version": "v",
                "subagent": {"parent_task_id": parent_id},
            },
            task_id=sub_id,
        )
        # The sub-agent row sits OUTSIDE the visible range on both
        # sides at once is impossible — use the min side.
        _set_timestamp(sub_id, 10.0)
        assert th._history_date_range() == (5_000.0, 5_000.0)

    def test_history_event_carries_date_range(self) -> None:
        id1, _ = th._add_task("first task", chat_id="c1")
        id2, _ = th._add_task("last task", chat_id="c2")
        _set_timestamp(id1, 1_234.5)
        _set_timestamp(id2, 6_789.5)

        server, events = _make_server()
        server._get_history(query=None, offset=0, generation=0)

        hist = [e for e in events if e.get("type") == "history"]
        assert len(hist) == 1
        assert hist[0]["dateRange"] == {"min": 1_234.5, "max": 6_789.5}

    def test_history_event_date_range_on_empty_db(self) -> None:
        server, events = _make_server()
        server._get_history(query=None, offset=0, generation=0)

        hist = [e for e in events if e.get("type") == "history"]
        assert len(hist) == 1
        assert hist[0]["dateRange"] == {"min": None, "max": None}

    def test_search_history_event_also_carries_date_range(self) -> None:
        """The range reflects the WHOLE db even for filtered queries,
        so the auto-fill never narrows below the true first/last."""
        id1, _ = th._add_task("alpha task", chat_id="c1")
        id2, _ = th._add_task("beta task", chat_id="c2")
        _set_timestamp(id1, 100.0)
        _set_timestamp(id2, 200.0)

        server, events = _make_server()
        server._get_history(query="beta", offset=0, generation=0)

        hist = [e for e in events if e.get("type") == "history"]
        assert len(hist) == 1
        assert [s["title"] for s in hist[0]["sessions"]] == ["beta task"]
        assert hist[0]["dateRange"] == {"min": 100.0, "max": 200.0}
