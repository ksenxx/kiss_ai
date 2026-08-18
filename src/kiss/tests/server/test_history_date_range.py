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

import kiss.agents.sorcar.persistence as th
from kiss.server.server import VSCodeServer
from kiss.tests.agents.sorcar.test_history_date_range import (  # noqa: F401
    _redirect,
    _restore,
    _set_timestamp,
)


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
