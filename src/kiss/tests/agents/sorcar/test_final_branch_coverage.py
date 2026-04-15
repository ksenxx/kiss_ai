"""Integration tests for remaining uncovered branches in sorcar/ and vscode/.

Targets:
  persistence.py: lines 125→129, 380→385, 403→404
  helpers.py: lines 133→134, 157→158
  server.py: lines 239→237, 241→237, 466→467, 670→671, 622 (remove pragma)

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

from kiss.agents.sorcar import persistence as th
from kiss.agents.vscode.helpers import rank_file_suggestions
from kiss.agents.vscode.server import VSCodeServer

_SavedState = tuple[Path, "sqlite3.Connection | None", Path]


def _redirect(tmpdir: str) -> _SavedState:
    """Redirect persistence to temp dir and return saved state."""
    old: _SavedState = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore(saved: _SavedState) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


# ---------------------------------------------------------------------------
# persistence.py — line 125→129: _get_db when DB file already exists
# ---------------------------------------------------------------------------


class TestGetDbWithExistingFile:
    """Cover the else branch of 'if not _DB_PATH.exists()' (line 125→129)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_db_when_file_already_exists(self) -> None:
        """_get_db skips stale WAL cleanup when DB file already exists."""
        # First call creates the DB file
        th._get_db()
        assert th._DB_PATH.exists()
        # Close connection but leave file on disk
        th._db_conn.close()  # type: ignore[union-attr]
        th._db_conn = None
        # Second call should skip the 'if not _DB_PATH.exists()' body → line 125→129
        db = th._get_db()
        assert db is not None


# ---------------------------------------------------------------------------
# persistence.py — line 380→385: _set_latest_chat_events with empty events
# ---------------------------------------------------------------------------


class TestSetLatestChatEventsEmpty:
    """Cover 'if has_ev:' False branch (line 380→385)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_set_empty_events(self) -> None:
        """Passing empty events list sets has_events=0 and skips insert."""
        task_id, chat_id = th._add_task("empty-events-task", chat_id=0)
        # First set some events, then clear them
        th._set_latest_chat_events(
            [{"type": "text_delta", "text": "hello"}], task_id=task_id
        )
        result = th._load_latest_chat_events_by_chat_id(chat_id)
        assert result is not None
        events = result["events"]
        assert isinstance(events, list)
        assert len(events) == 1
        # Now set empty events (line 380→385)
        th._set_latest_chat_events([], task_id=task_id)
        result = th._load_latest_chat_events_by_chat_id(chat_id)
        assert result is not None
        events = result["events"]
        assert isinstance(events, list)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# persistence.py — line 403→404: _append_chat_event with nonexistent task
# ---------------------------------------------------------------------------


class TestAppendChatEventNoTask:
    """Cover 'if resolved_task_id is None: return' (line 403→404)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_append_event_no_matching_task(self) -> None:
        """_append_chat_event returns early when task doesn't exist."""
        # Don't add any task, so resolved_task_id will be None
        th._append_chat_event({"type": "test"}, task="nonexistent-task-xyz")
        # No crash = success; the early return was taken


# ---------------------------------------------------------------------------
# helpers.py — lines 133→134 and 157→158: rank_file_suggestions with usage
# ---------------------------------------------------------------------------


class TestRankFileSuggestionsWithUsage:
    """Cover usage.get(path, 0) > 0 True branch and frequent loop."""

    def test_frequent_files_with_query(self) -> None:
        """Frequent files are filtered by query and sorted by end distance."""
        files = ["src/main.py", "src/main_test.py", "lib/main.py"]
        usage = {"src/main.py": 3, "lib/main.py": 1}
        result = rank_file_suggestions(files, "main", usage)
        frequent = [r for r in result if r["type"] == "frequent"]
        # Both src/main.py and lib/main.py match "main" and have usage
        assert len(frequent) == 2


# ---------------------------------------------------------------------------
# server.py — lines 239→237 and 241→237: _periodic_event_flush early exits
# ---------------------------------------------------------------------------


class TestPeriodicEventFlushEarlyExits:
    """Cover flush loop branches when task_id is None and events are empty."""

    def test_flush_with_no_task_id(self) -> None:
        """Flush loop exits early when agent._last_task_id is None."""
        server = VSCodeServer()
        agent = server._get_tab("0").agent
        agent._last_task_id = None
        stop = threading.Event()
        server._flush_interval = 0.05  # very fast flush
        rec_id = 999
        server.printer.start_recording(rec_id)
        # Run flush loop briefly — should skip because task_id is None
        t = threading.Thread(
            target=server._periodic_event_flush, args=(rec_id, stop, agent), daemon=True
        )
        t.start()
        time.sleep(0.15)  # allow 2-3 flush cycles
        stop.set()
        t.join(timeout=2)
        server.printer.stop_recording(rec_id)

    def test_flush_with_empty_events(self) -> None:
        """Flush loop skips DB write when events list is empty."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect(tmpdir)
        try:
            server = VSCodeServer()
            agent = server._get_tab("0").agent
            task_id, _ = th._add_task("flush-test")
            agent._last_task_id = task_id
            stop = threading.Event()
            server._flush_interval = 0.05
            rec_id = 888
            server.printer.start_recording(rec_id)
            # Don't broadcast anything → events empty → line 241→237
            t = threading.Thread(
                target=server._periodic_event_flush, args=(rec_id, stop, agent), daemon=True
            )
            t.start()
            time.sleep(0.15)
            stop.set()
            t.join(timeout=2)
            server.printer.stop_recording(rec_id)
        finally:
            if th._db_conn is not None:
                th._db_conn.close()
                th._db_conn = None
            _restore(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# server.py — line 466→467: _get_history with empty history
# ---------------------------------------------------------------------------


class TestGetHistoryBranches:
    """Cover _get_history branches for both empty and populated DB."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_server(self) -> tuple[VSCodeServer, list[dict]]:
        server = VSCodeServer()
        events: list[dict] = []
        orig = server.printer.broadcast

        def cap(ev: dict) -> None:
            events.append(ev)
            orig(ev)

        server.printer.broadcast = cap  # type: ignore[assignment]
        return server, events

    def test_get_history_with_entries(self) -> None:
        """_get_history with populated DB enters the loop (line 466→467)."""
        server, events = self._make_server()
        th._add_task("short task")
        th._add_task("a" * 60)  # long task for title truncation branch
        server._get_history(None, offset=0, generation=0)
        hist = [e for e in events if e.get("type") == "history"]
        assert len(hist) == 1
        sessions = hist[0]["sessions"]
        assert len(sessions) == 2
        # Verify truncation branch: long title gets "..."
        long_session = [s for s in sessions if len(s["preview"]) > 50][0]
        assert long_session["title"].endswith("...")

    def test_get_history_with_query(self) -> None:
        """_get_history with search query filters entries."""
        server, events = self._make_server()
        th._add_task("fix the bug")
        th._add_task("add feature")
        server._get_history("bug", offset=0, generation=0)
        hist = [e for e in events if e.get("type") == "history"]
        assert len(hist) == 1
        sessions = hist[0]["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["preview"] == "fix the bug"


# ---------------------------------------------------------------------------
# server.py — line 670→671: _complete with stale sequence number
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# server.py — line 622: remove # pragma: no branch
# ---------------------------------------------------------------------------


class TestCompleteFromActiveFileEqualSuffix:
    """Cover the False branch of 'if len(suffix) > len(best)' (line 622).

    When two candidates match with equal-length suffixes, the second
    iteration finds len(suffix) == len(best), making the condition False.
    """

    def test_equal_length_suffixes(self) -> None:
        """Two equal-length candidates: second doesn't replace first."""
        server = VSCodeServer()
        # Both "method_ab" and "method_cd" give suffix of length 2
        content = "method_ab method_cd"
        result = server._complete_from_active_file(
            "x method_", snapshot_content=content
        )
        # One of them wins (whichever is iterated first), length is 2
        assert len(result) == 2
        assert result in ("ab", "cd")
