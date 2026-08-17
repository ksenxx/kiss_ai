# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.agents.vscode.test_favorite_task``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.server.server import VSCodeServer


def _redirect(tmpdir: str):
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestHistoryIncludesFavoriteFlag:
    """``_get_history`` must populate ``is_favorite`` on every row."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_is_false_for_plain_task(self) -> None:
        th._add_task("plain")
        server, events = _make_server()

        server._get_history(query=None)

        hist = [e for e in events if e.get("type") == "history"]
        assert len(hist) == 1
        sessions = hist[0]["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["is_favorite"] is False

    def test_favorite_true_after_set(self) -> None:
        task_id, _ = th._add_task("starme")
        th._set_task_favorite(task_id, True)
        server, events = _make_server()

        server._get_history(query=None)

        sessions = [
            e for e in events if e.get("type") == "history"
        ][0]["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["is_favorite"] is True

    def test_favorite_false_after_unset(self) -> None:
        task_id, _ = th._add_task("toggle")
        th._set_task_favorite(task_id, True)
        th._set_task_favorite(task_id, False)
        server, events = _make_server()

        server._get_history(query=None)

        sessions = [
            e for e in events if e.get("type") == "history"
        ][0]["sessions"]
        assert sessions[0]["is_favorite"] is False


class TestSetFavoriteCommandDispatch:
    """Sending a ``setFavorite`` command must persist the flag."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_set_favorite_command_without_task_id_noop(self) -> None:
        """Missing taskId is silently dropped (no exception)."""
        server, events = _make_server()
        server._handle_command({"type": "setFavorite", "isFavorite": True})
        assert [e for e in events if e.get("type") == "error"] == []
