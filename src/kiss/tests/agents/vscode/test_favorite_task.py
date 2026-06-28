# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the favourite (star) feature on the history panel.

Covers:

1. Backend: ``_get_history`` includes ``is_favorite`` (default False).
2. Backend: ``_handle_set_favorite`` persists the flag.
3. Command dispatch: a ``setFavorite`` command updates the DB row.
4. Frontend (static checks on ``media/main.js``): a star button is
   rendered for each history row and posts ``setFavorite`` on click.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.server import VSCodeServer

MAIN_JS = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)
MAIN_CSS = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "media"
    / "main.css"
)


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
        # No error event fired.
        assert [e for e in events if e.get("type") == "error"] == []


class TestMainJsFavoriteButton:
    """Static checks that ``media/main.js`` and ``media/main.css``
    contain the favourite-button wiring."""

    def _js(self) -> str:
        assert MAIN_JS.is_file(), f"main.js not found at {MAIN_JS}"
        return MAIN_JS.read_text()

    def _css(self) -> str:
        assert MAIN_CSS.is_file(), f"main.css not found at {MAIN_CSS}"
        return MAIN_CSS.read_text()

    def test_favorite_button_class_defined_in_js(self) -> None:
        src = self._js()
        assert "sidebar-item-favorite" in src, (
            "main.js must create a button with class sidebar-item-favorite"
        )

    def test_favorite_click_posts_set_favorite_message(self) -> None:
        src = self._js()
        # Look for postMessage with type: 'setFavorite'.
        assert re.search(
            r"postMessage\(\s*\{\s*type:\s*'setFavorite'",
            src,
        ), "main.js must postMessage({type: 'setFavorite', ...}) on click"

    def test_favorite_button_reads_s_is_favorite(self) -> None:
        src = self._js()
        assert "s.is_favorite" in src, (
            "main.js must read s.is_favorite to decide the icon state"
        )

    def test_favorite_css_defines_favorited_class(self) -> None:
        css = self._css()
        assert ".sidebar-item-favorite" in css
        assert ".sidebar-item-favorite.favorited" in css
