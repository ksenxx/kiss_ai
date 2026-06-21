# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the History sidebar's per-task meta line at
the backend layer.

The History sidebar renders a single dot-separated meta line under
each row's workspace line::

    <model> • <wt|no-wt> • <parallel|sequential>
        • <auto-commit|manual-commit>

The four values come from the task's persisted ``extra`` JSON
(written by ``_TaskRunnerMixin._run_task_inner`` at task end), so
the server's ``_get_history`` must surface them on every session
payload::

    session["model"]            : str
    session["is_worktree"]      : bool
    session["is_parallel"]      : bool
    session["auto_commit_mode"] : bool

Rows whose persisted ``extra`` lacks any of these fields must fall
back to empty / False (the frontend interprets an empty ``model``
as "render no meta line").
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.server import VSCodeServer


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


def _history_sessions(events: list[dict]) -> list[dict]:
    hist = [e for e in events if e.get("type") == "history"]
    assert len(hist) == 1, f"expected one history event, got {len(hist)}"
    sessions = hist[0]["sessions"]
    assert isinstance(sessions, list)
    return sessions  # type: ignore[no-any-return]


class TestHistoryMetaFieldsSurfaced:
    """``_get_history`` must populate the four meta fields."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_defaults_for_task_without_extra(self) -> None:
        """A task with no persisted ``extra`` gets empty model and
        False for every boolean flag — the frontend will omit the
        meta line entirely (empty model)."""
        th._add_task("plain task")
        server, events = _make_server()

        server._get_history(query=None)

        sessions = _history_sessions(events)
        assert len(sessions) == 1
        s = sessions[0]
        assert s["model"] == ""
        assert s["is_worktree"] is False
        assert s["is_parallel"] is False
        assert s["auto_commit_mode"] is False

    def test_all_flags_on(self) -> None:
        task_id, _ = th._add_task("flags on")
        th._save_task_extra(
            {
                "model": "gpt-5",
                "is_worktree": True,
                "is_parallel": True,
                "auto_commit_mode": True,
            },
            task_id=task_id,
        )
        server, events = _make_server()

        server._get_history(query=None)

        s = _history_sessions(events)[0]
        assert s["model"] == "gpt-5"
        assert s["is_worktree"] is True
        assert s["is_parallel"] is True
        assert s["auto_commit_mode"] is True

    def test_all_flags_off(self) -> None:
        task_id, _ = th._add_task("flags off")
        th._save_task_extra(
            {
                "model": "claude-3.7-sonnet",
                "is_worktree": False,
                "is_parallel": False,
                "auto_commit_mode": False,
            },
            task_id=task_id,
        )
        server, events = _make_server()

        server._get_history(query=None)

        s = _history_sessions(events)[0]
        assert s["model"] == "claude-3.7-sonnet"
        assert s["is_worktree"] is False
        assert s["is_parallel"] is False
        assert s["auto_commit_mode"] is False

    def test_model_only_legacy_extra(self) -> None:
        """A row whose ``extra`` carries only ``model`` (no boolean
        flags) must surface the model verbatim and default every
        boolean to False — so the meta line still renders as
        ``<model> • no-wt • sequential • manual-commit``."""
        task_id, _ = th._add_task("legacy partial")
        th._save_task_extra({"model": "legacy-model"}, task_id=task_id)
        server, events = _make_server()

        server._get_history(query=None)

        s = _history_sessions(events)[0]
        assert s["model"] == "legacy-model"
        assert s["is_worktree"] is False
        assert s["is_parallel"] is False
        assert s["auto_commit_mode"] is False

    def test_non_string_model_falls_back_to_empty(self) -> None:
        """Garbage ``model`` (e.g. None or an int) must NOT crash
        and must surface as the empty string default — the frontend
        then omits the meta line entirely."""
        task_id, _ = th._add_task("garbage model")
        th._save_task_extra(
            {
                "model": None,
                "is_worktree": True,
                "is_parallel": True,
                "auto_commit_mode": True,
            },
            task_id=task_id,
        )
        server, events = _make_server()

        server._get_history(query=None)

        s = _history_sessions(events)[0]
        # ``None`` is not a string → empty model is kept; the
        # frontend will skip rendering the meta line for this row.
        assert s["model"] == ""
        # The booleans round-trip independently.
        assert s["is_worktree"] is True
        assert s["is_parallel"] is True
        assert s["auto_commit_mode"] is True

    def test_truthy_strings_coerced_to_bool(self) -> None:
        """A truthy non-bool value (e.g. ``"true"``) under one of
        the boolean keys must coerce via ``bool()`` — the goal is
        that legacy / hand-edited DB rows can't produce a non-bool
        in the session payload."""
        task_id, _ = th._add_task("truthy bool")
        th._save_task_extra(
            {
                "model": "x",
                "is_worktree": "yes",
                "is_parallel": 1,
                "auto_commit_mode": [0],
            },
            task_id=task_id,
        )
        server, events = _make_server()

        server._get_history(query=None)

        s = _history_sessions(events)[0]
        assert s["is_worktree"] is True
        assert s["is_parallel"] is True
        assert s["auto_commit_mode"] is True

    def test_falsy_values_coerced_to_false(self) -> None:
        """``None`` / ``""`` / ``0`` under boolean keys → False."""
        task_id, _ = th._add_task("falsy bool")
        th._save_task_extra(
            {
                "model": "y",
                "is_worktree": None,
                "is_parallel": "",
                "auto_commit_mode": 0,
            },
            task_id=task_id,
        )
        server, events = _make_server()

        server._get_history(query=None)

        s = _history_sessions(events)[0]
        assert s["is_worktree"] is False
        assert s["is_parallel"] is False
        assert s["auto_commit_mode"] is False
