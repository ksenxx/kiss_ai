"""Tests for SSE reconnection, shutdown timer, heartbeat, and browser close shutdown."""

from __future__ import annotations

import inspect
from pathlib import Path

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar import sorcar


def _redirect_history(tmpdir: str):
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore_history(saved):
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestShutdownTimerDuration:
    def test_shutdown_timers_are_short(self) -> None:
        source = inspect.getsource(sorcar.run_chatbot)
        assert "call_later(1.0," in source
        assert "call_later(10.0," not in source
        assert "Timer(120.0," not in source


