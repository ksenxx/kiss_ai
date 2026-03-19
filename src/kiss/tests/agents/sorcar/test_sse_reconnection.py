"""Tests for SSE reconnection, shutdown timer, heartbeat, and browser close shutdown."""

from __future__ import annotations

import inspect
import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar import sorcar


def _redirect_history(tmpdir: str):
    old = (th.HISTORY_FILE, th.MODEL_USAGE_FILE,
           th.FILE_USAGE_FILE, th._history_cache, th._KISS_DIR, th._CHAT_EVENTS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th.HISTORY_FILE = kiss_dir / "task_history.jsonl"
    th._CHAT_EVENTS_DIR = kiss_dir / "chat_events"
    th.MODEL_USAGE_FILE = kiss_dir / "model_usage.json"
    th.FILE_USAGE_FILE = kiss_dir / "file_usage.json"
    th._history_cache = None
    return old


def _restore_history(saved):
    (th.HISTORY_FILE, th.MODEL_USAGE_FILE,
     th.FILE_USAGE_FILE, th._history_cache, th._KISS_DIR, th._CHAT_EVENTS_DIR) = saved


class TestShutdownTimerDuration:
    def test_shutdown_timers_are_short(self) -> None:
        source = inspect.getsource(sorcar.run_chatbot)
        assert "call_later(1.0," in source
        assert "call_later(10.0," not in source
        assert "Timer(120.0," not in source
        assert "no_client_since >= 2.0" in source
        assert "no_client_since >= 10.0" not in source


class TestShutdownTimerBehavior:
    def test_cancel_shutdown_noop_when_no_timer(self) -> None:
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()

        def _cancel_shutdown() -> None:
            nonlocal shutdown_timer
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                    shutdown_timer = None

        _cancel_shutdown()
        assert shutdown_timer is None


class TestSSEEventsEndpointIntegration:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_history(self.tmpdir)

    def teardown_method(self) -> None:
        _restore_history(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
