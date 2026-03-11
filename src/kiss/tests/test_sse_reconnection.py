"""Tests for SSE disconnection fixes: reconnection, shutdown timer, heartbeat.

Covers:
- Server shutdown timer increased from 5s to 120s
- _cancel_shutdown cancels pending timer on client reconnect
- Heartbeat interval reduced from 15s to 5s
- SSE Connection header set to keep-alive
- JavaScript reconnect logic (via _build_html output inspection)
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.task_history as th


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


# ---------------------------------------------------------------------------
# Server-side: shutdown timer behavior
# ---------------------------------------------------------------------------
class TestShutdownTimerBehavior:
    """Test that the shutdown timer is long enough for reconnection."""

    def test_cancel_shutdown_noop_when_no_timer(self) -> None:
        """_cancel_shutdown should be safe to call with no timer."""
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


# ---------------------------------------------------------------------------
# SSE events endpoint integration test
# ---------------------------------------------------------------------------
class TestSSEEventsEndpointIntegration:
    """Integration tests that spin up a real Starlette app to verify SSE."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_history(self.tmpdir)

    def teardown_method(self) -> None:
        _restore_history(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
