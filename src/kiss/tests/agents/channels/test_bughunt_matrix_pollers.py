# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for round-2 bugs in matrix_agent.py and the Slack pollers.

Covered bugs (all reproduced end-to-end, no mock/patch libraries):

1. Slack pollers' ``_acquire_lock`` opened the lock file with mode ``"w"``,
   truncating the running holder's recorded PID before ``flock`` was even
   attempted.  Tested with a REAL second python subprocess contending for
   the same lock file.
2. Matrix backend drove one shared nio ``AsyncClient`` through a fresh
   ``asyncio.run()`` per method, so every call after the first failed with
   "Event loop is closed".  The fix routes all coroutines through one
   persistent background event loop; these tests exercise that loop's full
   lifecycle (two successive calls, disconnect, lazy restart) without nio.
3. The Slack DM poller lacked the startup ``min_ts`` watermark its sibling
   channel poller has, so it retroactively processed historical DMs.
   Tested against a real in-process HTTP server emulating the Slack API
   with a real ``slack_sdk.WebClient``.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from slack_sdk import WebClient

from kiss.agents.third_party_agents.matrix_agent import MatrixChannelBackend

_POLLER_MODULES = (
    "kiss.agents.third_party_agents.slack_sorcar_poller",
    "kiss.agents.third_party_agents.slack_channel_sorcar_poller",
)


def _contend_lock_from_subprocess(module_name: str, lock_path: Path) -> str:
    """Attempt to acquire ``lock_path`` via ``module_name`` in a real subprocess.

    Returns:
        The subprocess's combined stdout output.
    """
    code = (
        "import pathlib\n"
        f"import {module_name} as m\n"
        f"m.LOCK_FILE = pathlib.Path({str(lock_path)!r})\n"
        "m._acquire_lock()\n"
        "print('ACQUIRED')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


class TestPollerLockFile:
    """The lock file must not be truncated by a losing contender (bug 6)."""

    def _check_lock_preserves_holder_pid(self, module_name: str) -> None:
        import importlib

        mod = importlib.import_module(module_name)
        original_lock = mod.LOCK_FILE
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "poller.lock"
            setattr(mod, "LOCK_FILE", lock_path)  # noqa: B010
            fp = None
            try:
                fp = mod._acquire_lock()
                my_pid = f"{os.getpid()}\n"
                assert lock_path.read_text() == my_pid
                out = _contend_lock_from_subprocess(module_name, lock_path)
                assert "ACQUIRED" not in out, "second process must not get the lock"
                assert lock_path.read_text() == my_pid, (
                    "losing contender truncated/overwrote the holder's PID"
                )
            finally:
                setattr(mod, "LOCK_FILE", original_lock)  # noqa: B010
                if fp is not None:
                    fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
                    fp.close()

    def test_dm_poller_lock_preserves_holder_pid(self) -> None:
        """Losing DM-poller contender exits without clobbering the PID."""
        self._check_lock_preserves_holder_pid(_POLLER_MODULES[0])

    def test_channel_poller_lock_preserves_holder_pid(self) -> None:
        """Losing channel-poller contender exits without clobbering the PID."""
        self._check_lock_preserves_holder_pid(_POLLER_MODULES[1])


class TestMatrixPersistentLoop:
    """Matrix backend must reuse one event loop across calls (bug 1)."""

    def test_two_successive_calls_share_one_loop(self) -> None:
        """Two successive coroutine runs succeed and reuse the same loop."""
        backend = MatrixChannelBackend()
        try:
            assert backend._run(asyncio.sleep(0, result="first")) == "first"
            loop = backend._loop
            assert loop is not None
            # Pre-fix, the asyncio.run() pattern gave every call a fresh
            # loop and broke nio's cached aiohttp session on call two.
            assert backend._run(asyncio.sleep(0, result="second")) == "second"
            assert backend._loop is loop
            assert loop.is_running()
        finally:
            backend.disconnect()

    def test_disconnect_stops_loop_and_allows_restart(self) -> None:
        """disconnect() stops the loop; a later call lazily restarts it."""
        backend = MatrixChannelBackend()
        assert backend._run(asyncio.sleep(0, result=1)) == 1
        loop = backend._loop
        thread = backend._loop_thread
        assert loop is not None and thread is not None
        backend.disconnect()
        assert backend._loop is None
        assert backend._loop_thread is None
        assert not thread.is_alive()
        assert not loop.is_running()
        # Lazy restart after disconnect.
        try:
            assert backend._run(asyncio.sleep(0, result=7)) == 7
        finally:
            backend.disconnect()

    def test_coroutine_exceptions_propagate(self) -> None:
        """Exceptions raised inside coroutines surface to the caller."""
        backend = MatrixChannelBackend()

        async def _boom() -> None:
            raise ValueError("boom")

        try:
            try:
                backend._run(_boom())
            except ValueError as e:
                assert str(e) == "boom"
            else:  # pragma: no cover - failure path
                raise AssertionError("expected ValueError")
            # The loop must survive a failed coroutine.
            assert backend._run(asyncio.sleep(0, result="ok")) == "ok"
        finally:
            backend.disconnect()


_OLD_TS = "1000.000100"
_NEW_TS = "3000.100000"
_BOT_REPLY_OLD = {"ts": "1001.000000", "user": "UBOT", "text": "done"}
_BOT_REPLY_NEW = {"ts": "3001.000000", "user": "UBOT", "text": "done"}


class _SlackHandler(BaseHTTPRequestHandler):
    """Minimal Slack Web API emulator that records every request."""

    def do_POST(self) -> None:  # noqa: N802
        self._record_and_respond()

    def do_GET(self) -> None:  # noqa: N802
        self._record_and_respond()

    def _record_and_respond(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        parsed = urllib.parse.urlparse(self.path)
        server: Any = self.server
        server.requests.append({"path": parsed.path, "query": parsed.query, "body": body})
        if parsed.path.endswith("/conversations.history"):
            payload: dict[str, Any] = {
                "ok": True,
                "messages": [
                    {
                        "ts": _OLD_TS,
                        "user": "UKSEN",
                        "text": "old historical task",
                        "reply_count": 1,
                        "thread_ts": _OLD_TS,
                    },
                    {
                        "ts": _NEW_TS,
                        "user": "UKSEN",
                        "text": "new task",
                        "reply_count": 1,
                        "thread_ts": _NEW_TS,
                    },
                ],
            }
        elif parsed.path.endswith("/conversations.replies"):
            form = urllib.parse.parse_qs(body)
            ts = form.get("ts", [""])[0]
            if ts == _OLD_TS:
                parent = {
                    "ts": _OLD_TS,
                    "user": "UKSEN",
                    "text": "old historical task",
                    "thread_ts": _OLD_TS,
                }
                payload = {"ok": True, "messages": [parent, _BOT_REPLY_OLD]}
            else:
                parent = {
                    "ts": _NEW_TS,
                    "user": "UKSEN",
                    "text": "new task",
                    "thread_ts": _NEW_TS,
                }
                payload = {"ok": True, "messages": [parent, _BOT_REPLY_NEW]}
        else:
            payload = {"ok": True}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence request logging."""


class TestDmPollerWatermark:
    """The DM poller must skip messages older than the min_ts watermark."""

    def test_poll_once_skips_messages_older_than_min_ts(self) -> None:
        """Old DMs below the watermark are never fetched/processed."""
        import importlib

        poller = importlib.import_module(_POLLER_MODULES[0])
        server = ThreadingHTTPServer(("127.0.0.1", 0), _SlackHandler)
        server.requests = []  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            client = WebClient(token="xoxb-test", base_url=f"http://127.0.0.1:{port}/")
            state: dict[str, Any] = {"threads": {}, "min_ts": "2000.0"}
            poller._poll_once(client, "D1", "UBOT", "UKSEN", state)
            requests: list[dict[str, str]] = server.requests  # type: ignore[attr-defined]
            replies_ts = [
                urllib.parse.parse_qs(r["body"]).get("ts", [""])[0]
                for r in requests
                if r["path"].endswith("/conversations.replies")
            ]
            # The message newer than the watermark IS processed...
            assert _NEW_TS in replies_ts
            # ...but the historical message below the watermark is skipped.
            assert _OLD_TS not in replies_ts, (
                "DM poller retroactively processed a message older than min_ts"
            )
            # No Sorcar task ran (bot already replied in the new thread).
            posts = [r for r in requests if r["path"].endswith("/chat.postMessage")]
            assert posts == []
            assert state["threads"] == {}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)
