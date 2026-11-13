# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A loading page's CHATS rows go out before the transcript of the tab it shows.

The desktop page asks for its CHATS page (``getHistory``) right before
its ``ready``.  The page merges every registered machine's rows and a
machine that does not answer holds it for
:data:`~kiss.agents.sorcar.machines.HISTORY_FANOUT_WAIT` seconds,
while the ``ready`` handler had already queued the shown tab's
transcript (megabytes over a slow link) for the same connection, so
the rows waited behind it.  Now the transcript is queued only once
the page's reply is (:meth:`ServerApi.await_history_reply`, bounded by
:data:`~kiss.server.sorcar.HISTORY_FIRST_WAIT`), and a connection
that asked for no page waits for nothing.

A real daemon with a registered machine that never answers (a port
that takes the connection and says nothing), driven over its local
endpoint; no mocks.
"""

from __future__ import annotations

import json
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import persistence
from kiss.server.sorcar import HISTORY_FIRST_WAIT
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import _generate_self_signed_cert, open_local_connection


class TestHistoryBeforeTranscript(IsolatedAsyncioTestCase):
    """One daemon, one silent registered machine, one client loading a page."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-history-first-")
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        # Accepts in the kernel, never answers: the history fan-out
        # waits its bound for this machine's page.
        self.hole = socket.create_server(("127.0.0.1", 0))
        silent = mach.Machine("silent", f"https://127.0.0.1:{self.hole.getsockname()[1]}")
        silent.cert_file().parent.mkdir(parents=True, exist_ok=True)
        silent.cert_file().write_text(certfile.read_text(encoding="utf-8"), encoding="utf-8")
        mach.save_machines([silent])
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        mach.save_machines([])
        self.hole.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _frames_until_transcript(self, tab_id: str) -> list[dict[str, Any]]:
        """Return every frame up to and including *tab_id*'s ``task_events``."""
        frames: list[dict[str, Any]] = []
        while len(frames) < 300:
            line = await self.reader.readline()
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            frames.append(msg)
            if msg.get("type") == "task_events" and msg.get("tabId") == tab_id:
                return frames
        raise AssertionError("no transcript for the shown tab")

    async def test_the_rows_land_before_the_shown_tabs_transcript(self) -> None:
        task_id, chat_id = persistence._add_task("the chat on screen")
        persistence._queue_chat_event(
            {"type": "tool_result", "content": "x" * 10_000, "is_error": False,
             "tool_name": "Bash"},
            task_id=task_id,
        )
        persistence._flush_chat_events()
        persistence._save_task_result("done", task_id=task_id)
        started = time.monotonic()
        await self._send({"type": "getHistory"})
        await self._send({
            "type": "ready", "tabId": "shown", "restoredTabs": [
                {"tabId": "shown", "chatId": chat_id, "title": "the chat on screen", "workDir": ""},
            ],
        })
        frames = await self._frames_until_transcript("shown")
        kinds = [str(f.get("type")) for f in frames if f.get("type") in ("history", "task_events")]
        self.assertEqual(kinds[0], "history", f"the transcript came first: {kinds}")
        history = next(f for f in frames if f.get("type") == "history")
        self.assertIn(chat_id, [s.get("id") for s in history.get("sessions", [])])
        # The reply waited for the silent machine's page the fan-out's
        # bound, not the handler's: the page is what held the transcript.
        waited = time.monotonic() - started
        self.assertGreaterEqual(waited, mach.HISTORY_FANOUT_WAIT)
        self.assertLess(waited, HISTORY_FIRST_WAIT + mach.HISTORY_FANOUT_WAIT)

    async def test_a_page_that_asked_for_no_rows_gets_its_transcript_at_once(self) -> None:
        task_id, chat_id = persistence._add_task("a chat opened from a phone")
        persistence._save_task_result("done", task_id=task_id)
        started = time.monotonic()
        await self._send({
            "type": "ready", "tabId": "shown", "restoredTabs": [
                {"tabId": "shown", "chatId": chat_id, "title": "a chat", "workDir": ""},
            ],
        })
        frames = await self._frames_until_transcript("shown")
        self.assertNotIn("history", [f.get("type") for f in frames])
        self.assertLess(time.monotonic() - started, mach.HISTORY_FANOUT_WAIT)
