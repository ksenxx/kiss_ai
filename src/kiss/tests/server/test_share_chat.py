# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``shareChat`` / ``shareChatTasks`` API.

The chat webview's share button first sends ``shareChatTasks``: the
daemon answers with a ``share_tasks`` reply carrying the persisted
transcripts of ALL of the chat's tasks (oldest first) — the webview's
own DOM holds only the one task the session replay repainted.  The
webview assembles those transcripts (plus the live screen) into one
page body and sends it back via ``shareChat``; the daemon wraps it
into a standalone page
(``kiss.server.web_server._build_share_page``) and writes it to
``<workDir>/reports/chat-<chatId>.html``.  These tests drive the REAL
production path — a live :class:`RemoteAccessServer` dispatcher over a
real Unix-domain socket, exactly how the VS Code extension host
forwards the webview's commands, and over real WSS exactly like the
remote webapp — against a real history database, and assert on the
``share_tasks`` / ``share_done`` replies and on the page written to
disk.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import socket
import ssl
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import (
    _MAX_LINE_BYTES,
    _SHARE_TASKS_MAX_REPLY_BYTES,
    RemoteAccessServer,
)

_PASSWORD = "share-chat-test-password"


def _redirect_db(tmpdir: str) -> tuple[Any, Any, Any]:
    """Point the history database at a temp dir; return prior state.

    ``_get_db()`` invalidates cached per-thread connections whenever
    ``_DB_PATH`` changes, so reassigning the module globals is the
    supported test redirect (the same fixture every persistence test
    in this suite's directory uses).
    """
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_db(saved: tuple[Any, Any, Any]) -> None:
    """Undo :func:`_redirect_db`."""
    if th._db_conn is not None:
        th._db_conn.close()
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class _UdsServerTestCase(unittest.TestCase):
    """Harness: a live UDS dispatcher exactly like the extension's."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.work_dir = os.path.join(self.tmp.name, "workspace")
        os.makedirs(self.work_dir)
        self.sock_path = os.path.join(self.tmp.name, "sorcar-test.sock")
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        self.server._printer._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path
            ),
            self.loop,
        ).result(timeout=5)

    def tearDown(self) -> None:
        async def _shutdown() -> None:
            self.uds_server.close()
            await self.uds_server.wait_closed()

        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)],
            timeout=5,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def _roundtrip(
        self, cmds: list[dict[str, Any]], want_type: str
    ) -> dict[str, Any]:
        """Send *cmds* over one fresh UDS connection; return the first
        received event of type *want_type*."""

        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path,
                # A share_tasks reply can approach the transport's own
                # frame cap; the reader must accept what the daemon is
                # allowed to send.
                limit=_MAX_LINE_BYTES,
            )
            try:
                for cmd in cmds:
                    writer.write(json.dumps(cmd).encode() + b"\n")
                await writer.drain()
                while True:
                    line = await asyncio.wait_for(
                        reader.readline(), timeout=10
                    )
                    if not line:
                        raise AssertionError(
                            f"connection closed before a {want_type!r} event"
                        )
                    event: dict[str, Any] = json.loads(line)
                    if event.get("type") == want_type:
                        return event
            finally:
                writer.close()
                await writer.wait_closed()

        return asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )


class TestShareChatOverUds(_UdsServerTestCase):
    """``shareChat`` writes the shared page and answers ``share_done``."""

    def _share(self, **fields: Any) -> dict[str, Any]:
        cmd: dict[str, Any] = {"type": "shareChat", "workDir": self.work_dir}
        cmd.update(fields)
        return self._roundtrip([cmd], "share_done")

    BODY = (
        '<div id="task-panel" class="visible">'
        '<div id="task-panel-text">list files</div></div>'
        '<div id="output"><div class="ev tc collapsible">'
        '<div class="tc-h collapse-header">Bash</div>'
        "<pre>ls -la</pre></div></div>"
    )

    def test_share_writes_standalone_page_and_replies_ok(self) -> None:
        event = self._share(
            chatId="chat-42",
            html=self.BODY,
            title="My chat",
            tabId="tab-1",
        )
        self.assertTrue(event["ok"], event)
        self.assertEqual(event["tabId"], "tab-1")
        out = Path(self.work_dir) / "reports" / "chat-chat-42.html"
        self.assertEqual(event["path"], str(out))
        page = out.read_text(encoding="utf-8")
        # The transcript body travels verbatim.
        self.assertIn(self.BODY, page)
        self.assertIn("<title>My chat</title>", page)
        # Self-contained: the webview stylesheet, the highlight.js
        # theme, the VS Code palette and the collapse script are all
        # inlined, so the page needs no server and no other file.
        self.assertIn("#task-panel {", page)  # main.css
        self.assertIn(".collapse-preview", page)  # main.css
        self.assertIn(".hljs", page)  # highlight theme
        self.assertIn("--vscode-editor-background: #1e1e1e", page)
        self.assertIn("window.toggleThink", page)  # share.js
        # share.js's click delegation — a marker that appears in the
        # script only, never in this test's transcript body.
        self.assertIn(".closest('.collapse-header')", page)
        self.assertNotIn("{{", page.split("</title>")[0])

    def test_chat_id_is_sanitized_into_the_filename(self) -> None:
        event = self._share(chatId="a/b c!*", html=self.BODY)
        self.assertTrue(event["ok"], event)
        out = Path(self.work_dir) / "reports" / "chat-a-b-c.html"
        self.assertEqual(event["path"], str(out))
        self.assertTrue(out.is_file())

    def test_missing_html_field_is_rejected_by_the_catalog(self) -> None:
        event = self._roundtrip(
            [{"type": "shareChat", "chatId": "c", "tabId": "tab-9"}],
            "error",
        )
        self.assertEqual(
            event["text"], "Invalid shareChat command: missing html"
        )
        self.assertEqual(event["tabId"], "tab-9")

    def test_blank_html_reports_an_empty_chat(self) -> None:
        event = self._share(chatId="c1", html="   ")
        self.assertFalse(event["ok"])
        self.assertEqual(event["error"], "Nothing to share: the chat is empty")
        self.assertFalse((Path(self.work_dir) / "reports").exists())

    def test_non_string_chat_id_reports_missing_chat_id(self) -> None:
        event = self._share(chatId=7, html=self.BODY)
        self.assertFalse(event["ok"])
        self.assertEqual(event["error"], "Missing chat id")

    def test_all_punctuation_chat_id_falls_back_to_chat(self) -> None:
        event = self._share(chatId="///", html=self.BODY)
        self.assertTrue(event["ok"], event)
        self.assertEqual(
            event["path"],
            str(Path(self.work_dir) / "reports" / "chat-chat.html"),
        )

    def test_title_is_escaped_and_non_string_title_is_dropped(self) -> None:
        event = self._share(
            chatId="esc", html=self.BODY, title="<script>alert(1)</script>"
        )
        self.assertTrue(event["ok"], event)
        page = Path(event["path"]).read_text(encoding="utf-8")
        self.assertIn(
            "<title>&lt;script&gt;alert(1)&lt;/script&gt;</title>", page
        )
        event = self._share(chatId="esc2", html=self.BODY, title=17)
        self.assertTrue(event["ok"], event)
        page = Path(event["path"]).read_text(encoding="utf-8")
        self.assertIn("<title>KISS Sorcar chat</title>", page)

    def test_non_string_tab_id_is_normalized(self) -> None:
        event = self._share(chatId="tabless", html=self.BODY, tabId=99)
        self.assertTrue(event["ok"], event)
        self.assertEqual(event["tabId"], "")

    def test_unwritable_reports_dir_reports_the_os_error(self) -> None:
        # A FILE named "reports" makes mkdir(parents=True) raise.
        (Path(self.work_dir) / "reports").write_text("in the way")
        event = self._share(chatId="c2", html=self.BODY)
        self.assertFalse(event["ok"])
        self.assertIn("Failed to write the chat page", event["error"])

    def test_work_dir_comes_from_the_connection_when_not_sent(self) -> None:
        other = os.path.join(self.tmp.name, "other-workspace")
        os.makedirs(other)
        event = self._roundtrip(
            [
                {"type": "setWorkDir", "workDir": other},
                {"type": "shareChat", "chatId": "pinned", "html": self.BODY},
            ],
            "share_done",
        )
        self.assertTrue(event["ok"], event)
        self.assertEqual(
            event["path"],
            str(Path(other) / "reports" / "chat-pinned.html"),
        )


class TestShareChatTasksOverUds(_UdsServerTestCase):
    """``shareChatTasks`` lists every task of a chat, oldest first."""

    def setUp(self) -> None:
        super().setUp()
        self.saved_db = _redirect_db(self.tmp.name)

    def tearDown(self) -> None:
        _restore_db(self.saved_db)
        super().tearDown()

    def _tasks(self, **fields: Any) -> dict[str, Any]:
        cmd: dict[str, Any] = {"type": "shareChatTasks"}
        cmd.update(fields)
        return self._roundtrip([cmd], "share_tasks")

    def _seed(self, chat_id: str, task: str, content: str) -> str:
        """Persist one task with a single tool_call event; return id."""
        task_id, _ = th._add_task(task, chat_id=chat_id)
        th._append_chat_event(
            {"type": "tool_call", "name": "Bash", "command": content},
            task_id=task_id,
        )
        # timestamp is the ordering key; keep the rows distinct.
        time.sleep(0.01)
        return task_id

    def test_every_task_of_the_chat_is_returned_oldest_first(self) -> None:
        first = self._seed("chat-a", "first task", "ls -la")
        second = self._seed("chat-a", "second task", "pwd")
        self._seed("chat-OTHER", "foreign task", "whoami")
        event = self._tasks(chatId="chat-a", tabId="tab-7")
        self.assertEqual(event["tabId"], "tab-7")
        self.assertEqual(event["chatId"], "chat-a")
        self.assertFalse(event["truncated"])
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]], [first, second]
        )
        self.assertEqual(
            [t["task"] for t in event["tasks"]],
            ["first task", "second task"],
        )
        # Each task's events start with the ensured task_settings event
        # (synthesized from the row when the stream carries none).
        self.assertEqual(
            [t["events"][0]["type"] for t in event["tasks"]],
            ["task_settings", "task_settings"],
        )
        self.assertEqual(
            event["tasks"][0]["events"][1]["command"], "ls -la"
        )
        self.assertEqual(event["tasks"][1]["events"][1]["command"], "pwd")

    def test_subagent_rows_are_not_chat_tasks(self) -> None:
        parent = self._seed("chat-s", "parent task", "ls")
        sub = self._seed("chat-s", "sub-agent task", "internal")
        th._save_task_extra(
            {"subagent": {"parent_task_id": parent}}, task_id=sub
        )
        event = self._tasks(chatId="chat-s")
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]],
            [parent],
            "a sub-agent transcript replays inside its parent's panels,"
            " never as a chat task of its own",
        )

    def test_unknown_chat_id_yields_no_tasks(self) -> None:
        event = self._tasks(chatId="never-seen", tabId="tab-1")
        self.assertEqual(event["tasks"], [])
        self.assertFalse(event["truncated"])

    def test_blank_chat_id_yields_no_tasks(self) -> None:
        self._seed("chat-b", "some task", "ls")
        event = self._tasks(chatId="   ")
        self.assertEqual(event["tasks"], [])

    def test_non_string_chat_id_yields_no_tasks(self) -> None:
        event = self._tasks(chatId=7, tabId=42)
        self.assertEqual(event["tasks"], [])
        self.assertEqual(event["chatId"], "")
        self.assertEqual(event["tabId"], "")

    def test_missing_chat_id_is_rejected_by_the_catalog(self) -> None:
        event = self._roundtrip(
            [{"type": "shareChatTasks", "tabId": "tab-9"}], "error"
        )
        self.assertEqual(
            event["text"], "Invalid shareChatTasks command: missing chatId"
        )
        self.assertEqual(event["tabId"], "tab-9")

    def test_overlong_echo_ids_are_capped(self) -> None:
        # The echoed identifiers ride in every reply; a client must not
        # be able to push the reply past the frame cap by inflating
        # them.
        event = self._tasks(chatId="x" * 10_000, tabId="y" * 10_000)
        self.assertEqual(len(event["chatId"]), 256)
        self.assertEqual(len(event["tabId"]), 256)
        self.assertEqual(event["tasks"], [])

    def test_database_failure_reports_an_error(self) -> None:
        # A DIRECTORY at the database path makes sqlite3 unable to open
        # it — a real filesystem failure mode, not a simulated one.
        th._close_db()
        th._DB_PATH = Path(self.tmp.name) / "db-in-the-way"
        th._DB_PATH.mkdir()
        event = self._tasks(chatId="chat-x", tabId="tab-x")
        self.assertIn("Failed to load the chat history", event["error"])
        self.assertEqual(event["tasks"], [])
        self.assertEqual(event["tabId"], "tab-x")

    def test_oldest_tasks_are_dropped_when_the_reply_would_overflow(
        self,
    ) -> None:
        # Two tasks whose payloads together exceed the reply budget but
        # fit it one at a time: the OLDER one must be dropped — the
        # newest transcripts are the ones the webview cannot redraw
        # from its own DOM — and the reply must both fit the transport
        # frame and say it was truncated.
        big = "x" * (_SHARE_TASKS_MAX_REPLY_BYTES // 2 + 1024)
        self._seed("chat-t", "old task", big)
        newest = self._seed("chat-t", "new task", big)
        event = self._tasks(chatId="chat-t")
        self.assertTrue(event["truncated"])
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]], [newest]
        )


class TestShareChatOverWss(IsolatedAsyncioTestCase):
    """The remote webapp's share button path, over real WSS.

    Drives the exact frame sequence the browser produces — the
    ``auth`` handshake, then a ``shareChat`` catalog command on the
    authenticated socket — against a live :class:`RemoteAccessServer`,
    and asserts the page lands on disk and the ``share_done`` reply
    reaches the sending client.
    """

    async def asyncSetUp(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            self._port = int(sock.getsockname()[1])
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": _PASSWORD})
        self._work_dir = tempfile.mkdtemp()
        self._saved_db = _redirect_db(self._work_dir)
        self._server = RemoteAccessServer(
            host="127.0.0.1",
            port=self._port,
            work_dir=self._work_dir,
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        _restore_db(self._saved_db)
        await self._server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def test_share_over_wss_writes_page_and_replies(self) -> None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        async with await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=ctx,
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": _PASSWORD}))
            while True:
                msg = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=10)
                )
                if msg.get("type") == "auth_ok":
                    break
            await ws.send(json.dumps({
                "type": "shareChat",
                "chatId": "wss-chat",
                "title": "Remote chat",
                "html": TestShareChatOverUds.BODY,
                "tabId": "tab-wss",
            }))
            while True:
                event = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=10)
                )
                if event.get("type") == "share_done":
                    break
        self.assertTrue(event["ok"], event)
        self.assertEqual(event["tabId"], "tab-wss")
        out = Path(self._work_dir) / "reports" / "chat-wss-chat.html"
        self.assertEqual(event["path"], str(out))
        page = out.read_text(encoding="utf-8")
        self.assertIn(TestShareChatOverUds.BODY, page)
        self.assertIn("<title>Remote chat</title>", page)

    async def test_share_chat_tasks_over_wss_lists_every_task(self) -> None:
        first, _ = th._add_task("first task", chat_id="wss-chat-2")
        th._append_chat_event(
            {"type": "tool_call", "name": "Bash", "command": "ls -la"},
            task_id=first,
        )
        time.sleep(0.01)
        second, _ = th._add_task("second task", chat_id="wss-chat-2")
        th._append_chat_event(
            {"type": "tool_call", "name": "Bash", "command": "pwd"},
            task_id=second,
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        async with await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=ctx,
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": _PASSWORD}))
            while True:
                msg = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=10)
                )
                if msg.get("type") == "auth_ok":
                    break
            await ws.send(json.dumps({
                "type": "shareChatTasks",
                "chatId": "wss-chat-2",
                "tabId": "tab-wss-2",
            }))
            while True:
                event = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=10)
                )
                if event.get("type") == "share_tasks":
                    break
        self.assertEqual(event["tabId"], "tab-wss-2")
        self.assertEqual(event["chatId"], "wss-chat-2")
        self.assertFalse(event["truncated"])
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]], [first, second]
        )
        self.assertEqual(
            event["tasks"][1]["events"][0]["type"], "task_settings"
        )
        self.assertEqual(
            event["tasks"][1]["events"][1]["command"], "pwd"
        )


if __name__ == "__main__":
    unittest.main()
