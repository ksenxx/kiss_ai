# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``shareChat`` server API command.

The chat webview's share button serializes the highlighted tab's
static task panel and event panels and sends a ``shareChat`` command;
the daemon wraps them into a standalone page
(``kiss.server.web_server._build_share_page``) and writes it to
``<workDir>/reports/chat-<chatId>.html``.  These tests drive the REAL
production path — a live :class:`RemoteAccessServer` dispatcher over a
real Unix-domain socket, exactly how the VS Code extension host
forwards the webview's command — and assert on the ``share_done``
reply and on the page written to disk.
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
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer

_PASSWORD = "share-chat-test-password"


class TestShareChatOverUds(unittest.TestCase):
    """``shareChat`` writes the shared page and answers ``share_done``."""

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
                self.sock_path
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
        self._server = RemoteAccessServer(
            host="127.0.0.1",
            port=self._port,
            work_dir=self._work_dir,
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
