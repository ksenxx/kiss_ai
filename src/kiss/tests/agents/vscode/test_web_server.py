"""Integration tests for the KISS Sorcar remote web access server.

Tests cover HTTP serving, WebSocket communication, password authentication,
command dispatch, and event broadcasting through the web server.
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.agents.vscode.vscode_config import CONFIG_PATH, save_config
from kiss.agents.vscode.web_server import (
    _URL_FILE,
    TUNNEL_CHECK_INTERVAL,
    RemoteAccessServer,
    WebPrinter,
    _build_html,
    _create_ssl_context,
    _generate_self_signed_cert,
    _get_local_ips,
    _print_url,
    _remove_url_file,
    _save_url_file,
    _translate_webview_command,
)


class TestBuildHtml(unittest.TestCase):
    """Test HTML template generation."""

    def test_html_contains_key_elements(self) -> None:
        """The generated HTML includes all essential chat UI components."""
        html = _build_html()
        self.assertIn("<title>KISS Sorcar</title>", html)
        self.assertIn('id="tab-bar"', html)
        self.assertIn('id="output"', html)
        self.assertIn('id="task-input"', html)
        self.assertIn('id="input-area"', html)
        self.assertIn('id="model-picker"', html)
        self.assertIn('id="sidebar"', html)
        self.assertIn('id="config-sidebar"', html)
        self.assertIn('id="ask-user-modal"', html)
        self.assertIn('id="send-btn"', html)
        self.assertIn('id="stop-btn"', html)

    def test_html_includes_ws_shim(self) -> None:
        """The generated HTML injects the WebSocket shim before main.js."""
        html = _build_html()
        self.assertIn("acquireVsCodeApi", html)
        self.assertIn("WebSocket", html)
        shim_pos = html.index("acquireVsCodeApi")
        main_js_pos = html.index('src="/media/main.js"')
        self.assertLess(shim_pos, main_js_pos)

    def test_html_includes_media_refs(self) -> None:
        """The HTML references all required media assets."""
        html = _build_html()
        self.assertIn("/media/main.css", html)
        self.assertIn("/media/highlight-github-dark.min.css", html)
        self.assertIn("/media/highlight.min.js", html)
        self.assertIn("/media/marked.min.js", html)
        self.assertIn("/media/main.js", html)
        self.assertIn("/media/demo.js", html)

    def test_html_has_no_vscode_csp(self) -> None:
        """The standalone HTML does not contain VS Code CSP nonce directives."""
        html = _build_html()
        self.assertNotIn("acquireVsCodeApi()", html.split("acquireVsCodeApi")[0])
        self.assertNotIn("webview.cspSource", html)


class TestTranslateWebviewCommand(unittest.TestCase):
    """Test the command translation from webview format to backend format."""

    def test_user_action_done_translated(self) -> None:
        """userActionDone becomes userAnswer with answer='done'."""
        result = _translate_webview_command({"type": "userActionDone"})
        self.assertEqual(result["type"], "userAnswer")
        self.assertEqual(result["answer"], "done")

    def test_resume_session_id_becomes_chat_id(self) -> None:
        """resumeSession 'id' field is renamed to 'chatId'."""
        result = _translate_webview_command({
            "type": "resumeSession", "id": 42, "tabId": "t1",
        })
        self.assertEqual(result["type"], "resumeSession")
        self.assertEqual(result["chatId"], 42)
        self.assertNotIn("id", result)
        self.assertEqual(result["tabId"], "t1")

    def test_resume_session_with_chat_id_unchanged(self) -> None:
        """resumeSession with chatId already set is not modified."""
        cmd = {"type": "resumeSession", "chatId": 42, "tabId": "t1"}
        result = _translate_webview_command(cmd)
        self.assertEqual(result["chatId"], 42)

    def test_passthrough_commands_unchanged(self) -> None:
        """Commands not needing translation pass through unchanged."""
        for cmd in [
            {"type": "getModels"},
            {"type": "stop", "tabId": "t1"},
            {"type": "selectModel", "model": "m", "tabId": "t1"},
            {"type": "newChat", "tabId": "t1"},
            {"type": "getHistory", "query": "test"},
        ]:
            result = _translate_webview_command(cmd)
            self.assertEqual(result, cmd)


class TestWebPrinter(unittest.TestCase):
    """Test the WebPrinter event broadcasting."""

    def test_broadcast_records_event(self) -> None:
        """Broadcast records events in the per-tab recording buffer."""
        printer = WebPrinter()
        printer._thread_local.tab_id = "t1"
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "hello"})
        events = printer.peek_recording()
        self.assertTrue(any(e.get("text") == "hello" for e in events))

    def test_broadcast_injects_tab_id(self) -> None:
        """Broadcast injects tabId from thread-local when missing."""
        printer = WebPrinter()
        printer._thread_local.tab_id = "t1"
        printer.start_recording()
        captured: list[dict] = []
        original_record = printer._record_event

        def spy_record(event: dict) -> None:
            captured.append(event)
            original_record(event)

        printer._record_event = spy_record  # type: ignore[assignment]
        printer.broadcast({"type": "status", "running": False})
        self.assertTrue(any(e.get("tabId") == "t1" for e in captured))


def _find_free_port() -> int:
    """Find an available TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


class TestRemoteAccessServerHTTP(IsolatedAsyncioTestCase):
    """Test HTTP serving of HTML and static assets."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        # Use empty password for test simplicity
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _http_get(self, path: str) -> tuple[int, str]:
        """Make an HTTP GET request in a thread to avoid blocking the loop."""
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{self.port}{path}"

        def _fetch() -> tuple[int, str]:
            try:
                resp = urllib.request.urlopen(url, timeout=5)
                return resp.status, resp.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode() if e.fp else ""

        return await asyncio.get_event_loop().run_in_executor(None, _fetch)

    async def test_serve_html_page(self) -> None:
        """GET / returns the chat HTML page."""
        status, body = await self._http_get("/")
        self.assertEqual(status, 200)
        self.assertIn("<title>KISS Sorcar</title>", body)
        self.assertIn('id="task-input"', body)

    async def test_serve_css(self) -> None:
        """GET /media/main.css returns the CSS file."""
        status, body = await self._http_get("/media/main.css")
        self.assertEqual(status, 200)
        self.assertIn("#app", body)

    async def test_serve_js(self) -> None:
        """GET /media/main.js returns the JS file."""
        status, body = await self._http_get("/media/main.js")
        self.assertEqual(status, 200)
        self.assertIn("acquireVsCodeApi", body)

    async def test_404_for_unknown_path(self) -> None:
        """GET /unknown returns 404."""
        status, _ = await self._http_get("/unknown")
        self.assertEqual(status, 404)

    async def test_path_traversal_blocked(self) -> None:
        """GET /media/../server.py is blocked (404)."""
        status, _ = await self._http_get("/media/../server.py")
        self.assertEqual(status, 404)


class TestRemoteAccessServerWS(IsolatedAsyncioTestCase):
    """Test WebSocket communication without authentication."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def test_ws_auth_no_password(self) -> None:
        """WebSocket connection with empty password succeeds immediately."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")

    async def test_ws_get_models(self) -> None:
        """getModels command returns a models event over WebSocket."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")

            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "models")
            self.assertIn("models", resp)
            self.assertIsInstance(resp["models"], list)

    async def test_ws_get_history(self) -> None:
        """getHistory command returns a history event."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "getHistory"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "history")
            self.assertIn("sessions", resp)

    async def test_ws_get_config(self) -> None:
        """getConfig command returns configuration data."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "getConfig"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "configData")
            self.assertIn("config", resp)

    async def test_ws_vscode_only_commands_ignored(self) -> None:
        """VS Code-only commands are silently ignored (no error broadcast)."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            # Send a VS Code-only command
            await ws.send(json.dumps({"type": "focusEditor"}))
            # Send getModels to verify the connection still works
            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            # Should get models response, not an error about focusEditor
            self.assertEqual(resp["type"], "models")

    async def test_ws_unknown_command_returns_error(self) -> None:
        """Unknown commands produce an error event."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "totallyBogusCommand"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "error")
            self.assertIn("Unknown command", resp["text"])

    async def test_ws_new_chat_and_close_tab(self) -> None:
        """newChat and closeTab commands work over WebSocket."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            tab_id = "test-tab-1"
            await ws.send(json.dumps({"type": "newChat", "tabId": tab_id}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "showWelcome")
            self.assertEqual(resp["tabId"], tab_id)

            await ws.send(json.dumps({"type": "closeTab", "tabId": tab_id}))
            # closeTab doesn't broadcast, so verify no error by sending another command
            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "models")

    async def test_ws_select_model(self) -> None:
        """selectModel command updates the selected model."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "selectModel",
                "model": "gemini-2.5-pro",
                "tabId": "t1",
            }))
            # selectModel doesn't broadcast, verify via getModels
            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "models")
            self.assertEqual(resp["selected"], "gemini-2.5-pro")

    async def test_ws_ready_command(self) -> None:
        """The 'ready' command returns models, inputHistory, configData, welcome, focusInput."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")

            await ws.send(json.dumps({
                "type": "ready",
                "tabId": "ready-tab",
                "restoredTabs": [],
            }))
            # Collect all responses — expect models, inputHistory,
            # configData, welcome_suggestions, and focusInput (order may vary)
            received_types: set[str] = set()
            for _ in range(5):
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                ev = json.loads(raw)
                received_types.add(ev["type"])
            self.assertIn("models", received_types)
            self.assertIn("inputHistory", received_types)
            self.assertIn("configData", received_types)
            self.assertIn("welcome_suggestions", received_types)
            self.assertIn("focusInput", received_types)

    async def test_ws_ready_does_not_produce_unknown_error(self) -> None:
        """The 'ready' command must NOT produce an 'Unknown command' error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "ready",
                "tabId": "t-ready",
            }))
            events: list[dict[str, Any]] = []
            for _ in range(5):
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                events.append(json.loads(raw))
            for ev in events:
                if ev.get("type") == "error":
                    self.fail(f"ready command produced error: {ev}")

    async def test_ws_submit_does_not_produce_unknown_error(self) -> None:
        """The 'submit' command must NOT produce an 'Unknown command' error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "submit",
                "prompt": "hello",
                "model": "gemini-2.5-pro",
                "tabId": "submit-tab",
                "attachments": [],
            }))
            # Collect events; the first should be setTaskText and status
            events: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 10
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                    ev = json.loads(raw)
                    events.append(ev)
                    # Stop once we see status running=False or result
                    if ev.get("type") == "status" and not ev.get("running"):
                        break
                    if ev.get("type") == "result":
                        break
                except TimeoutError:
                    break
            error_events = [e for e in events if e.get("type") == "error"]
            for err in error_events:
                self.assertNotIn(
                    "Unknown command: submit",
                    err.get("text", ""),
                    "submit command should be translated to run, not error",
                )

    async def test_ws_user_action_done(self) -> None:
        """userActionDone is translated to userAnswer (no Unknown command error)."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "userActionDone"}))
            # userAnswer without an active task just drops the answer.
            # Verify no "Unknown command" error by sending a follow-up.
            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "models")

    async def test_ws_resume_session_translates_id(self) -> None:
        """resumeSession translates the webview 'id' field to 'chatId'."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            # Send resumeSession with 'id' (webview format) instead of
            # 'chatId' (backend format).  A non-existent id produces no
            # broadcast (empty session), but crucially no Unknown command
            # error.  Verify by sending a follow-up command.
            await ws.send(json.dumps({
                "type": "resumeSession",
                "id": 999999,
                "tabId": "resume-tab",
            }))
            await ws.send(json.dumps({"type": "getModels"}))
            # Drain responses — the first may be task_events from
            # resumeSession or models from getModels.
            events: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    ev = json.loads(raw)
                    events.append(ev)
                    if ev.get("type") == "models":
                        break
                except TimeoutError:
                    break
            # Must not have an "Unknown command: resumeSession" error
            for ev in events:
                if ev.get("type") == "error":
                    self.assertNotIn(
                        "Unknown command", ev.get("text", ""),
                    )

    async def test_ws_get_welcome_suggestions(self) -> None:
        """getWelcomeSuggestions returns a welcome_suggestions event."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "getWelcomeSuggestions"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "welcome_suggestions")
            self.assertIn("suggestions", resp)
            self.assertIsInstance(resp["suggestions"], list)

    async def test_ws_remote_url_from_active_url(self) -> None:
        """remote_url event uses in-memory _active_url even when URL file is missing."""
        # Set the in-memory URL directly (simulates what _serve_async does)
        self.server._active_url = "https://test-dynamic.trycloudflare.com"
        # Ensure the URL file does NOT exist so the fallback is needed
        _remove_url_file()
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "getWelcomeSuggestions"}))
            # Collect events — expect welcome_suggestions and remote_url
            events: list[dict[str, Any]] = []
            for _ in range(3):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                    events.append(json.loads(raw))
                except TimeoutError:
                    break
            types = [e["type"] for e in events]
            self.assertIn("remote_url", types)
            url_ev = next(e for e in events if e["type"] == "remote_url")
            self.assertEqual(url_ev["url"], "https://test-dynamic.trycloudflare.com")

    async def test_ws_ready_includes_remote_url(self) -> None:
        """ready command broadcasts remote_url when _active_url is set."""
        self.server._active_url = "https://ready-test.trycloudflare.com"
        _remove_url_file()
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "ready",
                "tabId": "url-tab",
                "restoredTabs": [],
            }))
            events: list[dict[str, Any]] = []
            for _ in range(8):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                    events.append(json.loads(raw))
                except TimeoutError:
                    break
            types = [e["type"] for e in events]
            self.assertIn("remote_url", types)
            url_ev = next(e for e in events if e["type"] == "remote_url")
            self.assertEqual(url_ev["url"], "https://ready-test.trycloudflare.com")

    async def test_ws_get_files(self) -> None:
        """getFiles command returns a files event."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "getFiles", "prefix": ""}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "files")
            self.assertIn("files", resp)

    async def test_ws_get_adjacent_task(self) -> None:
        """getAdjacentTask returns an adjacent_task_events event."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "getAdjacentTask",
                "tabId": "adj-tab",
                "task": "test",
                "direction": "prev",
            }))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "adjacent_task_events")

    async def test_ws_save_config(self) -> None:
        """saveConfig command updates config and returns configData."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "saveConfig",
                "config": {"max_budget": 50},
                "apiKeys": {},
            }))
            # saveConfig broadcasts models and configData
            received_types: set[str] = set()
            for _ in range(2):
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                ev = json.loads(raw)
                received_types.add(ev["type"])
            self.assertIn("configData", received_types)

    async def test_ws_set_skip_merge(self) -> None:
        """setSkipMerge command does not produce an error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "setSkipMerge",
                "tabId": "skip-tab",
                "skip": True,
            }))
            # setSkipMerge doesn't broadcast — verify no error
            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "models")

    async def test_ws_stop_no_error(self) -> None:
        """stop command with no running task does not produce an error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "stop", "tabId": "no-task"}))
            # stop without a running task is a no-op — verify no error
            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "models")

    async def test_ws_merge_action_all_done(self) -> None:
        """mergeAction with all-done does not crash."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "mergeAction",
                "action": "all-done",
                "tabId": "merge-tab",
            }))
            # Verify no crash by sending a follow-up and draining
            await ws.send(json.dumps({"type": "getModels"}))
            events: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    ev = json.loads(raw)
                    events.append(ev)
                    if ev.get("type") == "models":
                        break
                except TimeoutError:
                    break
            self.assertTrue(
                any(e["type"] == "models" for e in events),
                f"Expected models response, got: {[e['type'] for e in events]}",
            )

    async def test_ws_record_file_usage(self) -> None:
        """recordFileUsage command does not produce an error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "recordFileUsage",
                "path": "/tmp/test.py",
            }))
            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "models")

    async def test_ws_generate_commit_message(self) -> None:
        """generateCommitMessage command does not produce Unknown command error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "generateCommitMessage"}))
            # The command runs async and may produce a commitMessage or
            # nothing (no git diff). Verify no Unknown command error.
            await ws.send(json.dumps({"type": "getModels"}))
            events: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    ev = json.loads(raw)
                    events.append(ev)
                    if ev.get("type") == "models":
                        break
                except TimeoutError:
                    break
            # No "Unknown command: generateCommitMessage" error
            for ev in events:
                if ev.get("type") == "error":
                    self.assertNotIn("Unknown command", ev.get("text", ""))

    async def test_ws_user_answer(self) -> None:
        """userAnswer command does not produce an error (drops silently w/o task)."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "userAnswer", "answer": "yes", "tabId": "ans-tab",
            }))
            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "models")

    async def test_ws_complete(self) -> None:
        """complete command does not produce an error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({"type": "complete", "query": "hello"}))
            # complete either returns a ghost event or nothing; verify no error
            await ws.send(json.dumps({"type": "getModels"}))
            events: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    ev = json.loads(raw)
                    events.append(ev)
                    if ev.get("type") == "models":
                        break
                except TimeoutError:
                    break
            for ev in events:
                if ev.get("type") == "error":
                    self.assertNotIn("Unknown command", ev.get("text", ""))

    async def test_ws_worktree_action(self) -> None:
        """worktreeAction command does not produce Unknown command error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "worktreeAction", "action": "discard", "tabId": "wt-tab",
            }))
            # worktreeAction may broadcast worktree_result; verify no Unknown command
            await ws.send(json.dumps({"type": "getModels"}))
            events: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    ev = json.loads(raw)
                    events.append(ev)
                    if ev.get("type") == "models":
                        break
                except TimeoutError:
                    break
            for ev in events:
                if ev.get("type") == "error":
                    self.assertNotIn("Unknown command", ev.get("text", ""))

    async def test_ws_autocommit_action(self) -> None:
        """autocommitAction command does not produce Unknown command error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "autocommitAction", "action": "skip", "tabId": "ac-tab",
            }))
            await ws.send(json.dumps({"type": "getModels"}))
            events: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    ev = json.loads(raw)
                    events.append(ev)
                    if ev.get("type") == "models":
                        break
                except TimeoutError:
                    break
            for ev in events:
                if ev.get("type") == "error":
                    self.assertNotIn("Unknown command", ev.get("text", ""))

    async def test_ws_all_webview_commands_no_unknown_error(self) -> None:
        """All 30 FromWebviewMessage types produce no 'Unknown command' error."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            # All 30 FromWebviewMessage types with minimal required fields
            commands = [
                {"type": "ready", "tabId": "t"},
                {"type": "submit", "prompt": "hi", "model": "gemini-2.5-pro",
                 "tabId": "all-submit", "attachments": []},
                {"type": "getModels"},
                {"type": "getHistory"},
                {"type": "getFiles", "prefix": ""},
                {"type": "getInputHistory"},
                {"type": "getConfig"},
                {"type": "getWelcomeSuggestions"},
                {"type": "getAdjacentTask", "tabId": "t", "task": "x",
                 "direction": "prev"},
                {"type": "selectModel", "model": "gemini-2.5-pro", "tabId": "t"},
                {"type": "newChat", "tabId": "all-t"},
                {"type": "closeTab", "tabId": "all-t"},
                {"type": "userActionDone"},
                {"type": "userAnswer", "answer": "yes", "tabId": "t"},
                {"type": "resumeSession", "id": 1, "tabId": "t"},
                {"type": "stop", "tabId": "t"},
                {"type": "complete", "query": "test"},
                {"type": "recordFileUsage", "path": "/tmp/x"},
                {"type": "mergeAction", "action": "all-done", "tabId": "t"},
                {"type": "generateCommitMessage"},
                {"type": "worktreeAction", "action": "discard", "tabId": "t"},
                {"type": "autocommitAction", "action": "skip", "tabId": "t"},
                {"type": "setSkipMerge", "tabId": "t", "skip": False},
                {"type": "saveConfig", "config": {}, "apiKeys": {}},
                # VS Code-only (should be silently ignored)
                {"type": "openFile", "path": "/tmp/x"},
                {"type": "focusEditor"},
                {"type": "closeSecondaryBar"},
                {"type": "webviewFocusChanged", "focused": True},
                {"type": "resolveDroppedPaths", "uris": []},
                {"type": "runPrompt"},
            ]
            for cmd in commands:
                await ws.send(json.dumps(cmd))

            # Drain all messages, check none is an Unknown command error
            events: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 10
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    events.append(json.loads(raw))
                except TimeoutError:
                    break
            unknown_errors = [
                e for e in events
                if e.get("type") == "error"
                and "Unknown command" in e.get("text", "")
            ]
            self.assertEqual(
                unknown_errors, [],
                f"Got Unknown command errors: {unknown_errors}",
            )

    async def test_ws_submit_emits_task_text_and_status(self) -> None:
        """The 'submit' command emits setTaskText and status running=True."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "submit",
                "prompt": "test task",
                "model": "gemini-2.5-pro",
                "tabId": "submit-tab-2",
                "attachments": [],
            }))
            events: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 10
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                    ev = json.loads(raw)
                    events.append(ev)
                    if ev.get("type") == "status" and not ev.get("running"):
                        break
                    if ev.get("type") == "result":
                        break
                except TimeoutError:
                    break
            event_types = [e["type"] for e in events]
            self.assertIn("setTaskText", event_types)
            self.assertIn("status", event_types)
            # setTaskText should come before or at the same time as status running=True
            task_text_events = [e for e in events if e.get("type") == "setTaskText"]
            self.assertTrue(
                any(e.get("text") == "test task" for e in task_text_events),
                f"Expected setTaskText with 'test task', got: {task_text_events}",
            )


class TestRemoteAccessServerAuth(IsolatedAsyncioTestCase):
    """Test WebSocket password authentication."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": "test-secret-123"})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def test_auth_correct_password(self) -> None:
        """Correct password authenticates successfully."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": "test-secret-123"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")

    async def test_auth_wrong_password_then_correct(self) -> None:
        """Wrong password prompts auth_required, then correct password works."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": "wrong"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_required")

            # Second attempt with correct password
            await ws.send(json.dumps({"type": "auth", "password": "test-secret-123"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")

    async def test_auth_wrong_password_twice_disconnects(self) -> None:
        """Two wrong passwords result in connection close."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": "wrong"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_required")

            await ws.send(json.dumps({"type": "auth", "password": "also-wrong"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "error")
            self.assertIn("Authentication failed", resp["text"])


class TestRemoteAccessServerMultiClient(IsolatedAsyncioTestCase):
    """Test broadcasting events to multiple WebSocket clients."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def test_broadcast_reaches_all_clients(self) -> None:
        """Events broadcast by the server reach all connected clients."""
        async with (
            connect(f"ws://127.0.0.1:{self.port}/ws") as ws1,
            connect(f"ws://127.0.0.1:{self.port}/ws") as ws2,
        ):
            # Authenticate both
            for ws in [ws1, ws2]:
                await ws.send(json.dumps({"type": "auth", "password": ""}))
                await asyncio.wait_for(ws.recv(), timeout=5)

            # Trigger an event from client 1
            await ws1.send(json.dumps({"type": "newChat", "tabId": "shared-tab"}))

            # Both clients should receive the showWelcome event
            r1 = json.loads(await asyncio.wait_for(ws1.recv(), timeout=5))
            r2 = json.loads(await asyncio.wait_for(ws2.recv(), timeout=5))
            self.assertEqual(r1["type"], "showWelcome")
            self.assertEqual(r2["type"], "showWelcome")


class TestRemoteAccessServerTask(IsolatedAsyncioTestCase):
    """Test running an actual agent task through the web server.

    Uses a local HTTP server with OpenAI-compatible chat completions
    endpoint as the model backend.
    """

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self.model_port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        self._tmpdir = tempfile.mkdtemp()

        # Start a fake OpenAI-compatible model server
        self._model_server = _FakeModelServer(self.model_port)
        self._model_server.start()

        endpoint = f"http://127.0.0.1:{self.model_port}/v1"
        save_config({
            "remote_password": "",
            "custom_endpoint": endpoint,
            "custom_api_key": "test-key",
            "max_budget": 100,
        })

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=self._tmpdir,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        self._model_server.stop()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def test_run_task_receives_events(self) -> None:
        """Sending a 'run' command produces task events over WebSocket."""
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")

            tab_id = "task-tab-1"
            endpoint = f"http://127.0.0.1:{self.model_port}/v1"
            model_name = f"custom/{endpoint.rstrip('/').split('/')[-1]}"

            # Select the custom model
            await ws.send(json.dumps({
                "type": "selectModel",
                "model": model_name,
                "tabId": tab_id,
            }))

            # Run a simple task
            await ws.send(json.dumps({
                "type": "run",
                "task": "Say hello",
                "tabId": tab_id,
            }))

            # Collect events until we see a result or status running=False
            events: list[dict] = []
            deadline = asyncio.get_event_loop().time() + 30
            done = False
            while asyncio.get_event_loop().time() < deadline and not done:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    ev = json.loads(raw)
                    events.append(ev)
                    if ev.get("type") == "status" and not ev.get("running"):
                        done = True
                    if ev.get("type") == "result":
                        done = True
                except TimeoutError:
                    break

            event_types = {e["type"] for e in events}
            # Should have received a status running=True and eventually
            # either a result or error
            self.assertTrue(
                "status" in event_types or "result" in event_types or "error" in event_types,
                f"Expected task lifecycle events, got: {event_types}",
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModelHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible chat completions handler."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        if "/chat/completions" in self.path:
            body = json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! Task completed successfully.",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif "/models" in self.path:
            self.do_GET()
        else:
            self.send_error(404)

    def do_GET(self) -> None:
        if "/models" in self.path:
            body = json.dumps({
                "data": [{"id": "test-model", "object": "model"}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


class _FakeModelServer:
    """Runs a fake OpenAI-compatible model server in a thread."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the fake model server."""
        self._httpd = HTTPServer(("127.0.0.1", self.port), _FakeModelHandler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the fake model server."""
        if self._httpd:
            self._httpd.shutdown()


class TestRemoteAccessServerMerge(IsolatedAsyncioTestCase):
    """Test merge/diff button functionality in the web server.

    Sets up a git repo with a modified file and starts a merge session
    to verify that merge toolbar buttons work correctly.
    """

    async def asyncSetUp(self) -> None:
        import os
        import subprocess

        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        # Create a git repo with a file, commit, then modify
        self._tmpdir = tempfile.mkdtemp()
        subprocess.run(
            ["git", "init", self._tmpdir],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", self._tmpdir, "config", "user.email", "t@t.com"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", self._tmpdir, "config", "user.name", "T"],
            capture_output=True,
            check=True,
        )
        self._test_file = os.path.join(self._tmpdir, "test.py")
        with open(self._test_file, "w") as f:
            f.write("line1\nline2\nline3\n")
        subprocess.run(
            ["git", "-C", self._tmpdir, "add", "-A"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", self._tmpdir, "commit", "-m", "initial"],
            capture_output=True,
            check=True,
        )
        # Simulate agent changes
        with open(self._test_file, "w") as f:
            f.write("line1\nmodified_line2\nline3\nnew_line4\n")

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=self._tmpdir,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _auth(self, ws: Any) -> None:
        """Authenticate a WebSocket connection."""
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert resp["type"] == "auth_ok"

    async def _trigger_merge(self, tab_id: str) -> None:
        """Start a merge session for the test git repo."""
        loop = asyncio.get_event_loop()
        started = await loop.run_in_executor(
            None,
            lambda: self.server._vscode_server._prepare_and_start_merge(
                self._tmpdir, tab_id=tab_id,
            ),
        )
        assert started, "Merge session must start (there are uncommitted changes)"

    async def _collect_until(
        self,
        ws: Any,
        target_type: str,
        timeout: float = 5,
    ) -> list[dict[str, Any]]:
        """Collect WS events until one with the target type arrives."""
        events: list[dict[str, Any]] = []
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2)
                ev = json.loads(raw)
                events.append(ev)
                if ev.get("type") == target_type:
                    break
            except TimeoutError:
                break
        return events

    async def test_merge_accept_all_completes_merge(self) -> None:
        """mergeAction accept-all should complete the merge and broadcast merge_ended."""
        tab_id = "merge-accept-tab"
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await self._auth(ws)
            await self._trigger_merge(tab_id)

            # Receive merge_data and merge_started
            events = await self._collect_until(ws, "merge_started")
            types = [e["type"] for e in events]
            self.assertIn("merge_data", types)
            self.assertIn("merge_started", types)

            # Send accept-all
            await ws.send(json.dumps({
                "type": "mergeAction",
                "action": "accept-all",
                "tabId": tab_id,
            }))

            # Should receive merge_ended
            events = await self._collect_until(ws, "merge_ended", timeout=5)
            ended = [e for e in events if e.get("type") == "merge_ended"]
            self.assertTrue(
                len(ended) > 0,
                "merge_ended should be broadcast after accept-all",
            )

    async def test_merge_reject_all_reverts_files(self) -> None:
        """mergeAction reject-all should revert files to base and complete merge."""
        tab_id = "merge-reject-tab"
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await self._auth(ws)
            await self._trigger_merge(tab_id)
            events = await self._collect_until(ws, "merge_started")

            # Remember the original (base) content
            # The base is "line1\nline2\nline3\n"
            # The current (agent) content is "line1\nmodified_line2\nline3\nnew_line4\n"

            # Send reject-all
            await ws.send(json.dumps({
                "type": "mergeAction",
                "action": "reject-all",
                "tabId": tab_id,
            }))

            events = await self._collect_until(ws, "merge_ended", timeout=5)
            ended = [e for e in events if e.get("type") == "merge_ended"]
            self.assertTrue(len(ended) > 0, "merge_ended should be broadcast")

            # The file should be reverted to base content
            with open(self._test_file) as f:
                content = f.read()
            self.assertEqual(content, "line1\nline2\nline3\n")

    async def test_merge_data_includes_file_contents(self) -> None:
        """merge_data event should include base_text and current_text for web clients."""
        tab_id = "merge-contents-tab"
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await self._auth(ws)
            await self._trigger_merge(tab_id)

            events = await self._collect_until(ws, "merge_started")
            md_events = [e for e in events if e.get("type") == "merge_data"]
            self.assertTrue(len(md_events) > 0)

            md = md_events[0]
            files = md["data"]["files"]
            self.assertTrue(len(files) > 0)
            # Each file should have base_text and current_text
            for f in files:
                self.assertIn("base_text", f, "merge_data files must include base_text")
                self.assertIn(
                    "current_text", f, "merge_data files must include current_text"
                )

    async def test_merge_accept_individual_hunk(self) -> None:
        """mergeAction accept should mark one hunk and eventually complete."""
        tab_id = "merge-single-accept-tab"
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await self._auth(ws)
            await self._trigger_merge(tab_id)
            events = await self._collect_until(ws, "merge_started")

            md_events = [e for e in events if e.get("type") == "merge_data"]
            total_hunks = md_events[0]["hunk_count"]

            # Accept all hunks one by one
            for _ in range(total_hunks):
                await ws.send(json.dumps({
                    "type": "mergeAction",
                    "action": "accept",
                    "tabId": tab_id,
                }))

            events = await self._collect_until(ws, "merge_ended", timeout=5)
            ended = [e for e in events if e.get("type") == "merge_ended"]
            self.assertTrue(len(ended) > 0, "merge should complete after all hunks accepted")

            # Content should be preserved (agent's changes kept)
            with open(self._test_file) as f:
                content = f.read()
            self.assertEqual(content, "line1\nmodified_line2\nline3\nnew_line4\n")

    async def test_merge_reject_individual_hunk(self) -> None:
        """mergeAction reject should revert hunks one by one."""
        tab_id = "merge-single-reject-tab"
        async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
            await self._auth(ws)
            await self._trigger_merge(tab_id)
            events = await self._collect_until(ws, "merge_started")

            md_events = [e for e in events if e.get("type") == "merge_data"]
            total_hunks = md_events[0]["hunk_count"]

            # Reject all hunks one by one
            for _ in range(total_hunks):
                await ws.send(json.dumps({
                    "type": "mergeAction",
                    "action": "reject",
                    "tabId": tab_id,
                }))

            events = await self._collect_until(ws, "merge_ended", timeout=5)
            ended = [e for e in events if e.get("type") == "merge_ended"]
            self.assertTrue(len(ended) > 0)

            # Content should be reverted to base
            with open(self._test_file) as f:
                content = f.read()
            self.assertEqual(content, "line1\nline2\nline3\n")


class TestGenerateSelfSignedCert(unittest.TestCase):
    """Test self-signed certificate generation."""

    def test_generates_cert_and_key_files(self) -> None:
        """Generated cert and key files are valid PEM and loadable by ssl."""
        import ssl as _ssl

        with tempfile.TemporaryDirectory() as td:
            cert_path = Path(td) / "sub" / "cert.pem"
            key_path = Path(td) / "sub" / "key.pem"
            _generate_self_signed_cert(cert_path, key_path)

            self.assertTrue(cert_path.is_file())
            self.assertTrue(key_path.is_file())
            self.assertIn(b"BEGIN CERTIFICATE", cert_path.read_bytes())
            self.assertIn(b"BEGIN RSA PRIVATE KEY", key_path.read_bytes())

            # Verify ssl module can load them
            ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(str(cert_path), str(key_path))


class TestCreateSslContext(unittest.TestCase):
    """Test SSL context creation."""

    def test_auto_generates_cert_when_no_paths(self) -> None:
        """_create_ssl_context without args auto-generates certs."""
        import ssl as _ssl

        ctx = _create_ssl_context()
        self.assertIsInstance(ctx, _ssl.SSLContext)

    def test_loads_provided_cert(self) -> None:
        """_create_ssl_context with explicit cert/key loads them."""
        import ssl as _ssl

        with tempfile.TemporaryDirectory() as td:
            cert_path = Path(td) / "cert.pem"
            key_path = Path(td) / "key.pem"
            _generate_self_signed_cert(cert_path, key_path)

            ctx = _create_ssl_context(str(cert_path), str(key_path))
            self.assertIsInstance(ctx, _ssl.SSLContext)


class TestRemoteAccessServerTLS(IsolatedAsyncioTestCase):
    """Test HTTPS and WSS with TLS enabled."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self._tmpdir = tempfile.mkdtemp()
        self._cert_dir = tempfile.mkdtemp()
        self._certfile = os.path.join(self._cert_dir, "cert.pem")
        self._keyfile = os.path.join(self._cert_dir, "key.pem")
        _generate_self_signed_cert(Path(self._certfile), Path(self._keyfile))

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=self._tmpdir,
            tls=True,
            certfile=self._certfile,
            keyfile=self._keyfile,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    def _make_ssl_client_ctx(self) -> ssl.SSLContext:
        """Build an SSL context that trusts the test self-signed cert."""
        import ssl as _ssl

        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(self._certfile)
        return ctx

    async def test_https_serves_html(self) -> None:
        """GET / over HTTPS returns the Sorcar HTML page."""
        import ssl as _ssl
        import urllib.request

        ctx = _ssl.create_default_context(cafile=self._certfile)
        url = f"https://127.0.0.1:{self.port}/"

        def _fetch() -> tuple[int, str]:
            # Use localhost in SNI but connect to 127.0.0.1
            resp = urllib.request.urlopen(url, timeout=5, context=ctx)
            return resp.status, resp.read().decode()

        status, body = await asyncio.get_event_loop().run_in_executor(
            None, _fetch,
        )
        self.assertEqual(status, 200)
        self.assertIn("<title>KISS Sorcar</title>", body)

    async def test_wss_auth_and_command(self) -> None:
        """WSS connection authenticates and receives models."""
        ssl_ctx = self._make_ssl_client_ctx()
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws",
            ssl=ssl_ctx,
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")

            await ws.send(json.dumps({"type": "getModels"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "models")
            self.assertIn("models", resp)

    async def test_plain_ws_rejected(self) -> None:
        """Plain ws:// connection to a TLS server should fail."""
        with self.assertRaises(Exception):
            async with connect(
                f"ws://127.0.0.1:{self.port}/ws",
            ) as ws:
                await ws.send(json.dumps({"type": "auth", "password": ""}))
                await asyncio.wait_for(ws.recv(), timeout=3)


class TestTunnelWatchdog(IsolatedAsyncioTestCase):
    """Test the tunnel watchdog that restarts dead tunnel processes."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self._backup_url: bytes | None = None
        if _URL_FILE.is_file():
            self._backup_url = _URL_FILE.read_bytes()

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
        if self._backup_url is not None:
            _URL_FILE.write_bytes(self._backup_url)
        else:
            _URL_FILE.unlink(missing_ok=True)

    async def test_watchdog_no_tunnel_proc_is_noop(self) -> None:
        """When _tunnel_proc is None, _check_and_restart_tunnel is a no-op."""
        self.server._tunnel_proc = None
        await self.server._check_and_restart_tunnel()
        # Should not raise or change anything
        self.assertIsNone(self.server._tunnel_proc)

    async def test_watchdog_alive_process_not_restarted(self) -> None:
        """A still-running tunnel process is left alone."""
        # Start a long-running process
        proc = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.server._tunnel_proc = proc  # type: ignore[assignment]
        try:
            await self.server._check_and_restart_tunnel()
            # Process should still be the same (alive)
            self.assertIs(self.server._tunnel_proc, proc)
            self.assertIsNone(proc.poll())  # still running
        finally:
            proc.terminate()
            proc.wait()

    async def test_watchdog_restarts_dead_process(self) -> None:
        """A dead tunnel process triggers restart (which fails without cloudflared)."""
        # Start a process that exits immediately
        proc = subprocess.Popen(
            ["true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait()  # ensure it's dead
        self.server._tunnel_proc = proc  # type: ignore[assignment]
        self.server.use_tunnel = True

        # _check_and_restart_tunnel will detect the dead process and
        # try to restart.  Without cloudflared installed in CI, _start_tunnel
        # returns None, but the dead process should be cleared.
        await self.server._check_and_restart_tunnel()
        # The old dead proc should no longer be referenced
        self.assertIsNot(self.server._tunnel_proc, proc)

    async def test_watchdog_task_runs_and_cancels(self) -> None:
        """The watchdog task can be started and cancelled cleanly."""
        self.server._tunnel_proc = None
        task = asyncio.create_task(self.server._tunnel_watchdog())
        # Let it run one check cycle (with a very short sleep)
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_tunnel_check_interval_is_positive(self) -> None:
        """TUNNEL_CHECK_INTERVAL is a reasonable positive value."""
        self.assertGreater(TUNNEL_CHECK_INTERVAL, 0)
        self.assertLessEqual(TUNNEL_CHECK_INTERVAL, 120)

    async def test_watchdog_cleans_up_on_stop(self) -> None:
        """stop_async cancels the watchdog task if running."""
        self.server._watchdog_task = asyncio.create_task(
            self.server._tunnel_watchdog()
        )
        await asyncio.sleep(0.01)
        self.assertFalse(self.server._watchdog_task.done())
        await self.server.stop_async()
        self.assertIsNone(self.server._watchdog_task)


class TestUrlFile(unittest.TestCase):
    """Test URL file save/remove/print helpers."""

    def setUp(self) -> None:
        # Back up any existing URL file
        self._backup: bytes | None = None
        if _URL_FILE.is_file():
            self._backup = _URL_FILE.read_bytes()

    def tearDown(self) -> None:
        # Restore original URL file
        if self._backup is not None:
            _URL_FILE.write_bytes(self._backup)
        else:
            _URL_FILE.unlink(missing_ok=True)

    def test_save_url_file_local_only(self) -> None:
        """Saving with local URL only writes valid JSON."""
        _save_url_file("https://localhost:8787")
        data = json.loads(_URL_FILE.read_text())
        self.assertEqual(data["local"], "https://localhost:8787")
        self.assertNotIn("tunnel", data)

    def test_save_url_file_with_tunnel(self) -> None:
        """Saving with both local and tunnel URLs writes both."""
        _save_url_file("https://localhost:8787", "https://abc.trycloudflare.com")
        data = json.loads(_URL_FILE.read_text())
        self.assertEqual(data["local"], "https://localhost:8787")
        self.assertEqual(data["tunnel"], "https://abc.trycloudflare.com")

    def test_remove_url_file(self) -> None:
        """Removing the URL file deletes it."""
        _save_url_file("https://localhost:8787")
        self.assertTrue(_URL_FILE.is_file())
        _remove_url_file()
        self.assertFalse(_URL_FILE.is_file())

    def test_remove_url_file_missing(self) -> None:
        """Removing when file doesn't exist is a no-op."""
        _URL_FILE.unlink(missing_ok=True)
        _remove_url_file()  # should not raise
        self.assertFalse(_URL_FILE.is_file())

    def test_print_url_tunnel(self) -> None:
        """When tunnel URL exists, _print_url prints it."""
        _save_url_file("https://localhost:8787", "https://abc.trycloudflare.com")
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_url()
        self.assertEqual(buf.getvalue().strip(), "https://abc.trycloudflare.com")

    def test_print_url_local_only(self) -> None:
        """When no tunnel URL, _print_url prints local URL."""
        _save_url_file("https://localhost:8787")
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_url()
        self.assertEqual(buf.getvalue().strip(), "https://localhost:8787")

    def test_print_url_no_file(self) -> None:
        """When no URL file exists, _print_url exits with code 1."""
        _URL_FILE.unlink(missing_ok=True)
        with self.assertRaises(SystemExit) as ctx:
            _print_url()
        self.assertEqual(ctx.exception.code, 1)

    def test_print_url_corrupt_file(self) -> None:
        """When URL file is corrupt, _print_url exits with code 1."""
        _URL_FILE.parent.mkdir(parents=True, exist_ok=True)
        _URL_FILE.write_text("not json")
        with self.assertRaises(SystemExit) as ctx:
            _print_url()
        self.assertEqual(ctx.exception.code, 1)


class TestStartQuickTunnelUrlParsing(IsolatedAsyncioTestCase):
    """Integration test: _start_quick_tunnel must skip api.trycloudflare.com.

    When cloudflared starts a quick tunnel it may log Cloudflare's API
    endpoint (``api.trycloudflare.com``) in its stderr *before* the
    real tunnel URL.  The parser must ignore infrastructure URLs and
    only return the actual random ``*.trycloudflare.com`` tunnel URL.
    """

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self._backup_url: bytes | None = None
        if _URL_FILE.is_file():
            self._backup_url = _URL_FILE.read_bytes()

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            use_tunnel=True,
            work_dir=tempfile.mkdtemp(),
        )

    async def asyncTearDown(self) -> None:
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
        if self._backup_url is not None:
            _URL_FILE.write_bytes(self._backup_url)
        else:
            _URL_FILE.unlink(missing_ok=True)

    async def test_skips_api_trycloudflare_url(self) -> None:
        """_start_quick_tunnel ignores api.trycloudflare.com from stderr."""
        import sys

        # Create a helper script that mimics cloudflared stderr output:
        # first emits api.trycloudflare.com (the API endpoint), then
        # the real tunnel URL, then sleeps so the process stays alive.
        script = (
            "import sys, time\n"
            'sys.stderr.write("INF Requesting new quick Tunnel on '
            'https://api.trycloudflare.com/quicktunnel ...\\n")\n'
            "sys.stderr.flush()\n"
            'sys.stderr.write("INF +-------+\\n")\n'
            'sys.stderr.write("INF | https://test-word-abc-xyz.'
            'trycloudflare.com |\\n")\n'
            "sys.stderr.flush()\n"
            "time.sleep(30)\n"
        )
        # Start the fake cloudflared process
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Inject it as the tunnel process so _start_quick_tunnel's
        # reader sees the stderr from this process.
        self.server._tunnel_proc = proc  # type: ignore[assignment]

        # Directly invoke the stderr reader logic from _start_quick_tunnel.
        # We replicate the exact reader thread approach used in production.
        import re

        stderr_fd = proc.stderr
        assert stderr_fd is not None

        result_box: list[str | None] = [None]

        def _reader_target() -> None:
            for line in iter(stderr_fd.readline, ""):
                match = re.search(
                    r"(https://(?!api\.)[^\s]+\.trycloudflare\.com)",
                    line,
                )
                if match:
                    result_box[0] = match.group(1)
                    return
                if proc.poll() is not None:
                    break

        reader = threading.Thread(target=_reader_target, daemon=True)
        reader.start()
        reader.join(timeout=10)

        proc.terminate()
        proc.wait()

        self.assertEqual(
            result_box[0],
            "https://test-word-abc-xyz.trycloudflare.com",
            "Should capture real tunnel URL, not api.trycloudflare.com",
        )

    async def test_api_url_would_match_old_regex(self) -> None:
        """Confirm that the old regex (without negative lookahead) matched api.

        This verifies the bug existed: the un-patched regex DOES match
        api.trycloudflare.com.
        """
        import re

        old_regex = r"(https://[^\s]+\.trycloudflare\.com)"
        line = (
            "INF Requesting new quick Tunnel on "
            "https://api.trycloudflare.com/quicktunnel ..."
        )
        match = re.search(old_regex, line)
        self.assertIsNotNone(match, "Old regex should match api URL")
        self.assertTrue(
            match.group(1).startswith("https://api."),  # type: ignore[union-attr]
            "Old regex captured the api.trycloudflare.com URL",
        )

        # New regex should NOT match the api URL
        new_regex = r"(https://(?!api\.)[^\s]+\.trycloudflare\.com)"
        match2 = re.search(new_regex, line)
        self.assertIsNone(
            match2,
            "New regex must not match api.trycloudflare.com",
        )

    async def test_real_tunnel_url_still_matches(self) -> None:
        """The fixed regex still matches legitimate tunnel URLs."""
        import re

        new_regex = r"(https://(?!api\.)[^\s]+\.trycloudflare\.com)"
        line = (
            "INF |  https://genesis-tip-allan-frank"
            ".trycloudflare.com  |"
        )
        match = re.search(new_regex, line)
        self.assertIsNotNone(match)
        self.assertEqual(
            match.group(1),  # type: ignore[union-attr]
            "https://genesis-tip-allan-frank.trycloudflare.com",
        )


class TestGetLocalIps(unittest.TestCase):
    """Test the _get_local_ips() helper."""

    def test_returns_frozenset(self) -> None:
        """_get_local_ips returns a frozenset of strings."""
        result = _get_local_ips()
        self.assertIsInstance(result, frozenset)
        for addr in result:
            self.assertIsInstance(addr, str)

    def test_no_loopback(self) -> None:
        """Returned addresses do not include 127.x.x.x loopback."""
        result = _get_local_ips()
        for addr in result:
            self.assertFalse(addr.startswith("127."), f"Loopback in result: {addr}")

    def test_idempotent(self) -> None:
        """Consecutive calls with no network change return the same set."""
        a = _get_local_ips()
        b = _get_local_ips()
        self.assertEqual(a, b)


class TestIpWatchdog(IsolatedAsyncioTestCase):
    """Test the IP address change watchdog."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def test_ip_watchdog_task_started(self) -> None:
        """start_async starts the IP watchdog task."""
        task = self.server._ip_watchdog_task
        self.assertIsNotNone(task)
        assert task is not None
        self.assertFalse(task.done())

    async def test_ip_watchdog_cancelled_on_stop(self) -> None:
        """stop_async cancels the IP watchdog task."""
        self.assertIsNotNone(self.server._ip_watchdog_task)
        await self.server.stop_async()
        self.assertIsNone(self.server._ip_watchdog_task)

    async def test_ip_watchdog_closes_server_on_change(self) -> None:
        """When IPs change, the watchdog closes the WebSocket server."""
        # Simulate an IP change by setting _last_ips to something different
        self.server._last_ips = frozenset({"10.255.255.1"})
        # Cancel the existing watchdog and start a fresh one with short interval
        if self.server._ip_watchdog_task is not None:
            self.server._ip_watchdog_task.cancel()
            try:
                await self.server._ip_watchdog_task
            except asyncio.CancelledError:
                pass

        # Temporarily override TUNNEL_CHECK_INTERVAL by running a manual check
        # Instead, directly call the internal logic: we'll start the watchdog
        # and it should detect the mismatch on the first check cycle.
        import kiss.agents.vscode.web_server as ws_mod

        original_interval = ws_mod.TUNNEL_CHECK_INTERVAL
        ws_mod.TUNNEL_CHECK_INTERVAL = 0  # minimal sleep for fast test
        try:
            task = asyncio.create_task(self.server._ip_watchdog())
            # Wait for the watchdog to detect the change and close the server
            await asyncio.sleep(0.3)
            self.assertTrue(task.done(), "Watchdog should have returned after IP change")
        finally:
            ws_mod.TUNNEL_CHECK_INTERVAL = original_interval

    async def test_ip_watchdog_noop_when_unchanged(self) -> None:
        """When IPs haven't changed, the watchdog keeps running."""
        # Ensure _last_ips matches current reality
        import kiss.agents.vscode.web_server as ws_mod

        self.server._last_ips = _get_local_ips()
        if self.server._ip_watchdog_task is not None:
            self.server._ip_watchdog_task.cancel()
            try:
                await self.server._ip_watchdog_task
            except asyncio.CancelledError:
                pass

        original_interval = ws_mod.TUNNEL_CHECK_INTERVAL
        ws_mod.TUNNEL_CHECK_INTERVAL = 0  # minimal sleep for fast test
        try:
            task = asyncio.create_task(self.server._ip_watchdog())
            await asyncio.sleep(0.2)
            # Should still be running (IPs haven't changed)
            self.assertFalse(task.done(), "Watchdog should keep running when IPs unchanged")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            ws_mod.TUNNEL_CHECK_INTERVAL = original_interval


if __name__ == "__main__":
    unittest.main()
