"""Integration tests for the KISS Sorcar remote web access server.

Tests cover HTTP serving, WebSocket communication, password authentication,
command dispatch, and event broadcasting through the web server.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.agents.vscode.vscode_config import CONFIG_PATH, save_config
from kiss.agents.vscode.web_server import RemoteAccessServer, WebPrinter, _build_html


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
        """The 'ready' command returns models, inputHistory, configData, and focusInput."""
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
            # configData, and focusInput (order may vary)
            received_types: set[str] = set()
            for _ in range(4):
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                ev = json.loads(raw)
                received_types.add(ev["type"])
            self.assertIn("models", received_types)
            self.assertIn("inputHistory", received_types)
            self.assertIn("configData", received_types)
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
            for _ in range(4):
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


if __name__ == "__main__":
    unittest.main()
