"""Browser output streaming for KISS agents via SSE."""

import asyncio
import json
import queue
import time
import webbrowser
from collections.abc import AsyncGenerator
from typing import Any

from kiss.core.browser_ui import BaseBrowserPrinter, build_stream_viewer_html, find_free_port


class BrowserPrinter(BaseBrowserPrinter):
    def __init__(self) -> None:
        super().__init__()
        self._port = 0
        self._server: Any = None

    def start(self, open_browser: bool = True) -> None:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import HTMLResponse, StreamingResponse
        from starlette.routing import Route

        html_page = build_stream_viewer_html()
        printer = self

        async def index(request: Request) -> HTMLResponse:
            return HTMLResponse(html_page)

        async def events(request: Request) -> StreamingResponse:
            client_q: queue.Queue[dict[str, Any]] = queue.Queue()
            with printer._lock:
                printer._clients.append(client_q)

            async def generate() -> AsyncGenerator[str]:
                try:
                    while True:
                        try:
                            event = client_q.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.05)
                            continue
                        yield f"data: {json.dumps(event)}\n\n"
                        if event.get("type") == "done":
                            break
                except asyncio.CancelledError:
                    pass
                finally:
                    with printer._lock:
                        if client_q in printer._clients:
                            printer._clients.remove(client_q)

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        import threading

        app = Starlette(routes=[Route("/", index), Route("/events", events)])
        self._port = find_free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        threading.Thread(target=self._server.run, daemon=True).start()
        time.sleep(0.5)
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{self._port}")

    def stop(self) -> None:
        self.broadcast({"type": "done"})
        time.sleep(0.3)
        if self._server:
            self._server.should_exit = True
