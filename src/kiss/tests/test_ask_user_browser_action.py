"""Tests for ask_user_browser_action in web_use_tool.py and related components."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool

TEST_PAGE = b"""<!DOCTYPE html>
<html><head><title>CAPTCHA Page</title></head>
<body>
  <h1>Solve CAPTCHA</h1>
  <button id="verify">Verify</button>
</body></html>"""

OTHER_PAGE = b"""<!DOCTYPE html>
<html><head><title>Other Page</title></head>
<body>
  <h1>Other Page</h1>
  <a href="/">Back</a>
</body></html>"""


@pytest.fixture(scope="module")
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            pages = {"/": TEST_PAGE, "/other": OTHER_PAGE}
            content = pages.get(self.path, TEST_PAGE)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


# ---------------------------------------------------------------------------
# kiss/agents/sorcar/web_use_tool.py — WebUseTool
# ---------------------------------------------------------------------------

class TestAskUserBrowserAction:
    def test_with_url_navigates_first(self, http_server: str) -> None:
        """When url is provided, it navigates to that URL before returning."""
        tool = WebUseTool(user_data_dir=None, wait_for_user_callback=None)
        try:
            tool.go_to_url(http_server)
            result = tool.ask_user_browser_action(
                "Check this page", url=f"{http_server}/other"
            )
            assert "Other Page" in result
        finally:
            tool.close()

class TestWaitForUserBrowserCallback:
    """Test the server-side callback pattern using threading.Event."""

    def test_event_based_callback(self) -> None:
        """Simulates the full callback flow: broadcast, block, unblock."""
        broadcasts: list[dict] = []
        user_action_event: threading.Event | None = None

        def _wait_for_user_browser(instruction: str, url: str) -> None:
            nonlocal user_action_event
            event = threading.Event()
            user_action_event = event
            broadcasts.append({
                "type": "user_browser_action",
                "instruction": instruction,
                "url": url,
            })
            while not event.wait(timeout=0.1):
                pass  # In real code, would check stop event
            user_action_event = None

        # Simulate: agent calls callback in a thread, "user" sets event
        done = threading.Event()
        def agent_thread() -> None:
            _wait_for_user_browser("Solve CAPTCHA", "http://example.com")
            done.set()

        t = threading.Thread(target=agent_thread)
        t.start()

        # Small delay to let the callback start blocking
        import time
        time.sleep(0.2)
        assert len(broadcasts) == 1
        assert broadcasts[0]["instruction"] == "Solve CAPTCHA"
        assert user_action_event is not None

        # Simulate user clicking "I'm Done"
        user_action_event.set()
        t.join(timeout=5)
        assert done.is_set()
        assert not t.is_alive()

    def test_stop_event_interrupts_callback(self) -> None:
        """If stop event is set, callback raises KeyboardInterrupt."""
        current_stop_event = threading.Event()

        def _wait_for_user_browser(instruction: str, url: str) -> None:
            event = threading.Event()
            while not event.wait(timeout=0.1):
                if current_stop_event.is_set():
                    raise KeyboardInterrupt("Agent stopped while waiting for user")

        current_stop_event.set()
        with pytest.raises(KeyboardInterrupt, match="Agent stopped"):
            _wait_for_user_browser("Solve CAPTCHA", "http://example.com")
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
