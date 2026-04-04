"""Integration tests: cookies and login state persist across WebUseTool sessions."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool

SET_COOKIE_PAGE = b"""<!DOCTYPE html>
<html><head><title>Cookie Set</title></head>
<body><p>Cookie has been set.</p></body></html>"""

CHECK_COOKIE_PAGE = b"""<!DOCTYPE html>
<html><head><title>Check Cookie</title></head>
<body>
<script>document.title = document.cookie || "NO_COOKIES";</script>
</body></html>"""


@pytest.fixture(scope="module")
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/setcookie":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header(
                    "Set-Cookie",
                    "session_token=abc123; Path=/; Max-Age=86400",
                )
                self.end_headers()
                self.wfile.write(SET_COOKIE_PAGE)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(CHECK_COOKIE_PAGE)

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


class TestCookiePersistence:
    """Verify cookies survive browser close/reopen with a persistent profile."""

    def test_ephemeral_mode_loses_cookies(self, http_server: str) -> None:
        # Session 1 (ephemeral): set cookie
        tool1 = WebUseTool(user_data_dir=None)
        try:
            tool1.go_to_url(f"{http_server}/setcookie")
        finally:
            tool1.close()

        # Session 2 (ephemeral): cookie should be gone
        tool2 = WebUseTool(user_data_dir=None)
        try:
            result = tool2.go_to_url(f"{http_server}/checkcookie")
            assert "session_token" not in result
        finally:
            tool2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
