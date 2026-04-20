"""Tests for web_use_tool.py module."""

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.agents.sorcar.web_use_tool import (
    _SINGLETON_FILES,
    WebUseTool,
    _activate_app,
    _get_frontmost_app,
)

FORM_PAGE = b"""<!DOCTYPE html>
<html><head><title>Test Form</title></head>
<body>
  <h1>Test Form Page</h1>
  <a href="/second">Go to second page</a>
  <form>
    <label for="username">Username</label>
    <input type="text" id="username" name="username" placeholder="Enter username">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" placeholder="Enter password">
    <label for="color">Color</label>
    <select id="color" name="color">
      <option value="red">Red</option>
      <option value="green">Green</option>
      <option value="blue">Blue</option>
    </select>
    <label for="bio">Bio</label>
    <textarea id="bio" name="bio" placeholder="Bio"></textarea>
    <button type="submit">Submit</button>
  </form>
  <button id="action-btn" onclick="document.title='Clicked!'">Action</button>
  <div id="hover-target" onmouseover="this.textContent='Hovered!'"
       style="padding:20px;background:#eee;" role="button" tabindex="0">Hover me</div>
</body></html>"""

SECOND_PAGE = b"""<!DOCTYPE html>
<html><head><title>Second Page</title></head>
<body>
  <h1>Second Page</h1>
  <a href="/">Back to form</a>
  <p>Content on second page.</p>
</body></html>"""

LONG_PAGE = b"""<!DOCTYPE html>
<html><head><title>Long Page</title></head>
<body style="height: 5000px;">
  <h1>Top of page</h1>
  <div style="position: absolute; top: 3000px;">
    <p>Bottom content</p>
  </div>
</body></html>"""

ROLE_PAGE = b"""<!DOCTYPE html>
<html><head><title>Role Page</title></head>
<body>
  <div role="button" tabindex="0">Role Button</div>
  <div role="link" tabindex="0">Role Link</div>
  <div contenteditable="true" role="textbox" aria-label="Editable div">Editable div</div>
</body></html>"""

EMPTY_PAGE = b"""<!DOCTYPE html>
<html><head><title>Empty</title></head>
<body></body></html>"""

NEW_TAB_PAGE = b"""<!DOCTYPE html>
<html><head><title>New Tab Page</title></head>
<body>
  <a href="/second" target="_blank" id="newtab-link">Open in new tab</a>
</body></html>"""

KEY_PAGE = b"""<!DOCTYPE html>
<html><head><title>Key Test</title></head>
<body>
  <input type="text" id="key-input" onkeydown="this.value=event.key">
  <div id="key-result"></div>
</body></html>"""


@pytest.fixture(scope="module")
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            pages = {
                "/": FORM_PAGE,
                "/second": SECOND_PAGE,
                "/long": LONG_PAGE,
                "/roles": ROLE_PAGE,
                "/empty": EMPTY_PAGE,
                "/newtab": NEW_TAB_PAGE,
                "/keytest": KEY_PAGE,
            }
            content = pages.get(self.path, FORM_PAGE)
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


@pytest.fixture(scope="module")
def web_tool():
    tool = WebUseTool(user_data_dir=None, headless=True)
    yield tool
    tool.close()


class TestNavigation:

    def test_go_to_invalid_url(self, web_tool):
        result = web_tool.go_to_url("http://localhost:99999/nonexistent")
        assert "Error" in result


class TestCrashRecovery:
    """Verify the tool auto-recovers after Chromium/context dies unexpectedly.

    Simulates the "Google Chrome for Testing quit unexpectedly" scenario by
    closing the browser context out from under the tool.
    """

    def test_auto_relaunch_after_context_close(self, web_tool, http_server):
        web_tool.go_to_url(http_server + "/")
        assert web_tool._is_alive()
        # Simulate a crash: drop the browser context without notifying the tool.
        web_tool._context.close()
        assert not web_tool._is_alive()
        # Next call should transparently relaunch and succeed.
        result = web_tool.go_to_url(http_server + "/")
        assert "Test" in result
        assert web_tool._is_alive()


class TestSingletonLockCleanup:
    """Stale Singleton{Lock,Cookie,Socket} from a previously crashed Chromium
    must be removed before launching a persistent context."""

    def test_cleans_stale_singleton_files(self, tmp_path):
        (tmp_path / _SINGLETON_FILES[0]).symlink_to("stale-host-99999")
        (tmp_path / _SINGLETON_FILES[1]).write_text("stale")
        tool = WebUseTool(user_data_dir=str(tmp_path), headless=True)
        try:
            tool._clean_singleton_locks()
            for name in _SINGLETON_FILES:
                assert not (tmp_path / name).exists()
                assert not (tmp_path / name).is_symlink()
        finally:
            tool.close()

    def test_clean_singleton_locks_no_profile(self):
        """Called on an in-memory tool — no-op, does not raise."""
        tool = WebUseTool(user_data_dir=None, headless=True)
        tool._clean_singleton_locks()  # must not raise
        tool.close()


class TestFocusHelpers:
    """Tests for _get_frontmost_app and _activate_app focus management."""

    def test_get_frontmost_app_returns_string_on_macos(self):
        """On macOS, _get_frontmost_app should return the current app name."""
        result = _get_frontmost_app()
        if sys.platform == "darwin":
            assert isinstance(result, str)
            assert len(result) > 0
        else:
            assert result is None

    def test_activate_app_none_is_noop(self):
        """_activate_app(None) should silently do nothing."""
        _activate_app(None)  # must not raise

    def test_activate_app_with_valid_app(self):
        """_activate_app with a real app should not raise."""
        if sys.platform == "darwin":
            _activate_app("Finder")  # Finder is always running on macOS

    def test_activate_app_with_nonexistent_app(self):
        """_activate_app with a bogus name should not raise (best-effort)."""
        _activate_app("NonExistentApp12345")  # must not raise

    def test_ensure_browser_calls_focus_helpers(self, web_tool):
        """_ensure_browser should save and restore focus even in headless mode."""
        # Force a relaunch by dropping the context (exercises the finally block)
        web_tool._context.close()
        assert not web_tool._is_alive()
        web_tool._ensure_browser()
        assert web_tool._is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
