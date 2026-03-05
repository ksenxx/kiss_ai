"""Tests for web_use_tool.py module."""

import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kiss.agents.sorcar.web_use_tool import (
    KISS_PROFILE_DIR,
    WebUseTool,
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
    tool = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
    yield tool
    tool.close()


class TestNavigation:

    def test_go_to_invalid_url(self, web_tool):
        result = web_tool.go_to_url("http://localhost:99999/nonexistent")
        assert "Error" in result

    def test_go_to_empty_page(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/empty")
        assert "Page: Empty" in result


class TestAccessibilityTree:

    def test_get_page_content_tree(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.get_page_content()
        assert "Page: Test Form" in result
        assert "[" in result

    def test_get_page_content_text_only(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.get_page_content(text_only=True)
        assert "Page: Test Form" in result
        assert "Test Form Page" in result


class TestClick:

    def test_click_link(self, http_server, web_tool):
        dom = web_tool.go_to_url(http_server + "/")
        match = re.search(r"\[(\d+)\].*link.*Go to second page", dom)
        assert match, f"No link found:\n{dom}"
        link_id = int(match.group(1))
        result = web_tool.click(link_id)
        assert "Second Page" in result

    def test_hover(self, http_server, web_tool):
        dom = web_tool.go_to_url(http_server + "/")
        match = re.search(r"\[(\d+)\].*Hover me", dom)
        if match:
            hover_id = int(match.group(1))
            result = web_tool.click(hover_id, action="hover")
            assert "Page:" in result

    def test_hover_invalid_id(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.click(99999, action="hover")
        assert "Error" in result
        assert "not found" in result


class TestTypeText:
    def test_type_into_input(self, http_server, web_tool):
        dom = web_tool.go_to_url(http_server + "/")
        match = re.search(r"\[(\d+)\].*textbox.*[Uu]sername", dom)
        assert match, f"No username input found:\n{dom}"
        input_id = int(match.group(1))
        result = web_tool.type_text(input_id, "testuser")
        assert "testuser" in result

    def test_type_with_enter(self, http_server, web_tool):
        dom = web_tool.go_to_url(http_server + "/")
        match = re.search(r"\[(\d+)\].*textbox.*[Uu]sername", dom)
        assert match
        input_id = int(match.group(1))
        result = web_tool.type_text(input_id, "hello", press_enter=True)
        assert "Page:" in result

    def test_type_invalid_id(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.type_text(99999, "test")
        assert "Error" in result
        assert "not found" in result


class TestPressKey:

    def test_press_tab(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.press_key("Tab")
        assert "Page:" in result

    def test_press_invalid_key(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.press_key("NonExistentKey12345")
        assert "Error" in result


class TestScroll:

    def test_scroll_up(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/long")
        result = web_tool.scroll("up", 2)
        assert "Page:" in result


class TestTabManagement:
    def test_tab_list(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.go_to_url("tab:list")
        assert "Open tabs" in result
        assert "(active)" in result

    def test_tab_switch_valid(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.go_to_url("tab:0")
        assert "Page:" in result

    def test_tab_switch_invalid(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.go_to_url("tab:999")
        assert "Error" in result
        assert "out of range" in result


class TestScreenshot:
    def test_screenshot(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_screenshot.png")
            result = web_tool.screenshot(path)
            assert "Screenshot saved" in result
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0


class TestBrowserLifecycle:

    def test_close_and_reuse(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.close()
        assert "Browser closed" in result
        assert web_tool._page is None
        result = web_tool.go_to_url(http_server + "/")
        assert "Page: Test Form" in result

    def test_close_when_never_opened(self):
        tool = WebUseTool(user_data_dir=None)
        result = tool.close()
        assert "Browser closed" in result

    def test_get_tools_returns_all_methods(self):
        tool = WebUseTool(user_data_dir=None)
        tools = tool.get_tools()
        names = {t.__name__ for t in tools}
        assert names == {
            "go_to_url",
            "click",
            "type_text",
            "press_key",
            "scroll",
            "screenshot",
            "get_page_content",
        }


class TestAxTreeTruncation:
    def test_truncation(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool._get_ax_tree(max_chars=50)
        assert "... [truncated]" in result


class TestKissProfile:
    def test_kiss_profile_dir_is_under_home(self):
        assert ".kiss" in KISS_PROFILE_DIR
        assert "browser_profile" in KISS_PROFILE_DIR

    def test_default_constructor_uses_kiss_profile(self):
        tool = WebUseTool()
        assert tool.user_data_dir == KISS_PROFILE_DIR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
