"""Tests for web_use_tool.py module."""

import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kiss.core.web_use_tool import (
    KISS_PROFILE_DIR,
    WebUseTool,
    _number_interactive_elements,
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
    def test_go_to_url(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/")
        assert "Page: Test Form" in result
        assert "URL:" in result
        assert "[" in result

    def test_go_to_url_returns_interactive_elements(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/")
        assert "link" in result
        assert "textbox" in result
        assert "button" in result

    def test_go_to_second_page(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/second")
        assert "Page: Second Page" in result
        assert "Second Page" in result

    def test_go_to_invalid_url(self, web_tool):
        result = web_tool.go_to_url("http://localhost:99999/nonexistent")
        assert "Error" in result

    def test_go_to_empty_page(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/empty")
        assert "Page: Empty" in result


class TestAccessibilityTree:
    def test_tree_has_element_ids(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/")
        ids = re.findall(r"\[(\d+)\]", result)
        assert len(ids) >= 4

    def test_tree_shows_roles(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/")
        assert "textbox" in result
        assert "button" in result

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

    def test_tree_roles_page(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/roles")
        assert "button" in result
        assert "link" in result
        assert "Role Button" in result

    def test_tree_contenteditable(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/roles")
        assert "Editable div" in result


class TestClick:
    def test_click_button(self, http_server, web_tool):
        dom = web_tool.go_to_url(http_server + "/")
        match = re.search(r"\[(\d+)\].*button.*Action", dom)
        assert match, f"No Action button found:\n{dom}"
        btn_id = int(match.group(1))
        result = web_tool.click(btn_id)
        assert "Clicked!" in result

    def test_click_link(self, http_server, web_tool):
        dom = web_tool.go_to_url(http_server + "/")
        match = re.search(r"\[(\d+)\].*link.*Go to second page", dom)
        assert match, f"No link found:\n{dom}"
        link_id = int(match.group(1))
        result = web_tool.click(link_id)
        assert "Second Page" in result

    def test_click_invalid_id(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.click(99999)
        assert "Error" in result
        assert "not found" in result

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
    def test_press_escape(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.press_key("Escape")
        assert "Page:" in result

    def test_press_tab(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.press_key("Tab")
        assert "Page:" in result

    def test_press_page_down(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/long")
        result = web_tool.press_key("PageDown")
        assert "Page:" in result

    def test_press_invalid_key(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.press_key("NonExistentKey12345")
        assert "Error" in result


class TestScroll:
    def test_scroll_down(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/long")
        result = web_tool.scroll("down", 3)
        assert "Page:" in result

    def test_scroll_up(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/long")
        result = web_tool.scroll("up", 2)
        assert "Page:" in result

    def test_scroll_default(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/long")
        result = web_tool.scroll()
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
    def test_lazy_init(self):
        tool = WebUseTool(user_data_dir=None)
        assert tool._page is None
        assert tool._browser is None

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

    def test_constructor_accepts_browser_types(self):
        for browser_type in ["chromium", "firefox", "webkit"]:
            tool = WebUseTool(browser_type=browser_type, headless=True, user_data_dir=None)
            assert tool.browser_type == browser_type
            assert tool._page is None


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

    def test_explicit_none_gives_no_profile(self):
        tool = WebUseTool(user_data_dir=None)
        assert tool.user_data_dir is None

    def test_explicit_path_is_used(self):
        tool = WebUseTool(user_data_dir="/tmp/custom_profile")
        assert tool.user_data_dir == "/tmp/custom_profile"

    def test_no_profile_uses_regular_browser(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        assert web_tool._context is not None
        assert web_tool._browser is not None

    def test_user_data_dir_stored_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WebUseTool(
                browser_type="chromium", headless=True, user_data_dir=tmpdir
            )
            assert tool.user_data_dir == tmpdir
            assert tool._browser is None
            assert tool._context is None


class TestNumberInteractiveElements:
    def test_numbers_buttons(self):
        snapshot = '- button "OK"\n- button "Cancel"'
        result, elements = _number_interactive_elements(snapshot)
        assert '[1] button "OK"' in result
        assert '[2] button "Cancel"' in result
        assert len(elements) == 2
        assert elements[0] == {"role": "button", "name": "OK"}

    def test_skips_non_interactive(self):
        snapshot = '- heading "Title" [level=1]\n- button "Submit"'
        result, elements = _number_interactive_elements(snapshot)
        assert "[1]" not in result.split("\n")[0]
        assert '[1] button "Submit"' in result
        assert len(elements) == 1

    def test_handles_nameless_elements(self):
        snapshot = "- combobox"
        result, elements = _number_interactive_elements(snapshot)
        assert "[1] combobox" in result
        assert elements[0]["name"] == ""

    def test_preserves_indentation(self):
        snapshot = '  - link "Home"'
        result, elements = _number_interactive_elements(snapshot)
        assert '  - [1] link "Home"' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
