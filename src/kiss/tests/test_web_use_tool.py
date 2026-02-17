"""Tests for web_use_tool.py module."""

import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kiss.core.web_use_tool import WebUseTool

FORM_PAGE = b"""<!DOCTYPE html>
<html><head><title>Test Form</title></head>
<body>
  <h1>Test Form Page</h1>
  <a href="/second">Go to second page</a>
  <form>
    <input type="text" name="username" placeholder="Enter username">
    <input type="password" name="password" placeholder="Enter password">
    <select name="color">
      <option value="red">Red</option>
      <option value="green">Green</option>
      <option value="blue">Blue</option>
    </select>
    <textarea name="bio" placeholder="Bio"></textarea>
    <button type="submit">Submit</button>
  </form>
  <button id="action-btn" onclick="document.title='Clicked!'">Action</button>
  <div id="hover-target" onmouseover="this.textContent='Hovered!'"
       style="padding:20px;background:#eee;">Hover me</div>
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
  <div contenteditable="true">Editable div</div>
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
    tool = WebUseTool(browser_type="chromium", headless=True)
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
        assert "<a" in result
        assert "<input" in result
        assert "<button" in result
        assert "<select" in result
        assert "<textarea" in result
        assert "placeholder=" in result

    def test_go_to_second_page(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/second")
        assert "Page: Second Page" in result
        assert "Second Page" in result

    def test_go_back_via_execute_js(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        web_tool.go_to_url(http_server + "/second")
        result = web_tool.execute_js("history.back()")
        assert "Page: Test Form" in result

    def test_go_to_invalid_url(self, web_tool):
        result = web_tool.go_to_url("http://localhost:99999/nonexistent")
        assert "Error" in result

    def test_go_to_empty_page(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/empty")
        assert "Page: Empty" in result


class TestDOMTree:
    def test_dom_tree_has_element_ids(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/")
        ids = re.findall(r"\[(\d+)\]", result)
        assert len(ids) >= 5

    def test_dom_tree_shows_attributes(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/")
        assert 'type="text"' in result
        assert 'type="password"' in result
        assert 'name="username"' in result
        assert 'placeholder="Enter username"' in result

    def test_dom_tree_shows_select_options(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/")
        assert "<option>" in result
        assert "Red" in result
        assert "Green" in result
        assert "Blue" in result

    def test_get_page_content_dom(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.get_page_content()
        assert "Page: Test Form" in result
        assert "[" in result

    def test_get_page_content_text_only(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.get_page_content(text_only=True)
        assert "Page: Test Form" in result
        assert "Test Form Page" in result
        assert "[" not in result

    def test_dom_tree_roles(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/roles")
        assert 'role="button"' in result
        assert 'role="link"' in result
        assert "Role Button" in result

    def test_dom_tree_contenteditable(self, http_server, web_tool):
        result = web_tool.go_to_url(http_server + "/roles")
        assert "Editable div" in result


class TestClick:
    def test_click_button(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        dom = web_tool.get_page_content()
        match = re.search(r"\[(\d+)\] <button>Action</button>", dom)
        assert match, f"No Action button found in DOM:\n{dom}"
        btn_id = int(match.group(1))
        result = web_tool.click(btn_id)
        assert "Clicked!" in result

    def test_click_link(self, http_server, web_tool):
        dom = web_tool.go_to_url(http_server + "/")
        match = re.search(r'\[(\d+)\] <a href="/second"', dom)
        assert match, f"No link found in DOM:\n{dom}"
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
        match = re.search(r'\[(\d+)\] <input type="text"', dom)
        assert match, f"No text input found in DOM:\n{dom}"
        input_id = int(match.group(1))
        result = web_tool.type_text(input_id, "testuser")
        assert 'value="testuser"' in result

    def test_type_with_enter(self, http_server, web_tool):
        dom = web_tool.go_to_url(http_server + "/")
        match = re.search(r'\[(\d+)\] <input type="text"', dom)
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


class TestExecuteJs:
    def test_execute_js_returns_result(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.execute_js("document.title")
        assert "Result: Test Form" in result

    def test_execute_js_scroll(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/long")
        result = web_tool.execute_js("window.scrollBy(0, 1000)")
        assert "Page:" in result

    def test_execute_js_select_option(self, http_server, web_tool):
        dom = web_tool.go_to_url(http_server + "/")
        match = re.search(r"\[(\d+)\] <select", dom)
        assert match
        select_id = int(match.group(1))
        result = web_tool.execute_js(
            f"""(() => {{
                const el = window.__kiss_elements[{select_id}];
                el.value = 'green';
                el.dispatchEvent(new Event('change', {{bubbles: true}}));
                return el.value;
            }})()"""
        )
        assert "Result: green" in result

    def test_execute_js_extract_text(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.execute_js("document.body.innerText")
        assert "Result:" in result
        assert "Test Form Page" in result

    def test_execute_js_null_result(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.execute_js("void 0")
        assert "Page:" in result
        assert "Result:" not in result

    def test_execute_js_error(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool.execute_js("nonExistentFunction()")
        assert "Error" in result


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
        tool = WebUseTool()
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
        tool = WebUseTool()
        result = tool.close()
        assert "Browser closed" in result

    def test_get_tools_returns_all_methods(self):
        tool = WebUseTool()
        tools = tool.get_tools()
        names = {t.__name__ for t in tools}
        assert names == {
            "go_to_url",
            "click",
            "type_text",
            "press_key",
            "execute_js",
            "screenshot",
            "get_page_content",
        }

    def test_constructor_accepts_browser_types(self):
        for browser_type in ["chromium", "firefox", "webkit"]:
            tool = WebUseTool(browser_type=browser_type, headless=True)
            assert tool.browser_type == browser_type
            assert tool._page is None


class TestDOMTreeTruncation:
    def test_truncation(self, http_server, web_tool):
        web_tool.go_to_url(http_server + "/")
        result = web_tool._get_dom_tree(max_chars=50)
        assert "... [truncated]" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
