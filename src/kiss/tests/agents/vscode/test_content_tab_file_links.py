# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_content_tab_file_links and is intentionally
#   shadowed by test parameters of the same name)
"""End-to-end tests: clicking a file link in a remote-webapp chat tab.

When the user clicks a file link (``span[data-path]``) in a chat
webview served by the remote webapp, the frontend sends ``openFile``
over the WebSocket; :meth:`RemoteAccessServer._handle_open_file` reads
the file and replies with a ``fileContent`` event; ``media/main.js``
then opens the content in a SEPARATE content tab — code in a read-only
Monaco editor (with a ``pre``/highlight fallback when the CDN is
unreachable) and ``.html`` rendered as a webpage inside a sandboxed
iframe.

Content tabs must never interfere with the chat tabs of agents:

* opening/closing a content tab never sends ``closeTab``/``newTab``
  (or any other message) about a chat tab to the backend;
* the chat tab's input text and output DOM survive opening, switching
  and closing content tabs;
* closing a content tab leaves every chat tab intact.

The desktop remote page (the 1400px-wide default here) is the SPLIT
layout: the chat pane on the left stays on screen, and every content
tab is listed on the content pane's own row ``#content-tab-list`` on
the right, the shown one marked ``.active``.  The chat has no visible
tab of its own (a lone chat's group strip ``#tab-bar`` is hidden), so
readiness is waited for through the composer and ``window._testApi``.

These tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` over real ``wss://`` — no mocks.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)

# Every open tab (chats, sub-agents, content tabs), as the webview
# tracks them: chat tabs have no DOM entry unless their group strip is
# shown, so the DOM cannot be counted.
_OPEN_TAB_COUNT_JS = "window._testApi.openTabs().length"
# The content tab the content pane shows (split layout).
_ACTIVE_CONTENT_TAB = "#content-tab-list .chat-tab.content-tab.active"


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


def _open_page(browser, harness):
    """Open the webapp, wait for auth, and record every sent WS frame.

    Returns ``(context, page, sent_frames)`` where *sent_frames* is a
    live list of JSON-decoded frames the page sent over the WebSocket.
    """
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    sent_frames: list[dict] = []

    def _on_ws(ws) -> None:
        def _on_sent(payload) -> None:
            try:
                sent_frames.append(json.loads(payload))
            except Exception:
                pass

        ws.on("framesent", _on_sent)

    page.on("websocket", _on_ws)
    goto_retrying_network_change(page, harness.base_url + "/")
    _wait_ready(page)
    return context, page, sent_frames


def _wait_ready(page) -> None:
    """Wait until the webview has a chat on screen: the composer is
    visible and the test API reports an active (chat) tab.  A lone
    chat's group strip is hidden, so its ``.chat-tab`` entry cannot be
    waited for."""
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_function(
        "() => !!(window._testApi && window._testApi.getActiveTabId())",
        timeout=30000,
    )


def _inject_file_link(page, path: str, link_id: str) -> None:
    """Insert a file link span into the chat output, like linkified
    tool-call output would produce (``span.kiss-filelink[data-path]``).
    """
    page.evaluate(
        """([path, linkId]) => {
             const out = document.getElementById('output');
             const span = document.createElement('span');
             span.className = 'kiss-filelink';
             span.id = linkId;
             span.dataset.path = path;
             span.textContent = path;
             out.appendChild(span);
           }""",
        [path, link_id],
    )


class TestContentTabFileLinks:
    """Browser E2E: file links open formatted content tabs."""

    def test_code_link_opens_separate_tab_with_code(
        self, browser, harness,
    ) -> None:
        """Clicking a .py link opens a new content tab showing the code
        (Monaco, or the pre/code fallback when the CDN is unreachable)
        in the content pane while the chat, its composer and its draft
        stay on screen (split layout)."""
        context, page, sent = _open_page(browser, harness)
        try:
            page.fill("#task-input", "my precious draft")
            chat_tab_id = page.evaluate("() => window._testApi.getActiveTabId()")
            n_tabs_before = page.evaluate(_OPEN_TAB_COUNT_JS)
            _inject_file_link(
                page, str(harness.work_dir / "sample.py"), "lnk-code",
            )
            page.click("#lnk-code")
            page.wait_for_selector(_ACTIVE_CONTENT_TAB, timeout=30000)
            n_tabs_after = page.evaluate(_OPEN_TAB_COUNT_JS)
            assert n_tabs_after == n_tabs_before + 1
            label = page.locator(_ACTIVE_CONTENT_TAB + " .chat-tab-label")
            assert label.inner_text() == "sample.py"
            # The content tab is the content pane's, not the chat pane's:
            # the chat stays the active tab and nothing of it hides.
            assert page.evaluate("() => window._testApi.getActiveTabId()") == chat_tab_id
            assert page.locator("#tab-list .chat-tab.content-tab").count() == 0
            assert not page.evaluate(
                "() => document.body.classList.contains('content-tab-open')",
            )
            page.wait_for_selector(
                "#content-tab-area .content-tab-view", timeout=30000,
            )
            assert page.locator("#content-tab-area").is_visible()
            assert page.locator("#output").is_visible()
            assert page.locator("#input-area").is_visible()
            assert page.locator("#new-chat-btn").is_visible()
            assert page.locator("#more-btn").is_visible()
            assert page.locator("#task-input").is_visible()
            assert page.locator("#tricks-btn").is_visible()
            assert page.locator("#model-picker").is_visible()
            assert page.locator("#send-btn").is_visible()
            assert page.input_value("#task-input") == "my precious draft"
            page.wait_for_function(
                """() => {
                     const area = document.getElementById('content-tab-area');
                     if (!area) return false;
                     // Monaco renders spaces as U+00A0.
                     const text = area.innerText.replace(/\u00a0/g, ' ');
                     return text.includes('def greet');
                   }""",
                timeout=30000,
            )
            monaco_used = page.locator(
                "#content-tab-area .monaco-editor",
            ).count() > 0
            fallback_used = page.locator(
                "#content-tab-area .content-code-fallback",
            ).count() > 0
            assert monaco_used or fallback_used
            # A second file takes the content pane; clicking the first
            # tab's entry on the content row brings it back, and the
            # chat draft is untouched throughout.
            _inject_file_link(
                page, str(harness.work_dir / "page.html"), "lnk-code-2",
            )
            page.click("#lnk-code-2")
            page.wait_for_selector(
                "#content-tab-area .content-html-frame", timeout=30000,
            )
            assert page.locator(
                _ACTIVE_CONTENT_TAB + " .chat-tab-label",
            ).inner_text() == "page.html"
            assert page.locator("#content-tab-list .chat-tab.content-tab").count() == 2
            page.click("#content-tab-list .chat-tab.content-tab:has-text('sample.py')")
            page.wait_for_function(
                """() => document.querySelector(
                     '#content-tab-list .chat-tab.content-tab.active .chat-tab-label'
                   ).textContent === 'sample.py'""",
                timeout=10000,
            )
            assert page.locator("#content-tab-area .content-html-frame").is_hidden()
            assert page.evaluate("() => window._testApi.getActiveTabId()") == chat_tab_id
            assert page.input_value("#task-input") == "my precious draft"
            assert page.locator("#output").is_visible()
        finally:
            context.close()

    def test_html_link_renders_webpage_in_sandboxed_iframe(
        self, browser, harness,
    ) -> None:
        """Clicking an .html link renders the page in a sandboxed
        iframe inside a separate content tab."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(
                page, str(harness.work_dir / "page.html"), "lnk-html",
            )
            page.click("#lnk-html")
            page.wait_for_selector(
                "#content-tab-area .content-html-frame", timeout=30000,
            )
            iframe = page.locator("#content-tab-area .content-html-frame")
            assert iframe.get_attribute("sandbox") == "allow-scripts"
            frame = page.frame_locator("#content-tab-area .content-html-frame")
            assert (
                frame.locator("#marker").inner_text() == "KISS-HTML-MARKER"
            )
            label = page.locator(".chat-tab.content-tab .chat-tab-label")
            assert label.inner_text() == "page.html"
        finally:
            context.close()

    def test_md_link_renders_converted_html_in_sandboxed_iframe(
        self, browser, harness,
    ) -> None:
        """Clicking a .md link converts the markdown to HTML and renders
        the result in a sandboxed iframe inside a separate content tab —
        never the raw markdown source in a code view."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(
                page, str(harness.work_dir / "notes.md"), "lnk-md",
            )
            page.click("#lnk-md")
            page.wait_for_selector(
                "#content-tab-area .content-html-frame", timeout=30000,
            )
            iframe = page.locator("#content-tab-area .content-html-frame")
            assert iframe.get_attribute("sandbox") == "allow-scripts"
            frame = page.frame_locator("#content-tab-area .content-html-frame")
            assert frame.locator("h1").inner_text() == "KISS-MD-TITLE"
            assert frame.locator("strong").inner_text() == "bold"
            # The raw markdown syntax must not appear in the rendered page.
            assert "# KISS-MD-TITLE" not in frame.locator("body").inner_text()
            label = page.locator(".chat-tab.content-tab .chat-tab-label")
            assert label.inner_text() == "notes.md"
            # The Edit source surface stays dormant until its toggle is
            # clicked: the holder exists (hidden), but no editor was
            # mounted into it (see test_content_tab_preview_editing.py).
            holder = page.locator("#content-tab-area .content-monaco-holder")
            assert holder.count() == 1
            assert not holder.is_visible()
            assert page.locator("#content-tab-area .monaco-editor").count() == 0
        finally:
            context.close()

    def test_closing_content_tab_never_touches_backend_or_chat_tabs(
        self, browser, harness,
    ) -> None:
        """Opening and closing a content tab must not send closeTab (or
        any tab-lifecycle message) to the backend and must leave the
        chat tab fully intact."""
        context, page, sent = _open_page(browser, harness)
        try:
            page.fill("#task-input", "still here")
            chat_tab_id = page.evaluate("() => window._testApi.getActiveTabId()")
            _inject_file_link(
                page, str(harness.work_dir / "sample.py"), "lnk-close",
            )
            page.click("#lnk-close")
            page.wait_for_selector(_ACTIVE_CONTENT_TAB, timeout=30000)
            content_tab_id = page.locator(
                _ACTIVE_CONTENT_TAB,
            ).get_attribute("data-tab-id")
            sent.clear()
            page.click(_ACTIVE_CONTENT_TAB + " .chat-tab-close")
            page.wait_for_selector(
                ".chat-tab.content-tab", state="detached", timeout=30000,
            )
            page.wait_for_selector("#task-input", state="visible")
            assert page.input_value("#task-input") == "still here"
            # The content pane went away with its last tab; the chat is
            # untouched.
            assert not page.locator("#content-tab-area").is_visible()
            assert page.evaluate(
                "() => !document.body.classList.contains('content-pane-open')"
            )
            remaining = page.evaluate("() => window._testApi.openTabs()")
            assert [t["id"] for t in remaining if not t["isContentTab"]] == [chat_tab_id]
            assert not any(t["isContentTab"] for t in remaining)
            assert page.evaluate("() => window._testApi.getActiveTabId()") == chat_tab_id
            page.wait_for_timeout(500)
            for frame in sent:
                assert frame.get("type") != "closeTab"
                assert frame.get("tabId") != content_tab_id
        finally:
            context.close()

    def test_missing_file_shows_error_notification_no_tab(
        self, browser, harness,
    ) -> None:
        """A link to a nonexistent file shows an error toast and opens
        no content tab."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(
                page, str(harness.work_dir / "nope.py"), "lnk-missing",
            )
            page.click("#lnk-missing")
            page.wait_for_selector(
                ".kiss-notification-error", timeout=30000,
            )
            toast = page.locator(".kiss-notification-error")
            assert "File not found" in toast.inner_text()
            assert page.locator(".chat-tab.content-tab").count() == 0
            assert page.locator("#output").is_visible()
        finally:
            context.close()

    def test_relative_path_resolves_against_work_dir(
        self, browser, harness,
    ) -> None:
        """A relative file link resolves against the tab's work dir."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(page, "sample.py", "lnk-rel")
            page.click("#lnk-rel")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            page.wait_for_function(
                """() => {
                     const area = document.getElementById('content-tab-area');
                     if (!area) return false;
                     // Monaco renders spaces as U+00A0.
                     const text = area.innerText.replace(/\u00a0/g, ' ');
                     return text.includes('def greet');
                   }""",
                timeout=30000,
            )
        finally:
            context.close()

    def test_line_suffix_link_opens_content_tab(
        self, browser, harness,
    ) -> None:
        """A ``path:line`` link (as linkifyFilePaths produces) opens the
        file — the ``:line`` suffix is parsed off, not sent as path."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(
                page,
                str(harness.work_dir / "sample.py") + ":2",
                "lnk-line",
            )
            page.click("#lnk-line")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            label = page.locator(".chat-tab.content-tab .chat-tab-label")
            assert label.inner_text() == "sample.py"
        finally:
            context.close()

    # Waits until the content tab shows the code AROUND *marker* (a
    # `xNNN = NNN` line of longcode.py). Monaco virtualizes its DOM, so
    # innerText holds only the lines near the scroll position — seeing
    # the marker (and, when asked, NOT seeing `x1 =`, the first line)
    # proves the view actually jumped. The pre/code fallback renders
    # everything at once, so there the scroll offset is asserted instead.
    _JUMPED_TO_LINE_JS = """([marker, awayFromTop]) => {
         const area = document.getElementById('content-tab-area');
         if (!area) return false;
         const pre = area.querySelector('.content-code-fallback');
         if (pre) return !awayFromTop || pre.scrollTop > 0;
         const text = area.innerText.replace(/\\u00a0/g, ' ');
         if (!text.includes(marker)) return false;
         return !awayFromTop || !text.includes('x1 =');
       }"""

    def _write_long_file(self, harness) -> str:
        """Create a 300-line python file in the harness work dir."""
        path = harness.work_dir / "longcode.py"
        path.write_text(
            "".join(f"x{n} = {n}\n" for n in range(1, 301)),
        )
        return str(path)

    def test_line_suffix_link_jumps_to_line(self, browser, harness) -> None:
        """A ``path:250`` link on a 300-line file opens the content tab
        scrolled to line 250, matching the VS Code editor line jump."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = self._write_long_file(harness)
            _inject_file_link(page, path + ":250", "lnk-jump")
            page.click("#lnk-jump")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            page.wait_for_function(
                self._JUMPED_TO_LINE_JS, arg=["x250 =", True], timeout=30000,
            )
            open_file = [f for f in sent if f.get("type") == "openFile"]
            assert open_file and open_file[-1]["line"] == 250
            assert open_file[-1]["path"] == path
        finally:
            context.close()

    def test_out_of_range_line_clamps_to_last_line(
        self, browser, harness,
    ) -> None:
        """A ``:9999`` suffix on a 300-line file clamps to the end of
        the document instead of erroring, like VS Code does."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = self._write_long_file(harness)
            _inject_file_link(page, path + ":9999", "lnk-clamp")
            page.click("#lnk-clamp")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            page.wait_for_function(
                self._JUMPED_TO_LINE_JS, arg=["x300 = 300", True],
                timeout=30000,
            )
        finally:
            context.close()

    def test_fallback_pre_scrolls_to_line_when_cdn_unreachable(
        self, browser, harness,
    ) -> None:
        """With the Monaco CDN unreachable the pre/code fallback still
        honors the ``:NN`` suffix by scrolling to the line."""
        context = browser.new_context(ignore_https_errors=True)
        # Abort every CDN request so ensureMonaco() fails fast and the
        # pre/code fallback renders instead.
        context.route(
            "https://cdn.jsdelivr.net/**",
            lambda route: route.abort(),
        )
        page = context.new_page()
        try:
            goto_retrying_network_change(page, harness.base_url + "/")
            _wait_ready(page)
            path = self._write_long_file(harness)
            _inject_file_link(page, path + ":250", "lnk-fb")
            page.click("#lnk-fb")
            page.wait_for_selector(
                "#content-tab-area .content-code-fallback", timeout=30000,
            )
            page.wait_for_function(
                """() => {
                     const pre = document.querySelector(
                       '#content-tab-area .content-code-fallback');
                     return pre && pre.scrollTop > 0;
                   }""",
                timeout=30000,
            )
        finally:
            context.close()

    def test_hidden_open_jumps_when_tab_shown(self, browser, harness) -> None:
        """Hiding the content tab while the code is still loading must
        not lose the jump: the pending line is revealed when the tab
        becomes visible again."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = self._write_long_file(harness)
            _inject_file_link(page, path + ":250", "lnk-bg")
            page.click("#lnk-bg")
            page.wait_for_selector(_ACTIVE_CONTENT_TAB, timeout=30000)
            # Hide the content tab immediately by opening another file
            # over it in the content pane (the chat pane stays; a
            # content tab is never swapped out for the chat in the
            # split layout) — the editor then loads (or already loaded)
            # behind a display:none surface.
            _inject_file_link(page, str(harness.work_dir / "page.html"), "lnk-bg-2")
            page.click("#lnk-bg-2")
            page.wait_for_selector(
                "#content-tab-list .chat-tab.content-tab.active:has-text('page.html')",
                timeout=30000,
            )
            page.wait_for_timeout(2000)
            page.click(
                "#content-tab-list .chat-tab.content-tab:has-text('longcode.py')"
                " .chat-tab-label",
            )
            page.wait_for_function(
                self._JUMPED_TO_LINE_JS, arg=["x250 =", True], timeout=30000,
            )
        finally:
            context.close()

    def test_clicking_same_link_twice_reuses_tab(
        self, browser, harness,
    ) -> None:
        """Clicking the same file link twice opens exactly one tab."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(
                page, str(harness.work_dir / "sample.py"), "lnk-dup",
            )
            page.click("#lnk-dup")
            page.wait_for_selector(_ACTIVE_CONTENT_TAB, timeout=30000)
            # Another file takes the content pane first, so the second
            # click has to bring the existing tab back, not just leave
            # the shown one alone.
            _inject_file_link(page, str(harness.work_dir / "page.html"), "lnk-dup-2")
            page.click("#lnk-dup-2")
            page.wait_for_selector(
                "#content-tab-list .chat-tab.content-tab.active:has-text('page.html')",
                timeout=30000,
            )
            # The chat (and so the link) stays on screen in the split layout.
            page.wait_for_selector("#lnk-dup", state="visible")
            page.click("#lnk-dup")
            page.wait_for_selector(
                _ACTIVE_CONTENT_TAB + ":has-text('sample.py')", timeout=30000,
            )
            assert page.locator("#content-tab-list .chat-tab.content-tab").count() == 2
            assert page.locator(
                "#content-tab-list .chat-tab.content-tab:has-text('sample.py')",
            ).count() == 1
        finally:
            context.close()
