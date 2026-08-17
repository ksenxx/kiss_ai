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
* the chat tab's input text, output DOM, and tab-bar entry survive
  switching to and from content tabs;
* closing a content tab leaves every chat tab intact.

These tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` over real ``wss://`` — no mocks.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)


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
    page.goto(harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_selector(".chat-tab", timeout=30000)
    return context, page, sent_frames


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
        while the chat tab and its input text stay intact."""
        context, page, sent = _open_page(browser, harness)
        try:
            page.fill("#task-input", "my precious draft")
            real_tabs = page.locator(
                ".chat-tab:not(.chat-tab-add):not(.chat-tab-settings)",
            )
            n_tabs_before = real_tabs.count()
            _inject_file_link(
                page, str(harness.work_dir / "sample.py"), "lnk-code",
            )
            page.click("#lnk-code")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            n_tabs_after = real_tabs.count()
            assert n_tabs_after == n_tabs_before + 1
            label = page.locator(".chat-tab.content-tab .chat-tab-label")
            assert label.inner_text() == "sample.py"
            assert "active" in (
                page.locator(".chat-tab.content-tab").get_attribute("class")
            )
            page.wait_for_selector(
                "#content-tab-area .content-tab-view", timeout=30000,
            )
            assert page.locator("#output").is_hidden()
            assert page.locator("#input-area").is_hidden()
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
            page.click(".chat-tab:not(.content-tab) .chat-tab-label")
            page.wait_for_selector("#task-input", state="visible")
            assert page.input_value("#task-input") == "my precious draft"
            assert page.locator("#output").is_visible()
            assert page.locator("#content-tab-area").is_hidden()
            page.click(".chat-tab.content-tab .chat-tab-label")
            page.wait_for_selector("#content-tab-area", state="visible")
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
            assert page.locator(
                "#content-tab-area .content-monaco-holder",
            ).count() == 0
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
            chat_tab_id = page.locator(
                ".chat-tab:not(.chat-tab-add):not(.chat-tab-settings)",
            ).first.get_attribute("data-tab-id")
            _inject_file_link(
                page, str(harness.work_dir / "sample.py"), "lnk-close",
            )
            page.click("#lnk-close")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            content_tab_id = page.locator(
                ".chat-tab.content-tab",
            ).get_attribute("data-tab-id")
            sent.clear()
            page.click(".chat-tab.content-tab .chat-tab-close")
            page.wait_for_selector(
                ".chat-tab.content-tab", state="detached", timeout=30000,
            )
            page.wait_for_selector("#task-input", state="visible")
            assert page.input_value("#task-input") == "still here"
            remaining = page.locator(
                ".chat-tab:not(.chat-tab-add):not(.chat-tab-settings)",
            )
            assert remaining.count() >= 1
            assert remaining.first.get_attribute("data-tab-id") == chat_tab_id
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
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            page.click(".chat-tab:not(.content-tab) .chat-tab-label")
            page.wait_for_selector("#lnk-dup", state="visible")
            page.click("#lnk-dup")
            page.wait_for_selector(
                ".chat-tab.content-tab.active", timeout=30000,
            )
            assert page.locator(".chat-tab.content-tab").count() == 1
        finally:
            context.close()
