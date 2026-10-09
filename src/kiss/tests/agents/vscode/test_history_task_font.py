# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""History typography through real SQLite, WSS, rendering and Chromium layout."""

from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

from kiss.agents.sorcar import persistence
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.test_content_tab_file_links import _ServerHarness

_FONT = """el => {
    const style = getComputedStyle(el);
    return [style.fontFamily, style.fontSize, style.fontWeight, style.fontStyle];
}"""


@pytest.mark.parametrize("surface", ["remote", "extension"])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_history_task_text_matches_chat_header(
    surface: str, theme: str, tmp_path: Path,
) -> None:
    """Task text matches its header in grouped/flat views on both CSS surfaces.

    The extension styling uses the shared production page without the remote
    stylesheet/body class, as in the existing cross-surface layout tests.
    History is populated through real persistence, not injected UI messages.
    """
    harness = _ServerHarness()
    try:
        title = "History task typography matches its chat header"
        task_id, _ = persistence._add_task(
            title, "history-font-chat", {"work_dir": str(harness.work_dir)},
        )
        persistence._save_task_result("done", task_id=task_id)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(
                    ignore_https_errors=True, viewport={"width": 1200, "height": 900},
                )
                goto_retrying_network_change(page, harness.base_url + "/")
                page.wait_for_selector("#task-input", state="visible")
                page.wait_for_function(
                    "window._testApi && window._testApi.getActiveTabId()"
                )
                page.evaluate(
                    """([surface, theme]) => {
                        document.body.classList.toggle('light-theme', theme === 'light');
                        if (surface === 'extension') {
                            document.body.classList.remove('remote-chat');
                            document.body.classList.add('history-panel-mode', 'editor-tab-mode');
                            document.body.classList.add('vscode-' + theme);
                            document.querySelector('link[href*="remote-codex.css"]').remove();
                        }
                    }""",
                    [surface, theme],
                )
                header = page.locator(".history-chat-title").filter(has_text=title)
                expect(header).to_be_visible()
                header_font = header.evaluate(_FONT)
                assert header_font[2] == "400"
                # The chat's "Show details" chevron unfolds its task rows
                # (a click on the header itself opens the chat).
                page.locator(".history-chat-header .history-chat-collapse").click()
                text = page.locator("#history-list .running-item > .sidebar-item-text")
                expect(text).to_be_visible()
                expect(text).to_have_text(title)
                assert text.evaluate(_FONT) == header_font

                # Expanding task metadata and hovering must not change its font.
                page.locator("#history-list .running-item .sidebar-item-collapse").click()
                text.hover()
                assert text.evaluate(_FONT) == header_font
                page.screenshot(path=str(tmp_path / f"history-font-{surface}-{theme}.png"))

                # The same task retains its typography in the legacy flat list.
                page.click("#history-view-toggle")
                expect(page.locator("#history-list")).to_have_class("legacy-view")
                expect(text).to_be_visible()
                assert text.evaluate(_FONT) == header_font
            finally:
                browser.close()
    finally:
        harness.stop()
