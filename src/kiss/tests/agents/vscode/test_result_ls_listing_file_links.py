# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_content_tab_file_links and is intentionally
#   shadowed by test parameters of the same name)
"""Browser E2E: paths that highlight.js splits stay whole clickable links.

A result panel whose ``<pre><code>`` block holds a GNU ``ls -l`` listing
of absolute paths is tokenized by highlight.js (the ``/home/...``
listing auto-detects as Swift, whose grammar reads ``/home/`` and
``/kiss/`` as regexp literals).  Every path then spans several text
nodes and ``linkifyFilePaths()`` used to see only fragments, so the
reader got inert paths.  ``mergeSplitPathText()`` in ``media/main.js``
folds each split path back into one text node before linkifying.

This drives REAL Chromium (Playwright) against a REAL
:class:`RemoteAccessServer` over ``wss://``: the result frame is
delivered as a window ``message`` event exactly like the remote shim
does, the ``checkPaths`` -> ``pathsExist`` round-trip goes through the
real server against real files, and a click on the merged link opens a
real content tab.  The listing pins ``class="language-swift"`` so the
split does not depend on auto-detection heuristics of the hljs build.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_content_tab_file_links import _open_page
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


def _ls_listing(harness) -> str:
    """A GNU ``ls -ld "$PWD"/*`` style listing of the harness files."""
    wd = harness.work_dir
    return (
        f"-rw-rw-r--  1 ksen ksen  71830 Sep 15 07:29 {wd / 'sample.py'}\n"
        f"-rw-rw-r--  1 ksen ksen   2639 Sep  1 18:25 {wd / 'notes.md'}\n"
        f"-rw-rw-r--  1 ksen ksen  49260 Sep 15 23:23 {wd / 'page.html'}\n"
        f"-rw-rw-r--  1 ksen ksen 662614 Sep 15 03:58 {wd / 'missing.lock'}\n"
    )


def _deliver_result(page, summary_html: str) -> None:
    """Deliver a ``result`` frame the way the remote shim does."""
    page.evaluate(
        """(summary) => {
             window.dispatchEvent(new MessageEvent('message', {data: {
               type: 'result', summary, success: true,
               total_tokens: 1, cost: '$0.01',
             }}));
           }""",
        summary_html,
    )


class TestResultLsListingFileLinks:
    """Browser E2E: split-by-hljs listing paths are whole clickable links."""

    def test_split_paths_become_whole_links_and_open(
        self, browser, harness,
    ) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            listing = _ls_listing(harness)
            _deliver_result(
                page,
                '<h3>Output of <code>ls -ld "$PWD"/*</code></h3>'
                '<pre><code class="language-swift">'
                + listing
                + "</code></pre>",
            )
            sample = str(harness.work_dir / "sample.py")
            notes = str(harness.work_dir / "notes.md")
            html_file = str(harness.work_dir / "page.html")
            missing = str(harness.work_dir / "missing.lock")

            page.wait_for_selector(
                f'#output .rc-body pre code [data-path="{sample}"]',
                timeout=30000,
            )
            code = page.locator("#output .rc-body pre code")
            # Highlighting is kept: the block is a Swift-tokenized hljs block.
            assert "hljs" in (code.get_attribute("class") or "")
            assert code.locator('span[class^="hljs-"]').count() > 0
            # The block text is unchanged by the merge.
            assert code.evaluate("el => el.textContent") == listing

            link_paths = page.evaluate(
                """() => Array.from(
                     document.querySelectorAll('#output .rc-body [data-path]'),
                   ).map(el => [el.dataset.path, el.textContent])"""
            )
            assert sorted(link_paths) == sorted(
                [[sample, sample], [notes, notes], [html_file, html_file]],
            ), link_paths
            # No path-fragment links such as "/tmp/" and no leftovers.
            assert page.locator("#output [data-path-candidate]").count() == 0
            missing_state = page.evaluate(
                """(p) => {
                     const el = document.querySelector(
                       '#output [data-path-missing="' + p + '"]');
                     return el ? el.textContent : null;
                   }""",
                missing,
            )
            assert missing_state == missing

            # The whole-path link opens the real file in a content tab.
            real_tabs = page.locator(
                ".chat-tab:not(.chat-tab-add):not(.chat-tab-settings)",
            )
            n_before = real_tabs.count()
            page.click(f'#output .rc-body pre code [data-path="{notes}"]')
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            assert real_tabs.count() == n_before + 1
            label = page.locator(".chat-tab.content-tab .chat-tab-label")
            assert label.inner_text() == "notes.md"
            opens = [f for f in sent if f.get("type") == "openFile"]
            assert [f["path"] for f in opens] == [notes], json.dumps(opens)
        finally:
            context.close()
