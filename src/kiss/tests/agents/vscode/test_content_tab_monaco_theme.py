# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and intentionally shadowed by
#   test parameters of the same name)
"""End-to-end tests: a content tab's Monaco editor follows the page theme.

The remote webapp has a light/dark toggle (``#theme-btn``, persisted in
``localStorage``).  The Monaco editor of a file content tab must use
Monaco's light ``vs`` theme under the page's light theme and ``vs-dark``
under the dark one: both when it is created and when the user toggles
the page theme while editors are open.

These tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` and the real ``cdn.jsdelivr.net`` Monaco
bundle.  When the CDN is unreachable there is no editor and the tests
skip.
"""

from __future__ import annotations

from kiss.tests.agents.vscode.test_content_tab_editing import (
    _fresh_file,
    _open_editor,
)
from kiss.tests.agents.vscode.test_content_tab_file_links import (
    _inject_file_link,
    _open_page,
    browser,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)

_MONACO = "#content-tab-area .monaco-editor"


def _editor_themes(page) -> list[str]:
    """Return the Monaco theme class ('vs' / 'vs-dark') of every editor."""
    return list(page.evaluate(
        """() => [...document.querySelectorAll('#content-tab-area .monaco-editor')]
             .map(el => el.classList.contains('vs-dark') ? 'vs-dark'
                  : el.classList.contains('vs') ? 'vs' : 'other')""",
    ))


def _editor_background(page) -> str:
    return str(page.evaluate(
        "() => getComputedStyle(document.querySelector("
        "'#content-tab-area .monaco-editor .monaco-editor-background'"
        ")).backgroundColor",
    ))


def _toggle_theme(page) -> None:
    page.evaluate("() => document.getElementById('theme-btn').click()")


class TestContentTabMonacoTheme:
    """Browser E2E: the editor's colours track the page's light/dark theme."""

    def test_toggle_recolours_open_editors(self, browser, harness) -> None:
        """Dark page -> vs-dark editor; toggling to light recolours every
        open editor to vs, and toggling back restores vs-dark."""
        context, page, _sent = _open_page(browser, harness)
        try:
            page.evaluate("() => localStorage.removeItem('kissRemoteTheme')")
            page.reload()
            page.wait_for_selector(".chat-tab", timeout=30000)
            assert not page.evaluate(
                "() => document.body.classList.contains('light-theme')",
            )
            _open_editor(page, str(_fresh_file(harness, "theme_a.py")), "lnk-t1")
            # The chat output (and so the link) is hidden behind the
            # first content tab: click the second link from script.
            _inject_file_link(page, str(_fresh_file(harness, "theme_b.py")), "lnk-t2")
            page.evaluate("() => document.getElementById('lnk-t2').click()")
            page.wait_for_function(
                f"() => document.querySelectorAll('{_MONACO}').length === 2",
                timeout=30000,
            )
            assert _editor_themes(page) == ["vs-dark", "vs-dark"]
            assert _editor_background(page) == "rgb(30, 30, 30)"
            _toggle_theme(page)
            assert _editor_themes(page) == ["vs", "vs"]
            assert _editor_background(page) == "rgb(255, 255, 254)"
            _toggle_theme(page)
            assert _editor_themes(page) == ["vs-dark", "vs-dark"]
        finally:
            context.close()

    def test_editor_opened_under_saved_light_theme_is_light(
        self, browser, harness,
    ) -> None:
        """A page reloaded with the light theme saved creates its editor
        light from the start."""
        context, page, _sent = _open_page(browser, harness)
        try:
            page.evaluate("() => localStorage.setItem('kissRemoteTheme', 'light')")
            page.reload()
            page.wait_for_selector(".chat-tab", timeout=30000)
            assert page.evaluate(
                "() => document.body.classList.contains('light-theme')",
            )
            _open_editor(page, str(_fresh_file(harness, "theme_c.py")), "lnk-t3")
            assert _editor_themes(page) == ["vs"]
            assert _editor_background(page) == "rgb(255, 255, 254)"
        finally:
            context.close()
