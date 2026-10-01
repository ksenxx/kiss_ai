# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and intentionally shadowed by
#   test parameters of the same name)
"""End-to-end tests: a content tab's Monaco editor uses the page palette.

The remote webapp has a light/dark toggle (``#theme-btn``, persisted in
``localStorage``) and colours itself with VS Code's Dark/Light Modern
palette (the ``--vscode-*`` variables web_server.py injects).  The
Monaco editor of a file content tab must use the custom ``kiss-dark`` /
``kiss-light`` theme built from that palette, so its background,
selection and line-number colours are exactly the page's: both when the
editor is created and when the user toggles the page theme while
editors are open.

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
from kiss.tests.conftest import reload_retrying_network_change
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)

_MONACO = "#content-tab-area .monaco-editor"

# The editor of the visible content tab (hidden tabs' editors have no
# layout box, so their offsetParent is null).
_VISIBLE_EDITOR_JS = (
    "monaco.editor.getEditors().find(e => e.getDomNode().offsetParent)"
)

# Resolved rgb() colour of a page palette variable, read from a probe
# element under <body> (outside every .monaco-editor).
_PALETTE_JS = """(name) => {
  const probe = document.createElement('div');
  probe.style.backgroundColor = 'var(' + name + ')';
  document.body.appendChild(probe);
  const colour = getComputedStyle(probe).backgroundColor;
  probe.remove();
  return colour;
}"""


def _palette(page, name: str) -> str:
    return str(page.evaluate(_PALETTE_JS, name))


def _editor_themes(page) -> list[str]:
    """Return the theme type class ('vs' / 'vs-dark') of every editor.

    Monaco marks an editor only with its theme's base type, so the custom
    themes are told apart by their colours (see
    :func:`_assert_editor_matches_palette`)."""
    return list(page.evaluate(
        """() => [...document.querySelectorAll('#content-tab-area .monaco-editor')]
             .map(el => el.classList.contains('vs-dark') ? 'vs-dark'
                  : el.classList.contains('vs') ? 'vs' : 'other')""",
    ))


def _visible_editor_style(page, selector: str, prop: str) -> str:
    """Computed *prop* of the first *selector* match in the visible editor."""
    return str(page.evaluate(
        f"""([selector, prop]) => getComputedStyle(
               {_VISIBLE_EDITOR_JS}.getDomNode().querySelector(selector)
             )[prop]""",
        [selector, prop],
    ))


def _wait_visible_editor_style(
    page, selector: str, prop: str, expected: str,
) -> None:
    """Wait until the visible editor renders *selector* with *prop* ==
    *expected* (Monaco paints selections on the next animation frame)."""
    page.wait_for_function(
        f"""([selector, prop, expected]) => {{
               const el = {_VISIBLE_EDITOR_JS}.getDomNode()
                 .querySelector(selector);
               return !!el && getComputedStyle(el)[prop] === expected;
             }}""",
        arg=[selector, prop, expected],
        timeout=10000,
    )


def _assert_editor_matches_palette(page, background: str, selection: str) -> None:
    """The visible editor's colours are the page palette's: background,
    focused and unfocused selection, and line numbers.  *background* and
    *selection* pin the Modern values so the palette itself is checked."""
    assert _palette(page, "--vscode-editor-background") == background
    assert _palette(page, "--vscode-editor-selectionBackground") == selection
    assert _visible_editor_style(
        page, ".monaco-editor-background", "backgroundColor",
    ) == background
    assert _visible_editor_style(
        page, ".margin", "backgroundColor",
    ) == background
    page.evaluate(
        f"""() => {{
               const editor = {_VISIBLE_EDITOR_JS};
               editor.setSelection(new monaco.Range(1, 1, 1, 4));
               editor.focus();
             }}""",
    )
    _wait_visible_editor_style(page, ".selected-text", "backgroundColor", selection)
    _wait_visible_editor_style(
        page, ".active-line-number", "color",
        _palette(page, "--vscode-editorLineNumber-activeForeground"),
    )
    assert _visible_editor_style(
        page, ".line-numbers:not(.active-line-number)", "color",
    ) == _palette(page, "--vscode-editorLineNumber-foreground")
    page.evaluate("() => document.activeElement.blur()")
    _wait_visible_editor_style(
        page, ".selected-text", "backgroundColor",
        _palette(page, "--vscode-editor-inactiveSelectionBackground"),
    )


def _toggle_theme(page) -> None:
    page.evaluate("() => document.getElementById('theme-btn').click()")


class TestContentTabMonacoTheme:
    """Browser E2E: the editor's colours are the page's Modern palette."""

    def test_toggle_recolours_open_editors(self, browser, harness) -> None:
        """Light page (the default) -> kiss-light editors in Light Modern
        colours; toggling to dark recolours every open editor to kiss-dark
        in Dark Modern colours, and toggling back restores kiss-light."""
        context, page, _sent = _open_page(browser, harness)
        try:
            page.evaluate("() => localStorage.removeItem('kissRemoteTheme')")
            reload_retrying_network_change(page)
            page.wait_for_selector(".chat-tab", timeout=30000)
            assert page.evaluate(
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
            assert _editor_themes(page) == ["vs", "vs"]
            _assert_editor_matches_palette(
                page, "rgb(255, 255, 255)", "rgb(173, 214, 255)",
            )
            _toggle_theme(page)
            assert _editor_themes(page) == ["vs-dark", "vs-dark"]
            _assert_editor_matches_palette(
                page, "rgb(31, 31, 31)", "rgb(38, 79, 120)",
            )
            _toggle_theme(page)
            assert _editor_themes(page) == ["vs", "vs"]
            _assert_editor_matches_palette(
                page, "rgb(255, 255, 255)", "rgb(173, 214, 255)",
            )
        finally:
            context.close()

    def test_editor_opened_under_saved_dark_theme_is_dark(
        self, browser, harness,
    ) -> None:
        """A page reloaded with the dark theme saved creates its editor
        in the Dark Modern colours from the start."""
        context, page, _sent = _open_page(browser, harness)
        try:
            page.evaluate("() => localStorage.setItem('kissRemoteTheme', 'dark')")
            reload_retrying_network_change(page)
            page.wait_for_selector(".chat-tab", timeout=30000)
            assert not page.evaluate(
                "() => document.body.classList.contains('light-theme')",
            )
            _open_editor(page, str(_fresh_file(harness, "theme_c.py")), "lnk-t3")
            assert _editor_themes(page) == ["vs-dark"]
            _assert_editor_matches_palette(
                page, "rgb(31, 31, 31)", "rgb(38, 79, 120)",
            )
        finally:
            context.close()
