# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""The remote page defines the ``--theme-*`` design tokens.

``main.css`` resolves every colour and font through ``--theme-*``
variables; the page ``web_server._build_html`` serves must define each
of them, or the fonts and colours fall back to the browser's defaults.
The values are VS Code's default fonts and its "Dark Modern" theme, the
palette the chat page was designed against.
"""


from __future__ import annotations

import pytest

# VS Code's default fonts (all three platforms' stacks in turn) and its
# "Dark Modern" theme (dark_modern.json plus the colour registry
# defaults it inherits, e.g. terminal.ansi*).
THEME_VARS = {
    "--theme-font-size": "13px",
    "--theme-font-family": (
        '-apple-system, BlinkMacSystemFont, "Segoe WPC", "Segoe UI", '
        'system-ui, "Ubuntu", "Droid Sans", sans-serif'
    ),
    "--theme-editor-font-size": "13px",
    "--theme-editor-font-family": (
        'Menlo, Monaco, Consolas, "Droid Sans Mono", "Courier New", monospace'
    ),
    "--theme-editor-background": "#1f1f1f",
    "--theme-editor-foreground": "#cccccc",
    "--theme-descriptionForeground": "#9d9d9d",
    "--theme-panel-border": "#2b2b2b",
    "--theme-sideBar-background": "#181818",
    "--theme-textLink-foreground": "#4daafc",
    "--theme-button-background": "#0078d4",
    "--theme-focusBorder": "#0078d4",
    "--theme-input-background": "#313131",
    "--theme-input-border": "#3c3c3c",
    "--theme-terminal-ansiRed": "#cd3131",
    "--theme-terminal-ansiGreen": "#0dbc79",
    "--theme-terminal-ansiYellow": "#e5e510",
    "--theme-terminal-ansiMagenta": "#bc3fbc",
    "--theme-terminal-ansiCyan": "#11a8cd",
}


@pytest.mark.parametrize(("name", "value"), sorted(THEME_VARS.items()))
def test_remote_page_defines_theme_vars(name: str, value: str) -> None:
    """The page built by web_server.py defines every --theme-* token."""
    from kiss.server.web_server import _build_html

    html = _build_html()
    assert f"{name}: {value};" in html, (
        f"the remote page must define {name}: {value}"
    )
