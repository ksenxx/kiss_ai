# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""The remote page defines the ``--theme-*`` design tokens.

``main.css`` resolves every colour and font through ``--theme-*``
variables; the page ``web_server._build_html`` serves must define each
of them, or the fonts and colours fall back to the browser's defaults.
The values are the Claude apps' fallback font stacks and Claude Code
Desktop's dark palette, the look the chat page is designed against.
"""


from __future__ import annotations

import pytest

# The system UI and monospace stacks the Claude apps fall back to, and
# Claude Code Desktop's dark palette: a near-black canvas, panels one
# shade up, raised inputs two shades up, white hairlines at low alpha,
# one accent blue and a near-white primary button.
THEME_VARS = {
    "--theme-font-size": "13px",
    "--theme-font-family": (
        'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, '
        "sans-serif"
    ),
    "--theme-editor-font-family": (
        "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
        '"Liberation Mono", "Courier New", monospace'
    ),
    "--theme-editor-background": "#0b0b0b",
    "--theme-editor-foreground": "#f0efec",
    "--theme-descriptionForeground": "#9c9a92",
    "--theme-panel-border": "#ffffff1a",
    "--theme-sideBar-background": "#151515",
    "--theme-textLink-foreground": "#4d9fff",
    "--theme-button-background": "#efefef",
    "--theme-focusBorder": "#4d9fff",
    "--theme-input-background": "#1f1f1f",
    "--theme-input-border": "#ffffff26",
    "--theme-terminal-ansiRed": "#ff5a5a",
    "--theme-terminal-ansiGreen": "#4cc38a",
    "--theme-terminal-ansiYellow": "#e3c07b",
    "--theme-terminal-ansiMagenta": "#b796ff",
    "--theme-terminal-ansiCyan": "#5ac8d8",
}


@pytest.mark.parametrize(("name", "value"), sorted(THEME_VARS.items()))
def test_remote_page_defines_theme_vars(name: str, value: str) -> None:
    """The page built by web_server.py defines every --theme-* token."""
    from kiss.server.web_server import _build_html

    html = _build_html()
    assert f"{name}: {value};" in html, (
        f"the remote page must define {name}: {value}"
    )
