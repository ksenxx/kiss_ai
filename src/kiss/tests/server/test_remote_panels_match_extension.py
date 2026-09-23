# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.agents.vscode.test_remote_panels_match_extension``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import pytest

# VS Code's default fonts (src/vs/base/browser/fonts.ts and
# src/vs/editor/common/config/fontInfo.ts, all three platforms' stacks
# in turn) and the "Dark Modern" theme
# (extensions/theme-defaults/themes/dark_modern.json plus the colour
# registry defaults it inherits, e.g. terminal.ansi*).
VSCODE_VARS = {
    "--vscode-font-size": "14px",
    "--vscode-font-family": (
        '-apple-system, BlinkMacSystemFont, "Segoe WPC", "Segoe UI", '
        'system-ui, "Ubuntu", "Droid Sans", sans-serif'
    ),
    "--vscode-editor-font-size": "14px",
    "--vscode-editor-font-family": (
        'Menlo, Monaco, Consolas, "Droid Sans Mono", "Courier New", monospace'
    ),
    "--vscode-editor-background": "#1f1f1f",
    "--vscode-editor-foreground": "#cccccc",
    "--vscode-descriptionForeground": "#9d9d9d",
    "--vscode-panel-border": "#2b2b2b",
    "--vscode-sideBar-background": "#181818",
    "--vscode-textLink-foreground": "#4daafc",
    "--vscode-button-background": "#0078d4",
    "--vscode-focusBorder": "#0078d4",
    "--vscode-input-background": "#313131",
    "--vscode-input-border": "#3c3c3c",
    "--vscode-terminal-ansiRed": "#cd3131",
    "--vscode-terminal-ansiGreen": "#0dbc79",
    "--vscode-terminal-ansiYellow": "#e5e510",
    "--vscode-terminal-ansiMagenta": "#bc3fbc",
    "--vscode-terminal-ansiCyan": "#11a8cd",
}


@pytest.mark.parametrize(("name", "value"), sorted(VSCODE_VARS.items()))
def test_remote_page_defines_vscode_typography_vars(
    name: str, value: str
) -> None:
    """The remote page built by web_server.py must inject the same
    --vscode-* variables that the VS Code webview host provides (and
    that this test's extension reference page uses), so fonts resolve
    identically."""
    from kiss.server.web_server import _build_html

    html = _build_html()
    assert f"{name}: {value};" in html, (
        f"the remote page must define {name}: {value}"
    )
