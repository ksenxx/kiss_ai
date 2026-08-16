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

VSCODE_VARS = {
    "--vscode-font-size": "16px",
    "--vscode-font-family": (
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
        "'Helvetica Neue', Arial, sans-serif"
    ),
    "--vscode-editor-font-size": "16px",
    "--vscode-editor-font-family": (
        "Menlo, Monaco, 'Courier New', monospace"
    ),
    "--vscode-editor-background": "#1e1e1e",
    "--vscode-editor-foreground": "#cccccc",
    "--vscode-descriptionForeground": "#8b8b8b",
    "--vscode-panel-border": "#80808059",
    "--vscode-sideBar-background": "#252526",
    "--vscode-textLink-foreground": "#3794ff",
    "--vscode-terminal-ansiRed": "#f44747",
    "--vscode-terminal-ansiGreen": "#6a9955",
    "--vscode-terminal-ansiYellow": "#d7ba7d",
    "--vscode-terminal-ansiMagenta": "#c586c0",
    "--vscode-terminal-ansiCyan": "#4ec9b0",
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
