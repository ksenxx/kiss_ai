# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Server-only test extracted from ``kiss.tests.agents.vscode.test_codex_task_panel_style``.

Moved here because its full dependency closure touches only
kiss.server (it asserts on the typography variables that
``web_server._build_html`` injects, reading the server source
itself), per the task: relocate core+sorcar+server-only test
methods to tests/server.
"""

from __future__ import annotations

from pathlib import Path

import kiss.server.web_server

# Placement-independent: resolve the server source from the package
# rather than from this test file's location.
WEB_SERVER_PY = Path(kiss.server.web_server.__file__)


def test_remote_page_font_size_vars_match_task_panel() -> None:
    """The task panel sizes itself with --vscode-editor-font-size and
    the chat panels with rem units derived from --vscode-font-size;
    the remote page must inject the SAME 16px for both so panel
    contents and the task panel share one size."""
    src = WEB_SERVER_PY.read_text(encoding="utf-8")
    assert "--vscode-font-size: 16px" in src
    assert "--vscode-editor-font-size: 16px" in src
