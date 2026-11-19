# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Server-only test extracted from ``kiss.tests.webapp.test_codex_task_panel_style``.

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


def test_remote_page_injects_one_font_size_var() -> None:
    """The remote page injects ONE font-size variable:
    --theme-font-size, the root size every rem step of main.css's type
    scale derives from (the task panel's text included, one step up at
    --fs-lg; 13px, the size of Claude Code Desktop's sidebar and
    composer).  The server defines no second size, the former
    --theme-editor-font-size that nothing read, which could drift from
    it."""
    src = WEB_SERVER_PY.read_text(encoding="utf-8")
    assert "--theme-font-size: 13px" in src
    assert "--theme-editor-font-size" not in src
