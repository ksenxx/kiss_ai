# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ``#menu-btn`` ("Advanced options") placement and behaviour.

After the move, ``#menu-btn`` sits at the leftmost position inside
``#model-picker`` — *before* ``#model-btn`` — and clicking it toggles
the sidebar (opening it on the first in-panel tab, which is the
Running tab).  The tab-bar ``#history-btn`` has been removed, so
``#menu-btn`` is the only entry-point.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
_SORCAR_TS = _REPO_ROOT / "src" / "SorcarTab.ts"
_WEB_SERVER = _REPO_ROOT / "web_server.py"
_MAIN_JS = _REPO_ROOT / "media" / "main.js"


class TestMenuBtnPlacement(unittest.TestCase):
    """``#menu-btn`` must precede ``#model-btn`` in both templates."""

    def test_sorcar_tab_template_menu_btn_before_model_btn(self) -> None:
        html = _SORCAR_TS.read_text()
        menu_pos = html.index('id="menu-btn"')
        model_pos = html.index('id="model-btn"')
        upload_pos = html.index('id="upload-btn"')
        self.assertLess(menu_pos, model_pos)
        self.assertLess(model_pos, upload_pos)

    def test_web_server_template_menu_btn_before_model_btn(self) -> None:
        html = _WEB_SERVER.read_text()
        menu_pos = html.index('id="menu-btn"')
        model_pos = html.index('id="model-btn"')
        upload_pos = html.index('id="upload-btn"')
        self.assertLess(menu_pos, model_pos)
        self.assertLess(model_pos, upload_pos)


_MAIN_JS_TEXT = _MAIN_JS.read_text()


class TestMenuBtnHandlerWired(unittest.TestCase):
    """``main.js`` must wire ``#menu-btn`` to open the History sidebar."""

    def test_menu_btn_referenced(self) -> None:
        self.assertIn("getElementById('menu-btn')", _MAIN_JS_TEXT)
        self.assertIn("menuBtn", _MAIN_JS_TEXT)

    def test_menu_btn_click_opens_history_sidebar(self) -> None:
        """``menuBtn`` click handler must open the (history) sidebar."""
        # The handler is attached via ``menuBtn.addEventListener('click', ...)``.
        match = re.search(
            r"menuBtn\.addEventListener\(\s*'click'\s*,\s*([A-Za-z_$][\w$]*)",
            _MAIN_JS_TEXT,
        )
        assert match is not None, "menuBtn must register a click handler in main.js"
        handler_name = match.group(1)
        # The handler body must add 'open' to #sidebar and request the
        # latest history from the backend.
        body_match = re.search(
            r"function\s+"
            + re.escape(handler_name)
            + r"\s*\([^)]*\)\s*\{([\s\S]*?)\n\s{4}\}",
            _MAIN_JS_TEXT,
        )
        assert body_match is not None, (
            f"Could not locate body of {handler_name}()"
        )
        body = body_match.group(1)
        self.assertIn("sidebar.classList.add('open')", body)
        self.assertIn("getHistory", body)



if __name__ == "__main__":
    unittest.main()
