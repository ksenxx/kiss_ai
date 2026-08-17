# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Tests that stay behind from ``kiss.tests.agents.vscode.test_settings_git_commit_button``
(now ``kiss.tests.server.test_settings_git_commit_button``): they assert on assets outside
kiss.core/kiss.agents.sorcar/kiss.server (e.g. the real
chat.html/main.js media content), so they keep their original
location while the server-only majority of the file moved to
tests/server.
"""


from __future__ import annotations

import re
import unittest
from pathlib import Path

_MEDIA_DIR = (
    Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode" / "media"
)




class TestGitCommitButtonWiring(unittest.TestCase):
    """The webview carries the button and posts the command."""

    def test_button_markup_in_settings_row(self) -> None:
        html = (_MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
        i_row = html.find('class="config-update-row"')
        i_btn = html.find('id="autocommit-btn"')
        self.assertGreater(i_row, -1)
        self.assertGreater(i_btn, -1)
        self.assertLess(i_row, i_btn, "button must be in the settings row")
        tag = html[html.rfind("<button", 0, i_btn) : html.index(">", i_btn) + 1]
        self.assertIn('data-tooltip="git commit"', tag)
        self.assertIn("config-gitcommit-btn", tag)
        btn_block = html[i_btn : html.index("</button>", i_btn)]
        self.assertIn("<span>Git Commit</span>", btn_block)

    def test_main_js_posts_autocommit_action(self) -> None:
        js = (_MEDIA_DIR / "main.js").read_text(encoding="utf-8")
        self.assertIn("getElementById('autocommit-btn')", js)
        wiring = re.search(
            r"autocommitBtn\.addEventListener\('click',(.*?)\}\);\s*\}",
            js,
            re.DOTALL,
        )
        self.assertIsNotNone(wiring, "button click handler must exist")
        assert wiring is not None
        self.assertIn("api.autocommitAction(", wiring.group(1))
        self.assertIn("autocommitTargetTabId()", wiring.group(1))
        self.assertIn("setAutocommitInFlight(true)", wiring.group(1))
        self.assertIn("closeSettingsPanel()", wiring.group(1))
        self.assertIn("workDirForTab(commitTabId)", wiring.group(1))

    def test_gitcommit_css_present(self) -> None:
        main_css = (_MEDIA_DIR / "main.css").read_text(encoding="utf-8")
        remote_css = (
            _MEDIA_DIR / "remote-codex.css"
        ).read_text(encoding="utf-8")
        self.assertIn(".config-gitcommit-btn", main_css)
        self.assertIn(".config-gitcommit-btn", remote_css)


class TestGitCommitCommandCatalog(unittest.TestCase):
    """``autocommitAction`` is present in every command catalog."""

    def test_in_browser_catalog(self) -> None:
        api_js = (_MEDIA_DIR / "api.js").read_text(encoding="utf-8")
        match = re.search(
            r"const SORCAR_API_COMMANDS = \[(.*?)\];", api_js, re.DOTALL,
        )
        self.assertIsNotNone(match)
        assert match is not None
        names = re.findall(r"'([^']+)'", match.group(1))
        self.assertIn("autocommitAction", names)
