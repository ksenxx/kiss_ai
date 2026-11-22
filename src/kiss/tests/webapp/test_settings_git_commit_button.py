# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Tests that stay behind from ``kiss.tests.webapp.test_settings_git_commit_button``
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
    Path(__file__).resolve().parents[3] / "kiss" / "webapp" / "media"
)




class TestGitCommitButtonWiring(unittest.TestCase):
    """The page carries the button and posts the command."""

    def test_button_markup_on_the_task_info_strip(self) -> None:
        html = (_MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
        i_strip = html.find('id="meta-footer"')
        i_btn = html.find('id="autocommit-btn"')
        i_resizer = html.find('id="meta-resizer"')
        self.assertGreater(i_strip, -1)
        self.assertTrue(
            i_strip < i_btn < i_resizer,
            "button must be on the strip under the task-info panel",
        )
        tag = html[html.rfind("<button", 0, i_btn) : html.index(">", i_btn) + 1]
        self.assertIn('data-tooltip="Git commit"', tag)
        self.assertIn('aria-label="Git commit"', tag)
        self.assertIn('class="sidebar-tool-btn"', tag)

    def test_main_js_posts_autocommit_action(self) -> None:
        # Since 81905ace3 (editor-title git-commit button) the click
        # handler delegates to the shared triggerManualGitCommit()
        # helper instead of posting autocommitAction inline.
        js = (_MEDIA_DIR / "main.js").read_text(encoding="utf-8")
        self.assertIn("getElementById('autocommit-btn')", js)
        self.assertIn(
            "bindStripButton('autocommit-btn', triggerManualGitCommit);",
            js,
            "button click handler must exist",
        )
        helper = re.search(
            r"function triggerManualGitCommit\(\) \{(.*?)\n  \}",
            js,
            re.DOTALL,
        )
        self.assertIsNotNone(helper, "triggerManualGitCommit helper must exist")
        assert helper is not None
        self.assertIn("api.autocommitAction(", helper.group(1))
        self.assertIn("autocommitTargetTabId()", helper.group(1))
        self.assertIn("setAutocommitInFlight(true)", helper.group(1))
        self.assertIn("closeSettingsPanel()", helper.group(1))
        # The commit goes to the machine and folder the chat header's
        # drop-down shows for the tab.
        self.assertIn("tabBrowseWorkDir(commitTabId)", helper.group(1))
        self.assertIn("tabMachine(commitTabId)", helper.group(1))

    def test_gitcommit_css_present(self) -> None:
        main_css = (_MEDIA_DIR / "main.css").read_text(encoding="utf-8")
        remote_css = (
            _MEDIA_DIR / "remote-codex.css"
        ).read_text(encoding="utf-8")
        self.assertIn(".more-menu-item", main_css)
        self.assertIn(".more-menu-item", remote_css)


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
