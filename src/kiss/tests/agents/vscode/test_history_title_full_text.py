# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Tests that stay behind from ``kiss.tests.agents.vscode.test_history_title_full_text``
(now ``kiss.tests.server.test_history_title_full_text``): they assert on assets outside
kiss.core/kiss.agents.sorcar/kiss.server (e.g. the real
chat.html/main.js media content), so they keep their original
location while the server-only majority of the file moved to
tests/server.
"""


from __future__ import annotations

import unittest
from pathlib import Path


class TestRunningItemLineClampCSS(unittest.TestCase):
    """The CSS that controls how many lines history rows can show
    pins ``-webkit-line-clamp`` to 3 on ``.running-item`` rows, so a
    collapsed history task panel shows at most three lines of task text.
    History items render with ``className = 'sidebar-item running-item'``
    in ``renderHistory`` so this selector applies to them too.
    """

    def test_running_item_line_clamp_is_three(self) -> None:
        css_path = (
            Path(__file__).resolve().parents[3]
            / "agents"
            / "vscode"
            / "media"
            / "main.css"
        )
        css = css_path.read_text(encoding="utf-8")
        start = css.index(".running-item > .sidebar-item-text")
        block = css[start : start + 400]
        self.assertIn("-webkit-line-clamp: 3;", block)
        self.assertIn("line-clamp: 3;", block)
