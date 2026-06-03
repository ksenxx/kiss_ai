# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: for any regular tab that has run a task (whether
freshly created or loaded from history), the tab title shows a green
filled circle (●) on success and a red filled circle on failure
instead of the previous green ✓ / red ✗ glyphs.

Contract
--------
``renderTabBar`` in ``media/main.js`` constructs a ``<span>`` with the
``chat-tab-status`` class plus ``chat-tab-ok`` (green) or
``chat-tab-fail`` (red) depending on ``tab.lastTaskFailed``.  The
text content of that span must be the U+25CF BLACK CIRCLE glyph
(``●``) — the green/red colour is supplied entirely by the CSS class.
The previous ✓ (U+2713) / ✗ (U+2717) glyphs must not appear in this
branch.
"""

from __future__ import annotations

import re
from pathlib import Path

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss" / "agents" / "vscode" / "media" / "main.js"
)
_MAIN_CSS = (
    Path(__file__).resolve().parents[4]
    / "kiss" / "agents" / "vscode" / "media" / "main.css"
)


def _strip_js_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def _regular_tab_branch_source() -> str:
    """Return the ``else`` branch of ``renderTabBar`` (the non-subagent
    branch) with comments stripped, covering both the running-spinner
    and the finished-status-icon paths.
    """
    src = _MAIN_JS.read_text(encoding="utf-8")
    fn_idx = src.index("function renderTabBar(")
    branch_idx = src.index("if (tab.isSubagentTab) {", fn_idx)
    else_idx = src.index("} else {", branch_idx)
    # The else branch ends at the next ``}`` that closes the if/else
    # — locate it by scanning for the matching brace.
    depth = 1
    i = else_idx + len("} else {")
    while i < len(src) and depth > 0:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return _strip_js_comments(src[else_idx:i])


class TestRegularTabStatusUsesColouredCircle:
    """Static checks on ``media/main.js`` ``renderTabBar`` regular-tab
    (non-subagent) branch.
    """

    def test_status_icon_textcontent_is_filled_circle(self) -> None:
        """The status span must render U+25CF (●), not U+2713 / U+2717."""
        body = _regular_tab_branch_source()
        assert "\\u25CF" in body or "\\u25cf" in body or "●" in body, body
        # Old ✓ / ✗ glyphs (including their unicode escape forms) must
        # not be present in executable code anymore.
        assert "✓" not in body, body
        assert "✗" not in body, body
        assert "\\u2713" not in body, body
        assert "\\u2717" not in body, body

    def test_classes_still_drive_green_red_colour(self) -> None:
        """``chat-tab-ok`` / ``chat-tab-fail`` classes (which colour the
        circle green/red via CSS) must still be assigned to the icon.
        """
        body = _regular_tab_branch_source()
        assert "chat-tab-status chat-tab-ok" in body, body
        assert "chat-tab-status chat-tab-fail" in body, body
        # Selection is still keyed on ``tab.lastTaskFailed``.
        assert "tab.lastTaskFailed" in body, body

    def test_status_icon_only_when_tab_has_run_task(self) -> None:
        """The status circle must only be created when ``tab.hasRunTask``
        is true and the tab is not currently running — exactly the
        condition that applies to both freshly-created tabs that have
        completed a task and tabs loaded from history.
        """
        body = _regular_tab_branch_source()
        # The branch starts with ``if (tab.isRunning) { spinner } else
        # if (tab.hasRunTask) { status icon }``.
        m = re.search(
            r"if\s*\(\s*tab\.isRunning\s*\).*?\}\s*else\s+if\s*\(\s*"
            r"tab\.hasRunTask\s*\)\s*\{",
            body,
            flags=re.DOTALL,
        )
        assert m is not None, body

    def test_css_colours_status_circle_green_and_red(self) -> None:
        """``chat-tab-ok`` paints the circle in ``--green`` and
        ``chat-tab-fail`` in ``--red``.  These rules are what turn the
        ●-glyph into a coloured indicator.
        """
        css = _MAIN_CSS.read_text(encoding="utf-8")
        assert re.search(
            r"\.chat-tab-ok\s*\{[^}]*color:\s*var\(--green\)", css
        ), css
        assert re.search(
            r"\.chat-tab-fail\s*\{[^}]*color:\s*var\(--red\)", css
        ), css
