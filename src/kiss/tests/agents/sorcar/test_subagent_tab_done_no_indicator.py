"""Regression test: a non-running (``isDone=true``) sub-agent tab must
NOT render the green ✓ / red ✗ status icon in the tab title.

Contract
--------
The previous behaviour rendered ``✓`` (green) for done sub-agent tabs
and ``◉`` (purple, pulsing) for running ones.  An idle ✓ on a long-
finished history-reopened sub-agent tab adds noise without conveying
new information; the purple ``.subagent-tab`` accent already makes
the tab unambiguously a sub-agent tab.  So ``renderTabBar`` must
suppress the indicator entirely when ``tab.isSubagentTab`` is true
and ``tab.isDone`` is also true — and still render the running ``◉``
when ``tab.isDone`` is false.
"""

from __future__ import annotations

import re
from pathlib import Path

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss" / "agents" / "vscode" / "media" / "main.js"
)


def _strip_js_comments(src: str) -> str:
    """Drop ``//`` line comments and ``/* */`` block comments so glyphs
    in commentary text don't trip the executable-code checks below.
    """
    # Block comments first (non-greedy, DOTALL).
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    # Line comments.
    src = re.sub(r"//[^\n]*", "", src)
    return src


def _subagent_branch_source() -> str:
    """Return the JS source of the ``if (tab.isSubagentTab)`` branch
    inside ``renderTabBar``, with comments stripped.
    """
    src = _MAIN_JS.read_text(encoding="utf-8")
    # Anchor on the renderTabBar function so we don't accidentally
    # match unrelated occurrences of ``isSubagentTab`` elsewhere.
    fn_idx = src.index("function renderTabBar(")
    branch_idx = src.index("if (tab.isSubagentTab) {", fn_idx)
    # The matching ``} else {`` closes this branch.
    else_idx = src.index("} else {", branch_idx)
    return _strip_js_comments(src[branch_idx:else_idx])


class TestSubagentIndicatorSuppressedWhenDone:
    """Static checks on ``media/main.js`` ``renderTabBar`` subagent
    branch."""

    def test_done_branch_does_not_render_tick(self) -> None:
        """A done sub-agent tab must not emit a ✓ / \\u2713 glyph."""
        body = _subagent_branch_source()
        assert "✓" not in body, body
        assert "\\u2713" not in body, body
        # Defensive: also ensure no red-cross variant was reintroduced.
        assert "✗" not in body, body
        assert "\\u2717" not in body, body

    def test_running_indicator_still_rendered(self) -> None:
        """The pulsing ◉ for a running sub-agent must still be created."""
        body = _subagent_branch_source()
        assert "◉" in body, body
        # The ``subagent-indicator`` element must still be appended for
        # the running case so the purple pulse animation kicks in.
        assert "subagent-indicator" in body, body
        assert "el.appendChild(subIndicator)" in body, body

    def test_indicator_creation_is_guarded_by_not_is_done(self) -> None:
        """The indicator must be created only when ``!tab.isDone`` —
        guard prevents the done state from emitting any glyph at all.
        """
        body = _subagent_branch_source()
        # The whole indicator-construction block lives inside a
        # ``if (!tab.isDone) { ... }`` guard.
        m = re.search(r"if\s*\(\s*!tab\.isDone\s*\)\s*\{", body)
        assert m is not None, body
        # And the indicator-creation statement comes AFTER that guard.
        guard_end = m.end()
        appended_at = body.index("el.appendChild(subIndicator)")
        assert appended_at > guard_end, body
