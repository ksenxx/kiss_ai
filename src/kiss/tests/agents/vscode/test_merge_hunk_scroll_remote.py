# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for ``scrollHunkIntoView`` in ``media/main.js``.

The remote-web shell sets ``html, body { overflow: hidden }`` (so the page
itself never scrolls) and delegates scrolling to ``#output``.  Native
``Element.scrollIntoView`` is unreliable in that layout — Chromium/Webkit
sometimes try to scroll the (non-scrollable) document instead of bubbling
to the nearest scrollable ancestor, so clicking Accept/Reject/Prev/Next
in the inline merge toolbar would highlight the new hunk but leave it
off-screen.

These static-pattern tests pin the fix: ``scrollHunkIntoView`` must walk
up to the nearest scrollable ancestor explicitly and animate
``scrollTop`` to centre the hunk, only falling back to ``scrollIntoView``
when no scrollable ancestor is found.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_JS_PATH = (
    Path(__file__).resolve().parents[3]
    / "agents" / "vscode" / "media" / "main.js"
)


def _read_js() -> str:
    return _JS_PATH.read_text()


def _extract_function(js: str, name: str) -> str:
    """Return the source of ``function <name>(...)`` up to its matching brace."""
    start_match = re.search(r"function\s+" + re.escape(name) + r"\s*\(", js)
    assert start_match is not None, f"function {name} not found"
    i = js.index("{", start_match.end())
    depth = 1
    j = i + 1
    while j < len(js) and depth > 0:
        c = js[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        j += 1
    assert depth == 0, f"function {name} body not balanced"
    return js[start_match.start():j]


class TestScrollHunkIntoView(unittest.TestCase):
    """``scrollHunkIntoView`` must scroll the nearest scrollable ancestor."""

    def setUp(self) -> None:
        self.body = _extract_function(_read_js(), "scrollHunkIntoView")

    def test_walks_parent_chain_for_scrollable_ancestor(self) -> None:
        """Must walk ``parentElement`` and stop at an overflow-y auto/scroll node."""
        self.assertIn("parentElement", self.body)
        self.assertIn("getComputedStyle", self.body)
        # Accepts both 'auto' and 'scroll' overflow modes.
        self.assertRegex(self.body, r"overflowY|oy")
        self.assertIn("'auto'", self.body)
        self.assertIn("'scroll'", self.body)

    def test_requires_actual_overflow(self) -> None:
        """Stops only at ancestors whose scrollHeight exceeds clientHeight."""
        self.assertIn("scrollHeight", self.body)
        self.assertIn("clientHeight", self.body)

    def test_centres_hunk_in_container(self) -> None:
        """Computes target scrollTop centring the hunk inside the container."""
        # The target is derived from the hunk/container bounding rects and
        # the container's current scrollTop.
        self.assertIn("getBoundingClientRect", self.body)
        self.assertIn("scrollTop", self.body)

    def test_uses_container_scroll_not_only_native_scroll_into_view(self) -> None:
        """Must use ``container.scrollTo``/``scrollTop`` for the active path.

        Native ``scrollIntoView`` is allowed only as a fallback when no
        scrollable ancestor is found (so detached background-tab
        fragments still work).
        """
        self.assertIn("container.scrollTo", self.body)
        # There must be a fallback to native scrollIntoView for the
        # detached / no-ancestor case.
        self.assertIn("scrollIntoView", self.body)

    def test_smooth_behaviour(self) -> None:
        """Animates the scroll smoothly to match prior UX."""
        self.assertIn("'smooth'", self.body)

    def test_clamps_target_into_valid_range(self) -> None:
        """Clamps the computed target to ``[0, scrollHeight - clientHeight]``."""
        self.assertIn("Math.max", self.body)
        self.assertIn("Math.min", self.body)


class TestScrollHunkIntoViewCallSites(unittest.TestCase):
    """``scrollHunkIntoView`` is wired to both merge_data and merge_nav."""

    def test_called_on_initial_merge_data(self) -> None:
        """Initial render scrolls the first hunk into view."""
        js = _read_js()
        # In the merge_data case branch, scrollHunkIntoView(mdEl, 0, 0).
        self.assertRegex(js, r"scrollHunkIntoView\(mdEl,\s*0,\s*0\)")

    def test_called_on_every_merge_nav(self) -> None:
        """Every prev/next/accept/reject response scrolls to the new hunk."""
        js = _read_js()
        # merge_nav handler scrolls using ev.cur.fi/ev.cur.hi.
        self.assertIn(
            "scrollHunkIntoView(mergePanel, ev.cur.fi, ev.cur.hi)", js,
        )


if __name__ == "__main__":
    unittest.main()
