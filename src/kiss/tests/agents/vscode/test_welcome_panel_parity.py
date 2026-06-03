# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The welcome page must be identical between the VS Code extension
webview (``SorcarTab.ts``) and the standalone webapp
(``web_server._build_html``).

Both surfaces share the same ``main.css`` / ``main.js`` assets, so the
HTML for ``<div id="welcome">…</div>`` — the heading, intro paragraph,
remote-URL line, remote-password field (with show/hide toggle and its
two SVG icons), and the empty ``#suggestions`` container — must match
exactly (modulo whitespace) on both sides.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from kiss.agents.vscode import web_server

_SORCAR_TAB_TS = Path(web_server.__file__).parent / "src" / "SorcarTab.ts"


def _extract_welcome(text: str) -> str:
    """Return the ``<div id="welcome">…</div>`` block.

    Scans from the opening ``<div id="welcome">`` and counts nested
    ``<div>`` opens/closes until the matching ``</div>`` is reached.
    Collapses all whitespace runs to a single space so trivial
    indentation/line-break differences between the Python heredoc and
    the TypeScript template literal don't cause spurious failures.
    """
    m = re.search(r'<div id="welcome">', text)
    assert m is not None, "<div id=\"welcome\"> not found"
    depth = 1
    i = m.end()
    while i < len(text) and depth:
        nxt = text.find("<", i)
        if nxt < 0:
            break
        if text.startswith("<div", nxt) and text[nxt + 4] in " \t\n>":
            depth += 1
            i = nxt + 4
        elif text.startswith("</div>", nxt):
            depth -= 1
            if depth == 0:
                block = text[m.start() : nxt + len("</div>")]
                return re.sub(r"\s+", " ", block).strip()
            i = nxt + len("</div>")
        else:
            i = nxt + 1
    raise AssertionError("unbalanced <div id=\"welcome\"> block")


class TestWelcomePanelParity(unittest.TestCase):
    """The extension and webapp must render an identical welcome page."""

    def test_welcome_blocks_match(self) -> None:
        """The ``<div id="welcome">`` block is identical in both
        sources (after collapsing whitespace)."""
        webapp = _extract_welcome(web_server._build_html())
        extension = _extract_welcome(_SORCAR_TAB_TS.read_text())
        self.assertEqual(extension, webapp)


if __name__ == "__main__":
    unittest.main()
