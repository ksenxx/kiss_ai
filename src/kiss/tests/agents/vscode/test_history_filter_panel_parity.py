"""The History filter panel must be identical between the VS Code
extension webview (``SorcarTab.ts``) and the standalone webapp
(``web_server._build_html``).

Both surfaces share the same ``main.css`` / ``main.js`` assets, so the
HTML for ``.history-filter-bar`` — the row of checkboxes
(Running / Errored / Succeeded / Favorites) plus the From / To date
pickers — must match exactly (modulo whitespace) on both sides.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from kiss.agents.vscode import web_server

_SORCAR_TAB_TS = (
    Path(web_server.__file__).parent / "src" / "SorcarTab.ts"
)


def _extract_filter_bar(text: str) -> str:
    """Return the ``<div class="history-filter-bar">…</div>`` block."""
    m = re.search(
        r'<div class="history-filter-bar">.*?</div>',
        text,
        re.DOTALL,
    )
    assert m is not None, "history-filter-bar block not found"
    return re.sub(r"\s+", " ", m.group(0)).strip()


class TestHistoryFilterPanelParity(unittest.TestCase):
    """The extension and webapp must render identical filter bars."""

    def test_filter_bars_match(self) -> None:
        """The ``.history-filter-bar`` block is identical in both
        sources (after collapsing whitespace)."""
        webapp = _extract_filter_bar(web_server._build_html())
        extension = _extract_filter_bar(_SORCAR_TAB_TS.read_text())
        self.assertEqual(extension, webapp)


if __name__ == "__main__":
    unittest.main()
