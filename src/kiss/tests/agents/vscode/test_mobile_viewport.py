# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: remote webview fits mobile screens horizontally.

Verifies that:
- The viewport meta tag prevents zoom and sets width=device-width
- Key CSS rules prevent horizontal overflow on narrow screens
- The generated HTML from _build_html() includes mobile-safe constraints
"""

import re
from pathlib import Path

CSS_PATH = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media" / "main.css"
)


def _read_css() -> str:
    return CSS_PATH.read_text()


def _build_html() -> str:
    from kiss.agents.vscode.web_server import _build_html

    return _build_html()


def test_viewport_meta_has_device_width_and_max_scale() -> None:
    """The viewport meta must set width=device-width and maximum-scale=1."""
    html = _build_html()
    meta = re.search(r'<meta\s+name="viewport"\s+content="([^"]+)"', html)
    assert meta, "viewport meta tag not found"
    content = meta.group(1)
    assert "width=device-width" in content
    assert "maximum-scale=1" in content
















