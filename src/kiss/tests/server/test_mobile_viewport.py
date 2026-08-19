# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: remote webview fits mobile screens horizontally.

The viewport meta value is substituted by ``kiss.server.web_server``
itself (the ``{{VIEWPORT}}`` placeholder), so this test's dependency
closure is server-only; the file moved here from tests/agents/vscode.
"""

import re


def _build_html() -> str:
    from kiss.server.web_server import _build_html

    return _build_html()


def test_viewport_meta_has_device_width_and_max_scale() -> None:
    """The viewport meta must set width=device-width and maximum-scale=1."""
    html = _build_html()
    meta = re.search(r'<meta\s+name="viewport"\s+content="([^"]+)"', html)
    assert meta, "viewport meta tag not found"
    content = meta.group(1)
    assert "width=device-width" in content
    assert "maximum-scale=1" in content
