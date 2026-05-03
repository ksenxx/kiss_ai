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


def test_html_has_overflow_x_hidden() -> None:
    """html must have overflow-x: hidden to prevent horizontal scroll."""
    css = _read_css()
    match = re.search(r"html\s*\{([^}]+)\}", css)
    assert match, "html {} rule not found in main.css"
    rule = match.group(1)
    assert "overflow-x" in rule and "hidden" in rule


def test_html_has_max_width_100vw() -> None:
    """html must have max-width: 100vw."""
    css = _read_css()
    match = re.search(r"html\s*\{([^}]+)\}", css)
    assert match
    assert "max-width" in match.group(1) and "100vw" in match.group(1)


def test_app_has_overflow_x_hidden() -> None:
    """#app must have overflow-x: hidden."""
    css = _read_css()
    match = re.search(r"#app\s*\{([^}]+)\}", css)
    assert match, "#app {} rule not found"
    rule = match.group(1)
    assert "overflow-x" in rule and "hidden" in rule


def test_model_picker_flex_wrap() -> None:
    """#model-picker must use flex-wrap: wrap to avoid overflow."""
    css = _read_css()
    match = re.search(r"#model-picker\s*\{([^}]+)\}", css)
    assert match, "#model-picker {} rule not found"
    rule = match.group(1)
    assert "flex-wrap" in rule and "wrap" in rule


def test_model_dropdown_min_width_clamped() -> None:
    """#model-dropdown min-width must be clamped to viewport width."""
    css = _read_css()
    match = re.search(r"#model-dropdown\s*\{([^}]+)\}", css)
    assert match, "#model-dropdown {} rule not found"
    rule = match.group(1)
    assert "100vw" in rule or "calc" in rule


def test_input_footer_flex_wrap() -> None:
    """#input-footer must allow wrapping on narrow screens."""
    css = _read_css()
    match = re.search(r"#input-footer\s*\{([^}]+)\}", css)
    assert match, "#input-footer {} rule not found"
    rule = match.group(1)
    assert "flex-wrap" in rule and "wrap" in rule


def test_model_btn_max_width_clamped() -> None:
    """#model-btn max-width must not exceed viewport on small screens."""
    css = _read_css()
    match = re.search(r"#model-btn\s*\{([^}]+)\}", css)
    assert match, "#model-btn {} rule not found"
    rule = match.group(1)
    assert "max-width" in rule
    assert "vw" in rule or "min(" in rule


def test_body_has_max_width() -> None:
    """body must have max-width: 100vw."""
    css = _read_css()
    match = re.search(r"body\s*\{([^}]+)\}", css)
    assert match, "body {} rule not found"
    assert "max-width" in match.group(1)
