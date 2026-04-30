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


# ── Viewport meta tag ────────────────────────────────────────


def test_viewport_meta_has_device_width_and_max_scale() -> None:
    """The viewport meta must set width=device-width and maximum-scale=1."""
    html = _build_html()
    meta = re.search(r'<meta\s+name="viewport"\s+content="([^"]+)"', html)
    assert meta, "viewport meta tag not found"
    content = meta.group(1)
    assert "width=device-width" in content
    assert "maximum-scale=1" in content


# ── html element overflow constraint ─────────────────────────


def test_html_has_overflow_x_hidden() -> None:
    """html must have overflow-x: hidden to prevent horizontal scroll."""
    css = _read_css()
    # Find the html rule block
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


# ── #app overflow constraint ─────────────────────────────────


def test_app_has_overflow_x_hidden() -> None:
    """#app must have overflow-x: hidden."""
    css = _read_css()
    match = re.search(r"#app\s*\{([^}]+)\}", css)
    assert match, "#app {} rule not found"
    rule = match.group(1)
    assert "overflow-x" in rule and "hidden" in rule


# ── #model-picker wraps on narrow screens ────────────────────


def test_model_picker_flex_wrap() -> None:
    """#model-picker must use flex-wrap: wrap to avoid overflow."""
    css = _read_css()
    match = re.search(r"#model-picker\s*\{([^}]+)\}", css)
    assert match, "#model-picker {} rule not found"
    rule = match.group(1)
    assert "flex-wrap" in rule and "wrap" in rule


# ── #model-dropdown respects small screens ───────────────────


def test_model_dropdown_min_width_clamped() -> None:
    """#model-dropdown min-width must be clamped to viewport width."""
    css = _read_css()
    match = re.search(r"#model-dropdown\s*\{([^}]+)\}", css)
    assert match, "#model-dropdown {} rule not found"
    rule = match.group(1)
    # Must use min() or calc() with vw to clamp
    assert "100vw" in rule or "calc" in rule


# ── #input-footer wraps on narrow screens ────────────────────


def test_input_footer_flex_wrap() -> None:
    """#input-footer must allow wrapping on narrow screens."""
    css = _read_css()
    match = re.search(r"#input-footer\s*\{([^}]+)\}", css)
    assert match, "#input-footer {} rule not found"
    rule = match.group(1)
    assert "flex-wrap" in rule and "wrap" in rule


# ── #model-btn max-width is viewport-safe ────────────────────


def test_model_btn_max_width_clamped() -> None:
    """#model-btn max-width must not exceed viewport on small screens."""
    css = _read_css()
    # Search for the model-btn rule that contains max-width
    # Use a broader search since it might be multi-line
    match = re.search(r"#model-btn\s*\{([^}]+)\}", css)
    assert match, "#model-btn {} rule not found"
    rule = match.group(1)
    assert "max-width" in rule
    # Must use min() or vw to clamp
    assert "vw" in rule or "min(" in rule


# ── body has max-width constraint ────────────────────────────


def test_body_has_max_width() -> None:
    """body must have max-width: 100vw."""
    css = _read_css()
    match = re.search(r"body\s*\{([^}]+)\}", css)
    assert match, "body {} rule not found"
    assert "max-width" in match.group(1)
