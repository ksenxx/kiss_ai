"""Integration tests: every chat panel must expose a copy-to-clipboard button.

Feature: each panel rendered in the chat webview (tool calls, tool result
errors, bash stream output, the result card, system/user prompt panels,
Thoughts panels, and merge-info panels) must include a copy button that
writes the panel's plain text to the clipboard when clicked.

The implementation lives in ``main.js`` (``addCopyButton`` helper, plus
explicit calls for the headerless bash/result panels) and ``main.css``
(``.panel-copy-btn`` styles + ``.copyable`` positioning + updates to two
``.collapsed > :not(...)`` rules that previously hid every non-header
child).
"""

from __future__ import annotations

import re
from pathlib import Path

VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"
MAIN_JS = VSCODE_DIR / "media" / "main.js"
MAIN_CSS = VSCODE_DIR / "media" / "main.css"


def _read(path: Path) -> str:
    assert path.is_file(), f"file not found: {path}"
    return path.read_text()


def test_add_copy_button_helper_exists() -> None:
    """main.js must define an ``addCopyButton`` helper that uses the
    async clipboard API and stops click propagation so it does not
    trigger the collapsible header listener."""
    src = _read(MAIN_JS)
    assert re.search(r"function\s+addCopyButton\s*\(\s*panelEl\s*\)", src), (
        "main.js must define `function addCopyButton(panelEl)`"
    )
    # The helper must call navigator.clipboard.writeText
    assert "navigator.clipboard.writeText" in src, (
        "addCopyButton must use navigator.clipboard.writeText to copy text"
    )
    # It must mark the panel as copyable so the CSS positions the button.
    assert "classList.add('copyable')" in src, (
        "addCopyButton must add the 'copyable' class to the panel"
    )


def test_add_collapse_attaches_copy_button() -> None:
    """``addCollapse`` must invoke ``addCopyButton`` so every collapsible
    panel (tool call, prompt, Thoughts, tool-result error, merge-info)
    automatically gets a copy button."""
    src = _read(MAIN_JS)
    m = re.search(
        r"function addCollapse\([^)]*\)\s*\{(.*?)^\s{2}\}",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "could not locate function addCollapse in main.js"
    body = m.group(1)
    assert "addCopyButton(panelEl)" in body, (
        "addCollapse must call addCopyButton(panelEl) so every collapsible "
        "panel gets a copy button"
    )


def test_explicit_copy_buttons_for_headerless_panels() -> None:
    """The bash stream panel (inside a tool call), the bash output panel
    (successful tool result), and the result card don't go through
    addCollapse, so they must call addCopyButton directly."""
    src = _read(MAIN_JS)
    # bp: bash panel under .tc
    assert re.search(
        r"const\s+bp\s*=\s*mkEl\('div',\s*'bash-panel'\).*?addCopyButton\(bp\)",
        src,
        re.DOTALL,
    ), "the bash-panel inside a tool call (`bp`) must get addCopyButton(bp)"
    # op: success bash panel
    assert re.search(
        r"const\s+op\s*=\s*mkEl\('div',\s*'bash-panel'\).*?addCopyButton\(op\)",
        src,
        re.DOTALL,
    ), "the success bash-panel (`op`) must get addCopyButton(op)"
    # rc: result card
    assert re.search(
        r"hlBlock\(rc\);\s*addCopyButton\(rc\);",
        src,
    ), "the result card (`rc`) must get addCopyButton(rc) after hlBlock"


def test_collect_text_skips_panel_chrome() -> None:
    """``collectText`` must skip the copy button, the collapse chevron,
    and the collapse preview so neither the clipboard payload nor the
    collapsed-state preview repeats those UI-only fragments."""
    src = _read(MAIN_JS)
    m = re.search(
        r"function collectText\([^)]*\)\s*\{(.*?)^\s{2}\}",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "could not locate function collectText"
    body = m.group(1)
    for cls in ("panel-copy-btn", "collapse-chv", "collapse-preview"):
        assert f"'{cls}'" in body, (
            f"collectText must skip nodes with class '{cls}' so the copy "
            f"button / chevron / preview never leak into clipboard text"
        )


def test_panel_copy_button_css_present() -> None:
    """main.css must style ``.panel-copy-btn`` and make ``.copyable``
    the positioning context (``position: relative``)."""
    css = _read(MAIN_CSS)
    assert ".panel-copy-btn" in css, "main.css must style .panel-copy-btn"
    # `.copyable { position: relative; }` (whitespace-tolerant)
    assert re.search(
        r"\.copyable\s*\{[^}]*position:\s*relative", css
    ), ".copyable must set position: relative so the button anchors to the panel"
    # Button must be absolutely positioned.
    assert re.search(
        r"\.panel-copy-btn\s*\{[^}]*position:\s*absolute", css
    ), ".panel-copy-btn must use position: absolute"


def test_collapsed_rules_keep_copy_button_visible() -> None:
    """The two ``.collapsed > :not(<header>)`` rules previously hid every
    non-header direct child.  They must now also exempt
    ``.panel-copy-btn`` so the button stays clickable when the panel is
    collapsed."""
    css = _read(MAIN_CSS)
    # Stylelint's selector-not-notation rule requires complex :not() form:
    # :not(A, B) instead of :not(A):not(B).
    assert ".tc.collapsed > :not(.tc-h, .panel-copy-btn)" in css, (
        ".tc collapsed rule must exempt .panel-copy-btn (via the "
        "complex :not(.tc-h, .panel-copy-btn) selector) so the copy "
        "button stays visible when the tool-call panel is collapsed"
    )
    assert ".llm-panel.collapsed > :not(.llm-panel-hdr, .panel-copy-btn)" in css, (
        ".llm-panel collapsed rule must exempt .panel-copy-btn so the copy "
        "button stays visible when the Thoughts panel is collapsed"
    )
