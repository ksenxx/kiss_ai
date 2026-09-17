# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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
    assert re.search(
        r"const\s+bp\s*=\s*mkEl\('div',\s*'bash-panel'\).*?addCopyButton\(bp\)",
        src,
        re.DOTALL,
    ), "the bash-panel inside a tool call (`bp`) must get addCopyButton(bp)"
    assert re.search(
        r"const\s+op\s*=\s*mkEl\('div',\s*'bash-panel'\).*?addCopyButton\(op\)",
        src,
        re.DOTALL,
    ), "the success bash-panel (`op`) must get addCopyButton(op)"
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
    assert re.search(
        r"\.copyable\s*\{[^}]*position:\s*relative", css
    ), ".copyable must set position: relative so the button anchors to the panel"
    assert re.search(
        r"\.panel-copy-btn\s*\{[^}]*position:\s*absolute", css
    ), ".panel-copy-btn must use position: absolute"


def _collapsed_hide_rule_exemptions(css: str, panel: str) -> set[str]:
    """Return the ``:not(...)`` exemption list of *panel*'s collapsed
    display-none rule.

    Finds a rule whose FULL selector is exactly
    ``<panel>.collapsed > :not(<list>)`` (no descendant tail, matched
    from the end of the previous block or file start so a longer
    selector cannot satisfy it) and whose declaration block sets
    ``display: none`` — a match inside a comment, on an unrelated
    longer selector, or on a rule that does not hide the children
    would be a false positive.  Fails the calling test when no such
    rule exists.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)  # strip comments
    pattern = (
        r"(?:^|[};])\s*" + re.escape(panel)
        + r"\.collapsed\s*>\s*:not\(([^(){}]*)\)\s*\{([^{}]*)\}"
    )
    for match in re.finditer(pattern, css):
        if re.search(r"display:\s*none", match.group(2)):
            return {s.strip() for s in match.group(1).split(",")}
    raise AssertionError(
        f"main.css must contain a `{panel}.collapsed > :not(...)` rule "
        "with `display: none` hiding the collapsed panel's children"
    )


def _scan_quote(text: str, i: int) -> int:
    """Return the index just past the quoted string starting at *i*.

    ``text[i]`` must be ``'`` or ``"``; backslash escapes are skipped.
    An unterminated string consumes the rest of *text*.
    """
    quote = text[i]
    i += 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        i += 1
    return i


def _split_top_level(selectors: str) -> list[str]:
    """Split a CSS selector list at top-level commas only.

    A comma nested inside parentheses (``:not(...)``, ``:is(...)``),
    inside an attribute selector (``[data-x=", "]``) or inside a quoted
    string is not a selector-list separator.
    """
    parts: list[str] = []
    start = 0
    depth = 0
    i = 0
    while i < len(selectors):
        ch = selectors[i]
        if ch in "'\"":
            i = _scan_quote(selectors, i)
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(selectors[start:i])
            start = i + 1
        i += 1
    parts.append(selectors[start:])
    return parts


def _strip_not_groups(selector: str) -> str:
    """Remove every ``:not(...)`` group from *selector*.

    Tracks parenthesis/bracket depth and skips quoted strings so nested
    functional pseudos (``:not(:is(.x, .y), .z)``) and attribute values
    inside the ``:not(...)`` are consumed with it.
    """
    out: list[str] = []
    i = 0
    lowered = selector.lower()
    while i < len(selector):
        ch = selector[i]
        if ch in "'\"":
            end = _scan_quote(selector, i)
            out.append(selector[i:end])
            i = end
        elif lowered.startswith(":not(", i):
            depth = 1
            i += len(":not(")
            while i < len(selector) and depth:
                if selector[i] in "'\"":
                    i = _scan_quote(selector, i)
                    continue
                if selector[i] == "(":
                    depth += 1
                elif selector[i] == ")":
                    depth -= 1
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def test_no_collapsed_rule_targets_copy_button() -> None:
    """No hide rule may target ``.panel-copy-btn`` in a collapsed context.

    The exemption checks below prove the two known collapse rules skip
    the button, but a SEPARATE rule such as
    ``.collapsed > .panel-copy-btn { display: none; }`` would out-rank
    the button's own styles and hide it anyway while those checks still
    pass.  Reject any ``display: none`` / ``visibility: hidden`` rule
    whose selector mentions ``collapsed`` and matches the button
    outside a ``:not(...)`` exemption list.
    """
    css = re.sub(r"/\*.*?\*/", "", _read(MAIN_CSS), flags=re.S)
    offending = []
    for rule in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        hides = re.search(
            r"display\s*:\s*none|visibility\s*:\s*hidden",
            rule.group(2),
            re.IGNORECASE,
        )
        if not hides:
            continue
        # Split the selector list at TOP-LEVEL commas only (a comma
        # inside `:not(:is(.x, .y))` is not a list separator), then
        # remove each selector's `:not(...)` groups with a paren-depth
        # scan (a regex cannot skip nested functional pseudos): a rule
        # like `.tc.collapsed:not(:is(.x, .y), .z) > .panel-copy-btn`
        # must reduce to `.tc.collapsed > .panel-copy-btn` and be
        # flagged, while the legitimate exemption rules (whose
        # `.panel-copy-btn` lives inside the `:not(...)`) must not.
        for sel in _split_top_level(rule.group(1)):
            sel = _strip_not_groups(sel).strip()
            if "collapsed" in sel and ".panel-copy-btn" in sel:
                offending.append(sel)
    assert not offending, (
        "these main.css rules hide .panel-copy-btn in a collapsed "
        f"context: {offending}"
    )


def test_collapsed_rules_keep_copy_button_visible() -> None:
    """The two ``.collapsed > :not(<header>)`` rules previously hid every
    non-header direct child.  They must now also exempt
    ``.panel-copy-btn`` so the button stays clickable when the panel is
    collapsed."""
    css = _read(MAIN_CSS)
    tc_exempt = _collapsed_hide_rule_exemptions(css, ".tc")
    assert {".tc-h", ".panel-copy-btn"} <= tc_exempt, (
        ".tc collapsed rule must exempt .tc-h and .panel-copy-btn (its "
        f":not(...) list is {sorted(tc_exempt)}) so the copy button stays "
        "visible when the tool-call panel is collapsed"
    )
    llm_exempt = _collapsed_hide_rule_exemptions(css, ".llm-panel")
    assert {".llm-panel-hdr", ".panel-copy-btn"} <= llm_exempt, (
        ".llm-panel collapsed rule must exempt .llm-panel-hdr and "
        f".panel-copy-btn (its :not(...) list is {sorted(llm_exempt)}) so the "
        "copy button stays visible when the Thoughts panel is collapsed"
    )
