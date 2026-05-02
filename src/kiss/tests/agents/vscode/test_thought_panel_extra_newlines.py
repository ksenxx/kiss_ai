"""Regression test: thought panels must not show extra newlines after Markdown→HTML conversion.

Bug: The ``.txt`` element has ``white-space: pre-wrap`` for streaming raw
text deltas.  When ``text_end`` fires, the element gets the ``md-body``
class and its ``innerHTML`` is replaced by ``marked.parse()`` output.
Because ``marked.parse()`` produces HTML with literal ``\\n`` characters
between tags (e.g. ``<p>line1</p>\\n<p>line2</p>``), the still-active
``pre-wrap`` renders those ``\\n`` as visible line breaks **on top of**
the ``<p>`` margins — doubling the spacing after every line.

Fix: The CSS must reset ``white-space`` to ``normal`` when a ``.txt``
element also has the ``md-body`` class, so that only HTML semantics
(margins, ``<br>``) control spacing.
"""

from __future__ import annotations

import re
from pathlib import Path

MEDIA = Path(__file__).parent.parent.parent.parent / "agents" / "vscode" / "media"
MAIN_JS = MEDIA / "main.js"
MAIN_CSS = MEDIA / "main.css"


def _read(p: Path) -> str:
    assert p.is_file(), f"{p.name} not found at {p}"
    return p.read_text()


# ── Helpers ────────────────────────────────────────────────────────────


def _css_rules_for_selector(css: str, selector: str) -> list[str]:
    """Return all CSS declaration blocks whose selector matches *selector*.

    Performs a simple regex search — good enough for flat rule sets.
    """
    pattern = re.compile(
        re.escape(selector) + r"\s*\{([^}]*)\}",
        re.DOTALL,
    )
    return [m.group(1) for m in pattern.finditer(css)]


def _extract_case(src: str, label: str) -> str:
    """Extract the body of a ``case '<label>':`` arm until its ``break;``."""
    m = re.search(rf"case\s+'{label}':\s*(.*?)break;", src, re.DOTALL)
    assert m, f"could not locate case '{label}': in main.js"
    return m.group(1)


# ── Tests ──────────────────────────────────────────────────────────────


def test_txt_has_pre_wrap() -> None:
    """Precondition: ``.txt`` uses ``pre-wrap`` for streaming deltas."""
    css = _read(MAIN_CSS)
    blocks = _css_rules_for_selector(css, ".txt")
    assert blocks, "no .txt rule found in main.css"
    joined = " ".join(blocks)
    assert "pre-wrap" in joined, ".txt must have white-space: pre-wrap"


def test_text_end_adds_md_body_class() -> None:
    """``text_end`` handler must add ``md-body`` before setting innerHTML."""
    js = _read(MAIN_JS)
    body = _extract_case(js, "text_end")
    assert "md-body" in body, "text_end must add md-body class to .txt element"
    assert "marked.parse" in body, "text_end must call marked.parse"


def test_txt_md_body_resets_white_space() -> None:
    """When ``.txt`` gets ``md-body``, ``white-space`` must not be ``pre-wrap``.

    Without this, literal ``\\n`` in ``marked.parse()`` HTML output would
    render as visible line breaks, creating extra blank lines.
    """
    css = _read(MAIN_CSS)

    # Look for a rule that targets .txt elements with .md-body and sets
    # white-space to something other than pre-wrap.  Accepted selectors:
    #   .txt.md-body   or   .md-body.txt   or   .md-body (if it sets white-space)
    candidates = (
        _css_rules_for_selector(css, ".txt.md-body")
        + _css_rules_for_selector(css, ".md-body.txt")
    )

    has_reset = False
    for block in candidates:
        ws_match = re.search(r"white-space\s*:\s*([^;]+)", block)
        if ws_match and "pre-wrap" not in ws_match.group(1):
            has_reset = True
            break

    assert has_reset, (
        "main.css is missing a .txt.md-body rule that resets white-space "
        "away from pre-wrap.  After marked.parse() converts text to HTML, "
        "the literal \\n characters in the HTML source would be rendered as "
        "visible line breaks, adding extra blank lines after every line."
    )
