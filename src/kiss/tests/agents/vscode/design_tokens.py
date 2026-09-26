# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Expand main.css design tokens for tests that read CSS source text.

``media/main.css`` defines spacing, radius, shadow, z-index, motion and
semantic-colour tokens in its first ``:root`` block, and the stylesheets
use ``var(--token)`` instead of raw values.  Tests that check a rule's
literal value (``border-radius: 999px``, ``padding: 8px``) read the CSS
through :func:`inline_design_tokens`, so they keep asserting the value
the browser applies.
"""

from __future__ import annotations

import re
from pathlib import Path

MAIN_CSS = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media" / "main.css"
)

_TOKEN_NAME = re.compile(
    r"--(space|radius|shadow|scrim|z|dur|ease|status|favorite|on-accent|paper|ink|attention)\b"
)


def design_tokens() -> dict[str, str]:
    """Return ``{name: value}`` for every design token in main.css's ``:root``."""
    css = MAIN_CSS.read_text(encoding="utf-8")
    match = re.search(r":root\s*\{([^}]*)\}", css)
    assert match, "main.css has no :root block"
    root = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.S)
    return {
        name: value.strip()
        for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", root)
        if _TOKEN_NAME.match(name)
    }


def inline_design_tokens(css: str) -> str:
    """Replace every ``var(--token)`` in ``css`` with the token's value.

    Tokens defined in terms of other tokens are expanded until nothing
    changes.  Theme variables (``--vscode-*``, ``--bg``, ...) are kept.

    Args:
        css: Stylesheet source text.

    Returns:
        The same text with design-token references expanded.
    """
    tokens = design_tokens()
    while True:
        expanded = re.sub(
            r"var\((--[\w-]+)\)",
            lambda m: tokens.get(m.group(1), m.group(0)),
            css,
        )
        if expanded == css:
            return css
        css = expanded
