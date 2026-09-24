# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Product branding: the one place the product name and identity live.

Every user-visible name ("KISS Sorcar", the "KISS:" command prefix, the
agent's identity sentence in ``SYSTEM.md``) comes from
``src/kiss/agents/vscode/media/brand.json``.  The file sits in the
extension's ``media/`` directory because that is the single location
reachable by all three consumers: this Python package, the extension
host (``src/brand.ts``) and the shared ``chat.html`` template.  A custom
distribution re-brands the product by replacing that file (and the icon
files next to it) instead of editing source code: it drops its copies
into the git-ignored ``.brand/`` directory at the checkout root and
``install.sh`` swaps them in for the extension build (see "Brand
overlay" there), so the checked-in files always carry the stock brand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_MEDIA_DIR = Path(__file__).resolve().parents[1] / "agents" / "vscode" / "media"
BRAND_FILE = _MEDIA_DIR / "brand.json"

DEFAULT_BRAND: dict[str, str] = {
    "product_name": "KISS Sorcar",
    "short_name": "KISS",
    "tagline": "Your AI assistant. Ask me anything!",
    "identity": (
        "You are KISS Sorcar, an AI Assistant and a general-purpose multi-model, "
        "multi-modal, multi-agent AI Agent Framework researched and developed by "
        "Koushik Sen (ksen@berkeley.edu)."
    ),
    "extension_description": (
        "The open-source AI coding agent that beats Cursor and Claude Code on "
        "Terminal Bench. Free, local, bring your own API key."
    ),
}

_PLACEHOLDER_RE = re.compile(r"\{\{(PRODUCT_NAME|SHORT_NAME|TAGLINE|IDENTITY)\}\}")


def load_brand(path: Path = BRAND_FILE) -> dict[str, str]:
    """Return the brand strings from *path*, falling back to the defaults.

    Missing or malformed files and missing keys fall back key-by-key to
    :data:`DEFAULT_BRAND`, so a partial ``brand.json`` (only a new
    ``product_name``) is enough to re-brand the product.  Non-string
    values are ignored the same way.
    """
    brand = dict(DEFAULT_BRAND)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return brand
    if isinstance(loaded, dict):
        for key in DEFAULT_BRAND:
            value = loaded.get(key)
            if isinstance(value, str) and value:
                brand[key] = value
    return brand


BRAND = load_brand()
PRODUCT_NAME = BRAND["product_name"]
SHORT_NAME = BRAND["short_name"]


def render_brand(text: str, brand: dict[str, str] = BRAND) -> str:
    """Fill the brand placeholders in *text*.

    Recognised tokens: ``{{PRODUCT_NAME}}``, ``{{SHORT_NAME}}``,
    ``{{TAGLINE}}`` and ``{{IDENTITY}}``.  Used on the prompt files
    (``SYSTEM.md``, ``SYSTEM_LITE.md``) whose
    identity sentence is brand-specific.  Unknown ``{{...}}`` tokens are
    left untouched so other templating in the same file is unaffected.
    """
    return _PLACEHOLDER_RE.sub(lambda m: brand[m.group(1).lower()], text)
