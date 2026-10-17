# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A re-branded copy of the ``kiss`` package for end-to-end brand tests.

``kiss.core.brand`` reads the brand from ``webapp/media/brand.json``
next to the package at import time, exactly as an installed white-label
copy carries it after ``install.sh`` overlaid its ``.brand/`` files.
Tests that must see a *branded* product end to end (not the stock
"KISS Sorcar" this checkout is) copy the package into a temp directory
with the brand files replaced, and run the code under test in a
subprocess whose ``PYTHONPATH`` points at the copy.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1]

SEAMLESS_LOOP: dict[str, str] = {
    "product_name": "Seamless Loop",
    "short_name": "s10s",
    "tagline": "SeamlessLabs' AI assistant. Ask me anything!",
    "identity": "You are Seamless Loop (s10s for short), the AI assistant of SeamlessLabs.",
    "home_dir": ".s10s",
}
"""The brand of the Seamless Loop distribution (its ``brand.json``)."""


def make_branded_package(root: Path, brand: dict[str, str] = SEAMLESS_LOOP) -> Path:
    """Copy the ``kiss`` package under *root* with *brand* applied; return the ``src`` dir.

    The copy skips ``node_modules`` (the webapp's dev dependencies,
    never imported), the tests and byte-code caches.  ``brand.json`` is
    replaced by *brand*; ``kiss-icon.png`` by a distinct 1x1 PNG so a
    test can tell the brand's icon from the stock one by content.
    """
    src = root / "src"
    shutil.copytree(
        _PACKAGE,
        src / "kiss",
        ignore=shutil.ignore_patterns("node_modules", "tests", "__pycache__", "*.pyc"),
        symlinks=True,
    )
    media = src / "kiss" / "webapp" / "media"
    (media / "brand.json").write_text(json.dumps(brand, indent=2) + "\n", encoding="utf-8")
    (media / "kiss-icon.png").write_bytes(BRAND_ICON_PNG)
    return src


BRAND_ICON_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cfc0f01f000501020082a8d5f70000000049454e44ae426082"
)
"""A valid 1x1 RGBA PNG, the branded copy's ``kiss-icon.png``."""
