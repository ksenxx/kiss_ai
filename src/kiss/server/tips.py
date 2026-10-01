# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Load the tips shown by the chat webview.

Parses the bundled ``src/kiss/TIPS.md`` into a list of markdown tip
strings, one per ``# Tip`` section, and reads the "don't show tips
again" marker.  The daemon is the only reader: the remote webapp
builder (``web_server._build_html``) injects :func:`tips_data` as
``window.__TIPS__`` at page load, and the ``ready`` handler sends the
same dict as a ``tipsData`` event to every (re)connecting client, so
the VS Code extension paints tips without parsing anything itself.

The file path can be overridden via the ``KISS_TIPS_PATH`` environment
variable, which the test suite uses to pin deterministic tips.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from kiss.core.brand import render_brand
from kiss.core.config import kiss_home

_TIP_DELIMITER = re.compile(r"^# Tip.*$", re.MULTILINE)

TIPS_OPT_OUT_MARKER = "TIPS_DISABLED"
"""Basename, under ``$KISS_HOME``, of the "don't show tips again" marker."""


def tips_disabled() -> bool:
    """Whether the user opted out of the tips window on any surface.

    The marker ``$KISS_HOME/TIPS_DISABLED`` is written by the daemon's
    ``tipsOptOut`` API, which both the VS Code webview and the remote
    webapp call, so one choice holds everywhere.
    """
    return (kiss_home() / TIPS_OPT_OUT_MARKER).exists()


def _bundled_tips_path() -> Path:
    """Return the path to the bundled ``src/kiss/TIPS.md``.

    Honours the ``KISS_TIPS_PATH`` env override (used by the test
    suite), falling back to the file shipped inside the package.
    """
    override = os.environ.get("KISS_TIPS_PATH")
    if override:
        return Path(override)
    return Path(__file__).parent.parent / "TIPS.md"


def read_tips() -> list[str]:
    """Return one markdown string per ``# Tip`` section in ``TIPS.md``.

    Every line starting with ``# Tip`` begins a new tip; the tip body
    is the markdown text up to the next such line (or EOF), trimmed.
    Text before the first ``# Tip`` line and tips with empty bodies
    are skipped.  Returns ``[]`` when the file is missing or
    unreadable (graceful degradation — the chat webview simply shows
    no tips window).  Brand placeholders such as ``{{PRODUCT_NAME}}``
    are filled from ``kiss.core.brand`` first.
    """
    try:
        text = _bundled_tips_path().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    sections = _TIP_DELIMITER.split(render_brand(text))
    return [body.strip() for body in sections[1:] if body.strip()]


def tips_data(version: str) -> dict[str, object]:
    """The tips bootstrap both surfaces paint from (``window.__TIPS__``).

    ``show`` allows the auto-open unless the user opted out; the client
    (``tips.js``) then opens the window once per *version* per browser
    profile, i.e. on first use and again after every update.

    Args:
        version: The running version, sent so the client can key its
            once-per-version guard.
    """
    tips = read_tips()
    return {
        "tips": tips,
        "show": bool(tips) and not tips_disabled(),
        "version": version,
    }
