# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Load the fresh-install tips shown by the chat webview.

Python counterpart to ``getTips`` in ``SorcarTab.ts``: parses the
bundled ``src/kiss/TIPS.md`` into a list of markdown tip strings, one
per ``# Tip`` section.  The remote webapp builder
(``web_server._build_html``) injects the list as ``window.__TIPS__``
so the shared ``media/chat.html`` template never contains an
unsubstituted ``{{TIPS_JSON}}`` placeholder.

The file path can be overridden via the ``KISS_TIPS_PATH`` environment
variable, which the test suite uses to pin deterministic tips.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Every line starting with ``# Tip`` begins a new tip section.
_TIP_DELIMITER = re.compile(r"^# Tip.*$", re.MULTILINE)


def _bundled_tips_path() -> Path:
    """Return the path to the bundled ``src/kiss/TIPS.md``.

    Honours the ``KISS_TIPS_PATH`` env override (used by the test
    suite), falling back to the file shipped inside the package.
    """
    override = os.environ.get("KISS_TIPS_PATH")
    if override:
        return Path(override)
    # ``__file__`` is ``…/kiss/server/tips.py``; the bundled TIPS.md
    # lives at ``…/kiss/TIPS.md`` (one ``parent`` up from ``server/``).
    return Path(__file__).parent.parent / "TIPS.md"


def read_tips() -> list[str]:
    """Return one markdown string per ``# Tip`` section in ``TIPS.md``.

    Every line starting with ``# Tip`` begins a new tip; the tip body
    is the markdown text up to the next such line (or EOF), trimmed.
    Text before the first ``# Tip`` line and tips with empty bodies
    are skipped.  Returns ``[]`` when the file is missing or
    unreadable (graceful degradation — the chat webview simply shows
    no tips window).
    """
    try:
        text = _bundled_tips_path().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    sections = _TIP_DELIMITER.split(text)
    return [body.strip() for body in sections[1:] if body.strip()]
