# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The web app's "VS Code extension removed" notice.

The product left VS Code: its extension is gone from the repository and
will never be updated.  When ``install.sh`` finds a copy of that
extension on the machine (the user came through its Update button, or
the daemon's update ran for them) it writes ``$KISS_HOME/.vscode-retired``
and uninstalls the copy; the daemon then sends the notice below to every
page that connects (the ``ready`` handler's bootstrap events) while the
flag is fresh, so the browser the install opens says where the product
lives now.  The flag goes stale on its own: no acknowledgement to plumb
through the surfaces, and no notice forever.
"""

from __future__ import annotations

import time
from typing import Any

from kiss.core.brand import PRODUCT_NAME
from kiss.core.config import kiss_home

VSCODE_RETIRED_FLAG = ".vscode-retired"
"""Basename, under ``$KISS_HOME``, of the flag ``install.sh`` writes."""

NOTICE_DAYS = 7
"""How long after the install the notice keeps appearing."""

NOTICE_ID = "vscode-retired"
"""The notification id: one toast per page, replaced on reconnect."""


def vscode_retired_notice(loopback_url: str) -> dict[str, Any] | None:
    """The sticky ``notification`` event for a fresh flag, else ``None``.

    Args:
        loopback_url: The daemon's ``http://127.0.0.1:PORT`` URL the
            notice sends the user to.
    """
    try:
        written = (kiss_home() / VSCODE_RETIRED_FLAG).stat().st_mtime
    except OSError:
        return None
    if time.time() - written > NOTICE_DAYS * 86400:
        return None
    return {
        "type": "notification",
        "id": NOTICE_ID,
        "severity": "warning",
        "sticky": True,
        "title": "VS Code extension removed",
        "message": (
            f"The {PRODUCT_NAME} VS Code extension has been removed and will "
            f"never be updated. Use this web app instead: {loopback_url}"
        ),
    }
