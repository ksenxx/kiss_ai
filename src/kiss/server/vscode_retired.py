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
lives now.  The flag goes stale on its own, and the toast's "Do not show
this" checkbox ends it earlier: the ``dismissNotice`` command writes
``$KISS_HOME/.vscode-retired-dismissed``, which silences the notice for
good in every browser of this daemon, even after a later install
refreshes the flag.
"""

from __future__ import annotations

import time
from typing import Any

from kiss.core.brand import PRODUCT_NAME
from kiss.core.config import kiss_home
from kiss.server.install_problems import marker_is_fresh

VSCODE_RETIRED_FLAG = ".vscode-retired"
"""Basename, under ``$KISS_HOME``, of the flag ``install.sh`` writes."""

VSCODE_RETIRED_DISMISSED = ".vscode-retired-dismissed"
"""Basename, under ``$KISS_HOME``, of the "Do not show this again" marker."""

NOTICE_ID = "vscode-retired"
"""The notification id: one toast per page, replaced on reconnect."""


def vscode_retired_notice(loopback_url: str) -> dict[str, Any] | None:
    """The sticky ``notification`` event for a fresh flag, else ``None``.

    Fresh means written within :data:`kiss.server.install_problems.NOTICE_DAYS`.
    ``None`` also once the user ticked "Do not show this again" (see
    :func:`dismiss_vscode_retired_notice`).

    Args:
        loopback_url: The daemon's ``http://127.0.0.1:PORT`` URL the
            notice sends the user to.
    """
    if (kiss_home() / VSCODE_RETIRED_DISMISSED).exists():
        return None
    if not marker_is_fresh(kiss_home() / VSCODE_RETIRED_FLAG):
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
        "dontShowAgain": True,
    }


def dismiss_vscode_retired_notice() -> None:
    """Record the "Do not show this again" choice: write the dismissal marker.

    Best-effort, like the tips opt-out: an unwritable ``$KISS_HOME``
    only means the notice may show again until the flag goes stale.
    """
    marker = kiss_home() / VSCODE_RETIRED_DISMISSED
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    except OSError:
        pass
