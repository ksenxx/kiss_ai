# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Hand sign-in pages to the USER's default browser.

The implementation is :mod:`kiss.core.browser_handoff`; it lives in
``kiss.core`` because ``kiss.agents.sorcar.mcp_oauth`` needs it too and
Sorcar code may depend only on ``kiss.core``.  This module keeps the
connectors' import path.
"""

from __future__ import annotations

from kiss.core.browser_handoff import (
    _launch_commands,
    browser_handoff_note,
    open_in_default_browser,
    portal_handoff,
)

__all__ = [
    "_launch_commands",
    "browser_handoff_note",
    "open_in_default_browser",
    "portal_handoff",
]
