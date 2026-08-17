# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.agents.vscode.test_remote_webapp_auto_reload``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import re

from kiss.server.web_server import _WS_SHIM_JS


def test_shim_source_contains_reload_call():
    """Sanity check: the shipped shim source includes a reload call.

    A second, source-level guard so a future refactor that removes
    the reload (and is somehow missed by the browser-level test
    above — e.g. an inadvertent rename of the mock helpers) still
    fails loudly.  This is intentionally narrow: it only asserts
    that ``location.reload`` appears somewhere in the shim string.
    """
    assert re.search(r"location\s*\.\s*reload\s*\(", _WS_SHIM_JS), (
        "BUG: _WS_SHIM_JS must call location.reload() to recover from "
        "a server restart"
    )
