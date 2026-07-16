# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compatibility alias — module moved to :mod:`kiss.server.web_server`.

Importing this module yields the real :mod:`kiss.server.web_server`
module object (via a ``sys.modules`` alias), so attribute access,
monkeypatching, and the ``kiss-web`` console entry point resolved
through either import path affect the same module.  Running
``python -m kiss.agents.vscode.web_server`` still starts the daemon by
delegating to :func:`kiss.server.web_server.main`.
"""

import sys
from typing import Any

from kiss.server import web_server as _mod
from kiss.server.web_server import *  # noqa: F403 — precise types for static analysis

if __name__ == "__main__":
    _mod.main()

sys.modules[__name__] = _mod


def __getattr__(name: str) -> Any:  # pragma: no cover — static-analysis aid
    """Delegate attribute lookup to the relocated module (PEP 562)."""
    return getattr(_mod, name)
