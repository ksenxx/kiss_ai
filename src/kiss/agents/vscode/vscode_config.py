# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compatibility alias — module moved to :mod:`kiss.server.vscode_config`.

Importing this module yields the real :mod:`kiss.server.vscode_config`
module object (via a ``sys.modules`` alias), so attribute access and
monkeypatching through either import path affect the same module.
"""

import sys
from typing import Any

from kiss.server import vscode_config as _mod
from kiss.server.vscode_config import *  # noqa: F403 — precise types for static analysis

sys.modules[__name__] = _mod


def __getattr__(name: str) -> Any:  # pragma: no cover — static-analysis aid
    """Delegate attribute lookup to the relocated module (PEP 562)."""
    return getattr(_mod, name)
