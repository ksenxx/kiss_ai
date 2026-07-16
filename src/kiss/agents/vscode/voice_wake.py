# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Backward-compatibility alias — module moved to :mod:`kiss.server.voice_wake`.

Importing this module yields the real :mod:`kiss.server.voice_wake`
module object (via a ``sys.modules`` alias), so attribute access and
monkeypatching through either import path affect the same module.
``python -m kiss.agents.vscode.voice_wake`` still runs the listener
CLI by delegating to :func:`kiss.server.voice_wake.main`.
"""

import sys
from typing import Any

from kiss.server import voice_wake as _mod
from kiss.server.voice_wake import *  # noqa: F403 — precise types for static analysis

if __name__ == "__main__":
    sys.exit(_mod.main())

sys.modules[__name__] = _mod


def __getattr__(name: str) -> Any:  # pragma: no cover — static-analysis aid
    """Delegate attribute lookup to the relocated module (PEP 562)."""
    return getattr(_mod, name)
