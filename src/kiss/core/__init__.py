# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Core module for the KISS agent framework."""

from typing import Any

from kiss.core.config import Config
from kiss.core.kiss_error import KISSError

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "KISSError",
]


def __getattr__(name: str) -> Any:
    """Return the live ``DEFAULT_CONFIG`` from :mod:`kiss.core.config`.

    ``config_builder`` (and the VS Code settings panel) rebind
    ``kiss.core.config.DEFAULT_CONFIG`` at runtime; a static
    ``from ... import`` here would freeze a stale snapshot, so the
    package-attribute lookup is deferred and never cached.  A caller using
    ``from kiss.core import DEFAULT_CONFIG`` still creates a normal Python
    snapshot; callers that need rebinding visibility must access this module
    attribute (or ``config.DEFAULT_CONFIG``) each time.
    """
    if name == "DEFAULT_CONFIG":
        from kiss.core import config as config_module

        return config_module.DEFAULT_CONFIG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
