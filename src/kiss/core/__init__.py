# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Core module for the KISS agent framework.

``Config`` and ``KISSError`` are exported lazily: every ``kiss.*`` import
runs this file, and importing :mod:`kiss.core.config` here would load
pydantic and build ``DEFAULT_CONFIG`` (reading the API-key environment
variables) as a side effect of each of them.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiss.core.config import Config
    from kiss.core.kiss_error import KISSError

__all__ = [
    "Config",
    "KISSError",
]


def __getattr__(name: str) -> type:
    """Import an exported class on first use (PEP 562).

    Args:
        name: The attribute looked up on ``kiss.core``.

    Returns:
        :class:`~kiss.core.config.Config` or
        :class:`~kiss.core.kiss_error.KISSError`.

    Raises:
        AttributeError: For any other name, so that ``from kiss.core
            import <submodule>`` still imports the submodule.
    """
    if name == "Config":
        from kiss.core.config import Config

        return Config
    if name == "KISSError":
        from kiss.core.kiss_error import KISSError

        return KISSError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
