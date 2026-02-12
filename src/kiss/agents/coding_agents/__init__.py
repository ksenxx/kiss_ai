"""Coding agents for KISS framework."""

import kiss.agents.coding_agents.config  # type: ignore # noqa: F401
from kiss.core.base import CODING_INSTRUCTIONS, Base

try:
    from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
except ImportError:
    ClaudeCodingAgent = None  # type: ignore[assignment,misc]

__all__ = [
    "Base",
    "CODING_INSTRUCTIONS",
    "ClaudeCodingAgent",
]
