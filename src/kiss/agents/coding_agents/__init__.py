"""Coding agents for KISS framework."""

from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent
from kiss.core.base import CODING_INSTRUCTIONS, Base

try:
    from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
except ImportError:
    ClaudeCodingAgent = None  # type: ignore[assignment,misc]

try:
    from kiss.agents.coding_agents.gemini_cli_agent import GeminiCliAgent
except ImportError:
    GeminiCliAgent = None  # type: ignore[assignment,misc]

try:
    from kiss.agents.coding_agents.openai_codex_agent import OpenAICodexAgent
except ImportError:
    OpenAICodexAgent = None  # type: ignore[assignment,misc]

__all__ = [
    "Base",
    "CODING_INSTRUCTIONS",
    "ClaudeCodingAgent",
    "GeminiCliAgent",
    "KISSCodingAgent",
    "OpenAICodexAgent",
]
