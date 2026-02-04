"""Coding agents for KISS framework."""

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from kiss.agents.coding_agents.gemini_cli_agent import GeminiCliAgent
from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent
from kiss.agents.coding_agents.openai_codex_agent import OpenAICodexAgent
from kiss.core.base import CODING_INSTRUCTIONS, Base

__all__ = [
    "Base",
    "CODING_INSTRUCTIONS",
    "ClaudeCodingAgent",
    "GeminiCliAgent",
    "KISSCodingAgent",
    "OpenAICodexAgent",
]
