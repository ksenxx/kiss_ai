"""Coding agents for KISS framework."""

from kiss.core.base import Base, DEFAULT_SYSTEM_PROMPT
from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from kiss.agents.coding_agents.gemini_cli_agent import GeminiCliAgent
from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent
from kiss.agents.coding_agents.openai_codex_agent import OpenAICodexAgent

__all__ = [
    "Base",
    "DEFAULT_SYSTEM_PROMPT",
    "ClaudeCodingAgent",
    "GeminiCliAgent",
    "KISSCodingAgent",
    "OpenAICodexAgent",
]
