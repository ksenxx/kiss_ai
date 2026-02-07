# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""KISS agents package with pre-built agent implementations."""

from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent
from kiss.agents.kiss import (
    get_run_simple_coding_agent,
    prompt_refiner_agent,
    run_bash_task_in_sandboxed_ubuntu_latest,
)

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
    "ClaudeCodingAgent",
    "GeminiCliAgent",
    "KISSCodingAgent",
    "OpenAICodexAgent",
    "prompt_refiner_agent",
    "get_run_simple_coding_agent",
    "run_bash_task_in_sandboxed_ubuntu_latest",
]
