"""Agent execution for KISSClaw - runs KISSAgent for each group message."""

from __future__ import annotations

import logging
from pathlib import Path

from kiss.agents.kissclaw.config import KissClawConfig
from kiss.agents.kissclaw.types import AgentOutput

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are {assistant_name}, a helpful personal AI assistant.

## Group Context
Group: {group_name}
{group_memory}

## Messages
{messages}

## Instructions
- Respond helpfully and concisely to the messages above.
- If there are multiple messages, address the most recent ones.
- Use <internal>...</internal> tags for any internal reasoning you don't want shown to the user.
- Return ONLY your response text. Do not include any metadata or formatting.
"""


def run_agent(
    config: KissClawConfig,
    group_name: str,
    group_folder: str,
    formatted_messages: str,
    agent_fn: object | None = None,
) -> AgentOutput:
    """Run a KISSAgent to respond to messages.

    Args:
        config: KISSClaw configuration.
        group_name: Name of the group.
        group_folder: Folder name for the group.
        formatted_messages: XML-formatted messages.
        agent_fn: Optional callable override for testing. If provided, called
                  with (prompt) and should return a string.

    Returns:
        AgentOutput with status and result.
    """
    # Load group memory (CLAUDE.md equivalent)
    group_dir = Path(config.groups_dir) / group_folder
    group_dir.mkdir(parents=True, exist_ok=True)
    memory_file = group_dir / "MEMORY.md"
    group_memory = ""
    if memory_file.exists():
        group_memory = f"## Memory\n{memory_file.read_text()}"

    prompt = AGENT_SYSTEM_PROMPT.format(
        assistant_name=config.assistant_name,
        group_name=group_name,
        group_memory=group_memory,
        messages=formatted_messages,
    )

    if agent_fn is not None:
        try:
            result = str(agent_fn(prompt))  # type: ignore[operator]
            return AgentOutput(status="success", result=result)
        except Exception as e:
            return AgentOutput(status="error", error=str(e))

    # Use KISSAgent
    try:
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent(name=f"KissClaw-{group_folder}")
        result = agent.run(
            model_name=config.model_name,
            prompt_template=prompt,
            is_agentic=False,
            max_steps=config.max_steps,
            max_budget=config.max_budget,
        )
        return AgentOutput(status="success", result=result)
    except Exception as e:
        logger.exception("Agent error for group %s", group_name)
        return AgentOutput(status="error", error=str(e))
